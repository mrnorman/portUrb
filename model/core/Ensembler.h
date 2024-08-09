
#pragma once

#include "coupler.h"
#include <functional>

namespace core {

  struct Ensembler {

    struct Dimension {
      int                                       size;
      std::function<int(int)>                   func_nranks;
      std::function<void(int,core::Coupler &)>  func_coupler;
    };

    std::vector<Dimension> dimensions;


    void register_dimension( int size ,
                             std::function<int(int)> func_nranks ,
                             std::function<void(int,core::Coupler &)> func_coupler ) {
      dimensions.push_back({size,func_nranks,func_coupler});
    }


    void append_coupler_string( core::Coupler &coupler , std::string label , std::string val ) const {
      auto option = coupler.get_option<std::string>(label,"");
      if (option.empty()) { coupler.set_option<std::string>(label,val); }
      else                { coupler.set_option<std::string>(label,option+std::string("_")+val); }
      
    }


    ParallelComm create_coupler_comm( Coupler & coupler , int base_ranks = 1 , MPI_Comm comm_in = MPI_COMM_WORLD) {
      if (dimensions.size() == 0) yakl::yakl_throw("Trying to tensor ensemble with no dimensions");
      int nranks_tot, myrank;
      MPI_Comm_rank( comm_in , &myrank );
      MPI_Comm_size( comm_in , &nranks_tot );
      int rank_beg = 0;
      ParallelComm par_comm;
      if        (dimensions.size() == 1) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
          int nranks = base_ranks;
          nranks *= dimensions.at(0).func_nranks(i0);
          bool active = myrank >= rank_beg && myrank < rank_beg+nranks;
          par_comm.create( active , comm_in );
          if (active) {
            dimensions.at(0).func_coupler(i0,coupler);
          }
          rank_beg += nranks;
        }
      } else if (dimensions.size() == 2) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
          int nranks = base_ranks;
          nranks *= dimensions.at(0).func_nranks(i0);
          nranks *= dimensions.at(1).func_nranks(i1);
          bool active = myrank >= rank_beg && myrank < rank_beg+nranks;
          par_comm.create( active , comm_in );
          if (active) {
            dimensions.at(0).func_coupler(i0,coupler);
            dimensions.at(1).func_coupler(i1,coupler);
          }
          rank_beg += nranks;
        } }
      } else if (dimensions.size() == 3) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
        for (int i2=0; i2 < dimensions.at(2).size; i2++) {
          int nranks = base_ranks;
          nranks *= dimensions.at(0).func_nranks(i0);
          nranks *= dimensions.at(1).func_nranks(i1);
          nranks *= dimensions.at(2).func_nranks(i2);
          bool active = myrank >= rank_beg && myrank < rank_beg+nranks;
          par_comm.create( active , comm_in );
          if (active) {
            dimensions.at(0).func_coupler(i0,coupler);
            dimensions.at(1).func_coupler(i1,coupler);
            dimensions.at(2).func_coupler(i2,coupler);
          }
          rank_beg += nranks;
        } } }
      } else {
        yakl::yakl_throw("Requesting more ensemble dimensions than the 1-3 implemented");
      }
      return par_comm;
    }
  };

}
