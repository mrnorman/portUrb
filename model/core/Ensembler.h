
#pragma once

#include "coupler.h"
#include <functional>

namespace core {

  struct Ensembler {

    struct Dimension {
      int                                          size;
      std::function<int(int ind,core::Coupler &)>  func;
    };

    std::vector<Dimension> dimensions;


    void register_dimension( int size , std::function<int(int ind, core::Coupler &)> func ) {
      dimensions.push_back({size,func});
    }


    void append_coupler_string( core::Coupler &coupler , std::string label , std::string val ) const {
      auto option = coupler.get_option<std::string>(label,"");
      if (option.empty()) { coupler.set_option<std::string>(label,val); }
      else                { coupler.set_option<std::string>(label,option+std::string("_")+val); }
      
    }


    ParallelComm create_coupler_comm( Coupler & coupler , int base_ranks = 1 , MPI_Comm comm_in = MPI_COMM_WORLD) {
      if (dimensions.size() == 0) yakl::yakl_throw("Trying to tensor ensemble with no dimensions");
      int total = dimensions.at(0).size;
      for (int i=1; i < dimensions.size(); i++) { total *= dimensions.at(i).size; }
      int nranks, myrank;
      MPI_Comm_rank( comm_in , &myrank );
      MPI_Comm_size( comm_in , &nranks );
      if (nranks < total) {
        std::cerr << "Ensemble communicator has [" << nranks << "] ranks, which is less than the [" << total
                  << "] desired ensembles." << std::endl;
        yakl::yakl_throw("");
      }
      if (nranks > total) {
        if (myrank == 0) {
          std::cout << "WARNING: Launching with [" << nranks << "] ranks, which is more than the [" << total
                    << "] desired ensembles, meaning some ranks will be unused by this ensemble." << std::endl;
        }
      }
      int rank_beg = 0;
      ParallelComm par_comm;
      if        (dimensions.size() == 1) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
          int nranks  = base_ranks;
          nranks *= dimensions.at(0).func(i0,coupler);
          par_comm.create( myrank >= rank_beg && myrank <= rank_beg+nranks , comm_in );
          rank_beg += nranks;
        }
      } else if (dimensions.size() == 2) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
          int nranks  = base_ranks;
          nranks *= dimensions.at(0).func(i0,coupler);
          nranks *= dimensions.at(1).func(i1,coupler);
          par_comm.create( myrank >= rank_beg && myrank <= rank_beg+nranks , comm_in );
          rank_beg += nranks;
        } }
      } else if (dimensions.size() == 3) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
        for (int i2=0; i2 < dimensions.at(2).size; i2++) {
          int nranks  = base_ranks;
          nranks *= dimensions.at(0).func(i0,coupler);
          nranks *= dimensions.at(1).func(i1,coupler);
          nranks *= dimensions.at(2).func(i2,coupler);
          par_comm.create( myrank >= rank_beg && myrank <= rank_beg+nranks , comm_in );
          rank_beg += nranks;
        } } }
      } else if (dimensions.size() == 4) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
        for (int i2=0; i2 < dimensions.at(2).size; i2++) {
        for (int i3=0; i3 < dimensions.at(3).size; i3++) {
          int nranks  = base_ranks;
          nranks *= dimensions.at(0).func(i0,coupler);
          nranks *= dimensions.at(1).func(i1,coupler);
          nranks *= dimensions.at(2).func(i2,coupler);
          nranks *= dimensions.at(3).func(i3,coupler);
          par_comm.create( myrank >= rank_beg && myrank <= rank_beg+nranks , comm_in );
          rank_beg += nranks;
        } } } }
      } else if (dimensions.size() == 5) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
        for (int i2=0; i2 < dimensions.at(2).size; i2++) {
        for (int i3=0; i3 < dimensions.at(3).size; i3++) {
        for (int i4=0; i4 < dimensions.at(4).size; i4++) {
          int nranks  = base_ranks;
          nranks *= dimensions.at(0).func(i0,coupler);
          nranks *= dimensions.at(1).func(i1,coupler);
          nranks *= dimensions.at(2).func(i2,coupler);
          nranks *= dimensions.at(3).func(i3,coupler);
          nranks *= dimensions.at(4).func(i4,coupler);
          par_comm.create( myrank >= rank_beg && myrank <= rank_beg+nranks , comm_in );
          rank_beg += nranks;
        } } } } }
      } else if (dimensions.size() == 6) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
        for (int i2=0; i2 < dimensions.at(2).size; i2++) {
        for (int i3=0; i3 < dimensions.at(3).size; i3++) {
        for (int i4=0; i4 < dimensions.at(4).size; i4++) {
        for (int i5=0; i5 < dimensions.at(5).size; i5++) {
          int nranks  = base_ranks;
          nranks *= dimensions.at(0).func(i0,coupler);
          nranks *= dimensions.at(1).func(i1,coupler);
          nranks *= dimensions.at(2).func(i2,coupler);
          nranks *= dimensions.at(3).func(i3,coupler);
          nranks *= dimensions.at(4).func(i4,coupler);
          nranks *= dimensions.at(5).func(i5,coupler);
          par_comm.create( myrank >= rank_beg && myrank <= rank_beg+nranks , comm_in );
          rank_beg += nranks;
        } } } } } }
      } else if (dimensions.size() == 7) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
        for (int i2=0; i2 < dimensions.at(2).size; i2++) {
        for (int i3=0; i3 < dimensions.at(3).size; i3++) {
        for (int i4=0; i4 < dimensions.at(4).size; i4++) {
        for (int i5=0; i5 < dimensions.at(5).size; i5++) {
        for (int i6=0; i6 < dimensions.at(6).size; i6++) {
          int nranks  = base_ranks;
          nranks *= dimensions.at(0).func(i0,coupler);
          nranks *= dimensions.at(1).func(i1,coupler);
          nranks *= dimensions.at(2).func(i2,coupler);
          nranks *= dimensions.at(3).func(i3,coupler);
          nranks *= dimensions.at(4).func(i4,coupler);
          nranks *= dimensions.at(5).func(i5,coupler);
          nranks *= dimensions.at(6).func(i6,coupler);
          par_comm.create( myrank >= rank_beg && myrank <= rank_beg+nranks , comm_in );
          rank_beg += nranks;
        } } } } } } }
      } else if (dimensions.size() == 8) {
        for (int i0=0; i0 < dimensions.at(0).size; i0++) {
        for (int i1=0; i1 < dimensions.at(1).size; i1++) {
        for (int i2=0; i2 < dimensions.at(2).size; i2++) {
        for (int i3=0; i3 < dimensions.at(3).size; i3++) {
        for (int i4=0; i4 < dimensions.at(4).size; i4++) {
        for (int i5=0; i5 < dimensions.at(5).size; i5++) {
        for (int i6=0; i6 < dimensions.at(6).size; i6++) {
        for (int i7=0; i7 < dimensions.at(7).size; i7++) {
          int nranks  = base_ranks;
          nranks *= dimensions.at(0).func(i0,coupler);
          nranks *= dimensions.at(1).func(i1,coupler);
          nranks *= dimensions.at(2).func(i2,coupler);
          nranks *= dimensions.at(3).func(i3,coupler);
          nranks *= dimensions.at(4).func(i4,coupler);
          nranks *= dimensions.at(5).func(i5,coupler);
          nranks *= dimensions.at(6).func(i6,coupler);
          nranks *= dimensions.at(7).func(i7,coupler);
          par_comm.create( myrank >= rank_beg && myrank <= rank_beg+nranks , comm_in );
          rank_beg += nranks;
        } } } } } } } }
      } else {
        yakl::yakl_throw("Requesting more ensemble dimensions than the 1-8 implemented");
      }
      return par_comm;
    }
  };

}
