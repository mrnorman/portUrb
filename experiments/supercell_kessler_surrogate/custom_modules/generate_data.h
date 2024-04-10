
#pragma once

#include "coupler.h"
#include <random>

namespace custom_modules {

  struct DataGenerator {

    std::string fname;

    void init( core::Coupler &coupler ) {
      yakl::SimpleNetCDF nc;
      fname = std::string("supercell_kessler_data_task_") + std::to_string(coupler.get_myrank()) + 
              std::string(".nc");
      nc.create(fname);
      nc.createDim("nsamples");
      nc.close();
    }


    void generate_data( core::Coupler const & coupler ,
                        realConst4d           rho_d_1 ,
                        realConst4d           rho_v_1 ,
                        realConst4d           rho_c_1 ,
                        realConst4d           rho_p_1 ,
                        realConst4d           wvel_1  ,
                        realConst4d           temp_1  ,
                        realConst4d           rho_v_2 ,
                        realConst4d           rho_c_2 ,
                        realConst4d           rho_p_2 ,
                        real                  dt      ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nz      = coupler.get_nz();
      auto ny      = coupler.get_ny();
      auto nx      = coupler.get_nx();
      auto nens    = coupler.get_nens();
      auto ny_glob = coupler.get_ny_glob();
      auto nx_glob = coupler.get_nx_glob();
      auto i_beg   = coupler.get_i_beg();
      auto j_beg   = coupler.get_j_beg();
      size_t total_samples = coupler.get_option<int>("total_samples",1000000);
      size_t num_time_steps = static_cast<size_t>(std::ceil(coupler.get_option<real>("sim_time") / dt));
      size_t num_points = static_cast<size_t>(nz)*static_cast<size_t>(ny_glob)*static_cast<size_t>(nx_glob)*nens;
      real probability = static_cast<real>(total_samples) / static_cast<real>(num_time_steps*num_points);

      real4d c_to_v   ("c_to_v"   ,nz,ny,nx,nens);
      real4d c_to_p   ("c_to_p"   ,nz,ny,nx,nens);
      real4d p_adv_net("p_adv_net",nz,ny,nx,nens);
      
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        c_to_v   (k,j,i,iens) =  (rho_v_2(k,j,i,iens) - rho_v_1(k,j,i,iens));
        c_to_p   (k,j,i,iens) = -(rho_c_2(k,j,i,iens) - rho_c_1(k,j,i,iens)) - std::max(0._fp,c_to_v(k,j,i,iens));
        p_adv_net(k,j,i,iens) =  (rho_p_2(k,j,i,iens) - rho_p_1(k,j,i,iens)) - std::max(0._fp,c_to_p(k,j,i,iens));
      });

      auto rho_d_1_h   = rho_d_1  .createHostCopy().collapse();
      auto rho_v_1_h   = rho_v_1  .createHostCopy().collapse();
      auto rho_c_1_h   = rho_c_1  .createHostCopy().collapse();
      auto rho_p_1_h   = rho_p_1  .createHostCopy().collapse();
      auto wvel_1_h    = wvel_1   .createHostCopy().collapse();
      auto temp_1_h    = temp_1   .createHostCopy().collapse();
      auto c_to_v_h    = c_to_v   .createHostCopy().collapse();
      auto c_to_p_h    = c_to_p   .createHostCopy().collapse();
      auto p_adv_net_h = p_adv_net.createHostCopy().collapse();

      std::mt19937                            engine(std::clock());
      std::uniform_real_distribution<double>  dist(0, 1);
      std::vector<std::array<real,6>> inputs_vec;
      std::vector<std::array<real,3>> outputs_vec;
      for (int i=0; i < rho_d_1_h.size(); i++) {
        real rn = dist(engine);
        if (rn < probability) {
          inputs_vec .push_back( {rho_d_1_h(i),rho_v_1_h(i),rho_c_1_h(i),rho_p_1_h(i),wvel_1_h(i),temp_1_h(i)} );
          outputs_vec.push_back( {c_to_v_h(i),c_to_p_h(i),p_adv_net_h(i)} );
        }
      }

      int num_samples = inputs_vec.size();
      if (num_samples > 0) {
        yakl::SimpleNetCDF nc;
        nc.open(fname,yakl::NETCDF_MODE_WRITE);
        int ulindex = nc.getDimSize("nsamples");
        if ( ! nc.varExists("time_step_size"     ) ) nc.write(dt                              ,"time_step_size"     );
        if ( ! nc.varExists("only_two_dimensions") ) nc.write(coupler.get_ny_glob()==1 ? 0 : 1,"only_two_dimensions");
        if ( ! nc.varExists("dx"                 ) ) nc.write(coupler.get_dx  ()              ,"dx"                 );
        if ( ! nc.varExists("dy"                 ) ) nc.write(coupler.get_dy  ()              ,"dy"                 );
        if ( ! nc.varExists("dz"                 ) ) nc.write(coupler.get_dz  ()              ,"dz"                 );
        if ( ! nc.varExists("xlen"               ) ) nc.write(coupler.get_xlen()              ,"xlen"               );
        if ( ! nc.varExists("ylen"               ) ) nc.write(coupler.get_ylen()              ,"ylen"               );
        if ( ! nc.varExists("zlen"               ) ) nc.write(coupler.get_zlen()              ,"zlen"               );
        realHost1d inputs ("inputs" ,6);
        realHost1d outputs("outputs",3);
        for (int isamp = 0; isamp < num_samples; isamp++) {
          for (int i=0; i < 6; i++) { inputs (i) = inputs_vec [isamp][i]; }
          for (int i=0; i < 3; i++) { outputs(i) = outputs_vec[isamp][i]; }
          nc.write1( inputs  , "inputs"  , {"num_vars_in" } , ulindex , "nsamples" );
          nc.write1( outputs , "outputs" , {"num_vars_out"} , ulindex , "nsamples" );
          ulindex++;
        }
      }
    }

  };

}


