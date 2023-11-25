
#pragma once

#include "coupler.h"

namespace custom_modules {

  inline void nudge_winds( core::Coupler &coupler                            ,
                           real dt                                           ,
                           real time_scale                                   ,
                           real uvel_avg  = std::numeric_limits<real>::max() ,
                           real vvel_avg  = std::numeric_limits<real>::max() ,
                           real wvel_avg  = std::numeric_limits<real>::max() ,
                           real temp_avg  = std::numeric_limits<real>::max() ,
                           real rho_d_avg = std::numeric_limits<real>::max() ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nens    = coupler.get_nens();
    auto nz      = coupler.get_nz();
    auto ny      = coupler.get_ny();
    auto nx      = coupler.get_nx();
    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto rho_d   = coupler.get_data_manager_readwrite().get<real,4>("density_dry");
    auto uvel    = coupler.get_data_manager_readwrite().get<real,4>("uvel");
    auto vvel    = coupler.get_data_manager_readwrite().get<real,4>("vvel");
    auto wvel    = coupler.get_data_manager_readwrite().get<real,4>("wvel");
    auto temp    = coupler.get_data_manager_readwrite().get<real,4>("temp");
    bool do_r = rho_d_avg != std::numeric_limits<real>::max();
    bool do_u = uvel_avg  != std::numeric_limits<real>::max();
    bool do_v = vvel_avg  != std::numeric_limits<real>::max();
    bool do_w = wvel_avg  != std::numeric_limits<real>::max();
    bool do_t = temp_avg  != std::numeric_limits<real>::max();
    real3d r1("r1",nz,ny,nx);
    real3d u1("u1",nz,ny,nx);
    real3d v1("v1",nz,ny,nx);
    real3d w1("w1",nz,ny,nx);
    real3d t1("t1",nz,ny,nx);
    realHost2d sum_loc_host("sum_loc",5,nens);
    for (int iens=0; iens < nens; iens++) {
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        if (do_r) r1(k,j,i) = rho_d(k,j,i,iens);
        if (do_u) u1(k,j,i) = uvel (k,j,i,iens);
        if (do_v) v1(k,j,i) = vvel (k,j,i,iens);
        if (do_w) w1(k,j,i) = wvel (k,j,i,iens);
        if (do_t) t1(k,j,i) = temp (k,j,i,iens);
      });
      sum_loc_host(0,iens) = do_r ? yakl::intrinsics::sum(r1) : 0;
      sum_loc_host(1,iens) = do_u ? yakl::intrinsics::sum(u1) : 0;
      sum_loc_host(2,iens) = do_v ? yakl::intrinsics::sum(v1) : 0;
      sum_loc_host(3,iens) = do_w ? yakl::intrinsics::sum(w1) : 0;
      sum_loc_host(4,iens) = do_t ? yakl::intrinsics::sum(t1) : 0;
    }
    auto sum_glob_host = sum_loc_host.createHostObject();
    auto comm = MPI_COMM_WORLD;
    auto dtype = coupler.get_mpi_data_type();
    MPI_Allreduce( sum_loc_host.data() , sum_glob_host.data() , sum_loc_host.size() , dtype , MPI_SUM , comm );
    auto sum_glob = sum_glob_host.createDeviceCopy();
    real r_nx_ny_nz = 1./(nx_glob*ny_glob*nz);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
      if (do_r) rho_d(k,j,i,iens) += dt/time_scale*( rho_d_avg - sum_glob(0,iens)*r_nx_ny_nz );
      if (do_u) uvel (k,j,i,iens) += dt/time_scale*( uvel_avg  - sum_glob(1,iens)*r_nx_ny_nz );
      if (do_v) vvel (k,j,i,iens) += dt/time_scale*( vvel_avg  - sum_glob(2,iens)*r_nx_ny_nz );
      if (do_w) wvel (k,j,i,iens) += dt/time_scale*( wvel_avg  - sum_glob(3,iens)*r_nx_ny_nz );
      if (do_t) temp (k,j,i,iens) += dt/time_scale*( temp_avg  - sum_glob(4,iens)*r_nx_ny_nz );
    });
  }

}


