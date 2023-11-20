
#pragma once

#include "coupler.h"

namespace custom_modules {

  inline void nudge_winds( core::Coupler &coupler , real dt , real uvel_avg , real vvel_avg ,  real time_scale = 1 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nens    = coupler.get_nens();
    auto nz      = coupler.get_nz();
    auto ny      = coupler.get_ny();
    auto nx      = coupler.get_nx();
    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto uvel    = coupler.get_data_manager_readwrite().get<real,4>("uvel");
    auto vvel    = coupler.get_data_manager_readwrite().get<real,4>("vvel");
    real3d u1("u1",nz,ny,nx);
    real3d v1("v1",nz,ny,nx);
    realHost2d sum_loc_host("sum_loc",2,nens);
    for (int iens=0; iens < nens; iens++) {
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        u1(k,j,i) = uvel(k,j,i,iens);
        v1(k,j,i) = vvel(k,j,i,iens);
      });
      sum_loc_host(0,iens) = yakl::intrinsics::sum(u1);
      sum_loc_host(1,iens) = yakl::intrinsics::sum(v1);
    }
    auto sum_glob_host = sum_loc_host.createHostObject();
    auto comm = MPI_COMM_WORLD;
    auto dtype = coupler.get_mpi_data_type();
    MPI_Allreduce( sum_loc_host.data() , sum_glob_host.data() , sum_loc_host.size() , dtype , MPI_SUM , comm );
    auto sum_glob = sum_glob_host.createDeviceCopy();
    real r_nx_ny_nz = 1./(nx_glob*ny_glob*nz);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
      uvel(k,j,i,iens) += dt/time_scale*( uvel_avg - sum_glob(0,iens)*r_nx_ny_nz );
      vvel(k,j,i,iens) += dt/time_scale*( vvel_avg - sum_glob(1,iens)*r_nx_ny_nz );
    });
  }

}


