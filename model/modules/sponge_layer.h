
#pragma once

#include "coupler.h"

namespace modules {

  inline void sponge_layer( core::Coupler &coupler , real dt , real time_scale , real top_prop ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;

    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto nz      = coupler.get_nz  ();
    auto ny      = coupler.get_ny  ();
    auto nx      = coupler.get_nx  ();
    auto zlen    = coupler.get_zlen();
    auto dz      = coupler.get_dz  ();
    auto &dm     = coupler.get_data_manager_readwrite();

    auto dm_u = dm.get<real,3>("uvel");
    auto dm_v = dm.get<real,3>("vvel");
    auto dm_w = dm.get<real,3>("wvel");

    if (!dm.entry_exists("sponge_u_col")) {
      dm.register_and_allocate<real>("sponge_u_col","",{nz});
      dm.register_and_allocate<real>("sponge_v_col","",{nz});
      auto u_col = dm.get<real,1>("sponge_u_col");
      auto v_col = dm.get<real,1>("sponge_v_col");
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) {
        u_col(k) = 0;
        v_col(k) = 0;
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            u_col(k) += dm_u(k,j,i)*r_nx_ny;
            v_col(k) += dm_v(k,j,i)*r_nx_ny;
          }
        }
      });
      coupler.get_parallel_comm().all_reduce( u_col , MPI_SUM , "" ).deep_copy_to(u_col);
      coupler.get_parallel_comm().all_reduce( v_col , MPI_SUM , "" ).deep_copy_to(v_col);
    }
    auto u_col = dm.get<real,1>("sponge_u_col");
    auto v_col = dm.get<real,1>("sponge_v_col");

    real z1 = (1-top_prop)*zlen;
    real z2 = zlen;
    real p  = 3;
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      real z = (k+0.5_fp)*dz;
      if (z > z1) {
        real factor = std::pow((z-z1)/(z2-z1),p) * dt / time_scale;
        dm_u(k,j,i) = factor*u_col(k) + (1-factor)*dm_u(k,j,i);
        dm_v(k,j,i) = factor*v_col(k) + (1-factor)*dm_v(k,j,i);
        dm_w(k,j,i) = factor*0        + (1-factor)*dm_w(k,j,i);
      }
    });
  }

}

