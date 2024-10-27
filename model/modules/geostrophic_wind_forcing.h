
#pragma once

#include "coupler.h"

namespace modules {

  inline void geostrophic_wind_forcing( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz       = coupler.get_nz  ();
    auto ny       = coupler.get_ny  ();
    auto nx       = coupler.get_nx  ();
    auto &dm      = coupler.get_data_manager_readwrite();
    auto uvel     = dm.get<real,3>("uvel");
    auto vvel     = dm.get<real,3>("vvel");
    real fcor     = 2*7.2921e-5*std::sin(lat_g/180*M_PI);
    real3d utend("utend",nz,ny,nx);
    real3d vtend("vtend",nz,ny,nx);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      utend(k,j,i) =  fcor*(vvel(k,j,i)-v_g);
      vtend(k,j,i) = -fcor*(uvel(k,j,i)-u_g);
    });
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt*utend(k,j,i);
      vvel(k,j,i) += dt*vtend(k,j,i);
    });
  }

}

