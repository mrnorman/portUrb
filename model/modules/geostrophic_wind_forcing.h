
#pragma once

#include "coupler.h"

namespace modules {

  inline void geostrophic_wind_forcing( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz       = coupler.get_nz  ();
    auto ny       = coupler.get_ny  ();
    auto nx       = coupler.get_nx  ();
    auto nens     = coupler.get_nens();
    auto &dm      = coupler.get_data_manager_readwrite();
    auto uvel     = dm.get<real,4>("uvel");
    auto vvel     = dm.get<real,4>("vvel");
    real fcor     = 2*7.2921e-5*std::sin(lat_g/180*M_PI);
    real4d utend("utend",nz,ny,nx,nens);
    real4d vtend("vtend",nz,ny,nx,nens);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                      YAKL_LAMBDA (int k, int j, int i, int iens) {
      utend(k,j,i,iens) =  fcor*(vvel(k,j,i,iens)-v_g);
      vtend(k,j,i,iens) = -fcor*(uvel(k,j,i,iens)-u_g);
    });
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                      YAKL_LAMBDA (int k, int j, int i, int iens) {
      uvel(k,j,i,iens) += dt*utend(k,j,i,iens);
      vvel(k,j,i,iens) += dt*vtend(k,j,i,iens);
    });
  }

}

