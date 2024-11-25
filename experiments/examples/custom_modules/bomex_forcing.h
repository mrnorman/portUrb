
#pragma once

#include "main_header.h"
#include "profiles.h"
#include "coupler.h"
#include "TransformMatrices.h"
#include "hydrostasis.h"
#include <random>

namespace custom_modules {

  inline void bomex_forcing( core::Coupler & coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nx      = coupler.get_nx();
    auto ny      = coupler.get_ny();
    auto nz      = coupler.get_nz();
    auto dz      = coupler.get_dz();
    auto &dm     = coupler.get_data_manager_readwrite();
    auto uvel    = dm.get<real,3>("uvel"       );
    auto vvel    = dm.get<real,3>("vvel"       );
    auto wvel    = dm.get<real,3>("wvel"       );
    auto temp    = dm.get<real,3>("temp"       );
    auto rho_v   = dm.get<real,3>("water_vapor");
    real lat_g   = 14.941;
    real fcor    = 2*7.2921e-5*std::sin(lat_g/180*M_PI);
    auto w_tend = KOKKOS_LAMBDA (real z) -> real {
      real constexpr z0 = 0;
      real constexpr z1 = 1500;
      real constexpr z2 = 2100;
      real constexpr z3 = 3000;
      real constexpr v0 = 0;
      real constexpr v1 = -.65e-2;  // .65 cm / s
      real constexpr v2 = 0;
      real constexpr v3 = 0;
      if      (z >= z0 && z <= z1) { return v0+(v1-v0)/(z1-z0)*z; }
      else if (z >  z1 && z <= z2) { return v1+(v2-v1)/(z2-z1)*z; }
      else if (z >  z2 && z <= z3) { return v2+(v3-v2)/(z3-z2)*z; }
      return 0;
    };
    auto temp_tend = KOKKOS_LAMBDA (real z) -> real {
      real constexpr z0 = 0;
      real constexpr z1 = 1500;
      real constexpr z2 = 3000;
      real constexpr v0 = -2./86400.;  // -2K per day
      real constexpr v1 = -2./86400.;  // -2K per day
      real constexpr v2 = 0;
      if      (z >= z0 && z <= z1) { return v0+(v1-v0)/(z1-z0)*z; }
      else if (z >  z1 && z <= z2) { return v1+(v2-v1)/(z2-z1)*z; }
      return 0;
    };
    auto vapor_tend = KOKKOS_LAMBDA (real z) -> real {
      real constexpr z0 = 0;
      real constexpr z1 = 300;
      real constexpr z2 = 500;
      real constexpr z3 = 3000;
      real constexpr v0 = -1.2e-8;
      real constexpr v1 = -1.2e-8;
      real constexpr v2 = 0;
      real constexpr v3 = 0;
      if      (z >= z0 && z <= z1) { return v0+(v1-v0)/(z1-z0)*z; }
      else if (z >  z1 && z <= z2) { return v1+(v2-v1)/(z2-z1)*z; }
      else if (z >  z2 && z <= z3) { return v2+(v3-v2)/(z3-z2)*z; }
      return 0;
    };
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      real z = (k+0.5)*dz;
      real u = uvel(k,j,i);
      real v = vvel(k,j,i);
      real u_g = -10 + 1.8e-3*z;
      real v_g = 0;
      uvel (k,j,i) +=  dt*fcor*(v-v_g);
      vvel (k,j,i) += -dt*fcor*(u-u_g);
      wvel (k,j,i) +=  dt*w_tend    (z);
      temp (k,j,i) +=  dt*temp_tend (z);
      rho_v(k,j,i) +=  dt*vapor_tend(z);
    });
  }

}

