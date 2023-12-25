
#pragma once

namespace modules {

  // Currently ignoring stability / universal functions
  inline void apply_surface_fluxes( core::Coupler &coupler , real dtphys ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nx    = coupler.get_nx  ();
    auto ny    = coupler.get_ny  ();
    auto nz    = coupler.get_nz  ();
    auto nens  = coupler.get_nens();
    auto dx    = coupler.get_dx  ();
    auto dy    = coupler.get_dy  ();
    auto dz    = coupler.get_dz  ();
    auto p0    = coupler.get_option<real>("p0");
    auto R_d   = coupler.get_option<real>("R_d");
    auto cp_d  = coupler.get_option<real>("cp_d");
    auto dm_r  = coupler.get_data_manager_readwrite().get<real const,4>("density_dry");
    auto dm_u  = coupler.get_data_manager_readwrite().get<real      ,4>("uvel");
    auto dm_v  = coupler.get_data_manager_readwrite().get<real      ,4>("vvel");
    auto dm_T  = coupler.get_data_manager_readwrite().get<real      ,4>("temp");
    auto dm_Ts = coupler.get_data_manager_readwrite().get<real      ,4>("surface_temp");

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(ny,nx,nens) , YAKL_LAMBDA (int j, int i, int iens) {
      real constexpr prandtl    = 0.71;
      real constexpr roughness  = 0.1;
      real constexpr von_karman = 0.40;
      real r   = dm_r (0,j,i,iens);
      real u   = dm_u (0,j,i,iens);
      real v   = dm_v (0,j,i,iens);
      real T   = dm_T (0,j,i,iens);
      real th0 = dm_Ts(0,j,i,iens);
      real p   = r*R_d*T;
      real th  = T*std::pow( p0/p , R_d/cp_d );
      real lg  = std::log((z1+roughness)/roughness);
      real c_d = von_karman*von_karman / (lg*lg);
      real mag = std::sqrt(u*u+v*v);
      real u_new  = u  + dt*(0-c_d*u*mag               )/dz;
      real v_new  = v  + dt*(0-c_d*v*mag               )/dz;
      real th_new = th + dt*(0-c_d/prandtl*(th-th0)*mag)/dz;
      real T_new  = th_new*std::pow( p/p0 , R_d/cp_d );
      dm_u(0,j,i,iens) = u_new;
      dm_v(0,j,i,iens) = v_new;
      dm_T(0,j,i,iens) = T_new;
    });

  };

}

