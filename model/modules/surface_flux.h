
#pragma once

namespace modules {

  // Currently ignoring stability / universal functions
  inline void apply_surface_fluxes( core::Coupler &coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::c::Bounds;
    auto nx        = coupler.get_nx  ();
    auto ny        = coupler.get_ny  ();
    auto nz        = coupler.get_nz  ();
    auto nens      = coupler.get_nens();
    auto dx        = coupler.get_dx  ();
    auto dy        = coupler.get_dy  ();
    auto dz        = coupler.get_dz  ();
    auto p0        = coupler.get_option<real>("p0");
    auto R_d       = coupler.get_option<real>("R_d");
    auto cp_d      = coupler.get_option<real>("cp_d");
    auto roughness = coupler.get_option<real>("roughness",0.1);
    auto dm_r      = coupler.get_data_manager_readonly ().get<real const,4>("density_dry");
    auto dm_u      = coupler.get_data_manager_readwrite().get<real      ,4>("uvel");
    auto dm_v      = coupler.get_data_manager_readwrite().get<real      ,4>("vvel");
    auto dm_w      = coupler.get_data_manager_readwrite().get<real      ,4>("wvel");
    auto dm_T      = coupler.get_data_manager_readwrite().get<real      ,4>("temp");
    auto dm_Ts     = coupler.get_data_manager_readwrite().get<real      ,3>("surface_temp");
    auto immersed  = coupler.get_data_manager_readonly ().get<real const,4>("immersed_proportion_halos");
    int  hs        = (immersed.extent(2)-nx)/2;

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(ny,nx,nens) , YAKL_LAMBDA (int j, int i, int iens) {
      if (immersed(hs+0,hs+j,hs+i,iens) == 0) {
        real constexpr von_karman = 0.40;
        real r   = dm_r (0,j,i,iens);
        real u   = dm_u (0,j,i,iens);
        real v   = dm_v (0,j,i,iens);
        real T   = dm_T (0,j,i,iens);
        real th0 = dm_Ts(  j,i,iens);
        real p   = r*R_d*T;
        real th  = T*std::pow( p0/p , R_d/cp_d );
        real lg  = std::log((dz/2+roughness)/roughness);
        real c_d = von_karman*von_karman / (lg*lg);
        real mag = std::sqrt(u*u+v*v);
        real u_new = u - dt*c_d     *(u -0  )*mag/dz;
        real v_new = v - dt*c_d     *(v -0  )*mag/dz;
        real T_new = T - dt*c_d*cp_d*(th-th0)*mag/dz;
        // Don't allow friction to change sign of difference from suface values
        if ((u_new-0  )*(u-0  ) < 0) u_new = 0;
        if ((v_new-0  )*(v-0  ) < 0) v_new = 0;
        if ((T_new-th0)*(T-th0) < 0) T_new = th0;
        dm_u(0,j,i,iens) = u_new;
        dm_v(0,j,i,iens) = v_new;
        dm_T(0,j,i,iens) = T_new;
      }
    });

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
      if (immersed(hs+k,hs+j,hs+i,iens) == 0) {
        real constexpr von_karman = 0.40;
        real lgx   = std::log((dx/2+roughness)/roughness);
        real lgy   = std::log((dy/2+roughness)/roughness);
        real lgz   = std::log((dz/2+roughness)/roughness);
        real c_dx  = von_karman*von_karman / (lgx*lgx);
        real c_dy  = von_karman*von_karman / (lgy*lgy);
        real c_dz  = von_karman*von_karman / (lgz*lgz);
        real u     = dm_u(k,j,i,iens);
        real v     = dm_v(k,j,i,iens);
        real w     = dm_w(k,j,i,iens);
        real magx  = std::sqrt(v*v+w*w);
        real magy  = std::sqrt(u*u+w*w);
        real magz  = std::sqrt(u*u+v*v);
        real utend = 0;
        real vtend = 0;
        real wtend = 0;
        if (immersed(hs+k,hs+j,hs+i-1,iens) > 0) {
          vtend += -c_dx*v*magx/dx;
          wtend += -c_dx*w*magx/dx;
        }
        if (immersed(hs+k,hs+j,hs+i+1,iens) > 0) {
          vtend += -c_dx*v*magx/dx;
          wtend += -c_dx*w*magx/dx;
        }
        if (immersed(hs+k,hs+j-1,hs+i,iens) > 0) {
          utend += -c_dy*u*magy/dy;
          wtend += -c_dy*w*magy/dy;
        }
        if (immersed(hs+k,hs+j+1,hs+i,iens) > 0) {
          utend += -c_dy*u*magy/dy;
          wtend += -c_dy*w*magy/dy;
        }
        if (immersed(hs+k-1,hs+j,hs+i,iens) > 0 && k != 0) {
          utend += -c_dz*u*magz/dz;
          vtend += -c_dz*v*magz/dz;
        }
        if (immersed(hs+k+1,hs+j,hs+i,iens) > 0 && k != nz-1) {
          utend += -c_dz*u*magz/dz;
          vtend += -c_dz*v*magz/dz;
        }
        dm_u(k,j,i,iens) += dt*utend;
        dm_v(k,j,i,iens) += dt*vtend;
        dm_w(k,j,i,iens) += dt*wtend;
        // Don't allow friction to change sign of difference from suface values
        if (dm_u(k,j,i,iens)*u < 0) dm_u(k,j,i,iens) = 0;
        if (dm_v(k,j,i,iens)*v < 0) dm_v(k,j,i,iens) = 0;
        if (dm_w(k,j,i,iens)*w < 0) dm_w(k,j,i,iens) = 0;
      }
    });

  };

}

