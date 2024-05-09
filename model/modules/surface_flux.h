
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
    auto immersed  = coupler.get_data_manager_readonly ().get<real const,4>("immersed_proportion_halos");
    auto imm_rough = coupler.get_data_manager_readonly ().get<real const,4>("immersed_roughness_halos" );
    auto imm_khf   = coupler.get_data_manager_readonly ().get<real const,4>("immersed_khf_halos"       );
    int  hs        = (immersed.extent(2)-nx)/2;

    real4d tend_u("tend_u",nz,ny,nx,nens);
    real4d tend_v("tend_v",nz,ny,nx,nens);
    real4d tend_w("tend_w",nz,ny,nx,nens);
    real4d tend_T("tend_T",nz,ny,nx,nens);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                      YAKL_LAMBDA (int k, int j, int i, int iens) {
      tend_u(k,j,i,iens) = 0;
      tend_v(k,j,i,iens) = 0;
      tend_w(k,j,i,iens) = 0;
      tend_T(k,j,i,iens) = 0;
    });
    parallel_for( YAKL_AUTO_LABEL() , Bounds<4>({-1,nz},{-1,ny},{-1,nx},nens) ,
                                      YAKL_LAMBDA (int k, int j, int i, int iens) {
      if (immersed(hs+k,hs+j,hs+i,iens) > 0 && imm_rough(hs+k,hs+j,hs+i,iens) > 0) {
        real constexpr vk = 0.40;
        real lgx   = std::log((dx/2+roughness)/roughness);
        real lgy   = std::log((dy/2+roughness)/roughness);
        real lgz   = std::log((dz/2+roughness)/roughness);
        real c_dx  = vk*vk/(lgx*lgx);
        real c_dy  = vk*vk/(lgy*lgy);
        real c_dz  = vk*vk/(lgz*lgz);
        if (i >= 0 && i < nx && j >= 0 && j < ny && k-1 >= 0 && k-1 < nz && immersed(hs+k-1,hs+j,hs+i,iens) < 1) {
          real u = dm_u(k-1,j,i,iens);
          real v = dm_v(k-1,j,i,iens);
          real mag = std::sqrt(u*u+v*v);
          yakl::atomicAdd( tend_u(k-1,j,i,iens) , -c_dz*u*mag/dz );
          yakl::atomicAdd( tend_v(k-1,j,i,iens) , -c_dz*v*mag/dz );
          yakl::atomicAdd( tend_T(k-1,j,i,iens) , cp_d*imm_khf(hs+k,hs+j,hs+i,iens) );
        }
        if (i >= 0 && i < nx && j >= 0 && j < ny && k+1 >= 0 && k+1 < nz && immersed(hs+k+1,hs+j,hs+i,iens) < 1) {
          real u = dm_u(k+1,j,i,iens);
          real v = dm_v(k+1,j,i,iens);
          real mag = std::sqrt(u*u+v*v);
          yakl::atomicAdd( tend_u(k+1,j,i,iens) , -c_dz*u*mag/dz );
          yakl::atomicAdd( tend_v(k+1,j,i,iens) , -c_dz*v*mag/dz );
          yakl::atomicAdd( tend_T(k+1,j,i,iens) , cp_d*imm_khf(hs+k,hs+j,hs+i,iens) );
        }
        if (i >= 0 && i < nx && j-1 >= 0 && j-1 < ny && k >= 0 && k < nz && immersed(hs+k,hs+j-1,hs+i,iens) < 1) {
          real u = dm_u(k,j-1,i,iens);
          real w = dm_w(k,j-1,i,iens);
          real mag = std::sqrt(u*u+w*w);
          yakl::atomicAdd( tend_u(k,j-1,i,iens) , -c_dy*u*mag/dy );
          yakl::atomicAdd( tend_w(k,j-1,i,iens) , -c_dy*w*mag/dy );
          yakl::atomicAdd( tend_T(k,j-1,i,iens) , cp_d*imm_khf(hs+k,hs+j,hs+i,iens) );
        }
        if (i >= 0 && i < nx && j+1 >= 0 && j+1 < ny && k >= 0 && k < nz && immersed(hs+k,hs+j+1,hs+i,iens) < 1) {
          real u = dm_u(k,j+1,i,iens);
          real w = dm_w(k,j+1,i,iens);
          real mag = std::sqrt(u*u+w*w);
          yakl::atomicAdd( tend_u(k,j+1,i,iens) , -c_dy*u*mag/dy );
          yakl::atomicAdd( tend_w(k,j+1,i,iens) , -c_dy*w*mag/dy );
          yakl::atomicAdd( tend_T(k,j+1,i,iens) , cp_d*imm_khf(hs+k,hs+j,hs+i,iens) );
        }
        if (i-1 >= 0 && i-1 < nx && j >= 0 && j < ny && k >= 0 && k < nz && immersed(hs+k,hs+j,hs+i-1,iens) < 1) {
          real v = dm_v(k,j,i-1,iens);
          real w = dm_w(k,j,i-1,iens);
          real mag = std::sqrt(v*v+w*w);
          yakl::atomicAdd( tend_v(k,j,i-1,iens) , -c_dx*v*mag/dx );
          yakl::atomicAdd( tend_w(k,j,i-1,iens) , -c_dx*w*mag/dx );
          yakl::atomicAdd( tend_T(k,j,i-1,iens) , cp_d*imm_khf(hs+k,hs+j,hs+i,iens) );
        }
        if (i+1 >= 0 && i+1 < nx && j >= 0 && j < ny && k >= 0 && k < nz && immersed(hs+k,hs+j,hs+i+1,iens) < 1) {
          real v = dm_v(k,j,i+1,iens);
          real w = dm_w(k,j,i+1,iens);
          real mag = std::sqrt(v*v+w*w);
          yakl::atomicAdd( tend_v(k,j,i+1,iens) , -c_dx*v*mag/dx );
          yakl::atomicAdd( tend_w(k,j,i+1,iens) , -c_dx*w*mag/dx );
          yakl::atomicAdd( tend_T(k,j,i+1,iens) , cp_d*imm_khf(hs+k,hs+j,hs+i,iens) );
        }
      }
    });

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                      YAKL_LAMBDA (int k, int j, int i, int iens) {
      dm_u(k,j,i,iens) += dt*tend_u(k,j,i,iens);
      dm_v(k,j,i,iens) += dt*tend_v(k,j,i,iens);
      dm_w(k,j,i,iens) += dt*tend_w(k,j,i,iens);
      dm_T(k,j,i,iens) += dt*tend_T(k,j,i,iens);
    });

  };

}

