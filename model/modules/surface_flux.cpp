
#include "surface_flux.h"

namespace modules {

  // Currently ignoring stability / universal functions
  void apply_surface_fluxes( core::Coupler &coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    using yakl::c::Bounds;
    auto nx        = coupler.get_nx  ();
    auto ny        = coupler.get_ny  ();
    auto nz        = coupler.get_nz  ();
    auto dx        = coupler.get_dx  ();
    auto dy        = coupler.get_dy  ();
    auto dz        = coupler.get_dz  ();
    auto p0        = coupler.get_option<real>("p0");
    auto R_d       = coupler.get_option<real>("R_d");
    auto cp_d      = coupler.get_option<real>("cp_d");
    auto &dm       = coupler.get_data_manager_readwrite();
    auto dm_r      = dm.get<real const,3>("density_dry");
    auto dm_u      = dm.get<real      ,3>("uvel");
    auto dm_v      = dm.get<real      ,3>("vvel");
    auto dm_w      = dm.get<real      ,3>("wvel");
    auto dm_T      = dm.get<real      ,3>("temp");
    auto imm_prop  = dm.get<real const,3>("immersed_proportion_halos");
    auto imm_rough = dm.get<real const,3>("immersed_roughness_halos" );
    auto imm_temp  = dm.get<real const,3>("immersed_temp_halos"      );
    auto imm_khf   = dm.get<real const,3>("immersed_khf_halos"       );
    auto sfc_rough = dm.get<real const,2>("surface_roughness" );
    auto sfc_temp  = dm.get<real const,2>("surface_temp"      );
    auto sfc_khf   = dm.get<real const,2>("surface_khf"       );
    int  hs        = (imm_prop.extent(2)-nx)/2;

    real3d tend_u("tend_u",nz,ny,nx);
    real3d tend_v("tend_v",nz,ny,nx);
    real3d tend_w("tend_w",nz,ny,nx);
    real3d tend_T("tend_T",nz,ny,nx);

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      tend_u(k,j,i) = 0;
      tend_v(k,j,i) = 0;
      tend_w(k,j,i) = 0;
      tend_T(k,j,i) = 0;
    });

    // Fluxes at the bottom surface
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
      int k = 0;
      if (imm_prop(hs+k,hs+j,hs+i) < 1) {
        real roughness = sfc_rough(j,i);                  // Roughness length
        real vk   = 0.40;                                 // von karmann constant
        real lgz  = std::log((dz/2+roughness)/roughness); // log term
        real c_dz = vk*vk/(lgz*lgz);                      // Multiplier
        real r    = dm_r  (k,j,i);                        // Density
        real u    = dm_u  (k,j,i);                        // u-velocity
        real v    = dm_v  (k,j,i);                        // v-velocity
        real T    = dm_T  (k,j,i);                        // temperature
        real T0   = sfc_temp(j,i);                        // Density
        real hf   = sfc_khf (j,i)*cp_d*r;                 // heat flux
        real mag  = std::sqrt(u*u+v*v);                   // transvers velocity magnitude
        yakl::atomicAdd( tend_u(k,j,i) , -c_dz*(u-0 )*mag/dz );
        yakl::atomicAdd( tend_v(k,j,i) , -c_dz*(v-0 )*mag/dz );
        yakl::atomicAdd( tend_T(k,j,i) , -c_dz*(T-T0)*mag/dz );
        yakl::atomicAdd( tend_T(k,j,i) ,        hf       /dz );
      }
    });

    // Fluxes at immersed interfaces
    parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,{-1,ny},{-1,nx}) , YAKL_LAMBDA (int k, int j, int i) {
      if (imm_prop(hs+k,hs+j,hs+i) > 0 && imm_rough(hs+k,hs+j,hs+i) > 0) {
        real roughness = imm_rough(hs+k,hs+j,hs+i);
        real vk   = 0.40;
        real lgx  = std::log((dx/2+roughness)/roughness);
        real lgy  = std::log((dy/2+roughness)/roughness);
        real lgz  = std::log((dz/2+roughness)/roughness);
        real c_dx = vk*vk/(lgx*lgx);
        real c_dy = vk*vk/(lgy*lgy);
        real c_dz = vk*vk/(lgz*lgz);
        if (i >= 0 && i < nx && j >= 0 && j < ny && k-1 >= 0 && k-1 < nz && imm_prop(hs+k-1,hs+j,hs+i) < 1) {
          real r   = dm_r(k-1,j,i);
          real u   = dm_u(k-1,j,i);
          real v   = dm_v(k-1,j,i);
          real T   = dm_T(k-1,j,i);
          real T0  = imm_temp(hs+k,hs+j,hs+i);  if (T0 == 0) T0 = T;
          real hf  = imm_khf (hs+k,hs+j,hs+i)*cp_d*r;
          real mag = std::sqrt(u*u+v*v);
          yakl::atomicAdd( tend_u(k-1,j,i) , -c_dz*(u-0 )*mag/dz );
          yakl::atomicAdd( tend_v(k-1,j,i) , -c_dz*(v-0 )*mag/dz );
          yakl::atomicAdd( tend_T(k-1,j,i) , -c_dz*(T-T0)*mag/dz );
          yakl::atomicAdd( tend_T(k-1,j,i) ,        hf       /dz );
        }
        if (i >= 0 && i < nx && j >= 0 && j < ny && k+1 >= 0 && k+1 < nz && imm_prop(hs+k+1,hs+j,hs+i) < 1) {
          real r   = dm_r(k+1,j,i);
          real u   = dm_u(k+1,j,i);
          real v   = dm_v(k+1,j,i);
          real T   = dm_T(k+1,j,i);
          real T0  = imm_temp(hs+k,hs+j,hs+i);  if (T0 == 0) T0 = T;
          real hf  = imm_khf (hs+k,hs+j,hs+i)*cp_d*r;
          real mag = std::sqrt(u*u+v*v);
          yakl::atomicAdd( tend_u(k+1,j,i) , -c_dz*(u-0 )*mag/dz );
          yakl::atomicAdd( tend_v(k+1,j,i) , -c_dz*(v-0 )*mag/dz );
          yakl::atomicAdd( tend_T(k+1,j,i) , -c_dz*(T-T0)*mag/dz );
          yakl::atomicAdd( tend_T(k+1,j,i) ,        hf       /dz );
        }
        if (i >= 0 && i < nx && j-1 >= 0 && j-1 < ny && k >= 0 && k < nz && imm_prop(hs+k,hs+j-1,hs+i) < 1) {
          real r   = dm_r(k,j-1,i);
          real u   = dm_u(k,j-1,i);
          real w   = dm_w(k,j-1,i);
          real T   = dm_T(k,j-1,i);
          real T0  = imm_temp(hs+k,hs+j,hs+i);  if (T0 == 0) T0 = T;
          real hf  = imm_khf (hs+k,hs+j,hs+i)*cp_d*r;
          real mag = std::sqrt(u*u+w*w);
          yakl::atomicAdd( tend_u(k,j-1,i) , -c_dz*(u-0 )*mag/dy );
          yakl::atomicAdd( tend_w(k,j-1,i) , -c_dz*(w-0 )*mag/dy );
          yakl::atomicAdd( tend_T(k,j-1,i) , -c_dz*(T-T0)*mag/dy );
          yakl::atomicAdd( tend_T(k,j-1,i) ,        hf       /dy );
        }
        if (i >= 0 && i < nx && j+1 >= 0 && j+1 < ny && k >= 0 && k < nz && imm_prop(hs+k,hs+j+1,hs+i) < 1) {
          real r   = dm_r(k,j+1,i);
          real u   = dm_u(k,j+1,i);
          real w   = dm_w(k,j+1,i);
          real T   = dm_T(k,j+1,i);
          real T0  = imm_temp(hs+k,hs+j,hs+i);  if (T0 == 0) T0 = T;
          real hf  = imm_khf (hs+k,hs+j,hs+i)*cp_d*r;
          real mag = std::sqrt(u*u+w*w);
          yakl::atomicAdd( tend_u(k,j+1,i) , -c_dz*(u-0 )*mag/dy );
          yakl::atomicAdd( tend_w(k,j+1,i) , -c_dz*(w-0 )*mag/dy );
          yakl::atomicAdd( tend_T(k,j+1,i) , -c_dz*(T-T0)*mag/dy );
          yakl::atomicAdd( tend_T(k,j+1,i) ,        hf       /dy );
        }
        if (i-1 >= 0 && i-1 < nx && j >= 0 && j < ny && k >= 0 && k < nz && imm_prop(hs+k,hs+j,hs+i-1) < 1) {
          real r   = dm_r(k,j,i-1);
          real v   = dm_v(k,j,i-1);
          real w   = dm_w(k,j,i-1);
          real T   = dm_T(k,j,i-1);
          real T0  = imm_temp(hs+k,hs+j,hs+i);  if (T0 == 0) T0 = T;
          real hf  = imm_khf (hs+k,hs+j,hs+i)*cp_d*r;
          real mag = std::sqrt(v*v+w*w);
          yakl::atomicAdd( tend_v(k,j,i-1) , -c_dz*(v-0 )*mag/dx );
          yakl::atomicAdd( tend_w(k,j,i-1) , -c_dz*(w-0 )*mag/dx );
          yakl::atomicAdd( tend_T(k,j,i-1) , -c_dz*(T-T0)*mag/dx );
          yakl::atomicAdd( tend_T(k,j,i-1) ,        hf       /dx );
        }
        if (i+1 >= 0 && i+1 < nx && j >= 0 && j < ny && k >= 0 && k < nz && imm_prop(hs+k,hs+j,hs+i+1) < 1) {
          real r   = dm_r(k,j,i+1);
          real v   = dm_v(k,j,i+1);
          real w   = dm_w(k,j,i+1);
          real T   = dm_T(k,j,i+1);
          real T0  = imm_temp(hs+k,hs+j,hs+i);  if (T0 == 0) T0 = T;
          real hf  = imm_khf (hs+k,hs+j,hs+i)*cp_d*r;
          real mag = std::sqrt(v*v+w*w);
          yakl::atomicAdd( tend_v(k,j,i+1) , -c_dz*(v-0 )*mag/dx );
          yakl::atomicAdd( tend_w(k,j,i+1) , -c_dz*(w-0 )*mag/dx );
          yakl::atomicAdd( tend_T(k,j,i+1) , -c_dz*(T-T0)*mag/dx );
          yakl::atomicAdd( tend_T(k,j,i+1) ,        hf       /dx );
        }
      }
    });

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      dm_u(k,j,i) += dt*tend_u(k,j,i);
      dm_v(k,j,i) += dt*tend_v(k,j,i);
      dm_w(k,j,i) += dt*tend_w(k,j,i);
      dm_T(k,j,i) += dt*tend_T(k,j,i);
    });

  };

}

