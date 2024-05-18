
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
    auto sfc_temp  = dm.get<real const,2>("surface_temp_halos"       );
    int  hs        = 1;

    core::MultiField<real,3> field;
    field.add_field( dm_T );
    auto field_halo = coupler.create_and_exchange_halos( field , hs );
    auto temp = field_halo.get_field(0);
    parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(hs,ny+2*hs,nx+2*hs) , YAKL_LAMBDA (int kk, int j, int i) {
      temp(      kk,j,i) = sfc_temp(j,i) == 0 ? temp(hs,j,i) : sfc_temp(j,i);
      temp(hs+nz+kk,j,i) = temp(hs+nz-1,j,i);
    });

    real3d tend_u("tend_u",nz,ny,nx);
    real3d tend_v("tend_v",nz,ny,nx);
    real3d tend_w("tend_w",nz,ny,nx);
    real3d tend_T("tend_T",nz,ny,nx);

    real vk = 0.40;   // von karman constant

    parallel_for( YAKL_AUTO_LABEL() , Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      real r = dm_r(k,j,i);
      real u = dm_u(k,j,i);
      real v = dm_v(k,j,i);
      real w = dm_w(k,j,i);
      real T = dm_T(k,j,i);
      tend_u(k,j,i) = 0;
      tend_v(k,j,i) = 0;
      tend_w(k,j,i) = 0;
      tend_T(k,j,i) = 0;
      int indk, indj, indi;
      indk = hs+k;  indj = hs+j;  indi = hs+i-1;
      if (imm_prop(indk,indj,indi) > 0) {
        real roughness = imm_rough(indk,indj,indi);
        real lgx  = std::log((dx/2+roughness)/roughness);
        real c_dx = vk*vk/(lgx*lgx);
        real T0  = imm_temp(indk,indj,indi);  if (T0 == 0) T0 = temp(indk,indj,indi);
        real hf  = imm_khf (indk,indj,indi)*cp_d*r;
        real mag = std::sqrt(v*v+w*w);
        tend_v(k,j,i) += -c_dx*(v-0 )*mag/dx;
        tend_w(k,j,i) += -c_dx*(w-0 )*mag/dx;
        tend_T(k,j,i) += -c_dx*(T-T0)*mag/dx;
        tend_T(k,j,i) +=        hf       /dx;
      }
      indk = hs+k;  indj = hs+j;  indi = hs+i+1;
      if (imm_prop(indk,indj,indi) > 0) {
        real roughness = imm_rough(indk,indj,indi);
        real lgx  = std::log((dx/2+roughness)/roughness);
        real c_dx = vk*vk/(lgx*lgx);
        real T0  = imm_temp(indk,indj,indi);  if (T0 == 0) T0 = temp(indk,indj,indi);
        real hf  = imm_khf (indk,indj,indi)*cp_d*r;
        real mag = std::sqrt(v*v+w*w);
        tend_v(k,j,i) += -c_dx*(v-0 )*mag/dx;
        tend_w(k,j,i) += -c_dx*(w-0 )*mag/dx;
        tend_T(k,j,i) += -c_dx*(T-T0)*mag/dx;
        tend_T(k,j,i) +=        hf       /dx;
      }
      indk = hs+k;  indj = hs+j-1;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        real roughness = imm_rough(indk,indj,indi);
        real lgy  = std::log((dy/2+roughness)/roughness);
        real c_dy = vk*vk/(lgy*lgy);
        real T0  = imm_temp(indk,indj,indi);  if (T0 == 0) T0 = temp(indk,indj,indi);
        real hf  = imm_khf (indk,indj,indi)*cp_d*r;
        real mag = std::sqrt(u*u+w*w);
        tend_u(k,j,i) += -c_dy*(u-0 )*mag/dy;
        tend_w(k,j,i) += -c_dy*(w-0 )*mag/dy;
        tend_T(k,j,i) += -c_dy*(T-T0)*mag/dy;
        tend_T(k,j,i) +=        hf       /dy;
      }
      indk = hs+k;  indj = hs+j+1;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        real roughness = imm_rough(indk,indj,indi);
        real lgy  = std::log((dy/2+roughness)/roughness);
        real c_dy = vk*vk/(lgy*lgy);
        real T0  = imm_temp(indk,indj,indi);  if (T0 == 0) T0 = temp(indk,indj,indi);
        real hf  = imm_khf (indk,indj,indi)*cp_d*r;
        real mag = std::sqrt(u*u+w*w);
        tend_u(k,j,i) += -c_dy*(u-0 )*mag/dy;
        tend_w(k,j,i) += -c_dy*(w-0 )*mag/dy;
        tend_T(k,j,i) += -c_dy*(T-T0)*mag/dy;
        tend_T(k,j,i) +=        hf       /dy;
      }
      indk = hs+k-1;  indj = hs+j;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        real roughness = imm_rough(indk,indj,indi);
        real lgz  = std::log((dz/2+roughness)/roughness);
        real c_dz = vk*vk/(lgz*lgz);
        real T0  = imm_temp(indk,indj,indi);  if (T0 == 0) T0 = temp(indk,indj,indi);
        real hf  = imm_khf (indk,indj,indi)*cp_d*r;
        real mag = std::sqrt(u*u+v*v);
        tend_u(k,j,i) += -c_dz*(u-0 )*mag/dz;
        tend_v(k,j,i) += -c_dz*(v-0 )*mag/dz;
        tend_T(k,j,i) += -c_dz*(T-T0)*mag/dz;
        tend_T(k,j,i) +=        hf       /dz;
      }
      indk = hs+k+1;  indj = hs+j;  indi = hs+i;
      if (imm_prop(indk,indj,indi) > 0) {
        real roughness = imm_rough(indk,indj,indi);
        real lgz  = std::log((dz/2+roughness)/roughness);
        real c_dz = vk*vk/(lgz*lgz);
        real T0  = imm_temp(indk,indj,indi);  if (T0 == 0) T0 = temp(indk,indj,indi);
        real hf  = imm_khf (indk,indj,indi)*cp_d*r;
        real mag = std::sqrt(u*u+v*v);
        tend_u(k,j,i) += -c_dz*(u-0 )*mag/dz;
        tend_v(k,j,i) += -c_dz*(v-0 )*mag/dz;
        tend_T(k,j,i) += -c_dz*(T-T0)*mag/dz;
        tend_T(k,j,i) +=        hf       /dz;
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

