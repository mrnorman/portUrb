
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  struct LES_Closure {
    int static constexpr hs        = 1;
    int static constexpr num_state = 5;
    int static constexpr idR = 0;
    int static constexpr idU = 1;
    int static constexpr idV = 2;
    int static constexpr idW = 3;
    int static constexpr idT = 4;


    void init( core::Coupler &coupler ) const {
      auto nx  = coupler.get_nx  ();
      auto ny  = coupler.get_ny  ();
      auto nz  = coupler.get_nz  ();
      auto &dm = coupler.get_data_manager_readwrite();
      coupler.add_tracer( "TKE" , "mass-weighted TKE" , true , false , false );
      dm.get<real,3>("TKE") = 0.1;
    }



    void apply( core::Coupler &coupler , real dtphys ) const {
      using yikl::parallel_for;
      using yikl::SimpleBounds;
      auto nx             = coupler.get_nx  ();
      auto ny             = coupler.get_ny  ();
      auto nz             = coupler.get_nz  ();
      auto dx             = coupler.get_dx  ();
      auto dy             = coupler.get_dy  ();
      auto dz             = coupler.get_dz  ();
      auto enable_gravity = coupler.get_option<bool>("enable_gravity" , true );
      auto grav           = coupler.get_option<real>("grav");
      auto nu             = coupler.get_option<real>("kinematic_viscosity",0);
      auto dns            = coupler.get_option<bool>("dns",false);
      auto &dm            = coupler.get_data_manager_readwrite();
      real delta          = std::pow( dx*dy*dz , 1._fp/3._fp );
      auto immersed       = dm.get<real const,3>("immersed_proportion_halos");
      real constexpr Pr = 0.7;

      real4d state , tracers;
      real3d tke;
      convert_coupler_to_dynamics( coupler , state , tracers , tke );
      auto num_tracers = tracers.extent(0);

      core::MultiField<real,3> fields;
      for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<3>(l,0,0,0) ); }
      for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<3>(l,0,0,0) ); }
      fields.add_field( tke );
      coupler.halo_exchange( fields , hs );
      halo_bcs( coupler , state , tracers , tke );

      real3d flux_ru_x     ("flux_ru_x"                 ,nz  ,ny  ,nx+1);
      real3d flux_rv_x     ("flux_rv_x"                 ,nz  ,ny  ,nx+1);
      real3d flux_rw_x     ("flux_rw_x"                 ,nz  ,ny  ,nx+1);
      real3d flux_rt_x     ("flux_rt_x"                 ,nz  ,ny  ,nx+1);
      real3d flux_tke_x    ("flux_tke_x"                ,nz  ,ny  ,nx+1);
      real4d flux_tracers_x("flux_tracers_x",num_tracers,nz  ,ny  ,nx+1);
      real3d flux_ru_y     ("flux_ru_y"                 ,nz  ,ny+1,nx  );
      real3d flux_rv_y     ("flux_rv_y"                 ,nz  ,ny+1,nx  );
      real3d flux_rw_y     ("flux_rw_y"                 ,nz  ,ny+1,nx  );
      real3d flux_rt_y     ("flux_rt_y"                 ,nz  ,ny+1,nx  );
      real3d flux_tke_y    ("flux_tke_y"                ,nz  ,ny+1,nx  );
      real4d flux_tracers_y("flux_tracers_y",num_tracers,nz  ,ny+1,nx  );
      real3d flux_ru_z     ("flux_ru_z"                 ,nz+1,ny  ,nx  );
      real3d flux_rv_z     ("flux_rv_z"                 ,nz+1,ny  ,nx  );
      real3d flux_rw_z     ("flux_rw_z"                 ,nz+1,ny  ,nx  );
      real3d flux_rt_z     ("flux_rt_z"                 ,nz+1,ny  ,nx  );
      real3d flux_tke_z    ("flux_tke_z"                ,nz+1,ny  ,nx  );
      real4d flux_tracers_z("flux_tracers_z",num_tracers,nz+1,ny  ,nx  );
      real3d tke_source    ("tke_source"                ,nz  ,ny  ,nx  );

      real visc_max_x = 0.25_fp*dx*dx/dtphys;
      real visc_max_y = 0.25_fp*dy*dy/dtphys;
      real visc_max_z = 0.25_fp*dz*dz/dtphys;

      // Buoyancy source
      // TKE dissipation
      // Shear production

      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny+1,nx+1) , KOKKOS_LAMBDA (int k, int j, int i) {
        if (j < ny && k < nz) {
          if (immersed(hs+k,hs+j,hs+i-1) == 1 || immersed(hs+k,hs+j,hs+i) == 1) {
            flux_ru_x (k,j,i) = 0;
            flux_rv_x (k,j,i) = 0;
            flux_rw_x (k,j,i) = 0;
            flux_rt_x (k,j,i) = 0;
            flux_tke_x(k,j,i) = 0;
            for (int tr=0; tr < num_tracers; tr++) { flux_tracers_x(tr,k,j,i) = 0; }
          } else {
            // Derivatives valid at interface i-1/2
            real du_dz = 0.5_fp * ( (state(idU,hs+k+1,hs+j,hs+i-1)-state(idU,hs+k-1,hs+j,hs+i-1))/(2*dz) +
                                    (state(idU,hs+k+1,hs+j,hs+i  )-state(idU,hs+k-1,hs+j,hs+i  ))/(2*dz) );
            real dw_dz = 0.5_fp * ( (state(idW,hs+k+1,hs+j,hs+i-1)-state(idW,hs+k-1,hs+j,hs+i-1))/(2*dz) +
                                    (state(idW,hs+k+1,hs+j,hs+i  )-state(idW,hs+k-1,hs+j,hs+i  ))/(2*dz) );
            real dt_dz = 0.5_fp * ( (state(idT,hs+k+1,hs+j,hs+i-1)-state(idT,hs+k-1,hs+j,hs+i-1))/(2*dz) +
                                    (state(idT,hs+k+1,hs+j,hs+i  )-state(idT,hs+k-1,hs+j,hs+i  ))/(2*dz) );
            real du_dy = 0.5_fp * ( (state(idU,hs+k,hs+j+1,hs+i-1)-state(idU,hs+k,hs+j-1,hs+i-1))/(2*dy) +
                                    (state(idU,hs+k,hs+j+1,hs+i  )-state(idU,hs+k,hs+j-1,hs+i  ))/(2*dy) );
            real dv_dy = 0.5_fp * ( (state(idV,hs+k,hs+j+1,hs+i-1)-state(idV,hs+k,hs+j-1,hs+i-1))/(2*dy) +
                                    (state(idV,hs+k,hs+j+1,hs+i  )-state(idV,hs+k,hs+j-1,hs+i  ))/(2*dy) );
            real du_dx = (state(idU,hs+k,hs+j,hs+i) - state(idU,hs+k,hs+j,hs+i-1))/dx;
            real dv_dx = (state(idV,hs+k,hs+j,hs+i) - state(idV,hs+k,hs+j,hs+i-1))/dx;
            real dw_dx = (state(idW,hs+k,hs+j,hs+i) - state(idW,hs+k,hs+j,hs+i-1))/dx;
            real dt_dx = (state(idT,hs+k,hs+j,hs+i) - state(idT,hs+k,hs+j,hs+i-1))/dx;
            real dK_dx = (tke      (hs+k,hs+j,hs+i) - tke      (hs+k,hs+j,hs+i-1))/dx;
            // Quantities at interface i-1/2
            real rho         = 0.5_fp * ( state(idR,hs+k,hs+j,hs+i-1) + state(idR,hs+k,hs+j,hs+i) );
            real K           = 0.5_fp * ( tke      (hs+k,hs+j,hs+i-1) + tke      (hs+k,hs+j,hs+i) );
            real t           = 0.5_fp * ( state(idT,hs+k,hs+j,hs+i-1) + state(idT,hs+k,hs+j,hs+i) );
            real N           = dt_dz >= 0 && enable_gravity ? std::sqrt(grav/t*dt_dz) : 0;
            real ell         = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20_fp) , delta );
            real km          = 0.1_fp * ell * std::sqrt(K);
            real Pr_t        = delta / (1+2*ell);
            real visc_tot    = dns ? nu : std::min( km+nu         , 0.5_fp*visc_max_x );
            real visc_tot_th = dns ? nu : std::min( km/Pr_t+nu/Pr , 0.5_fp*visc_max_x );
            flux_ru_x (k,j,i) = -rho*visc_tot   *(du_dx + du_dx - 2._fp/3._fp*(du_dx+dv_dy+dw_dz));
            flux_rv_x (k,j,i) = -rho*visc_tot   *(dv_dx + du_dy                                  );
            flux_rw_x (k,j,i) = -rho*visc_tot   *(dw_dx + du_dz                                  );
            flux_rt_x (k,j,i) = -rho*visc_tot_th*(dt_dx                                          );
            flux_tke_x(k,j,i) = -rho*visc_tot*2 *(dK_dx                                          );
            for (int tr=0; tr < num_tracers; tr++) {
              dt_dx = (tracers(tr,hs+k,hs+j,hs+i) - tracers(tr,hs+k,hs+j,hs+i-1))/dx;
              flux_tracers_x(tr,k,j,i) = -rho*visc_tot*dt_dx;
            }
          }
        }
        if (i < nx && k < nz) {
          if (immersed(hs+k,hs+j-1,hs+i) == 1 || immersed(hs+k,hs+j,hs+i) == 1) {
            flux_ru_y (k,j,i) = 0;
            flux_rv_y (k,j,i) = 0;
            flux_rw_y (k,j,i) = 0;
            flux_rt_y (k,j,i) = 0;
            flux_tke_y(k,j,i) = 0;
            for (int tr=0; tr < num_tracers; tr++) { flux_tracers_y(tr,k,j,i) = 0; }
          } else {
            // Derivatives valid at interface j-1/2
            real dv_dz = 0.5_fp * ( (state(idV,hs+k+1,hs+j-1,hs+i)-state(idV,hs+k-1,hs+j-1,hs+i))/(2*dz) +
                                    (state(idV,hs+k+1,hs+j  ,hs+i)-state(idV,hs+k-1,hs+j  ,hs+i))/(2*dz) );
            real dw_dz = 0.5_fp * ( (state(idW,hs+k+1,hs+j-1,hs+i)-state(idW,hs+k-1,hs+j-1,hs+i))/(2*dz) +
                                    (state(idW,hs+k+1,hs+j  ,hs+i)-state(idW,hs+k-1,hs+j  ,hs+i))/(2*dz) );
            real dt_dz = 0.5_fp * ( (state(idT,hs+k+1,hs+j-1,hs+i)-state(idT,hs+k-1,hs+j-1,hs+i))/(2*dz) +
                                    (state(idT,hs+k+1,hs+j  ,hs+i)-state(idT,hs+k-1,hs+j  ,hs+i))/(2*dz) );
            real du_dx = 0.5_fp * ( (state(idU,hs+k,hs+j-1,hs+i+1)-state(idU,hs+k,hs+j-1,hs+i-1))/(2*dx) +
                                    (state(idU,hs+k,hs+j  ,hs+i+1)-state(idU,hs+k,hs+j  ,hs+i-1))/(2*dx) );
            real dv_dx = 0.5_fp * ( (state(idV,hs+k,hs+j-1,hs+i+1)-state(idV,hs+k,hs+j-1,hs+i-1))/(2*dx) +
                                    (state(idV,hs+k,hs+j  ,hs+i+1)-state(idV,hs+k,hs+j  ,hs+i-1))/(2*dx) );
            real du_dy = (state(idU,hs+k,hs+j,hs+i) - state(idU,hs+k,hs+j-1,hs+i))/dy;
            real dv_dy = (state(idV,hs+k,hs+j,hs+i) - state(idV,hs+k,hs+j-1,hs+i))/dy;
            real dw_dy = (state(idW,hs+k,hs+j,hs+i) - state(idW,hs+k,hs+j-1,hs+i))/dy;
            real dt_dy = (state(idT,hs+k,hs+j,hs+i) - state(idT,hs+k,hs+j-1,hs+i))/dy;
            real dK_dy = (tke      (hs+k,hs+j,hs+i) - tke      (hs+k,hs+j-1,hs+i))/dy;
            // Quantities at interface j-1/2
            real rho  = 0.5_fp * ( state(idR,hs+k,hs+j-1,hs+i) + state(idR,hs+k,hs+j,hs+i) );
            real K    = 0.5_fp * ( tke      (hs+k,hs+j-1,hs+i) + tke      (hs+k,hs+j,hs+i) );
            real t    = 0.5_fp * ( state(idT,hs+k,hs+j-1,hs+i) + state(idT,hs+k,hs+j,hs+i) );
            real N    = dt_dz >= 0 && enable_gravity ? std::sqrt(grav/t*dt_dz) : 0;
            real ell  = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20_fp) , delta );
            real km   = 0.1_fp * ell * std::sqrt(K);
            real Pr_t = delta / (1+2*ell);
            real visc_tot    = dns ? nu : std::min( km+nu         , 0.5_fp*visc_max_y );
            real visc_tot_th = dns ? nu : std::min( km/Pr_t+nu/Pr , 0.5_fp*visc_max_y );
            flux_ru_y (k,j,i) = -rho*visc_tot   *(du_dy + dv_dx                                  );
            flux_rv_y (k,j,i) = -rho*visc_tot   *(dv_dy + dv_dy - 2._fp/3._fp*(du_dx+dv_dy+dw_dz));
            flux_rw_y (k,j,i) = -rho*visc_tot   *(dw_dy + dv_dz                                  );
            flux_rt_y (k,j,i) = -rho*visc_tot_th*(dt_dy                                          );
            flux_tke_y(k,j,i) = -rho*visc_tot*2 *(dK_dy                                          );
            for (int tr=0; tr < num_tracers; tr++) {
              dt_dy = (tracers(tr,hs+k,hs+j,hs+i) - tracers(tr,hs+k,hs+j-1,hs+i))/dy;
              flux_tracers_y(tr,k,j,i) = -rho*visc_tot*dt_dy;
            }
          }
        }
        if (i < nx && j < ny) {
          if (immersed(hs+k-1,hs+j,hs+i) == 1 || immersed(hs+k,hs+j,hs+i) == 1) {
            flux_ru_z (k,j,i) = 0;
            flux_rv_z (k,j,i) = 0;
            flux_rw_z (k,j,i) = 0;
            flux_rt_z (k,j,i) = 0;
            flux_tke_z(k,j,i) = 0;
            for (int tr=0; tr < num_tracers; tr++) { flux_tracers_z(tr,k,j,i) = 0; }
          } else {
            // Derivatives valid at interface k-1/2
            real du_dx = 0.5_fp * ( (state(idU,hs+k-1,hs+j,hs+i+1) - state(idU,hs+k-1,hs+j,hs+i-1))/(2*dx) +
                                    (state(idU,hs+k  ,hs+j,hs+i+1) - state(idU,hs+k  ,hs+j,hs+i-1))/(2*dx) );
            real dw_dx = 0.5_fp * ( (state(idW,hs+k-1,hs+j,hs+i+1) - state(idW,hs+k-1,hs+j,hs+i-1))/(2*dx) +
                                    (state(idW,hs+k  ,hs+j,hs+i+1) - state(idW,hs+k  ,hs+j,hs+i-1))/(2*dx) );
            real dv_dy = 0.5_fp * ( (state(idV,hs+k-1,hs+j+1,hs+i) - state(idV,hs+k-1,hs+j-1,hs+i))/(2*dy) +
                                    (state(idV,hs+k  ,hs+j+1,hs+i) - state(idV,hs+k  ,hs+j-1,hs+i))/(2*dy) );
            real dw_dy = 0.5_fp * ( (state(idW,hs+k-1,hs+j+1,hs+i) - state(idW,hs+k-1,hs+j-1,hs+i))/(2*dy) +
                                    (state(idW,hs+k  ,hs+j+1,hs+i) - state(idW,hs+k  ,hs+j-1,hs+i))/(2*dy) );
            real du_dz = (state(idU,hs+k,hs+j,hs+i) - state(idU,hs+k-1,hs+j,hs+i))/dz;
            real dv_dz = (state(idV,hs+k,hs+j,hs+i) - state(idV,hs+k-1,hs+j,hs+i))/dz;
            real dw_dz = (state(idW,hs+k,hs+j,hs+i) - state(idW,hs+k-1,hs+j,hs+i))/dz;
            real dt_dz = (state(idT,hs+k,hs+j,hs+i) - state(idT,hs+k-1,hs+j,hs+i))/dz;
            real dK_dz = (tke      (hs+k,hs+j,hs+i) - tke      (hs+k-1,hs+j,hs+i))/dz;
            // Quantities at interface k-1/2
            real rho  = 0.5_fp * ( state(idR,hs+k-1,hs+j,hs+i) + state(idR,hs+k,hs+j,hs+i) );
            real K    = 0.5_fp * ( tke      (hs+k-1,hs+j,hs+i) + tke      (hs+k,hs+j,hs+i) );
            real t    = 0.5_fp * ( state(idT,hs+k-1,hs+j,hs+i) + state(idT,hs+k,hs+j,hs+i) );
            real N    = dt_dz >= 0 && enable_gravity ? std::sqrt(grav/t*dt_dz) : 0;
            real ell  = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20_fp) , delta );
            real km   = 0.1_fp * ell * std::sqrt(K);
            real Pr_t = delta / (1+2*ell);
            real visc_tot    = dns ? nu : std::min( km+nu         , 0.5_fp*visc_max_z );
            real visc_tot_th = dns ? nu : std::min( km/Pr_t+nu/Pr , 0.5_fp*visc_max_z );
            flux_ru_z (k,j,i) = -rho*visc_tot   *(du_dz + dw_dx                                  );
            flux_rv_z (k,j,i) = -rho*visc_tot   *(dv_dz + dw_dy                                  );
            flux_rw_z (k,j,i) = -rho*visc_tot   *(dw_dz + dw_dz - 2._fp/3._fp*(du_dx+dv_dy+dw_dz));
            flux_rt_z (k,j,i) = -rho*visc_tot_th*(dt_dz                                          );
            flux_tke_z(k,j,i) = -rho*visc_tot*2 *(dK_dz                                          );
            for (int tr=0; tr < num_tracers; tr++) {
              dt_dz = (tracers(tr,hs+k,hs+j,hs+i) - tracers(tr,hs+k-1,hs+j,hs+i))/dz;
              flux_tracers_z(tr,k,j,i) = -rho*visc_tot*dt_dz;
            }
          }
        }
      });

      halo_bcs( coupler , state , tracers , tke );

      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho   = state(idR,hs+k,hs+j,hs+i);
        real K     = tke      (hs+k,hs+j,hs+i);
        real t     = state(idT,hs+k,hs+j,hs+i);
        real dt_dz = ( state(idT,hs+k+1,hs+j,hs+i) - state(idT,hs+k-1,hs+j,hs+i) ) / (2*dz);
        real N     = dt_dz >= 0 && enable_gravity ? std::sqrt(grav/t*dt_dz) : 0;
        real ell   = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20_fp) , delta );
        real km    = 0.1_fp * ell * std::sqrt(K);
        real Pr_t  = delta / (1+2*ell);
        // Compute tke cell-averaged source
        tke_source(k,j,i) = 0;
        if (immersed(hs+k,hs+j,hs+i) < 1) {
          // Buoyancy source
          if (enable_gravity) {
            tke_source  (k,j,i) += -(grav*rho*km)/(t*Pr_t)*dt_dz;
          }
          // TKE dissipation
          tke_source  (k,j,i) -=  rho*(0.19_fp + 0.51_fp*ell/delta)/delta*std::pow(K,1.5_fp);
          // Shear production
          int im1 = immersed(hs+k,hs+j,hs+i-1) > 0 ? i : i-1;
          int ip1 = immersed(hs+k,hs+j,hs+i+1) > 0 ? i : i+1;
          int jm1 = immersed(hs+k,hs+j-1,hs+i) > 0 ? j : j-1;
          int jp1 = immersed(hs+k,hs+j+1,hs+i) > 0 ? j : j+1;
          int km1 = immersed(hs+k-1,hs+j,hs+i) > 0 ? k : k-1;
          int kp1 = immersed(hs+k+1,hs+j,hs+i) > 0 ? k : k+1;
          real du_dx = ( state(idU,hs+k,hs+j,hs+i+1) - state(idU,hs+k,hs+j,hs+i-1) ) / (2*dx);
          real dv_dx = ( state(idV,hs+k,hs+j,hs+ip1) - state(idV,hs+k,hs+j,hs+im1) ) / (2*dx);
          real dw_dx = ( state(idW,hs+k,hs+j,hs+ip1) - state(idW,hs+k,hs+j,hs+im1) ) / (2*dx);
          real du_dy = ( state(idU,hs+k,hs+jp1,hs+i) - state(idU,hs+k,hs+jm1,hs+i) ) / (2*dy);
          real dv_dy = ( state(idV,hs+k,hs+j+1,hs+i) - state(idV,hs+k,hs+j-1,hs+i) ) / (2*dy);
          real dw_dy = ( state(idW,hs+k,hs+jp1,hs+i) - state(idW,hs+k,hs+jm1,hs+i) ) / (2*dy);
          real du_dz = ( state(idU,hs+kp1,hs+j,hs+i) - state(idU,hs+km1,hs+j,hs+i) ) / (2*dz);
          real dv_dz = ( state(idV,hs+kp1,hs+j,hs+i) - state(idV,hs+km1,hs+j,hs+i) ) / (2*dz);
          real dw_dz = ( state(idW,hs+k+1,hs+j,hs+i) - state(idW,hs+k-1,hs+j,hs+i) ) / (2*dz);
          real j1_i1 = (du_dx + du_dx) * du_dx;
          real j1_i2 = (dv_dx + du_dy) * dv_dx;
          real j1_i3 = (dw_dx + du_dz) * dw_dx;
          real j2_i1 = (du_dy + dv_dx) * du_dy;
          real j2_i2 = (dv_dy + dv_dy) * dv_dy;
          real j2_i3 = (dw_dy + dv_dz) * dw_dy;
          real j3_i1 = (du_dz + dw_dx) * du_dz;
          real j3_i2 = (dv_dz + dw_dy) * dv_dz;
          real j3_i3 = (dw_dz + dw_dz) * dw_dz;
          tke_source   (k,j,i) += rho*km*(j1_i1 + j1_i2 + j1_i3 + j2_i1 + j2_i2 + j2_i3 + j3_i1 + j3_i2 + j3_i3);
        }
      });

      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real tend_ru  = -(flux_ru_x (k,j,i+1) - flux_ru_x (k,j,i)) / dx -
                         (flux_ru_y (k,j+1,i) - flux_ru_y (k,j,i)) / dy -
                         (flux_ru_z (k+1,j,i) - flux_ru_z (k,j,i)) / dz;
        real tend_rv  = -(flux_rv_x (k,j,i+1) - flux_rv_x (k,j,i)) / dx -
                         (flux_rv_y (k,j+1,i) - flux_rv_y (k,j,i)) / dy -
                         (flux_rv_z (k+1,j,i) - flux_rv_z (k,j,i)) / dz;
        real tend_rw  = -(flux_rw_x (k,j,i+1) - flux_rw_x (k,j,i)) / dx -
                         (flux_rw_y (k,j+1,i) - flux_rw_y (k,j,i)) / dy -
                         (flux_rw_z (k+1,j,i) - flux_rw_z (k,j,i)) / dz;
        real tend_rt  = -(flux_rt_x (k,j,i+1) - flux_rt_x (k,j,i)) / dx -
                         (flux_rt_y (k,j+1,i) - flux_rt_y (k,j,i)) / dy -
                         (flux_rt_z (k+1,j,i) - flux_rt_z (k,j,i)) / dz;
        real tend_tke = -(flux_tke_x(k,j,i+1) - flux_tke_x(k,j,i)) / dx -
                         (flux_tke_y(k,j+1,i) - flux_tke_y(k,j,i)) / dy -
                         (flux_tke_z(k+1,j,i) - flux_tke_z(k,j,i)) / dz + tke_source(k,j,i);

        state(idU,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        state(idU,hs+k,hs+j,hs+i) += dtphys * tend_ru ;

        state(idV,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        state(idV,hs+k,hs+j,hs+i) += dtphys * tend_rv ;

        state(idW,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        state(idW,hs+k,hs+j,hs+i) += dtphys * tend_rw ;

        state(idT,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        state(idT,hs+k,hs+j,hs+i) += dtphys * tend_rt ;

        tke      (hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
        tke      (hs+k,hs+j,hs+i) += dtphys * tend_tke;
        tke      (hs+k,hs+j,hs+i) = std::max( 0._fp , tke(hs+k,hs+j,hs+i) );

        for (int tr=0; tr < num_tracers; tr++) {
          real tend_tracer = -(flux_tracers_x(tr,k,j,i+1) - flux_tracers_x(tr,k,j,i)) / dx -
                              (flux_tracers_y(tr,k,j+1,i) - flux_tracers_y(tr,k,j,i)) / dy -
                              (flux_tracers_z(tr,k+1,j,i) - flux_tracers_z(tr,k,j,i)) / dz;
          tracers(tr,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i);
          tracers(tr,hs+k,hs+j,hs+i) += dtphys * tend_tracer;
        }
      });

      convert_dynamics_to_coupler( coupler , state , tracers , tke );
    }



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ,
                                      real3d              &tke     ) const {
      using yikl::parallel_for;
      using yikl::SimpleBounds;
      auto nx           = coupler.get_nx();
      auto ny           = coupler.get_ny();
      auto nz           = coupler.get_nz();
      auto R_d          = coupler.get_option<real>("R_d"    );
      auto R_v          = coupler.get_option<real>("R_v"    );
      auto gamma        = coupler.get_option<real>("gamma_d");
      auto C0           = coupler.get_option<real>("C0"     );
      auto &dm          = coupler.get_data_manager_readonly();
      auto tracer_names = coupler.get_tracer_names();
      auto dm_rho_d     = dm.get<real const,3>("density_dry");
      auto dm_uvel      = dm.get<real const,3>("uvel"       );
      auto dm_vvel      = dm.get<real const,3>("vvel"       );
      auto dm_wvel      = dm.get<real const,3>("wvel"       );
      auto dm_temp      = dm.get<real const,3>("temp"       );
      auto dm_tke       = dm.get<real const,3>("TKE"        );
      core::MultiField<real const,3> dm_tracers;
      for (int tr=0; tr < tracer_names.size(); tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        if (diffuse) dm_tracers.add_field( dm.get<real const,3>(tracer_names[tr]) );
      }
      auto num_tracers = dm_tracers.size();
      state   = real4d("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      tracers = real4d("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      tke     = real3d("tke"                ,nz+2*hs,ny+2*hs,nx+2*hs);
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i);
        state(idR,hs+k,hs+j,hs+i) = rho_d;
        state(idU,hs+k,hs+j,hs+i) = dm_uvel(k,j,i);
        state(idV,hs+k,hs+j,hs+i) = dm_vvel(k,j,i);
        state(idW,hs+k,hs+j,hs+i) = dm_wvel(k,j,i);
        state(idT,hs+k,hs+j,hs+i) = pow( rho_d*R_d*dm_temp(k,j,i)/C0 , 1._fp / gamma ) / rho_d;
        tke      (hs+k,hs+j,hs+i) = std::max( 0._fp , dm_tke (k,j,i) / rho_d );
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i) = dm_tracers(tr,k,j,i)/rho_d; }
      });
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst4d    state   ,
                                      realConst4d    tracers ,
                                      realConst3d    tke     ) const {
      using yikl::parallel_for;
      using yikl::SimpleBounds;
      auto nx           = coupler.get_nx();
      auto ny           = coupler.get_ny();
      auto nz           = coupler.get_nz();
      auto R_d          = coupler.get_option<real>("R_d"    );
      auto R_v          = coupler.get_option<real>("R_v"    );
      auto gamma        = coupler.get_option<real>("gamma_d");
      auto C0           = coupler.get_option<real>("C0"     );
      auto &dm          = coupler.get_data_manager_readwrite();
      auto tracer_names = coupler.get_tracer_names();
      auto dm_rho_d     = dm.get<real,3>("density_dry");
      auto dm_uvel      = dm.get<real,3>("uvel"       );
      auto dm_vvel      = dm.get<real,3>("vvel"       );
      auto dm_wvel      = dm.get<real,3>("wvel"       );
      auto dm_temp      = dm.get<real,3>("temp"       );
      auto dm_tke       = dm.get<real,3>("TKE"        );
      core::MultiField<real,3> dm_tracers;
      for (int tr=0; tr < tracer_names.size(); tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        if (diffuse) dm_tracers.add_field( dm.get<real,3>(tracer_names[tr]) );
      }
      auto num_tracers = dm_tracers.size();
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real rho_d = state(idR,hs+k,hs+j,hs+i);
        dm_rho_d(k,j,i) = rho_d;
        dm_uvel (k,j,i) = state(idU,hs+k,hs+j,hs+i) / rho_d;
        dm_vvel (k,j,i) = state(idV,hs+k,hs+j,hs+i) / rho_d;
        dm_wvel (k,j,i) = state(idW,hs+k,hs+j,hs+i) / rho_d;
        dm_temp (k,j,i) = C0 * pow( state(idT,hs+k,hs+j,hs+i) , gamma ) / ( rho_d * R_d );
        dm_tke  (k,j,i) = tke(hs+k,hs+j,hs+i);
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i) = tracers(tr,hs+k,hs+j,hs+i); }
      });
    }



    void halo_bcs( core::Coupler const & coupler ,
                     real4d        const & state   ,
                     real4d        const & tracers ,
                     real3d        const & tke     ) const {
      using yikl::parallel_for;
      using yikl::SimpleBounds;
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto dz             = coupler.get_dz();
      auto num_tracers    = tracers.extent(0);
      auto px             = coupler.get_px();
      auto py             = coupler.get_py();
      auto nproc_x        = coupler.get_nproc_x();
      auto nproc_y        = coupler.get_nproc_y();
      auto &neigh         = coupler.get_neighbor_rankid_matrix();
      auto &dm            = coupler.get_data_manager_readonly();
      auto grav           = coupler.get_option<real>("grav");
      auto gamma          = coupler.get_option<real>("gamma_d");
      auto C0             = coupler.get_option<real>("C0");
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);
      auto bc_z           = coupler.get_option<std::string>("bc_z","solid_wall");
      if (!enable_gravity) grav = 0;

      if (coupler.get_option<std::string>("bc_x") == "precursor" && coupler.get_px() == 0) {
        parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,hs) , KOKKOS_LAMBDA (int k, int j, int ii) {
          for (int l=0; l < num_state  ; l++) state  (l,hs+k,hs+j,ii) = state  (l,hs+k,hs+j,hs+0);
          for (int l=0; l < num_tracers; l++) tracers(l,hs+k,hs+j,ii) = tracers(l,hs+k,hs+j,hs+0);
          tke(hs+k,hs+j,ii) = tke(hs+k,hs+j,hs+0);
        });
      }

      if (coupler.get_option<std::string>("bc_x") == "precursor" && coupler.get_px() == coupler.get_nproc_x()-1) {
        parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,hs) , KOKKOS_LAMBDA (int k, int j, int ii) {
          for (int l=0; l < num_state  ; l++) state  (l,hs+k,hs+j,hs+nx+ii) = state  (l,hs+k,hs+j,hs+nx-1);
          for (int l=0; l < num_tracers; l++) tracers(l,hs+k,hs+j,hs+nx+ii) = tracers(l,hs+k,hs+j,hs+nx-1);
          tke(hs+k,hs+j,hs+nx+ii) = tke(hs+k,hs+j,hs+nx-1);
        });
      }

      if (coupler.get_option<std::string>("bc_y") == "precursor" && coupler.get_py() == 0) {
        parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,hs,nx) , KOKKOS_LAMBDA (int k, int jj, int i) {
          for (int l=0; l < num_state  ; l++) state  (l,hs+k,jj,hs+i) = state  (l,hs+k,hs+0,hs+i);
          for (int l=0; l < num_tracers; l++) tracers(l,hs+k,jj,hs+i) = tracers(l,hs+k,hs+0,hs+i);
          tke(hs+k,jj,hs+i) = tke(hs+k,hs+0,hs+i);
        });
      }

      if (coupler.get_option<std::string>("bc_y") == "precursor" && coupler.get_py() == coupler.get_nproc_y()-1) {
        parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,hs,nx) , KOKKOS_LAMBDA (int k, int jj, int i) {
          for (int l=0; l < num_state  ; l++) state  (l,hs+k,hs+ny+jj,hs+i) = state  (l,hs+k,hs+ny-1,hs+i);
          for (int l=0; l < num_tracers; l++) tracers(l,hs+k,hs+ny+jj,hs+i) = tracers(l,hs+k,hs+ny-1,hs+i);
          tke(hs+k,hs+ny+jj,hs+i) = tke(hs+k,hs+ny-1,hs+i);
        });
      }

      // z-direction BC's
      if (bc_z == "solid_wall") {
        parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) ,
                                          KOKKOS_LAMBDA (int kk, int j, int i) {
          state(idU,      kk,j,i) = state(idU,hs+0   ,j,i);
          state(idV,      kk,j,i) = state(idV,hs+0   ,j,i);
          state(idW,      kk,j,i) = 0;
          state(idT,      kk,j,i) = state(idT,hs+0   ,j,i);
          tke  (          kk,j,i) = tke  (    hs+0   ,j,i);
          state(idU,hs+nz+kk,j,i) = state(idU,hs+nz-1,j,i);
          state(idV,hs+nz+kk,j,i) = state(idV,hs+nz-1,j,i);
          state(idW,hs+nz+kk,j,i) = 0;
          state(idT,hs+nz+kk,j,i) = state(idT,hs+nz-1,j,i);
          tke  (    hs+nz+kk,j,i) = tke  (    hs+nz-1,j,i);
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,j,i) = tracers(l,hs+0   ,j,i);
            tracers(l,hs+nz+kk,j,i) = tracers(l,hs+nz-1,j,i);
          }
          {
            int  k0       = hs;
            int  k        = k0-1-kk;
            real rho0     = state(idR,k0,j,i);
            real theta0   = state(idT,k0,j,i);
            real rho0_gm1 = std::pow(rho0  ,gamma-1);
            real theta0_g = std::pow(theta0,gamma  );
            state(idR,k,j,i) = std::pow( rho0_gm1 + grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) ,
                                         1._fp/(gamma-1) );
          }
          {
            int  k0       = hs+nz-1;
            int  k        = k0+1+kk;
            real rho0     = state(idR,k0,j,i);
            real theta0   = state(idT,k0,j,i);
            real rho0_gm1 = std::pow(rho0  ,gamma-1);
            real theta0_g = std::pow(theta0,gamma  );
            state(idR,k,j,i) = std::pow( rho0_gm1 - grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) ,
                                         1._fp/(gamma-1) );
          }
        });
      } else if (bc_z == "periodic") {
        parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) ,
                                          KOKKOS_LAMBDA (int kk, int j, int i) {
          state(idR,      kk,j,i) = state(idR,nz+kk,j,i);
          state(idU,      kk,j,i) = state(idU,nz+kk,j,i);
          state(idV,      kk,j,i) = state(idV,nz+kk,j,i);
          state(idW,      kk,j,i) = state(idW,nz+kk,j,i);
          state(idT,      kk,j,i) = state(idT,nz+kk,j,i);
          tke  (          kk,j,i) = tke  (    nz+kk,j,i);
          state(idR,hs+nz+kk,j,i) = state(idR,hs+kk,j,i);
          state(idU,hs+nz+kk,j,i) = state(idU,hs+kk,j,i);
          state(idV,hs+nz+kk,j,i) = state(idV,hs+kk,j,i);
          state(idW,hs+nz+kk,j,i) = state(idW,hs+kk,j,i);
          state(idT,hs+nz+kk,j,i) = state(idT,hs+kk,j,i);
          tke  (    hs+nz+kk,j,i) = tke  (    hs+kk,j,i);
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,j,i) = tracers(l,nz+kk,j,i);
            tracers(l,hs+nz+kk,j,i) = tracers(l,hs+kk,j,i);
          }
        });
      } else {
        Kokkos::abort("ERROR: Specified invalid bc_z in coupler options");
      }
    }



    void halo_bcs_zero_vel( core::Coupler const & coupler ,
                            real4d        const & state   ,
                            real4d        const & tracers ,
                            real3d        const & tke     ) const {
      using yikl::parallel_for;
      using yikl::SimpleBounds;
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto dz             = coupler.get_dz();
      auto num_tracers    = tracers.extent(0);
      auto px             = coupler.get_px();
      auto py             = coupler.get_py();
      auto nproc_x        = coupler.get_nproc_x();
      auto nproc_y        = coupler.get_nproc_y();
      auto &neigh         = coupler.get_neighbor_rankid_matrix();
      auto &dm            = coupler.get_data_manager_readonly();
      auto grav           = coupler.get_option<real>("grav");
      auto gamma          = coupler.get_option<real>("gamma_d");
      auto C0             = coupler.get_option<real>("C0");
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);
      auto bc_z           = coupler.get_option<std::string>("bc_z","solid_wall");
      // z-direction BC's
      if (bc_z == "solid_wall") {
        parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) ,
                                          KOKKOS_LAMBDA (int kk, int j, int i) {
          state(idU,      kk,j,i) = 0;
          state(idV,      kk,j,i) = 0;
        });
      }
    }
  };

}

