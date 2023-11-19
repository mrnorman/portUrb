
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  struct LES_Closure {
    int static constexpr hs        = 1;
    int static constexpr num_state = 5;


    void init( core::Coupler &coupler , real dt ) {
      coupler.add_tracer( "TKE" , "mass-weighted TKE" , true , false , false );
      coupler.get_data_manager_readwrite().get<real,4>("TKE") = 0;
      if (! coupler.option_exists("R_d" )) coupler.set_option<real>("R_d" ,287. );
      if (! coupler.option_exists("cp_d")) coupler.set_option<real>("cp_d",1003.);
      if (! coupler.option_exists("R_v" )) coupler.set_option<real>("R_v" ,461. );
      if (! coupler.option_exists("cp_v")) coupler.set_option<real>("cp_v",1859 );
      if (! coupler.option_exists("p0"  )) coupler.set_option<real>("p0"  ,1.e5 );
      if (! coupler.option_exists("grav")) coupler.set_option<real>("grav",9.81 );
      auto R_d  = coupler.get_option<real>("R_d" );
      auto cp_d = coupler.get_option<real>("cp_d");
      auto R_v  = coupler.get_option<real>("R_v" );
      auto cp_v = coupler.get_option<real>("cp_v");
      auto p0   = coupler.get_option<real>("p0"  );
      auto grav = coupler.get_option<real>("grav");
      if (! coupler.option_exists("cv_d"   )) coupler.set_option<real>("cv_d"   ,cp_d - R_d );
      auto cv_d = coupler.get_option<real>("cv_d");
      if (! coupler.option_exists("gamma_d")) coupler.set_option<real>("gamma_d",cp_d / cv_d);
      if (! coupler.option_exists("kappa_d")) coupler.set_option<real>("kappa_d",R_d  / cp_d);
      if (! coupler.option_exists("cv_v"   )) coupler.set_option<real>("cv_v"   ,R_v - cp_v );
      auto gamma = coupler.get_option<real>("gamma_d");
      auto kappa = coupler.get_option<real>("kappa_d");
      if (! coupler.option_exists("C0")) coupler.set_option<real>("C0" , pow( R_d * pow( p0 , -kappa ) , gamma ));
      auto C0    = coupler.get_option<real>("C0");
    }



    void apply( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx  ();
      auto ny    = coupler.get_ny  ();
      auto nz    = coupler.get_nz  ();
      auto nens  = coupler.get_nens();
      auto dx    = coupler.get_dx  ();
      auto dy    = coupler.get_dy  ();
      auto dz    = coupler.get_dz  ();
      auto &dm   = coupler.get_data_manager_readwrite();
      auto grav  = coupler.get_option<real>("grav");
      auto r     = dm.get<real,4>("density_dry");
      auto u     = dm.get<real,4>("uvel");
      auto v     = dm.get<real,4>("vvel");
      auto w     = dm.get<real,4>("wvel");
      auto T     = dm.get<real,4>("temp");
      real delta = std::pow( dx*dy*dz , 1._fp/3._fp );

      real5d state , tracers;
      real4d tke;
      convert_coupler_to_dynamics( coupler , state , tracers , tke );
      auto num_tracers = tracers.extent(0);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        state(idU,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens);
        state(idV,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens);
        state(idW,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens);
        state(idT,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens);
        tke      (hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens);
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens); }
      });

      halo_exchange( coupler , state , tracers , tke );

      real4d flux_ru_x     ("flux_ru_x"                 ,nz,ny,nx+1,nens);
      real4d flux_rv_x     ("flux_rv_x"                 ,nz,ny,nx+1,nens);
      real4d flux_rw_x     ("flux_rw_x"                 ,nz,ny,nx+1,nens);
      real4d flux_rt_x     ("flux_rt_x"                 ,nz,ny,nx+1,nens);
      real5d flux_tracers_x("flux_tracers_x",num_tracers,nz,ny,nx+1,nens);
      real4d flux_ru_y     ("flux_ru_y"                 ,nz,ny+1,nx,nens);
      real4d flux_rv_y     ("flux_rv_y"                 ,nz,ny+1,nx,nens);
      real4d flux_rw_y     ("flux_rw_y"                 ,nz,ny+1,nx,nens);
      real4d flux_rt_y     ("flux_rt_y"                 ,nz,ny+1,nx,nens);
      real5d flux_tracers_y("flux_tracers_y",num_tracers,nz,ny+1,nx,nens);
      real4d flux_ru_z     ("flux_ru_z"                 ,nz+1,ny,nx,nens);
      real4d flux_rv_z     ("flux_rv_z"                 ,nz+1,ny,nx,nens);
      real4d flux_rw_z     ("flux_rw_z"                 ,nz+1,ny,nx,nens);
      real4d flux_rt_z     ("flux_rt_z"                 ,nz+1,ny,nx,nens);
      real5d flux_tracers_z("flux_tracers_z",num_tracers,nz+1,ny,nx,nens);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz+1,ny+1,nx+1,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        if (j < ny && k < nz) {
          real rho   = 0.5_fp * ( state(idR,hs+k,hs+j,hs+i-1,iens) + state(idR,hs+k,hs+j,hs+i,iens) );
          real K     = 0.5_fp * ( tke      (hs+k,hs+j,hs+i-1,iens) + tke      (hs+k,hs+j,hs+i,iens) );
          real t     = 0.5_fp * ( state(idT,hs+k,hs+j,hs+i-1,iens) + state(idT,hs+k,hs+j,hs+i,iens) );
          real dt_dz = 0.5_fp * ( (state(idT,hs+k+1,hs+j,hs+i-1,iens)-state(idT,hs+k-1,hs+j,hs+i-1,iens))/(2*dz) +
                                  (state(idT,hs+k+1,hs+j,hs+i  ,iens)-state(idT,hs+k-1,hs+j,hs+i  ,iens))/(2*dz) );
          real N     = grav/t*dt_dz;
          real ell   = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20) , delta );
          real km    = 0.1_fp * ell * std::sqrt(K);
          real Pr    = 1+2*ell/delta;
          real du_dy = 0.5_fp * ( (state(idU,hs+k,hs+j+1,hs+i-1) - state(idU,hs+k,hs+j-1,hs+i-1))/(2*dy) +
                                  (state(idU,hs+k,hs+j+1,hs+i  ) - state(idU,hs+k,hs+j-1,hs+i  ))/(2*dy) );
          real du_dz = 0.5_fp * ( (state(idU,hs+k+1,hs+j,hs+i-1) - state(idU,hs+k-1,hs+j,hs+i-1))/(2*dz) +
                                  (state(idU,hs+k+1,hs+j,hs+i  ) - state(idU,hs+k-1,hs+j,hs+i  ))/(2*dz) );
          real du_dx = (state(idU,hs+k,hs+j,hs+i) - state(idU,hs+k,hs+j,hs+i-1))/dx;
          real dv_dx = (state(idV,hs+k,hs+j,hs+i) - state(idV,hs+k,hs+j,hs+i-1))/dx;
          real dw_dx = (state(idW,hs+k,hs+j,hs+i) - state(idW,hs+k,hs+j,hs+i-1))/dx;
          real dt_dx = (state(idT,hs+k,hs+j,hs+i) - state(idT,hs+k,hs+j,hs+i-1))/dx;
          flux_ru_x(k,j,i,iens) = -rho*km   *(du_dx + du_dx);
          flux_rv_x(k,j,i,iens) = -rho*km   *(dv_dx + du_dy);
          flux_rw_x(k,j,i,iens) = -rho*km   *(dw_dx + du_dz);
          flux_rt_x(k,j,i,iens) = -rho*km/Pr* dt_dx;
          for (int tr=0; tr < num_tracers; tr++) {
            dt_dx = (tracers(tr,hs+k,hs+j,hs+i) - tracers(tr,hs+k,hs+j,hs+i-1))/dx;
            flux_tracers_x(tr,k,j,i,iens) = -rho*km/Pr*dt_dx;
          }
        }
        if (i < nx && k < nz) {
          real rho   = 0.5_fp * ( state(idR,hs+k,hs+j-1,hs+i,iens) + state(idR,hs+k,hs+j,hs+i,iens) );
          real K     = 0.5_fp * ( tke      (hs+k,hs+j-1,hs+i,iens) + tke      (hs+k,hs+j,hs+i,iens) );
          real t     = 0.5_fp * ( state(idT,hs+k,hs+j-1,hs+i,iens) + state(idT,hs+k,hs+j,hs+i,iens) );
          real dt_dz = 0.5_fp * ( (state(idT,hs+k+1,hs+j-1,hs+i,iens)-state(idT,hs+k-1,hs+j-1,hs+i,iens))/(2*dz) +
                                  (state(idT,hs+k+1,hs+j  ,hs+i,iens)-state(idT,hs+k-1,hs+j  ,hs+i,iens))/(2*dz) );
          real N     = grav/t*dt_dz;
          real ell   = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20) , delta );
          real km    = 0.1_fp * ell * std::sqrt(K);
          real Pr    = 1+2*ell/delta;
          real dv_dx = 0.5_fp * ( (state(idV,hs+k,hs+j-1,hs+i+1) - state(idV,hs+k,hs+j-1,hs+i-1))/(2*dx) +
                                  (state(idV,hs+k,hs+j  ,hs+i+1) - state(idV,hs+k,hs+j  ,hs+i-1))/(2*dx) );
          real dv_dz = 0.5_fp * ( (state(idV,hs+k+1,hs+j-1,hs+i) - state(idV,hs+k-1,hs+j-1,hs+i))/(2*dz) +
                                  (state(idV,hs+k+1,hs+j  ,hs+i) - state(idV,hs+k-1,hs+j  ,hs+i))/(2*dz) );
          real du_dy = (state(idU,hs+k,hs+j,hs+i) - state(idU,hs+k,hs+j-1,hs+i))/dy;
          real dv_dy = (state(idV,hs+k,hs+j,hs+i) - state(idV,hs+k,hs+j-1,hs+i))/dy;
          real dw_dy = (state(idW,hs+k,hs+j,hs+i) - state(idW,hs+k,hs+j-1,hs+i))/dy;
          real dt_dy = (state(idT,hs+k,hs+j,hs+i) - state(idT,hs+k,hs+j-1,hs+i))/dy;
          flux_ru_y(k,j,i,iens) = -rho*km   *(du_dy + dv_dx);
          flux_rv_y(k,j,i,iens) = -rho*km   *(dv_dy + dv_dy);
          flux_rw_y(k,j,i,iens) = -rho*km   *(dw_dy + dv_dz);
          flux_rt_y(k,j,i,iens) = -rho*km/Pr* dt_dy;
          for (int tr=0; tr < num_tracers; tr++) {
            dt_dy = (tracers(tr,hs+k,hs+j,hs+i) - tracers(tr,hs+k,hs+j-1,hs+i))/dy;
            flux_tracers_y(tr,k,j,i,iens) = -rho*km/Pr*dt_dy;
          }
        }
        if (i < nx && j < ny) {
          real rho   = 0.5_fp * ( state(idR,hs+k-1,hs+j,hs+i,iens) + state(idR,hs+k,hs+j,hs+i,iens) );
          real K     = 0.5_fp * ( tke      (hs+k-1,hs+j,hs+i,iens) + tke      (hs+k,hs+j,hs+i,iens) );
          real t     = 0.5_fp * ( state(idT,hs+k-1,hs+j,hs+i,iens) + state(idT,hs+k,hs+j,hs+i,iens) );
          real dt_dz = (state(idT,hs+k,hs+j,hs+i,iens) - state(idT,hs+k-1,hs+j,hs+i,iens))/dz;
          real N     = grav/t*dt_dz;
          real ell   = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20) , delta );
          real km    = 0.1_fp * ell * std::sqrt(K);
          real Pr    = 1+2*ell/delta;
          real dw_dx = 0.5_fp * ( (state(idW,hs+k-1,hs+j,hs+i+1) - state(idW,hs+k-1,hs+j,hs+i-1))/(2*dx) +
                                  (state(idW,hs+k  ,hs+j,hs+i+1) - state(idW,hs+k  ,hs+j,hs+i-1))/(2*dx) );
          real dw_dy = 0.5_fp * ( (state(idW,hs+k-1,hs+j+1,hs+i) - state(idW,hs+k-1,hs+j-1,hs+i))/(2*dy) +
                                  (state(idW,hs+k  ,hs+j+1,hs+i) - state(idW,hs+k  ,hs+j-1,hs+i))/(2*dy) );
          real du_dz = (state(idU,hs+k,hs+j,hs+i) - state(idU,hs+k-1,hs+j,hs+i))/dz;
          real dv_dz = (state(idV,hs+k,hs+j,hs+i) - state(idV,hs+k-1,hs+j,hs+i))/dz;
          real dw_dz = (state(idW,hs+k,hs+j,hs+i) - state(idW,hs+k-1,hs+j,hs+i))/dz;
          flux_ru_z(k,j,i,iens) = -rho*km   *(du_dz + dw_dx);
          flux_rv_z(k,j,i,iens) = -rho*km   *(dv_dz + dw_dy);
          flux_rw_z(k,j,i,iens) = -rho*km   *(dw_dz + dw_dz);
          flux_rt_z(k,j,i,iens) = -rho*km/Pr* dt_dz;
          for (int tr=0; tr < num_tracers; tr++) {
            dt_dz = (tracers(tr,hs+k,hs+j,hs+i) - tracers(tr,hs+k-1,hs+j,hs+i))/dz;
            flux_tracers_z(tr,k,j,i,iens) = -rho*km/Pr*dt_dz;
          }
        }
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        state(idU,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
        state(idV,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
        state(idW,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
        state(idT,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
        tke      (hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens); }
      });





      convert_dynamics_to_coupler( coupler , state , tracers , tke );
    }



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real5d              &state   ,
                                      real5d              &tracers ,
                                      real4d              &tke     ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens         = coupler.get_nens();
      auto nx           = coupler.get_nx();
      auto ny           = coupler.get_ny();
      auto nz           = coupler.get_nz();
      auto R_d          = coupler.get_option<real>("R_d"    );
      auto R_v          = coupler.get_option<real>("R_v"    );
      auto gamma        = coupler.get_option<real>("gamma_d");
      auto C0           = coupler.get_option<real>("C0"     );
      auto &dm          = coupler.get_data_manager_readonly();
      auto tracer_names = coupler.get_tracer_names();
      auto dm_rho_d     = dm.get<real const,4>("density_dry");
      auto dm_uvel      = dm.get<real const,4>("uvel"       );
      auto dm_vvel      = dm.get<real const,4>("vvel"       );
      auto dm_wvel      = dm.get<real const,4>("wvel"       );
      auto dm_temp      = dm.get<real const,4>("temp"       );
      // Gather all tracer, set idTKE, and set tracer_adds_mass
      int idTKE = -1;
      core::MultiField<real const,4> dm_tracers;
      for (int tr=0; tr < tracer_names.size(); tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        if (diffuse) dm_diffuse_tracers.add_field( dm.get<real const,4>(tracer_names[tr]) );
        if (tracers[tr] == "TKE") idTKE = tr;
      }
      auto num_tracers = dm_tracers.size();
      state   = real5d("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      tracers = real5d("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      tke     = real4d("tke"                ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho_d = dm_rho_d(k,j,i,iens);
        state(idR,hs+k,hs+j,hs+i,iens) = rho_d;
        state(idU,hs+k,hs+j,hs+i,iens) = rho_d * dm_uvel (k,j,i,iens);
        state(idV,hs+k,hs+j,hs+i,iens) = rho_d * dm_vvel (k,j,i,iens);
        state(idW,hs+k,hs+j,hs+i,iens) = rho_d * dm_wvel (k,j,i,iens);
        state(idT,hs+k,hs+j,hs+i,iens) = rho_d * pow( rho_d*R_d*dm_temp(k,j,i,iens)/C0 , 1._fp / gamma ) / rho_d;
        tke      (hs+k,hs+j,hs+i,iens) = dm_tracers(idTKE,k,j,i,iens);
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) = dm_tracers(tr,k,j,i,iens); }
      });
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst5d    state   ,
                                      realConst5d    tracers ,
                                      realConst4d    tke     ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens         = coupler.get_nens();
      auto nx           = coupler.get_nx();
      auto ny           = coupler.get_ny();
      auto nz           = coupler.get_nz();
      auto R_d          = coupler.get_option<real>("R_d"    );
      auto R_v          = coupler.get_option<real>("R_v"    );
      auto gamma        = coupler.get_option<real>("gamma_d");
      auto C0           = coupler.get_option<real>("C0"     );
      auto &dm          = coupler.get_data_manager_readwrite();
      auto tracer_names = coupler.get_tracer_names();
      auto dm_rho_d     = dm.get<real,4>("density_dry");
      auto dm_uvel      = dm.get<real,4>("uvel"       );
      auto dm_vvel      = dm.get<real,4>("vvel"       );
      auto dm_wvel      = dm.get<real,4>("wvel"       );
      auto dm_temp      = dm.get<real,4>("temp"       );
      // Gather all tracer, set idTKE, and set tracer_adds_mass
      int idTKE = -1;
      core::MultiField<real,4> dm_tracers;
      for (int tr=0; tr < tracer_names.size(); tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        if (diffuse) dm_diffuse_tracers.add_field( dm.get<real,4>(tracer_names[tr]) );
        if (tracers[tr] == "TKE") idTKE = tr;
      }
      auto num_tracers = dm_tracers.size();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho_d = state(idR,hs+k,hs+j,hs+i,iens);
        dm_rho_d(k,j,i,iens) = rho_d;
        dm_uvel (k,j,i,iens) = state(idU,hs+k,hs+j,hs+i,iens) / rho_d;
        dm_vvel (k,j,i,iens) = state(idV,hs+k,hs+j,hs+i,iens) / rho_d;
        dm_wvel (k,j,i,iens) = state(idW,hs+k,hs+j,hs+i,iens) / rho_d;
        dm_temp (k,j,i,iens) = C0 * pow( state(idT,hs+k,hs+j,hs+i,iens) , gamma ) / ( rho_d * R_d );
        dm_tracers(idTKE,k,j,i,iens) = tke(hs+k,hs+j,hs+i,iens);
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens); }
      });
    }



    void halo_exchange( core::Coupler const & coupler ,
                        real5d        const & state   ,
                        real5d        const & tracers ,
                        real4d        const & tke     ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens           = coupler.get_nens();
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto dz             = coupler.get_dz();
      auto num_tracers    = coupler.get_num_tracers();
      auto px             = coupler.get_px();
      auto py             = coupler.get_py();
      auto nproc_x        = coupler.get_nproc_x();
      auto nproc_y        = coupler.get_nproc_y();
      auto bc_x           = coupler.get_option<int >("bc_x");
      auto bc_y           = coupler.get_option<int>("bc_y");
      auto bc_z           = coupler.get_option<int >("bc_z");
      auto &neigh         = coupler.get_neighbor_rankid_matrix();
      auto dtype          = coupler.get_mpi_data_type();
      auto enable_gravity = coupler.get_option<bool>("enable_gravity");
      auto grav           = coupler.get_option<real>("grav");
      auto gamma          = coupler.get_option<real>("gamma_d");
      auto C0             = coupler.get_option<real>("C0");
      MPI_Request sReq [2];
      MPI_Request rReq [2];
      MPI_Status  sStat[2];
      MPI_Status  rStat[2];
      auto comm = MPI_COMM_WORLD;
      int npack = num_state + num_tracers+1;

      // x-direction exchanges
      {
        real5d halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs,nens);
        real5d halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs,nens);
        real5d halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs,nens);
        real5d halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,ny,hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
          if        (v < num_state) {
            halo_send_buf_W(v,k,j,ii,iens) = state  (v          ,hs+k,hs+j,hs+ii,iens);
            halo_send_buf_E(v,k,j,ii,iens) = state  (v          ,hs+k,hs+j,nx+ii,iens);
          } else if (v < num_state + num_tracers) {
            halo_send_buf_W(v,k,j,ii,iens) = tracers(v-num_state,hs+k,hs+j,hs+ii,iens);
            halo_send_buf_E(v,k,j,ii,iens) = tracers(v-num_state,hs+k,hs+j,nx+ii,iens);
          } else {
            halo_send_buf_W(v,k,j,ii,iens) = tke(hs+k,hs+j,hs+ii,iens);
            halo_send_buf_E(v,k,j,ii,iens) = tke(hs+k,hs+j,nx+ii,iens);
          }
        });
        yakl::timer_start("halo_exchange_mpi");
        #ifdef MW_GPU_AWARE_MPI
          yakl::fence();
          MPI_Irecv( halo_recv_buf_W.data() , halo_recv_buf_W.size() , dtype , neigh(1,0) , 0 , comm , &rReq[0] );
          MPI_Irecv( halo_recv_buf_E.data() , halo_recv_buf_E.size() , dtype , neigh(1,2) , 1 , comm , &rReq[1] );
          MPI_Isend( halo_send_buf_W.data() , halo_send_buf_W.size() , dtype , neigh(1,0) , 1 , comm , &sReq[0] );
          MPI_Isend( halo_send_buf_E.data() , halo_send_buf_E.size() , dtype , neigh(1,2) , 0 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("halo_exchange_mpi");
        #else
          realHost5d halo_send_buf_W_host("halo_send_buf_W_host",npack,nz,ny,hs,nens);
          realHost5d halo_send_buf_E_host("halo_send_buf_E_host",npack,nz,ny,hs,nens);
          realHost5d halo_recv_buf_W_host("halo_recv_buf_W_host",npack,nz,ny,hs,nens);
          realHost5d halo_recv_buf_E_host("halo_recv_buf_E_host",npack,nz,ny,hs,nens);
          MPI_Irecv( halo_recv_buf_W_host.data() , halo_recv_buf_W_host.size() , dtype , neigh(1,0) , 0 , comm , &rReq[0] );
          MPI_Irecv( halo_recv_buf_E_host.data() , halo_recv_buf_E_host.size() , dtype , neigh(1,2) , 1 , comm , &rReq[1] );
          halo_send_buf_W.deep_copy_to(halo_send_buf_W_host);
          halo_send_buf_E.deep_copy_to(halo_send_buf_E_host);
          yakl::fence();
          MPI_Isend( halo_send_buf_W_host.data() , halo_send_buf_W_host.size() , dtype , neigh(1,0) , 1 , comm , &sReq[0] );
          MPI_Isend( halo_send_buf_E_host.data() , halo_send_buf_E_host.size() , dtype , neigh(1,2) , 0 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("halo_exchange_mpi");
          halo_recv_buf_W_host.deep_copy_to(halo_recv_buf_W);
          halo_recv_buf_E_host.deep_copy_to(halo_recv_buf_E);
        #endif
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,ny,hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
          if        (v < num_state) {
            state  (v          ,hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
            state  (v          ,hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
          } else if (v < num_state + num_tracers) {
            tracers(v-num_state,hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
            tracers(v-num_state,hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
          } else {
            tke(hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
            tke(hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
          }
        });
      }

      // y-direction exchanges
      {
        real5d halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx+2*hs,nens);
        real5d halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx+2*hs,nens);
        real5d halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx+2*hs,nens);
        real5d halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx+2*hs,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          if        (v < num_state) {
            halo_send_buf_S(v,k,jj,i,iens) = state  (v          ,hs+k,hs+jj,i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = state  (v          ,hs+k,ny+jj,i,iens);
          } else if (v < num_state + num_tracers) {
            halo_send_buf_S(v,k,jj,i,iens) = tracers(v-num_state,hs+k,hs+jj,i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = tracers(v-num_state,hs+k,ny+jj,i,iens);
          } else {
            halo_send_buf_S(v,k,jj,i,iens) = tke(hs+k,hs+jj,i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = tke(hs+k,ny+jj,i,iens);
          }
        });
        yakl::timer_start("halo_exchange_mpi");
        #ifdef MW_GPU_AWARE_MPI
          yakl::fence();
          MPI_Irecv( halo_recv_buf_S.data() , halo_recv_buf_S.size() , dtype , neigh(0,1) , 2 , comm , &rReq[0] );
          MPI_Irecv( halo_recv_buf_N.data() , halo_recv_buf_N.size() , dtype , neigh(2,1) , 3 , comm , &rReq[1] );
          MPI_Isend( halo_send_buf_S.data() , halo_send_buf_S.size() , dtype , neigh(0,1) , 3 , comm , &sReq[0] );
          MPI_Isend( halo_send_buf_N.data() , halo_send_buf_N.size() , dtype , neigh(2,1) , 2 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("halo_exchange_mpi");
        #else
          realHost5d halo_send_buf_S_host("halo_send_buf_S_host",npack,nz,hs,nx+2*hs,nens);
          realHost5d halo_send_buf_N_host("halo_send_buf_N_host",npack,nz,hs,nx+2*hs,nens);
          realHost5d halo_recv_buf_S_host("halo_recv_buf_S_host",npack,nz,hs,nx+2*hs,nens);
          realHost5d halo_recv_buf_N_host("halo_recv_buf_N_host",npack,nz,hs,nx+2*hs,nens);
          MPI_Irecv( halo_recv_buf_S_host.data() , halo_recv_buf_S_host.size() , dtype , neigh(0,1) , 2 , comm , &rReq[0] );
          MPI_Irecv( halo_recv_buf_N_host.data() , halo_recv_buf_N_host.size() , dtype , neigh(2,1) , 3 , comm , &rReq[1] );
          halo_send_buf_S.deep_copy_to(halo_send_buf_S_host);
          halo_send_buf_N.deep_copy_to(halo_send_buf_N_host);
          yakl::fence();
          MPI_Isend( halo_send_buf_S_host.data() , halo_send_buf_S_host.size() , dtype , neigh(0,1) , 3 , comm , &sReq[0] );
          MPI_Isend( halo_send_buf_N_host.data() , halo_send_buf_N_host.size() , dtype , neigh(2,1) , 2 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("halo_exchange_mpi");
          halo_recv_buf_S_host.deep_copy_to(halo_recv_buf_S);
          halo_recv_buf_N_host.deep_copy_to(halo_recv_buf_N);
        #endif
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          if        (v < num_state) {
            state  (v          ,hs+k,      jj,i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            state  (v          ,hs+k,ny+hs+jj,i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          } else if (v < num_state + num_tracers) {
            tracers(v-num_state,hs+k,      jj,i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            tracers(v-num_state,hs+k,ny+hs+jj,i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          } else {
            tke(hs+k,      jj,i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            tke(hs+k,ny+hs+jj,i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          }
        });
      }

      // x-direction non-periodic BC's
      if (bc_x == BC_WALL || bc_x == BC_OPEN) {
        if (px == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,hs,nens) ,
                                            YAKL_LAMBDA (int k, int j, int ii, int iens) {
            state(idR,hs+k,hs+j,ii,iens) =                     state(idR,hs+k,hs+j,hs+0,iens);
            state(idU,hs+k,hs+j,ii,iens) = bc_x==BC_WALL ? 0 : state(idU,hs+k,hs+j,hs+0,iens);
            state(idV,hs+k,hs+j,ii,iens) = bc_x==BC_WALL ? 0 : state(idV,hs+k,hs+j,hs+0,iens);
            state(idW,hs+k,hs+j,ii,iens) = bc_x==BC_WALL ? 0 : state(idW,hs+k,hs+j,hs+0,iens);
            state(idT,hs+k,hs+j,ii,iens) =                     state(idT,hs+k,hs+j,hs+0,iens);
            tke      (hs+k,hs+j,ii,iens) = bc_x==BC_WALL ? 0 : tke      (hs+k,hs+j,hs+0,iens);
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,ii,iens) = tracers(l,hs+k,hs+j,hs+0,iens); }
          });
        }
        if (px == nproc_x-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,hs,nens) ,
                                            YAKL_LAMBDA (int k, int j, int ii, int iens) {
            state(idR,hs+k,hs+j,hs+nx+ii,iens) =                     state(idR,hs+k,hs+j,hs+nx-1,iens);
            state(idU,hs+k,hs+j,hs+nx+ii,iens) = bc_x==BC_WALL ? 0 : state(idU,hs+k,hs+j,hs+nx-1,iens);
            state(idV,hs+k,hs+j,hs+nx+ii,iens) = bc_x==BC_WALL ? 0 : state(idV,hs+k,hs+j,hs+nx-1,iens);
            state(idW,hs+k,hs+j,hs+nx+ii,iens) = bc_x==BC_WALL ? 0 : state(idW,hs+k,hs+j,hs+nx-1,iens);
            state(idT,hs+k,hs+j,hs+nx+ii,iens) =                     state(idT,hs+k,hs+j,hs+nx-1,iens);
            tke      (hs+k,hs+j,hs+nx+ii,iens) = bc_x==BC_WALL ? 0 : tke      (hs+k,hs+j,hs+nx-1,iens);
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+nx+ii,iens) = tracers(l,hs+k,hs+j,hs+nx-1,iens); }
          });
        }
      }

      //y-direction non-periodic BC's
      if (bc_y == BC_WALL || bc_y == BC_OPEN) {
        if (py == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,hs,nx+2*hs,nens) ,
                                            YAKL_LAMBDA (int k, int jj, int i, int iens) {
            state(idR,hs+k,jj,i,iens) =                     state(idR,hs+k,hs+0,i,iens);
            state(idU,hs+k,jj,i,iens) = bc_y==BC_WALL ? 0 : state(idU,hs+k,hs+0,i,iens);
            state(idV,hs+k,jj,i,iens) = bc_y==BC_WALL ? 0 : state(idV,hs+k,hs+0,i,iens);
            state(idW,hs+k,jj,i,iens) = bc_y==BC_WALL ? 0 : state(idW,hs+k,hs+0,i,iens);
            state(idT,hs+k,jj,i,iens) =                     state(idT,hs+k,hs+0,i,iens);
            tke      (hs+k,jj,i,iens) = bc_y==BC_WALL ? 0 : tke      (hs+k,hs+0,i,iens);
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,jj,i,iens) = tracers(l,hs+k,hs+0,i,iens); }
          });
        }
        if (py == nproc_y-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,hs,nx+2*hs,nens) ,
                                            YAKL_LAMBDA (int k, int jj, int i, int iens) {
            state(idR,hs+k,hs+ny+jj,i,iens) =                     state(idR,hs+k,hs+ny-1,i,iens);
            state(idU,hs+k,hs+ny+jj,i,iens) = bc_y==BC_WALL ? 0 : state(idU,hs+k,hs+ny-1,i,iens);
            state(idV,hs+k,hs+ny+jj,i,iens) = bc_y==BC_WALL ? 0 : state(idV,hs+k,hs+ny-1,i,iens);
            state(idW,hs+k,hs+ny+jj,i,iens) = bc_y==BC_WALL ? 0 : state(idW,hs+k,hs+ny-1,i,iens);
            state(idT,hs+k,hs+ny+jj,i,iens) =                     state(idT,hs+k,hs+ny-1,i,iens);
            tke      (hs+k,hs+ny+jj,i,iens) = bc_y==BC_WALL ? 0 : tke      (hs+k,hs+ny-1,i,iens);
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+ny+jj,i,iens) = tracers(l,hs+k,hs+ny-1,i,iens); }
          });
        }
      }

      // z-direction BC's
      if (bc_z == BC_PERIODIC) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(hs,ny+2*hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int kk, int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            state(l,      kk,j,i,iens) = state(l,      kk+nz,j,i,iens);
            state(l,hs+nz+kk,j,i,iens) = state(l,hs+nz+kk-nz,j,i,iens);
          }
          tke(      kk,j,i,iens) = tke(      kk+nz,j,i,iens);
          tke(hs+nz+kk,j,i,iens) = tke(hs+nz+kk-nz,j,i,iens);
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,j,i,iens) = tracers(l,      kk+nz,j,i,iens);
            tracers(l,hs+nz+kk,j,i,iens) = tracers(l,hs+nz+kk-nz,j,i,iens);
          }
        });
      } else if (bc_z == BC_WALL || bc_z == BC_OPEN) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(hs,ny+2*hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int kk, int j, int i, int iens) {
          state(idU,      kk,j,i,iens) = bc_z==BC_WALL ? 0 : state(idU,hs+0   ,j,i,iens);
          state(idV,      kk,j,i,iens) = bc_z==BC_WALL ? 0 : state(idV,hs+0   ,j,i,iens);
          state(idW,      kk,j,i,iens) = bc_z==BC_WALL ? 0 : state(idW,hs+0   ,j,i,iens);
          state(idT,      kk,j,i,iens) =                     state(idT,hs+0   ,j,i,iens);
          tke  (          kk,j,i,iens) = bc_z==BC_WALL ? 0 : tke  (    hs+0   ,j,i,iens);
          state(idU,hs+nz+kk,j,i,iens) = bc_z==BC_WALL ? 0 : state(idU,hs+nz-1,j,i,iens);
          state(idV,hs+nz+kk,j,i,iens) = bc_z==BC_WALL ? 0 : state(idV,hs+nz-1,j,i,iens);
          state(idW,hs+nz+kk,j,i,iens) = bc_z==BC_WALL ? 0 : state(idW,hs+nz-1,j,i,iens);
          state(idT,hs+nz+kk,j,i,iens) =                     state(idT,hs+nz-1,j,i,iens);
          tke  (    hs+nz+kk,j,i,iens) = bc_z==BC_WALL ? 0 : tke  (    hs+nz-1,j,i,iens);
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,j,i,iens) = tracers(l,hs+0   ,j,i,iens);
            tracers(l,hs+nz+kk,j,i,iens) = tracers(l,hs+nz-1,j,i,iens);
          }
          if (enable_gravity) {
            {
              int  k0       = hs;
              int  k        = k0-1-kk;
              real rho0     = state(idR,k0,j,i,iens);
              real theta0   = state(idT,k0,j,i,iens);
              real rho0_gm1 = std::pow(rho0  ,gamma-1);
              real theta0_g = std::pow(theta0,gamma  );
              state(idR,k,j,i,iens) = std::pow( rho0_gm1 + grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) ,
                                                1._fp/(gamma-1) );
            }
            {
              int  k0       = hs+nz-1;
              int  k        = k0+1+kk;
              real rho0     = state(idR,k0,j,i,iens);
              real theta0   = state(idT,k0,j,i,iens);
              real rho0_gm1 = std::pow(rho0  ,gamma-1);
              real theta0_g = std::pow(theta0,gamma  );
              state(idR,k,j,i,iens) = std::pow( rho0_gm1 - grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) ,
                                                1._fp/(gamma-1) );
            }
          } else {
            state(idR,      kk,j,i,iens) = state(idR,hs+0   ,j,i,iens);
            state(idR,hs+nz+kk,j,i,iens) = state(idR,hs+nz-1,j,i,iens);
          }
        });
      }
    }


  };

}

