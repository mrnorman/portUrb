
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "PPM_Limiter.h"
#include "WenoLimiter.h"
#include <random>
#include <sstream>

namespace modules {

  // This clas simplements an A-grid (collocated) cell-centered Finite-Volume method with an upwind Godunov Riemanns
  // solver at cell edges, high-order-accurate reconstruction, Weighted Essentially Non-Oscillatory (WENO) limiting,
  // and a third-order-accurate three-stage Strong Stability Preserving Runge-Kutta time stepping.
  // The dycore prognoses full density, u-, v-, and w-momenta, and mass-weighted potential temperature
  // Since the coupler state is dry density, u-, v-, and w-velocity, and temperature, we need to convert to and from
  // the coupler state.

  struct Dynamics_Euler_Stratified_WenoFV {
    // Order of accuracy (numerical convergence for smooth flows) for the dynamical core
    #ifndef MW_ORD
      int  static constexpr ord = 5;
    #else
      int  static constexpr ord = MW_ORD;
    #endif
    int  static constexpr tord = 3;
    int  static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")
    int  static constexpr num_state = 5;
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature
    // IDs for boundary conditions
    int  static constexpr BC_PERIODIC = 0;
    int  static constexpr BC_OPEN     = 1;
    int  static constexpr BC_WALL     = 2;
    // DIM IDs
    int  static constexpr DIR_X = 0;
    int  static constexpr DIR_Y = 1;
    int  static constexpr DIR_Z = 2;
    // Class data (not use inside parallel_for)
    bool dim_switch;



    Dynamics_Euler_Stratified_WenoFV() { dim_switch = true; }



    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real constexpr maxwave = 350 + 40;
      real cfl = 0.45;
      return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
    }



    void time_step(core::Coupler &coupler, real dt_phys) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers             = coupler.get_num_tracers();
      auto nens                    = coupler.get_nens();
      auto nx                      = coupler.get_nx();
      auto ny                      = coupler.get_ny();
      auto nz                      = coupler.get_nz();
      real5d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real5d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      convert_coupler_to_dynamics( coupler , state , tracers );
      real dt_dyn = compute_time_step( coupler );
      int ncycles = (int) std::ceil( dt_phys / dt_dyn );
      dt_dyn = dt_phys / ncycles;
      for (int icycle = 0; icycle < ncycles; icycle++) {
        if (dim_switch) {
          time_step_euler(coupler,state,tracers,dt_dyn,DIR_X);
          time_step_euler(coupler,state,tracers,dt_dyn,DIR_Y);
          time_step_euler(coupler,state,tracers,dt_dyn,DIR_Z);
        } else {
          time_step_euler(coupler,state,tracers,dt_dyn,DIR_Z);
          time_step_euler(coupler,state,tracers,dt_dyn,DIR_Y);
          time_step_euler(coupler,state,tracers,dt_dyn,DIR_X);
        }
        dim_switch = ! dim_switch;
      }
      convert_dynamics_to_coupler( coupler , state , tracers );
    }



    void time_step_euler( core::Coupler & coupler ,
                          real5d const  & state   ,
                          real5d const  & tracers ,
                          real            dt_dyn  ,
                          int             dir) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto tracer_positive = coupler.get_data_manager_readonly().get<bool const,1>("tracer_positive");
      real5d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx,nens);
      real5d tracers_tend("tracers_tend",num_tracers,nz,ny,nx,nens);
      if      (dir == DIR_X) { compute_tendencies_x( coupler , state , state_tend , tracers , tracers_tend , dt_dyn ); }
      else if (dir == DIR_Y) { compute_tendencies_y( coupler , state , state_tend , tracers , tracers_tend , dt_dyn ); }
      else if (dir == DIR_Z) { compute_tendencies_z( coupler , state , state_tend , tracers , tracers_tend , dt_dyn ); }
      real mx = std::max(num_state,num_tracers);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(mx,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        if (l < num_state) {
          state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn * state_tend  (l,k,j,i,iens);
        }
        if (l < num_tracers) {
          tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn * tracers_tend(l,k,j,i,iens);
          if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
        }
      });
    }



    void compute_tendencies_x( core::Coupler       & coupler      ,
                               real5d        const & state        ,
                               real5d        const & state_tend   ,
                               real5d        const & tracers      ,
                               real5d        const & tracers_tend ,
                               real                  dt           ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens              = coupler.get_nens();
      auto nx                = coupler.get_nx();
      auto ny                = coupler.get_ny();
      auto nz                = coupler.get_nz();
      auto dx                = coupler.get_dx();
      auto dy                = coupler.get_dy();
      auto dz                = coupler.get_dz();
      auto sim2d             = coupler.is_sim2d();
      auto C0                = coupler.get_option<real>("C0"     );
      auto gamma             = coupler.get_option<real>("gamma_d");
      auto num_tracers       = coupler.get_num_tracers();
      auto &dm               = coupler.get_data_manager_readonly();
      auto hy_pressure_cells = dm.get<real const,2>("hy_pressure_cells");
      auto tracer_positive   = dm.get<bool const,1>("tracer_positive"  );
      SArray<real,2,ord,tord> coefs_to_gll;
      TransformMatrices::coefs_to_gll_lower(coefs_to_gll);
      SArray<real,2,tord,tord> g2d2g;
      {
        SArray<real,2,tord,tord> g2c;
        SArray<real,2,tord,tord> c2d;
        SArray<real,2,tord,tord> c2g;
        TransformMatrices::gll_to_coefs  ( g2c );
        TransformMatrices::coefs_to_deriv( c2d );
        TransformMatrices::coefs_to_gll  ( c2g );
        using yakl::intrinsics::matmul_cr;
        using yakl::componentwise::operator/;
        g2d2g = matmul_cr( c2g , matmul_cr( c2d , g2c ) ) / dx;
      }
      real r_dx = 1./dx;

      real4d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rt = state(idT,hs+k,hs+j,hs+i,iens);
        pressure(hs+k,hs+j,hs+i,iens) = C0*std::pow(rt,gamma) - hy_pressure_cells(hs+k,iens);
      });

      core::MultiField<real,4> fields;
      for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<4>(l,0,0,0,0) ); }
      for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<4>(l,0,0,0,0) ); }
      fields.add_field( pressure );
      if (ord > 1) halo_exchange_x( coupler , fields );

      real6d state_limits   ("state_limits"   ,2,num_state  ,nz,ny,nx+1,nens);
      real6d tracers_limits ("tracers_limits" ,2,num_tracers,nz,ny,nx+1,nens);
      real5d pressure_limits("pressure_limits",2            ,nz,ny,nx+1,nens);

      limiter::WenoLimiter<ord> limiter(0.1,1,2,1,1.e3);

      core::MultiField<real,4> advec_fields;
      advec_fields.add_field(state.slice<4>(idV,0,0,0,0));
      advec_fields.add_field(state.slice<4>(idW,0,0,0,0));
      advec_fields.add_field(state.slice<4>(idT,0,0,0,0));
      for (int tr=0; tr < num_tracers; tr++) { advec_fields.add_field(tracers.slice<4>(tr,0,0,0,0)); }
      core::MultiField<real,4> advec_limits_L;
      advec_limits_L.add_field(state_limits.slice<4>(0,idV,0,0,0,0));
      advec_limits_L.add_field(state_limits.slice<4>(0,idW,0,0,0,0));
      advec_limits_L.add_field(state_limits.slice<4>(0,idT,0,0,0,0));
      for (int tr=0; tr < num_tracers; tr++) { advec_limits_L.add_field(tracers_limits.slice<4>(0,tr,0,0,0,0)); }
      core::MultiField<real,4> advec_limits_R;
      advec_limits_R.add_field(state_limits.slice<4>(1,idV,0,0,0,0));
      advec_limits_R.add_field(state_limits.slice<4>(1,idW,0,0,0,0));
      advec_limits_R.add_field(state_limits.slice<4>(1,idT,0,0,0,0));
      for (int tr=0; tr < num_tracers; tr++) { advec_limits_R.add_field(tracers_limits.slice<4>(1,tr,0,0,0,0)); }

      real constexpr cs2 = 350*350;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,2,tord,tord> r, ru, p, ruu;
        SArray<real,1,ord> stencil;
        for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idR,hs+k,hs+j,i+ii,iens); }
        reconstruct_gll_values(stencil,r ,coefs_to_gll,limiter);
        for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idU,hs+k,hs+j,i+ii,iens); }
        reconstruct_gll_values(stencil,ru,coefs_to_gll,limiter);
        for (int ii=0; ii < ord; ii++) { stencil(ii) = pressure( hs+k,hs+j,i+ii,iens); }
        reconstruct_gll_values(stencil,p ,coefs_to_gll,limiter);
        for (int kt=0; kt < tord; kt++) {
          for (int ii=0; ii < tord; ii++) { ruu(kt,ii) = kt==0 ? ru(0,ii)*ru(0,ii)/r(0,ii) : 0; }
        }
        // Compute higher-order time derivatives for state & non-linears
        for (int kt=0; kt < tord-1; kt++) {
          // Compute derivative of fluxes to compute next time derivative
          for (int ii=0; ii<tord; ii++) {
            real der_ru    = 0;
            real der_ruu_p = 0;
            for (int s=0; s < tord; s++) {
              der_ru    += g2d2g(s,ii)*(ru (kt,s)          );
              der_ruu_p += g2d2g(s,ii)*(ruu(kt,s) + p(kt,s));
            }
            r (kt+1,ii) = -der_ru   /(kt+1);
            ru(kt+1,ii) = -der_ruu_p/(kt+1);
            p (kt+1,ii) = -der_ru   /(kt+1)*cs2;
          }
          // Compute non-linear fluxes based off of new state data
          for (int ii=0; ii < tord; ii++) {
            real tot_ruu = 0;
            for (int ind_rt=0; ind_rt <= kt+1; ind_rt++) {
              tot_ruu += ru(ind_rt,ii)*ru(kt+1-ind_rt,ii) - r(ind_rt,ii)*ruu(kt+1-ind_rt,ii);
            }
            ruu(kt+1,ii) = tot_ruu / r(0,ii);
          }
        }
        real mult = dt;
        real r_tavg_0       = r (0,0     );
        real ru_tavg_0      = ru(0,0     );
        real r_tavg_tordm1  = r (0,tord-1);
        real ru_tavg_tordm1 = ru(0,tord-1);
        for (int kt=1; kt < tord; kt++) {
          p (0,0     )   += mult * p (kt,0     ) / (kt+1);
          r_tavg_0       += mult * r (kt,0     ) / (kt+1);
          ru_tavg_0      += mult * ru(kt,0     ) / (kt+1);
          p (0,tord-1)   += mult * p (kt,tord-1) / (kt+1);
          r_tavg_tordm1  += mult * r (kt,tord-1) / (kt+1);
          ru_tavg_tordm1 += mult * ru(kt,tord-1) / (kt+1);
          mult *= dt;
        }
        pressure_limits(1    ,k,j,i  ,iens) = p(0,0     );
        state_limits   (1,idR,k,j,i  ,iens) = r_tavg_0 ;
        state_limits   (1,idU,k,j,i  ,iens) = ru_tavg_0;
        pressure_limits(0    ,k,j,i+1,iens) = p(0,tord-1);
        state_limits   (0,idR,k,j,i+1,iens) = r_tavg_tordm1 ;
        state_limits   (0,idU,k,j,i+1,iens) = ru_tavg_tordm1;
        #pragma no_unroll
        for (int l=0; l < advec_fields.size(); l++) {
          SArray<real,1,ord> stencil;
          SArray<real,2,tord,tord> rt, rut;
          for (int ii=0; ii < ord; ii++) { stencil(ii) = advec_fields(l,hs+k,hs+j,i+ii,iens); }
          reconstruct_gll_values(stencil,rt,coefs_to_gll,limiter);
          for (int kt=0; kt < tord; kt++) {
            for (int ii=0; ii < tord; ii++) { rut(kt,ii) = kt==0 ? ru(0,ii)*rt(0,ii)/r(0,ii) : 0; }
          }
          // Compute higher-order time derivatives for state & non-linears
          for (int kt=0; kt < tord-1; kt++) {
            for (int ii=0; ii<tord; ii++) {
              real der_rut = 0;
              for (int s=0; s < tord; s++) { der_rut += g2d2g(s,ii)*rut(kt,s); }
              rt(kt+1,ii) = -der_rut/(kt+1);
            }
            for (int ii=0; ii < tord; ii++) {
              real tot_rut = 0;
              for (int ind_rt=0; ind_rt <= kt+1; ind_rt++) {
                tot_rut += ru(ind_rt,ii)*rt(kt+1-ind_rt,ii) - r(ind_rt,ii)*rut(kt+1-ind_rt,ii);
              }
              rut(kt+1,ii) = tot_rut / r(0,ii);
            }
          }
          // Compute time-average for tracers
          real mult = dt;
          for (int kt=1; kt < tord; kt++) {
            rt(0,0     ) += mult * rt(kt,0     ) / (kt+1);
            rt(0,tord-1) += mult * rt(kt,tord-1) / (kt+1);
            mult *= dt;
          }
          advec_limits_R(l,k,j,i  ,iens) = rt(0,0     );
          advec_limits_L(l,k,j,i+1,iens) = rt(0,tord-1);
        }
      });
      
      edge_exchange_x( coupler , state_limits , tracers_limits , pressure_limits );

      using yakl::COLON;
      auto state_flux   = state_limits  .slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_flux = tracers_limits.slice<5>(0,COLON,COLON,COLON,COLON,COLON);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      real constexpr cs   = 350.;
      real constexpr r_cs = 1./cs;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx+1,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        // Acoustically upwind mass flux and pressure
        real ru_L = state_limits   (0,idU,k,j,i,iens);   real ru_R = state_limits   (1,idU,k,j,i,iens);
        real rt_L = state_limits   (0,idT,k,j,i,iens);   real rt_R = state_limits   (1,idT,k,j,i,iens);
        real p_L  = pressure_limits(0    ,k,j,i,iens);   real p_R  = pressure_limits(1    ,k,j,i,iens);
        real w1 = 0.5_fp * (p_R-cs*ru_R);
        real w2 = 0.5_fp * (p_L+cs*ru_L);
        real p_upw  = w1 + w2;
        real ru_upw = (w2-w1)*r_cs;
        // Advectively upwind everything else
        int ind = ru_upw > 0 ? 0 : 1;
        real r_rupw = 1._fp / state_limits(ind,idR,k,j,i,iens);
        state_flux(idR,k,j,i,iens) = ru_upw;
        state_flux(idU,k,j,i,iens) = ru_upw*state_limits(ind,idU,k,j,i,iens)*r_rupw + p_upw;
        state_flux(idV,k,j,i,iens) = ru_upw*state_limits(ind,idV,k,j,i,iens)*r_rupw;
        state_flux(idW,k,j,i,iens) = ru_upw*state_limits(ind,idW,k,j,i,iens)*r_rupw;
        state_flux(idT,k,j,i,iens) = ru_upw*state_limits(ind,idT,k,j,i,iens)*r_rupw;
        for (int tr=0; tr < num_tracers; tr++) {
          tracers_flux(tr,k,j,i,iens) = ru_upw*tracers_limits(ind,tr,k,j,i,iens)*r_rupw;
        }
      });

      // Compute tendencies as the flux divergence + gravity source term
      real mx = std::max(num_state,num_tracers);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(mx,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        if (l < num_state) {
          state_tend  (l,k,j,i,iens) = -( state_flux  (l,k,j,i+1,iens) - state_flux  (l,k,j,i,iens) ) * r_dx;
          if (l == idV && sim2d) state_tend(l,k,j,i,iens) = 0;
        }
        if (l < num_tracers) {
          tracers_tend(l,k,j,i,iens) = -( tracers_flux(l,k,j,i+1,iens) - tracers_flux(l,k,j,i,iens) ) * r_dx;
        }
      });
    }



    void compute_tendencies_y( core::Coupler       & coupler      ,
                               real5d        const & state        ,
                               real5d        const & state_tend   ,
                               real5d        const & tracers      ,
                               real5d        const & tracers_tend ,
                               real                  dt           ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens              = coupler.get_nens();
      auto nx                = coupler.get_nx();
      auto ny                = coupler.get_ny();
      auto nz                = coupler.get_nz();
      auto dx                = coupler.get_dx();
      auto dy                = coupler.get_dy();
      auto dz                = coupler.get_dz();
      auto sim2d             = coupler.is_sim2d();
      auto C0                = coupler.get_option<real>("C0"     );
      auto gamma             = coupler.get_option<real>("gamma_d");
      auto num_tracers       = coupler.get_num_tracers();
      auto &dm               = coupler.get_data_manager_readonly();
      auto hy_pressure_cells = dm.get<real const,2>("hy_pressure_cells");
      auto tracer_positive   = dm.get<bool const,1>("tracer_positive"  );
      SArray<real,2,ord,tord> coefs_to_gll;
      TransformMatrices::coefs_to_gll_lower(coefs_to_gll);
      SArray<real,2,tord,tord> g2d2g;
      {
        SArray<real,2,tord,tord> g2c;
        SArray<real,2,tord,tord> c2d;
        SArray<real,2,tord,tord> c2g;
        TransformMatrices::gll_to_coefs  ( g2c );
        TransformMatrices::coefs_to_deriv( c2d );
        TransformMatrices::coefs_to_gll  ( c2g );
        using yakl::intrinsics::matmul_cr;
        using yakl::componentwise::operator/;
        g2d2g = matmul_cr( c2g , matmul_cr( c2d , g2c ) ) / dy;
      }
      real r_dy = 1./dy;

      real4d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rt = state(idT,hs+k,hs+j,hs+i,iens);
        pressure(hs+k,hs+j,hs+i,iens) = C0*std::pow(rt,gamma) - hy_pressure_cells(hs+k,iens);
      });

      core::MultiField<real,4> fields;
      for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<4>(l,0,0,0,0) ); }
      for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<4>(l,0,0,0,0) ); }
      fields.add_field( pressure );
      if (ord > 1) halo_exchange_y( coupler , fields );

      real6d state_limits   ("state_limits"   ,2,num_state  ,nz,ny+1,nx,nens);
      real6d tracers_limits ("tracers_limits" ,2,num_tracers,nz,ny+1,nx,nens);
      real5d pressure_limits("pressure_limits",2            ,nz,ny+1,nx,nens);

      limiter::WenoLimiter<ord> limiter(0.0,1,2,1,1.e3);

      core::MultiField<real,4> advec_fields;
      advec_fields.add_field(state.slice<4>(idU,0,0,0,0));
      advec_fields.add_field(state.slice<4>(idW,0,0,0,0));
      advec_fields.add_field(state.slice<4>(idT,0,0,0,0));
      for (int tr=0; tr < num_tracers; tr++) { advec_fields.add_field(tracers.slice<4>(tr,0,0,0,0)); }
      core::MultiField<real,4> advec_limits_L;
      advec_limits_L.add_field(state_limits.slice<4>(0,idU,0,0,0,0));
      advec_limits_L.add_field(state_limits.slice<4>(0,idW,0,0,0,0));
      advec_limits_L.add_field(state_limits.slice<4>(0,idT,0,0,0,0));
      for (int tr=0; tr < num_tracers; tr++) { advec_limits_L.add_field(tracers_limits.slice<4>(0,tr,0,0,0,0)); }
      core::MultiField<real,4> advec_limits_R;
      advec_limits_R.add_field(state_limits.slice<4>(1,idU,0,0,0,0));
      advec_limits_R.add_field(state_limits.slice<4>(1,idW,0,0,0,0));
      advec_limits_R.add_field(state_limits.slice<4>(1,idT,0,0,0,0));
      for (int tr=0; tr < num_tracers; tr++) { advec_limits_R.add_field(tracers_limits.slice<4>(1,tr,0,0,0,0)); }

      real constexpr cs2 = 350*350;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,2,tord,tord> r, rv, p, rvv;
        SArray<real,1,ord> stencil;
        for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idR,hs+k,j+jj,hs+i,iens); }
        reconstruct_gll_values(stencil,r ,coefs_to_gll,limiter);
        for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idV,hs+k,j+jj,hs+i,iens); }
        reconstruct_gll_values(stencil,rv,coefs_to_gll,limiter);
        for (int jj=0; jj < ord; jj++) { stencil(jj) = pressure( hs+k,j+jj,hs+i,iens); }
        reconstruct_gll_values(stencil,p ,coefs_to_gll,limiter);
        // Initialize non-linears to zero for higher-order time derivatives
        for (int kt=0; kt < tord; kt++) {
          for (int jj=0; jj < tord; jj++) {
            rvv(kt,jj) = kt==0 ? rv(0,jj)*rv(0,jj)/r(0,jj) : 0;
          }
        }
        // Compute higher-order time derivatives for state & non-linears
        for (int kt=0; kt < tord-1; kt++) {
          // Compute derivative of fluxes to compute next time derivative
          for (int jj=0; jj<tord; jj++) {
            real der_rv    = 0;
            real der_rvv_p = 0;
            for (int s=0; s < tord; s++) {
              der_rv    += g2d2g(s,jj)*(rv (kt,s)          );
              der_rvv_p += g2d2g(s,jj)*(rvv(kt,s) + p(kt,s));
            }
            r (kt+1,jj) = -der_rv   /(kt+1);
            rv(kt+1,jj) = -der_rvv_p/(kt+1);
            p (kt+1,jj) = -der_rv   /(kt+1)*cs2;
          }
          // Compute non-linear fluxes based off of new state data
          for (int jj=0; jj < tord; jj++) {
            real tot_rvv = 0;
            for (int ind_rt=0; ind_rt <= kt+1; ind_rt++) {
              tot_rvv += rv(ind_rt,jj)*rv(kt+1-ind_rt,jj) - r(ind_rt,jj)*rvv(kt+1-ind_rt,jj);
            }
            rvv(kt+1,jj) = tot_rvv/r(0,jj);
          }
        }
        real mult = dt;
        real r_tavg_0       = r (0,0     );
        real rv_tavg_0      = rv(0,0     );
        real r_tavg_tordm1  = r (0,tord-1);
        real rv_tavg_tordm1 = rv(0,tord-1);
        for (int kt=1; kt < tord; kt++) {
          p (0,0     )   += mult * p (kt,0     ) / (kt+1);
          r_tavg_0       += mult * r (kt,0     ) / (kt+1);
          rv_tavg_0      += mult * rv(kt,0     ) / (kt+1);
          p (0,tord-1)   += mult * p (kt,tord-1) / (kt+1);
          r_tavg_tordm1  += mult * r (kt,tord-1) / (kt+1);
          rv_tavg_tordm1 += mult * rv(kt,tord-1) / (kt+1);
          mult *= dt;
        }
        pressure_limits(1    ,k,j  ,i,iens) = p(0,0     );
        state_limits   (1,idR,k,j  ,i,iens) = r_tavg_0 ;
        state_limits   (1,idV,k,j  ,i,iens) = rv_tavg_0;
        pressure_limits(0    ,k,j+1,i,iens) = p(0,tord-1);
        state_limits   (0,idR,k,j+1,i,iens) = r_tavg_tordm1 ;
        state_limits   (0,idV,k,j+1,i,iens) = rv_tavg_tordm1;
        #pragma no_unroll
        for (int l=0; l < advec_fields.size(); l++) {
          SArray<real,1,ord> stencil;
          SArray<real,2,tord,tord> rt, rvt;
          // Compute GLL points for zeroth-order time derivative
          for (int jj=0; jj < ord; jj++) { stencil(jj) = advec_fields(l,hs+k,j+jj,hs+i,iens); }
          reconstruct_gll_values(stencil,rt,coefs_to_gll,limiter);
          for (int kt=0; kt < tord; kt++) {
            for (int jj=0; jj < tord; jj++) { rvt(kt,jj) = kt==0 ? rv(0,jj)*rt(0,jj)/r(0,jj) : 0; }
          }
          // Compute higher-order time derivatives for state & non-linears
          for (int kt=0; kt < tord-1; kt++) {
            // Compute derivative of fluxes to compute next time derivative
            for (int jj=0; jj<tord; jj++) {
              real der_rvt = 0;
              for (int s=0; s < tord; s++) { der_rvt += g2d2g(s,jj)* rvt(kt,s); }
              rt(kt+1,jj) = -der_rvt/(kt+1);
            }
            // Compute non-linear fluxes based off of new state data
            for (int jj=0; jj < tord; jj++) {
              real tot_rvt = 0;
              for (int ind_rt=0; ind_rt <= kt+1; ind_rt++) {
                tot_rvt += rv(ind_rt,jj)*rt(kt+1-ind_rt,jj) - r(ind_rt,jj)*rvt(kt+1-ind_rt,jj);
              }
              rvt(kt+1,jj) = tot_rvt / r(0,jj);
            }
          }
          // Compute time-average for tracers
          real mult = dt;
          for (int kt=1; kt < tord; kt++) {
            rt(0,0     ) += mult * rt(kt,0     ) / (kt+1);
            rt(0,tord-1) += mult * rt(kt,tord-1) / (kt+1);
            mult *= dt;
          }
          advec_limits_R(l,k,j  ,i,iens) = rt(0,0     );
          advec_limits_L(l,k,j+1,i,iens) = rt(0,tord-1);
        }
      });
      
      edge_exchange_y( coupler , state_limits , tracers_limits , pressure_limits );

      using yakl::COLON;
      auto state_flux   = state_limits  .slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_flux = tracers_limits.slice<5>(0,COLON,COLON,COLON,COLON,COLON);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      real constexpr cs   = 350.;
      real constexpr r_cs = 1./cs;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny+1,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        // Acoustically upwind mass flux and pressure
        real rv_L = state_limits   (0,idV,k,j,i,iens);   real rv_R = state_limits   (1,idV,k,j,i,iens);
        real rt_L = state_limits   (0,idT,k,j,i,iens);   real rt_R = state_limits   (1,idT,k,j,i,iens);
        real p_L  = pressure_limits(0    ,k,j,i,iens);   real p_R  = pressure_limits(1    ,k,j,i,iens);
        real w1 = 0.5_fp * (p_R-cs*rv_R);
        real w2 = 0.5_fp * (p_L+cs*rv_L);
        real p_upw  = w1 + w2;
        real rv_upw = (w2-w1)*r_cs;
        // Advectively upwind everything else
        int ind = rv_upw > 0 ? 0 : 1;
        real r_rupw = 1._fp / state_limits(ind,idR,k,j,i,iens);
        state_flux(idR,k,j,i,iens) = rv_upw;
        state_flux(idU,k,j,i,iens) = rv_upw*state_limits(ind,idU,k,j,i,iens)*r_rupw;
        state_flux(idV,k,j,i,iens) = rv_upw*state_limits(ind,idV,k,j,i,iens)*r_rupw + p_upw;
        state_flux(idW,k,j,i,iens) = rv_upw*state_limits(ind,idW,k,j,i,iens)*r_rupw;
        state_flux(idT,k,j,i,iens) = rv_upw*state_limits(ind,idT,k,j,i,iens)*r_rupw;
        for (int tr=0; tr < num_tracers; tr++) {
          tracers_flux(tr,k,j,i,iens) = rv_upw*tracers_limits(ind,tr,k,j,i,iens)*r_rupw;
        }
      });

      // Compute tendencies as the flux divergence + gravity source term
      real mx = std::max(num_state,num_tracers);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(mx,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        if (l < num_state) {
          state_tend  (l,k,j,i,iens) = -( state_flux  (l,k,j+1,i,iens) - state_flux  (l,k,j,i,iens) ) * r_dy;
        }
        if (l < num_tracers) {
          tracers_tend(l,k,j,i,iens) = -( tracers_flux(l,k,j+1,i,iens) - tracers_flux(l,k,j,i,iens) ) * r_dy;
        }
      });
    }



    void compute_tendencies_z( core::Coupler       & coupler      ,
                               real5d        const & state        ,
                               real5d        const & state_tend   ,
                               real5d        const & tracers      ,
                               real5d        const & tracers_tend ,
                               real                  dt           ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens              = coupler.get_nens();
      auto nx                = coupler.get_nx();
      auto ny                = coupler.get_ny();
      auto nz                = coupler.get_nz();
      auto dx                = coupler.get_dx();
      auto dy                = coupler.get_dy();
      auto dz                = coupler.get_dz();
      auto sim2d             = coupler.is_sim2d();
      auto num_tracers       = coupler.get_num_tracers();
      auto C0                = coupler.get_option<real>("C0"     );
      auto grav              = coupler.get_option<real>("grav"   );
      auto gamma             = coupler.get_option<real>("gamma_d");
      auto save_pressure_z   = coupler.get_option<bool>("save_pressure_z",false);
      auto tracer_positive   = coupler.get_data_manager_readonly().get<bool const,1>("tracer_positive");
      auto hy_dens_gll       = coupler.get_data_manager_readonly().get<real const,3>("hy_dens_gll"  );
      auto hy_dens_cells     = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_cells");
      auto hy_pressure_cells = coupler.get_data_manager_readonly().get<real const,2>("hy_pressure_cells");
      SArray<real,2,ord,tord> coefs_to_gll;
      TransformMatrices::coefs_to_gll_lower(coefs_to_gll);
      SArray<real,2,tord,tord> g2d2g;
      {
        SArray<real,2,tord,tord> g2c;
        SArray<real,2,tord,tord> c2d;
        SArray<real,2,tord,tord> c2g;
        TransformMatrices::gll_to_coefs  ( g2c );
        TransformMatrices::coefs_to_deriv( c2d );
        TransformMatrices::coefs_to_gll  ( c2g );
        using yakl::intrinsics::matmul_cr;
        using yakl::componentwise::operator/;
        g2d2g = matmul_cr( c2g , matmul_cr( c2d , g2c ) ) / dz;
      }
      real r_dz = 1./dz;

      // Since tracers are full mass, it's helpful before reconstruction to remove the background density for potentially
      // more accurate reconstructions of tracer concentrations
      real4d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rdens = 1._fp / state(idR,hs+k,hs+j,hs+i,iens);
        real rt = state(idT,hs+k,hs+j,hs+i,iens);
        pressure( hs+k,hs+j,hs+i,iens) = C0*std::pow(rt,gamma) - hy_pressure_cells(hs+k,iens);
        state(idU,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idV,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idW,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idT,hs+k,hs+j,hs+i,iens) *= rdens;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) *= rdens; }
      });

      if (ord > 1) halo_boundary_conditions_z( coupler , state , tracers , pressure );

      real6d state_limits   ("state_limits"   ,2,num_state  ,nz+1,ny,nx,nens);
      real6d tracers_limits ("tracers_limits" ,2,num_tracers,nz+1,ny,nx,nens);
      real5d pressure_limits("pressure_limits",2            ,nz+1,ny,nx,nens);

      limiter::WenoLimiter<ord> limiter(0.0,1,2,1,1.e3);

      core::MultiField<real,4> advec_fields;
      advec_fields.add_field(state.slice<4>(idU,0,0,0,0));
      advec_fields.add_field(state.slice<4>(idV,0,0,0,0));
      advec_fields.add_field(state.slice<4>(idT,0,0,0,0));
      for (int tr=0; tr < num_tracers; tr++) { advec_fields.add_field(tracers.slice<4>(tr,0,0,0,0)); }
      core::MultiField<real,4> advec_limits_L;
      advec_limits_L.add_field(state_limits.slice<4>(0,idU,0,0,0,0));
      advec_limits_L.add_field(state_limits.slice<4>(0,idV,0,0,0,0));
      advec_limits_L.add_field(state_limits.slice<4>(0,idT,0,0,0,0));
      for (int tr=0; tr < num_tracers; tr++) { advec_limits_L.add_field(tracers_limits.slice<4>(0,tr,0,0,0,0)); }
      core::MultiField<real,4> advec_limits_R;
      advec_limits_R.add_field(state_limits.slice<4>(1,idU,0,0,0,0));
      advec_limits_R.add_field(state_limits.slice<4>(1,idV,0,0,0,0));
      advec_limits_R.add_field(state_limits.slice<4>(1,idT,0,0,0,0));
      for (int tr=0; tr < num_tracers; tr++) { advec_limits_R.add_field(tracers_limits.slice<4>(1,tr,0,0,0,0)); }

      real constexpr cs2 = 350*350;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,2,tord,tord> r, rw, p, rww;
        SArray<real,1,ord> stencil;
        for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idR,k+kk,hs+j,hs+i,iens); }
        reconstruct_gll_values(stencil,r ,coefs_to_gll,limiter);
        for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idW,k+kk,hs+j,hs+i,iens); }
        reconstruct_gll_values(stencil,rw,coefs_to_gll,limiter);
        for (int kk=0; kk < ord; kk++) { stencil(kk) = pressure( k+kk,hs+j,hs+i,iens); }
        reconstruct_gll_values(stencil,p ,coefs_to_gll,limiter);
        if (k == 0   ) { rw(0,0     ) = 0; }
        if (k == nz-1) { rw(0,tord-1) = 0; }
        // Compute non-linears for zeroth-order time derivative
        for (int kk=0; kk < tord; kk++) { rw(0,kk) *= r(0,kk); }
        // Initialize non-linears to zero for higher-order time derivatives
        for (int kt=0; kt < tord; kt++) {
          for (int kk=0; kk < tord; kk++) { rww(kt,kk) = kt==0 ? rw(0,kk)*rw(0,kk)/r(0,kk) : 0; }
        }
        // Compute higher-order time derivatives for state & non-linears
        for (int kt=0; kt < tord-1; kt++) {
          // Compute derivative of fluxes to compute next time derivative
          for (int kk=0; kk<tord; kk++) {
            real der_rw    = 0;
            real der_rww_p = 0;
            for (int s=0; s < tord; s++) {
              der_rw    += g2d2g(s,kk)*(rw (kt,s)          );
              der_rww_p += g2d2g(s,kk)*(rww(kt,s) + p(kt,s));
            }
            der_rww_p += grav * ( kt==0 ? r(kt,kk)-hy_dens_gll(kk,k,iens) : r(kt,kk) );
            r (kt+1,kk) = -der_rw   /(kt+1);
            rw(kt+1,kk) = -der_rww_p/(kt+1);
            p (kt+1,kk) = -der_rw   /(kt+1)*cs2;
          }
          if (k == 0   ) { rw(kt+1,0     ) = 0; }
          if (k == nz-1) { rw(kt+1,tord-1) = 0; }
          // Compute non-linear fluxes based off of new state data
          for (int kk=0; kk < tord; kk++) {
            real tot_rww = 0;
            for (int ind_rt=0; ind_rt <= kt+1; ind_rt++) {
              tot_rww += rw(ind_rt,kk)*rw(kt+1-ind_rt,kk) - r(ind_rt,kk)*rww(kt+1-ind_rt,kk);
            }
            rww(kt+1,kk) = tot_rww / r(0,kk);
          }
        }
        // Compute time average for all terms except r and ru, which are needed later
        real mult = dt;
        real r_tavg_0       = r (0,0     );
        real rw_tavg_0      = rw(0,0     );
        real r_tavg_tordm1  = r (0,tord-1);
        real rw_tavg_tordm1 = rw(0,tord-1);
        for (int kt=1; kt < tord; kt++) {
          p (0,0     )   += mult * p (kt,0     ) / (kt+1);
          r_tavg_0       += mult * r (kt,0     ) / (kt+1);
          rw_tavg_0      += mult * rw(kt,0     ) / (kt+1);
          p (0,tord-1)   += mult * p (kt,tord-1) / (kt+1);
          r_tavg_tordm1  += mult * r (kt,tord-1) / (kt+1);
          rw_tavg_tordm1 += mult * rw(kt,tord-1) / (kt+1);
          mult *= dt;
        }
        pressure_limits(1    ,k  ,j,i,iens) = p (0,0     );
        state_limits   (1,idR,k  ,j,i,iens) = r_tavg_0      ;
        state_limits   (1,idW,k  ,j,i,iens) = rw_tavg_0     ;
        pressure_limits(0    ,k+1,j,i,iens) = p (0,tord-1);
        state_limits   (0,idR,k+1,j,i,iens) = r_tavg_tordm1 ;
        state_limits   (0,idW,k+1,j,i,iens) = rw_tavg_tordm1;
        #pragma no_unroll
        for (int l=0; l < advec_fields.size(); l++) {
          SArray<real,1,ord> stencil;
          SArray<real,2,tord,tord> rt, rwt;
          // Compute GLL points for zeroth-order time derivative
          for (int kk=0; kk < ord; kk++) { stencil(kk) = advec_fields(l,k+kk,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,rt,coefs_to_gll,limiter);
          // Compute non-linears for zeroth-order time derivative
          for (int kk=0; kk < tord; kk++) { rt(0,kk) *= r(0,kk); }
          for (int kt=0; kt < tord; kt++) {
            for (int kk=0; kk < tord; kk++) { rwt(kt,kk) = kt==0 ? rw(0,kk)*rt(0,kk)/r(0,kk) : 0; }
          }
          // Compute higher-order time derivatives for state & non-linears
          for (int kt=0; kt < tord-1; kt++) {
            // Compute derivative of fluxes to compute next time derivative
            for (int kk=0; kk<tord; kk++) {
              real der_rwt = 0;
              for (int s=0; s < tord; s++) { der_rwt += g2d2g(s,kk)* rwt(kt,s); }
              rt(kt+1,kk) = -der_rwt/(kt+1);
            }
            // Compute non-linear fluxes based off of new state data
            for (int kk=0; kk < tord; kk++) {
              real tot_rwt = 0;
              for (int ind_rt=0; ind_rt <= kt+1; ind_rt++) {
                tot_rwt += rw(ind_rt,kk)*rt(kt+1-ind_rt,kk) - r(ind_rt,kk)*rwt(kt+1-ind_rt,kk);
              }
              rwt(kt+1,kk) = tot_rwt / r(0,kk);
            }
          }
          // Compute time-average for tracers
          real mult = dt;
          for (int kt=1; kt < tord; kt++) {
            rt(0,0     ) += mult * rt(kt,0     ) / (kt+1);
            rt(0,tord-1) += mult * rt(kt,tord-1) / (kt+1);
            mult *= dt;
          }
          advec_limits_R(l,k  ,j,i,iens) = rt(0,0     );
          advec_limits_L(l,k+1,j,i,iens) = rt(0,tord-1);
        }
      });
      
      edge_boundary_conditions_z( coupler , state_limits , tracers_limits , pressure_limits );

      using yakl::COLON;
      auto state_flux   = state_limits  .slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_flux = tracers_limits.slice<5>(0,COLON,COLON,COLON,COLON,COLON);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      real constexpr cs   = 350.;
      real constexpr r_cs = 1./cs;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz+1,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        // Acoustically upwind mass flux and pressure
        real rw_L = state_limits   (0,idW,k,j,i,iens);   real rw_R = state_limits   (1,idW,k,j,i,iens);
        real rt_L = state_limits   (0,idT,k,j,i,iens);   real rt_R = state_limits   (1,idT,k,j,i,iens);
        real p_L  = pressure_limits(0    ,k,j,i,iens);   real p_R  = pressure_limits(1    ,k,j,i,iens);
        real w1 = 0.5_fp * (p_R-cs*rw_R);
        real w2 = 0.5_fp * (p_L+cs*rw_L);
        real p_upw  = w1 + w2;
        real rw_upw = (w2-w1)*r_cs;
        // Advectively upwind everything else
        int ind = rw_upw > 0 ? 0 : 1;
        real r_rupw = 1._fp / state_limits(ind,idR,k,j,i,iens);
        state_flux(idR,k,j,i,iens) = rw_upw;
        state_flux(idU,k,j,i,iens) = rw_upw*state_limits(ind,idU,k,j,i,iens)*r_rupw;
        state_flux(idV,k,j,i,iens) = rw_upw*state_limits(ind,idV,k,j,i,iens)*r_rupw;
        state_flux(idW,k,j,i,iens) = rw_upw*state_limits(ind,idW,k,j,i,iens)*r_rupw + p_upw;
        state_flux(idT,k,j,i,iens) = rw_upw*state_limits(ind,idT,k,j,i,iens)*r_rupw;
        for (int tr=0; tr < num_tracers; tr++) {
          tracers_flux(tr,k,j,i,iens) = rw_upw*tracers_limits(ind,tr,k,j,i,iens)*r_rupw;
        }
      });

      // Compute tendencies as the flux divergence + gravity source term
      real mx = std::max(num_state,num_tracers);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(mx,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        if (l < num_state) {
          if (l != idR) state(l,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
          state_tend  (l,k,j,i,iens) = -( state_flux  (l,k+1,j,i,iens) - state_flux  (l,k,j,i,iens) ) * r_dz;
          if (l == idW) state_tend(l,k,j,i,iens) += -grav*(state(idR,hs+k,hs+j,hs+i,iens) - hy_dens_cells(hs+k,iens));
        }
        if (l < num_tracers) {
          tracers(l,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
          tracers_tend(l,k,j,i,iens) = -( tracers_flux(l,k+1,j,i,iens) - tracers_flux(l,k,j,i,iens) ) * r_dz;
        }
      });
    }



    // ord stencil cell averages to two GLL point values via high-order reconstruction and WENO limiting
    template <class LIMITER>
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>       const & stencil      ,
                                                    SArray<real,2,tord,tord>       & gll          ,
                                                    SArray<real,2,ord ,tord> const & coefs_to_gll ,
                                                    LIMITER                  const & limiter    ) {
      // Reconstruct values
      SArray<real,1,ord> wenoCoefs;
      limiter.compute_limited_coefs( stencil , wenoCoefs );
      // Transform ord weno coefficients into 2 GLL points
      for (int ii=0; ii<tord; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) { tmp += coefs_to_gll(s,ii) * wenoCoefs(s); }
        gll(0,ii) = tmp;
      }
    }



    // ord stencil cell averages to two GLL point values via high-order reconstruction and WENO limiting
    template <class LIMITER>
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>      const & stencil      ,
                                                    SArray<real,1,tord>           & gll          ,
                                                    SArray<real,2,ord,tord> const & coefs_to_gll ,
                                                    LIMITER                 const & limiter    ) {
      // Reconstruct values
      SArray<real,1,ord> wenoCoefs;
      limiter.compute_limited_coefs( stencil , wenoCoefs );
      // Transform ord weno coefficients into 2 GLL points
      for (int ii=0; ii<tord; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) { tmp += coefs_to_gll(s,ii) * wenoCoefs(s); }
        gll(ii) = tmp;
      }
    }



    void halo_exchange_x( core::Coupler const & coupler , core::MultiField<real,4> & fields ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens   = coupler.get_nens();
      auto nx     = coupler.get_nx();
      auto ny     = coupler.get_ny();
      auto nz     = coupler.get_nz();
      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto dtype  = coupler.get_mpi_data_type();
      MPI_Request sReq [2], rReq [2];
      MPI_Status  sStat[2], rStat[2];
      auto comm = MPI_COMM_WORLD;
      int npack = fields.size();

      real5d halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs,nens);
      real5d halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs,nens);
      real5d halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs,nens);
      real5d halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs,nens);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,ny,hs,nens) ,
                                        YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
        halo_send_buf_W(v,k,j,ii,iens) = fields(v,hs+k,hs+j,hs+ii,iens);
        halo_send_buf_E(v,k,j,ii,iens) = fields(v,hs+k,hs+j,nx+ii,iens);
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
        auto halo_send_buf_W_host = halo_send_buf_W.createHostObject();
        auto halo_send_buf_E_host = halo_send_buf_E.createHostObject();
        auto halo_recv_buf_W_host = halo_recv_buf_W.createHostObject();
        auto halo_recv_buf_E_host = halo_recv_buf_E.createHostObject();
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
        fields(v,hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
        fields(v,hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
      });
    }



    void halo_exchange_y( core::Coupler const & coupler , core::MultiField<real,4> & fields ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens   = coupler.get_nens();
      auto nx     = coupler.get_nx();
      auto ny     = coupler.get_ny();
      auto nz     = coupler.get_nz();
      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto dtype  = coupler.get_mpi_data_type();
      MPI_Request sReq [2], rReq [2];
      MPI_Status  sStat[2], rStat[2];
      auto comm = MPI_COMM_WORLD;
      int npack = fields.size();

      real5d halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx+2*hs,nens);
      real5d halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx+2*hs,nens);
      real5d halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx+2*hs,nens);
      real5d halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx+2*hs,nens);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,hs,nx+2*hs,nens) ,
                                        YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
        halo_send_buf_S(v,k,jj,i,iens) = fields(v,hs+k,hs+jj,i,iens);
        halo_send_buf_N(v,k,jj,i,iens) = fields(v,hs+k,ny+jj,i,iens);
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
        auto halo_send_buf_S_host = halo_send_buf_S.createHostObject();
        auto halo_send_buf_N_host = halo_send_buf_N.createHostObject();
        auto halo_recv_buf_S_host = halo_recv_buf_S.createHostObject();
        auto halo_recv_buf_N_host = halo_recv_buf_N.createHostObject();
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
        fields(v,hs+k,      jj,i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
        fields(v,hs+k,ny+hs+jj,i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
      });
    }



    void halo_boundary_conditions_z( core::Coupler const & coupler  ,
                                     real5d        const & state    ,
                                     real5d        const & tracers  ,
                                     real4d        const & pressure ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens           = coupler.get_nens();
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto dz             = coupler.get_dz();
      auto num_tracers    = coupler.get_num_tracers();
      auto &neigh         = coupler.get_neighbor_rankid_matrix();
      auto dtype          = coupler.get_mpi_data_type();
      auto grav           = coupler.get_option<real>("grav");
      auto gamma          = coupler.get_option<real>("gamma_d");
      auto C0             = coupler.get_option<real>("C0");
      auto hy_dens_cells  = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_cells");
      auto hy_theta_cells = coupler.get_data_manager_readonly().get<real const,2>("hy_theta_cells");

      // z-direction BC's
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(hs,ny+2*hs,nx+2*hs,nens) ,
                                        YAKL_LAMBDA (int kk, int j, int i, int iens) {
        state(idR,kk,j,i,iens) = hy_dens_cells (kk,iens);
        state(idU,kk,j,i,iens) = state(idU,hs+0,j,i,iens);
        state(idV,kk,j,i,iens) = state(idV,hs+0,j,i,iens);
        state(idW,kk,j,i,iens) = 0;
        state(idT,kk,j,i,iens) = hy_theta_cells(kk,iens);
        pressure( kk,j,i,iens) = pressure( hs+0,j,i,iens);
        state(idR,hs+nz+kk,j,i,iens) = hy_dens_cells (hs+nz+kk,iens);
        state(idU,hs+nz+kk,j,i,iens) = state(idU,hs+nz-1,j,i,iens);
        state(idV,hs+nz+kk,j,i,iens) = state(idV,hs+nz-1,j,i,iens);
        state(idW,hs+nz+kk,j,i,iens) = 0;
        state(idT,hs+nz+kk,j,i,iens) = hy_theta_cells(hs+nz+kk,iens);
        pressure( hs+nz+kk,j,i,iens) = pressure( hs+nz-1,j,i,iens);
        for (int l=0; l < num_tracers; l++) {
          tracers(l,      kk,j,i,iens) = 0;
          tracers(l,hs+nz+kk,j,i,iens) = 0;
        }
      });
    }



    void edge_exchange_x( core::Coupler const & coupler           ,
                          real6d        const & state_limits_x    ,
                          real6d        const & tracers_limits_x  ,
                          real5d        const & pressure_limits_x ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto &neigh      = coupler.get_neighbor_rankid_matrix();
      auto dtype       = coupler.get_mpi_data_type();
      auto comm        = MPI_COMM_WORLD;
      int npack = num_state + num_tracers + 1;
      MPI_Request sReq [2];
      MPI_Request rReq [2];
      MPI_Status  sStat[2];
      MPI_Status  rStat[2];

      real4d edge_send_buf_W("edge_send_buf_W",npack,nz,ny,nens);
      real4d edge_send_buf_E("edge_send_buf_E",npack,nz,ny,nens);
      real4d edge_recv_buf_W("edge_recv_buf_W",npack,nz,ny,nens);
      real4d edge_recv_buf_E("edge_recv_buf_E",npack,nz,ny,nens);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,nens) , YAKL_LAMBDA (int v, int k, int j, int iens) {
        if        (v < num_state) {
          edge_send_buf_W(v,k,j,iens) = state_limits_x  (1,v          ,k,j,0 ,iens);
          edge_send_buf_E(v,k,j,iens) = state_limits_x  (0,v          ,k,j,nx,iens);
        } else if (v < num_state + num_tracers) {                    
          edge_send_buf_W(v,k,j,iens) = tracers_limits_x(1,v-num_state,k,j,0 ,iens);
          edge_send_buf_E(v,k,j,iens) = tracers_limits_x(0,v-num_state,k,j,nx,iens);
        } else {
          edge_send_buf_W(v,k,j,iens) = pressure_limits_x(1,k,j,0 ,iens);
          edge_send_buf_E(v,k,j,iens) = pressure_limits_x(0,k,j,nx,iens);
        }
      });
      yakl::timer_start("edge_exchange_mpi");
      #ifdef MW_GPU_AWARE_MPI
        yakl::fence();
        MPI_Irecv( edge_recv_buf_W.data() , edge_recv_buf_W.size() , dtype , neigh(1,0) , 4 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E.data() , edge_recv_buf_E.size() , dtype , neigh(1,2) , 5 , comm , &rReq[1] );
        MPI_Isend( edge_send_buf_W.data() , edge_send_buf_W.size() , dtype , neigh(1,0) , 5 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_E.data() , edge_send_buf_E.size() , dtype , neigh(1,2) , 4 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        yakl::timer_stop("edge_exchange_mpi");
      #else
        realHost4d edge_send_buf_W_host("edge_send_buf_W_host",npack,nz,ny,nens);
        realHost4d edge_send_buf_E_host("edge_send_buf_E_host",npack,nz,ny,nens);
        realHost4d edge_recv_buf_W_host("edge_recv_buf_W_host",npack,nz,ny,nens);
        realHost4d edge_recv_buf_E_host("edge_recv_buf_E_host",npack,nz,ny,nens);
        MPI_Irecv( edge_recv_buf_W_host.data() , edge_recv_buf_W_host.size() , dtype , neigh(1,0) , 4 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E_host.data() , edge_recv_buf_E_host.size() , dtype , neigh(1,2) , 5 , comm , &rReq[1] );
        edge_send_buf_W.deep_copy_to(edge_send_buf_W_host);
        edge_send_buf_E.deep_copy_to(edge_send_buf_E_host);
        yakl::fence();
        MPI_Isend( edge_send_buf_W_host.data() , edge_send_buf_W_host.size() , dtype , neigh(1,0) , 5 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_E_host.data() , edge_send_buf_E_host.size() , dtype , neigh(1,2) , 4 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        yakl::timer_stop("edge_exchange_mpi");
        edge_recv_buf_W_host.deep_copy_to(edge_recv_buf_W);
        edge_recv_buf_E_host.deep_copy_to(edge_recv_buf_E);
      #endif
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,nens) ,
                                        YAKL_LAMBDA (int v, int k, int j, int iens) {
        if        (v < num_state) {
          state_limits_x  (0,v          ,k,j,0 ,iens) = edge_recv_buf_W(v,k,j,iens);
          state_limits_x  (1,v          ,k,j,nx,iens) = edge_recv_buf_E(v,k,j,iens);
        } else if (v < num_state + num_tracers) {
          tracers_limits_x(0,v-num_state,k,j,0 ,iens) = edge_recv_buf_W(v,k,j,iens);
          tracers_limits_x(1,v-num_state,k,j,nx,iens) = edge_recv_buf_E(v,k,j,iens);
        } else {
          pressure_limits_x(0,k,j,0 ,iens) = edge_recv_buf_W(v,k,j,iens);
          pressure_limits_x(1,k,j,nx,iens) = edge_recv_buf_E(v,k,j,iens);
        }
      });
    }



    void edge_exchange_y( core::Coupler const & coupler           ,
                          real6d        const & state_limits_y    ,
                          real6d        const & tracers_limits_y  ,
                          real5d        const & pressure_limits_y ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto &neigh      = coupler.get_neighbor_rankid_matrix();
      auto dtype       = coupler.get_mpi_data_type();
      auto comm        = MPI_COMM_WORLD;
      int npack = num_state + num_tracers + 1;
      MPI_Request sReq [2];
      MPI_Request rReq [2];
      MPI_Status  sStat[2];
      MPI_Status  rStat[2];

      real4d edge_send_buf_S("edge_send_buf_S",npack,nz,nx,nens);
      real4d edge_send_buf_N("edge_send_buf_N",npack,nz,nx,nens);
      real4d edge_recv_buf_S("edge_recv_buf_S",npack,nz,nx,nens);
      real4d edge_recv_buf_N("edge_recv_buf_N",npack,nz,nx,nens);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,nx,nens) ,
                                        YAKL_LAMBDA (int v, int k, int i, int iens) {
        if        (v < num_state) {
          edge_send_buf_S(v,k,i,iens) = state_limits_y  (1,v          ,k,0 ,i,iens);
          edge_send_buf_N(v,k,i,iens) = state_limits_y  (0,v          ,k,ny,i,iens);
        } else if (v < num_state + num_tracers) {                    
          edge_send_buf_S(v,k,i,iens) = tracers_limits_y(1,v-num_state,k,0 ,i,iens);
          edge_send_buf_N(v,k,i,iens) = tracers_limits_y(0,v-num_state,k,ny,i,iens);
        } else {
          edge_send_buf_S(v,k,i,iens) = pressure_limits_y(1,k,0 ,i,iens);
          edge_send_buf_N(v,k,i,iens) = pressure_limits_y(0,k,ny,i,iens);
        }
      });
      yakl::timer_start("edge_exchange_mpi");
      #ifdef MW_GPU_AWARE_MPI
        yakl::fence();
        MPI_Irecv( edge_recv_buf_S.data() , edge_recv_buf_S.size() , dtype , neigh(0,1) , 6 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_N.data() , edge_recv_buf_N.size() , dtype , neigh(2,1) , 7 , comm , &rReq[1] );
        MPI_Isend( edge_send_buf_S.data() , edge_send_buf_S.size() , dtype , neigh(0,1) , 7 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_N.data() , edge_send_buf_N.size() , dtype , neigh(2,1) , 6 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        yakl::timer_stop("edge_exchange_mpi");
      #else
        realHost4d edge_send_buf_S_host("edge_send_buf_S_host",npack,nz,nx,nens);
        realHost4d edge_send_buf_N_host("edge_send_buf_N_host",npack,nz,nx,nens);
        realHost4d edge_recv_buf_S_host("edge_recv_buf_S_host",npack,nz,nx,nens);
        realHost4d edge_recv_buf_N_host("edge_recv_buf_N_host",npack,nz,nx,nens);
        MPI_Irecv( edge_recv_buf_S_host.data() , edge_recv_buf_S_host.size() , dtype , neigh(0,1) , 6 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_N_host.data() , edge_recv_buf_N_host.size() , dtype , neigh(2,1) , 7 , comm , &rReq[1] );
        edge_send_buf_S.deep_copy_to(edge_send_buf_S_host);
        edge_send_buf_N.deep_copy_to(edge_send_buf_N_host);
        yakl::fence();
        MPI_Isend( edge_send_buf_S_host.data() , edge_send_buf_S_host.size() , dtype , neigh(0,1) , 7 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_N_host.data() , edge_send_buf_N_host.size() , dtype , neigh(2,1) , 6 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        yakl::timer_stop("edge_exchange_mpi");
        edge_recv_buf_S_host.deep_copy_to(edge_recv_buf_S);
        edge_recv_buf_N_host.deep_copy_to(edge_recv_buf_N);
      #endif
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,nx,nens) ,
                                        YAKL_LAMBDA (int v, int k, int i, int iens) {
        if        (v < num_state) {
          state_limits_y  (0,v          ,k,0 ,i,iens) = edge_recv_buf_S(v,k,i,iens);
          state_limits_y  (1,v          ,k,ny,i,iens) = edge_recv_buf_N(v,k,i,iens);
        } else if (v < num_state + num_tracers) {
          tracers_limits_y(0,v-num_state,k,0 ,i,iens) = edge_recv_buf_S(v,k,i,iens);
          tracers_limits_y(1,v-num_state,k,ny,i,iens) = edge_recv_buf_N(v,k,i,iens);
        } else {
          pressure_limits_y(0,k,0 ,i,iens) = edge_recv_buf_S(v,k,i,iens);
          pressure_limits_y(1,k,ny,i,iens) = edge_recv_buf_N(v,k,i,iens);
        }
      });
    }



    void edge_boundary_conditions_z( core::Coupler const & coupler           ,
                                     real6d        const & state_limits_z    ,
                                     real6d        const & tracers_limits_z  ,
                                     real5d        const & pressure_limits_z ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto hy_dens_edges  = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_edges" );
      auto hy_theta_edges = coupler.get_data_manager_readonly().get<real const,2>("hy_theta_edges");
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(ny,nx,nens) ,
                                        YAKL_LAMBDA (int j, int i, int iens) {
        state_limits_z(0,idR,0 ,j,i,iens) = hy_dens_edges(0,iens);
        state_limits_z(1,idR,0 ,j,i,iens) = hy_dens_edges(0,iens);
        state_limits_z(0,idU,0 ,j,i,iens) = state_limits_z(1,idU,0 ,j,i,iens);
        state_limits_z(0,idV,0 ,j,i,iens) = state_limits_z(1,idV,0 ,j,i,iens);
        state_limits_z(0,idW,0 ,j,i,iens) = 0;
        state_limits_z(1,idW,0 ,j,i,iens) = 0;
        state_limits_z(0,idT,0 ,j,i,iens) = hy_theta_edges(0,iens);
        state_limits_z(1,idT,0 ,j,i,iens) = hy_theta_edges(0,iens);
        state_limits_z(0,idR,nz,j,i,iens) = hy_dens_edges(nz,iens);
        state_limits_z(1,idR,nz,j,i,iens) = hy_dens_edges(nz,iens);
        state_limits_z(1,idU,nz,j,i,iens) = state_limits_z(0,idU,nz,j,i,iens);
        state_limits_z(1,idV,nz,j,i,iens) = state_limits_z(0,idV,nz,j,i,iens);
        state_limits_z(0,idW,nz,j,i,iens) = 0;
        state_limits_z(1,idW,nz,j,i,iens) = 0;
        state_limits_z(0,idT,nz,j,i,iens) = hy_theta_edges(nz,iens);
        state_limits_z(1,idT,nz,j,i,iens) = hy_theta_edges(nz,iens);
        for (int l=0; l < num_tracers; l++) {
          tracers_limits_z(0,l,0 ,j,i,iens) = 0;
          tracers_limits_z(1,l,0 ,j,i,iens) = 0;
          tracers_limits_z(0,l,nz,j,i,iens) = 0;
          tracers_limits_z(1,l,nz,j,i,iens) = 0;
        }
        pressure_limits_z(0,0 ,j,i,iens) = pressure_limits_z(1,0 ,j,i,iens);
        pressure_limits_z(1,nz,j,i,iens) = pressure_limits_z(0,nz,j,i,iens);
      });
    }



    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto dz          = coupler.get_dz();
      auto nx_glob     = coupler.get_nx_glob();
      auto ny_glob     = coupler.get_ny_glob();
      auto num_tracers = coupler.get_num_tracers();
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0"     );
      auto grav        = coupler.get_option<real>("grav"   );

      num_tracers = coupler.get_num_tracers();
      bool1d tracer_adds_mass("tracer_adds_mass",num_tracers);
      bool1d tracer_positive ("tracer_positive" ,num_tracers);
      auto tracer_adds_mass_host = tracer_adds_mass.createHostCopy();
      auto tracer_positive_host  = tracer_positive .createHostCopy();
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass, diffuse;
        coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass , diffuse );
        tracer_positive_host (tr) = positive;
        tracer_adds_mass_host(tr) = adds_mass;
      }
      tracer_positive_host .deep_copy_to(tracer_positive );
      tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);
      auto &dm = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<bool>("tracer_adds_mass","",{num_tracers});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);
      dm.register_and_allocate<bool>("tracer_positive","",{num_tracers});
      auto dm_tracer_positive = dm.get<bool,1>("tracer_positive");
      tracer_positive.deep_copy_to(dm_tracer_positive);

      real5d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);  state   = 0;
      real5d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);  tracers = 0;
      convert_coupler_to_dynamics( coupler , state , tracers );
      dm.register_and_allocate<real>("hy_dens_cells"    ,"",{nz+2*hs,nens});
      dm.register_and_allocate<real>("hy_theta_cells"   ,"",{nz+2*hs,nens});
      dm.register_and_allocate<real>("hy_pressure_cells","",{nz+2*hs,nens});
      auto r = dm.get<real,2>("hy_dens_cells"    );    r = 0;
      auto t = dm.get<real,2>("hy_theta_cells"   );    t = 0;
      auto p = dm.get<real,2>("hy_pressure_cells");    p = 0;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz+2*hs,nens) , YAKL_LAMBDA (int k, int iens) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            r(k,iens) += state(idR,k,hs+j,hs+i,iens);
            t(k,iens) += state(idT,k,hs+j,hs+i,iens) / state(idR,k,hs+j,hs+i,iens);
            p(k,iens) += C0 * std::pow( state(idT,k,hs+j,hs+i,iens) , gamma );
          }
        }
      });
      auto r_loc = r .createHostCopy();    auto r_glob = r .createHostObject();
      auto t_loc = t .createHostCopy();    auto t_glob = t .createHostObject();
      auto p_loc = p .createHostCopy();    auto p_glob = p .createHostObject();
      auto dtype = coupler.get_mpi_data_type();
      MPI_Allreduce( r_loc.data() , r_glob.data() , r.size() , dtype , MPI_SUM , MPI_COMM_WORLD );
      MPI_Allreduce( t_loc.data() , t_glob.data() , t.size() , dtype , MPI_SUM , MPI_COMM_WORLD );
      MPI_Allreduce( p_loc.data() , p_glob.data() , p.size() , dtype , MPI_SUM , MPI_COMM_WORLD );
      r_glob.deep_copy_to(r);
      t_glob.deep_copy_to(t);
      p_glob.deep_copy_to(p);
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz+2*hs,nens) , YAKL_LAMBDA (int k, int iens) {
        r(k,iens) *= r_nx_ny;
        t(k,iens) *= r_nx_ny;
        p(k,iens) *= r_nx_ny;
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(hs,nens) , YAKL_LAMBDA (int kk, int iens) {
        {
          int  k0       = hs;
          int  k        = k0-1-kk;
          real rho0     = r(k0,iens);
          real theta0   = t(k0,iens);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k,iens) = std::pow( rho0_gm1 + grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k,iens) = theta0;
          p(k,iens) = C0*std::pow(r(k,iens)*theta0,gamma);
        }
        {
          int  k0       = hs+nz-1;
          int  k        = k0+1+kk;
          real rho0     = r(k0,iens);
          real theta0   = t(k0,iens);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k,iens) = std::pow( rho0_gm1 - grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k,iens) = theta0;
          p(k,iens) = C0*std::pow(r(k,iens)*theta0,gamma);
        }
      });

      auto create_immersed_proportion_halos = [] (core::Coupler &coupler) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;;
        auto nz   = coupler.get_nz  ();
        auto ny   = coupler.get_ny  ();
        auto nx   = coupler.get_nx  ();
        auto nens = coupler.get_nens();
        auto &neigh       = coupler.get_neighbor_rankid_matrix();
        auto dtype        = coupler.get_mpi_data_type();
        auto &dm  = coupler.get_data_manager_readwrite();
        if (! dm.entry_exists("immersed_proportion_halos")) {
          dm.register_and_allocate<real>("immersed_proportion_halos","",{nz+2*hs,ny+2*hs,nx+2*hs,nens});
        }
        if (! dm.entry_exists("fully_immersed_halos")) {
          dm.register_and_allocate<bool>("fully_immersed_halos","",{nz+2*hs,ny+2*hs,nx+2*hs,nens});
        }
        auto immersed_proportion       = dm.get<real const,4>("immersed_proportion"      );
        auto immersed_proportion_halos = dm.get<real      ,4>("immersed_proportion_halos");  immersed_proportion_halos = 0;
        auto fully_immersed_halos      = dm.get<bool      ,4>("fully_immersed_halos"     );  fully_immersed_halos      = false;
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          immersed_proportion_halos(hs+k,hs+j,hs+i,iens) = immersed_proportion(k,j,i,iens);
        });
        MPI_Request sReq [2];
        MPI_Request rReq [2];
        MPI_Status  sStat[2];
        MPI_Status  rStat[2];
        auto comm = MPI_COMM_WORLD;
        // x-direction exchange
        real4d halo_send_buf_W("halo_send_buf_W",nz,ny,hs,nens);
        real4d halo_send_buf_E("halo_send_buf_E",nz,ny,hs,nens);
        real4d halo_recv_buf_W("halo_recv_buf_W",nz,ny,hs,nens);
        real4d halo_recv_buf_E("halo_recv_buf_E",nz,ny,hs,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,hs,nens) ,
                                          YAKL_LAMBDA (int k, int j, int ii, int iens) {
          halo_send_buf_W(k,j,ii,iens) = immersed_proportion_halos(hs+k,hs+j,hs+ii,iens);
          halo_send_buf_E(k,j,ii,iens) = immersed_proportion_halos(hs+k,hs+j,nx+ii,iens);
        });
        realHost4d halo_send_buf_W_host("halo_send_buf_W_host",nz,ny,hs,nens);
        realHost4d halo_send_buf_E_host("halo_send_buf_E_host",nz,ny,hs,nens);
        realHost4d halo_recv_buf_W_host("halo_recv_buf_W_host",nz,ny,hs,nens);
        realHost4d halo_recv_buf_E_host("halo_recv_buf_E_host",nz,ny,hs,nens);
        MPI_Irecv( halo_recv_buf_W_host.data() , halo_recv_buf_W_host.size() , dtype , neigh(1,0) , 0 , comm , &rReq[0] );
        MPI_Irecv( halo_recv_buf_E_host.data() , halo_recv_buf_E_host.size() , dtype , neigh(1,2) , 1 , comm , &rReq[1] );
        halo_send_buf_W.deep_copy_to(halo_send_buf_W_host);
        halo_send_buf_E.deep_copy_to(halo_send_buf_E_host);
        yakl::fence();
        MPI_Isend( halo_send_buf_W_host.data() , halo_send_buf_W_host.size() , dtype , neigh(1,0) , 1 , comm , &sReq[0] );
        MPI_Isend( halo_send_buf_E_host.data() , halo_send_buf_E_host.size() , dtype , neigh(1,2) , 0 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        halo_recv_buf_W_host.deep_copy_to(halo_recv_buf_W);
        halo_recv_buf_E_host.deep_copy_to(halo_recv_buf_E);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,hs,nens) ,
                                          YAKL_LAMBDA (int k, int j, int ii, int iens) {
          immersed_proportion_halos(hs+k,hs+j,      ii,iens) = halo_recv_buf_W(k,j,ii,iens);
          immersed_proportion_halos(hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(k,j,ii,iens);
        });
        // y-direction exchange
        real4d halo_send_buf_S("halo_send_buf_S",nz,hs,nx+2*hs,nens);
        real4d halo_send_buf_N("halo_send_buf_N",nz,hs,nx+2*hs,nens);
        real4d halo_recv_buf_S("halo_recv_buf_S",nz,hs,nx+2*hs,nens);
        real4d halo_recv_buf_N("halo_recv_buf_N",nz,hs,nx+2*hs,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int k, int jj, int i, int iens) {
          halo_send_buf_S(k,jj,i,iens) = immersed_proportion_halos(hs+k,hs+jj,i,iens);
          halo_send_buf_N(k,jj,i,iens) = immersed_proportion_halos(hs+k,ny+jj,i,iens);
        });
        realHost4d halo_send_buf_S_host("halo_send_buf_S_host",nz,hs,nx+2*hs,nens);
        realHost4d halo_send_buf_N_host("halo_send_buf_N_host",nz,hs,nx+2*hs,nens);
        realHost4d halo_recv_buf_S_host("halo_recv_buf_S_host",nz,hs,nx+2*hs,nens);
        realHost4d halo_recv_buf_N_host("halo_recv_buf_N_host",nz,hs,nx+2*hs,nens);
        MPI_Irecv( halo_recv_buf_S_host.data() , halo_recv_buf_S_host.size() , dtype , neigh(0,1) , 2 , comm , &rReq[0] );
        MPI_Irecv( halo_recv_buf_N_host.data() , halo_recv_buf_N_host.size() , dtype , neigh(2,1) , 3 , comm , &rReq[1] );
        halo_send_buf_S.deep_copy_to(halo_send_buf_S_host);
        halo_send_buf_N.deep_copy_to(halo_send_buf_N_host);
        yakl::fence();
        MPI_Isend( halo_send_buf_S_host.data() , halo_send_buf_S_host.size() , dtype , neigh(0,1) , 3 , comm , &sReq[0] );
        MPI_Isend( halo_send_buf_N_host.data() , halo_send_buf_N_host.size() , dtype , neigh(2,1) , 2 , comm , &sReq[1] );
        MPI_Waitall(2, sReq, sStat);
        MPI_Waitall(2, rReq, rStat);
        halo_recv_buf_S_host.deep_copy_to(halo_recv_buf_S);
        halo_recv_buf_N_host.deep_copy_to(halo_recv_buf_N);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int k, int jj, int i, int iens) {
          immersed_proportion_halos(hs+k,      jj,i,iens) = halo_recv_buf_S(k,jj,i,iens);
          immersed_proportion_halos(hs+k,ny+hs+jj,i,iens) = halo_recv_buf_N(k,jj,i,iens);
        });
        // z-direction boundaries
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(hs,ny+2*hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int kk, int j, int i, int iens) {
          immersed_proportion_halos(      kk,j,i,iens) = 1;
          immersed_proportion_halos(hs+nz+kk,j,i,iens) = 1;
        });
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz+2*hs,ny+2*hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          fully_immersed_halos(k,j,i,iens) = immersed_proportion_halos(k,j,i,iens) == 1;
        });
      };

      auto compute_hydrostasis_edges = [] (core::Coupler &coupler) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;;
        auto nz   = coupler.get_nz  ();
        auto ny   = coupler.get_ny  ();
        auto nx   = coupler.get_nx  ();
        auto nens = coupler.get_nens();
        auto &dm  = coupler.get_data_manager_readwrite();
        if (! dm.entry_exists("hy_dens_gll"   )) dm.register_and_allocate<real>("hy_dens_gll"   ,"",{tord,nz,nens});
        if (! dm.entry_exists("hy_dens_edges" )) dm.register_and_allocate<real>("hy_dens_edges" ,"",{nz+1,nens});
        if (! dm.entry_exists("hy_theta_edges")) dm.register_and_allocate<real>("hy_theta_edges","",{nz+1,nens});
        auto hy_dens_cells  = dm.get<real const,2>("hy_dens_cells" );
        auto hy_theta_cells = dm.get<real const,2>("hy_theta_cells");
        auto hy_dens_edges  = dm.get<real      ,2>("hy_dens_edges" );
        auto hy_theta_edges = dm.get<real      ,2>("hy_theta_edges");
        auto hy_dens_gll    = dm.get<real      ,3>("hy_dens_gll"   );
        SArray<real,2,ord,tord> sten_to_gll;
        TransformMatrices::sten_to_gll_lower(sten_to_gll);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz+1,nens) , YAKL_LAMBDA (int k, int iens) {
          hy_dens_edges(k,iens) = std::exp( -1./12.*std::log(hy_dens_cells(hs+k-2,iens)) +
                                             7./12.*std::log(hy_dens_cells(hs+k-1,iens)) +
                                             7./12.*std::log(hy_dens_cells(hs+k  ,iens)) +
                                            -1./12.*std::log(hy_dens_cells(hs+k+1,iens)) );
          hy_theta_edges(k,iens) = -1./12.*hy_theta_cells(hs+k-2,iens) +
                                    7./12.*hy_theta_cells(hs+k-1,iens) +
                                    7./12.*hy_theta_cells(hs+k  ,iens) +
                                   -1./12.*hy_theta_cells(hs+k+1,iens);
          if (k < nz) {
            for (int ii=0; ii<tord; ii++) {
              real tmp = 0;
              for (int s=0; s < ord; s++) { tmp += sten_to_gll(s,ii) * std::log(hy_dens_cells(k+s,iens)); }
              hy_dens_gll(ii,k,iens) = std::exp(tmp);
            }
          }
        });
      };

      create_immersed_proportion_halos( coupler );
      compute_hydrostasis_edges       ( coupler );

      // These are needed for a proper restart
      coupler.register_output_variable<real>( "immersed_proportion" , core::Coupler::DIMS_3D      );
      coupler.register_output_variable<real>( "surface_temp"        , core::Coupler::DIMS_SURFACE );
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto i_beg = coupler.get_i_beg();
        auto j_beg = coupler.get_j_beg();
        auto nz    = coupler.get_nz();
        auto ny    = coupler.get_ny();
        auto nx    = coupler.get_nx();
        auto nens  = coupler.get_nens();
        nc.redef();
        nc.create_dim( "z_halo" , coupler.get_nz()+2*hs );
        nc.create_var<real>( "hy_dens_cells"     , {"z_halo","ens"});
        nc.create_var<real>( "hy_theta_cells"    , {"z_halo","ens"});
        nc.create_var<real>( "hy_pressure_cells" , {"z_halo","ens"});
        nc.create_var<real>( "theta"             , {"z","y","x","ens"});
        nc.enddef();
        nc.begin_indep_data();
        auto &dm = coupler.get_data_manager_readonly();
        if (coupler.is_mainproc()) nc.write( dm.get<real const,2>("hy_dens_cells"    ) , "hy_dens_cells"     );
        if (coupler.is_mainproc()) nc.write( dm.get<real const,2>("hy_theta_cells"   ) , "hy_theta_cells"    );
        if (coupler.is_mainproc()) nc.write( dm.get<real const,2>("hy_pressure_cells") , "hy_pressure_cells" );
        nc.end_indep_data();
        real5d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
        real5d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);
        convert_coupler_to_dynamics( coupler , state , tracers );
        std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg,0};
        using yakl::componentwise::operator/;
        real4d data("data",nz,ny,nx,nens);
        yakl::c::parallel_for( yakl::c::Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          data(k,j,i,iens) = state(idT,hs+k,hs+j,hs+i,iens) / state(idR,hs+k,hs+j,hs+i,iens);
        });
        nc.write_all(data,"theta",start_3d);
      } );
      coupler.register_overwrite_with_restart_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto &dm = coupler.get_data_manager_readwrite();
        nc.read_all(dm.get<real,2>("hy_dens_cells"    ),"hy_dens_cells"    ,{0,0});
        nc.read_all(dm.get<real,2>("hy_theta_cells"   ),"hy_theta_cells"   ,{0,0});
        nc.read_all(dm.get<real,2>("hy_pressure_cells"),"hy_pressure_cells",{0,0});
        create_immersed_proportion_halos( coupler );
        compute_hydrostasis_edges       ( coupler );
      } );
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler , realConst5d state , realConst5d tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto R_d         = coupler.get_option<real>("R_d"    );
      auto R_v         = coupler.get_option<real>("R_v"    );
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0"     );
      auto idWV        = coupler.get_option<int >("idWV"   );
      auto num_tracers = coupler.get_num_tracers();
      auto &dm = coupler.get_data_manager_readwrite();
      auto dm_rho_d = dm.get<real,4>("density_dry");
      auto dm_uvel  = dm.get<real,4>("uvel"       );
      auto dm_vvel  = dm.get<real,4>("vvel"       );
      auto dm_wvel  = dm.get<real,4>("wvel"       );
      auto dm_temp  = dm.get<real,4>("temp"       );
      auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      core::MultiField<real,4> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real,4>(tracer_names[tr]) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho   = state(idR,hs+k,hs+j,hs+i,iens);
        real u     = state(idU,hs+k,hs+j,hs+i,iens) / rho;
        real v     = state(idV,hs+k,hs+j,hs+i,iens) / rho;
        real w     = state(idW,hs+k,hs+j,hs+i,iens) / rho;
        real theta = state(idT,hs+k,hs+j,hs+i,iens) / rho;
        real press = C0 * pow( rho*theta , gamma );
        real rho_v = tracers(idWV,hs+k,hs+j,hs+i,iens);
        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,hs+j,hs+i,iens); }
        real temp = press / ( rho_d * R_d + rho_v * R_v );
        dm_rho_d(k,j,i,iens) = rho_d;
        dm_uvel (k,j,i,iens) = u;
        dm_vvel (k,j,i,iens) = v;
        dm_wvel (k,j,i,iens) = w;
        dm_temp (k,j,i,iens) = temp;
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens); }
      });
    }



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler , real5d &state , real5d &tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto R_d         = coupler.get_option<real>("R_d"    );
      auto R_v         = coupler.get_option<real>("R_v"    );
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0"     );
      auto idWV        = coupler.get_option<int >("idWV"   );
      auto num_tracers = coupler.get_num_tracers();
      auto &dm = coupler.get_data_manager_readonly();
      auto dm_rho_d = dm.get<real const,4>("density_dry");
      auto dm_uvel  = dm.get<real const,4>("uvel"       );
      auto dm_vvel  = dm.get<real const,4>("vvel"       );
      auto dm_wvel  = dm.get<real const,4>("wvel"       );
      auto dm_temp  = dm.get<real const,4>("temp"       );
      auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      core::MultiField<real const,4> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real const,4>(tracer_names[tr]) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho_d = dm_rho_d(k,j,i,iens);
        real u     = dm_uvel (k,j,i,iens);
        real v     = dm_vvel (k,j,i,iens);
        real w     = dm_wvel (k,j,i,iens);
        real temp  = dm_temp (k,j,i,iens);
        real rho_v = dm_tracers(idWV,k,j,i,iens);
        real press = rho_d * R_d * temp + rho_v * R_v * temp;
        real rho = rho_d;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i,iens); }
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;
        state(idR,hs+k,hs+j,hs+i,iens) = rho;
        state(idU,hs+k,hs+j,hs+i,iens) = rho * u;
        state(idV,hs+k,hs+j,hs+i,iens) = rho * v;
        state(idW,hs+k,hs+j,hs+i,iens) = rho * w;
        state(idT,hs+k,hs+j,hs+i,iens) = rho * theta;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) = dm_tracers(tr,k,j,i,iens); }
      });
    }


  };

}


