
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
#include "PPM_Limiter.h"
#include "MinmodLimiter.h"
#include <random>
#include <sstream>

namespace modules {

  // This clas simplements an A-grid (collocated) cell-centered Finite-Volume method with an upwind Godunov Riemanns
  // solver at cell edges, high-order-accurate reconstruction, Weighted Essentially Non-Oscillatory (WENO) limiting,
  // and Strong Stability Preserving Runge-Kutta time stepping.
  // The dycore prognoses full density, u-, v-, and w-momenta, and mass-weighted potential temperature
  // Since the coupler state is dry density, u-, v-, and w-velocity, and temperature, we need to convert to and from
  // the coupler state.
  // This dynamical core supports immersed boundaries (fully immersed only. Partially immersed are ignored). Immersed
  // boundaries will have no-slip wall BC's, and surface fluxes are applied in a separate module to model friction
  // based on a prescribed roughness length with Monin-Obukhov thoery.
  // You'll notice the dimensions are nz,ny,nx,nens. The nens dimension supports running multiple ensembles in parallel
  // at the same time and allowing them to contribute to the GPU threading.

  struct Dynamics_Euler_Stratified_WenoFV {
    // Order of accuracy (numerical convergence for smooth flows) for the dynamical core
    #ifndef MW_ORD
      int  static constexpr ord = 5;
    #else
      int  static constexpr ord = MW_ORD;
    #endif
    int  static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")
    int  static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature



    Dynamics_Euler_Stratified_WenoFV() { }



    // Use CFL criterion to determine the time step. Currently hardwired
    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real constexpr maxwave = 350 + 40;
      real cfl = 0.45;
      return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
    }



    // Perform a time step
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
      for (int icycle = 0; icycle < ncycles; icycle++) { time_step_rk_3_3(coupler,state,tracers,dt_dyn); }
      convert_dynamics_to_coupler( coupler , state , tracers );
    }



    // CFL 2.0 (Differs from paper, but this is the true value for this high-order FV scheme)
    // s-stage third-order low-storage SSPRK with two state registers
    // https://link.springer.com/content/pdf/10.1007/s10915-008-9239-z.pdf
    void time_step_rk_s_3( core::Coupler & coupler ,
                           real5d const  & state   ,
                           real5d const  & tracers ,
                           real            dt_dyn  ,
                           int s = 9 ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      // Array to determine which tracers must remain positive
      auto tracer_positive = coupler.get_data_manager_readonly().get<bool const,1>("tracer_positive");
      // Parameters from the paper
      int n = sqrt(s);
      int r = s-n;
      // To hold tendencies
      real5d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx,nens);
      real5d tracers_tend("tracers_tend",num_tracers,nz,ny,nx,nens);

      //////////////
      // Part 1
      //////////////
      for (int istage = 1; istage <= ((n-1)*(n-2))/2; istage++) {
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/r);
        // Apply tendencies for state and tracers
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/r;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/r;
            // Ensure positive tracers stay positive
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
      }
      // Create copies (see paper)
      real5d state_tmp   = state  .createDeviceCopy();
      real5d tracers_tmp = tracers.createDeviceCopy();
      ///////////////////
      // Part 2
      ///////////////////
      for (int istage = ((n-1)*(n-2))/2+1; istage <= (n*(n+1))/2-1; istage++) {
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/r);
        // Apply tendencies for state and tracers
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/r;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/r;
            // Ensure positive tracers stay positive
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
      }
      ///////////////////////////////
      // Intermission  (See paper)
      ///////////////////////////////
      compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,(n-1)/(2*n-1.)*dt_dyn/r);
      // Apply tendencies for state and tracers
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        for (int l = 0; l < num_state  ; l++) {
          state  (l,hs+k,hs+j,hs+i,iens) =  n   /(2*n-1.)*state_tmp  (l,hs+k,hs+j,hs+i,iens) +
                                           (n-1)/(2*n-1.)*state      (l,hs+k,hs+j,hs+i,iens) +
                                           (n-1)/(2*n-1.)*dt_dyn/r*state_tend  (l,k,j,i,iens);
        }
        for (int l = 0; l < num_tracers; l++) {
          tracers(l,hs+k,hs+j,hs+i,iens) =  n   /(2*n-1.)*tracers_tmp(l,hs+k,hs+j,hs+i,iens) +
                                           (n-1)/(2*n-1.)*tracers    (l,hs+k,hs+j,hs+i,iens) +
                                           (n-1)/(2*n-1.)*dt_dyn/r*tracers_tend(l,k,j,i,iens);
          // Ensure positive tracers stay positive
          if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
        }
      });
      //////////////
      // Part 3
      //////////////
      for (int istage = (n*(n+1))/2+1; istage <= s; istage++) {
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/r);
        // Apply tendencies for state and tracers
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/r;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/r;
            // Ensure positive tracers stay positive
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
      }
    }



    // CFL 2.0 (Differs from paper, but this is the true value for this high-order FV scheme)
    // fourth-order ten stage low-storage SSPRK with two registers
    // https://link.springer.com/content/pdf/10.1007/s10915-008-9239-z.pdf
    void time_step_rk_10_4( core::Coupler & coupler ,
                            real5d const  & state   ,
                            real5d const  & tracers ,
                            real            dt_dyn  ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto tracer_positive = coupler.get_data_manager_readonly().get<bool const,1>("tracer_positive");

      // If first-order, then do forward Euler
      if (ord == 1) {
        real5d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx,nens);
        real5d tracers_tend("tracers_tend",num_tracers,nz,ny,nx,nens);
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn * state_tend  (l,k,j,i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn * tracers_tend(l,k,j,i,iens);
            // Ensure positive tracers stay positive
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers(l,hs+k,hs+j,hs+i,iens) );
          }
        });
      } else {
        // Temporary registers for time stepping
        real5d state_tmp   = state  .createDeviceCopy();
        real5d tracers_tmp = tracers.createDeviceCopy();
        // To hold tendencies
        real5d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx,nens);
        real5d tracers_tend("tracers_tend",num_tracers,nz,ny,nx,nens);
        //////////////
        // Stages 1-5
        //////////////
        for (int istage = 1; istage <= 5; istage++) {
          compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/6);
          // Apply tendencies
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            for (int l = 0; l < num_state  ; l++) {
              state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/6;
            }
            for (int l = 0; l < num_tracers; l++) {
              tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/6;
              // Ensure positive tracers stay positive
              if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
            }
          });
        }
        //////////////////////////////
        // Intermissions (see paper)
        //////////////////////////////
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i,iens) = state_tmp  (l,hs+k,hs+j,hs+i,iens)/25 + 9*state    (l,hs+k,hs+j,hs+i,iens)/25;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i,iens) = tracers_tmp(l,hs+k,hs+j,hs+i,iens)/25 + 9*tracers  (l,hs+k,hs+j,hs+i,iens)/25;
            // Ensure positive tracers stay positive
            if (tracer_positive(l)) tracers_tmp(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers_tmp(l,hs+k,hs+j,hs+i,iens));
          }
        });
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = 15*state_tmp  (l,hs+k,hs+j,hs+i,iens) - 5*state    (l,hs+k,hs+j,hs+i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = 15*tracers_tmp(l,hs+k,hs+j,hs+i,iens) - 5*tracers  (l,hs+k,hs+j,hs+i,iens);
            // Ensure positive tracers stay positive
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
        //////////////
        // Stages 6-9
        //////////////
        for (int istage = 6; istage <= 9; istage++) {
          compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/6);
          // Apply tendencies
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            for (int l = 0; l < num_state  ; l++) {
              state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/6;
            }
            for (int l = 0; l < num_tracers; l++) {
              tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/6;
              // Ensure positive tracers stay positive
              if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
            }
          });
        }
        //////////////
        // Stage 10
        //////////////
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/10);
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state_tmp  (l,hs+k,hs+j,hs+i,iens) + 3*state    (l,hs+k,hs+j,hs+i,iens)/5 +
                                             dt_dyn*state_tend  (l,k,j,i,iens)/10;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers_tmp(l,hs+k,hs+j,hs+i,iens) + 3*tracers  (l,hs+k,hs+j,hs+i,iens)/5 +
                                             dt_dyn*tracers_tend(l,k,j,i,iens)/10;
            // Ensure positive tracers stay positive
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
      }
    }



    // CFL 0.45 (Differs from paper, but this is the true value for this high-order FV scheme)
    // Third-order, three-stage SSPRK method
    // https://link.springer.com/content/pdf/10.1007/s10915-008-9239-z.pdf
    void time_step_rk_3_3( core::Coupler & coupler ,
                           real5d const  & state   ,
                           real5d const  & tracers ,
                           real            dt_dyn  ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto tracer_positive = coupler.get_data_manager_readonly().get<bool const,1>("tracer_positive");

      // if first-order, use forward Euler time stepping
      if (ord == 1) {
        real5d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx,nens);
        real5d tracers_tend("tracers_tend",num_tracers,nz,ny,nx,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
          tracers_tend(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens);
        });
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn * state_tend  (l,k,j,i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn * tracers_tend(l,k,j,i,iens);
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers(l,hs+k,hs+j,hs+i,iens) );
            }
          }
        });
      } else {
        // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
        real5d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
        real5d tracers_tmp ("tracers_tmp" ,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);
        // To hold tendencies
        real5d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     ,nens);
        real5d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     ,nens);
        //////////////
        // Stage 1
        //////////////
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn * state_tend  (l,k,j,i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn * tracers_tend(l,k,j,i,iens);
            // Ensure positive tracers stay positive
            if (tracer_positive(l)) tracers_tmp(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i,iens) );
          }
        });
        //////////////
        // Stage 2
        //////////////
        compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/4.);
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i,iens) = (3._fp/4._fp) * state      (l,hs+k,hs+j,hs+i,iens) + 
                                                 (1._fp/4._fp) * state_tmp  (l,hs+k,hs+j,hs+i,iens) +
                                                 (1._fp/4._fp) * dt_dyn * state_tend  (l,k,j,i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i,iens) = (3._fp/4._fp) * tracers    (l,hs+k,hs+j,hs+i,iens) + 
                                                 (1._fp/4._fp) * tracers_tmp(l,hs+k,hs+j,hs+i,iens) +
                                                 (1._fp/4._fp) * dt_dyn * tracers_tend(l,k,j,i,iens);
            // Ensure positive tracers stay positive
            if (tracer_positive(l))  tracers_tmp(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i,iens) );
          }
        });
        //////////////
        // Stage 3
        //////////////
        compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,2.*dt_dyn/3.);
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = (1._fp/3._fp) * state      (l,hs+k,hs+j,hs+i,iens) +
                                             (2._fp/3._fp) * state_tmp  (l,hs+k,hs+j,hs+i,iens) +
                                             (2._fp/3._fp) * dt_dyn * state_tend  (l,k,j,i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = (1._fp/3._fp) * tracers    (l,hs+k,hs+j,hs+i,iens) +
                                             (2._fp/3._fp) * tracers_tmp(l,hs+k,hs+j,hs+i,iens) +
                                             (2._fp/3._fp) * dt_dyn * tracers_tend(l,k,j,i,iens);
            // Ensure positive tracers stay positive
            if (tracer_positive(l))  tracers(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers(l,hs+k,hs+j,hs+i,iens) );
          }
        });
      }
    }



    // Once you encounter an immersed boundary, set zero derivative boundary conditions
    YAKL_INLINE static void modify_stencil_immersed_der0( SArray<real,1,ord>       & stencil  ,
                                                          SArray<bool,1,ord> const & immersed ) {
      // Don't modify the stencils of immersed cells
      if (! immersed(hs)) {
        // Move out from the center of the stencil. once you encounter a boundary, enforce zero derivative,
        //     which is essentially replication of the last in-domain value
        for (int i2=hs+1; i2<ord; i2++) {
          if (immersed(i2)) { for (int i3=i2; i3<ord; i3++) { stencil(i3) = stencil(i2-1); }; break; }
        }
        for (int i2=hs-1; i2>=0 ; i2--) {
          if (immersed(i2)) { for (int i3=i2; i3>=0 ; i3--) { stencil(i3) = stencil(i2+1); }; break; }
        }
      }
    }



    // The issue with having zero-value boundary conditions is that nothing can flux through that boundary
    // Often the next cell interface's flux needs to "feel" the zero value in order to avoid fluxing too much
    // into the cell adjacent to the boundary.
    template <class LIM3, class LIM5>
    YAKL_INLINE static void do_recon( SArray<real,1,5>   const & stencil  ,
                                      SArray<real,1,2>         & gll      ,
                                      SArray<real,2,3,2> const & s2g3     ,
                                      SArray<real,2,5,2> const & s2g5     ,
                                      SArray<real,2,3,2> const & c2g3     ,
                                      SArray<real,2,5,2> const & c2g5     ,
                                      LIM3               const & lim3     ,
                                      LIM5               const & lim5     ,
                                      SArray<bool,1,5>   const & immersed ,
                                      bool                       zero_vel ,
                                      bool                       use_weno ) {
      if (use_weno) {
        if (zero_vel) {
          if        (immersed(hs-1) || immersed(hs+1)) {
            SArray<real,1,3> stencil3;
            for (int i=0; i < 3; i++) { stencil3(i) = stencil(i+1); }
            reconstruct_gll_values(stencil3,gll,s2g3);
          } else if (immersed(hs-2) || immersed(hs+2)) {
            reconstruct_gll_values(stencil ,gll,s2g5);
          } else {
            reconstruct_gll_values(stencil ,gll,c2g5,lim5);
          }
        } else {
          reconstruct_gll_values(stencil ,gll,c2g5,lim5);
        }
      } else {
        reconstruct_gll_values(stencil,gll,s2g5);
      }
    }



    // Compute semi-discrete tendencies in x, y, and z directions
    // Fully split in dimensions, and coupled together inside RK stages
    // dt is not used at the moment
    void compute_tendencies( core::Coupler       & coupler      ,
                             real5d        const & state        ,
                             real5d        const & state_tend   ,
                             real5d        const & tracers      ,
                             real5d        const & tracers_tend ,
                             real                  dt           ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      using std::min;
      using std::max;

      auto nens                      = coupler.get_nens();
      auto nx                        = coupler.get_nx();    // Proces-local number of cells
      auto ny                        = coupler.get_ny();    // Proces-local number of cells
      auto nz                        = coupler.get_nz();    // Total vertical cells
      auto dx                        = coupler.get_dx();    // grid spacing
      auto dy                        = coupler.get_dy();    // grid spacing
      auto dz                        = coupler.get_dz();    // grid spacing
      auto sim2d                     = coupler.is_sim2d();  // Is this a 2-D simulation?
      auto C0                        = coupler.get_option<real>("C0"     );  // pressure = C0*pow(rho*theta,gamma)
      auto grav                      = coupler.get_option<real>("grav"   );  // Gravity
      auto gamma                     = coupler.get_option<real>("gamma_d");  // cp_dry / cv_dry (about 1.4)
      auto latitude                  = coupler.get_option<real>("latitude"); // For coriolis
      auto num_tracers               = coupler.get_num_tracers();            // Number of tracers
      auto &dm                       = coupler.get_data_manager_readonly();  // Grab read-only data manager
      auto use_weno                  = coupler.get_option<bool>("use_weno",true);
      auto tracer_positive           = dm.get<bool const,1>("tracer_positive"          ); // Is a tracer
                                                                                          //   positive-definite?
      auto immersed_proportion_halos = dm.get<real const,4>("immersed_proportion_halos"); // Proportion of immersed
                                                                                          //   material in each cell
      auto fully_immersed_halos      = dm.get<bool const,4>("fully_immersed_halos"     ); // Is a cell fullly immersed?
      auto hy_dens_cells             = dm.get<real const,2>("hy_dens_cells"            ); // Hydrostatic density
      auto hy_theta_cells            = dm.get<real const,2>("hy_theta_cells"           ); // Hydrostatic potential
                                                                                          //  temperature
      auto hy_dens_edges             = dm.get<real const,2>("hy_dens_edges"            ); // Hydrostatic density
      auto hy_theta_edges            = dm.get<real const,2>("hy_theta_edges"           ); // Hydrostatic potential
                                                                                          //  temperature
      auto hy_pressure_cells         = dm.get<real const,2>("hy_pressure_cells"        ); // Hydrostatic pressure
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      SArray<real,2,3,2> c2g3, s2g3;
      SArray<real,2,5,2> c2g5, s2g5;
      SArray<real,2,3,3> s2c3;
      SArray<real,2,5,5> s2c5;
      TransformMatrices::coefs_to_gll_lower(c2g3);
      TransformMatrices::coefs_to_gll_lower(c2g5);
      TransformMatrices::sten_to_coefs     (s2c3);
      TransformMatrices::sten_to_coefs     (s2c5);
      s2g3 = yakl::intrinsics::matmul_cr(c2g3,s2c3); // Matrix-matrix multiply in column-row format
      s2g5 = yakl::intrinsics::matmul_cr(c2g5,s2c5); // Matrix-matrix multiply in column-row format
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real r_dz = 1./dz; // reciprocal of grid spacing
      real4d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs,nens); // Holds pressure perturbation
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

      // Compute pressure perturbation. Divide rho*u, rho*v, rho*w, rho*theta, rho*tracer by density to get
      //   velocities, theta, and tracer concentrations to make boundary conditions easier to implement.
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        pressure(hs+k,hs+j,hs+i,iens) = C0*std::pow(state(idT,hs+k,hs+j,hs+i,iens),gamma) - hy_pressure_cells(hs+k,iens);
        real rdens = 1._fp / state(idR,hs+k,hs+j,hs+i,iens);
        state(idU,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idV,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idW,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idT,hs+k,hs+j,hs+i,iens) *= rdens;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) *= rdens; }
      });

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      core::MultiField<real,4> fields;
      for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<4>(l,0,0,0,0) ); }
      for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<4>(l,0,0,0,0) ); }
      fields.add_field( pressure );
      if (ord > 1) halo_exchange( coupler , fields );

      halo_boundary_conditions( coupler , state , tracers , pressure );

      // To hold left and right interface recontsructions for each cell in each direction
      real6d state_limits_x   ("state_limits_x"   ,2,num_state  ,nz,ny,nx+1,nens);
      real6d state_limits_y   ("state_limits_y"   ,2,num_state  ,nz,ny+1,nx,nens);
      real6d state_limits_z   ("state_limits_z"   ,2,num_state  ,nz+1,ny,nx,nens);
      real6d tracers_limits_x ("tracers_limits_x" ,2,num_tracers,nz,ny,nx+1,nens);
      real6d tracers_limits_y ("tracers_limits_y" ,2,num_tracers,nz,ny+1,nx,nens);
      real6d tracers_limits_z ("tracers_limits_z" ,2,num_tracers,nz+1,ny,nx,nens);
      real5d pressure_limits_x("pressure_limits_x",2            ,nz,ny,nx+1,nens);
      real5d pressure_limits_y("pressure_limits_y",2            ,nz,ny+1,nx,nens);
      real5d pressure_limits_z("pressure_limits_z",2            ,nz+1,ny,nx,nens);

      // Create a WENO limiter
      limiter::WenoLimiter<3> lim3(0.1,1,1,1.e2);
      limiter::WenoLimiter<5> lim5(0.1,1,2,1,1.e3);

      // The kernels below are split because if they are not, the AMD compiler does a very poor job with register reuse
      //   and the code slows down significantly on AMD GPUs.
      // Reconstruct cell-edge estimates of density for each cell for each dimension
      // In interpolation, do not use WENO in the presence of immersed boundaries because Dirichlet BCs would likely get
      //   ignored in WENO interpolation.
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        SArray<real,1,2  > gll;
        { // Density x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idR,hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { immersed(ii) = fully_immersed_halos(hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { if (immersed(ii)) stencil(ii) = hy_dens_cells(hs+k,iens); }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_x(1,idR,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idR,k,j,i+1,iens) = gll(1);
        }
        { // Density y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idR,hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { immersed(jj) = fully_immersed_halos(hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { if (immersed(jj)) stencil(jj) = hy_dens_cells(hs+k,iens); }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_y(1,idR,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idR,k,j+1,i,iens) = gll(1);
        }
        { // Density z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idR,k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { immersed(kk) = fully_immersed_halos(k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { if (immersed(kk)) stencil(kk) = hy_dens_cells(k+kk,iens); }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_z(1,idR,k  ,j,i,iens) = gll(0);
          state_limits_z(0,idR,k+1,j,i,iens) = gll(1);
        }
      });
      // Reconstruct cell-edge estimates of u-velocity for each cell for each dimension
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        SArray<real,1,2  > gll;
        { // u-vel x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idU,hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { immersed(ii) = fully_immersed_halos(hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { if (immersed(ii)) stencil(ii) = 0; }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , true  , use_weno );
          state_limits_x(1,idU,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idU,k,j,i+1,iens) = gll(1);
        }
        { // u-vel y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idU,hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { immersed(jj) = fully_immersed_halos(hs+k,j+jj,hs+i,iens); }
          modify_stencil_immersed_der0( stencil , immersed );
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_y(1,idU,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idU,k,j+1,i,iens) = gll(1);
        }
        { // u-vel z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idU,k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { immersed(kk) = fully_immersed_halos(k+kk,hs+j,hs+i,iens); }
          modify_stencil_immersed_der0( stencil , immersed );
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_z(1,idU,k  ,j,i,iens) = gll(0);
          state_limits_z(0,idU,k+1,j,i,iens) = gll(1);
        }
      });
      // Reconstruct cell-edge estimates of v-velocity for each cell for each dimension
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        SArray<real,1,2  > gll;
        { // v-vel x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idV,hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { immersed(ii) = fully_immersed_halos(hs+k,hs+j,i+ii,iens); }
          modify_stencil_immersed_der0( stencil , immersed );
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_x(1,idV,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idV,k,j,i+1,iens) = gll(1);
        }
        { // v-vel y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idV,hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { immersed(jj) = fully_immersed_halos(hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { if (immersed(jj)) stencil(jj) = 0; }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , true  , use_weno );
          state_limits_y(1,idV,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idV,k,j+1,i,iens) = gll(1);
        }
        { // v-vel z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idV,k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { immersed(kk) = fully_immersed_halos(k+kk,hs+j,hs+i,iens); }
          modify_stencil_immersed_der0( stencil , immersed );
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_z(1,idV,k  ,j,i,iens) = gll(0);
          state_limits_z(0,idV,k+1,j,i,iens) = gll(1);
        }
      });
      // Reconstruct cell-edge estimates of w-velocity for each cell for each dimension
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        SArray<real,1,2  > gll;
        { // w-vel x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idW,hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { immersed(ii) = fully_immersed_halos(hs+k,hs+j,i+ii,iens); }
          modify_stencil_immersed_der0( stencil , immersed );
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_x(1,idW,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idW,k,j,i+1,iens) = gll(1);
        }
        { // w-vel y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idW,hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { immersed(jj) = fully_immersed_halos(hs+k,j+jj,hs+i,iens); }
          modify_stencil_immersed_der0( stencil , immersed );
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_y(1,idW,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idW,k,j+1,i,iens) = gll(1);
        }
        { // w-vel z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idW,k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { immersed(kk) = fully_immersed_halos(k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { if (immersed(kk)) stencil(kk) = 0; }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , true  , use_weno );
          state_limits_z(1,idW,k  ,j,i,iens) = gll(0);
          state_limits_z(0,idW,k+1,j,i,iens) = gll(1);
        }
      });
      // Reconstruct cell-edge estimates of potential temperature for each cell for each dimension
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        SArray<real,1,2  > gll;
        { // Theta x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idT,hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { immersed(ii) = fully_immersed_halos(hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { if (immersed(ii)) stencil(ii) = hy_theta_cells(hs+k,iens); }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_x(1,idT,k,j,i  ,iens) = gll(0);
          state_limits_x(0,idT,k,j,i+1,iens) = gll(1);
        }
        { // Theta y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idT,hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { immersed(jj) = fully_immersed_halos(hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { if (immersed(jj)) stencil(jj) = hy_theta_cells(hs+k,iens); }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_y(1,idT,k,j  ,i,iens) = gll(0);
          state_limits_y(0,idT,k,j+1,i,iens) = gll(1);
        }
        { // Theta z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idT,k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { immersed(kk) = fully_immersed_halos(k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { if (immersed(kk)) stencil(kk) = hy_theta_cells(k+kk,iens); }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          state_limits_z(1,idT,k  ,j,i,iens) = gll(0);
          state_limits_z(0,idT,k+1,j,i,iens) = gll(1);
        }
      });
      // Reconstruct cell-edge estimates of pressure for each cell for each dimension
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        SArray<real,1,2  > gll;
        { // Pressure x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = pressure(hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { immersed(ii) = fully_immersed_halos(hs+k,hs+j,i+ii,iens); }
          modify_stencil_immersed_der0( stencil , immersed );
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          pressure_limits_x(1,k,j,i  ,iens) = gll(0);
          pressure_limits_x(0,k,j,i+1,iens) = gll(1);
        }
        { // Pressure y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = pressure(hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { immersed(jj) = fully_immersed_halos(hs+k,j+jj,hs+i,iens); }
          modify_stencil_immersed_der0( stencil , immersed );
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          pressure_limits_y(1,k,j  ,i,iens) = gll(0);
          pressure_limits_y(0,k,j+1,i,iens) = gll(1);
        }
        { // Pressure z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = pressure(k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { immersed(kk) = fully_immersed_halos(k+kk,hs+j,hs+i,iens); }
          modify_stencil_immersed_der0( stencil , immersed );
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          pressure_limits_z(1,k  ,j,i,iens) = gll(0);
          pressure_limits_z(0,k+1,j,i,iens) = gll(1);
        }
      });
      // Reconstruct cell-edge estimates of tracers for each cell for each dimension
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        SArray<real,1,2  > gll;
        { // Tracer x
          for (int ii=0; ii < ord; ii++) { stencil(ii) = tracers(l,hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { immersed(ii) = fully_immersed_halos(hs+k,hs+j,i+ii,iens); }
          for (int ii=0; ii < ord; ii++) { if (immersed(ii)) stencil(ii) = 0; }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          tracers_limits_x(1,l,k,j,i  ,iens) = gll(0);
          tracers_limits_x(0,l,k,j,i+1,iens) = gll(1);
        }
        { // Tracer y
          for (int jj=0; jj < ord; jj++) { stencil(jj) = tracers(l,hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { immersed(jj) = fully_immersed_halos(hs+k,j+jj,hs+i,iens); }
          for (int jj=0; jj < ord; jj++) { if (immersed(jj)) stencil(jj) = 0; }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          tracers_limits_y(1,l,k,j  ,i,iens) = gll(0);
          tracers_limits_y(0,l,k,j+1,i,iens) = gll(1);
        }
        { // Tracer z
          for (int kk=0; kk < ord; kk++) { stencil(kk) = tracers(l,k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { immersed(kk) = fully_immersed_halos(k+kk,hs+j,hs+i,iens); }
          for (int kk=0; kk < ord; kk++) { if (immersed(kk)) stencil(kk) = 0; }
          do_recon( stencil , gll , s2g3 , s2g5 , c2g3 , c2g5 , lim3 , lim5 , immersed , false , use_weno );
          tracers_limits_z(1,l,k  ,j,i,iens) = gll(0);
          tracers_limits_z(0,l,k+1,j,i,iens) = gll(1);
        }
      });
      
      // Perform periodic horizontal exchange of cell-edge data, and implement vertical boundary conditions
      edge_exchange( coupler , state_limits_x , tracers_limits_x , pressure_limits_x ,
                               state_limits_y , tracers_limits_y , pressure_limits_y ,
                               state_limits_z , tracers_limits_z , pressure_limits_z );

      // To save on space, slice the limits arrays to store single-valued interface fluxes
      using yakl::COLON;
      auto state_flux_x   = state_limits_x  .slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto state_flux_y   = state_limits_y  .slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto state_flux_z   = state_limits_z  .slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_flux_x = tracers_limits_x.slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_flux_y = tracers_limits_y.slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_flux_z = tracers_limits_z.slice<5>(0,COLON,COLON,COLON,COLON,COLON);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      // Speed of sound and its reciprocal. Using a constant speed of sound for upwinding
      real constexpr cs   = 350.;
      real constexpr r_cs = 1./cs;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz+1,ny+1,nx+1,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        if (j < ny && k < nz) {
          // Apply boundary conditions if encountering an immersed boundary
          bool left_cell_immersed  = fully_immersed_halos(hs+k,hs+j,hs+i-1,iens);
          bool right_cell_immersed = fully_immersed_halos(hs+k,hs+j,hs+i  ,iens);
          if (!left_cell_immersed && right_cell_immersed) {
            // Use only left values
            state_limits_x   (0,idU,k,j,i,iens) = 0;
            state_limits_x   (1,idU,k,j,i,iens) = 0;
            state_limits_x   (0,idR,k,j,i,iens) = hy_dens_cells (hs+k,iens);
            state_limits_x   (1,idR,k,j,i,iens) = hy_dens_cells (hs+k,iens);
            state_limits_x   (0,idT,k,j,i,iens) = hy_theta_cells(hs+k,iens);
            state_limits_x   (1,idT,k,j,i,iens) = hy_theta_cells(hs+k,iens);
            state_limits_x   (1,idV,k,j,i,iens) = state_limits_x   (0,idV,k,j,i,iens);
            state_limits_x   (1,idW,k,j,i,iens) = state_limits_x   (0,idW,k,j,i,iens);
            pressure_limits_x(1    ,k,j,i,iens) = pressure_limits_x(0    ,k,j,i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_limits_x(0,tr,k,j,i,iens) = 0;
              tracers_limits_x(1,tr,k,j,i,iens) = 0;
            }
          }
          if (!right_cell_immersed && left_cell_immersed) {
            // Use only right values
            state_limits_x   (0,idU,k,j,i,iens) = 0;
            state_limits_x   (1,idU,k,j,i,iens) = 0;
            state_limits_x   (0,idR,k,j,i,iens) = hy_dens_cells (hs+k,iens);
            state_limits_x   (1,idR,k,j,i,iens) = hy_dens_cells (hs+k,iens);
            state_limits_x   (0,idT,k,j,i,iens) = hy_theta_cells(hs+k,iens);
            state_limits_x   (1,idT,k,j,i,iens) = hy_theta_cells(hs+k,iens);
            state_limits_x   (0,idV,k,j,i,iens) = state_limits_x   (1,idV,k,j,i,iens);
            state_limits_x   (0,idW,k,j,i,iens) = state_limits_x   (1,idW,k,j,i,iens);
            pressure_limits_x(0    ,k,j,i,iens) = pressure_limits_x(1    ,k,j,i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_limits_x(0,tr,k,j,i,iens) = 0;
              tracers_limits_x(1,tr,k,j,i,iens) = 0;
            }
          }
          // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
          real r_L = state_limits_x   (0,idR,k,j,i,iens);   real r_R = state_limits_x   (1,idR,k,j,i,iens);
          real u_L = state_limits_x   (0,idU,k,j,i,iens);   real u_R = state_limits_x   (1,idU,k,j,i,iens);
          real t_L = state_limits_x   (0,idT,k,j,i,iens);   real t_R = state_limits_x   (1,idT,k,j,i,iens);
          real p_L = pressure_limits_x(0    ,k,j,i,iens);   real p_R = pressure_limits_x(1    ,k,j,i,iens);
          real w1 = 0.5_fp * (p_R-cs*r_R*u_R);
          real w2 = 0.5_fp * (p_L+cs*r_L*u_L);
          real p_upw  = w1 + w2;
          real ru_upw = (w2-w1)*r_cs;
          // Advectively upwind everything else
          int ind = ru_upw > 0 ? 0 : 1;
          state_flux_x(idR,k,j,i,iens) = ru_upw;
          state_flux_x(idU,k,j,i,iens) = ru_upw*state_limits_x(ind,idU,k,j,i,iens) + p_upw;
          state_flux_x(idV,k,j,i,iens) = ru_upw*state_limits_x(ind,idV,k,j,i,iens);
          state_flux_x(idW,k,j,i,iens) = ru_upw*state_limits_x(ind,idW,k,j,i,iens);
          state_flux_x(idT,k,j,i,iens) = ru_upw*state_limits_x(ind,idT,k,j,i,iens);
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_x(tr,k,j,i,iens) = ru_upw*tracers_limits_x(ind,tr,k,j,i,iens);
          }
        }
        if (i < nx && k < nz && !sim2d) {
          // Apply boundary conditions if encountering an immersed boundary
          bool left_cell_immersed  = fully_immersed_halos(hs+k,hs+j-1,hs+i,iens);
          bool right_cell_immersed = fully_immersed_halos(hs+k,hs+j  ,hs+i,iens);
          if (!left_cell_immersed && right_cell_immersed) {
            // Use only left values
            state_limits_y   (0,idV,k,j,i,iens) = 0;
            state_limits_y   (1,idV,k,j,i,iens) = 0;
            state_limits_y   (0,idR,k,j,i,iens) = hy_dens_cells (hs+k,iens);
            state_limits_y   (1,idR,k,j,i,iens) = hy_dens_cells (hs+k,iens);
            state_limits_y   (0,idT,k,j,i,iens) = hy_theta_cells(hs+k,iens);
            state_limits_y   (1,idT,k,j,i,iens) = hy_theta_cells(hs+k,iens);
            state_limits_y   (1,idU,k,j,i,iens) = state_limits_y   (0,idU,k,j,i,iens);
            state_limits_y   (1,idW,k,j,i,iens) = state_limits_y   (0,idW,k,j,i,iens);
            pressure_limits_y(1    ,k,j,i,iens) = pressure_limits_y(0    ,k,j,i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_limits_y(0,tr,k,j,i,iens) = 0;
              tracers_limits_y(1,tr,k,j,i,iens) = 0;
            }
          }
          if (!right_cell_immersed && left_cell_immersed) {
            // Use only right values
            state_limits_y   (0,idV,k,j,i,iens) = 0;
            state_limits_y   (1,idV,k,j,i,iens) = 0;
            state_limits_y   (0,idR,k,j,i,iens) = hy_dens_cells (hs+k,iens);
            state_limits_y   (1,idR,k,j,i,iens) = hy_dens_cells (hs+k,iens);
            state_limits_y   (0,idT,k,j,i,iens) = hy_theta_cells(hs+k,iens);
            state_limits_y   (1,idT,k,j,i,iens) = hy_theta_cells(hs+k,iens);
            state_limits_y   (0,idU,k,j,i,iens) = state_limits_y   (1,idU,k,j,i,iens);
            state_limits_y   (0,idW,k,j,i,iens) = state_limits_y   (1,idW,k,j,i,iens);
            pressure_limits_y(0    ,k,j,i,iens) = pressure_limits_y(1    ,k,j,i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_limits_y(0,tr,k,j,i,iens) = 0;
              tracers_limits_y(1,tr,k,j,i,iens) = 0;
            }
          }
          // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
          real r_L = state_limits_y   (0,idR,k,j,i,iens);   real r_R = state_limits_y   (1,idR,k,j,i,iens);
          real v_L = state_limits_y   (0,idV,k,j,i,iens);   real v_R = state_limits_y   (1,idV,k,j,i,iens);
          real t_L = state_limits_y   (0,idT,k,j,i,iens);   real t_R = state_limits_y   (1,idT,k,j,i,iens);
          real p_L = pressure_limits_y(0    ,k,j,i,iens);   real p_R = pressure_limits_y(1    ,k,j,i,iens);
          real w1 = 0.5_fp * (p_R-cs*r_R*v_R);
          real w2 = 0.5_fp * (p_L+cs*r_L*v_L);
          real p_upw  = w1 + w2;
          real rv_upw = (w2-w1)*r_cs;
          // Advectively upwind everything else
          int ind = rv_upw > 0 ? 0 : 1;
          state_flux_y(idR,k,j,i,iens) = rv_upw;
          state_flux_y(idU,k,j,i,iens) = rv_upw*state_limits_y(ind,idU,k,j,i,iens);
          state_flux_y(idV,k,j,i,iens) = rv_upw*state_limits_y(ind,idV,k,j,i,iens) + p_upw;
          state_flux_y(idW,k,j,i,iens) = rv_upw*state_limits_y(ind,idW,k,j,i,iens);
          state_flux_y(idT,k,j,i,iens) = rv_upw*state_limits_y(ind,idT,k,j,i,iens);
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_y(tr,k,j,i,iens) = rv_upw*tracers_limits_y(ind,tr,k,j,i,iens);
          }
        }
        if (i < nx && j < ny) {
          // Apply boundary conditions if encountering an immersed boundary
          bool left_cell_immersed  = fully_immersed_halos(hs+k-1,hs+j,hs+i,iens);
          bool right_cell_immersed = fully_immersed_halos(hs+k  ,hs+j,hs+i,iens);
          if (!left_cell_immersed && right_cell_immersed && k != 0 && k != nz) {
            // Use only left values
            state_limits_z   (0,idW,k,j,i,iens) = 0;
            state_limits_z   (1,idW,k,j,i,iens) = 0;
            state_limits_z   (0,idR,k,j,i,iens) = hy_dens_edges (k,iens);
            state_limits_z   (1,idR,k,j,i,iens) = hy_dens_edges (k,iens);
            state_limits_z   (0,idT,k,j,i,iens) = hy_theta_edges(k,iens);
            state_limits_z   (1,idT,k,j,i,iens) = hy_theta_edges(k,iens);
            state_limits_z   (1,idU,k,j,i,iens) = state_limits_z   (0,idU,k,j,i,iens);
            state_limits_z   (1,idV,k,j,i,iens) = state_limits_z   (0,idV,k,j,i,iens);
            pressure_limits_z(1    ,k,j,i,iens) = pressure_limits_z(0    ,k,j,i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_limits_z(0,tr,k,j,i,iens) = 0;
              tracers_limits_z(1,tr,k,j,i,iens) = 0;
            }
          }
          if (!right_cell_immersed && left_cell_immersed && k != 0 && k != nz) {
            // Use only right values
            state_limits_z   (0,idW,k,j,i,iens) = 0;
            state_limits_z   (1,idW,k,j,i,iens) = 0;
            state_limits_z   (0,idR,k,j,i,iens) = hy_dens_edges (k,iens);
            state_limits_z   (1,idR,k,j,i,iens) = hy_dens_edges (k,iens);
            state_limits_z   (0,idT,k,j,i,iens) = hy_theta_edges(k,iens);
            state_limits_z   (1,idT,k,j,i,iens) = hy_theta_edges(k,iens);
            state_limits_z   (0,idU,k,j,i,iens) = state_limits_z   (1,idU,k,j,i,iens);
            state_limits_z   (0,idV,k,j,i,iens) = state_limits_z   (1,idV,k,j,i,iens);
            pressure_limits_z(0    ,k,j,i,iens) = pressure_limits_z(1    ,k,j,i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_limits_z(0,tr,k,j,i,iens) = 0;
              tracers_limits_z(1,tr,k,j,i,iens) = 0;
            }
          }
          // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
          real r_L = state_limits_z   (0,idR,k,j,i,iens);   real r_R = state_limits_z   (1,idR,k,j,i,iens);
          real w_L = state_limits_z   (0,idW,k,j,i,iens);   real w_R = state_limits_z   (1,idW,k,j,i,iens);
          real t_L = state_limits_z   (0,idT,k,j,i,iens);   real t_R = state_limits_z   (1,idT,k,j,i,iens);
          real p_L = pressure_limits_z(0    ,k,j,i,iens);   real p_R = pressure_limits_z(1    ,k,j,i,iens);
          real w1 = 0.5_fp * (p_R-cs*r_R*w_R);
          real w2 = 0.5_fp * (p_L+cs*r_L*w_L);
          real p_upw  = w1 + w2;
          real rw_upw = (w2-w1)*r_cs;
          // Advectively upwind everything else
          int ind = rw_upw > 0 ? 0 : 1;
          state_flux_z(idR,k,j,i,iens) = rw_upw;
          state_flux_z(idU,k,j,i,iens) = rw_upw*state_limits_z(ind,idU,k,j,i,iens);
          state_flux_z(idV,k,j,i,iens) = rw_upw*state_limits_z(ind,idV,k,j,i,iens);
          state_flux_z(idW,k,j,i,iens) = rw_upw*state_limits_z(ind,idW,k,j,i,iens) + p_upw;
          state_flux_z(idT,k,j,i,iens) = rw_upw*state_limits_z(ind,idT,k,j,i,iens);
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_z(tr,k,j,i,iens) = rw_upw*tracers_limits_z(ind,tr,k,j,i,iens);
          }
        }
        // Multiply density back onto the state values before calculating tendencies
        if (i < nx && j < ny && k < nz) {
          real dens = state(idR,hs+k,hs+j,hs+i,iens);
          state(idU,hs+k,hs+j,hs+i,iens) *= dens;
          state(idV,hs+k,hs+j,hs+i,iens) *= dens;
          state(idW,hs+k,hs+j,hs+i,iens) *= dens;
          state(idT,hs+k,hs+j,hs+i,iens) *= dens;
          for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) *= dens; }
        }
      });

      // Compute tendencies as the flux divergence + gravity source term + coriolis
      // If the cell is immersed, do not allow a tendency
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        for (int l = 0; l < num_state; l++) {
          state_tend(l,k,j,i,iens) = -( state_flux_x(l,k,j,i+1,iens) - state_flux_x(l,k,j,i,iens) ) * r_dx
                                     -( state_flux_y(l,k,j+1,i,iens) - state_flux_y(l,k,j,i,iens) ) * r_dy
                                     -( state_flux_z(l,k+1,j,i,iens) - state_flux_z(l,k,j,i,iens) ) * r_dz;
          if (l == idV && sim2d) state_tend(l,k,j,i,iens) = 0;
          if (l == idW) state_tend(l,k,j,i,iens) += -grav*(state(idR,hs+k,hs+j,hs+i,iens) - hy_dens_cells(hs+k,iens));
          if (latitude != 0 && !sim2d && l == idU) state_tend(l,k,j,i,iens) += fcor*state(idV,hs+k,hs+j,hs+i,iens);
          if (latitude != 0 && !sim2d && l == idV) state_tend(l,k,j,i,iens) -= fcor*state(idU,hs+k,hs+j,hs+i,iens);
          if (fully_immersed_halos(hs+k,hs+j,hs+i,iens)) state_tend(l,k,j,i,iens) = 0;
        }
        for (int l = 0; l < num_tracers; l++) {
          tracers_tend(l,k,j,i,iens) = -( tracers_flux_x(l,k,j,i+1,iens) - tracers_flux_x(l,k,j,i,iens) ) * r_dx
                                       -( tracers_flux_y(l,k,j+1,i,iens) - tracers_flux_y(l,k,j,i,iens) ) * r_dy 
                                       -( tracers_flux_z(l,k+1,j,i,iens) - tracers_flux_z(l,k,j,i,iens) ) * r_dz;
          if (fully_immersed_halos(hs+k,hs+j,hs+i,iens)) tracers_tend(l,k,j,i,iens) = 0;
        }
      });
    }



    // Use a WENO limiter (or any limiter) to compute coefficients. Then sample polynomial at cell edges
    template <yakl::index_t ord, class LIMITER>
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>     const & stencil      ,
                                                    SArray<real,1,2  >           & gll          ,
                                                    SArray<real,2,ord,2>   const & coefs_to_gll ,
                                                    LIMITER                const & limiter      ) {
      // Apply WENO to compute coefficients from the stencil
      SArray<real,1,ord> wenoCoefs;
      limiter.compute_limited_coefs( stencil , wenoCoefs );
      // Compute left and right cell edge estimates from coefficients
      for (int ii=0; ii<2; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) { tmp += coefs_to_gll(s,ii) * wenoCoefs(s); }
        gll(ii) = tmp;
      }
    }



    // Project stencil averages to cell-edge interpolations (No limiter)
    template <yakl::index_t ord>
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>   const & stencil     ,
                                                    SArray<real,1,2  >         & gll         ,
                                                    SArray<real,2,ord,2> const & sten_to_gll ) {
      // Compute left and right cell edge estimates from stencil cell averages
      for (int ii=0; ii<2; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) { tmp += sten_to_gll(s,ii) * stencil(s); }
        gll(ii) = tmp;
      }
    }



    // Exchange halo values periodically in the horizontal, and apply vertical no-slip solid-wall BC's
    void halo_exchange( core::Coupler const & coupler , core::MultiField<real,4> & fields ) const {
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

      // x-direction exchanges
      {
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

      // y-direction exchanges
      {
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
    }



    void halo_boundary_conditions( core::Coupler const & coupler  ,
                                   real5d        const & state    ,
                                   real5d        const & tracers  ,
                                   real4d        const & pressure ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens          = coupler.get_nens();
      auto nx            = coupler.get_nx();
      auto ny            = coupler.get_ny();
      auto nz            = coupler.get_nz();
      auto dz            = coupler.get_dz();
      auto num_tracers   = coupler.get_num_tracers();
      auto &neigh        = coupler.get_neighbor_rankid_matrix();
      auto dtype         = coupler.get_mpi_data_type();
      auto grav          = coupler.get_option<real>("grav");
      auto gamma         = coupler.get_option<real>("gamma_d");
      auto C0            = coupler.get_option<real>("C0");
      auto hy_dens_cells = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_cells");

      // z-direction BC's
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(hs,ny+2*hs,nx+2*hs,nens) ,
                                        YAKL_LAMBDA (int kk, int j, int i, int iens) {
        state(idR,kk,j,i,iens) = hy_dens_cells (kk,iens);
        state(idU,kk,j,i,iens) = state(idU,hs+0,j,i,iens);
        state(idV,kk,j,i,iens) = state(idV,hs+0,j,i,iens);
        state(idW,kk,j,i,iens) = 0;
        state(idT,kk,j,i,iens) = state(idT,hs+0,j,i,iens);
        pressure( kk,j,i,iens) = pressure( hs+0,j,i,iens);
        state(idR,hs+nz+kk,j,i,iens) = hy_dens_cells(hs+nz+kk,iens);
        state(idU,hs+nz+kk,j,i,iens) = state(idU,hs+nz-1,j,i,iens);
        state(idV,hs+nz+kk,j,i,iens) = state(idV,hs+nz-1,j,i,iens);
        state(idW,hs+nz+kk,j,i,iens) = 0;
        state(idT,hs+nz+kk,j,i,iens) = state(idT,hs+nz-1,j,i,iens);
        pressure( hs+nz+kk,j,i,iens) = pressure( hs+nz-1,j,i,iens);
        for (int l=0; l < num_tracers; l++) {
          tracers(l,      kk,j,i,iens) = tracers(l,hs+0   ,j,i,iens);
          tracers(l,hs+nz+kk,j,i,iens) = tracers(l,hs+nz-1,j,i,iens);
        }
      });
    }



    void edge_exchange( core::Coupler const & coupler           ,
                        real6d        const & state_limits_x    ,
                        real6d        const & tracers_limits_x  ,
                        real5d        const & pressure_limits_x ,
                        real6d        const & state_limits_y    ,
                        real6d        const & tracers_limits_y  ,
                        real5d        const & pressure_limits_y ,
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
      auto &neigh      = coupler.get_neighbor_rankid_matrix();
      auto dtype       = coupler.get_mpi_data_type();
      auto comm        = MPI_COMM_WORLD;
      int npack = num_state + num_tracers+1;
      MPI_Request sReq [2];
      MPI_Request rReq [2];
      MPI_Status  sStat[2];
      MPI_Status  rStat[2];

      // x-exchange
      {
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

      // y-direction exchange
      {
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

      // z-direction BC's
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(ny,nx,nens) ,
                                        YAKL_LAMBDA (int j, int i, int iens) {
        state_limits_z(0,idR,0 ,j,i,iens) = state_limits_z(1,idR,0 ,j,i,iens);
        state_limits_z(0,idU,0 ,j,i,iens) = state_limits_z(1,idU,0 ,j,i,iens);
        state_limits_z(0,idV,0 ,j,i,iens) = state_limits_z(1,idV,0 ,j,i,iens);
        state_limits_z(0,idW,0 ,j,i,iens) = 0;  state_limits_z(1,idW,0 ,j,i,iens) = 0;
        state_limits_z(0,idT,0 ,j,i,iens) = state_limits_z(1,idT,0 ,j,i,iens);
        state_limits_z(1,idR,nz,j,i,iens) = state_limits_z(0,idR,nz,j,i,iens);
        state_limits_z(1,idU,nz,j,i,iens) = state_limits_z(0,idU,nz,j,i,iens);
        state_limits_z(1,idV,nz,j,i,iens) = state_limits_z(0,idV,nz,j,i,iens);
        state_limits_z(1,idW,nz,j,i,iens) = 0;  state_limits_z(0,idW,nz,j,i,iens) = 0;
        state_limits_z(1,idT,nz,j,i,iens) = state_limits_z(0,idT,nz,j,i,iens);
        for (int l=0; l < num_tracers; l++) {
          tracers_limits_z(0,l,0 ,j,i,iens) = tracers_limits_z(1,l,0 ,j,i,iens);
          tracers_limits_z(1,l,nz,j,i,iens) = tracers_limits_z(0,l,nz,j,i,iens);
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
        if (! dm.entry_exists("hy_dens_edges" )) dm.register_and_allocate<real>("hy_dens_edges" ,"",{nz+1,nens});
        if (! dm.entry_exists("hy_theta_edges")) dm.register_and_allocate<real>("hy_theta_edges","",{nz+1,nens});
        auto hy_dens_cells  = dm.get<real const,2>("hy_dens_cells" );
        auto hy_theta_cells = dm.get<real const,2>("hy_theta_cells");
        auto hy_dens_edges  = dm.get<real      ,2>("hy_dens_edges" );
        auto hy_theta_edges = dm.get<real      ,2>("hy_theta_edges");
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz+1,nens) , YAKL_LAMBDA (int k, int iens) {
          hy_dens_edges(k,iens) = std::exp( -1./12.*std::log(hy_dens_cells(hs+k-2,iens)) +
                                             7./12.*std::log(hy_dens_cells(hs+k-1,iens)) +
                                             7./12.*std::log(hy_dens_cells(hs+k  ,iens)) +
                                            -1./12.*std::log(hy_dens_cells(hs+k+1,iens)) );
          hy_theta_edges(k,iens) = -1./12.*hy_theta_cells(hs+k-2,iens) +
                                    7./12.*hy_theta_cells(hs+k-1,iens) +
                                    7./12.*hy_theta_cells(hs+k  ,iens) +
                                   -1./12.*hy_theta_cells(hs+k+1,iens);
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


