
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



    Dynamics_Euler_Stratified_WenoFV() { }



    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real constexpr maxwave = 350 + 40;
      real cfl = 2.00;
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
      for (int icycle = 0; icycle < ncycles; icycle++) { time_step_rk_s_3(coupler,state,tracers,dt_dyn); }
      convert_dynamics_to_coupler( coupler , state , tracers );
    }



    void enforce_immersed_boundaries( core::Coupler & coupler ,
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
      auto use_immersed_boundaries = coupler.get_option<bool>("use_immersed_boundaries");
      auto immersed_proportion     = coupler.get_data_manager_readonly().get<real const,4>("immersed_proportion");
      auto hy_dens_cells           = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_cells"      );
      auto hy_dens_theta_cells     = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_theta_cells");
      if (use_immersed_boundaries) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          real prop = immersed_proportion(k,j,i,iens);
          if (prop > 0) {
            real r_prop = 1._fp / std::max( prop , 1.e-6_fp );
            real mult_s = 1._fp / (1.e-6 + r_prop*r_prop*r_prop);
            real mult_w = 1._fp / (1.e-6 + r_prop*r_prop*r_prop);
            state(idR,hs+k,hs+j,hs+i,iens) = mult_s*hy_dens_cells      (k,iens) + (1-mult_s)*state(idR,hs+k,hs+j,hs+i,iens);
            state(idU,hs+k,hs+j,hs+i,iens) = mult_w*0                           + (1-mult_w)*state(idU,hs+k,hs+j,hs+i,iens);
            state(idV,hs+k,hs+j,hs+i,iens) = mult_w*0                           + (1-mult_w)*state(idV,hs+k,hs+j,hs+i,iens);
            state(idW,hs+k,hs+j,hs+i,iens) = mult_w*0                           + (1-mult_w)*state(idW,hs+k,hs+j,hs+i,iens);
            state(idT,hs+k,hs+j,hs+i,iens) = mult_w*hy_dens_theta_cells(k,iens) + (1-mult_w)*state(idT,hs+k,hs+j,hs+i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers(tr,hs+k,hs+j,hs+i,iens) = mult_w*0 + (1-mult_w)*tracers(tr,hs+k,hs+j,hs+i,iens);
            }
          }
        });
      }
    }



    // CFL 2.0
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
      auto tracer_positive = coupler.get_data_manager_readonly().get<bool const,1>("tracer_positive");
      int n = sqrt(s);
      int r = s-n;
      real5d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx,nens);
      real5d tracers_tend("tracers_tend",num_tracers,nz,ny,nx,nens);
      int nstages = 0;

      enforce_immersed_boundaries(coupler,state,tracers,dt_dyn);

      //////////////
      // Part 1
      //////////////
      for (int istage = 1; istage <= ((n-1)*(n-2))/2; istage++) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int tr=0; tr < num_tracers; tr++) { tracers_tend(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens); }
        });
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/r);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/r;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/r;
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
        nstages++;
      }
      real5d state_tmp   = state  .createDeviceCopy();
      real5d tracers_tmp = tracers.createDeviceCopy();
      ///////////////////
      // Part 2
      ///////////////////
      for (int istage = ((n-1)*(n-2))/2+1; istage <= (n*(n+1))/2-1; istage++) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int tr=0; tr < num_tracers; tr++) { tracers_tend(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens); }
        });
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/r);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/r;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/r;
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
        nstages++;
      }
      ///////////////////
      // Intermission
      ///////////////////
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        for (int tr=0; tr < num_tracers; tr++) {
          tracers_tend(tr,k,j,i,iens) =  n   /(2*n-1.)*tracers_tmp(tr,hs+k,hs+j,hs+i,iens) +
                                        (n-1)/(2*n-1.)*tracers    (tr,hs+k,hs+j,hs+i,iens);
        }
      });
      compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,(n-1)/(2*n-1.)*dt_dyn/r);
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
          if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
        }
      });
      nstages++;
      //////////////
      // Part 3
      //////////////
      for (int istage = (n*(n+1))/2+1; istage <= s; istage++) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int tr=0; tr < num_tracers; tr++) { tracers_tend(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens); }
        });
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/r);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/r;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/r;
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
        nstages++;
      }

      if (nstages != s) yakl::yakl_throw("ERROR: Incorrect implementation");

      enforce_immersed_boundaries(coupler,state,tracers,dt_dyn);
    }



    // CFL 2.0
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

      enforce_immersed_boundaries(coupler,state,tracers,dt_dyn);

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
        real5d state_tmp   = state  .createDeviceCopy();
        real5d tracers_tmp = tracers.createDeviceCopy();
        real5d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx,nens);
        real5d tracers_tend("tracers_tend",num_tracers,nz,ny,nx,nens);
        //////////////
        // Stages 1-5
        //////////////
        for (int istage = 1; istage <= 5; istage++) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            for (int tr=0; tr < num_tracers; tr++) { tracers_tend(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens); }
          });
          compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/6);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            for (int l = 0; l < num_state  ; l++) {
              state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/6;
            }
            for (int l = 0; l < num_tracers; l++) {
              tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/6;
              if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
            }
          });
        }
        ///////////////////
        // Intermissions
        ///////////////////
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i,iens) = state_tmp  (l,hs+k,hs+j,hs+i,iens)/25 + 9*state    (l,hs+k,hs+j,hs+i,iens)/25;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i,iens) = tracers_tmp(l,hs+k,hs+j,hs+i,iens)/25 + 9*tracers  (l,hs+k,hs+j,hs+i,iens)/25;
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
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
        //////////////
        // Stages 6-9
        //////////////
        for (int istage = 6; istage <= 9; istage++) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            for (int tr=0; tr < num_tracers; tr++) { tracers_tend(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens); }
          });
          compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/6);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            for (int l = 0; l < num_state  ; l++) {
              state  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn*state_tend  (l,k,j,i,iens)/6;
            }
            for (int l = 0; l < num_tracers; l++) {
              tracers(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn*tracers_tend(l,k,j,i,iens)/6;
              if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
            }
          });
        }
        //////////////
        // Stage 10
        //////////////
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_tend(tr,k,j,i,iens) = tracers_tmp(tr,hs+k,hs+j,hs+i,iens) + 3*tracers(tr,hs+k,hs+j,hs+i,iens)/5;
          }
        });
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn/10);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state  (l,hs+k,hs+j,hs+i,iens) = state_tmp  (l,hs+k,hs+j,hs+i,iens) + 3*state    (l,hs+k,hs+j,hs+i,iens)/5 +
                                             dt_dyn*state_tend  (l,k,j,i,iens)/10;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers(l,hs+k,hs+j,hs+i,iens) = tracers_tmp(l,hs+k,hs+j,hs+i,iens) + 3*tracers  (l,hs+k,hs+j,hs+i,iens)/5 +
                                             dt_dyn*tracers_tend(l,k,j,i,iens)/10;
            if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i,iens) = std::max(0._fp,tracers(l,hs+k,hs+j,hs+i,iens));
          }
        });
      }

      enforce_immersed_boundaries(coupler,state,tracers,dt_dyn);
    }



    // CFL 0.45
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

      enforce_immersed_boundaries(coupler,state,tracers,dt_dyn);

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
        real5d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     ,nens);
        real5d tracers_tmp ("tracers_tmp" ,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);
        real5d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     ,nens);
        //////////////
        // Stage 1
        //////////////
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_tend(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens);
          }
        });
        compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
        // Apply tendencies
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l = 0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i,iens) = state  (l,hs+k,hs+j,hs+i,iens) + dt_dyn * state_tend  (l,k,j,i,iens);
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i,iens) = tracers(l,hs+k,hs+j,hs+i,iens) + dt_dyn * tracers_tend(l,k,j,i,iens);
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers_tmp(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i,iens) );
            }
          }
        });
        //////////////
        // Stage 2
        //////////////
        // enforce_immersed_boundaries(coupler,state_tmp,tracers_tmp,dt_dyn);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_tend(tr,k,j,i,iens) = (3._fp/4._fp) * tracers    (tr,hs+k,hs+j,hs+i,iens) + 
                                          (1._fp/4._fp) * tracers_tmp(tr,hs+k,hs+j,hs+i,iens);
          }
        });
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
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers_tmp(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i,iens) );
            }
          }
        });
        //////////////
        // Stage 3
        //////////////
        // enforce_immersed_boundaries(coupler,state_tmp,tracers_tmp,dt_dyn);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_tend(tr,k,j,i,iens) = (1._fp/3._fp) * tracers    (tr,hs+k,hs+j,hs+i,iens) + 
                                          (2._fp/3._fp) * tracers_tmp(tr,hs+k,hs+j,hs+i,iens);
          }
        });
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
            // For machine precision negative values after FCT-enforced positivity application
            if (tracer_positive(l)) {
              tracers(l,hs+k,hs+j,hs+i,iens) = std::max( 0._fp , tracers(l,hs+k,hs+j,hs+i,iens) );
            }
          }
        });
      }

      enforce_immersed_boundaries(coupler,state,tracers,dt_dyn);
    }



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

      auto use_immersed_boundaries = coupler.get_option<bool>("use_immersed_boundaries");
      auto nens                    = coupler.get_nens();
      auto nx                      = coupler.get_nx();
      auto ny                      = coupler.get_ny();
      auto nz                      = coupler.get_nz();
      auto dx                      = coupler.get_dx();
      auto dy                      = coupler.get_dy();
      auto dz                      = coupler.get_dz();
      auto sim2d                   = coupler.is_sim2d();
      auto C0                      = coupler.get_option<real>("C0"     );
      auto grav                    = coupler.get_option<real>("grav"   );
      auto gamma                   = coupler.get_option<real>("gamma_d");
      auto save_pressure_z         = coupler.get_option<bool>("save_pressure_z",false);
      auto enable_gravity          = coupler.get_option<bool>("enable_gravity");
      auto nu                      = coupler.get_option<real>("dns_nu" );
      auto num_tracers             = coupler.get_num_tracers();
      auto tracer_positive         = coupler.get_data_manager_readonly().get<bool const,1>("tracer_positive"    );
      auto immersed_proportion     = coupler.get_data_manager_readonly().get<real const,4>("immersed_proportion");
      auto pressure_mult           = coupler.get_data_manager_readonly().get<real const,2>("pressure_mult");
      SArray<real,2,ord,2> coefs_to_gll;
      TransformMatrices::coefs_to_gll_lower(coefs_to_gll);
      real r_dx = 1./dx;
      real r_dy = 1./dy;
      real r_dz = 1./dz;
      real4d pressure_z;
      if (save_pressure_z) pressure_z = coupler.get_data_manager_readwrite().get<real,4>("pressure_z");
      real Pr = 0.71;

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rdens = 1._fp / state(idR,hs+k,hs+j,hs+i,iens);
        state(idU,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idV,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idW,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idT,hs+k,hs+j,hs+i,iens) *= rdens;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) *= rdens; }
      });

      if (ord > 1) halo_exchange( coupler , state , tracers );

      real6d state_limits_x  ("state_limits_x"  ,2,num_state  ,nz,ny,nx+1,nens);
      real6d tracers_limits_x("tracers_limits_x",2,num_tracers,nz,ny,nx+1,nens);
      real6d state_limits_y  ("state_limits_y"  ,2,num_state  ,nz,ny+1,nx,nens);
      real6d tracers_limits_y("tracers_limits_y",2,num_tracers,nz,ny+1,nx,nens);
      real6d state_limits_z  ("state_limits_z"  ,2,num_state  ,nz+1,ny,nx,nens);
      real6d tracers_limits_z("tracers_limits_z",2,num_tracers,nz+1,ny,nx,nens);

      limiter::WenoLimiter<ord> limiter(0.1,1,2,1,1.e3);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil;
        SArray<real,1,2  > gll;
        for (int l=0; l < num_state; l++) {
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(l,hs+k,hs+j,i+ii,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          state_limits_x(1,l,k,j,i  ,iens) = gll(0);
          state_limits_x(0,l,k,j,i+1,iens) = gll(1);
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(l,hs+k,j+jj,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          state_limits_y(1,l,k,j  ,i,iens) = gll(0);
          state_limits_y(0,l,k,j+1,i,iens) = gll(1);
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(l,k+kk,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          state_limits_z(1,l,k  ,j,i,iens) = gll(0);
          state_limits_z(0,l,k+1,j,i,iens) = gll(1);
        }
        for (int l=0; l < num_tracers; l++) {
          for (int ii=0; ii < ord; ii++) { stencil(ii) = tracers(l,hs+k,hs+j,i+ii,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          tracers_limits_x(1,l,k,j,i  ,iens) = gll(0);
          tracers_limits_x(0,l,k,j,i+1,iens) = gll(1);
          for (int jj=0; jj < ord; jj++) { stencil(jj) = tracers(l,hs+k,j+jj,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          tracers_limits_y(1,l,k,j  ,i,iens) = gll(0);
          tracers_limits_y(0,l,k,j+1,i,iens) = gll(1);
          for (int kk=0; kk < ord; kk++) { stencil(kk) = tracers(l,k+kk,hs+j,hs+i,iens); }
          reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
          tracers_limits_z(1,l,k  ,j,i,iens) = gll(0);
          tracers_limits_z(0,l,k+1,j,i,iens) = gll(1);
        }
      });
      
      edge_exchange( coupler , state_limits_x , tracers_limits_x ,
                               state_limits_y , tracers_limits_y ,
                               state_limits_z , tracers_limits_z );

      using yakl::COLON;
      auto state_flux_x   = state_limits_x  .slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_flux_x = tracers_limits_x.slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_mult_x = tracers_limits_x.slice<5>(1,COLON,COLON,COLON,COLON,COLON);
      auto state_flux_y   = state_limits_y  .slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_flux_y = tracers_limits_y.slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_mult_y = tracers_limits_y.slice<5>(1,COLON,COLON,COLON,COLON,COLON);
      auto state_flux_z   = state_limits_z  .slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_flux_z = tracers_limits_z.slice<5>(0,COLON,COLON,COLON,COLON,COLON);
      auto tracers_mult_z = tracers_limits_z.slice<5>(1,COLON,COLON,COLON,COLON,COLON);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz+1,ny+1,nx+1,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real constexpr cs   = 350.;
        real constexpr r_cs = 1./cs;
        if (j < ny && k < nz) {
          // Acoustically upwind mass flux and pressure
          real r_L = state_limits_x(0,idR,k,j,i,iens);   real r_R = state_limits_x(1,idR,k,j,i,iens);
          real u_L = state_limits_x(0,idU,k,j,i,iens);   real u_R = state_limits_x(1,idU,k,j,i,iens);
          real t_L = state_limits_x(0,idT,k,j,i,iens);   real t_R = state_limits_x(1,idT,k,j,i,iens);
          real p_L = C0*std::pow(r_L*t_L,gamma)      ;   real p_R = C0*std::pow(r_R*t_R,gamma)      ;
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
            tracers_mult_x(tr,k,j,i,iens) = 1;
          }
          if (nu > 0) {
            real dr_dx = ( state(idR,hs+k,hs+j,hs+i,iens) - state(idR,hs+k,hs+j,hs+i-1,iens) ) / dx;
            real du_dx = ( state(idU,hs+k,hs+j,hs+i,iens) - state(idU,hs+k,hs+j,hs+i-1,iens) ) / dx;
            real dv_dx = ( state(idV,hs+k,hs+j,hs+i,iens) - state(idV,hs+k,hs+j,hs+i-1,iens) ) / dx;
            real dw_dx = ( state(idW,hs+k,hs+j,hs+i,iens) - state(idW,hs+k,hs+j,hs+i-1,iens) ) / dx;
            real dt_dx = ( state(idT,hs+k,hs+j,hs+i,iens) - state(idT,hs+k,hs+j,hs+i-1,iens) ) / dx;
            real du_dy = 0.5_fp*((state(idU,hs+k,hs+j+1,hs+i-1,iens)-state(idU,hs+k,hs+j-1,hs+i-1,iens))/(2*dy) +
                                 (state(idU,hs+k,hs+j+1,hs+i  ,iens)-state(idU,hs+k,hs+j-1,hs+i  ,iens))/(2*dy) );
            real dv_dy = 0.5_fp*((state(idV,hs+k,hs+j+1,hs+i-1,iens)-state(idV,hs+k,hs+j-1,hs+i-1,iens))/(2*dy) +
                                 (state(idV,hs+k,hs+j+1,hs+i  ,iens)-state(idV,hs+k,hs+j-1,hs+i  ,iens))/(2*dy) );
            real du_dz = 0.5_fp*((state(idU,hs+k+1,hs+j,hs+i-1,iens)-state(idU,hs+k-1,hs+j,hs+i-1,iens))/(2*dz) +
                                 (state(idU,hs+k+1,hs+j,hs+i  ,iens)-state(idU,hs+k-1,hs+j,hs+i  ,iens))/(2*dz) );
            real dw_dz = 0.5_fp*((state(idW,hs+k+1,hs+j,hs+i-1,iens)-state(idW,hs+k-1,hs+j,hs+i-1,iens))/(2*dz) +
                                 (state(idW,hs+k+1,hs+j,hs+i  ,iens)-state(idW,hs+k-1,hs+j,hs+i  ,iens))/(2*dz) );
            real r = 0.5_fp * (r_L+r_R);
            // state_flux_x(idR,k,j,i,iens) -= r*nu/Pr*(dr_dx                                              );
            state_flux_x(idU,k,j,i,iens) -= r*nu   *(du_dx + du_dx - 2._fp/3._fp*(du_dx + dv_dy + dw_dz));
            state_flux_x(idV,k,j,i,iens) -= r*nu   *(dv_dx + du_dy                                      );
            state_flux_x(idW,k,j,i,iens) -= r*nu   *(dw_dx + du_dz                                      );
            state_flux_x(idT,k,j,i,iens) -= r*nu/Pr*(dt_dx                                              );
            for (int tr=0; tr < num_tracers; tr++) {
              real dt_dx = ( tracers(tr,hs+k,hs+j,hs+i,iens) - tracers(tr,hs+k,hs+j,hs+i-1,iens) ) / dx;
              tracers_flux_x(tr,k,j,i,iens) -= r*nu/Pr*dt_dx;
            }
          }
        }
        if (i < nx && k < nz && !sim2d) {
          // Acoustically upwind mass flux and pressure
          real r_L = state_limits_y(0,idR,k,j,i,iens);   real r_R = state_limits_y(1,idR,k,j,i,iens);
          real v_L = state_limits_y(0,idV,k,j,i,iens);   real v_R = state_limits_y(1,idV,k,j,i,iens);
          real t_L = state_limits_y(0,idT,k,j,i,iens);   real t_R = state_limits_y(1,idT,k,j,i,iens);
          real p_L = C0*std::pow(r_L*t_L,gamma)      ;   real p_R = C0*std::pow(r_R*t_R,gamma)      ;
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
            tracers_mult_y(tr,k,j,i,iens) = 1;
          }
          if (nu > 0) {
            real dr_dy = ( state(idR,hs+k,hs+j,hs+i,iens) - state(idR,hs+k,hs+j-1,hs+i,iens) ) / dy;
            real du_dy = ( state(idU,hs+k,hs+j,hs+i,iens) - state(idU,hs+k,hs+j-1,hs+i,iens) ) / dy;
            real dv_dy = ( state(idV,hs+k,hs+j,hs+i,iens) - state(idV,hs+k,hs+j-1,hs+i,iens) ) / dy;
            real dw_dy = ( state(idW,hs+k,hs+j,hs+i,iens) - state(idW,hs+k,hs+j-1,hs+i,iens) ) / dy;
            real dt_dy = ( state(idT,hs+k,hs+j,hs+i,iens) - state(idT,hs+k,hs+j-1,hs+i,iens) ) / dy;
            real du_dx = 0.5_fp*((state(idU,hs+k,hs+j-1,hs+i+1,iens)-state(idU,hs+k,hs+j-1,hs+i-1,iens))/(2*dx) +
                                 (state(idU,hs+k,hs+j  ,hs+i+1,iens)-state(idU,hs+k,hs+j  ,hs+i-1,iens))/(2*dx) );
            real dv_dx = 0.5_fp*((state(idV,hs+k,hs+j-1,hs+i+1,iens)-state(idV,hs+k,hs+j-1,hs+i-1,iens))/(2*dx) +
                                 (state(idV,hs+k,hs+j  ,hs+i+1,iens)-state(idV,hs+k,hs+j  ,hs+i-1,iens))/(2*dx) );
            real dv_dz = 0.5_fp*((state(idV,hs+k+1,hs+j-1,hs+i,iens)-state(idV,hs+k-1,hs+j-1,hs+i,iens))/(2*dz) +
                                 (state(idV,hs+k+1,hs+j  ,hs+i,iens)-state(idV,hs+k-1,hs+j  ,hs+i,iens))/(2*dz) );
            real dw_dz = 0.5_fp*((state(idW,hs+k+1,hs+j-1,hs+i,iens)-state(idW,hs+k-1,hs+j-1,hs+i,iens))/(2*dz) +
                                 (state(idW,hs+k+1,hs+j  ,hs+i,iens)-state(idW,hs+k-1,hs+j  ,hs+i,iens))/(2*dz) );
            real r = 0.5_fp * (r_L+r_R);
            // state_flux_y(idR,k,j,i,iens) -= r*nu/Pr*(dr_dy                                              );
            state_flux_y(idU,k,j,i,iens) -= r*nu   *(du_dy + dv_dx                                      );
            state_flux_y(idV,k,j,i,iens) -= r*nu   *(dv_dy + dv_dy - 2._fp/3._fp*(du_dx + dv_dy + dw_dz));
            state_flux_y(idW,k,j,i,iens) -= r*nu   *(dw_dy + dv_dz                                      );
            state_flux_y(idT,k,j,i,iens) -= r*nu/Pr*(dt_dy                                              );
            for (int tr=0; tr < num_tracers; tr++) {
              real dt_dy = ( tracers(tr,hs+k,hs+j,hs+i,iens) - tracers(tr,hs+k,hs+j-1,hs+i,iens) ) / dy;
              tracers_flux_y(tr,k,j,i,iens) -= r*nu/Pr*dt_dy;
            }
          }
        }
        if (i < nx && j < ny) {
          // Acoustically upwind mass flux and pressure
          real r_L = state_limits_z(0,idR,k,j,i,iens);   real r_R = state_limits_z(1,idR,k,j,i,iens);
          real w_L = state_limits_z(0,idW,k,j,i,iens);   real w_R = state_limits_z(1,idW,k,j,i,iens);
          real t_L = state_limits_z(0,idT,k,j,i,iens);   real t_R = state_limits_z(1,idT,k,j,i,iens);
          real p_L = C0*std::pow(r_L*t_L,gamma)      ;   real p_R = C0*std::pow(r_R*t_R,gamma)      ;
          real w1 = 0.5_fp * (p_R-cs*r_R*w_R);
          real w2 = 0.5_fp * (p_L+cs*r_L*w_L);
          real p_upw  = w1 + w2;
          real rw_upw = (w2-w1)*r_cs;
          p_upw *= pressure_mult(k,iens);
          // Advectively upwind everything else
          int ind = rw_upw > 0 ? 0 : 1;
          state_flux_z(idR,k,j,i,iens) = rw_upw;
          state_flux_z(idU,k,j,i,iens) = rw_upw*state_limits_z(ind,idU,k,j,i,iens);
          state_flux_z(idV,k,j,i,iens) = rw_upw*state_limits_z(ind,idV,k,j,i,iens);
          state_flux_z(idW,k,j,i,iens) = rw_upw*state_limits_z(ind,idW,k,j,i,iens) + p_upw;
          state_flux_z(idT,k,j,i,iens) = rw_upw*state_limits_z(ind,idT,k,j,i,iens);
          if (save_pressure_z) pressure_z(k,j,i,iens) = p_upw;
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_z(tr,k,j,i,iens) = rw_upw*tracers_limits_z(ind,tr,k,j,i,iens);
            tracers_mult_z(tr,k,j,i,iens) = 1;
          }
          if (nu > 0) {
            real dr_dz = ( state(idR,hs+k,hs+j,hs+i,iens) - state(idR,hs+k-1,hs+j,hs+i,iens) ) / dz;
            real du_dz = ( state(idU,hs+k,hs+j,hs+i,iens) - state(idU,hs+k-1,hs+j,hs+i,iens) ) / dz;
            real dv_dz = ( state(idV,hs+k,hs+j,hs+i,iens) - state(idV,hs+k-1,hs+j,hs+i,iens) ) / dz;
            real dw_dz = ( state(idW,hs+k,hs+j,hs+i,iens) - state(idW,hs+k-1,hs+j,hs+i,iens) ) / dz;
            real dt_dz = ( state(idT,hs+k,hs+j,hs+i,iens) - state(idT,hs+k-1,hs+j,hs+i,iens) ) / dz;
            real du_dx = 0.5_fp*((state(idU,hs+k-1,hs+j,hs+i+1,iens)-state(idU,hs+k-1,hs+j,hs+i-1,iens))/(2*dx) +
                                 (state(idU,hs+k  ,hs+j,hs+i+1,iens)-state(idU,hs+k  ,hs+j,hs+i-1,iens))/(2*dx) );
            real dw_dx = 0.5_fp*((state(idW,hs+k-1,hs+j,hs+i+1,iens)-state(idW,hs+k-1,hs+j,hs+i-1,iens))/(2*dx) +
                                 (state(idW,hs+k  ,hs+j,hs+i+1,iens)-state(idW,hs+k  ,hs+j,hs+i-1,iens))/(2*dx) );
            real dv_dy = 0.5_fp*((state(idV,hs+k-1,hs+j+1,hs+i,iens)-state(idV,hs+k-1,hs+j-1,hs+i,iens))/(2*dy) +
                                 (state(idV,hs+k  ,hs+j+1,hs+i,iens)-state(idV,hs+k  ,hs+j-1,hs+i,iens))/(2*dy) );
            real dw_dy = 0.5_fp*((state(idW,hs+k-1,hs+j+1,hs+i,iens)-state(idW,hs+k-1,hs+j-1,hs+i,iens))/(2*dy) +
                                 (state(idW,hs+k  ,hs+j+1,hs+i,iens)-state(idW,hs+k  ,hs+j-1,hs+i,iens))/(2*dy) );
            real r = 0.5_fp * (r_L+r_R);
            // state_flux_z(idR,k,j,i,iens) -= r*nu/Pr*(dr_dz                                              );
            state_flux_z(idU,k,j,i,iens) -= r*nu   *(du_dz + dw_dx                                      );
            state_flux_z(idV,k,j,i,iens) -= r*nu   *(dv_dz + dw_dy                                      );
            state_flux_z(idW,k,j,i,iens) -= r*nu   *(dw_dz + dw_dz - 2._fp/3._fp*(du_dx + dv_dy + dw_dz));
            state_flux_z(idT,k,j,i,iens) -= r*nu/Pr*(dt_dz                                              );
            for (int tr=0; tr < num_tracers; tr++) {
              real dt_dz = ( tracers(tr,hs+k,hs+j,hs+i,iens) - tracers(tr,hs+k-1,hs+j,hs+i,iens) ) / dz;
              tracers_flux_z(tr,k,j,i,iens) -= r*nu/Pr*dt_dz;
            }
          }
        }
      });


      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real dens = state(idR,hs+k,hs+j,hs+i,iens);
        state(idU,hs+k,hs+j,hs+i,iens) *= dens;
        state(idV,hs+k,hs+j,hs+i,iens) *= dens;
        state(idW,hs+k,hs+j,hs+i,iens) *= dens;
        state(idT,hs+k,hs+j,hs+i,iens) *= dens;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) *= dens; }
      });

      // Flux Corrected Transport to enforce positivity for tracer species that must remain non-negative
      // This looks like it has a race condition, but it does not. Only one of the adjacent cells can ever change
      // a given edge flux because it's only changed if its sign oriented outward from a cell.
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
        if (tracer_positive(tr)) {
          real mass_available = max(tracers_tend(tr,k,j,i,iens),0._fp) * dx * dy * dz;
          real flux_out = (max(tracers_flux_x(tr,k,j,i+1,iens),0._fp)-min(tracers_flux_x(tr,k,j,i,iens),0._fp))*r_dx +
                          (max(tracers_flux_y(tr,k,j+1,i,iens),0._fp)-min(tracers_flux_y(tr,k,j,i,iens),0._fp))*r_dy +
                          (max(tracers_flux_z(tr,k+1,j,i,iens),0._fp)-min(tracers_flux_z(tr,k,j,i,iens),0._fp))*r_dz;
          real mass_out = (flux_out) * dt * dx * dy * dz;
          if (mass_out > mass_available) {
            real mult = mass_available / mass_out;
            if (tracers_flux_x(tr,k,j,i+1,iens) > 0) { tracers_mult_x(tr,k,j,i+1,iens) = mult; }
            if (tracers_flux_x(tr,k,j,i  ,iens) < 0) { tracers_mult_x(tr,k,j,i  ,iens) = mult; }
            if (tracers_flux_y(tr,k,j+1,i,iens) > 0) { tracers_mult_y(tr,k,j+1,i,iens) = mult; }
            if (tracers_flux_y(tr,k,j  ,i,iens) < 0) { tracers_mult_y(tr,k,j  ,i,iens) = mult; }
            if (tracers_flux_z(tr,k+1,j,i,iens) > 0) { tracers_mult_z(tr,k+1,j,i,iens) = mult; }
            if (tracers_flux_z(tr,k  ,j,i,iens) < 0) { tracers_mult_z(tr,k  ,j,i,iens) = mult; }
          }
        }
      });

      fct_mult_exchange( coupler , tracers_mult_x , tracers_mult_y , tracers_mult_z );

      // Compute tendencies as the flux divergence + gravity source term
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        for (int l = 0; l < num_state; l++) {
          state_tend(l,k,j,i,iens) = -( state_flux_x(l,k,j,i+1,iens) - state_flux_x(l,k,j,i,iens) ) * r_dx
                                     -( state_flux_y(l,k,j+1,i,iens) - state_flux_y(l,k,j,i,iens) ) * r_dy
                                     -( state_flux_z(l,k+1,j,i,iens) - state_flux_z(l,k,j,i,iens) ) * r_dz;
          if (l == idV && sim2d) state_tend(l,k,j,i,iens) = 0;
          if (enable_gravity && l == idW) state_tend(l,k,j,i,iens) += -grav*state(idR,hs+k,hs+j,hs+i,iens);
        }
        for (int l = 0; l < num_tracers; l++) {
          tracers_tend(l,k,j,i,iens) = -( tracers_flux_x(l,k,j,i+1,iens)*tracers_mult_x(l,k,j,i+1,iens) -
                                          tracers_flux_x(l,k,j,i  ,iens)*tracers_mult_x(l,k,j,i  ,iens) ) * r_dx
                                       -( tracers_flux_y(l,k,j+1,i,iens)*tracers_mult_y(l,k,j+1,i,iens) -
                                          tracers_flux_y(l,k,j  ,i,iens)*tracers_mult_y(l,k,j  ,i,iens) ) * r_dy 
                                       -( tracers_flux_z(l,k+1,j,i,iens)*tracers_mult_z(l,k+1,j,i,iens) -
                                          tracers_flux_z(l,k  ,j,i,iens)*tracers_mult_z(l,k  ,j,i,iens) ) * r_dz;
        }
      });
    }



    // ord stencil cell averages to two GLL point values via high-order reconstruction and WENO limiting
    template <class LIMITER>
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>     const & stencil      ,
                                                    SArray<real,1,2  >           & gll          ,
                                                    SArray<real,2,ord,2>   const & coefs_to_gll ,
                                                    LIMITER                const & limiter      ) {
      SArray<real,1,ord> wenoCoefs;
      limiter.compute_limited_coefs( stencil , wenoCoefs );
      for (int ii=0; ii<2; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) {
          tmp += coefs_to_gll(s,ii) * wenoCoefs(s);
        }
        gll(ii) = tmp;
      }
    }



    void get_BCs( core::Coupler const &coupler , int &bc_x , int &bc_y , int &bc_z ) const {
      auto bc_x_str = coupler.get_option<std::string>("bc_x");
      auto bc_y_str = coupler.get_option<std::string>("bc_y");
      auto bc_z_str = coupler.get_option<std::string>("bc_z");
      if (bc_x_str == "periodic") bc_x = BC_PERIODIC;
      if (bc_y_str == "periodic") bc_y = BC_PERIODIC;
      if (bc_z_str == "periodic") bc_z = BC_PERIODIC;
      if (bc_x_str == "wall") bc_x = BC_WALL;
      if (bc_y_str == "wall") bc_y = BC_WALL;
      if (bc_z_str == "wall") bc_z = BC_WALL;
      if (bc_x_str == "open") bc_x = BC_OPEN;
      if (bc_y_str == "open") bc_y = BC_OPEN;
      if (bc_z_str == "open") bc_z = BC_OPEN;
    }



    void halo_exchange( core::Coupler const & coupler  ,
                        real5d        const & state    ,
                        real5d        const & tracers  ) const {
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
      auto &neigh         = coupler.get_neighbor_rankid_matrix();
      auto dtype          = coupler.get_mpi_data_type();
      auto enable_gravity = coupler.get_option<bool>("enable_gravity");
      auto grav           = coupler.get_option<real>("grav");
      auto gamma          = coupler.get_option<real>("gamma_d");
      auto C0             = coupler.get_option<real>("C0");
      int bc_x , bc_y , bc_z;
      get_BCs( coupler , bc_x , bc_y , bc_z );
      MPI_Request sReq [2];
      MPI_Request rReq [2];
      MPI_Status  sStat[2];
      MPI_Status  rStat[2];
      auto comm = MPI_COMM_WORLD;
      int npack = num_state + num_tracers;

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
          }
        });
      }

      // y-direction exchanges
      {
        real5d halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx,nens);
        real5d halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx,nens);
        real5d halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx,nens);
        real5d halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,hs,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          if        (v < num_state) {
            halo_send_buf_S(v,k,jj,i,iens) = state  (v          ,hs+k,hs+jj,hs+i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = state  (v          ,hs+k,ny+jj,hs+i,iens);
          } else if (v < num_state + num_tracers) {
            halo_send_buf_S(v,k,jj,i,iens) = tracers(v-num_state,hs+k,hs+jj,hs+i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = tracers(v-num_state,hs+k,ny+jj,hs+i,iens);
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
          realHost5d halo_send_buf_S_host("halo_send_buf_S_host",npack,nz,hs,nx,nens);
          realHost5d halo_send_buf_N_host("halo_send_buf_N_host",npack,nz,hs,nx,nens);
          realHost5d halo_recv_buf_S_host("halo_recv_buf_S_host",npack,nz,hs,nx,nens);
          realHost5d halo_recv_buf_N_host("halo_recv_buf_N_host",npack,nz,hs,nx,nens);
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
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,hs,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          if        (v < num_state) {
            state  (v          ,hs+k,      jj,hs+i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            state  (v          ,hs+k,ny+hs+jj,hs+i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          } else if (v < num_state + num_tracers) {
            tracers(v-num_state,hs+k,      jj,hs+i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            tracers(v-num_state,hs+k,ny+hs+jj,hs+i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          }
        });
      }

      // x-direction non-periodic BC's
      if (bc_x == BC_WALL || bc_x == BC_OPEN) {
        if (px == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,hs,nens) ,
                                            YAKL_LAMBDA (int k, int j, int ii, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state(l,hs+k,hs+j,ii,iens) = 0; }
              else                             { state(l,hs+k,hs+j,ii,iens) = state(l,hs+k,hs+j,hs+0,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,ii,iens) = tracers(l,hs+k,hs+j,hs+0,iens); }
          });
        }
        if (px == nproc_x-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,hs,nens) ,
                                            YAKL_LAMBDA (int k, int j, int ii, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state(l,hs+k,hs+j,hs+nx+ii,iens) = 0; }
              else                             { state(l,hs+k,hs+j,hs+nx+ii,iens) = state(l,hs+k,hs+j,hs+nx-1,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+nx+ii,iens) = tracers(l,hs+k,hs+j,hs+nx-1,iens); }
          });
        }
      }

      //y-direction non-periodic BC's
      if (bc_y == BC_WALL || bc_y == BC_OPEN) {
        if (py == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,hs,nx,nens) ,
                                            YAKL_LAMBDA (int k, int jj, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state(l,hs+k,jj,hs+i,iens) = 0; }
              else                             { state(l,hs+k,jj,hs+i,iens) = state(l,hs+k,hs+0,hs+i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,jj,hs+i,iens) = tracers(l,hs+k,hs+0,hs+i,iens); }
          });
        }
        if (py == nproc_y-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,hs,nx,nens) ,
                                            YAKL_LAMBDA (int k, int jj, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state(l,hs+k,hs+ny+jj,hs+i,iens) = 0; }
              else                             { state(l,hs+k,hs+ny+jj,hs+i,iens) = state(l,hs+k,hs+ny-1,hs+i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+ny+jj,hs+i,iens) = tracers(l,hs+k,hs+ny-1,hs+i,iens); }
          });
        }
      }

      // z-direction BC's
      if (bc_z == BC_PERIODIC) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(hs,ny,nx,nens) ,
                                          YAKL_LAMBDA (int kk, int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            state(l,      kk,hs+j,hs+i,iens) = state(l,      kk+nz,hs+j,hs+i,iens);
            state(l,hs+nz+kk,hs+j,hs+i,iens) = state(l,hs+nz+kk-nz,hs+j,hs+i,iens);
          }
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,hs+j,hs+i,iens) = tracers(l,      kk+nz,hs+j,hs+i,iens);
            tracers(l,hs+nz+kk,hs+j,hs+i,iens) = tracers(l,hs+nz+kk-nz,hs+j,hs+i,iens);
          }
        });
      } else if (bc_z == BC_WALL || bc_z == BC_OPEN) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(hs,ny,nx,nens) ,
                                          YAKL_LAMBDA (int kk, int j, int i, int iens) {
          // Lower bound of 1 below is on purpose to not write density BC's here. Those are done further down
          for (int l=1; l < num_state; l++) {
            if ((l==idW || l==idV || l==idU) && bc_z == BC_WALL) {
              state(l,      kk,hs+j,hs+i,iens) = 0;
              state(l,hs+nz+kk,hs+j,hs+i,iens) = 0;
            } else {
              state(l,      kk,hs+j,hs+i,iens) = state(l,hs+0   ,hs+j,hs+i,iens);
              state(l,hs+nz+kk,hs+j,hs+i,iens) = state(l,hs+nz-1,hs+j,hs+i,iens);
            }
          }
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,hs+j,hs+i,iens) = tracers(l,hs+0   ,hs+j,hs+i,iens);
            tracers(l,hs+nz+kk,hs+j,hs+i,iens) = tracers(l,hs+nz-1,hs+j,hs+i,iens);
          }
          if (enable_gravity) {
            {
              int  k0       = hs;
              int  k        = k0-1-kk;
              real rho0     = state(idR,k0,hs+j,hs+i,iens);
              real theta0   = state(idT,k0,hs+j,hs+i,iens);
              real rho0_gm1 = std::pow(rho0  ,gamma-1);
              real theta0_g = std::pow(theta0,gamma  );
              state(idR,k,hs+j,hs+i,iens) = std::pow( rho0_gm1 + grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) ,
                                                      1._fp/(gamma-1) );
            }
            {
              int  k0       = hs+nz-1;
              int  k        = k0+1+kk;
              real rho0     = state(idR,k0,hs+j,hs+i,iens);
              real theta0   = state(idT,k0,hs+j,hs+i,iens);
              real rho0_gm1 = std::pow(rho0  ,gamma-1);
              real theta0_g = std::pow(theta0,gamma  );
              state(idR,k,hs+j,hs+i,iens) = std::pow( rho0_gm1 - grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) ,
                                                      1._fp/(gamma-1) );
            }
          } else {
            state(idR,      kk,hs+j,hs+i,iens) = state(idR,hs+0   ,hs+j,hs+i,iens);
            state(idR,hs+nz+kk,hs+j,hs+i,iens) = state(idR,hs+nz-1,hs+j,hs+i,iens);
          }
        });
      }
    }



    void edge_exchange( core::Coupler const & coupler           ,
                        real6d        const & state_limits_x    ,
                        real6d        const & tracers_limits_x  ,
                        real6d        const & state_limits_y    ,
                        real6d        const & tracers_limits_y  ,
                        real6d        const & state_limits_z    ,
                        real6d        const & tracers_limits_z  ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();
      auto &neigh      = coupler.get_neighbor_rankid_matrix();
      auto dtype       = coupler.get_mpi_data_type();
      auto comm        = MPI_COMM_WORLD;
      int bc_x , bc_y , bc_z;
      get_BCs( coupler , bc_x , bc_y , bc_z );
      int npack = num_state + num_tracers;
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
          }
        });
      }

      // x-direction non-periodic BC's
      if (bc_x == BC_WALL || bc_x == BC_OPEN) {
        if (px == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nens) ,
                                            YAKL_LAMBDA (int k, int j, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state_limits_x(0,l,k,j,0,iens) = 0; state_limits_x(1,l,k,j,0,iens) = 0; }
              else                             { state_limits_x(0,l,k,j,0,iens) = state_limits_x(1,l,k,j,0,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_x(0,l,k,j,0,iens) = tracers_limits_x(1,l,k,j,0,iens); }
          });
        } else if (px == nproc_x-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nens) ,
                                            YAKL_LAMBDA (int k, int j, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state_limits_x(0,l,k,j,nx,iens) = 0; state_limits_x(1,l,k,j,nx,iens) = 0; }
              else                             { state_limits_x(1,l,k,j,nx,iens) = state_limits_x(0,l,k,j,nx,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_x(1,l,k,j,nx,iens) = tracers_limits_x(0,l,k,j,nx,iens); }
          });
        }
      }

      // y-direction non-periodic BC's
      if (bc_y == BC_WALL || bc_y == BC_OPEN) {
        if (py == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,nx,nens) ,
                                            YAKL_LAMBDA (int k, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state_limits_y(0,l,k,0,i,iens) = 0; state_limits_y(1,l,k,0,i,iens) = 0; }
              else                             { state_limits_y(0,l,k,0,i,iens) = state_limits_y(1,l,k,0,i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_y(0,l,k,0,i,iens) = tracers_limits_y(1,l,k,0,i,iens); }
          });
        } else if (py == nproc_y-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,nx,nens) ,
                                            YAKL_LAMBDA (int k, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state_limits_y(0,l,k,ny,i,iens) = 0; state_limits_y(1,l,k,ny,i,iens) = 0; }
              else                             { state_limits_y(1,l,k,ny,i,iens) = state_limits_y(0,l,k,ny,i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_y(1,l,k,ny,i,iens) = tracers_limits_y(0,l,k,ny,i,iens); }
          });
        }
      }

      // z-direction BC's
      if (bc_z == BC_PERIODIC) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(ny,nx,nens) ,
                                          YAKL_LAMBDA (int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            state_limits_z(0,l,0 ,j,i,iens) = state_limits_z(0,l,nz,j,i,iens);
            state_limits_z(1,l,nz,j,i,iens) = state_limits_z(1,l,0 ,j,i,iens);
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(0,l,0 ,j,i,iens) = tracers_limits_z(0,l,nz,j,i,iens);
            tracers_limits_z(1,l,nz,j,i,iens) = tracers_limits_z(1,l,0 ,j,i,iens);
          }
        });
      } else if (bc_z == BC_WALL || bc_z == BC_OPEN) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(ny,nx,nens) ,
                                          YAKL_LAMBDA (int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            if ((l==idW || l==idV || l==idU) && bc_z == BC_WALL) {
              state_limits_z(0,l,0 ,j,i,iens) = 0;  state_limits_z(1,l,0 ,j,i,iens) = 0;
              state_limits_z(0,l,nz,j,i,iens) = 0;  state_limits_z(1,l,nz,j,i,iens) = 0;
            } else {
              state_limits_z(0,l,0 ,j,i,iens) = state_limits_z(1,l,0 ,j,i,iens);
              state_limits_z(1,l,nz,j,i,iens) = state_limits_z(0,l,nz,j,i,iens);
            }
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(0,l,0 ,j,i,iens) = tracers_limits_z(1,l,0 ,j,i,iens);
            tracers_limits_z(1,l,nz,j,i,iens) = tracers_limits_z(0,l,nz,j,i,iens);
          }
        });
      }
    }



    void fct_mult_exchange( core::Coupler const &coupler ,
                            real5d const &tracers_mult_x ,
                            real5d const &tracers_mult_y ,
                            real5d const &tracers_mult_z ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();
      auto &neigh      = coupler.get_neighbor_rankid_matrix();
      auto dtype       = coupler.get_mpi_data_type();
      int bc_x , bc_y , bc_z;
      get_BCs( coupler , bc_x , bc_y , bc_z );
      auto comm = MPI_COMM_WORLD;
      int npack = num_tracers;
      MPI_Request sReq [2];
      MPI_Request rReq [2];
      MPI_Status  sStat[2];
      MPI_Status  rStat[2];

      // x-direction exchange
      {
        real4d     edge_send_buf_W     ("edge_send_buf_W"     ,npack,nz,ny,nens);
        real4d     edge_send_buf_E     ("edge_send_buf_E"     ,npack,nz,ny,nens);
        real4d     edge_recv_buf_W     ("edge_recv_buf_W"     ,npack,nz,ny,nens);
        real4d     edge_recv_buf_E     ("edge_recv_buf_E"     ,npack,nz,ny,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,nens) ,
                                          YAKL_LAMBDA (int v, int k, int j, int iens) {
          edge_send_buf_W(v,k,j,iens) = tracers_mult_x(v,k,j,0 ,iens);
          edge_send_buf_E(v,k,j,iens) = tracers_mult_x(v,k,j,nx,iens);
        });
        yakl::timer_start("edge_exchange_mpi");
        #ifdef MW_GPU_AWARE_MPI
          yakl::fence();
          MPI_Irecv( edge_recv_buf_W.data() , edge_recv_buf_W.size() , dtype , neigh(1,0) , 8 , comm , &rReq[0] );
          MPI_Irecv( edge_recv_buf_E.data() , edge_recv_buf_E.size() , dtype , neigh(1,2) , 9 , comm , &rReq[1] );
          MPI_Isend( edge_send_buf_W.data() , edge_send_buf_W.size() , dtype , neigh(1,0) , 9 , comm , &sReq[0] );
          MPI_Isend( edge_send_buf_E.data() , edge_send_buf_E.size() , dtype , neigh(1,2) , 8 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("edge_exchange_mpi");
        #else
          realHost4d edge_send_buf_W_host("edge_send_buf_W_host",npack,nz,ny,nens);
          realHost4d edge_send_buf_E_host("edge_send_buf_E_host",npack,nz,ny,nens);
          realHost4d edge_recv_buf_W_host("edge_recv_buf_W_host",npack,nz,ny,nens);
          realHost4d edge_recv_buf_E_host("edge_recv_buf_E_host",npack,nz,ny,nens);
          MPI_Irecv( edge_recv_buf_W_host.data() , edge_recv_buf_W_host.size() , dtype , neigh(1,0) , 8 , comm , &rReq[0] );
          MPI_Irecv( edge_recv_buf_E_host.data() , edge_recv_buf_E_host.size() , dtype , neigh(1,2) , 9 , comm , &rReq[1] );
          edge_send_buf_W.deep_copy_to(edge_send_buf_W_host);
          edge_send_buf_E.deep_copy_to(edge_send_buf_E_host);
          yakl::fence();
          MPI_Isend( edge_send_buf_W_host.data() , edge_send_buf_W_host.size() , dtype , neigh(1,0) , 9 , comm , &sReq[0] );
          MPI_Isend( edge_send_buf_E_host.data() , edge_send_buf_E_host.size() , dtype , neigh(1,2) , 8 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("edge_exchange_mpi");
          edge_recv_buf_W_host.deep_copy_to(edge_recv_buf_W);
          edge_recv_buf_E_host.deep_copy_to(edge_recv_buf_E);
        #endif
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,nens) ,
                                          YAKL_LAMBDA (int v, int k, int j, int iens) {
          tracers_mult_x(v,k,j,0 ,iens) = std::min( edge_recv_buf_W(v,k,j,iens) , tracers_mult_x(v,k,j,0 ,iens) );
          tracers_mult_x(v,k,j,nx,iens) = std::min( edge_recv_buf_E(v,k,j,iens) , tracers_mult_x(v,k,j,nx,iens) );
        });
      }

      // y-direction exchange
      {
        real4d     edge_send_buf_S     ("edge_send_buf_S"     ,npack,nz,nx,nens);
        real4d     edge_send_buf_N     ("edge_send_buf_N"     ,npack,nz,nx,nens);
        real4d     edge_recv_buf_S     ("edge_recv_buf_S"     ,npack,nz,nx,nens);
        real4d     edge_recv_buf_N     ("edge_recv_buf_N"     ,npack,nz,nx,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int i, int iens) {
          edge_send_buf_S(v,k,i,iens) = tracers_mult_y(v,k,0 ,i,iens);
          edge_send_buf_N(v,k,i,iens) = tracers_mult_y(v,k,ny,i,iens);
        });
        yakl::timer_start("edge_exchange_mpi");
        #ifdef MW_GPU_AWARE_MPI
          yakl::fence();
          MPI_Irecv( edge_recv_buf_S.data() , edge_recv_buf_S.size() , dtype , neigh(0,1) , 10 , comm , &rReq[0] );
          MPI_Irecv( edge_recv_buf_N.data() , edge_recv_buf_N.size() , dtype , neigh(2,1) , 11 , comm , &rReq[1] );
          MPI_Isend( edge_send_buf_S.data() , edge_send_buf_S.size() , dtype , neigh(0,1) , 11 , comm , &sReq[0] );
          MPI_Isend( edge_send_buf_N.data() , edge_send_buf_N.size() , dtype , neigh(2,1) , 10 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("edge_exchange_mpi");
        #else
          realHost4d edge_send_buf_S_host("edge_send_buf_S_host",npack,nz,nx,nens);
          realHost4d edge_send_buf_N_host("edge_send_buf_N_host",npack,nz,nx,nens);
          realHost4d edge_recv_buf_S_host("edge_recv_buf_S_host",npack,nz,nx,nens);
          realHost4d edge_recv_buf_N_host("edge_recv_buf_N_host",npack,nz,nx,nens);
          MPI_Irecv( edge_recv_buf_S_host.data() , edge_recv_buf_S_host.size() , dtype , neigh(0,1) , 10 , comm , &rReq[0] );
          MPI_Irecv( edge_recv_buf_N_host.data() , edge_recv_buf_N_host.size() , dtype , neigh(2,1) , 11 , comm , &rReq[1] );
          edge_send_buf_S.deep_copy_to(edge_send_buf_S_host);
          edge_send_buf_N.deep_copy_to(edge_send_buf_N_host);
          yakl::fence();
          MPI_Isend( edge_send_buf_S_host.data() , edge_send_buf_S_host.size() , dtype , neigh(0,1) , 11 , comm , &sReq[0] );
          MPI_Isend( edge_send_buf_N_host.data() , edge_send_buf_N_host.size() , dtype , neigh(2,1) , 10 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("edge_exchange_mpi");
          edge_recv_buf_S_host.deep_copy_to(edge_recv_buf_S);
          edge_recv_buf_N_host.deep_copy_to(edge_recv_buf_N);
        #endif
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int i, int iens) {
          tracers_mult_y(v,k,0 ,i,iens) = std::min( edge_recv_buf_S(v,k,i,iens) , tracers_mult_y(v,k,0 ,i,iens) );
          tracers_mult_y(v,k,ny,i,iens) = std::min( edge_recv_buf_N(v,k,i,iens) , tracers_mult_y(v,k,ny,i,iens) );
        });
      }

      // z-direwciton exchange
      if (bc_z == BC_PERIODIC) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(ny,nx,nens) ,
                                          YAKL_LAMBDA (int j, int i, int iens) {
          for (int l=0; l < num_tracers; l++) {
            real mn = std::min( tracers_mult_z(l,0 ,j,i,iens) , tracers_mult_z(l,nz,j,i,iens) );
            tracers_mult_z(l,0 ,j,i,iens) = mn;
            tracers_mult_z(l,nz,j,i,iens) = mn;
          }
        });
      }
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
      auto sim2d       = coupler.is_sim2d();
      auto num_tracers = coupler.get_num_tracers();
      auto grav        = coupler.get_option<real>("grav",9.81);

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

      real5d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real5d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      convert_coupler_to_dynamics( coupler , state , tracers );

      dm.register_and_allocate<real>("hy_dens_cells"      ,"",{nz  ,nens});
      dm.register_and_allocate<real>("hy_dens_theta_cells","",{nz  ,nens});
      dm.register_and_allocate<real>("pressure_mult"      ,"",{nz+1,nens});
      auto r             = dm.get<real,2>("hy_dens_cells"      );    r             = 0;
      auto rt            = dm.get<real,2>("hy_dens_theta_cells");    rt            = 0;
      auto pressure_mult = dm.get<real,2>("pressure_mult"      );    pressure_mult = 1;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,nens) , YAKL_LAMBDA (int k, int iens) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            r (k,iens) += state(idR,hs+k,hs+j,hs+i,iens);
            rt(k,iens) += state(idT,hs+k,hs+j,hs+i,iens);
          }
        }
      });
      auto r_loc  = r .createHostCopy();    auto r_glob  = r .createHostObject();
      auto rt_loc = rt.createHostCopy();    auto rt_glob = rt.createHostObject();
      auto dtype = coupler.get_mpi_data_type();
      MPI_Allreduce( r_loc .data() , r_glob .data() , r .size() , dtype , MPI_SUM , MPI_COMM_WORLD );
      MPI_Allreduce( rt_loc.data() , rt_glob.data() , rt.size() , dtype , MPI_SUM , MPI_COMM_WORLD );
      r_glob .deep_copy_to(r );
      rt_glob.deep_copy_to(rt);
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,nens) , YAKL_LAMBDA (int k, int iens) {
        r (k,iens) *= r_nx_ny;
        rt(k,iens) *= r_nx_ny;
      });

      if (coupler.get_option<bool>("enable_gravity",true)) {
        // Compute forcing due to pressure gradient only in the vertical direction
        coupler.set_option<bool>("save_pressure_z",true);
        dm.register_and_allocate<real>("pressure_z","",{nz+1,ny,nx,nens});
        auto pressure_z = dm.get<real,4>("pressure_z");
        real dt_dummy = 1.;
        real5d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx,nens);
        real5d tracers_tend("tracers_tend",num_tracers,nz,ny,nx,nens);
        compute_tendencies( coupler , state , state_tend , tracers , tracers_tend , dt_dummy );
        real3d vars_loc("vars_loc",2,nz+1,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz+1,nens) , YAKL_LAMBDA (int k, int iens) {
          real tot1 = 0;
          real tot2 = 0;
          for (int j=0; j < ny; j++) {
            for (int i=0; i < nx; i++) {
              tot1 += pressure_z(k,j,i,iens);
              if (k < nz) tot2 += state(idR,hs+k,hs+j,hs+i,iens);
            }
          }
          vars_loc(0,k,iens) = tot1;
          vars_loc(1,k,iens) = tot2;
        });
        auto vars_loc_host = vars_loc.createHostCopy();
        auto vars_host     = vars_loc.createHostCopy();
        MPI_Allreduce( vars_loc_host.data()        ,  // sendbuf
                       vars_host.data()            ,  // recvbuf
                       vars_loc_host.size()        ,  // count
                       coupler.get_mpi_data_type() ,  // type
                       MPI_SUM                     ,  // operation
                       MPI_COMM_WORLD              ); // communicator
        realHost2d pressure_mult_host("pressure_mult",nz+1,nens);
        real r_nx_ny = 1./(nx_glob*ny_glob);
        for (int iens=0; iens < nens; iens++) { pressure_mult_host(0,iens) = 1; }
        for (int k=1; k < nz+1; k++) {
          for (int iens=0; iens < nens; iens++) {
            real dens_k        = vars_host(1,k-1,iens) * r_nx_ny;
            real p_actual_km12 = vars_host(0,k-1,iens) * r_nx_ny;
            real p_actual_kp12 = vars_host(0,k  ,iens) * r_nx_ny;
            real p_mult_km12   = pressure_mult_host(k-1,iens);
            real p_hydro_kp12  = p_actual_km12*p_mult_km12 - dens_k*grav*dz;
            pressure_mult_host(k,iens) = p_hydro_kp12 / p_actual_kp12;
          }
        }
        pressure_mult_host.deep_copy_to(pressure_mult);
        coupler.set_option<bool>("save_pressure_z",false);
      }
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


