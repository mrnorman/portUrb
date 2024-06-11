
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include <random>
#include <sstream>

namespace modules {

  struct Dynamics_Euler_Stratified_Jacobian {
    // Order of accuracy (numerical convergence for smooth flows) for the dynamical core
    // 9th-order
    yakl::index_t static constexpr ord  = 9;
    yakl::index_t static constexpr ngll = 6;
    // // 7th-order
    // yakl::index_t static constexpr ord  = 7;
    // yakl::index_t static constexpr ngll = 5;
    // // 5th-order
    // yakl::index_t static constexpr ord  = 5;
    // yakl::index_t static constexpr ngll = 4;
    // // 3rd-order
    // yakl::index_t static constexpr ord  = 3;
    // yakl::index_t static constexpr ngll = 3;
    int static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-velocity
    int  static constexpr idV = 2;  // v-velocity
    int  static constexpr idW = 3;  // w-velocity
    int  static constexpr idP = 4;  // Pressure


    template <yakl::index_t N>
    YAKL_INLINE static void normalize( SArray<real,1,N> &s ) {
      real mn = s(0);
      real mx = s(0);
      for (int i=1; i < N; i++) {
        mn = std::min( mn , s(i) );
        mx = std::max( mx , s(i) );
      }
      real scale = 1;
      if (mx-mn > 1.e-10) scale = mx-mn;
      for (int i=0; i < N; i++) { s(i) = (s(i) - mn) / scale; }
    }



    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real constexpr maxwave = 350 + 50;
      real cfl = coupler.get_option<real>("cfl",0.6);
      return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
    }
    // real compute_time_step( core::Coupler const &coupler ) const {
    //   using yakl::c::parallel_for;
    //   using yakl::c::SimpleBounds;
    //   auto nx = coupler.get_nx();
    //   auto ny = coupler.get_ny();
    //   auto nz = coupler.get_nz();
    //   auto dx = coupler.get_dx();
    //   auto dy = coupler.get_dy();
    //   auto dz = coupler.get_dz();
    //   auto R_d = coupler.get_option<real>("R_d");
    //   auto gamma = coupler.get_option<real>("gamma_d");
    //   auto &dm = coupler.get_data_manager_readonly();
    //   auto rho_d = dm.get<real const,3>("density_dry");
    //   auto uvel  = dm.get<real const,3>("uvel"       );
    //   auto vvel  = dm.get<real const,3>("vvel"       );
    //   auto wvel  = dm.get<real const,3>("wvel"       );
    //   auto temp  = dm.get<real const,3>("temp"       );
    //   real3d dt3d("dt3d",nz,ny,nx);
    //   real cfl = coupler.get_option<real>("cfl",0.6);
    //   parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
    //     real r = rho_d(k,j,i);
    //     real u = uvel (k,j,i);
    //     real v = vvel (k,j,i);
    //     real w = wvel (k,j,i);
    //     real T = temp (k,j,i);
    //     real p = r*R_d*T;
    //     real cs = std::sqrt(gamma*p/r);
    //     real dtx = cfl*dx/(std::abs(u)+cs);
    //     real dty = cfl*dy/(std::abs(v)+cs);
    //     real dtz = cfl*dz/(std::abs(w)+cs);
    //     dt3d(k,j,i) = std::min(std::min(dtx,dty),dtz);
    //   });
    //   real maxwave = yakl::intrinsics::minval(dt3d);
    //   return coupler.get_parallel_comm().all_reduce( maxwave , MPI_MIN );
    // }


    // Perform a time step
    void time_step(core::Coupler &coupler, real dt_phys) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("time_step");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers             = coupler.get_num_tracers();
      auto nx                      = coupler.get_nx();
      auto ny                      = coupler.get_ny();
      auto nz                      = coupler.get_nz();
      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      state   = 0;
      tracers = 0;
      convert_coupler_to_dynamics( coupler , state , tracers );
      real dt_dyn = compute_time_step( coupler );
      int ncycles = (int) std::ceil( dt_phys / dt_dyn );
      ncycles = 1;
      dt_dyn = dt_phys / ncycles;
      for (int icycle = 0; icycle < ncycles; icycle++) {
        time_step_rk_3_3_advect(coupler,state,tracers,dt_dyn);
        int acoustic_cycles = coupler.get_option<int>("acoustic_cycles",1);
        for (int isub = 0; isub < acoustic_cycles; isub++) {
          time_step_rk_3_3_acoust(coupler,state,tracers,dt_dyn/acoustic_cycles);
        }
      }
      convert_dynamics_to_coupler( coupler , state , tracers );
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("time_step");
      #endif
    }


    // CFL 0.45 (Differs from paper, but this is the true value for this high-order FV scheme)
    // Third-order, three-stage SSPRK method
    // https://link.springer.com/content/pdf/10.1007/s10915-008-9239-z.pdf
    void time_step_rk_3_3_advect( core::Coupler & coupler ,
                                   real4d const  & state   ,
                                   real4d const  & tracers ,
                                   real            dt_dyn  ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("time_step_rk_3_3_advect");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto &dm         = coupler.get_data_manager_readonly();
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");
      // SSPRK3 requires temporary arrays to hold intermediate state and tracers arrays
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers_tmp ("tracers_tmp" ,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      // To hold tendencies
      real4d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     );
      real4d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     );

      enforce_immersed_boundaries( coupler , state , tracers , dt_dyn/2 );

      //////////////
      // Stage 1
      //////////////
      compute_tendencies_advect(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i) + dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,hs+k,hs+j,hs+i) = tracers(l,hs+k,hs+j,hs+i) + dt_dyn * tracers_tend(l,k,j,i);
          // Ensure positive tracers stay positive
          if (tracer_positive(l)) tracers_tmp(l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i) );
        }
      });
      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp , dt_dyn/2 );
      //////////////
      // Stage 2
      //////////////
      compute_tendencies_advect(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/4.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,hs+k,hs+j,hs+i) = (3._fp/4._fp) * state      (l,hs+k,hs+j,hs+i) + 
                                          (1._fp/4._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                          (1._fp/4._fp) * dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,hs+k,hs+j,hs+i) = (3._fp/4._fp) * tracers    (l,hs+k,hs+j,hs+i) + 
                                          (1._fp/4._fp) * tracers_tmp(l,hs+k,hs+j,hs+i) +
                                          (1._fp/4._fp) * dt_dyn * tracers_tend(l,k,j,i);
          // Ensure positive tracers stay positive
          if (tracer_positive(l))  tracers_tmp(l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers_tmp(l,hs+k,hs+j,hs+i) );
        }
      });
      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp , dt_dyn/2 );
      //////////////
      // Stage 3
      //////////////
      compute_tendencies_advect(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,2.*dt_dyn/3.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state  (l,hs+k,hs+j,hs+i) = (1._fp/3._fp) * state      (l,hs+k,hs+j,hs+i) +
                                      (2._fp/3._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                      (2._fp/3._fp) * dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers(l,hs+k,hs+j,hs+i) = (1._fp/3._fp) * tracers    (l,hs+k,hs+j,hs+i) +
                                      (2._fp/3._fp) * tracers_tmp(l,hs+k,hs+j,hs+i) +
                                      (2._fp/3._fp) * dt_dyn * tracers_tend(l,k,j,i);
          // Ensure positive tracers stay positive
          if (tracer_positive(l))  tracers(l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers(l,hs+k,hs+j,hs+i) );
        }
      });

      enforce_immersed_boundaries( coupler , state , tracers , dt_dyn/2 );
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("time_step_rk_3_3_advect");
      #endif
    }


    // CFL 0.45 (Differs from paper, but this is the true value for this high-order FV scheme)
    // Third-order, three-stage SSPRK method
    // https://link.springer.com/content/pdf/10.1007/s10915-008-9239-z.pdf
    void time_step_rk_3_3_acoust( core::Coupler & coupler ,
                                  real4d const  & state   ,
                                  real4d const  & tracers ,
                                  real            dt_dyn  ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("time_step_rk_3_3_acoust");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto &dm         = coupler.get_data_manager_readonly();
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     );

      enforce_immersed_boundaries( coupler , state , real4d() , dt_dyn/2 );

      //////////////
      // Stage 1
      //////////////
      compute_tendencies_acoust(coupler,state,state_tend,dt_dyn);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state,nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        state_tmp  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i) + dt_dyn * state_tend  (l,k,j,i);
      });
      enforce_immersed_boundaries( coupler , state_tmp , real4d() , dt_dyn/2 );
      //////////////
      // Stage 2
      //////////////
      compute_tendencies_acoust(coupler,state_tmp,state_tend,dt_dyn/4.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state,nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        state_tmp  (l,hs+k,hs+j,hs+i) = (3._fp/4._fp) * state      (l,hs+k,hs+j,hs+i) + 
                                        (1._fp/4._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                        (1._fp/4._fp) * dt_dyn * state_tend  (l,k,j,i);
      });
      enforce_immersed_boundaries( coupler , state_tmp , real4d() , dt_dyn/2 );
      //////////////
      // Stage 3
      //////////////
      compute_tendencies_acoust(coupler,state_tmp,state_tend,2.*dt_dyn/3.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state,nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        state  (l,hs+k,hs+j,hs+i) = (1._fp/3._fp) * state      (l,hs+k,hs+j,hs+i) +
                                    (2._fp/3._fp) * state_tmp  (l,hs+k,hs+j,hs+i) +
                                    (2._fp/3._fp) * dt_dyn * state_tend  (l,k,j,i);
      });

      enforce_immersed_boundaries( coupler , state , real4d() , dt_dyn/2 );
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("time_step_rk_3_3_acoust");
      #endif
    }


    // CFL 0.45 (Differs from paper, but this is the true value for this high-order FV scheme)
    // Third-order, three-stage SSPRK method
    // https://link.springer.com/content/pdf/10.1007/s10915-008-9239-z.pdf
    void time_step_kgrk_5_3_acoust( core::Coupler & coupler ,
                                    real4d const  & state   ,
                                    real4d const  & tracers ,
                                    real            dt_dyn  ) const {
       // ! KG 3nd order 5 stage:   CFL=sqrt( 4^2 -1) = 3.87
       // ! but nonlinearly only 2nd order
       // ! u1 = u0 + dt/5 RHS(u0)
       // ! u2 = u0 + dt/5 RHS(u1)
       // ! u3 = u0 + dt/3 RHS(u2)
       // ! u4 = u0 + dt/2 RHS(u3)
       // ! u5 = u0 + dt   RHS(u4)
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto &dm         = coupler.get_data_manager_readonly();
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");
      real4d state_tmp   ("state_tmp"   ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     );

      enforce_immersed_boundaries( coupler , state , real4d() , dt_dyn );

      //////////////
      // Stage 1
      //////////////
      compute_tendencies_acoust(coupler,state,state_tend,dt_dyn/5.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state_tmp(l,hs+k,hs+j,hs+i) = state(l,hs+k,hs+j,hs+i) + dt_dyn/5.*state_tend(l,k,j,i);
      });
      enforce_immersed_boundaries( coupler , state_tmp , real4d() , dt_dyn );
      //////////////
      // Stage 2
      //////////////
      compute_tendencies_acoust(coupler,state_tmp,state_tend,dt_dyn/5.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state_tmp(l,hs+k,hs+j,hs+i) = state(l,hs+k,hs+j,hs+i) + dt_dyn/5.*state_tend(l,k,j,i);
      });
      enforce_immersed_boundaries( coupler , state_tmp , real4d() , dt_dyn );
      //////////////
      // Stage 3
      //////////////
      compute_tendencies_acoust(coupler,state_tmp,state_tend,dt_dyn/3.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state_tmp(l,hs+k,hs+j,hs+i) = state(l,hs+k,hs+j,hs+i) + dt_dyn/3.*state_tend(l,k,j,i);
      });
      enforce_immersed_boundaries( coupler , state_tmp , real4d() , dt_dyn );
      //////////////
      // Stage 4
      //////////////
      compute_tendencies_acoust(coupler,state_tmp,state_tend,dt_dyn/2.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state_tmp(l,hs+k,hs+j,hs+i) = state(l,hs+k,hs+j,hs+i) + dt_dyn/2.*state_tend(l,k,j,i);
      });
      enforce_immersed_boundaries( coupler , state_tmp , real4d() , dt_dyn );
      //////////////
      // Stage 5
      //////////////
      compute_tendencies_acoust(coupler,state_tmp,state_tend,dt_dyn/1.);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        state    (l,hs+k,hs+j,hs+i) = state(l,hs+k,hs+j,hs+i) + dt_dyn/1.*state_tend(l,k,j,i);
      });
      enforce_immersed_boundaries( coupler , state , real4d() , dt_dyn );
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("time_step_rk_3_3_acoust");
      #endif
    }



    void enforce_immersed_boundaries( core::Coupler const & coupler ,
                                      real4d        const & state   ,
                                      real4d        const & tracers ,
                                      real                  dt      ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("enforce_immersed_boundaries");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers       = coupler.get_num_tracers();
      auto nx                = coupler.get_nx();
      auto ny                = coupler.get_ny();
      auto nz                = coupler.get_nz();
      auto immersed_power    = coupler.get_option<real>("immersed_power",4);
      auto &dm               = coupler.get_data_manager_readonly();
      auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells");
      auto immersed_prop     = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion

      real immersed_tau = dt;

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real mult = dt/immersed_tau * std::pow( immersed_prop(hs+k,hs+j,hs+i) , immersed_power );
        // TODO: Find a way to calculate drag in here
        // Density
        {
          auto &var = state(idR,hs+k,hs+j,hs+i);
          real  target = hy_dens_cells(hs+k);
          var = var + (target - var)*mult;
        }
        // u-momentum
        {
          auto &var = state(idU,hs+k,hs+j,hs+i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // v-momentum
        {
          auto &var = state(idV,hs+k,hs+j,hs+i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // w-momentum
        {
          auto &var = state(idW,hs+k,hs+j,hs+i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // pressure perturbation (zero == hydrostasis)
        {
          auto &var = state(idP,hs+k,hs+j,hs+i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
        // Tracers
        if (tracers.initialized()) {
          for (int tr=0; tr < num_tracers; tr++) {
            auto &var = tracers(tr,hs+k,hs+j,hs+i);
            real  target = 0;
            var = var + (target - var)*mult;
          }
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("enforce_immersed_boundaries");
      #endif
    }



    // Once you encounter an immersed boundary, set zero derivative boundary conditions
    template <class FP, yakl::index_t ORD>
    YAKL_INLINE static void modify_stencil_immersed_der0( SArray<FP  ,1,ORD>       & stencil  ,
                                                          SArray<bool,1,ORD> const & immersed ) {
      int constexpr hs = (ORD-1)/2;
      // Don't modify the stencils of immersed cells
      if (! immersed(hs)) {
        // Move out from the center of the stencil. once you encounter a boundary, enforce zero derivative,
        //     which is essentially replication of the last in-domain value
        for (int i2=hs+1; i2<ORD; i2++) {
          if (immersed(i2)) { for (int i3=i2; i3<ORD; i3++) { stencil(i3) = stencil(i2-1); }; break; }
        }
        for (int i2=hs-1; i2>=0 ; i2--) {
          if (immersed(i2)) { for (int i3=i2; i3>=0 ; i3--) { stencil(i3) = stencil(i2+1); }; break; }
        }
      }
    }



    // Compute semi-discrete tendencies in x, y, and z directions
    // Fully split in dimensions, and coupled together inside RK stages
    // dt is not used at the moment
    void compute_tendencies_advect( core::Coupler       & coupler      ,
                                    real4d        const & state        ,
                                    real4d        const & state_tend   ,
                                    real4d        const & tracers      ,
                                    real4d        const & tracers_tend ,
                                    real                  dt           ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("compute_tendencies_advect");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx                = coupler.get_nx();    // Proces-local number of cells
      auto ny                = coupler.get_ny();    // Proces-local number of cells
      auto nz                = coupler.get_nz();    // Total vertical cells
      auto dx                = coupler.get_dx();    // grid spacing
      auto dy                = coupler.get_dy();    // grid spacing
      auto dz                = coupler.get_dz();    // grid spacing
      auto sim2d             = coupler.is_sim2d();  // Is this a 2-D simulation?
      auto enable_gravity    = coupler.get_option<bool>("enable_gravity",true);
      auto C0                = coupler.get_option<real>("C0"     );  // pressure = C0*pow(rho*theta,gamma)
      auto grav              = coupler.get_option<real>("grav"   );  // Gravity
      auto gamma             = coupler.get_option<real>("gamma_d");  // cp_dry / cv_dry (about 1.4)
      auto latitude          = coupler.get_option<real>("latitude",0); // For coriolis
      auto num_tracers       = coupler.get_num_tracers();            // Number of tracers
      auto &dm               = coupler.get_data_manager_readonly();  // Grab read-only data manager
      auto tracer_positive   = dm.get<bool const,1>("tracer_positive"      ); // Is a tracer positive-definite?
      auto immersed_prop     = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
      auto any_immersed      = dm.get<bool const,3>("dycore_any_immersed10"); // Are any immersed in 3-D halo?
      auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells");
      auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real r_dz = 1./dz; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

      yakl::SArray<real,2,ord,ngll> s2g;
      yakl::SArray<real,2,ord,2   > s2e;
      yakl::SArray<real,2,ord,ngll> s2d2g;
      yakl::SArray<real,1,ngll> gll_pts, gll_wts;
      using yakl::intrinsics::matmul_cr;

      {
        yakl::SArray<real,2,ord,ord > s2c;
        yakl::SArray<real,2,ord,ord > c2d;
        yakl::SArray<real,2,ord,ngll> c2g;
        yakl::SArray<real,2,ord,2   > c2e;
        TransformMatrices::sten_to_coefs     (s2c);
        TransformMatrices::coefs_to_deriv    (c2d);
        TransformMatrices::coefs_to_gll_lower(c2g);
        TransformMatrices::coefs_to_gll_lower(c2e);
        s2g   = matmul_cr( c2g , s2c );
        s2e   = matmul_cr( c2e , s2c );
        s2d2g = matmul_cr( c2g , matmul_cr( c2d , s2c ) );
        TransformMatrices::get_gll_points ( gll_pts );
        TransformMatrices::get_gll_weights( gll_wts );
      }

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      {
        core::MultiField<real,3> fields;
        for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<3>(l,0,0,0) ); }
        for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<3>(l,0,0,0) ); }
        if (ord > 1) coupler.halo_exchange( fields , hs );
        halo_boundary_conditions( coupler , state , tracers );
      }

      // Create arrays to hold cell interface interpolations
      real5d state_limits_x   ("state_limits_x"   ,2,num_state  ,nz,ny,nx+1); state_limits_x   = 0;
      real5d state_limits_y   ("state_limits_y"   ,2,num_state  ,nz,ny+1,nx); state_limits_y   = 0;
      real5d state_limits_z   ("state_limits_z"   ,2,num_state  ,nz+1,ny,nx); state_limits_z   = 0;
      real5d tracers_limits_x ("tracers_limits_x" ,2,num_tracers,nz,ny,nx+1); tracers_limits_x = 0;
      real5d tracers_limits_y ("tracers_limits_y" ,2,num_tracers,nz,ny+1,nx); tracers_limits_y = 0;
      real5d tracers_limits_z ("tracers_limits_z" ,2,num_tracers,nz+1,ny,nx); tracers_limits_z = 0;

      state_tend   = 0;
      tracers_tend = 0;

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) ,
                                        YAKL_LAMBDA (int k, int j, int i) {
        // ADVECTIVE
        {
          SArray<real,1,ord > stencil;
          // u-velocity
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idU,hs+k,hs+j,i+ii); }
          auto u_vals = matmul_cr( s2g , stencil );
          // immersed
          SArray<bool,1,ord> immersed;
          for (int ii=0; ii<ord; ii++) { immersed(ii) = immersed_prop(hs+k,hs+j,i+ii) > 0;}
          // State
          for (int l=0; l < num_state; l++) {
            // Gather stencil
            for (int ii=0; ii < ord; ii++) { stencil(ii) = state(l,hs+k,hs+j,i+ii); }
            if (l == idV || l == idW) modify_stencil_immersed_der0( stencil , immersed );
            // Compute local tendency from derivative
            auto der = matmul_cr( s2d2g , stencil );
            real tend = 0;
            for (int ii=0; ii < ngll; ii++) { tend += -u_vals(ii)*der(ii)/dx*gll_wts(ii); }
            state_tend(l,k,j,i) += tend;
            // Store interface values for wave propagation
            auto val = matmul_cr( s2e , stencil );
            state_limits_x(1,l,k,j,i  ) = val(0);
            state_limits_x(0,l,k,j,i+1) = val(1);
          }
          // Tracers
          for (int l=0; l < num_tracers; l++) {
            // Gather stencil
            for (int ii=0; ii < ord; ii++) { stencil(ii) = tracers(l,hs+k,hs+j,i+ii); }
            // Compute local tendency from derivative
            auto der = matmul_cr( s2d2g , stencil );
            real tend = 0;
            for (int ii=0; ii < ngll; ii++) { tend += -u_vals(ii)*der(ii)/dx*gll_wts(ii); }
            tracers_tend(l,k,j,i) += tend;
            // Store interface values for wave propagation
            auto val = matmul_cr( s2e , stencil );
            tracers_limits_x(1,l,k,j,i  ) = val(0);
            tracers_limits_x(0,l,k,j,i+1) = val(1);
          }
        }
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) ,
                                        YAKL_LAMBDA (int k, int j, int i) {
        // ADVECTION
        {
          SArray<real,1,ord > stencil;
          // v-velocity
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idV,hs+k,j+jj,hs+i); }
          auto v_vals = matmul_cr( s2g , stencil );
          // immersed
          SArray<bool,1,ord> immersed;
          for (int jj=0; jj<ord; jj++) { immersed(jj) = immersed_prop(hs+k,j+jj,hs+i) > 0;}
          // State
          for (int l=0; l < num_state; l++) {
            // Gather stencil
            for (int jj=0; jj < ord; jj++) { stencil(jj) = state(l,hs+k,j+jj,hs+i); }
            if (l == idU || l == idW) modify_stencil_immersed_der0( stencil , immersed );
            // Compute local tendency from derivative
            auto der = matmul_cr( s2d2g , stencil );
            real tend = 0;
            for (int jj=0; jj < ngll; jj++) { tend += -v_vals(jj)*der(jj)/dy*gll_wts(jj); }
            state_tend(l,k,j,i) += tend;
            // Store interface values for wave propagation
            auto val = matmul_cr( s2e , stencil );
            state_limits_y(1,l,k,j  ,i) = val(0);
            state_limits_y(0,l,k,j+1,i) = val(1);
          }
          // Tracers
          for (int l=0; l < num_tracers; l++) {
            // Gather stencil
            for (int jj=0; jj < ord; jj++) { stencil(jj) = tracers(l,hs+k,j+jj,hs+i); }
            // Compute local tendency from derivative
            auto der = matmul_cr( s2d2g , stencil );
            real tend = 0;
            for (int jj=0; jj < ngll; jj++) { tend += -v_vals(jj)*der(jj)/dy*gll_wts(jj); }
            tracers_tend(l,k,j,i) += tend;
            // Store interface values for wave propagation
            auto val = matmul_cr( s2e , stencil );
            tracers_limits_y(1,l,k,j  ,i) = val(0);
            tracers_limits_y(0,l,k,j+1,i) = val(1);
          }
        }
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) ,
                                        YAKL_LAMBDA (int k, int j, int i) {
        // ADVECTIVE
        {
          SArray<real,1,ord > stencil;
          // w-velocity
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idW,k+kk,hs+j,hs+i); }
          auto w_vals = matmul_cr( s2g , stencil );
          // immersed
          SArray<bool,1,ord> immersed;
          for (int kk=0; kk<ord; kk++) { immersed(kk) = immersed_prop(k+kk,hs+j,hs+i) > 0;}
          // State
          for (int l=0; l < num_state; l++) {
            // Gather stencil
            for (int kk=0; kk < ord; kk++) { stencil(kk) = state(l,k+kk,hs+j,hs+i); }
            if (l == idP) {
              for (int kk=0; kk < ord; kk++) { stencil(kk) += hy_pressure_cells(k+kk); }
            }
            if (l == idU || l == idV) modify_stencil_immersed_der0( stencil , immersed );
            // Compute local tendency from derivative
            auto der = matmul_cr( s2d2g , stencil );
            real tend = 0;
            for (int kk=0; kk < ngll; kk++) { tend += -w_vals(kk)*der(kk)/dz*gll_wts(kk); }
            state_tend(l,k,j,i) += tend;
            // Store interface values for wave propagation
            auto val = matmul_cr( s2e , stencil );
            state_limits_z(1,l,k  ,j,i) = val(0);
            state_limits_z(0,l,k+1,j,i) = val(1);
          }
          // Tracers
          for (int l=0; l < num_tracers; l++) {
            // Gather stencil
            for (int kk=0; kk < ord; kk++) { stencil(kk) = tracers(l,k+kk,hs+j,hs+i); }
            // Compute local tendency from derivative
            auto der = matmul_cr( s2d2g , stencil );
            real tend = 0;
            for (int kk=0; kk < ngll; kk++) { tend += -w_vals(kk)*der(kk)/dz*gll_wts(kk); }
            tracers_tend(l,k,j,i) += tend;
            // Store interface values for wave propagation
            auto val = matmul_cr( s2e , stencil );
            tracers_limits_z(1,l,k  ,j,i) = val(0);
            tracers_limits_z(0,l,k+1,j,i) = val(1);
          }
        }
      });

      // Perform periodic horizontal exchange of cell-edge data, and implement vertical boundary conditions
      edge_exchange( coupler , state_limits_x , tracers_limits_x ,
                               state_limits_y , tracers_limits_y ,
                               state_limits_z , tracers_limits_z );

      // To save on space, slice the limits arrays to store single-valued interface fluxes
      real5d state_prop_x  ("state_prop_x  ",2,num_state  ,nz  ,ny  ,nx+1);
      real5d state_prop_y  ("state_prop_y  ",2,num_state  ,nz  ,ny+1,nx  );
      real5d state_prop_z  ("state_prop_z  ",2,num_state  ,nz+1,ny  ,nx  );
      real5d tracers_prop_x("tracers_prop_x",2,num_tracers,nz  ,ny  ,nx+1);
      real5d tracers_prop_y("tracers_prop_y",2,num_tracers,nz  ,ny+1,nx  );
      real5d tracers_prop_z("tracers_prop_z",2,num_tracers,nz+1,ny  ,nx  );

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      // Speed of sound and its reciprocal. Using a constant speed of sound for upwinding
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i) {
        if (j < ny && k < nz) {
          // Advective
          {
            real u = 0.5_fp * ( state_limits_x(0,idU,k,j,i) + state_limits_x(1,idU,k,j,i) );
            int ind = u > 0 ? 1 : 0;
            for (int l=0; l < num_state; l++) {
              state_prop_x  (  ind,l,k,j,i) = u*(state_limits_x  (1,l,k,j,i)-state_limits_x  (0,l,k,j,i));
              state_prop_x  (1-ind,l,k,j,i) = 0;
            }
            for (int l=0; l < num_tracers; l++) {
              tracers_prop_x(  ind,l,k,j,i) = u*(tracers_limits_x(1,l,k,j,i)-tracers_limits_x(0,l,k,j,i));
              tracers_prop_x(1-ind,l,k,j,i) = 0;
            }
          }
        }
        if (i < nx && k < nz && !sim2d) {
          // Advective
          {
            real v = 0.5_fp * ( state_limits_y(0,idV,k,j,i) + state_limits_y(1,idV,k,j,i) );
            int ind = v > 0 ? 1 : 0;
            for (int l=0; l < num_state; l++) {
              state_prop_y  (  ind,l,k,j,i) = v*(state_limits_y  (1,l,k,j,i)-state_limits_y  (0,l,k,j,i));
              state_prop_y  (1-ind,l,k,j,i) = 0;
            }
            for (int l=0; l < num_tracers; l++) {
              tracers_prop_y(  ind,l,k,j,i) = v*(tracers_limits_y(1,l,k,j,i)-tracers_limits_y(0,l,k,j,i));
              tracers_prop_y(1-ind,l,k,j,i) = 0;
            }
          }
        }
        if (i < nx && j < ny) {
          // Advective
          {
            real w = 0.5_fp * ( state_limits_z(0,idW,k,j,i) + state_limits_z(1,idW,k,j,i) );
            int ind = w > 0 ? 1 : 0;
            for (int l=0; l < num_state; l++) {
              state_prop_z  (  ind,l,k,j,i) = w*(state_limits_z  (1,l,k,j,i)-state_limits_z  (0,l,k,j,i));
              state_prop_z  (1-ind,l,k,j,i) = 0;
            }
            for (int l=0; l < num_tracers; l++) {
              tracers_prop_z(  ind,l,k,j,i) = w*(tracers_limits_z(1,l,k,j,i)-tracers_limits_z(0,l,k,j,i));
              tracers_prop_z(1-ind,l,k,j,i) = 0;
            }
          }
        }
      });

      // Compute tendencies as the flux divergence + gravity source term + coriolis
      int mx = std::max(num_state,num_tracers);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(mx,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_tracers) {
          tracers_tend(l,k,j,i) += -( tracers_prop_x(0,l,k,j,i+1) + tracers_prop_x(1,l,k,j,i) ) * r_dx
                                   -( tracers_prop_y(0,l,k,j+1,i) + tracers_prop_y(1,l,k,j,i) ) * r_dy 
                                   -( tracers_prop_z(0,l,k+1,j,i) + tracers_prop_z(1,l,k,j,i) ) * r_dz;
        }
        if (l < num_state) {
          state_tend  (l,k,j,i) += -( state_prop_x  (0,l,k,j,i+1) + state_prop_x  (1,l,k,j,i) ) * r_dx
                                   -( state_prop_y  (0,l,k,j+1,i) + state_prop_y  (1,l,k,j,i) ) * r_dy
                                   -( state_prop_z  (0,l,k+1,j,i) + state_prop_z  (1,l,k,j,i) ) * r_dz;
          if (l == idV && sim2d) state_tend(l,k,j,i) = 0;
          if (latitude != 0 && !sim2d && l == idU) state_tend(l,k,j,i) += fcor*state(idV,hs+k,hs+j,hs+i);
          if (latitude != 0 && !sim2d && l == idV) state_tend(l,k,j,i) -= fcor*state(idU,hs+k,hs+j,hs+i);
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("compute_tendencies_advect");
      #endif
    }



    void compute_tendencies_acoust( core::Coupler       & coupler      ,
                                    real4d        const & state        ,
                                    real4d        const & state_tend   ,
                                    real                  dt           ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("compute_tendencies_acoust");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx                = coupler.get_nx();    // Proces-local number of cells
      auto ny                = coupler.get_ny();    // Proces-local number of cells
      auto nz                = coupler.get_nz();    // Total vertical cells
      auto dx                = coupler.get_dx();    // grid spacing
      auto dy                = coupler.get_dy();    // grid spacing
      auto dz                = coupler.get_dz();    // grid spacing
      auto sim2d             = coupler.is_sim2d();  // Is this a 2-D simulation?
      auto enable_gravity    = coupler.get_option<bool>("enable_gravity",true);
      auto C0                = coupler.get_option<real>("C0"     );  // pressure = C0*pow(rho*theta,gamma)
      auto grav              = coupler.get_option<real>("grav"   );  // Gravity
      auto gamma             = coupler.get_option<real>("gamma_d");  // cp_dry / cv_dry (about 1.4)
      auto latitude          = coupler.get_option<real>("latitude",0); // For coriolis
      auto &dm               = coupler.get_data_manager_readonly();
      auto immersed_prop     = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
      auto any_immersed      = dm.get<bool const,3>("dycore_any_immersed10"); // Are any immersed in 3-D halo?
      auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells");
      auto hy_dens_edges     = dm.get<real const,1>("hy_dens_edges");
      auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real r_dz = 1./dz; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

      yakl::SArray<real,2,ord,2   > s2e;
      using yakl::intrinsics::matmul_cr;
      {
        yakl::SArray<real,2,ord,ord > s2c;
        yakl::SArray<real,2,ord,2   > c2e;
        TransformMatrices::sten_to_coefs     (s2c);
        TransformMatrices::coefs_to_gll_lower(c2e);
        s2e = matmul_cr( c2e , s2c );
      }

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      {
        core::MultiField<real,3> fields_x, fields_y;
        fields_x.add_field( state.slice<3>(idR,0,0,0) );
        fields_x.add_field( state.slice<3>(idU,0,0,0) );
        fields_x.add_field( state.slice<3>(idP,0,0,0) );
        if (ord > 1) coupler.halo_exchange_x( fields_x , hs );
        fields_y.add_field( state.slice<3>(idR,0,0,0) );
        fields_y.add_field( state.slice<3>(idV,0,0,0) );
        fields_y.add_field( state.slice<3>(idP,0,0,0) );
        if (ord > 1) coupler.halo_exchange_y( fields_y , hs );
        halo_boundary_conditions_acoust( coupler , state );
      }

      // Create arrays to hold cell interface interpolations
      real4d u_limits_x("u_limits_x",2,nz,ny,nx+1);
      real4d p_limits_x("p_limits_x",2,nz,ny,nx+1);
      real4d v_limits_y("v_limits_y",2,nz,ny+1,nx);
      real4d p_limits_y("p_limits_y",2,nz,ny+1,nx);
      real4d w_limits_z("w_limits_z",2,nz+1,ny,nx);
      real4d p_limits_z("p_limits_z",2,nz+1,ny,nx);

      real constexpr cs   = 350.;
      real constexpr cs2  = cs*cs;
      real constexpr r_cs = 1./cs;

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) ,
                                        YAKL_LAMBDA (int k, int j, int i) {
        // SArray<bool,1,ord > immersed;
        // for (int ii=0; ii < ord; ii++) { immersed(ii) = immersed_prop(hs+k,hs+j,i+ii) > 0; }
        SArray<real,1,ord > stencil;
        // u-vel
        for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idU,hs+k,hs+j,i+ii); }
        auto val = matmul_cr( s2e , stencil );
        u_limits_x(1,k,j,i  ) = val(0);
        u_limits_x(0,k,j,i+1) = val(1);
        state_tend(idR,k,j,i) = -hy_dens_cells(hs+k)*(val(1)-val(0))/dx;
        state_tend(idP,k,j,i) = -hy_dens_cells(hs+k)*(val(1)-val(0))/dx*cs2;
        // pressure
        for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idP,hs+k,hs+j,i+ii); }
        // modify_stencil_immersed_der0( stencil , immersed );
        val = matmul_cr( s2e , stencil );
        p_limits_x(1,k,j,i  ) = val(0);
        p_limits_x(0,k,j,i+1) = val(1);
        state_tend(idU,k,j,i) = -(val(1)-val(0))/dx/hy_dens_cells(hs+k);
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) ,
                                        YAKL_LAMBDA (int k, int j, int i) {
        // SArray<bool,1,ord > immersed;
        // for (int jj=0; jj < ord; jj++) { immersed(jj) = immersed_prop(hs+k,j+jj,hs+i) > 0; }
        SArray<real,1,ord > stencil;
        // v-vel
        for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idV,hs+k,j+jj,hs+i); }
        auto val = matmul_cr( s2e , stencil );
        v_limits_y(1,k,j  ,i) = val(0);
        v_limits_y(0,k,j+1,i) = val(1);
        state_tend(idR,k,j,i) += -hy_dens_cells(hs+k)*(val(1)-val(0))/dy;
        state_tend(idP,k,j,i) += -hy_dens_cells(hs+k)*(val(1)-val(0))/dy*cs2;
        // pressure
        for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idP,hs+k,j+jj,hs+i); }
        // modify_stencil_immersed_der0( stencil , immersed );
        val = matmul_cr( s2e , stencil );
        p_limits_y(1,k,j  ,i) = val(0);
        p_limits_y(0,k,j+1,i) = val(1);
        state_tend(idV,k,j,i) = -(val(1)-val(0))/dy/hy_dens_cells(hs+k);
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) ,
                                        YAKL_LAMBDA (int k, int j, int i) {
        // SArray<bool,1,ord > immersed;
        // for (int kk=0; kk < ord; kk++) { immersed(kk) = immersed_prop(k+kk,hs+j,hs+i) > 0; }
        SArray<real,1,ord > stencil;
        // w-vel
        for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idW,k+kk,hs+j,hs+i); }
        auto val = matmul_cr( s2e , stencil );
        w_limits_z(1,k  ,j,i) = val(0);
        w_limits_z(0,k+1,j,i) = val(1);
        state_tend(idR,k,j,i) += -hy_dens_cells(hs+k)*(val(1)-val(0))/dz;
        state_tend(idP,k,j,i) += -hy_dens_cells(hs+k)*(val(1)-val(0))/dz*cs2;
        // pressure
        for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idP,k+kk,hs+j,hs+i); }
        // modify_stencil_immersed_der0( stencil , immersed );
        val = matmul_cr( s2e , stencil );
        p_limits_z(1,k  ,j,i) = val(0);
        p_limits_z(0,k+1,j,i) = val(1);
        state_tend(idW,k,j,i) = -(val(1)-val(0))/dz/hy_dens_cells(hs+k);
      });

      // Perform periodic horizontal exchange of cell-edge data, and implement vertical boundary conditions
      edge_exchange_acoust( coupler , u_limits_x , p_limits_x ,
                                      v_limits_y , p_limits_y ,
                                      w_limits_z , p_limits_z );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i) {
        if (j < ny && k < nz) {
          // Acoustic
          {
            real r  = hy_dens_cells(hs+k);
            real du = u_limits_x(1,k,j,i) - u_limits_x(0,k,j,i);
            real dp = p_limits_x(1,k,j,i) - p_limits_x(0,k,j,i);
            u_limits_x(0,k,j,i) = 0.5_fp*(-du*cs    + dp/r );
            p_limits_x(0,k,j,i) = 0.5_fp*( du*r*cs2 - dp*cs);
            u_limits_x(1,k,j,i) = 0.5_fp*( du*cs    + dp/r );
            p_limits_x(1,k,j,i) = 0.5_fp*( du*r*cs2 + dp*cs);
          }
        }
        if (i < nx && k < nz && !sim2d) {
          // Acoustic
          {
            real r  = hy_dens_cells(hs+k);
            real dv = v_limits_y(1,k,j,i) - v_limits_y(0,k,j,i);
            real dp = p_limits_y(1,k,j,i) - p_limits_y(0,k,j,i);
            v_limits_y(0,k,j,i) = 0.5_fp*(-dv*cs    + dp/r );
            p_limits_y(0,k,j,i) = 0.5_fp*( dv*r*cs2 - dp*cs);
            v_limits_y(1,k,j,i) = 0.5_fp*( dv*cs    + dp/r );
            p_limits_y(1,k,j,i) = 0.5_fp*( dv*r*cs2 + dp*cs);
          }
        }
        if (i < nx && j < ny) {
          // Acoustic
          {
            real r  = hy_dens_edges(k);
            real dw = w_limits_z(1,k,j,i) - w_limits_z(0,k,j,i);
            real dp = p_limits_z(1,k,j,i) - p_limits_z(0,k,j,i);
            w_limits_z(0,k,j,i) = 0.5_fp*(-dw*cs    + dp/r );
            p_limits_z(0,k,j,i) = 0.5_fp*( dw*r*cs2 - dp*cs);
            w_limits_z(1,k,j,i) = 0.5_fp*( dw*cs    + dp/r );
            p_limits_z(1,k,j,i) = 0.5_fp*( dw*r*cs2 + dp*cs);
          }
        }
      });

      // Compute tendencies as the flux divergence + gravity source term + coriolis
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        state_tend(idR,k,j,i) += -( p_limits_x(0,k,j,i+1) + p_limits_x(1,k,j,i) ) * r_dx/cs2
                                 -( p_limits_y(0,k,j+1,i) + p_limits_y(1,k,j,i) ) * r_dy/cs2
                                 -( p_limits_z(0,k+1,j,i) + p_limits_z(1,k,j,i) ) * r_dz/cs2;

        state_tend(idU,k,j,i) += -( u_limits_x(0,k,j,i+1) + u_limits_x(1,k,j,i) ) * r_dx;

        state_tend(idV,k,j,i) += -( v_limits_y(0,k,j+1,i) + v_limits_y(1,k,j,i) ) * r_dy;

        state_tend(idW,k,j,i) += -( w_limits_z(0,k+1,j,i) + w_limits_z(1,k,j,i) ) * r_dz;

        state_tend(idP,k,j,i) += -( p_limits_x(0,k,j,i+1) + p_limits_x(1,k,j,i) ) * r_dx
                                 -( p_limits_y(0,k,j+1,i) + p_limits_y(1,k,j,i) ) * r_dy
                                 -( p_limits_z(0,k+1,j,i) + p_limits_z(1,k,j,i) ) * r_dz;

        if (enable_gravity) {
          state_tend(idW,k,j,i) += -grav*(state(idR,hs+k,hs+j,hs+i) - hy_dens_cells(hs+k))/hy_dens_cells(hs+k);
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("compute_tendencies_acoust");
      #endif
    }



    void halo_boundary_conditions( core::Coupler const & coupler ,
                                   real4d        const & state   ,
                                   real4d        const & tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("halo_boundary_conditions");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx                = coupler.get_nx();
      auto ny                = coupler.get_ny();
      auto nz                = coupler.get_nz();
      auto num_tracers       = coupler.get_num_tracers();
      auto bc_z              = coupler.get_option<std::string>("bc_z","solid_wall");
      auto &dm               = coupler.get_data_manager_readonly();
      auto surface_temp      = dm.get<real const,2>("surface_temp");
      auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells");

      // z-direction BC's
      if (bc_z == "solid_wall") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny,nx) , YAKL_LAMBDA (int kk, int j, int i) {
          state(idR,kk,hs+j,hs+i) = hy_dens_cells(kk);
          state(idU,kk,hs+j,hs+i) = state(idU,hs+0,hs+j,hs+i);
          state(idV,kk,hs+j,hs+i) = state(idV,hs+0,hs+j,hs+i);
          state(idW,kk,hs+j,hs+i) = 0;
          state(idP,kk,hs+j,hs+i) = state(idP,hs+0,hs+j,hs+i);
          state(idR,hs+nz+kk,hs+j,hs+i) = hy_dens_cells(hs+nz+kk);
          state(idU,hs+nz+kk,hs+j,hs+i) = state(idU,hs+nz-1,hs+j,hs+i);
          state(idV,hs+nz+kk,hs+j,hs+i) = state(idV,hs+nz-1,hs+j,hs+i);
          state(idW,hs+nz+kk,hs+j,hs+i) = 0;
          state(idP,hs+nz+kk,hs+j,hs+i) = state(idP,hs+nz-1,hs+j,hs+i);
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,hs+j,hs+i) = 0;
            tracers(l,hs+nz+kk,hs+j,hs+i) = 0;
          }
        });
      } else if (bc_z == "periodic") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny,nx) , YAKL_LAMBDA (int kk, int j, int i) {
          state(idR,      kk,hs+j,hs+i) = state(idR,nz+kk,hs+j,hs+i);
          state(idU,      kk,hs+j,hs+i) = state(idU,nz+kk,hs+j,hs+i);
          state(idV,      kk,hs+j,hs+i) = state(idV,nz+kk,hs+j,hs+i);
          state(idW,      kk,hs+j,hs+i) = state(idW,nz+kk,hs+j,hs+i);
          state(idP,      kk,hs+j,hs+i) = state(idP,nz+kk,hs+j,hs+i);
          state(idR,hs+nz+kk,hs+j,hs+i) = state(idR,hs+kk,hs+j,hs+i);
          state(idU,hs+nz+kk,hs+j,hs+i) = state(idU,hs+kk,hs+j,hs+i);
          state(idV,hs+nz+kk,hs+j,hs+i) = state(idV,hs+kk,hs+j,hs+i);
          state(idW,hs+nz+kk,hs+j,hs+i) = state(idW,hs+kk,hs+j,hs+i);
          state(idP,hs+nz+kk,hs+j,hs+i) = state(idP,hs+kk,hs+j,hs+i);
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,hs+j,hs+i) = tracers(l,nz+kk,hs+j,hs+i);
            tracers(l,hs+nz+kk,hs+j,hs+i) = tracers(l,hs+kk,hs+j,hs+i);
          }
        });
      } else {
        yakl::yakl_throw("ERROR: Specified invalid bc_z in coupler options");
      }
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("halo_boundary_conditions");
      #endif
    }



    void halo_boundary_conditions_acoust( core::Coupler const & coupler ,
                                          real4d        const & state   ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("halo_boundary_conditions");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx                = coupler.get_nx();
      auto ny                = coupler.get_ny();
      auto nz                = coupler.get_nz();
      auto bc_z              = coupler.get_option<std::string>("bc_z","solid_wall");
      auto &dm               = coupler.get_data_manager_readonly();
      auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells");

      // z-direction BC's
      if (bc_z == "solid_wall") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny,nx) , YAKL_LAMBDA (int kk, int j, int i) {
          state(idR,kk,hs+j,hs+i) = hy_dens_cells(kk);
          state(idW,kk,hs+j,hs+i) = 0;
          state(idP,kk,hs+j,hs+i) = state(idP,hs+0,hs+j,hs+i);
          state(idR,hs+nz+kk,hs+j,hs+i) = hy_dens_cells(hs+nz+kk);
          state(idW,hs+nz+kk,hs+j,hs+i) = 0;
          state(idP,hs+nz+kk,hs+j,hs+i) = state(idP,hs+nz-1,hs+j,hs+i);
        });
      } else if (bc_z == "periodic") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny,nx) , YAKL_LAMBDA (int kk, int j, int i) {
          state(idR,      kk,hs+j,hs+i) = state(idR,nz+kk,hs+j,hs+i);
          state(idW,      kk,hs+j,hs+i) = state(idW,nz+kk,hs+j,hs+i);
          state(idP,      kk,hs+j,hs+i) = state(idP,nz+kk,hs+j,hs+i);
          state(idR,hs+nz+kk,hs+j,hs+i) = state(idR,hs+kk,hs+j,hs+i);
          state(idW,hs+nz+kk,hs+j,hs+i) = state(idW,hs+kk,hs+j,hs+i);
          state(idP,hs+nz+kk,hs+j,hs+i) = state(idP,hs+kk,hs+j,hs+i);
        });
      } else {
        yakl::yakl_throw("ERROR: Specified invalid bc_z in coupler options");
      }
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("halo_boundary_conditions");
      #endif
    }



    void edge_exchange( core::Coupler const & coupler           ,
                        real5d        const & state_limits_x    ,
                        real5d        const & tracers_limits_x  ,
                        real5d        const & state_limits_y    ,
                        real5d        const & tracers_limits_y  ,
                        real5d        const & state_limits_z    ,
                        real5d        const & tracers_limits_z  ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("edge_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto num_tracers    = coupler.get_num_tracers();
      auto &neigh         = coupler.get_neighbor_rankid_matrix();
      auto bc_z           = coupler.get_option<std::string>("bc_z","solid_wall");
      auto &dm            = coupler.get_data_manager_readonly();
      auto surface_temp   = dm.get<real const,2>("surface_temp"  );
      int npack = num_state + num_tracers;

      // x-exchange
      {
        real3d edge_send_buf_W("edge_send_buf_W",npack,nz,ny);
        real3d edge_send_buf_E("edge_send_buf_E",npack,nz,ny);
        real3d edge_recv_buf_W("edge_recv_buf_W",npack,nz,ny);
        real3d edge_recv_buf_E("edge_recv_buf_E",npack,nz,ny);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
          if        (v < num_state) {
            edge_send_buf_W(v,k,j) = state_limits_x  (1,v          ,k,j,0 );
            edge_send_buf_E(v,k,j) = state_limits_x  (0,v          ,k,j,nx);
          } else if (v < num_state + num_tracers) {                    
            edge_send_buf_W(v,k,j) = tracers_limits_x(1,v-num_state,k,j,0 );
            edge_send_buf_E(v,k,j) = tracers_limits_x(0,v-num_state,k,j,nx);
          }
        });
        coupler.get_parallel_comm().send_receive<real,3>( { {edge_recv_buf_W,neigh(1,0),4} , {edge_recv_buf_E,neigh(1,2),5} } ,
                                                          { {edge_send_buf_W,neigh(1,0),5} , {edge_send_buf_E,neigh(1,2),4} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
          if        (v < num_state) {
            state_limits_x  (0,v          ,k,j,0 ) = edge_recv_buf_W(v,k,j);
            state_limits_x  (1,v          ,k,j,nx) = edge_recv_buf_E(v,k,j);
          } else if (v < num_state + num_tracers) {
            tracers_limits_x(0,v-num_state,k,j,0 ) = edge_recv_buf_W(v,k,j);
            tracers_limits_x(1,v-num_state,k,j,nx) = edge_recv_buf_E(v,k,j);
          }
        });
      }

      // y-direction exchange
      {
        real3d edge_send_buf_S("edge_send_buf_S",npack,nz,nx);
        real3d edge_send_buf_N("edge_send_buf_N",npack,nz,nx);
        real3d edge_recv_buf_S("edge_recv_buf_S",npack,nz,nx);
        real3d edge_recv_buf_N("edge_recv_buf_N",npack,nz,nx);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if        (v < num_state) {
            edge_send_buf_S(v,k,i) = state_limits_y  (1,v          ,k,0 ,i);
            edge_send_buf_N(v,k,i) = state_limits_y  (0,v          ,k,ny,i);
          } else if (v < num_state + num_tracers) {                    
            edge_send_buf_S(v,k,i) = tracers_limits_y(1,v-num_state,k,0 ,i);
            edge_send_buf_N(v,k,i) = tracers_limits_y(0,v-num_state,k,ny,i);
          }
        });
        coupler.get_parallel_comm().send_receive<real,3>( { {edge_recv_buf_S,neigh(0,1),6} , {edge_recv_buf_N,neigh(2,1),7} } ,
                                                          { {edge_send_buf_S,neigh(0,1),7} , {edge_send_buf_N,neigh(2,1),6} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if        (v < num_state) {
            state_limits_y  (0,v          ,k,0 ,i) = edge_recv_buf_S(v,k,i);
            state_limits_y  (1,v          ,k,ny,i) = edge_recv_buf_N(v,k,i);
          } else if (v < num_state + num_tracers) {
            tracers_limits_y(0,v-num_state,k,0 ,i) = edge_recv_buf_S(v,k,i);
            tracers_limits_y(1,v-num_state,k,ny,i) = edge_recv_buf_N(v,k,i);
          }
        });
      }

      if (bc_z == "solid_wall") {
        // z-direction BC's
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
          // Dirichlet
          state_limits_z(0,idW,0 ,j,i) = 0;
          state_limits_z(1,idW,0 ,j,i) = 0;
          state_limits_z(0,idW,nz,j,i) = 0;
          state_limits_z(1,idW,nz,j,i) = 0;
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(0,l,0 ,j,i) = 0;
            tracers_limits_z(1,l,0 ,j,i) = 0;
            tracers_limits_z(0,l,nz,j,i) = 0;
            tracers_limits_z(1,l,nz,j,i) = 0;
          }
          // Neumann
          state_limits_z(0,idR,0 ,j,i) = state_limits_z(1,idR,0 ,j,i);
          state_limits_z(0,idU,0 ,j,i) = state_limits_z(1,idU,0 ,j,i);
          state_limits_z(0,idV,0 ,j,i) = state_limits_z(1,idV,0 ,j,i);
          state_limits_z(0,idP,0 ,j,i) = state_limits_z(1,idP,0 ,j,i);
          state_limits_z(1,idR,nz,j,i) = state_limits_z(0,idR,nz,j,i);
          state_limits_z(1,idU,nz,j,i) = state_limits_z(0,idU,nz,j,i);
          state_limits_z(1,idV,nz,j,i) = state_limits_z(0,idV,nz,j,i);
          state_limits_z(1,idP,nz,j,i) = state_limits_z(0,idP,nz,j,i);
        });
      } else if (bc_z == "periodic") {
        // z-direction BC's
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
          state_limits_z(0,idR,0 ,j,i) = state_limits_z(0,idR,nz,j,i);
          state_limits_z(0,idU,0 ,j,i) = state_limits_z(0,idU,nz,j,i);
          state_limits_z(0,idV,0 ,j,i) = state_limits_z(0,idV,nz,j,i);
          state_limits_z(0,idW,0 ,j,i) = state_limits_z(0,idW,nz,j,i);
          state_limits_z(0,idP,0 ,j,i) = state_limits_z(0,idP,nz,j,i);
          state_limits_z(1,idR,nz,j,i) = state_limits_z(1,idR,0 ,j,i);
          state_limits_z(1,idU,nz,j,i) = state_limits_z(1,idU,0 ,j,i);
          state_limits_z(1,idV,nz,j,i) = state_limits_z(1,idV,0 ,j,i);
          state_limits_z(1,idW,nz,j,i) = state_limits_z(1,idW,0 ,j,i);
          state_limits_z(1,idP,nz,j,i) = state_limits_z(1,idP,0 ,j,i);
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(0,l,0 ,j,i) = tracers_limits_z(0,l,nz,j,i);
            tracers_limits_z(1,l,nz,j,i) = tracers_limits_z(1,l,0 ,j,i);
          }
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("edge_exchange");
      #endif
    }



    void edge_exchange_acoust( core::Coupler const & coupler    ,
                               real4d        const & u_limits_x ,
                               real4d        const & p_limits_x ,
                               real4d        const & v_limits_y ,
                               real4d        const & p_limits_y ,
                               real4d        const & w_limits_z ,
                               real4d        const & p_limits_z ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("edge_exchange");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto &neigh         = coupler.get_neighbor_rankid_matrix();
      auto bc_z           = coupler.get_option<std::string>("bc_z","solid_wall");
      auto &dm            = coupler.get_data_manager_readonly();
      auto surface_temp   = dm.get<real const,2>("surface_temp"  );
      int npack = 2;

      // x-exchange
      {
        real3d edge_send_buf_W("edge_send_buf_W",npack,nz,ny);
        real3d edge_send_buf_E("edge_send_buf_E",npack,nz,ny);
        real3d edge_recv_buf_W("edge_recv_buf_W",npack,nz,ny);
        real3d edge_recv_buf_E("edge_recv_buf_E",npack,nz,ny);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
          edge_send_buf_W(0,k,j) = u_limits_x(1,k,j,0 );
          edge_send_buf_E(0,k,j) = u_limits_x(0,k,j,nx);
          edge_send_buf_W(1,k,j) = p_limits_x(1,k,j,0 );
          edge_send_buf_E(1,k,j) = p_limits_x(0,k,j,nx);
        });
        coupler.get_parallel_comm().send_receive<real,3>( { {edge_recv_buf_W,neigh(1,0),4} , {edge_recv_buf_E,neigh(1,2),5} } ,
                                                          { {edge_send_buf_W,neigh(1,0),5} , {edge_send_buf_E,neigh(1,2),4} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
          u_limits_x(0,k,j,0 ) = edge_recv_buf_W(0,k,j);
          u_limits_x(1,k,j,nx) = edge_recv_buf_E(0,k,j);
          p_limits_x(0,k,j,0 ) = edge_recv_buf_W(1,k,j);
          p_limits_x(1,k,j,nx) = edge_recv_buf_E(1,k,j);
        });
      }

      // y-direction exchange
      {
        real3d edge_send_buf_S("edge_send_buf_S",npack,nz,nx);
        real3d edge_send_buf_N("edge_send_buf_N",npack,nz,nx);
        real3d edge_recv_buf_S("edge_recv_buf_S",npack,nz,nx);
        real3d edge_recv_buf_N("edge_recv_buf_N",npack,nz,nx);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          edge_send_buf_S(0,k,i) = v_limits_y(1,k,0 ,i);
          edge_send_buf_N(0,k,i) = v_limits_y(0,k,ny,i);
          edge_send_buf_S(1,k,i) = p_limits_y(1,k,0 ,i);
          edge_send_buf_N(1,k,i) = p_limits_y(0,k,ny,i);
        });
        coupler.get_parallel_comm().send_receive<real,3>( { {edge_recv_buf_S,neigh(0,1),6} , {edge_recv_buf_N,neigh(2,1),7} } ,
                                                          { {edge_send_buf_S,neigh(0,1),7} , {edge_send_buf_N,neigh(2,1),6} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          v_limits_y(0,k,0 ,i) = edge_recv_buf_S(0,k,i);
          v_limits_y(1,k,ny,i) = edge_recv_buf_N(0,k,i);
          p_limits_y(0,k,0 ,i) = edge_recv_buf_S(1,k,i);
          p_limits_y(1,k,ny,i) = edge_recv_buf_N(1,k,i);
        });
      }

      if (bc_z == "solid_wall") {
        // z-direction BC's
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
          w_limits_z(0,0 ,j,i) = 0;
          w_limits_z(1,0 ,j,i) = 0;
          w_limits_z(0,nz,j,i) = 0;
          w_limits_z(1,nz,j,i) = 0;
          p_limits_z(0,0 ,j,i) = p_limits_z(1,0 ,j,i);
          p_limits_z(1,nz,j,i) = p_limits_z(0,nz,j,i);
        });
      } else if (bc_z == "periodic") {
        // z-direction BC's
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
          w_limits_z(0,0 ,j,i) = w_limits_z(0,nz,j,i);
          p_limits_z(0,0 ,j,i) = p_limits_z(0,nz,j,i);
          w_limits_z(1,nz,j,i) = w_limits_z(1,0 ,j,i);
          p_limits_z(1,nz,j,i) = p_limits_z(1,0 ,j,i);
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("edge_exchange");
      #endif
    }



    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("init");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto dz             = coupler.get_dz();
      auto nx_glob        = coupler.get_nx_glob();
      auto ny_glob        = coupler.get_ny_glob();
      auto num_tracers    = coupler.get_num_tracers();
      auto gamma          = coupler.get_option<real>("gamma_d");
      auto C0             = coupler.get_option<real>("C0"     );
      auto grav           = coupler.get_option<real>("grav"   );
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);

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

      dm.register_and_allocate<real>("hy_dens_cells"    ,"",{nz+2*hs});
      dm.register_and_allocate<real>("hy_theta_cells"   ,"",{nz+2*hs});
      dm.register_and_allocate<real>("hy_pressure_cells","",{nz+2*hs});
      auto hy_r = dm.get<real,1>("hy_dens_cells"    );    hy_r = 0;
      auto hy_t = dm.get<real,1>("hy_theta_cells"   );    hy_t = 0;
      auto hy_p = dm.get<real,1>("hy_pressure_cells");    hy_p = 0;

      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);  state   = 0;
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);  tracers = 0;
      convert_coupler_to_dynamics( coupler , state , tracers );
      parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , YAKL_LAMBDA (int k) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            hy_r(k) += state(idR,k,hs+j,hs+i);
            hy_p(k) += state(idP,k,hs+j,hs+i);
            hy_t(k) += std::pow( state(idP,k,hs+j,hs+i)/C0 , 1._fp/gamma ) / state(idR,k,hs+j,hs+i);
          }
        }
      });
      coupler.get_parallel_comm().all_reduce( hy_r , MPI_SUM ).deep_copy_to(hy_r);
      coupler.get_parallel_comm().all_reduce( hy_t , MPI_SUM ).deep_copy_to(hy_t);
      coupler.get_parallel_comm().all_reduce( hy_p , MPI_SUM ).deep_copy_to(hy_p);
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , YAKL_LAMBDA (int k) {
        hy_r(k) *= r_nx_ny;
        hy_t(k) *= r_nx_ny;
        hy_p(k) *= r_nx_ny;
      });
      parallel_for( YAKL_AUTO_LABEL() , hs , YAKL_LAMBDA (int kk) {
        {
          int  k0       = hs;
          int  k        = k0-1-kk;
          real rho0     = hy_r(k0);
          real theta0   = hy_t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          hy_r(k) = std::pow( rho0_gm1 + grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          hy_t(k) = theta0;
          hy_p(k) = C0*std::pow(hy_r(k)*theta0,gamma);
        }
        {
          int  k0       = hs+nz-1;
          int  k        = k0+1+kk;
          real rho0     = hy_r(k0);
          real theta0   = hy_t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          hy_r(k) = std::pow( rho0_gm1 - grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          hy_t(k) = theta0;
          hy_p(k) = C0*std::pow(hy_r(k)*theta0,gamma);
        }
      });

      auto create_immersed_proportion_halos = [] (core::Coupler &coupler) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;;
        auto nz     = coupler.get_nz  ();
        auto ny     = coupler.get_ny  ();
        auto nx     = coupler.get_nx  ();
        auto bc_z   = coupler.get_option<std::string>("bc_z","solid_wall");
        auto &dm    = coupler.get_data_manager_readwrite();
        if (!dm.entry_exists("dycore_immersed_proportion_halos")) {
          auto immersed_prop = dm.get<real,3>("immersed_proportion");
          core::MultiField<real,3> fields;
          fields.add_field( immersed_prop  );
          auto fields_halos = coupler.create_and_exchange_halos( fields , hs );
          dm.register_and_allocate<real>("dycore_immersed_proportion_halos","",{nz+2*hs,ny+2*hs,nx+2*hs},
                                         {"z_halod","y_halod","x_halod"});
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) , YAKL_LAMBDA (int kk, int j, int i) {
            fields_halos(0,      kk,j,i) = 1;
            fields_halos(0,hs+nz+kk,j,i) = 1;
          });
          fields_halos.get_field(0).deep_copy_to( dm.get<real,3>("dycore_immersed_proportion_halos") );

          {
            int hsnew = 2;
            dm.register_and_allocate<bool>("dycore_any_immersed2","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed2");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              YAKL_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii)) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
          {
            int hsnew = 4;
            dm.register_and_allocate<bool>("dycore_any_immersed4","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed4");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              YAKL_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii)) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
          {
            int hsnew = 6;
            dm.register_and_allocate<bool>("dycore_any_immersed6","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed6");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              YAKL_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii)) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
          {
            int hsnew = 8;
            dm.register_and_allocate<bool>("dycore_any_immersed8","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed8");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              YAKL_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii)) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
          {
            int hsnew = 10;
            dm.register_and_allocate<bool>("dycore_any_immersed10","",{nz,ny,nx},{"z","y","x"});
            auto any_immersed = dm.get<bool,3>("dycore_any_immersed10");
            auto fields_halos_larger = coupler.create_and_exchange_halos( fields , hsnew );
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hsnew,ny+2*hsnew,nx+2*hsnew) ,
                                              YAKL_LAMBDA (int kk, int j, int i) {
              fields_halos_larger(0,         kk,j,i) = 0;
              fields_halos_larger(0,hsnew+nz+kk,j,i) = 0;
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
              any_immersed(k,j,i) = false;
              for (int kk=0; kk < hsnew*2+1; kk++) {
                for (int jj=0; jj < hsnew*2+1; jj++) {
                  for (int ii=0; ii < hsnew*2+1; ii++) {
                    if (fields_halos_larger(0,k+kk,j+jj,i+ii)) any_immersed(k,j,i) = true;
                  }
                }
              }
            });
          }
        }
      };

      auto compute_hydrostasis_edges = [] (core::Coupler &coupler) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;;
        auto nz   = coupler.get_nz  ();
        auto ny   = coupler.get_ny  ();
        auto nx   = coupler.get_nx  ();
        auto &dm  = coupler.get_data_manager_readwrite();
        if (! dm.entry_exists("hy_dens_edges"    )) dm.register_and_allocate<real>("hy_dens_edges"    ,"",{nz+1});
        if (! dm.entry_exists("hy_theta_edges"   )) dm.register_and_allocate<real>("hy_theta_edges"   ,"",{nz+1});
        if (! dm.entry_exists("hy_pressure_edges")) dm.register_and_allocate<real>("hy_pressure_edges","",{nz+1});
        auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"    );
        auto hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"   );
        auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
        auto hy_dens_edges     = dm.get<real      ,1>("hy_dens_edges"    );
        auto hy_theta_edges    = dm.get<real      ,1>("hy_theta_edges"   );
        auto hy_pressure_edges = dm.get<real      ,1>("hy_pressure_edges");
        if (ord < 5) {
          parallel_for( YAKL_AUTO_LABEL() , nz+1 , YAKL_LAMBDA (int k) {
            hy_dens_edges(k) = std::exp( 0.5_fp*std::log(hy_dens_cells(hs+k-1)) +
                                         0.5_fp*std::log(hy_dens_cells(hs+k  )) );
            hy_theta_edges(k) = 0.5_fp*hy_theta_cells(hs+k-1) +
                                0.5_fp*hy_theta_cells(hs+k  ) ;
            hy_pressure_edges(k) = std::exp( 0.5_fp*std::log(hy_pressure_cells(hs+k-1)) +
                                             0.5_fp*std::log(hy_pressure_cells(hs+k  )) );
          });
        } else {
          parallel_for( YAKL_AUTO_LABEL() , nz+1 , YAKL_LAMBDA (int k) {
            hy_dens_edges(k) = std::exp( -1./12.*std::log(hy_dens_cells(hs+k-2)) +
                                          7./12.*std::log(hy_dens_cells(hs+k-1)) +
                                          7./12.*std::log(hy_dens_cells(hs+k  )) +
                                         -1./12.*std::log(hy_dens_cells(hs+k+1)) );
            hy_theta_edges(k) = -1./12.*hy_theta_cells(hs+k-2) +
                                 7./12.*hy_theta_cells(hs+k-1) +
                                 7./12.*hy_theta_cells(hs+k  ) +
                                -1./12.*hy_theta_cells(hs+k+1);
            hy_pressure_edges(k) = std::exp( -1./12.*std::log(hy_pressure_cells(hs+k-2)) +
                                              7./12.*std::log(hy_pressure_cells(hs+k-1)) +
                                              7./12.*std::log(hy_pressure_cells(hs+k  )) +
                                             -1./12.*std::log(hy_pressure_cells(hs+k+1)) );
          });
        }
      };

      create_immersed_proportion_halos( coupler );
      compute_hydrostasis_edges       ( coupler );

      // These are needed for a proper restart
      coupler.register_output_variable<real>( "immersed_proportion" , core::Coupler::DIMS_3D      );
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto i_beg = coupler.get_i_beg();
        auto j_beg = coupler.get_j_beg();
        auto nz    = coupler.get_nz();
        auto ny    = coupler.get_ny();
        auto nx    = coupler.get_nx();
        nc.redef();
        nc.create_dim( "z_halo" , coupler.get_nz()+2*hs );
        nc.create_var<real>( "hy_dens_cells"     , {"z_halo"});
        nc.create_var<real>( "hy_theta_cells"    , {"z_halo"});
        nc.create_var<real>( "hy_pressure_cells" , {"z_halo"});
        nc.create_var<real>( "theta"             , {"z","y","x"});
        nc.create_var<real>( "pressure_pert"     , {"z","y","x"});
        nc.enddef();
        nc.begin_indep_data();
        auto &dm = coupler.get_data_manager_readonly();
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_dens_cells"    ) , "hy_dens_cells"     );
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_theta_cells"   ) , "hy_theta_cells"    );
        if (coupler.is_mainproc()) nc.write( dm.get<real const,1>("hy_pressure_cells") , "hy_pressure_cells" );
        nc.end_indep_data();
        real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
        real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
        convert_coupler_to_dynamics( coupler , state , tracers );
        std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg};
        using yakl::componentwise::operator/;
        real3d data("data",nz,ny,nx);
        auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          real r = state(idR,hs+k,hs+j,hs+i);
          real p = state(idP,hs+k,hs+j,hs+i)+hy_pressure_cells(hs+k);
          data(k,j,i) = std::pow( p/C0 , 1._fp/gamma ) / r;
        });
        nc.write_all(data,"theta",start_3d);
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = state(idP,hs+k,hs+j,hs+i);
        });
        nc.write_all(data,"pressure_pert",start_3d);
      } );
      coupler.register_overwrite_with_restart_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto &dm = coupler.get_data_manager_readwrite();
        nc.read_all(dm.get<real,1>("hy_dens_cells"    ),"hy_dens_cells"    ,{0});
        nc.read_all(dm.get<real,1>("hy_theta_cells"   ),"hy_theta_cells"   ,{0});
        nc.read_all(dm.get<real,1>("hy_pressure_cells"),"hy_pressure_cells",{0});
        create_immersed_proportion_halos( coupler );
        compute_hydrostasis_edges       ( coupler );
      } );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("init");
      #endif
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst4d    state   ,
                                      realConst4d    tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("convert_dynamics_to_coupler");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
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
      auto dm_rho_d = dm.get<real,3>("density_dry");
      auto dm_uvel  = dm.get<real,3>("uvel"       );
      auto dm_vvel  = dm.get<real,3>("vvel"       );
      auto dm_wvel  = dm.get<real,3>("wvel"       );
      auto dm_temp  = dm.get<real,3>("temp"       );
      auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
      auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      core::MultiField<real,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real,3>(tracer_names[tr]) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real rho = state(idR,hs+k,hs+j,hs+i);
        real sm = 1;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) sm += tracers(tr,hs+k,hs+j,hs+i); }
        real rho_d = rho / sm;
        real press = state(idP,hs+k,hs+j,hs+i) + hy_pressure_cells(hs+k);
        real rho_v = tracers(idWV,hs+k,hs+j,hs+i)*rho_d;
        real temp = press / ( rho_d * R_d + rho_v * R_v );
        dm_rho_d(k,j,i) = rho_d;
        dm_uvel (k,j,i) = state(idU,hs+k,hs+j,hs+i);
        dm_vvel (k,j,i) = state(idV,hs+k,hs+j,hs+i);
        dm_wvel (k,j,i) = state(idW,hs+k,hs+j,hs+i);
        dm_temp (k,j,i) = temp;
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i) = tracers(tr,hs+k,hs+j,hs+i)*rho_d; }
      });
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("convert_dynamics_to_coupler");
      #endif
    }



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_start("convert_coupler_to_dynamics");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
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
      auto dm_rho_d = dm.get<real const,3>("density_dry");
      auto dm_uvel  = dm.get<real const,3>("uvel"       );
      auto dm_vvel  = dm.get<real const,3>("vvel"       );
      auto dm_wvel  = dm.get<real const,3>("wvel"       );
      auto dm_temp  = dm.get<real const,3>("temp"       );
      auto tracer_adds_mass  = dm.get<bool const,1>("tracer_adds_mass");
      auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
      core::MultiField<real const,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real const,3>(tracer_names[tr]) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i);
        real rho   = rho_d;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i); }
        real temp  = dm_temp(k,j,i);
        real rho_v = dm_tracers(idWV,k,j,i);
        real press = rho_d * R_d * temp + rho_v * R_v * temp;
        state(idR,hs+k,hs+j,hs+i) = rho;
        state(idU,hs+k,hs+j,hs+i) = dm_uvel(k,j,i);
        state(idV,hs+k,hs+j,hs+i) = dm_vvel(k,j,i);
        state(idW,hs+k,hs+j,hs+i) = dm_wvel(k,j,i);
        state(idP,hs+k,hs+j,hs+i) = press - hy_pressure_cells(hs+k);
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i) = dm_tracers(tr,k,j,i)/rho_d; }
      });
      #ifdef YAKL_AUTO_PROFILE
        coupler.get_parallel_comm().barrier();
        yakl::timer_stop("convert_coupler_to_dynamics");
      #endif
    }


  };

}


