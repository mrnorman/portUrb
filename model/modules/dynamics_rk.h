
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
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
  // You'll notice the dimensions are nz,ny,nx.

  struct Dynamics_Euler_Stratified_WenoFV {
    // Order of accuracy (numerical convergence for smooth flows) for the dynamical core
    #ifndef PORTURB_ORD
      yakl::index_t static constexpr ord = 9;
    #else
      yakl::index_t static constexpr ord = PORTURB_ORD;
    #endif
    int static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature


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
      real cfl = coupler.get_option<real>("cfl",0.15);
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
    //   real cfl = coupler.get_option<real>("cfl",0.15);
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
      convert_coupler_to_dynamics( coupler , state , tracers );
      real dt_dyn = compute_time_step( coupler );
      int ncycles = (int) std::ceil( dt_phys / dt_dyn );
      dt_dyn = dt_phys / ncycles;
      // // SSPRK33
      // SArray<real,2,3,3> ssprk33_A;
      // SArray<real,1,3> ssprk33_b;
      // ssprk33_A = 0;
      // ssprk33_A(1,0) = 1;
      // ssprk33_A(2,0) = 1./4.;
      // ssprk33_A(2,1) = 1./4.;
      // ssprk33_b(0)   = 1./6.;
      // ssprk33_b(1)   = 1./6.;
      // ssprk33_b(2)   = 2./3.;
      for (int icycle = 0; icycle < ncycles; icycle++) {
        // time_step_generic_rk(coupler,A,b,state,tracers,dt_dyn);
        time_step_rk_3_3(coupler,state,tracers,dt_dyn);
      }
      convert_dynamics_to_coupler( coupler , state , tracers );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step");
      #endif
    }



    template <yakl::index_t N>
    void time_step_generic_rk( core::Coupler            & coupler ,
                               SArray<real,2,N,N> const & A       ,
                               SArray<real,1,N>   const & b       ,
                               real4d             const & state   ,
                               real4d             const & tracers ,
                               real                       dt      ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      real5d state_rk_tend  ("state_rk_tend"  ,N,num_state  ,nz,ny,nx);
      real5d tracers_rk_tend("tracers_rk_tend",N,num_tracers,nz,ny,nx);
      real4d state_tmp  ("state_tmp"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      real4d tracers_tmp("tracers_tmp",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      enforce_immersed_boundaries( coupler , state , tracers , dt );
      for (int is = 0; is < N; is++) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          for (int l=0; l < num_state  ; l++) {
            state_tmp  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i);
            for (int l2=0; l2 <= is-1; l2++) { state_tmp  (l,hs+k,hs+j,hs+i) += dt*A(is,l2)*state_rk_tend  (l2,l,k,j,i); }
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_tmp(l,hs+k,hs+j,hs+i) = tracers(l,hs+k,hs+j,hs+i);
            for (int l2=0; l2 <= is-1; l2++) { tracers_tmp(l,hs+k,hs+j,hs+i) += dt*A(is,l2)*tracers_rk_tend(l2,l,k,j,i); }
          }
        });
        auto state_tend   = state_rk_tend  .slice<4>(is,0,0,0,0);
        auto tracers_tend = tracers_rk_tend.slice<4>(is,0,0,0,0);
        enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp , dt );
        compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend);
      }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        for (int is = 0; is < N; is++) {
          for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) += dt*b(is)*state_rk_tend  (is,l,k,j,i); }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) += dt*b(is)*tracers_rk_tend(is,l,k,j,i); }
        }
      });
      enforce_immersed_boundaries( coupler , state , tracers , dt );
    }



    void time_step_sspab3( core::Coupler & coupler ,
                           real4d const  & state   ,
                           real4d const  & tracers ,
                           real            dt_dyn  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_sspab3");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto &dm         = coupler.get_data_manager_readwrite();
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");

      if ( ! dm.entry_exists("state_tend_register") ) {
        dm.register_and_allocate<real>("state_register"  ,"",{3,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs});
        dm.register_and_allocate<real>("tracers_register","",{3,num_tracers,nz+2*hs,ny+2*hs,nx+2*hs});
        dm.register_and_allocate<real>("state_tend_register"  ,"",{3,num_state  ,nz,ny,nx});
        dm.register_and_allocate<real>("tracers_tend_register","",{3,num_tracers,nz,ny,nx});
        dm.get<real,5>("state_tend_register"  ) = 0;
        dm.get<real,5>("tracers_tend_register") = 0;
        auto state_register   = dm.get<real,5>("state_register"  );
        auto tracers_register = dm.get<real,5>("tracers_register");
        state  .deep_copy_to(state_register  .slice<4>(0,0,0,0,0));
        state  .deep_copy_to(state_register  .slice<4>(1,0,0,0,0));
        state  .deep_copy_to(state_register  .slice<4>(2,0,0,0,0));
        tracers.deep_copy_to(tracers_register.slice<4>(0,0,0,0,0));
        tracers.deep_copy_to(tracers_register.slice<4>(1,0,0,0,0));
        tracers.deep_copy_to(tracers_register.slice<4>(2,0,0,0,0));
        coupler.set_option<int>("ab3_current_index",2);
      }
      auto state_register        = dm.get<real,5>("state_register"       );
      auto tracers_register      = dm.get<real,5>("tracers_register"     );
      auto state_tend_register   = dm.get<real,5>("state_tend_register"  );
      auto tracers_tend_register = dm.get<real,5>("tracers_tend_register");
      auto ind_n   = coupler.get_option<int>("ab3_current_index");
      auto ind_nm1 = ind_n-1;   if (ind_nm1 < 0) ind_nm1 += 3;
      auto ind_nm2 = ind_n-2;   if (ind_nm2 < 0) ind_nm2 += 3;

      enforce_immersed_boundaries( coupler , state , tracers , dt_dyn );

      auto state_tend_nm2   = state_tend_register  .slice<4>(ind_nm2,0,0,0,0);
      auto tracers_tend_nm2 = tracers_tend_register.slice<4>(ind_nm2,0,0,0,0);
      auto state_tend_nm1   = state_tend_register  .slice<4>(ind_nm1,0,0,0,0);
      auto tracers_tend_nm1 = tracers_tend_register.slice<4>(ind_nm1,0,0,0,0);
      auto state_tend_n     = state_tend_register  .slice<4>(ind_n  ,0,0,0,0);
      auto tracers_tend_n   = tracers_tend_register.slice<4>(ind_n  ,0,0,0,0);

      auto state_nm2   = state_register  .slice<4>(ind_nm2,0,0,0,0);
      auto tracers_nm2 = tracers_register.slice<4>(ind_nm2,0,0,0,0);
      auto state_nm1   = state_register  .slice<4>(ind_nm1,0,0,0,0);
      auto tracers_nm1 = tracers_register.slice<4>(ind_nm1,0,0,0,0);
      auto state_n     = state_register  .slice<4>(ind_n  ,0,0,0,0);
      auto tracers_n   = tracers_register.slice<4>(ind_n  ,0,0,0,0);
      state  .deep_copy_to(state_n  );
      tracers.deep_copy_to(tracers_n);

      compute_tendencies(coupler,state,state_tend_n,tracers,tracers_tend_n);
      
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state  (l,hs+k,hs+j,hs+i) =  1.908535476882378 * state_n  (l,hs+k,hs+j,hs+i)      +
                                      -1.334951446162515 * state_nm1(l,hs+k,hs+j,hs+i)      +
                                       0.426415969280137 * state_nm2(l,hs+k,hs+j,hs+i)      +
                                       1.502575553858997 * dt_dyn * state_tend_n  (l,k,j,i) +
                                      -1.654746338401493 * dt_dyn * state_tend_nm1(l,k,j,i) +
                                       0.670051276940255 * dt_dyn * state_tend_nm2(l,k,j,i);
        } else {
          l -= num_state;
          tracers(l,hs+k,hs+j,hs+i) =  1.908535476882378 * tracers_n  (l,hs+k,hs+j,hs+i)      +
                                      -1.334951446162515 * tracers_nm1(l,hs+k,hs+j,hs+i)      +
                                       0.426415969280137 * tracers_nm2(l,hs+k,hs+j,hs+i)      +
                                       1.502575553858997 * dt_dyn * tracers_tend_n  (l,k,j,i) +
                                      -1.654746338401493 * dt_dyn * tracers_tend_nm1(l,k,j,i) +
                                       0.670051276940255 * dt_dyn * tracers_tend_nm2(l,k,j,i);
          if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers(l,hs+k,hs+j,hs+i) );
        }
      });

      ind_n++;   if (ind_n > 2) ind_n -= 3;
      coupler.set_option<int>("ab3_current_index",ind_n);

      enforce_immersed_boundaries( coupler , state , tracers , dt_dyn );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_sspab3");
      #endif
    }



    void time_step_rk_3_3( core::Coupler & coupler ,
                          real4d const  & state   ,
                          real4d const  & tracers ,
                          real            dt_dyn  ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_rk_3_3");
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
      compute_tendencies(coupler,state,state_tend,tracers,tracers_tend);
      // Apply tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tmp  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i) + dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers_tmp(l,hs+k,hs+j,hs+i) = tracers(l,hs+k,hs+j,hs+i) + dt_dyn * tracers_tend(l,k,j,i);
        }
      });

      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp , dt_dyn/2 );

      //////////////
      // Stage 2
      //////////////
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend);
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
        }
      });

      enforce_immersed_boundaries( coupler , state_tmp , tracers_tmp , dt_dyn/2 );

      //////////////
      // Stage 3
      //////////////
      compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend);
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
        yakl::timer_stop("time_step_rk_3_3");
      #endif
    }


    void enforce_immersed_boundaries( core::Coupler const & coupler ,
                                      real4d        const & state   ,
                                      real4d        const & tracers ,
                                      real                  dt      ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("enforce_immersed_boundaries");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers    = coupler.get_num_tracers();
      auto nx             = coupler.get_nx();
      auto ny             = coupler.get_ny();
      auto nz             = coupler.get_nz();
      auto immersed_power = coupler.get_option<real>("immersed_power",4);
      auto &dm            = coupler.get_data_manager_readonly();
      auto hy_dens_cells  = dm.get<float const,1>("hy_dens_cells" ); // Hydrostatic density
      auto hy_theta_cells = dm.get<float const,1>("hy_theta_cells"); // Hydrostatic potential temperature
      auto immersed_prop  = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion

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
        // density*theta
        {
          auto &var = state(idT,hs+k,hs+j,hs+i);
          real  target = hy_dens_cells(hs+k)*hy_theta_cells(hs+k);
          var = var + (target - var)*mult;
        }
        // Tracers
        for (int tr=0; tr < num_tracers; tr++) {
          auto &var = tracers(tr,hs+k,hs+j,hs+i);
          real  target = 0;
          var = var + (target - var)*mult;
        }
      });
      #ifdef YAKL_AUTO_PROFILE
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



    void compute_tendencies( core::Coupler       & coupler      ,
                             real4d        const & state        ,
                             real4d        const & state_tend   ,
                             real4d        const & tracers      ,
                             real4d        const & tracers_tend ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("compute_tendencies");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx                = coupler.get_nx();    // Proces-local number of cells
      auto  ny                = coupler.get_ny();    // Proces-local number of cells
      auto  nz                = coupler.get_nz();    // Total vertical cells
      auto  dx                = coupler.get_dx();    // grid spacing
      auto  dy                = coupler.get_dy();    // grid spacing
      auto  dz                = coupler.get_dz();    // grid spacing
      auto  sim2d             = coupler.is_sim2d();  // Is this a 2-D simulation?
      auto  enable_gravity    = coupler.get_option<bool>("enable_gravity",true);
      auto  C0                = coupler.get_option<real>("C0"     );  // pressure = C0*pow(rho*theta,gamma)
      auto  grav              = coupler.get_option<real>("grav"   );  // Gravity
      auto  gamma             = coupler.get_option<real>("gamma_d");  // cp_dry / cv_dry (about 1.4)
      auto  latitude          = coupler.get_option<real>("latitude",0); // For coriolis
      auto  num_tracers       = coupler.get_num_tracers();            // Number of tracers
      auto  &dm               = coupler.get_data_manager_readonly();  // Grab read-only data manager
      auto  tracer_positive   = dm.get<bool const,1>("tracer_positive"          ); // Is a tracer positive-definite?
      auto  immersed_prop     = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
      auto  any_immersed      = dm.get<bool const,3>("dycore_any_immersed10"    ); // Are any immersed in 3-D halo?
      auto  hy_dens_cells     = dm.get<float const,1>("hy_dens_cells"            ); // Hydrostatic density
      auto  hy_theta_cells    = dm.get<float const,1>("hy_theta_cells"           ); // Hydrostatic potential temperature
      auto  hy_dens_edges     = dm.get<float const,1>("hy_dens_edges"            ); // Hydrostatic density
      auto  hy_theta_edges    = dm.get<float const,1>("hy_theta_edges"           ); // Hydrostatic potential temperature
      auto  hy_pressure_cells = dm.get<float const,1>("hy_pressure_cells"        ); // Hydrostatic pressure
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real r_dz = 1./dz; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

      SArray<real,2,ord,2> s2e;
      SArray<real,2,ord,ord> s2c;
      SArray<real,2,ord,2> c2e;
      TransformMatrices::sten_to_coefs(s2c);
      TransformMatrices::coefs_to_gll_lower(c2e);
      using yakl::intrinsics::matmul_cr;
      s2e = matmul_cr( c2e , s2c );

      float4d state_loc  ("stateloc"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
      float4d tracers_loc("tracersloc",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
      float3d pressure   ("pressure"              ,nz+2*hs,ny+2*hs,nx+2*hs); // Holds pressure perturbation

      // Compute pressure
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        pressure(hs+k,hs+j,hs+i) = C0*std::pow(state(idT,hs+k,hs+j,hs+i),gamma) - hy_pressure_cells(hs+k);
        real r_r = 1._fp / state(idR,hs+k,hs+j,hs+i);
        state_loc(idR,hs+k,hs+j,hs+i) = state(idR,hs+k,hs+j,hs+i);
        for (int l=1; l < num_state  ; l++) { state_loc  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i)*r_r; }
        for (int l=0; l < num_tracers; l++) { tracers_loc(l,hs+k,hs+j,hs+i) = tracers(l,hs+k,hs+j,hs+i)*r_r; }
        state_loc(idR,hs+k,hs+j,hs+i) -= hy_dens_cells (hs+k);
        state_loc(idT,hs+k,hs+j,hs+i) -= hy_theta_cells(hs+k);
      });

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      {
        core::MultiField<float,3> fields;
        for (int l=0; l < num_state  ; l++) { fields.add_field( state_loc  .slice<3>(l,0,0,0) ); }
        for (int l=0; l < num_tracers; l++) { fields.add_field( tracers_loc.slice<3>(l,0,0,0) ); }
        fields.add_field( pressure );
        if (ord > 1) coupler.halo_exchange_x( fields , hs );
        if (ord > 1) coupler.halo_exchange_y( fields , hs );
        halo_boundary_conditions( coupler , state_loc , tracers_loc , pressure );
      }

      typedef limiter::WenoLimiter<float,ord> Limiter;

      // Create arrays to hold cell interface interpolations
      float5d state_limits_x   ("state_limits_x"   ,2,num_state  ,nz,ny,nx+1);
      float5d state_limits_y   ("state_limits_y"   ,2,num_state  ,nz,ny+1,nx);
      float5d state_limits_z   ("state_limits_z"   ,2,num_state  ,nz+1,ny,nx);
      float4d pressure_limits_x("pressure_limits_x",2            ,nz,ny,nx+1);
      float4d pressure_limits_y("pressure_limits_y",2            ,nz,ny+1,nx);
      float4d pressure_limits_z("pressure_limits_z",2            ,nz+1,ny,nx);
      float5d tracers_limits_x ("tracers_limits_x" ,2,num_tracers,nz,ny,nx+1);
      float5d tracers_limits_y ("tracers_limits_y" ,2,num_tracers,nz,ny+1,nx);
      float5d tracers_limits_z ("tracers_limits_z" ,2,num_tracers,nz+1,ny,nx);
      
      // Aggregate fields for interpolation
      core::MultiField<float,3> fields;
      core::MultiField<float,3> lim_x_L, lim_x_R, lim_y_L, lim_y_R, lim_z_L, lim_z_R;
      int idP = 5;
      for (int l=0; l < num_state; l++) {
        fields .add_field(state_loc     .slice<3>(  l,0,0,0));
        lim_x_L.add_field(state_limits_x.slice<3>(0,l,0,0,0));
        lim_x_R.add_field(state_limits_x.slice<3>(1,l,0,0,0));
        lim_y_L.add_field(state_limits_y.slice<3>(0,l,0,0,0));
        lim_y_R.add_field(state_limits_y.slice<3>(1,l,0,0,0));
        lim_z_L.add_field(state_limits_z.slice<3>(0,l,0,0,0));
        lim_z_R.add_field(state_limits_z.slice<3>(1,l,0,0,0));
      }
      fields .add_field(pressure         .slice<3>(  0,0,0));
      lim_x_L.add_field(pressure_limits_x.slice<3>(0,0,0,0));
      lim_x_R.add_field(pressure_limits_x.slice<3>(1,0,0,0));
      lim_y_L.add_field(pressure_limits_y.slice<3>(0,0,0,0));
      lim_y_R.add_field(pressure_limits_y.slice<3>(1,0,0,0));
      lim_z_L.add_field(pressure_limits_z.slice<3>(0,0,0,0));
      lim_z_R.add_field(pressure_limits_z.slice<3>(1,0,0,0));
      for (int l=0; l < num_tracers; l++) {
        fields .add_field(tracers_loc     .slice<3>(  l,0,0,0));
        lim_x_L.add_field(tracers_limits_x.slice<3>(0,l,0,0,0));
        lim_x_R.add_field(tracers_limits_x.slice<3>(1,l,0,0,0));
        lim_y_L.add_field(tracers_limits_y.slice<3>(0,l,0,0,0));
        lim_y_R.add_field(tracers_limits_y.slice<3>(1,l,0,0,0));
        lim_z_L.add_field(tracers_limits_z.slice<3>(0,l,0,0,0));
        lim_z_R.add_field(tracers_limits_z.slice<3>(1,l,0,0,0));
      }

      // X-direction interpolation
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(fields.size(),nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        SArray<float,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        for (int ii=0; ii<ord; ii++) {
          immersed(ii) = immersed_prop(hs+k,hs+j,i+ii) > 0;
          stencil (ii) = fields     (l,hs+k,hs+j,i+ii);
        }
        if (l == idV || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        bool map = ! any_immersed(k,j,i);
        Limiter::compute_limited_edges( stencil , lim_x_R(l,k,j,i) , lim_x_L(l,k,j,i+1) ,
                                        { map , immersed(hs-1) , immersed(hs+1)} );
      });

      // Y-direction interpolation
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(fields.size(),nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        SArray<float,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        for (int jj=0; jj<ord; jj++) {
          immersed(jj) = immersed_prop(hs+k,j+jj,hs+i) > 0;
          stencil (jj) = fields     (l,hs+k,j+jj,hs+i);
        }
        if (l == idU || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        bool map = ! any_immersed(k,j,i);
        Limiter::compute_limited_edges( stencil , lim_y_R(l,k,j,i) , lim_y_L(l,k,j+1,i) ,
                                        { map , immersed(hs-1) , immersed(hs+1)} );
      });

      // Z-direction interpolation
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(fields.size(),nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        SArray<float,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        for (int kk=0; kk<ord; kk++) {
          immersed(kk) = immersed_prop(k+kk,hs+j,hs+i) > 0;
          stencil (kk) = fields     (l,k+kk,hs+j,hs+i);
        }
        if (l == idU || l == idV || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        bool map = ! any_immersed(k,j,i);
        Limiter::compute_limited_edges( stencil , lim_z_R(l,k,j,i) , lim_z_L(l,k+1,j,i) ,
                                        { map , immersed(hs-1) , immersed(hs+1)} );
      });

      // Perform periodic horizontal exchange of cell-edge data, and implement vertical boundary conditions
      edge_exchange( coupler , state_limits_x , tracers_limits_x , pressure_limits_x ,
                               state_limits_y , tracers_limits_y , pressure_limits_y ,
                               state_limits_z , tracers_limits_z , pressure_limits_z );

      // To save on space, slice the limits arrays to store single-valued interface fluxes
      using yakl::COLON;
      auto state_flux_x   = state_limits_x  .slice<4>(0,COLON,COLON,COLON,COLON);
      auto state_flux_y   = state_limits_y  .slice<4>(0,COLON,COLON,COLON,COLON);
      auto state_flux_z   = state_limits_z  .slice<4>(0,COLON,COLON,COLON,COLON);
      auto tracers_flux_x = tracers_limits_x.slice<4>(0,COLON,COLON,COLON,COLON);
      auto tracers_flux_y = tracers_limits_y.slice<4>(0,COLON,COLON,COLON,COLON);
      auto tracers_flux_z = tracers_limits_z.slice<4>(0,COLON,COLON,COLON,COLON);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      // Speed of sound and its reciprocal. Using a constant speed of sound for upwinding
      float constexpr cs   = 350.;
      float constexpr r_cs = 1./cs;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i) {
        if (j < ny && k < nz) {
          state_limits_x(0,idR,k,j,i) += hy_dens_cells (hs+k);
          state_limits_x(1,idR,k,j,i) += hy_dens_cells (hs+k);
          state_limits_x(0,idT,k,j,i) += hy_theta_cells(hs+k);
          state_limits_x(1,idT,k,j,i) += hy_theta_cells(hs+k);
          // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
          float r_L  = state_limits_x  (0,idR,k,j,i)    ;   float r_R  = state_limits_x   (1,idR,k,j,i)    ;
          float ru_L = state_limits_x  (0,idU,k,j,i)*r_L;   float ru_R = state_limits_x   (1,idU,k,j,i)*r_R;
          float p_L  = pressure_limits_x(0   ,k,j,i)    ;   float p_R  = pressure_limits_x(1    ,k,j,i)    ;
          float p_upw  = 0.5_fp*(p_L  + p_R  - cs*(ru_R-ru_L)   );
          float ru_upw = 0.5_fp*(ru_L + ru_R -    (p_R -p_L )/cs);
          // Advectively upwind everything else
          int ind = ru_upw > 0 ? 0 : 1;
          state_flux_x(idR,k,j,i) = ru_upw;
          state_flux_x(idU,k,j,i) = ru_upw*state_limits_x(ind,idU,k,j,i) + p_upw;
          state_flux_x(idV,k,j,i) = ru_upw*state_limits_x(ind,idV,k,j,i);
          state_flux_x(idW,k,j,i) = ru_upw*state_limits_x(ind,idW,k,j,i);
          state_flux_x(idT,k,j,i) = ru_upw*state_limits_x(ind,idT,k,j,i);
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_x(tr,k,j,i) = ru_upw*tracers_limits_x(ind,tr,k,j,i);
          }
        }
        if (i < nx && k < nz && !sim2d) {
          state_limits_y(0,idR,k,j,i) += hy_dens_cells (hs+k);
          state_limits_y(1,idR,k,j,i) += hy_dens_cells (hs+k);
          state_limits_y(0,idT,k,j,i) += hy_theta_cells(hs+k);
          state_limits_y(1,idT,k,j,i) += hy_theta_cells(hs+k);
          // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
          float r_L  = state_limits_y  (0,idR,k,j,i)    ;   float r_R  = state_limits_y   (1,idR,k,j,i)    ;
          float rv_L = state_limits_y  (0,idV,k,j,i)*r_L;   float rv_R = state_limits_y   (1,idV,k,j,i)*r_R;
          float p_L  = pressure_limits_y(0   ,k,j,i)    ;   float p_R  = pressure_limits_y(1    ,k,j,i)    ;
          float p_upw  = 0.5_fp*(p_L  + p_R  - cs*(rv_R-rv_L)   );
          float rv_upw = 0.5_fp*(rv_L + rv_R -    (p_R -p_L )/cs);
          // Advectively upwind everything else
          int ind = rv_upw > 0 ? 0 : 1;
          state_flux_y(idR,k,j,i) = rv_upw;
          state_flux_y(idU,k,j,i) = rv_upw*state_limits_y(ind,idU,k,j,i);
          state_flux_y(idV,k,j,i) = rv_upw*state_limits_y(ind,idV,k,j,i) + p_upw;
          state_flux_y(idW,k,j,i) = rv_upw*state_limits_y(ind,idW,k,j,i);
          state_flux_y(idT,k,j,i) = rv_upw*state_limits_y(ind,idT,k,j,i);
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_y(tr,k,j,i) = rv_upw*tracers_limits_y(ind,tr,k,j,i);
          }
        }
        if (i < nx && j < ny) {
          state_limits_z(0,idR,k,j,i) += hy_dens_edges (k);
          state_limits_z(1,idR,k,j,i) += hy_dens_edges (k);
          state_limits_z(0,idT,k,j,i) += hy_theta_edges(k);
          state_limits_z(1,idT,k,j,i) += hy_theta_edges(k);
          // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
          float r_L  = state_limits_z  (0,idR,k,j,i)    ;   float r_R  = state_limits_z   (1,idR,k,j,i)    ;
          float rw_L = state_limits_z  (0,idW,k,j,i)*r_L;   float rw_R = state_limits_z   (1,idW,k,j,i)*r_R;
          float p_L  = pressure_limits_z(0   ,k,j,i)    ;   float p_R  = pressure_limits_z(1    ,k,j,i)    ;
          float p_upw  = 0.5_fp*(p_L  + p_R  - cs*(rw_R-rw_L)   );
          float rw_upw = 0.5_fp*(rw_L + rw_R -    (p_R -p_L )/cs);
          // Advectively upwind everything else
          int ind = rw_upw > 0 ? 0 : 1;
          state_flux_z(idR,k,j,i) = rw_upw;
          state_flux_z(idU,k,j,i) = rw_upw*state_limits_z(ind,idU,k,j,i);
          state_flux_z(idV,k,j,i) = rw_upw*state_limits_z(ind,idV,k,j,i);
          state_flux_z(idW,k,j,i) = rw_upw*state_limits_z(ind,idW,k,j,i) + p_upw;
          state_flux_z(idT,k,j,i) = rw_upw*state_limits_z(ind,idT,k,j,i);
          for (int tr=0; tr < num_tracers; tr++) {
            tracers_flux_z(tr,k,j,i) = rw_upw*tracers_limits_z(ind,tr,k,j,i);
          }
        }
      });

      // Compute tendencies as the flux divergence + gravity source term + coriolis
      int mx = std::max(num_state,num_tracers);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(mx,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tend(l,k,j,i) = -( state_flux_x(l,k,j,i+1) - state_flux_x(l,k,j,i) ) * r_dx
                                -( state_flux_y(l,k,j+1,i) - state_flux_y(l,k,j,i) ) * r_dy
                                -( state_flux_z(l,k+1,j,i) - state_flux_z(l,k,j,i) ) * r_dz;
          if (l == idV && sim2d) state_tend(l,k,j,i) = 0;
          if (l == idW && enable_gravity) {
            state_tend(l,k,j,i) += -grav*(state(idR,hs+k,hs+j,hs+i) - hy_dens_cells(hs+k));
          }
          if (latitude != 0 && !sim2d && l == idU) state_tend(l,k,j,i) += fcor*state(idV,hs+k,hs+j,hs+i);
          if (latitude != 0 && !sim2d && l == idV) state_tend(l,k,j,i) -= fcor*state(idU,hs+k,hs+j,hs+i);
        }
        if (l < num_tracers) {
          tracers_tend(l,k,j,i) = -( tracers_flux_x(l,k,j,i+1) - tracers_flux_x(l,k,j,i) ) * r_dx
                                  -( tracers_flux_y(l,k,j+1,i) - tracers_flux_y(l,k,j,i) ) * r_dy 
                                  -( tracers_flux_z(l,k+1,j,i) - tracers_flux_z(l,k,j,i) ) * r_dz;
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("compute_tendencies");
      #endif
    }



    void halo_boundary_conditions( core::Coupler const & coupler  ,
                                   float4d       const & state    ,
                                   float4d       const & tracers  ,
                                   float3d       const & pressure ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("halo_boundary_conditions");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx              = coupler.get_nx();
      auto ny              = coupler.get_ny();
      auto nz              = coupler.get_nz();
      auto num_tracers     = coupler.get_num_tracers();
      auto bc_z            = coupler.get_option<std::string>("bc_z","solid_wall");
      auto &dm             = coupler.get_data_manager_readonly();
      auto hy_dens_cells   = dm.get<float const,1>("hy_dens_cells" );
      auto hy_theta_cells  = dm.get<float const,1>("hy_theta_cells");
      auto surface_temp    = dm.get<real const,2>("surface_temp");

      // z-direction BC's
      if (bc_z == "solid_wall") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny,nx) , YAKL_LAMBDA (int kk, int j, int i) {
          state(idR,kk,hs+j,hs+i) = state(idR,hs+0,hs+j,hs+i);
          state(idU,kk,hs+j,hs+i) = state(idU,hs+0,hs+j,hs+i);
          state(idV,kk,hs+j,hs+i) = state(idV,hs+0,hs+j,hs+i);
          state(idW,kk,hs+j,hs+i) = 0;
          state(idT,kk,hs+j,hs+i) = state(idT,hs+0,hs+j,hs+i);
          pressure( kk,hs+j,hs+i) = pressure (hs+0,hs+j,hs+i);
          state(idR,hs+nz+kk,hs+j,hs+i) = state(idR,hs+nz-1,hs+j,hs+i);
          state(idU,hs+nz+kk,hs+j,hs+i) = state(idU,hs+nz-1,hs+j,hs+i);
          state(idV,hs+nz+kk,hs+j,hs+i) = state(idV,hs+nz-1,hs+j,hs+i);
          state(idW,hs+nz+kk,hs+j,hs+i) = 0;
          state(idT,hs+nz+kk,hs+j,hs+i) = state(idT,hs+nz-1,hs+j,hs+i);
          pressure( hs+nz+kk,hs+j,hs+i) = pressure (hs+nz-1,hs+j,hs+i);
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
          state(idT,      kk,hs+j,hs+i) = state(idT,nz+kk,hs+j,hs+i);
          pressure(       kk,hs+j,hs+i) = pressure( nz+kk,hs+j,hs+i);
          state(idR,hs+nz+kk,hs+j,hs+i) = state(idR,hs+kk,hs+j,hs+i);
          state(idU,hs+nz+kk,hs+j,hs+i) = state(idU,hs+kk,hs+j,hs+i);
          state(idV,hs+nz+kk,hs+j,hs+i) = state(idV,hs+kk,hs+j,hs+i);
          state(idW,hs+nz+kk,hs+j,hs+i) = state(idW,hs+kk,hs+j,hs+i);
          state(idT,hs+nz+kk,hs+j,hs+i) = state(idT,hs+kk,hs+j,hs+i);
          pressure( hs+nz+kk,hs+j,hs+i) = pressure( hs+kk,hs+j,hs+i);
          for (int l=0; l < num_tracers; l++) {
            tracers(l,      kk,hs+j,hs+i) = tracers(l,nz+kk,hs+j,hs+i);
            tracers(l,hs+nz+kk,hs+j,hs+i) = tracers(l,hs+kk,hs+j,hs+i);
          }
        });
      } else {
        yakl::yakl_throw("ERROR: Specified invalid bc_z in coupler options");
      }
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("halo_boundary_conditions");
      #endif
    }



    void edge_exchange( core::Coupler const & coupler           ,
                        float5d       const & state_limits_x    ,
                        float5d       const & tracers_limits_x  ,
                        float4d       const & pressure_limits_x ,
                        float5d       const & state_limits_y    ,
                        float5d       const & tracers_limits_y  ,
                        float4d       const & pressure_limits_y ,
                        float5d       const & state_limits_z    ,
                        float5d       const & tracers_limits_z  ,
                        float4d       const & pressure_limits_z ) const {
      #ifdef YAKL_AUTO_PROFILE
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
      auto hy_dens_edges  = dm.get<float const,1>("hy_dens_edges" );
      auto hy_theta_edges = dm.get<float const,1>("hy_theta_edges");
      auto surface_temp   = dm.get<real const,2>("surface_temp"  );
      int npack = num_state + num_tracers+1;

      // x-exchange
      {
        float3d edge_send_buf_W("edge_send_buf_W",npack,nz,ny);
        float3d edge_send_buf_E("edge_send_buf_E",npack,nz,ny);
        float3d edge_recv_buf_W("edge_recv_buf_W",npack,nz,ny);
        float3d edge_recv_buf_E("edge_recv_buf_E",npack,nz,ny);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
          if        (v < num_state) {
            edge_send_buf_W(v,k,j) = state_limits_x  (1,v          ,k,j,0 );
            edge_send_buf_E(v,k,j) = state_limits_x  (0,v          ,k,j,nx);
          } else if (v < num_state + num_tracers) {                    
            edge_send_buf_W(v,k,j) = tracers_limits_x(1,v-num_state,k,j,0 );
            edge_send_buf_E(v,k,j) = tracers_limits_x(0,v-num_state,k,j,nx);
          } else {
            edge_send_buf_W(v,k,j) = pressure_limits_x(1,k,j,0 );
            edge_send_buf_E(v,k,j) = pressure_limits_x(0,k,j,nx);
          }
        });
        coupler.get_parallel_comm().send_receive<float,3>( { {edge_recv_buf_W,neigh(1,0),4} , {edge_recv_buf_E,neigh(1,2),5} } ,
                                                           { {edge_send_buf_W,neigh(1,0),5} , {edge_send_buf_E,neigh(1,2),4} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
          if        (v < num_state) {
            state_limits_x  (0,v          ,k,j,0 ) = edge_recv_buf_W(v,k,j);
            state_limits_x  (1,v          ,k,j,nx) = edge_recv_buf_E(v,k,j);
          } else if (v < num_state + num_tracers) {
            tracers_limits_x(0,v-num_state,k,j,0 ) = edge_recv_buf_W(v,k,j);
            tracers_limits_x(1,v-num_state,k,j,nx) = edge_recv_buf_E(v,k,j);
          } else {
            pressure_limits_x(0,k,j,0 ) = edge_recv_buf_W(v,k,j);
            pressure_limits_x(1,k,j,nx) = edge_recv_buf_E(v,k,j);
          }
        });
      }

      // y-direction exchange
      {
        float3d edge_send_buf_S("edge_send_buf_S",npack,nz,nx);
        float3d edge_send_buf_N("edge_send_buf_N",npack,nz,nx);
        float3d edge_recv_buf_S("edge_recv_buf_S",npack,nz,nx);
        float3d edge_recv_buf_N("edge_recv_buf_N",npack,nz,nx);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if        (v < num_state) {
            edge_send_buf_S(v,k,i) = state_limits_y  (1,v          ,k,0 ,i);
            edge_send_buf_N(v,k,i) = state_limits_y  (0,v          ,k,ny,i);
          } else if (v < num_state + num_tracers) {                    
            edge_send_buf_S(v,k,i) = tracers_limits_y(1,v-num_state,k,0 ,i);
            edge_send_buf_N(v,k,i) = tracers_limits_y(0,v-num_state,k,ny,i);
          } else {
            edge_send_buf_S(v,k,i) = pressure_limits_y(1,k,0 ,i);
            edge_send_buf_N(v,k,i) = pressure_limits_y(0,k,ny,i);
          }
        });
        coupler.get_parallel_comm().send_receive<float,3>( { {edge_recv_buf_S,neigh(0,1),6} , {edge_recv_buf_N,neigh(2,1),7} } ,
                                                           { {edge_send_buf_S,neigh(0,1),7} , {edge_send_buf_N,neigh(2,1),6} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if        (v < num_state) {
            state_limits_y  (0,v          ,k,0 ,i) = edge_recv_buf_S(v,k,i);
            state_limits_y  (1,v          ,k,ny,i) = edge_recv_buf_N(v,k,i);
          } else if (v < num_state + num_tracers) {
            tracers_limits_y(0,v-num_state,k,0 ,i) = edge_recv_buf_S(v,k,i);
            tracers_limits_y(1,v-num_state,k,ny,i) = edge_recv_buf_N(v,k,i);
          } else {
            pressure_limits_y(0,k,0 ,i) = edge_recv_buf_S(v,k,i);
            pressure_limits_y(1,k,ny,i) = edge_recv_buf_N(v,k,i);
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
          state_limits_z   (0,idR,0 ,j,i) = state_limits_z   (1,idR,0 ,j,i);
          state_limits_z   (0,idU,0 ,j,i) = state_limits_z   (1,idU,0 ,j,i);
          state_limits_z   (0,idV,0 ,j,i) = state_limits_z   (1,idV,0 ,j,i);
          state_limits_z   (0,idT,0 ,j,i) = state_limits_z   (1,idT,0 ,j,i);
          pressure_limits_z(0    ,0 ,j,i) = pressure_limits_z(1    ,0 ,j,i);
          state_limits_z   (1,idR,nz,j,i) = state_limits_z   (0,idR,nz,j,i);
          state_limits_z   (1,idU,nz,j,i) = state_limits_z   (0,idU,nz,j,i);
          state_limits_z   (1,idV,nz,j,i) = state_limits_z   (0,idV,nz,j,i);
          state_limits_z   (1,idT,nz,j,i) = state_limits_z   (0,idT,nz,j,i);
          pressure_limits_z(1    ,nz,j,i) = pressure_limits_z(0    ,nz,j,i);
        });
      } else if (bc_z == "periodic") {
        // z-direction BC's
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
          state_limits_z   (0,idR,0 ,j,i) = state_limits_z   (0,idR,nz,j,i);
          state_limits_z   (0,idU,0 ,j,i) = state_limits_z   (0,idU,nz,j,i);
          state_limits_z   (0,idV,0 ,j,i) = state_limits_z   (0,idV,nz,j,i);
          state_limits_z   (0,idW,0 ,j,i) = state_limits_z   (0,idW,nz,j,i);
          state_limits_z   (0,idT,0 ,j,i) = state_limits_z   (0,idT,nz,j,i);
          pressure_limits_z(0    ,0 ,j,i) = pressure_limits_z(0    ,nz,j,i);
          state_limits_z   (1,idR,nz,j,i) = state_limits_z   (1,idR,0 ,j,i);
          state_limits_z   (1,idU,nz,j,i) = state_limits_z   (1,idU,0 ,j,i);
          state_limits_z   (1,idV,nz,j,i) = state_limits_z   (1,idV,0 ,j,i);
          state_limits_z   (1,idW,nz,j,i) = state_limits_z   (1,idW,0 ,j,i);
          state_limits_z   (1,idT,nz,j,i) = state_limits_z   (1,idT,0 ,j,i);
          pressure_limits_z(1    ,nz,j,i) = pressure_limits_z(1    ,0 ,j,i);
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(0,l,0 ,j,i) = tracers_limits_z(0,l,nz,j,i);
            tracers_limits_z(1,l,nz,j,i) = tracers_limits_z(1,l,0 ,j,i);
          }
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("edge_exchange");
      #endif
    }



    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) const {
      #ifdef YAKL_AUTO_PROFILE
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

      real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);  state   = 0;
      real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);  tracers = 0;
      convert_coupler_to_dynamics( coupler , state , tracers );
      dm.register_and_allocate<float>("hy_dens_cells"    ,"",{nz+2*hs});
      dm.register_and_allocate<float>("hy_theta_cells"   ,"",{nz+2*hs});
      dm.register_and_allocate<float>("hy_pressure_cells","",{nz+2*hs});
      auto r = dm.get<float,1>("hy_dens_cells"    );    r = 0;
      auto t = dm.get<float,1>("hy_theta_cells"   );    t = 0;
      auto p = dm.get<float,1>("hy_pressure_cells");    p = 0;
      parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , YAKL_LAMBDA (int k) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            r(k) += state(idR,k,hs+j,hs+i);
            t(k) += state(idT,k,hs+j,hs+i) / state(idR,k,hs+j,hs+i);
            p(k) += C0 * std::pow( state(idT,k,hs+j,hs+i) , gamma );
          }
        }
      });
      coupler.get_parallel_comm().all_reduce( r , MPI_SUM ).deep_copy_to(r);
      coupler.get_parallel_comm().all_reduce( t , MPI_SUM ).deep_copy_to(t);
      coupler.get_parallel_comm().all_reduce( p , MPI_SUM ).deep_copy_to(p);
      real r_nx_ny = 1./(nx_glob*ny_glob);
      parallel_for( YAKL_AUTO_LABEL() , nz+2*hs , YAKL_LAMBDA (int k) {
        r(k) *= r_nx_ny;
        t(k) *= r_nx_ny;
        p(k) *= r_nx_ny;
      });
      parallel_for( YAKL_AUTO_LABEL() , hs , YAKL_LAMBDA (int kk) {
        {
          int  k0       = hs;
          int  k        = k0-1-kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 + grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k) = theta0;
          p(k) = C0*std::pow(r(k)*theta0,gamma);
        }
        {
          int  k0       = hs+nz-1;
          int  k        = k0+1+kk;
          real rho0     = r(k0);
          real theta0   = t(k0);
          real rho0_gm1 = std::pow(rho0  ,gamma-1);
          real theta0_g = std::pow(theta0,gamma  );
          r(k) = std::pow( rho0_gm1 - grav*(gamma-1)*dz*(kk+1)/(gamma*C0*theta0_g) , 1._fp/(gamma-1) );
          t(k) = theta0;
          p(k) = C0*std::pow(r(k)*theta0,gamma);
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
        if (! dm.entry_exists("hy_dens_edges" )) dm.register_and_allocate<float>("hy_dens_edges" ,"",{nz+1});
        if (! dm.entry_exists("hy_theta_edges")) dm.register_and_allocate<float>("hy_theta_edges","",{nz+1});
        auto hy_dens_cells  = dm.get<float const,1>("hy_dens_cells" );
        auto hy_theta_cells = dm.get<float const,1>("hy_theta_cells");
        auto hy_dens_edges  = dm.get<float      ,1>("hy_dens_edges" );
        auto hy_theta_edges = dm.get<float      ,1>("hy_theta_edges");
        if (ord < 5) {
          parallel_for( YAKL_AUTO_LABEL() , nz+1 , YAKL_LAMBDA (int k) {
            hy_dens_edges(k) = std::exp( 0.5_fp*std::log(hy_dens_cells(hs+k-1)) +
                                         0.5_fp*std::log(hy_dens_cells(hs+k  )) );
            hy_theta_edges(k) = 0.5_fp*hy_theta_cells(hs+k-1) +
                                0.5_fp*hy_theta_cells(hs+k  ) ;
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
        nc.create_var<float>( "hy_dens_cells"     , {"z_halo"});
        nc.create_var<float>( "hy_theta_cells"    , {"z_halo"});
        nc.create_var<float>( "hy_pressure_cells" , {"z_halo"});
        nc.create_var<real>( "theta_pert"        , {"z","y","x"});
        nc.create_var<real>( "pressure_pert"     , {"z","y","x"});
        nc.create_var<real>( "density_pert"      , {"z","y","x"});
        nc.enddef();
        nc.begin_indep_data();
        auto &dm = coupler.get_data_manager_readonly();
        if (coupler.is_mainproc()) nc.write( dm.get<float const,1>("hy_dens_cells"    ) , "hy_dens_cells"     );
        if (coupler.is_mainproc()) nc.write( dm.get<float const,1>("hy_theta_cells"   ) , "hy_theta_cells"    );
        if (coupler.is_mainproc()) nc.write( dm.get<float const,1>("hy_pressure_cells") , "hy_pressure_cells" );
        nc.end_indep_data();
        real4d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs);
        real4d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs);
        convert_coupler_to_dynamics( coupler , state , tracers );
        std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg};
        using yakl::componentwise::operator/;
        real3d data("data",nz,ny,nx);
        auto hy_dens_cells = dm.get<float const,1>("hy_dens_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = state(idR,hs+k,hs+j,hs+i) - hy_dens_cells(hs+k);
        });
        nc.write_all(data,"density_pert",start_3d);
        auto hy_theta_cells = dm.get<float const,1>("hy_theta_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = state(idT,hs+k,hs+j,hs+i) / state(idR,hs+k,hs+j,hs+i) - hy_theta_cells(hs+k);
        });
        nc.write_all(data,"theta_pert",start_3d);
        auto hy_pressure_cells = dm.get<float const,1>("hy_pressure_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = C0 * std::pow( state(idT,hs+k,hs+j,hs+i) , gamma ) - hy_pressure_cells(hs+k);
        });
        nc.write_all(data,"pressure_pert",start_3d);
      } );
      coupler.register_overwrite_with_restart_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        auto &dm = coupler.get_data_manager_readwrite();
        nc.read_all(dm.get<float,1>("hy_dens_cells"    ),"hy_dens_cells"    ,{0});
        nc.read_all(dm.get<float,1>("hy_theta_cells"   ),"hy_theta_cells"   ,{0});
        nc.read_all(dm.get<float,1>("hy_pressure_cells"),"hy_pressure_cells",{0});
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
      auto  nx          = coupler.get_nx();
      auto  ny          = coupler.get_ny();
      auto  nz          = coupler.get_nz();
      auto  R_d         = coupler.get_option<real>("R_d"    );
      auto  R_v         = coupler.get_option<real>("R_v"    );
      auto  gamma       = coupler.get_option<real>("gamma_d");
      auto  C0          = coupler.get_option<real>("C0"     );
      auto  idWV        = coupler.get_option<int >("idWV"   );
      auto  num_tracers = coupler.get_num_tracers();
      auto  &dm = coupler.get_data_manager_readwrite();
      auto  dm_rho_d = dm.get<real,3>("density_dry");
      auto  dm_uvel  = dm.get<real,3>("uvel"       );
      auto  dm_vvel  = dm.get<real,3>("vvel"       );
      auto  dm_wvel  = dm.get<real,3>("wvel"       );
      auto  dm_temp  = dm.get<real,3>("temp"       );
      auto  tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      core::MultiField<real,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real,3>(tracer_names[tr]) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real rho   = state(idR,hs+k,hs+j,hs+i);
        real u     = state(idU,hs+k,hs+j,hs+i) / rho;
        real v     = state(idV,hs+k,hs+j,hs+i) / rho;
        real w     = state(idW,hs+k,hs+j,hs+i) / rho;
        real theta = state(idT,hs+k,hs+j,hs+i) / rho;
        real press = C0 * pow( rho*theta , gamma );
        real rho_v = tracers(idWV,hs+k,hs+j,hs+i);
        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,hs+j,hs+i); }
        real temp = press / ( rho_d * R_d + rho_v * R_v );
        dm_rho_d(k,j,i) = rho_d;
        dm_uvel (k,j,i) = u;
        dm_vvel (k,j,i) = v;
        dm_wvel (k,j,i) = w;
        dm_temp (k,j,i) = temp;
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i) = tracers(tr,hs+k,hs+j,hs+i); }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_dynamics_to_coupler");
      #endif
    }



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("convert_coupler_to_dynamics");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto  nx          = coupler.get_nx();
      auto  ny          = coupler.get_ny();
      auto  nz          = coupler.get_nz();
      auto  R_d         = coupler.get_option<real>("R_d"    );
      auto  R_v         = coupler.get_option<real>("R_v"    );
      auto  gamma       = coupler.get_option<real>("gamma_d");
      auto  C0          = coupler.get_option<real>("C0"     );
      auto  idWV        = coupler.get_option<int >("idWV"   );
      auto  num_tracers = coupler.get_num_tracers();
      auto  &dm = coupler.get_data_manager_readonly();
      auto  dm_rho_d = dm.get<real const,3>("density_dry");
      auto  dm_uvel  = dm.get<real const,3>("uvel"       );
      auto  dm_vvel  = dm.get<real const,3>("vvel"       );
      auto  dm_wvel  = dm.get<real const,3>("wvel"       );
      auto  dm_temp  = dm.get<real const,3>("temp"       );
      auto  tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      core::MultiField<real const,3> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real const,3>(tracer_names[tr]) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real rho_d = dm_rho_d(k,j,i);
        real u     = dm_uvel (k,j,i);
        real v     = dm_vvel (k,j,i);
        real w     = dm_wvel (k,j,i);
        real temp  = dm_temp (k,j,i);
        real rho_v = dm_tracers(idWV,k,j,i);
        real press = rho_d * R_d * temp + rho_v * R_v * temp;
        real rho = rho_d;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i); }
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;
        state(idR,hs+k,hs+j,hs+i) = rho;
        state(idU,hs+k,hs+j,hs+i) = rho * u;
        state(idV,hs+k,hs+j,hs+i) = rho * v;
        state(idW,hs+k,hs+j,hs+i) = rho * w;
        state(idT,hs+k,hs+j,hs+i) = rho * theta;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i) = dm_tracers(tr,k,j,i); }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("convert_coupler_to_dynamics");
      #endif
    }


  };

}


