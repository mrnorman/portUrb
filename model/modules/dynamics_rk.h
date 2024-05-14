
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
    #ifndef MW_ORD
      yakl::index_t static constexpr ord = 9;
    #else
      yakl::index_t static constexpr ord = MW_ORD;
    #endif
    int static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature



    Dynamics_Euler_Stratified_WenoFV();



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



    // Use CFL criterion to determine the time step. Currently hardwired
    real compute_time_step( core::Coupler const &coupler ) const;


    // Perform a time step
    void time_step(core::Coupler &coupler, real dt_phys) const;


    // CFL 0.45 (Differs from paper, but this is the true value for this high-order FV scheme)
    // Third-order, three-stage SSPRK method
    // https://link.springer.com/content/pdf/10.1007/s10915-008-9239-z.pdf
    void time_step_rk_3_3( core::Coupler & coupler ,
                           real4d const  & state   ,
                           real4d const  & tracers ,
                           real            dt_dyn  ) const;


    void enforce_immersed_boundaries( core::Coupler const & coupler ,
                                      real4d        const & state   ,
                                      real4d        const & tracers ,
                                      real                  dt      ) const;



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
    void compute_tendencies( core::Coupler       & coupler      ,
                             real4d        const & state        ,
                             real4d        const & state_tend   ,
                             real4d        const & tracers      ,
                             real4d        const & tracers_tend ,
                             real                  dt           ) const;



    void halo_boundary_conditions( core::Coupler const & coupler  ,
                                   real4d        const & state    ,
                                   real4d        const & tracers  ,
                                   real3d        const & pressure ) const;



    void edge_exchange( core::Coupler const & coupler           ,
                        real5d        const & state_limits_x    ,
                        real5d        const & tracers_limits_x  ,
                        real4d        const & pressure_limits_x ,
                        real5d        const & state_limits_y    ,
                        real5d        const & tracers_limits_y  ,
                        real4d        const & pressure_limits_y ,
                        real5d        const & state_limits_z    ,
                        real5d        const & tracers_limits_z  ,
                        real4d        const & pressure_limits_z ) const;



    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) const;



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler , realConst4d state , realConst4d tracers ) const;



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler , real4d &state , real4d &tracers ) const;


  };

}


