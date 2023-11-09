
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
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

  class Dynamics_Euler_Stratified_WenoFV {
    public:

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
    // IDs for the test cases
    int  static constexpr DATA_THERMAL   = 0;
    int  static constexpr DATA_SUPERCELL = 1;
    int  static constexpr DATA_CITY      = 2;
    int  static constexpr DATA_BUILDING  = 3;
    // IDs for boundary conditions
    int  static constexpr BC_PERIODIC = 0;
    int  static constexpr BC_OPEN     = 1;
    int  static constexpr BC_WALL     = 2;
    // Class data (not use inside parallel_for)
    real etime;    // Elapsed time
    real out_freq; // Frequency out file output
    int  num_out;  // Number of outputs produced thus far


    // Compute the maximum stable time step using very conservative assumptions about max wind speed
    real compute_time_step( core::Coupler const &coupler ) const {
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      real constexpr maxwave = 350 + 80;
      real cfl = 0.9;
      return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
    }


    int static constexpr DIR_X = 0;
    int static constexpr DIR_Y = 1;
    int static constexpr DIR_Z = 2;

    bool dir_switch = true;


    void time_step(core::Coupler &coupler, real &dt_phys) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      if (dir_switch) {
        time_step_rk(coupler,dt_phys,DIR_X);
        time_step_rk(coupler,dt_phys,DIR_Y);
        time_step_rk(coupler,dt_phys,DIR_Z);
      } else {
        time_step_rk(coupler,dt_phys,DIR_Z);
        time_step_rk(coupler,dt_phys,DIR_Y);
        time_step_rk(coupler,dt_phys,DIR_X);
      }
      dir_switch = ! dir_switch;

      // Advance the dycore's tracking of total ellapsed time
      etime += dt_phys;
      // Do output and inform the user if it's time to do output
      if (out_freq >= 0. && etime / out_freq >= num_out+1) {
        output( coupler , etime );
        num_out++;
        // Let the user know what the max vertical velocity is to ensure the model hasn't crashed
        auto &dm = coupler.get_data_manager_readonly();
        auto u = dm.get_collapsed<real const>("uvel");
        auto v = dm.get_collapsed<real const>("vvel");
        auto w = dm.get_collapsed<real const>("wvel");
        auto mag = u.createDeviceObject();
        parallel_for( YAKL_AUTO_LABEL() , mag.size() , YAKL_LAMBDA (int i) {
          mag(i) = std::sqrt( u(i)*u(i) + v(i)*v(i) + w(i)*w(i) );
        });
        real wind_mag_loc = yakl::intrinsics::maxval(mag);
        real wind_mag;
        auto mpi_data_type = coupler.get_mpi_data_type();
        MPI_Reduce( &wind_mag_loc , &wind_mag , 1 , mpi_data_type , MPI_MAX , 0 , MPI_COMM_WORLD );
        if (coupler.is_mainproc()) {
          std::cout << "Etime , dtphys, wind_mag: " << std::scientific << std::setw(10) << etime    << " , " 
                                                    << std::scientific << std::setw(10) << dt_phys  << " , "
                                                    << std::scientific << std::setw(10) << wind_mag << std::endl;
        }
      }
    }


    // Perform a single time step using SSPRK3 time stepping
    void time_step_rk(core::Coupler &coupler, real &dt_phys, int dir) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      auto num_tracers         = coupler.get_num_tracers();
      auto nens                = coupler.get_nens();
      auto nx                  = coupler.get_nx();
      auto ny                  = coupler.get_ny();
      auto nz                  = coupler.get_nz();
      auto tracer_positive     = coupler.get_data_manager_readonly().get<bool const,1>("tracer_positive");
      auto immersed_proportion = coupler.get_data_manager_readonly().get<real const,4>("immersed_proportion");
      auto hy_dens_cells       = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_cells"      );
      auto hy_dens_theta_cells = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_theta_cells");

      // Create arrays to hold state and tracers with halos on the left and right of the domain
      // Cells [0:hs-1] are the left halos, and cells [nx+hs:nx+2*hs-1] are the right halos
      real5d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real5d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);

      // Populate the state and tracers arrays using data from the coupler, convert to the dycore's desired state
      convert_coupler_to_dynamics( coupler , state , tracers );

      // Get the max stable time step for the dynamics. dt_phys might be > dt_dyn, meaning we would need to sub-cycle
      real dt_dyn = compute_time_step( coupler );

      // Get the number of sub-cycles we need, and set the dynamics time step accordingly
      int ncycles = (int) std::ceil( dt_phys / dt_dyn );
      dt_dyn = dt_phys / ncycles;

      for (int icycle = 0; icycle < ncycles; icycle++) {
        if (ord == 1) {
          real5d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     ,nens);
          real5d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     ,nens);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
            tracers_tend(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens);
          });
          compute_tendencies( coupler , state , state_tend , tracers , tracers_tend , dt_dyn , dir );
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
            for (int l = 0; l < num_state; l++) {
              state_tend  (l ,k,j,i,iens) = state  (l ,hs+k,hs+j,hs+i,iens);
            }
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_tend(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens);
            }
            real prop = immersed_proportion(k,j,i,iens);
            if (prop > 0) {
              state(idR,hs+k,hs+j,hs+i,iens) = hy_dens_cells(k,iens);
              state(idU,hs+k,hs+j,hs+i,iens) = 0;
              state(idV,hs+k,hs+j,hs+i,iens) = 0;
              state(idW,hs+k,hs+j,hs+i,iens) = 0;
              state(idT,hs+k,hs+j,hs+i,iens) = hy_dens_theta_cells(k,iens);
            }
          });
          compute_tendencies( coupler , state , state_tend , tracers , tracers_tend , dt_dyn , dir );
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
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            for (int l = 0; l < num_state; l++) {
              state_tend  (l ,k,j,i,iens) = (3._fp/4._fp) * state      (l ,hs+k,hs+j,hs+i,iens) + 
                                            (1._fp/4._fp) * state_tmp  (l ,hs+k,hs+j,hs+i,iens);
            }
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_tend(tr,k,j,i,iens) = (3._fp/4._fp) * tracers    (tr,hs+k,hs+j,hs+i,iens) + 
                                            (1._fp/4._fp) * tracers_tmp(tr,hs+k,hs+j,hs+i,iens);
            }
            real prop = immersed_proportion(k,j,i,iens);
            if (prop > 0) {
              state_tmp(idR,hs+k,hs+j,hs+i,iens) = hy_dens_cells(k,iens);
              state_tmp(idU,hs+k,hs+j,hs+i,iens) = 0;
              state_tmp(idV,hs+k,hs+j,hs+i,iens) = 0;
              state_tmp(idW,hs+k,hs+j,hs+i,iens) = 0;
              state_tmp(idT,hs+k,hs+j,hs+i,iens) = hy_dens_theta_cells(k,iens);
            }
          });
          compute_tendencies( coupler , state_tmp , state_tend , tracers_tmp , tracers_tend , (1._fp/4._fp) * dt_dyn , dir );
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
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            for (int l = 0; l < num_state  ; l++) {
              state_tend  (l ,k,j,i,iens) = (1._fp/3._fp) * state      (l ,hs+k,hs+j,hs+i,iens) + 
                                            (2._fp/3._fp) * state_tmp  (l ,hs+k,hs+j,hs+i,iens);
            }
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_tend(tr,k,j,i,iens) = (1._fp/3._fp) * tracers    (tr,hs+k,hs+j,hs+i,iens) + 
                                            (2._fp/3._fp) * tracers_tmp(tr,hs+k,hs+j,hs+i,iens);
            }
            real prop = immersed_proportion(k,j,i,iens);
            if (prop > 0) {
              state_tmp(idR,hs+k,hs+j,hs+i,iens) = hy_dens_cells(k,iens);
              state_tmp(idU,hs+k,hs+j,hs+i,iens) = 0;
              state_tmp(idV,hs+k,hs+j,hs+i,iens) = 0;
              state_tmp(idW,hs+k,hs+j,hs+i,iens) = 0;
              state_tmp(idT,hs+k,hs+j,hs+i,iens) = hy_dens_theta_cells(k,iens);
            }
          });
          compute_tendencies( coupler , state_tmp , state_tend , tracers_tmp , tracers_tend , (2._fp/3._fp) * dt_dyn , dir );
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
            real prop = immersed_proportion(k,j,i,iens);
            if (prop > 0) {
              state(idR,hs+k,hs+j,hs+i,iens) = hy_dens_cells(k,iens);
              state(idU,hs+k,hs+j,hs+i,iens) = 0;
              state(idV,hs+k,hs+j,hs+i,iens) = 0;
              state(idW,hs+k,hs+j,hs+i,iens) = 0;
              state(idT,hs+k,hs+j,hs+i,iens) = hy_dens_theta_cells(k,iens);
            }
          });
        }
      }

      // Convert the dycore's state back to the coupler's state
      convert_dynamics_to_coupler( coupler , state , tracers );
    }


    // Compute the tendencies for state and tracers for one semi-discretized step inside the RK integrator
    // Tendencies are the time rate of change for a quantity
    // Coupler is non-const because we are writing to the flux variables
    void compute_tendencies( core::Coupler       & coupler      ,
                             real5d        const & state        ,
                             real5d        const & state_tend   ,
                             real5d        const & tracers      ,
                             real5d        const & tracers_tend ,
                             real                  dt           ,
                             int                   dir          ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      using std::min;
      using std::max;

      auto use_immersed_boundaries = coupler.get_option<bool>("use_immersed_boundaries");
      auto earthrot                = coupler.get_option<real>("earthrot");
      auto fcor                    = 2*earthrot*sin(coupler.get_option<real>("latitude"));
      auto nens                    = coupler.get_nens();
      auto nx                      = coupler.get_nx();
      auto ny                      = coupler.get_ny();
      auto nz                      = coupler.get_nz();
      auto dx                      = coupler.get_dx();
      auto dy                      = coupler.get_dy();
      auto dz                      = coupler.get_dz();
      auto sim2d                   = coupler.is_sim2d();
      auto C0                      = coupler.get_option<real>("C0"     );
      auto gamma                   = coupler.get_option<real>("gamma_d");
      auto grav                    = coupler.get_option<real>("grav"   );
      auto enable_gravity          = coupler.get_option<bool>("enable_gravity",true);
      auto num_tracers             = coupler.get_num_tracers();

      SArray<real,2,ord,2> coefs_to_gll;
      TransformMatrices::coefs_to_gll_lower(coefs_to_gll);

      auto &dm = coupler.get_data_manager_readonly();
      auto tracer_positive     = dm.get<bool const,1>("tracer_positive"    );
      auto hy_dens_cells       = dm.get<real const,2>("hy_dens_cells"      );
      auto hy_dens_theta_cells = dm.get<real const,2>("hy_dens_theta_cells");
      auto pressure_mult       = dm.get<real const,2>("pressure_mult"      );

      // Since tracers are full mass, it's helpful before reconstruction to remove the background density for potentially
      // more accurate reconstructions of tracer concentrations
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        state(idU,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens);
        state(idV,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens);
        state(idW,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens);
        state(idT,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens);
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) /= state(idR,hs+k,hs+j,hs+i,iens); }
      });

      if (ord > 1) halo_exchange( coupler , state , tracers );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        state(idU,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
        state(idV,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
        state(idW,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
        state(idT,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens);
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) *= state(idR,hs+k,hs+j,hs+i,iens); }
      });

      weno::WenoLimiter<ord> limiter;

      if (dir == DIR_X) {

        // These arrays store high-order-accurate samples of the state and tracers at cell edges after cell-centered recon
        real6d state_limits_x   ("state_limits_x"   ,num_state  ,2,nz  ,ny  ,nx+1,nens);  state_limits_x    = 0;
        real6d state_limits_y   ("state_limits_y"   ,num_state  ,2,nz  ,ny+1,nx  ,nens);  state_limits_y    = 0;
        real6d state_limits_z   ("state_limits_z"   ,num_state  ,2,nz+1,ny  ,nx  ,nens);  state_limits_z    = 0;
        real6d tracers_limits_x ("tracers_limits_x" ,num_tracers,2,nz  ,ny  ,nx+1,nens);  tracers_limits_x  = 0;
        real6d tracers_limits_y ("tracers_limits_y" ,num_tracers,2,nz  ,ny+1,nx  ,nens);  tracers_limits_y  = 0;
        real6d tracers_limits_z ("tracers_limits_z" ,num_tracers,2,nz+1,ny  ,nx  ,nens);  tracers_limits_z  = 0;

        // Compute samples of state and tracers at cell edges using cell-centered reconstructions at high-order with WENO
        // At the end of this, we will have two samples per cell edge in each dimension, one from each adjacent cell.
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          real cs = 350;
          real u  = state(idU,hs+k,hs+j,hs+i,iens);
          real v  = state(idV,hs+k,hs+j,hs+i,iens);
          real w  = state(idW,hs+k,hs+j,hs+i,iens);
          real t  = state(idT,hs+k,hs+j,hs+i,iens);
          ////////////////////////////////////////////////////////
          // X-direction
          ////////////////////////////////////////////////////////
          // State
          {
            SArray<real,1,ord> stencil;
            SArray<real,1,2  > g1, g2, g3, g4, g5;
            for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idR,hs+k,hs+j,i+ii,iens); }
            reconstruct_gll_values(stencil,g1,coefs_to_gll,limiter);
            for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idU,hs+k,hs+j,i+ii,iens); }
            reconstruct_gll_values(stencil,g2,coefs_to_gll,limiter);
            for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idV,hs+k,hs+j,i+ii,iens); }
            reconstruct_gll_values(stencil,g3,coefs_to_gll,limiter);
            for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idW,hs+k,hs+j,i+ii,iens); }
            reconstruct_gll_values(stencil,g4,coefs_to_gll,limiter);
            for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idT,hs+k,hs+j,i+ii,iens); }
            reconstruct_gll_values(stencil,g5,coefs_to_gll,limiter);

            real r  = g1(0);
            real ru = g2(0);
            real rv = g3(0);
            real rw = g4(0);
            real rt = g5(0);
            // r  = std::max(std::min(state(idR,hs+k,hs+j,hs+i-1,iens),state(idR,hs+k,hs+j,hs+i,iens)),r );
            // ru = std::max(std::min(state(idU,hs+k,hs+j,hs+i-1,iens),state(idU,hs+k,hs+j,hs+i,iens)),ru);
            // rv = std::max(std::min(state(idV,hs+k,hs+j,hs+i-1,iens),state(idV,hs+k,hs+j,hs+i,iens)),rv);
            // rw = std::max(std::min(state(idW,hs+k,hs+j,hs+i-1,iens),state(idW,hs+k,hs+j,hs+i,iens)),rw);
            // rt = std::max(std::min(state(idT,hs+k,hs+j,hs+i-1,iens),state(idT,hs+k,hs+j,hs+i,iens)),rt);
            // r  = std::min(std::max(state(idR,hs+k,hs+j,hs+i-1,iens),state(idR,hs+k,hs+j,hs+i,iens)),r );
            // ru = std::min(std::max(state(idU,hs+k,hs+j,hs+i-1,iens),state(idU,hs+k,hs+j,hs+i,iens)),ru);
            // rv = std::min(std::max(state(idV,hs+k,hs+j,hs+i-1,iens),state(idV,hs+k,hs+j,hs+i,iens)),rv);
            // rw = std::min(std::max(state(idW,hs+k,hs+j,hs+i-1,iens),state(idW,hs+k,hs+j,hs+i,iens)),rw);
            // rt = std::min(std::max(state(idT,hs+k,hs+j,hs+i-1,iens),state(idT,hs+k,hs+j,hs+i,iens)),rt);
            state_limits_x(idR,1,k,j,i  ,iens) = r;
            state_limits_x(idU,1,k,j,i  ,iens) = ru;
            state_limits_x(idV,1,k,j,i  ,iens) = rv;
            state_limits_x(idW,1,k,j,i  ,iens) = rw;
            state_limits_x(idT,1,k,j,i  ,iens) = rt;

            r  = g1(1);
            ru = g2(1);
            rv = g3(1);
            rw = g4(1);
            rt = g5(1);
            // r  = std::max(std::min(state(idR,hs+k,hs+j,hs+i,iens),state(idR,hs+k,hs+j,hs+i+1,iens)),r );
            // ru = std::max(std::min(state(idU,hs+k,hs+j,hs+i,iens),state(idU,hs+k,hs+j,hs+i+1,iens)),ru);
            // rv = std::max(std::min(state(idV,hs+k,hs+j,hs+i,iens),state(idV,hs+k,hs+j,hs+i+1,iens)),rv);
            // rw = std::max(std::min(state(idW,hs+k,hs+j,hs+i,iens),state(idW,hs+k,hs+j,hs+i+1,iens)),rw);
            // rt = std::max(std::min(state(idT,hs+k,hs+j,hs+i,iens),state(idT,hs+k,hs+j,hs+i+1,iens)),rt);
            // r  = std::min(std::max(state(idR,hs+k,hs+j,hs+i,iens),state(idR,hs+k,hs+j,hs+i+1,iens)),r );
            // ru = std::min(std::max(state(idU,hs+k,hs+j,hs+i,iens),state(idU,hs+k,hs+j,hs+i+1,iens)),ru);
            // rv = std::min(std::max(state(idV,hs+k,hs+j,hs+i,iens),state(idV,hs+k,hs+j,hs+i+1,iens)),rv);
            // rw = std::min(std::max(state(idW,hs+k,hs+j,hs+i,iens),state(idW,hs+k,hs+j,hs+i+1,iens)),rw);
            // rt = std::min(std::max(state(idT,hs+k,hs+j,hs+i,iens),state(idT,hs+k,hs+j,hs+i+1,iens)),rt);
            state_limits_x(idR,0,k,j,i+1,iens) = r ; 
            state_limits_x(idU,0,k,j,i+1,iens) = ru;
            state_limits_x(idV,0,k,j,i+1,iens) = rv;
            state_limits_x(idW,0,k,j,i+1,iens) = rw;
            state_limits_x(idT,0,k,j,i+1,iens) = rt;
          }
          // Tracers
          for (int l=0; l < num_tracers; l++) {
            // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
            SArray<real,1,ord> stencil;
            SArray<real,1,2>   gll;
            for (int s=0; s < ord; s++) { stencil(s) = tracers(l,hs+k,hs+j,i+s,iens); }
            reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
            tracers_limits_x(l,1,k,j,i  ,iens) = gll(0);
            tracers_limits_x(l,0,k,j,i+1,iens) = gll(1);
          }
        });

        edge_exchange( coupler           ,
                       state_limits_x    ,
                       tracers_limits_x  ,
                       state_limits_y    ,
                       tracers_limits_y  ,
                       state_limits_z    ,
                       tracers_limits_z  );

        real5d state_flux_x  ("state_flux_x"  ,num_state  ,nz  ,ny  ,nx+1,nens);
        real5d state_flux_y  ("state_flux_y"  ,num_state  ,nz  ,ny+1,nx  ,nens);
        real5d state_flux_z  ("state_flux_z"  ,num_state  ,nz+1,ny  ,nx  ,nens);
        real5d tracers_flux_x("tracers_flux_x",num_tracers,nz  ,ny  ,nx+1,nens);
        real5d tracers_flux_y("tracers_flux_y",num_tracers,nz  ,ny+1,nx  ,nens);
        real5d tracers_flux_z("tracers_flux_z",num_tracers,nz+1,ny  ,nx  ,nens);

        auto save_pressure_z = coupler.get_option<bool>("save_pressure_z",false);
        real4d pressure_z;
        if (save_pressure_z) pressure_z = coupler.get_data_manager_readwrite().get<real,4>("pressure_z");

        // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz+1,ny+1,nx+1,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          // X-direction
          if (j < ny && k < nz) {
            real r_L = state_limits_x(idR,0,k,j,i,iens)    ;    real r_R = state_limits_x(idR,1,k,j,i,iens)    ;
            real u_L = state_limits_x(idU,0,k,j,i,iens)/r_L;    real u_R = state_limits_x(idU,1,k,j,i,iens)/r_R;
            real v_L = state_limits_x(idV,0,k,j,i,iens)/r_L;    real v_R = state_limits_x(idV,1,k,j,i,iens)/r_R;
            real w_L = state_limits_x(idW,0,k,j,i,iens)/r_L;    real w_R = state_limits_x(idW,1,k,j,i,iens)/r_R;
            real t_L = state_limits_x(idT,0,k,j,i,iens)/r_L;    real t_R = state_limits_x(idT,1,k,j,i,iens)/r_R;
            real p_L = C0*std::pow(r_L*t_L,gamma)          ;    real p_R = C0*std::pow(r_R*t_R,gamma)          ;
            real r = 0.5_fp*(r_L+r_R);
            real u = 0.5_fp*(u_L+u_R);
            real v = 0.5_fp*(v_L+v_R);
            real w = 0.5_fp*(w_L+w_R);
            real t = 0.5_fp*(t_L+t_R);
            real p = 0.5_fp*(p_L+p_R);
            real cs2 = gamma*p/r;
            real cs  = std::sqrt(cs2);
            real w1 =  u*(r_R)/(2*cs) - (r_R*u_R)/(2*cs) + (r_R*t_R)/(2*t);
            real w2 = -u*(r_L)/(2*cs) + (r_L*u_L)/(2*cs) + (r_L*t_L)/(2*t);
            real w3, w4, w5;
            if (u > 0) {
              w3 = (r_L)     -   (r_L*t_L)/t;
              w4 = (r_L*v_L) - v*(r_L*t_L)/t;
              w5 = (r_L*w_L) - w*(r_L*t_L)/t;
            } else {
              w3 = (r_R)     -   (r_R*t_R)/t;
              w4 = (r_R*v_R) - v*(r_R*t_R)/t;
              w5 = (r_R*w_R) - w*(r_R*t_R)/t;
            }
            real r_upw  = w1        + w2        + w3            ;
            real ru_upw = w1*(u-cs) + w2*(u+cs) + w3*u          ;
            real rv_upw = w1*v      + w2*v             + w4     ;
            real rw_upw = w1*w      + w2*w                  + w5;
            real rt_upw = w1*t      + w2*t                      ;
            state_flux_x(idR,k,j,i,iens) = ru_upw;
            state_flux_x(idU,k,j,i,iens) = ru_upw*ru_upw/r_upw + C0*std::pow(rt_upw,gamma);
            state_flux_x(idV,k,j,i,iens) = ru_upw*rv_upw/r_upw;
            state_flux_x(idW,k,j,i,iens) = ru_upw*rw_upw/r_upw;
            state_flux_x(idT,k,j,i,iens) = ru_upw*rt_upw/r_upw;
            int uind = u > 0 ? 0 : 1;
            r_upw = state_limits_x(idR,uind,k,j,i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_flux_x(tr,k,j,i,iens) = ru_upw*tracers_limits_x(tr,uind,k,j,i,iens)/r_upw;
            }


            // state_flux_x(idR,k,j,i,iens) = 0.5*((r_R*u_R        )+(r_L*u_L        )-(std::abs(u)+cs)*((r_R    )-(r_L    )));
            // state_flux_x(idU,k,j,i,iens) = 0.5*((r_R*u_R*u_R+p_R)+(r_L*u_L*u_L+p_L)-(std::abs(u)+cs)*((r_R*u_R)-(r_L*u_L)));
            // state_flux_x(idV,k,j,i,iens) = 0.5*((r_R*u_R*v_R    )+(r_L*u_L*v_L    )-(std::abs(u)+cs)*((r_R*v_R)-(r_L*v_L)));
            // state_flux_x(idW,k,j,i,iens) = 0.5*((r_R*u_R*w_R    )+(r_L*u_L*w_L    )-(std::abs(u)+cs)*((r_R*w_R)-(r_L*w_L)));
            // state_flux_x(idT,k,j,i,iens) = 0.5*((r_R*u_R*t_R    )+(r_L*u_L*t_L    )-(std::abs(u)+cs)*((r_R*t_R)-(r_L*t_L)));
            // for (int tr=0; tr < num_tracers; tr++) {
            //   real t_L = tracers_limits_x(tr,0,k,j,i,iens)/state_limits_x(idR,0,k,j,i,iens);
            //   real t_R = tracers_limits_x(tr,1,k,j,i,iens)/state_limits_x(idR,1,k,j,i,iens);
            //   tracers_flux_x(tr,k,j,i,iens) = 0.5*((r_R*u_R*t_R)+(r_L*u_L*t_L)-(std::abs(u)+cs)*((r_R*t_R)-(r_L*t_L)));
            // }
          }
        });

        // Deallocate state and tracer limits because they are no longer needed
        state_limits_x    = real6d();
        state_limits_y    = real6d();
        state_limits_z    = real6d();
        tracers_limits_x  = real6d();
        tracers_limits_y  = real6d();
        tracers_limits_z  = real6d();

        real5d tracers_mult_x("tracers_mult_x",num_tracers,nz,ny,nx+1,nens);
        real5d tracers_mult_y("tracers_mult_y",num_tracers,nz,ny+1,nx,nens);
        real5d tracers_mult_z("tracers_mult_z",num_tracers,nz+1,ny,nx,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz+1,ny+1,nx+1,nens) ,
                                          YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
          if (j < ny && k < nz) tracers_mult_x(tr,k,j,i,iens) = 1;
          if (i < nx && k < nz) tracers_mult_y(tr,k,j,i,iens) = 1;
          if (i < nx && j < ny) tracers_mult_z(tr,k,j,i,iens) = 1;
        });

        // Flux Corrected Transport to enforce positivity for tracer species that must remain non-negative
        // This looks like it has a race condition, but it does not. Only one of the adjacent cells can ever change
        // a given edge flux because it's only changed if its sign oriented outward from a cell.
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
          if (tracer_positive(tr)) {
            real mass_available = max(tracers_tend(tr,k,j,i,iens),0._fp) * dx * dy * dz;
            real flux_out_x = ( max(tracers_flux_x(tr,k,j,i+1,iens),0._fp) - min(tracers_flux_x(tr,k,j,i,iens),0._fp) ) / dx;
            real mass_out = (flux_out_x) * dt * dx * dy * dz;
            if (mass_out > mass_available) {
              real mult = mass_available / mass_out;
              if (tracers_flux_x(tr,k,j,i+1,iens) > 0) { tracers_mult_x(tr,k,j,i+1,iens) = mult; }
              if (tracers_flux_x(tr,k,j,i  ,iens) < 0) { tracers_mult_x(tr,k,j,i  ,iens) = mult; }
            }
          }
        });

        fct_mult_exchange( coupler , tracers_mult_x , tracers_mult_y , tracers_mult_z );

        // Compute tendencies as the flux divergence + gravity source term
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          // Now overwrite state_tend and tracers_tend with this stage's tendencies
          for (int l = 0; l < num_state; l++) {
            state_tend(l,k,j,i,iens) = -( state_flux_x(l,k  ,j  ,i+1,iens) - state_flux_x(l,k,j,i,iens) ) / dx;
            if (l == idV && sim2d) state_tend(l,k,j,i,iens) = 0;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tend(l,k,j,i,iens) = -( tracers_flux_x(l,k  ,j  ,i+1,iens)*tracers_mult_x(l,k  ,j  ,i+1,iens) -
                                            tracers_flux_x(l,k  ,j  ,i  ,iens)*tracers_mult_x(l,k  ,j  ,i  ,iens) ) / dx;
          }
        });

      } else if (dir == DIR_Y) {

        // These arrays store high-order-accurate samples of the state and tracers at cell edges after cell-centered recon
        real6d state_limits_x   ("state_limits_x"   ,num_state  ,2,nz  ,ny  ,nx+1,nens);  state_limits_x    = 0;
        real6d state_limits_y   ("state_limits_y"   ,num_state  ,2,nz  ,ny+1,nx  ,nens);  state_limits_y    = 0;
        real6d state_limits_z   ("state_limits_z"   ,num_state  ,2,nz+1,ny  ,nx  ,nens);  state_limits_z    = 0;
        real6d tracers_limits_x ("tracers_limits_x" ,num_tracers,2,nz  ,ny  ,nx+1,nens);  tracers_limits_x  = 0;
        real6d tracers_limits_y ("tracers_limits_y" ,num_tracers,2,nz  ,ny+1,nx  ,nens);  tracers_limits_y  = 0;
        real6d tracers_limits_z ("tracers_limits_z" ,num_tracers,2,nz+1,ny  ,nx  ,nens);  tracers_limits_z  = 0;

        // Compute samples of state and tracers at cell edges using cell-centered reconstructions at high-order with WENO
        // At the end of this, we will have two samples per cell edge in each dimension, one from each adjacent cell.
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          real cs = 350;
          real u  = state(idU,hs+k,hs+j,hs+i,iens);
          real v  = state(idV,hs+k,hs+j,hs+i,iens);
          real w  = state(idW,hs+k,hs+j,hs+i,iens);
          real t  = state(idT,hs+k,hs+j,hs+i,iens);
          ////////////////////////////////////////////////////////
          // Y-direction
          ////////////////////////////////////////////////////////
          // If we're simulating in only 2-D, then do not compute y-direction tendencies
          if (!sim2d) {
            // State
            {
              SArray<real,1,ord> stencil;
              SArray<real,1,2  > g1, g2, g3, g4, g5;
              for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idR,hs+k,j+jj,hs+i,iens); }
              reconstruct_gll_values(stencil,g1,coefs_to_gll,limiter);
              for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idU,hs+k,j+jj,hs+i,iens); }
              reconstruct_gll_values(stencil,g2,coefs_to_gll,limiter);
              for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idV,hs+k,j+jj,hs+i,iens); }
              reconstruct_gll_values(stencil,g3,coefs_to_gll,limiter);
              for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idW,hs+k,j+jj,hs+i,iens); }
              reconstruct_gll_values(stencil,g4,coefs_to_gll,limiter);
              for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idT,hs+k,j+jj,hs+i,iens); }
              reconstruct_gll_values(stencil,g5,coefs_to_gll,limiter);

              real r  = g1(0);
              real ru = g2(0);
              real rv = g3(0);
              real rw = g4(0);
              real rt = g5(0);
              // r  = std::max(std::min(state(idR,hs+k,hs+j-1,hs+i,iens),state(idR,hs+k,hs+j,hs+i,iens)),r );
              // ru = std::max(std::min(state(idU,hs+k,hs+j-1,hs+i,iens),state(idU,hs+k,hs+j,hs+i,iens)),ru);
              // rv = std::max(std::min(state(idV,hs+k,hs+j-1,hs+i,iens),state(idV,hs+k,hs+j,hs+i,iens)),rv);
              // rw = std::max(std::min(state(idW,hs+k,hs+j-1,hs+i,iens),state(idW,hs+k,hs+j,hs+i,iens)),rw);
              // rt = std::max(std::min(state(idT,hs+k,hs+j-1,hs+i,iens),state(idT,hs+k,hs+j,hs+i,iens)),rt);
              // r  = std::min(std::max(state(idR,hs+k,hs+j-1,hs+i,iens),state(idR,hs+k,hs+j,hs+i,iens)),r );
              // ru = std::min(std::max(state(idU,hs+k,hs+j-1,hs+i,iens),state(idU,hs+k,hs+j,hs+i,iens)),ru);
              // rv = std::min(std::max(state(idV,hs+k,hs+j-1,hs+i,iens),state(idV,hs+k,hs+j,hs+i,iens)),rv);
              // rw = std::min(std::max(state(idW,hs+k,hs+j-1,hs+i,iens),state(idW,hs+k,hs+j,hs+i,iens)),rw);
              // rt = std::min(std::max(state(idT,hs+k,hs+j-1,hs+i,iens),state(idT,hs+k,hs+j,hs+i,iens)),rt);
              state_limits_y(idR,1,k,j  ,i,iens) = r ;
              state_limits_y(idU,1,k,j  ,i,iens) = ru;
              state_limits_y(idV,1,k,j  ,i,iens) = rv;
              state_limits_y(idW,1,k,j  ,i,iens) = rw;
              state_limits_y(idT,1,k,j  ,i,iens) = rt;

              r  = g1(1);
              ru = g2(1);
              rv = g3(1);
              rw = g4(1);
              rt = g5(1);
              // r  = std::max(std::min(state(idR,hs+k,hs+j,hs+i,iens),state(idR,hs+k,hs+j+1,hs+i,iens)),r );
              // ru = std::max(std::min(state(idU,hs+k,hs+j,hs+i,iens),state(idU,hs+k,hs+j+1,hs+i,iens)),ru);
              // rv = std::max(std::min(state(idV,hs+k,hs+j,hs+i,iens),state(idV,hs+k,hs+j+1,hs+i,iens)),rv);
              // rw = std::max(std::min(state(idW,hs+k,hs+j,hs+i,iens),state(idW,hs+k,hs+j+1,hs+i,iens)),rw);
              // rt = std::max(std::min(state(idT,hs+k,hs+j,hs+i,iens),state(idT,hs+k,hs+j+1,hs+i,iens)),rt);
              // r  = std::min(std::max(state(idR,hs+k,hs+j,hs+i,iens),state(idR,hs+k,hs+j+1,hs+i,iens)),r );
              // ru = std::min(std::max(state(idU,hs+k,hs+j,hs+i,iens),state(idU,hs+k,hs+j+1,hs+i,iens)),ru);
              // rv = std::min(std::max(state(idV,hs+k,hs+j,hs+i,iens),state(idV,hs+k,hs+j+1,hs+i,iens)),rv);
              // rw = std::min(std::max(state(idW,hs+k,hs+j,hs+i,iens),state(idW,hs+k,hs+j+1,hs+i,iens)),rw);
              // rt = std::min(std::max(state(idT,hs+k,hs+j,hs+i,iens),state(idT,hs+k,hs+j+1,hs+i,iens)),rt);
              state_limits_y(idR,0,k,j+1,i,iens) = r ;
              state_limits_y(idU,0,k,j+1,i,iens) = ru;
              state_limits_y(idV,0,k,j+1,i,iens) = rv;
              state_limits_y(idW,0,k,j+1,i,iens) = rw;
              state_limits_y(idT,0,k,j+1,i,iens) = rt;
            }
            // Tracers
            for (int l=0; l < num_tracers; l++) {
              // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
              SArray<real,1,ord> stencil;
              SArray<real,1,2>   gll;
              for (int s=0; s < ord; s++) { stencil(s) = tracers(l,hs+k,j+s,hs+i,iens); }
              reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
              tracers_limits_y(l,1,k,j  ,i,iens) = gll(0);
              tracers_limits_y(l,0,k,j+1,i,iens) = gll(1);
            }
          } else {
            for (int l=0; l < num_state; l++) {
              state_limits_y(l,1,k,j  ,i,iens) = 0;
              state_limits_y(l,0,k,j+1,i,iens) = 0;
            }
            for (int l=0; l < num_tracers; l++) {
              tracers_limits_y(l,1,k,j  ,i,iens) = 0;
              tracers_limits_y(l,0,k,j+1,i,iens) = 0;
            }
          }
        });

        edge_exchange( coupler           ,
                       state_limits_x    ,
                       tracers_limits_x  ,
                       state_limits_y    ,
                       tracers_limits_y  ,
                       state_limits_z    ,
                       tracers_limits_z  );

        real5d state_flux_x  ("state_flux_x"  ,num_state  ,nz  ,ny  ,nx+1,nens);
        real5d state_flux_y  ("state_flux_y"  ,num_state  ,nz  ,ny+1,nx  ,nens);
        real5d state_flux_z  ("state_flux_z"  ,num_state  ,nz+1,ny  ,nx  ,nens);
        real5d tracers_flux_x("tracers_flux_x",num_tracers,nz  ,ny  ,nx+1,nens);
        real5d tracers_flux_y("tracers_flux_y",num_tracers,nz  ,ny+1,nx  ,nens);
        real5d tracers_flux_z("tracers_flux_z",num_tracers,nz+1,ny  ,nx  ,nens);

        // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz+1,ny+1,nx+1,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          // Y-direction
          // If we are simulating in 2-D, then do not do Riemann in the y-direction
          if ( (! sim2d) && i < nx && k < nz) {
            real r_L = state_limits_y(idR,0,k,j,i,iens)    ;    real r_R = state_limits_y(idR,1,k,j,i,iens)    ;
            real u_L = state_limits_y(idU,0,k,j,i,iens)/r_L;    real u_R = state_limits_y(idU,1,k,j,i,iens)/r_R;
            real v_L = state_limits_y(idV,0,k,j,i,iens)/r_L;    real v_R = state_limits_y(idV,1,k,j,i,iens)/r_R;
            real w_L = state_limits_y(idW,0,k,j,i,iens)/r_L;    real w_R = state_limits_y(idW,1,k,j,i,iens)/r_R;
            real t_L = state_limits_y(idT,0,k,j,i,iens)/r_L;    real t_R = state_limits_y(idT,1,k,j,i,iens)/r_R;
            real p_L = C0*std::pow(r_L*t_L,gamma)          ;    real p_R = C0*std::pow(r_R*t_R,gamma)          ;
            real r = 0.5_fp*(r_L+r_R);
            real u = 0.5_fp*(u_L+u_R);
            real v = 0.5_fp*(v_L+v_R);
            real w = 0.5_fp*(w_L+w_R);
            real t = 0.5_fp*(t_L+t_R);
            real p = 0.5_fp*(p_L+p_R);
            real cs2 = gamma*p/r;
            real cs  = std::sqrt(cs2);
            real w1 =  v*(r_R)/(2*cs) - (r_R*v_R)/(2*cs) + (r_R*t_R)/(2*t);
            real w2 = -v*(r_L)/(2*cs) + (r_L*v_L)/(2*cs) + (r_L*t_L)/(2*t);
            real w3, w4, w5;
            if (v > 0) {
              w3 = (r_L)     -   (r_L*t_L)/t;
              w4 = (r_L*u_L) - u*(r_L*t_L)/t;
              w5 = (r_L*w_L) - w*(r_L*t_L)/t;
            } else {
              w3 = (r_R)     -   (r_R*t_R)/t;
              w4 = (r_R*u_R) - u*(r_R*t_R)/t;
              w5 = (r_R*w_R) - w*(r_R*t_R)/t;
            }
            real r_upw  = w1        + w2        + w3            ;
            real ru_upw = w1*u      + w2*u             + w4     ;
            real rv_upw = w1*(v-cs) + w2*(v+cs) + w3*v          ;
            real rw_upw = w1*w      + w2*w                  + w5;
            real rt_upw = w1*t      + w2*t                      ;
            state_flux_y(idR,k,j,i,iens) = rv_upw;
            state_flux_y(idU,k,j,i,iens) = rv_upw*ru_upw/r_upw;
            state_flux_y(idV,k,j,i,iens) = rv_upw*rv_upw/r_upw + C0*std::pow(rt_upw,gamma);
            state_flux_y(idW,k,j,i,iens) = rv_upw*rw_upw/r_upw;
            state_flux_y(idT,k,j,i,iens) = rv_upw*rt_upw/r_upw;
            int uind = v > 0 ? 0 : 1;
            r_upw = state_limits_y(idR,uind,k,j,i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_flux_y(tr,k,j,i,iens) = rv_upw*tracers_limits_y(tr,uind,k,j,i,iens)/r_upw;
            }


            // state_flux_y(idR,k,j,i,iens) = 0.5*((r_R*v_R        )+(r_L*v_L        )-(std::abs(v)+cs)*((r_R    )-(r_L    )));
            // state_flux_y(idU,k,j,i,iens) = 0.5*((r_R*v_R*u_R    )+(r_L*v_L*u_L    )-(std::abs(v)+cs)*((r_R*u_R)-(r_L*u_L)));
            // state_flux_y(idV,k,j,i,iens) = 0.5*((r_R*v_R*v_R+p_R)+(r_L*v_L*v_L+p_L)-(std::abs(v)+cs)*((r_R*v_R)-(r_L*v_L)));
            // state_flux_y(idW,k,j,i,iens) = 0.5*((r_R*v_R*w_R    )+(r_L*v_L*w_L    )-(std::abs(v)+cs)*((r_R*w_R)-(r_L*w_L)));
            // state_flux_y(idT,k,j,i,iens) = 0.5*((r_R*v_R*t_R    )+(r_L*v_L*t_L    )-(std::abs(v)+cs)*((r_R*t_R)-(r_L*t_L)));
            // for (int tr=0; tr < num_tracers; tr++) {
            //   real t_L = tracers_limits_y(tr,0,k,j,i,iens)/state_limits_y(idR,0,k,j,i,iens);
            //   real t_R = tracers_limits_y(tr,1,k,j,i,iens)/state_limits_y(idR,1,k,j,i,iens);
            //   tracers_flux_y(tr,k,j,i,iens) = 0.5*((r_R*v_R*t_R)+(r_L*v_L*t_L)-(std::abs(v)+cs)*((r_R*t_R)-(r_L*t_L)));
            // }
          } else if (i < nx && k < nz) {
            state_flux_y(idR,k,j,i,iens) = 0;
            state_flux_y(idU,k,j,i,iens) = 0;
            state_flux_y(idV,k,j,i,iens) = 0;
            state_flux_y(idW,k,j,i,iens) = 0;
            state_flux_y(idT,k,j,i,iens) = 0;
            for (int tr=0; tr < num_tracers; tr++) { tracers_flux_y(tr,k,j,i,iens) = 0; }
          }
        });

        // Deallocate state and tracer limits because they are no longer needed
        state_limits_x    = real6d();
        state_limits_y    = real6d();
        state_limits_z    = real6d();
        tracers_limits_x  = real6d();
        tracers_limits_y  = real6d();
        tracers_limits_z  = real6d();

        real5d tracers_mult_x("tracers_mult_x",num_tracers,nz,ny,nx+1,nens);
        real5d tracers_mult_y("tracers_mult_y",num_tracers,nz,ny+1,nx,nens);
        real5d tracers_mult_z("tracers_mult_z",num_tracers,nz+1,ny,nx,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz+1,ny+1,nx+1,nens) ,
                                          YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
          if (j < ny && k < nz) tracers_mult_x(tr,k,j,i,iens) = 1;
          if (i < nx && k < nz) tracers_mult_y(tr,k,j,i,iens) = 1;
          if (i < nx && j < ny) tracers_mult_z(tr,k,j,i,iens) = 1;
        });

        // Flux Corrected Transport to enforce positivity for tracer species that must remain non-negative
        // This looks like it has a race condition, but it does not. Only one of the adjacent cells can ever change
        // a given edge flux because it's only changed if its sign oriented outward from a cell.
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
          if (tracer_positive(tr)) {
            real mass_available = max(tracers_tend(tr,k,j,i,iens),0._fp) * dx * dy * dz;
            real flux_out_y = ( max(tracers_flux_y(tr,k,j+1,i,iens),0._fp) - min(tracers_flux_y(tr,k,j,i,iens),0._fp) ) / dy;
            real mass_out = (flux_out_y) * dt * dx * dy * dz;
            if (mass_out > mass_available) {
              real mult = mass_available / mass_out;
              if (tracers_flux_y(tr,k,j+1,i,iens) > 0) { tracers_mult_y(tr,k,j+1,i,iens) = mult; }
              if (tracers_flux_y(tr,k,j  ,i,iens) < 0) { tracers_mult_y(tr,k,j  ,i,iens) = mult; }
            }
          }
        });

        fct_mult_exchange( coupler , tracers_mult_x , tracers_mult_y , tracers_mult_z );

        // Compute tendencies as the flux divergence + gravity source term
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          // Now overwrite state_tend and tracers_tend with this stage's tendencies
          for (int l = 0; l < num_state; l++) {
            state_tend(l,k,j,i,iens) = -( state_flux_y(l,k  ,j+1,i  ,iens) - state_flux_y(l,k,j,i,iens) ) / dy;
            if (l == idV && sim2d) state_tend(l,k,j,i,iens) = 0;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tend(l,k,j,i,iens) = -( tracers_flux_y(l,k  ,j+1,i  ,iens)*tracers_mult_y(l,k  ,j+1,i  ,iens) -
                                            tracers_flux_y(l,k  ,j  ,i  ,iens)*tracers_mult_y(l,k  ,j  ,i  ,iens) ) / dy;
          }
        });

      } else if (dir == DIR_Z) {

        // These arrays store high-order-accurate samples of the state and tracers at cell edges after cell-centered recon
        real6d state_limits_x   ("state_limits_x"   ,num_state  ,2,nz  ,ny  ,nx+1,nens);  state_limits_x    = 0;
        real6d state_limits_y   ("state_limits_y"   ,num_state  ,2,nz  ,ny+1,nx  ,nens);  state_limits_y    = 0;
        real6d state_limits_z   ("state_limits_z"   ,num_state  ,2,nz+1,ny  ,nx  ,nens);  state_limits_z    = 0;
        real6d tracers_limits_x ("tracers_limits_x" ,num_tracers,2,nz  ,ny  ,nx+1,nens);  tracers_limits_x  = 0;
        real6d tracers_limits_y ("tracers_limits_y" ,num_tracers,2,nz  ,ny+1,nx  ,nens);  tracers_limits_y  = 0;
        real6d tracers_limits_z ("tracers_limits_z" ,num_tracers,2,nz+1,ny  ,nx  ,nens);  tracers_limits_z  = 0;

        // Compute samples of state and tracers at cell edges using cell-centered reconstructions at high-order with WENO
        // At the end of this, we will have two samples per cell edge in each dimension, one from each adjacent cell.
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          real cs = 350;
          real u  = state(idU,hs+k,hs+j,hs+i,iens);
          real v  = state(idV,hs+k,hs+j,hs+i,iens);
          real w  = state(idW,hs+k,hs+j,hs+i,iens);
          real t  = state(idT,hs+k,hs+j,hs+i,iens);
          ////////////////////////////////////////////////////////
          // Z-direction
          ////////////////////////////////////////////////////////
          // State
          {
            SArray<real,1,ord> stencil;
            SArray<real,1,2  > g1, g2, g3, g4, g5;
            for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idR,k+kk,hs+j,hs+i,iens); }
            reconstruct_gll_values(stencil,g1,coefs_to_gll,limiter);
            for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idU,k+kk,hs+j,hs+i,iens); }
            reconstruct_gll_values(stencil,g2,coefs_to_gll,limiter);
            for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idV,k+kk,hs+j,hs+i,iens); }
            reconstruct_gll_values(stencil,g3,coefs_to_gll,limiter);
            for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idW,k+kk,hs+j,hs+i,iens); }
            reconstruct_gll_values(stencil,g4,coefs_to_gll,limiter);
            for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idT,k+kk,hs+j,hs+i,iens); }
            reconstruct_gll_values(stencil,g5,coefs_to_gll,limiter);

            real r  = g1(0);
            real ru = g2(0);
            real rv = g3(0);
            real rw = g4(0);
            real rt = g5(0);
            // r  = std::max(std::min(state(idR,hs+k-1,hs+j,hs+i,iens),state(idR,hs+k,hs+j,hs+i,iens)),r );
            // ru = std::max(std::min(state(idU,hs+k-1,hs+j,hs+i,iens),state(idU,hs+k,hs+j,hs+i,iens)),ru);
            // rv = std::max(std::min(state(idV,hs+k-1,hs+j,hs+i,iens),state(idV,hs+k,hs+j,hs+i,iens)),rv);
            // rw = std::max(std::min(state(idW,hs+k-1,hs+j,hs+i,iens),state(idW,hs+k,hs+j,hs+i,iens)),rw);
            // rt = std::max(std::min(state(idT,hs+k-1,hs+j,hs+i,iens),state(idT,hs+k,hs+j,hs+i,iens)),rt);
            // r  = std::min(std::max(state(idR,hs+k-1,hs+j,hs+i,iens),state(idR,hs+k,hs+j,hs+i,iens)),r );
            // ru = std::min(std::max(state(idU,hs+k-1,hs+j,hs+i,iens),state(idU,hs+k,hs+j,hs+i,iens)),ru);
            // rv = std::min(std::max(state(idV,hs+k-1,hs+j,hs+i,iens),state(idV,hs+k,hs+j,hs+i,iens)),rv);
            // rw = std::min(std::max(state(idW,hs+k-1,hs+j,hs+i,iens),state(idW,hs+k,hs+j,hs+i,iens)),rw);
            // rt = std::min(std::max(state(idT,hs+k-1,hs+j,hs+i,iens),state(idT,hs+k,hs+j,hs+i,iens)),rt);
            state_limits_z(idR,1,k  ,j,i,iens) = r ;
            state_limits_z(idU,1,k  ,j,i,iens) = ru;
            state_limits_z(idV,1,k  ,j,i,iens) = rv;
            state_limits_z(idW,1,k  ,j,i,iens) = rw;
            state_limits_z(idT,1,k  ,j,i,iens) = rt;

            r  = g1(1);
            ru = g2(1);
            rv = g3(1);
            rw = g4(1);
            rt = g5(1);
            // r  = std::max(std::min(state(idR,hs+k,hs+j,hs+i,iens),state(idR,hs+k+1,hs+j,hs+i,iens)),r );
            // ru = std::max(std::min(state(idU,hs+k,hs+j,hs+i,iens),state(idU,hs+k+1,hs+j,hs+i,iens)),ru);
            // rv = std::max(std::min(state(idV,hs+k,hs+j,hs+i,iens),state(idV,hs+k+1,hs+j,hs+i,iens)),rv);
            // rw = std::max(std::min(state(idW,hs+k,hs+j,hs+i,iens),state(idW,hs+k+1,hs+j,hs+i,iens)),rw);
            // rt = std::max(std::min(state(idT,hs+k,hs+j,hs+i,iens),state(idT,hs+k+1,hs+j,hs+i,iens)),rt);
            // r  = std::min(std::max(state(idR,hs+k,hs+j,hs+i,iens),state(idR,hs+k+1,hs+j,hs+i,iens)),r );
            // ru = std::min(std::max(state(idU,hs+k,hs+j,hs+i,iens),state(idU,hs+k+1,hs+j,hs+i,iens)),ru);
            // rv = std::min(std::max(state(idV,hs+k,hs+j,hs+i,iens),state(idV,hs+k+1,hs+j,hs+i,iens)),rv);
            // rw = std::min(std::max(state(idW,hs+k,hs+j,hs+i,iens),state(idW,hs+k+1,hs+j,hs+i,iens)),rw);
            // rt = std::min(std::max(state(idT,hs+k,hs+j,hs+i,iens),state(idT,hs+k+1,hs+j,hs+i,iens)),rt);
            state_limits_z(idR,0,k+1,j,i,iens) = r ;
            state_limits_z(idU,0,k+1,j,i,iens) = ru;
            state_limits_z(idV,0,k+1,j,i,iens) = rv;
            state_limits_z(idW,0,k+1,j,i,iens) = rw;
            state_limits_z(idT,0,k+1,j,i,iens) = rt;
          }
          // Tracers
          for (int l=0; l < num_tracers; l++) {
            // Gather the stencil of cell averages, and use WENO to compute values at the cell edges (i.e., 2 GLL points)
            SArray<real,1,ord> stencil;
            SArray<real,1,2>   gll;
            for (int s=0; s < ord; s++) { stencil(s) = tracers(l,k+s,hs+j,hs+i,iens); }
            reconstruct_gll_values(stencil,gll,coefs_to_gll,limiter);
            tracers_limits_z(l,1,k  ,j,i,iens) = gll(0);
            tracers_limits_z(l,0,k+1,j,i,iens) = gll(1);
          }
        });

        edge_exchange( coupler           ,
                       state_limits_x    ,
                       tracers_limits_x  ,
                       state_limits_y    ,
                       tracers_limits_y  ,
                       state_limits_z    ,
                       tracers_limits_z  );

        real5d state_flux_x  ("state_flux_x"  ,num_state  ,nz  ,ny  ,nx+1,nens);
        real5d state_flux_y  ("state_flux_y"  ,num_state  ,nz  ,ny+1,nx  ,nens);
        real5d state_flux_z  ("state_flux_z"  ,num_state  ,nz+1,ny  ,nx  ,nens);
        real5d tracers_flux_x("tracers_flux_x",num_tracers,nz  ,ny  ,nx+1,nens);
        real5d tracers_flux_y("tracers_flux_y",num_tracers,nz  ,ny+1,nx  ,nens);
        real5d tracers_flux_z("tracers_flux_z",num_tracers,nz+1,ny  ,nx  ,nens);

        auto save_pressure_z = coupler.get_option<bool>("save_pressure_z",false);
        real4d pressure_z;
        if (save_pressure_z) pressure_z = coupler.get_data_manager_readwrite().get<real,4>("pressure_z");

        // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz+1,ny+1,nx+1,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          // Z-direction
          if (i < nx && j < ny) {
            real r_L = state_limits_z(idR,0,k,j,i,iens)    ;    real r_R = state_limits_z(idR,1,k,j,i,iens)    ;
            real u_L = state_limits_z(idU,0,k,j,i,iens)/r_L;    real u_R = state_limits_z(idU,1,k,j,i,iens)/r_R;
            real v_L = state_limits_z(idV,0,k,j,i,iens)/r_L;    real v_R = state_limits_z(idV,1,k,j,i,iens)/r_R;
            real w_L = state_limits_z(idW,0,k,j,i,iens)/r_L;    real w_R = state_limits_z(idW,1,k,j,i,iens)/r_R;
            real t_L = state_limits_z(idT,0,k,j,i,iens)/r_L;    real t_R = state_limits_z(idT,1,k,j,i,iens)/r_R;
            real p_L = C0*std::pow(r_L*t_L,gamma)          ;    real p_R = C0*std::pow(r_R*t_R,gamma)          ;
            real r = 0.5_fp*(r_L+r_R);
            real u = 0.5_fp*(u_L+u_R);
            real v = 0.5_fp*(v_L+v_R);
            real w = 0.5_fp*(w_L+w_R);
            real t = 0.5_fp*(t_L+t_R);
            real p = 0.5_fp*(p_L+p_R);
            real cs2 = gamma*p/r;
            real cs  = std::sqrt(cs2);
            real w1 =  w*(r_R)/(2*cs) - (r_R*w_R)/(2*cs) + (r_R*t_R)/(2*t);
            real w2 = -w*(r_L)/(2*cs) + (r_L*w_L)/(2*cs) + (r_L*t_L)/(2*t);
            real w3, w4, w5;
            if (w > 0) {
              w3 = (r_L)     -   (r_L*t_L)/t;
              w4 = (r_L*u_L) - u*(r_L*t_L)/t;
              w5 = (r_L*v_L) - v*(r_L*t_L)/t;
            } else {
              w3 = (r_R)     -   (r_R*t_R)/t;
              w4 = (r_R*u_R) - u*(r_R*t_R)/t;
              w5 = (r_R*v_R) - v*(r_R*t_R)/t;
            }
            real r_upw  = w1        + w2        + w3            ;
            real ru_upw = w1*u      + w2*u             + w4     ;
            real rv_upw = w1*v      + w2*v                  + w5;
            real rw_upw = w1*(w-cs) + w2*(w+cs) + w3*w          ;
            real rt_upw = w1*t      + w2*t                      ;
            state_flux_z(idR,k,j,i,iens) = rw_upw;
            state_flux_z(idU,k,j,i,iens) = rw_upw*ru_upw/r_upw;
            state_flux_z(idV,k,j,i,iens) = rw_upw*rv_upw/r_upw;
            state_flux_z(idW,k,j,i,iens) = rw_upw*rw_upw/r_upw + C0*std::pow(rt_upw,gamma);
            state_flux_z(idT,k,j,i,iens) = rw_upw*rt_upw/r_upw;
            int uind = w > 0 ? 0 : 1;
            r_upw = state_limits_z(idR,uind,k,j,i,iens);
            for (int tr=0; tr < num_tracers; tr++) {
              tracers_flux_z(tr,k,j,i,iens) = rw_upw*tracers_limits_z(tr,uind,k,j,i,iens)/r_upw;
            }


            // state_flux_z(idR,k,j,i,iens) = 0.5*((r_R*w_R        )+(r_L*w_L        )-(std::abs(w)+cs)*((r_R    )-(r_L    )));
            // state_flux_z(idU,k,j,i,iens) = 0.5*((r_R*w_R*u_R    )+(r_L*w_L*u_L    )-(std::abs(w)+cs)*((r_R*u_R)-(r_L*u_L)));
            // state_flux_z(idV,k,j,i,iens) = 0.5*((r_R*w_R*v_R    )+(r_L*w_L*v_L    )-(std::abs(w)+cs)*((r_R*v_R)-(r_L*v_L)));
            // state_flux_z(idW,k,j,i,iens) = 0.5*((r_R*w_R*w_R+p_R)+(r_L*w_L*w_L+p_L)-(std::abs(w)+cs)*((r_R*w_R)-(r_L*w_L)));
            // state_flux_z(idT,k,j,i,iens) = 0.5*((r_R*w_R*t_R    )+(r_L*w_L*t_L    )-(std::abs(w)+cs)*((r_R*t_R)-(r_L*t_L)));
            // for (int tr=0; tr < num_tracers; tr++) {
            //   real t_L = tracers_limits_z(tr,0,k,j,i,iens)/state_limits_z(idR,0,k,j,i,iens);
            //   real t_R = tracers_limits_z(tr,1,k,j,i,iens)/state_limits_z(idR,1,k,j,i,iens);
            //   tracers_flux_z(tr,k,j,i,iens) = 0.5*((r_R*w_R*t_R)+(r_L*w_L*t_L)-(std::abs(w)+cs)*((r_R*t_R)-(r_L*t_L)));
            // }
          }
        });

        // Deallocate state and tracer limits because they are no longer needed
        state_limits_x    = real6d();
        state_limits_y    = real6d();
        state_limits_z    = real6d();
        tracers_limits_x  = real6d();
        tracers_limits_y  = real6d();
        tracers_limits_z  = real6d();

        real5d tracers_mult_x("tracers_mult_x",num_tracers,nz,ny,nx+1,nens);
        real5d tracers_mult_y("tracers_mult_y",num_tracers,nz,ny+1,nx,nens);
        real5d tracers_mult_z("tracers_mult_z",num_tracers,nz+1,ny,nx,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz+1,ny+1,nx+1,nens) ,
                                          YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
          if (j < ny && k < nz) tracers_mult_x(tr,k,j,i,iens) = 1;
          if (i < nx && k < nz) tracers_mult_y(tr,k,j,i,iens) = 1;
          if (i < nx && j < ny) tracers_mult_z(tr,k,j,i,iens) = 1;
        });

        // Flux Corrected Transport to enforce positivity for tracer species that must remain non-negative
        // This looks like it has a race condition, but it does not. Only one of the adjacent cells can ever change
        // a given edge flux because it's only changed if its sign oriented outward from a cell.
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_tracers,nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int tr, int k, int j, int i, int iens) {
          if (tracer_positive(tr)) {
            real mass_available = max(tracers_tend(tr,k,j,i,iens),0._fp) * dx * dy * dz;
            real flux_out_z = ( max(tracers_flux_z(tr,k+1,j,i,iens),0._fp) - min(tracers_flux_z(tr,k,j,i,iens),0._fp) ) / dz;
            real mass_out = (flux_out_z) * dt * dx * dy * dz;
            if (mass_out > mass_available) {
              real mult = mass_available / mass_out;
              if (tracers_flux_z(tr,k+1,j,i,iens) > 0) { tracers_mult_z(tr,k+1,j,i,iens) = mult; }
              if (tracers_flux_z(tr,k  ,j,i,iens) < 0) { tracers_mult_z(tr,k  ,j,i,iens) = mult; }
            }
          }
        });

        fct_mult_exchange( coupler , tracers_mult_x , tracers_mult_y , tracers_mult_z );

        // Compute tendencies as the flux divergence + gravity source term
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          // Now overwrite state_tend and tracers_tend with this stage's tendencies
          for (int l = 0; l < num_state; l++) {
            state_tend(l,k,j,i,iens) = -( state_flux_z(l,k+1,j  ,i  ,iens) - state_flux_z(l,k,j,i,iens) ) / dz;
            if (l == idW && enable_gravity) state_tend(l,k,j,i,iens) += -grav * state(idR,hs+k,hs+j,hs+i,iens);
            if (l == idV && sim2d) state_tend(l,k,j,i,iens) = 0;
          }
          for (int l = 0; l < num_tracers; l++) {
            tracers_tend(l,k,j,i,iens) = -( tracers_flux_z(l,k+1,j  ,i  ,iens)*tracers_mult_z(l,k+1,j  ,i  ,iens) -
                                            tracers_flux_z(l,k  ,j  ,i  ,iens)*tracers_mult_z(l,k  ,j  ,i  ,iens) ) / dz;
          }
        });

      }


    }



    // ord stencil cell averages to two GLL point values via high-order reconstruction and WENO limiting
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>     const &stencil      ,
                                                    SArray<real,1,2>             &gll          ,
                                                    SArray<real,2,ord,2>   const &coefs_to_gll ,
                                                    weno::WenoLimiter<ord> const &limiter    ) {
      // Reconstruct values
      SArray<real,1,ord> wenoCoefs;
      limiter.compute_limited_coefs( stencil , wenoCoefs );
      // Transform ord weno coefficients into 2 GLL points
      for (int ii=0; ii<2; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) {
          tmp += coefs_to_gll(s,ii) * wenoCoefs(s);
        }
        gll(ii) = tmp;
      }
    }


    void halo_exchange( core::Coupler const & coupler  ,
                        real5d        const & state    ,
                        real5d        const & tracers  ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto dz          = coupler.get_dz();
      auto num_tracers = coupler.get_num_tracers();
      auto sim2d       = coupler.is_sim2d();
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();
      auto bc_x        = coupler.get_option<int>("bc_x");
      auto bc_y        = coupler.get_option<int>("bc_y");
      auto bc_z        = coupler.get_option<int>("bc_z");
      auto grav        = coupler.get_option<real>("grav");
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0");

      int npack = num_state + num_tracers;

      realHost5d halo_send_buf_W_host("halo_send_buf_W_host",npack,nz,ny,hs,nens);
      realHost5d halo_send_buf_E_host("halo_send_buf_E_host",npack,nz,ny,hs,nens);
      realHost5d halo_send_buf_S_host("halo_send_buf_S_host",npack,nz,hs,nx,nens);
      realHost5d halo_send_buf_N_host("halo_send_buf_N_host",npack,nz,hs,nx,nens);
      realHost5d halo_recv_buf_S_host("halo_recv_buf_S_host",npack,nz,hs,nx,nens);
      realHost5d halo_recv_buf_N_host("halo_recv_buf_N_host",npack,nz,hs,nx,nens);
      realHost5d halo_recv_buf_W_host("halo_recv_buf_W_host",npack,nz,ny,hs,nens);
      realHost5d halo_recv_buf_E_host("halo_recv_buf_E_host",npack,nz,ny,hs,nens);
      real5d     halo_send_buf_W     ("halo_send_buf_W"     ,npack,nz,ny,hs,nens);
      real5d     halo_send_buf_E     ("halo_send_buf_E"     ,npack,nz,ny,hs,nens);
      real5d     halo_send_buf_S     ("halo_send_buf_S"     ,npack,nz,hs,nx,nens);
      real5d     halo_send_buf_N     ("halo_send_buf_N"     ,npack,nz,hs,nx,nens);
      real5d     halo_recv_buf_W     ("halo_recv_buf_W"     ,npack,nz,ny,hs,nens);
      real5d     halo_recv_buf_E     ("halo_recv_buf_E"     ,npack,nz,ny,hs,nens);
      real5d     halo_recv_buf_S     ("halo_recv_buf_S"     ,npack,nz,hs,nx,nens);
      real5d     halo_recv_buf_N     ("halo_recv_buf_N"     ,npack,nz,hs,nx,nens);

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

      if (!sim2d) {
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
      }

      yakl::fence();
      yakl::timer_start("halo_exchange_mpi");

      MPI_Request sReq[4];
      MPI_Request rReq[4];

      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto dtype = coupler.get_mpi_data_type();
      auto comm = MPI_COMM_WORLD;

      #ifdef MW_GPU_AWARE_MPI
        yakl::fence();

        //Pre-post the receives
        MPI_Irecv( halo_recv_buf_W.data() , halo_recv_buf_W.size() , dtype , neigh(1,0) , 0 , comm , &rReq[0] );
        MPI_Irecv( halo_recv_buf_E.data() , halo_recv_buf_E.size() , dtype , neigh(1,2) , 1 , comm , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( halo_recv_buf_S.data() , halo_recv_buf_S.size() , dtype , neigh(0,1) , 2 , comm , &rReq[2] );
          MPI_Irecv( halo_recv_buf_N.data() , halo_recv_buf_N.size() , dtype , neigh(2,1) , 3 , comm , &rReq[3] );
        }

        //Send the data
        MPI_Isend( halo_send_buf_W.data() , halo_send_buf_W.size() , dtype , neigh(1,0) , 1 , comm , &sReq[0] );
        MPI_Isend( halo_send_buf_E.data() , halo_send_buf_E.size() , dtype , neigh(1,2) , 0 , comm , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( halo_send_buf_S.data() , halo_send_buf_S.size() , dtype , neigh(0,1) , 3 , comm , &sReq[2] );
          MPI_Isend( halo_send_buf_N.data() , halo_send_buf_N.size() , dtype , neigh(2,1) , 2 , comm , &sReq[3] );
        }

        MPI_Status  sStat[4];
        MPI_Status  rStat[4];

        //Wait for the sends and receives to finish
        if (sim2d) {
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
        } else {
          MPI_Waitall(4, sReq, sStat);
          MPI_Waitall(4, rReq, rStat);
        }
        yakl::timer_stop("halo_exchange_mpi");
      #else
        //Pre-post the receives
        MPI_Irecv( halo_recv_buf_W_host.data() , halo_recv_buf_W_host.size() , dtype , neigh(1,0) , 0 , comm , &rReq[0] );
        MPI_Irecv( halo_recv_buf_E_host.data() , halo_recv_buf_E_host.size() , dtype , neigh(1,2) , 1 , comm , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( halo_recv_buf_S_host.data() , halo_recv_buf_S_host.size() , dtype , neigh(0,1) , 2 , comm , &rReq[2] );
          MPI_Irecv( halo_recv_buf_N_host.data() , halo_recv_buf_N_host.size() , dtype , neigh(2,1) , 3 , comm , &rReq[3] );
        }

        halo_send_buf_W.deep_copy_to(halo_send_buf_W_host);
        halo_send_buf_E.deep_copy_to(halo_send_buf_E_host);
        if (!sim2d) {
          halo_send_buf_S.deep_copy_to(halo_send_buf_S_host);
          halo_send_buf_N.deep_copy_to(halo_send_buf_N_host);
        }

        yakl::fence();

        //Send the data
        MPI_Isend( halo_send_buf_W_host.data() , halo_send_buf_W_host.size() , dtype , neigh(1,0) , 1 , comm , &sReq[0] );
        MPI_Isend( halo_send_buf_E_host.data() , halo_send_buf_E_host.size() , dtype , neigh(1,2) , 0 , comm , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( halo_send_buf_S_host.data() , halo_send_buf_S_host.size() , dtype , neigh(0,1) , 3 , comm , &sReq[2] );
          MPI_Isend( halo_send_buf_N_host.data() , halo_send_buf_N_host.size() , dtype , neigh(2,1) , 2 , comm , &sReq[3] );
        }

        MPI_Status  sStat[4];
        MPI_Status  rStat[4];

        //Wait for the sends and receives to finish
        if (sim2d) {
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
        } else {
          MPI_Waitall(4, sReq, sStat);
          MPI_Waitall(4, rReq, rStat);
        }
        yakl::timer_stop("halo_exchange_mpi");

        halo_recv_buf_W_host.deep_copy_to(halo_recv_buf_W);
        halo_recv_buf_E_host.deep_copy_to(halo_recv_buf_E);
        if (!sim2d) {
          halo_recv_buf_S_host.deep_copy_to(halo_recv_buf_S);
          halo_recv_buf_N_host.deep_copy_to(halo_recv_buf_N);
        }
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

      if (!sim2d) {
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

      ////////////////////////////////////
      // Begin boundary conditions
      ////////////////////////////////////
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
            if ((l == idW || l == idV || l == idU) && bc_z == BC_WALL) {
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
        });
      }
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
      auto sim2d       = coupler.is_sim2d();
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();
      auto bc_x        = coupler.get_option<int>("bc_x");
      auto bc_y        = coupler.get_option<int>("bc_y");
      auto bc_z        = coupler.get_option<int>("bc_z");

      int npack = num_state + num_tracers;

      realHost4d edge_send_buf_S_host("edge_send_buf_S_host",npack,nz,nx,nens);
      realHost4d edge_send_buf_N_host("edge_send_buf_N_host",npack,nz,nx,nens);
      realHost4d edge_send_buf_W_host("edge_send_buf_W_host",npack,nz,ny,nens);
      realHost4d edge_send_buf_E_host("edge_send_buf_E_host",npack,nz,ny,nens);
      realHost4d edge_recv_buf_S_host("edge_recv_buf_S_host",npack,nz,nx,nens);
      realHost4d edge_recv_buf_N_host("edge_recv_buf_N_host",npack,nz,nx,nens);
      realHost4d edge_recv_buf_W_host("edge_recv_buf_W_host",npack,nz,ny,nens);
      realHost4d edge_recv_buf_E_host("edge_recv_buf_E_host",npack,nz,ny,nens);
      real4d     edge_send_buf_W     ("edge_send_buf_W"     ,npack,nz,ny,nens);
      real4d     edge_send_buf_E     ("edge_send_buf_E"     ,npack,nz,ny,nens);
      real4d     edge_send_buf_S     ("edge_send_buf_S"     ,npack,nz,nx,nens);
      real4d     edge_send_buf_N     ("edge_send_buf_N"     ,npack,nz,nx,nens);
      real4d     edge_recv_buf_W     ("edge_recv_buf_W"     ,npack,nz,ny,nens);
      real4d     edge_recv_buf_E     ("edge_recv_buf_E"     ,npack,nz,ny,nens);
      real4d     edge_recv_buf_S     ("edge_recv_buf_S"     ,npack,nz,nx,nens);
      real4d     edge_recv_buf_N     ("edge_recv_buf_N"     ,npack,nz,nx,nens);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,nens) , YAKL_LAMBDA (int v, int k, int j, int iens) {
        if        (v < num_state) {
          edge_send_buf_W(v,k,j,iens) = state_limits_x  (v          ,1,k,j,0 ,iens);
          edge_send_buf_E(v,k,j,iens) = state_limits_x  (v          ,0,k,j,nx,iens);
        } else if (v < num_state + num_tracers) {
          edge_send_buf_W(v,k,j,iens) = tracers_limits_x(v-num_state,1,k,j,0 ,iens);
          edge_send_buf_E(v,k,j,iens) = tracers_limits_x(v-num_state,0,k,j,nx,iens);
        }
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int i, int iens) {
          if        (v < num_state) {
            edge_send_buf_S(v,k,i,iens) = state_limits_y  (v          ,1,k,0 ,i,iens);
            edge_send_buf_N(v,k,i,iens) = state_limits_y  (v          ,0,k,ny,i,iens);
          } else if (v < num_state + num_tracers) {
            edge_send_buf_S(v,k,i,iens) = tracers_limits_y(v-num_state,1,k,0 ,i,iens);
            edge_send_buf_N(v,k,i,iens) = tracers_limits_y(v-num_state,0,k,ny,i,iens);
          }
        });
      }

      yakl::fence();
      yakl::timer_start("edge_exchange_mpi");

      MPI_Request sReq[4];
      MPI_Request rReq[4];

      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto dtype = coupler.get_mpi_data_type();
      auto comm = MPI_COMM_WORLD;

      #ifdef MW_GPU_AWARE_MPI
        yakl::fence();

        //Pre-post the receives
        MPI_Irecv( edge_recv_buf_W.data() , edge_recv_buf_W.size() , dtype , neigh(1,0) , 4 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E.data() , edge_recv_buf_E.size() , dtype , neigh(1,2) , 5 , comm , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( edge_recv_buf_S.data() , edge_recv_buf_S.size() , dtype , neigh(0,1) , 6 , comm , &rReq[2] );
          MPI_Irecv( edge_recv_buf_N.data() , edge_recv_buf_N.size() , dtype , neigh(2,1) , 7 , comm , &rReq[3] );
        }

        //Send the data
        MPI_Isend( edge_send_buf_W.data() , edge_send_buf_W.size() , dtype , neigh(1,0) , 5 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_E.data() , edge_send_buf_E.size() , dtype , neigh(1,2) , 4 , comm , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( edge_send_buf_S.data() , edge_send_buf_S.size() , dtype , neigh(0,1) , 7 , comm , &sReq[2] );
          MPI_Isend( edge_send_buf_N.data() , edge_send_buf_N.size() , dtype , neigh(2,1) , 6 , comm , &sReq[3] );
        }

        MPI_Status  sStat[4];
        MPI_Status  rStat[4];

        //Wait for the sends and receives to finish
        if (sim2d) {
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
        } else {
          MPI_Waitall(4, sReq, sStat);
          MPI_Waitall(4, rReq, rStat);
        }
        yakl::timer_stop("edge_exchange_mpi");
      #else
        //Pre-post the receives
        MPI_Irecv( edge_recv_buf_W_host.data() , edge_recv_buf_W_host.size() , dtype , neigh(1,0) , 4 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E_host.data() , edge_recv_buf_E_host.size() , dtype , neigh(1,2) , 5 , comm , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( edge_recv_buf_S_host.data() , edge_recv_buf_S_host.size() , dtype , neigh(0,1) , 6 , comm , &rReq[2] );
          MPI_Irecv( edge_recv_buf_N_host.data() , edge_recv_buf_N_host.size() , dtype , neigh(2,1) , 7 , comm , &rReq[3] );
        }

        edge_send_buf_W.deep_copy_to(edge_send_buf_W_host);
        edge_send_buf_E.deep_copy_to(edge_send_buf_E_host);
        if (!sim2d) {
          edge_send_buf_S.deep_copy_to(edge_send_buf_S_host);
          edge_send_buf_N.deep_copy_to(edge_send_buf_N_host);
        }

        yakl::fence();

        //Send the data
        MPI_Isend( edge_send_buf_W_host.data() , edge_send_buf_W_host.size() , dtype , neigh(1,0) , 5 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_E_host.data() , edge_send_buf_E_host.size() , dtype , neigh(1,2) , 4 , comm , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( edge_send_buf_S_host.data() , edge_send_buf_S_host.size() , dtype , neigh(0,1) , 7 , comm , &sReq[2] );
          MPI_Isend( edge_send_buf_N_host.data() , edge_send_buf_N_host.size() , dtype , neigh(2,1) , 6 , comm , &sReq[3] );
        }

        MPI_Status  sStat[4];
        MPI_Status  rStat[4];

        //Wait for the sends and receives to finish
        if (sim2d) {
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
        } else {
          MPI_Waitall(4, sReq, sStat);
          MPI_Waitall(4, rReq, rStat);
        }
        yakl::timer_stop("edge_exchange_mpi");

        edge_recv_buf_W_host.deep_copy_to(edge_recv_buf_W);
        edge_recv_buf_E_host.deep_copy_to(edge_recv_buf_E);
        if (!sim2d) {
          edge_recv_buf_S_host.deep_copy_to(edge_recv_buf_S);
          edge_recv_buf_N_host.deep_copy_to(edge_recv_buf_N);
        }
      #endif

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,nens) ,
                                        YAKL_LAMBDA (int v, int k, int j, int iens) {
        if        (v < num_state) {
          state_limits_x  (v          ,0,k,j,0 ,iens) = edge_recv_buf_W(v,k,j,iens);
          state_limits_x  (v          ,1,k,j,nx,iens) = edge_recv_buf_E(v,k,j,iens);
        } else if (v < num_state + num_tracers) {
          tracers_limits_x(v-num_state,0,k,j,0 ,iens) = edge_recv_buf_W(v,k,j,iens);
          tracers_limits_x(v-num_state,1,k,j,nx,iens) = edge_recv_buf_E(v,k,j,iens);
        }
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int i, int iens) {
          if        (v < num_state) {
            state_limits_y  (v          ,0,k,0 ,i,iens) = edge_recv_buf_S(v,k,i,iens);
            state_limits_y  (v          ,1,k,ny,i,iens) = edge_recv_buf_N(v,k,i,iens);
          } else if (v < num_state + num_tracers) {
            tracers_limits_y(v-num_state,0,k,0 ,i,iens) = edge_recv_buf_S(v,k,i,iens);
            tracers_limits_y(v-num_state,1,k,ny,i,iens) = edge_recv_buf_N(v,k,i,iens);
          }
        });
      }

      /////////////////////////////////
      // Begin boundary conditions
      /////////////////////////////////
      if (bc_z == BC_PERIODIC) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(ny,nx,nens) ,
                                          YAKL_LAMBDA (int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            state_limits_z(l,0,0 ,j,i,iens) = state_limits_z(l,0,nz,j,i,iens);
            state_limits_z(l,1,nz,j,i,iens) = state_limits_z(l,1,0 ,j,i,iens);
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(l,0,0 ,j,i,iens) = tracers_limits_z(l,0,nz,j,i,iens);
            tracers_limits_z(l,1,nz,j,i,iens) = tracers_limits_z(l,1,0 ,j,i,iens);
          }
        });
      } else if (bc_z == BC_WALL || bc_z == BC_OPEN) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(ny,nx,nens) ,
                                          YAKL_LAMBDA (int j, int i, int iens) {
          for (int l=0; l < num_state; l++) {
            if (l == idW && bc_z == BC_WALL) {
              state_limits_z(l,0,0 ,j,i,iens) = 0;
              state_limits_z(l,1,0 ,j,i,iens) = 0;
              state_limits_z(l,0,nz,j,i,iens) = 0;
              state_limits_z(l,1,nz,j,i,iens) = 0;
            } else {
              state_limits_z(l,0,0 ,j,i,iens) = state_limits_z(l,1,0 ,j,i,iens);
              state_limits_z(l,1,nz,j,i,iens) = state_limits_z(l,0,nz,j,i,iens);
            }
          }
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(l,0,0 ,j,i,iens) = tracers_limits_z(l,1,0 ,j,i,iens);
            tracers_limits_z(l,1,nz,j,i,iens) = tracers_limits_z(l,0,nz,j,i,iens);
          }
        });
      }
      if (bc_x == BC_WALL || bc_x == BC_OPEN) {
        if (px == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nens) ,
                                            YAKL_LAMBDA (int k, int j, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state_limits_x(l,0,k,j,0,iens) = 0; state_limits_x(l,1,k,j,0,iens) = 0; }
              else                             { state_limits_x(l,0,k,j,0,iens) = state_limits_x(l,1,k,j,0,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_x(l,0,k,j,0,iens) = tracers_limits_x(l,1,k,j,0,iens); }
          });
        } else if (px == nproc_x-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nens) ,
                                            YAKL_LAMBDA (int k, int j, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idU && bc_x == BC_WALL) { state_limits_x(l,0,k,j,nx,iens) = 0; state_limits_x(l,1,k,j,nx,iens) = 0; }
              else                             { state_limits_x(l,1,k,j,nx,iens) = state_limits_x(l,0,k,j,nx,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_x(l,1,k,j,nx,iens) = tracers_limits_x(l,0,k,j,nx,iens); }
          });
        }
      }
      if (bc_y == BC_WALL || bc_y == BC_OPEN) {
        if (py == 0) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,nx,nens) ,
                                            YAKL_LAMBDA (int k, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state_limits_y(l,0,k,0,i,iens) = 0; state_limits_y(l,1,k,0,i,iens) = 0; }
              else                             { state_limits_y(l,0,k,0,i,iens) = state_limits_y(l,1,k,0,i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_y(l,0,k,0,i,iens) = tracers_limits_y(l,1,k,0,i,iens); }
          });
        } else if (py == nproc_y-1) {
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,nx,nens) ,
                                            YAKL_LAMBDA (int k, int i, int iens) {
            for (int l=0; l < num_state; l++) {
              if (l == idV && bc_y == BC_WALL) { state_limits_y(l,0,k,ny,i,iens) = 0; state_limits_y(l,1,k,ny,i,iens) = 0; }
              else                             { state_limits_y(l,1,k,ny,i,iens) = state_limits_y(l,0,k,ny,i,iens); }
            }
            for (int l=0; l < num_tracers; l++) { tracers_limits_y(l,1,k,ny,i,iens) = tracers_limits_y(l,0,k,ny,i,iens); }
          });
        }
      }
    }



    void fct_mult_exchange(core::Coupler const &coupler , real5d const &tracers_mult_x ,
                                                          real5d const &tracers_mult_y ,
                                                          real5d const &tracers_mult_z ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto num_tracers = coupler.get_num_tracers();
      auto sim2d       = coupler.is_sim2d();
      auto px          = coupler.get_px();
      auto py          = coupler.get_py();
      auto nproc_x     = coupler.get_nproc_x();
      auto nproc_y     = coupler.get_nproc_y();
      auto bc_x        = coupler.get_option<int>("bc_x");
      auto bc_y        = coupler.get_option<int>("bc_y");
      auto bc_z        = coupler.get_option<int>("bc_z");

      int npack = num_tracers;
      realHost4d edge_send_buf_S_host("edge_send_buf_S_host",npack,nz,nx,nens);
      realHost4d edge_send_buf_N_host("edge_send_buf_N_host",npack,nz,nx,nens);
      realHost4d edge_send_buf_W_host("edge_send_buf_W_host",npack,nz,ny,nens);
      realHost4d edge_send_buf_E_host("edge_send_buf_E_host",npack,nz,ny,nens);
      realHost4d edge_recv_buf_S_host("edge_recv_buf_S_host",npack,nz,nx,nens);
      realHost4d edge_recv_buf_N_host("edge_recv_buf_N_host",npack,nz,nx,nens);
      realHost4d edge_recv_buf_W_host("edge_recv_buf_W_host",npack,nz,ny,nens);
      realHost4d edge_recv_buf_E_host("edge_recv_buf_E_host",npack,nz,ny,nens);
      real4d     edge_send_buf_W     ("edge_send_buf_W"     ,npack,nz,ny,nens);
      real4d     edge_send_buf_E     ("edge_send_buf_E"     ,npack,nz,ny,nens);
      real4d     edge_send_buf_S     ("edge_send_buf_S"     ,npack,nz,nx,nens);
      real4d     edge_send_buf_N     ("edge_send_buf_N"     ,npack,nz,nx,nens);
      real4d     edge_recv_buf_W     ("edge_recv_buf_W"     ,npack,nz,ny,nens);
      real4d     edge_recv_buf_E     ("edge_recv_buf_E"     ,npack,nz,ny,nens);
      real4d     edge_recv_buf_S     ("edge_recv_buf_S"     ,npack,nz,nx,nens);
      real4d     edge_recv_buf_N     ("edge_recv_buf_N"     ,npack,nz,nx,nens);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,nens) ,
                                        YAKL_LAMBDA (int v, int k, int j, int iens) {
        edge_send_buf_W(v,k,j,iens) = tracers_mult_x(v,k,j,0 ,iens);
        edge_send_buf_E(v,k,j,iens) = tracers_mult_x(v,k,j,nx,iens);
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int i, int iens) {
          edge_send_buf_S(v,k,i,iens) = tracers_mult_y(v,k,0 ,i,iens);
          edge_send_buf_N(v,k,i,iens) = tracers_mult_y(v,k,ny,i,iens);
        });
      }

      yakl::fence();
      yakl::timer_start("edge_exchange_mpi");

      MPI_Request sReq[4];
      MPI_Request rReq[4];

      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto dtype = coupler.get_mpi_data_type();
      auto comm = MPI_COMM_WORLD;

      #ifdef MW_GPU_AWARE_MPI
        yakl::fence();

        //Pre-post the receives
        MPI_Irecv( edge_recv_buf_W.data() , edge_recv_buf_W.size() , dtype , neigh(1,0) , 4 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E.data() , edge_recv_buf_E.size() , dtype , neigh(1,2) , 5 , comm , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( edge_recv_buf_S.data() , edge_recv_buf_S.size() , dtype , neigh(0,1) , 6 , comm , &rReq[2] );
          MPI_Irecv( edge_recv_buf_N.data() , edge_recv_buf_N.size() , dtype , neigh(2,1) , 7 , comm , &rReq[3] );
        }

        //Send the data
        MPI_Isend( edge_send_buf_W.data() , edge_send_buf_W.size() , dtype , neigh(1,0) , 5 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_E.data() , edge_send_buf_E.size() , dtype , neigh(1,2) , 4 , comm , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( edge_send_buf_S.data() , edge_send_buf_S.size() , dtype , neigh(0,1) , 7 , comm , &sReq[2] );
          MPI_Isend( edge_send_buf_N.data() , edge_send_buf_N.size() , dtype , neigh(2,1) , 6 , comm , &sReq[3] );
        }

        MPI_Status  sStat[4];
        MPI_Status  rStat[4];

        //Wait for the sends and receives to finish
        if (sim2d) {
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
        } else {
          MPI_Waitall(4, sReq, sStat);
          MPI_Waitall(4, rReq, rStat);
        }
        yakl::timer_stop("edge_exchange_mpi");
      #else
        //Pre-post the receives
        MPI_Irecv( edge_recv_buf_W_host.data() , edge_recv_buf_W_host.size() , dtype , neigh(1,0) , 4 , comm , &rReq[0] );
        MPI_Irecv( edge_recv_buf_E_host.data() , edge_recv_buf_E_host.size() , dtype , neigh(1,2) , 5 , comm , &rReq[1] );
        if (!sim2d) {
          MPI_Irecv( edge_recv_buf_S_host.data() , edge_recv_buf_S_host.size() , dtype , neigh(0,1) , 6 , comm , &rReq[2] );
          MPI_Irecv( edge_recv_buf_N_host.data() , edge_recv_buf_N_host.size() , dtype , neigh(2,1) , 7 , comm , &rReq[3] );
        }

        edge_send_buf_W.deep_copy_to(edge_send_buf_W_host);
        edge_send_buf_E.deep_copy_to(edge_send_buf_E_host);
        if (!sim2d) {
          edge_send_buf_S.deep_copy_to(edge_send_buf_S_host);
          edge_send_buf_N.deep_copy_to(edge_send_buf_N_host);
        }

        yakl::fence();

        //Send the data
        MPI_Isend( edge_send_buf_W_host.data() , edge_send_buf_W_host.size() , dtype , neigh(1,0) , 5 , comm , &sReq[0] );
        MPI_Isend( edge_send_buf_E_host.data() , edge_send_buf_E_host.size() , dtype , neigh(1,2) , 4 , comm , &sReq[1] );
        if (!sim2d) {
          MPI_Isend( edge_send_buf_S_host.data() , edge_send_buf_S_host.size() , dtype , neigh(0,1) , 7 , comm , &sReq[2] );
          MPI_Isend( edge_send_buf_N_host.data() , edge_send_buf_N_host.size() , dtype , neigh(2,1) , 6 , comm , &sReq[3] );
        }

        MPI_Status  sStat[4];
        MPI_Status  rStat[4];

        //Wait for the sends and receives to finish
        if (sim2d) {
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
        } else {
          MPI_Waitall(4, sReq, sStat);
          MPI_Waitall(4, rReq, rStat);
        }
        yakl::timer_stop("edge_exchange_mpi");

        edge_recv_buf_W_host.deep_copy_to(edge_recv_buf_W);
        edge_recv_buf_E_host.deep_copy_to(edge_recv_buf_E);
        if (!sim2d) {
          edge_recv_buf_S_host.deep_copy_to(edge_recv_buf_S);
          edge_recv_buf_N_host.deep_copy_to(edge_recv_buf_N);
        }
      #endif

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,ny,nens) ,
                                        YAKL_LAMBDA (int v, int k, int j, int iens) {
        tracers_mult_x(v,k,j,0 ,iens) = std::min( edge_recv_buf_W(v,k,j,iens) , tracers_mult_x(v,k,j,0 ,iens) );
        tracers_mult_x(v,k,j,nx,iens) = std::min( edge_recv_buf_E(v,k,j,iens) , tracers_mult_x(v,k,j,nx,iens) );
      });

      if (!sim2d) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(npack,nz,nx,nens) ,
                                          YAKL_LAMBDA (int v, int k, int i, int iens) {
          tracers_mult_y(v,k,0 ,i,iens) = std::min( edge_recv_buf_S(v,k,i,iens) , tracers_mult_y(v,k,0 ,i,iens) );
          tracers_mult_y(v,k,ny,i,iens) = std::min( edge_recv_buf_N(v,k,i,iens) , tracers_mult_y(v,k,ny,i,iens) );
        });
      }

      /////////////////////////////////
      // Begin boundary conditions
      /////////////////////////////////
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
      // If boundary conditions are wall, then w-velocity is set to zero, and there is no flux.
      // TODO: Implement something for radiating BC's
    }


    // Creates initial data at a point in space for the rising moist thermal test case
    YAKL_INLINE static void thermal(real x, real y, real z, real xlen, real ylen, real grav, real C0, real gamma,
                                    real cp, real p0, real R_d, real R_v, real &rho, real &u, real &v, real &w,
                                    real &theta, real &rho_v, real &hr, real &ht) {
      hydro_const_theta(z,grav,C0,cp,p0,gamma,R_d,hr,ht);
      real rho_d   = hr;
      u            = 0.;
      v            = 0.;
      w            = 0.;
      real theta_d = ht + sample_ellipse_cosine(2._fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.);
      real p_d     = C0 * pow( rho_d*theta_d , gamma );
      real temp    = p_d / rho_d / R_d;
      real sat_pv  = saturation_vapor_pressure(temp);
      real sat_rv  = sat_pv / R_v / temp;
      rho_v        = sample_ellipse_cosine(0.8_fp  ,  x,y,z  ,  xlen/2,ylen/2,2000.  ,  2000.,2000.,2000.) * sat_rv;
      real p       = rho_d * R_d * temp + rho_v * R_v * temp;
      rho          = rho_d + rho_v;
      theta        = std::pow( p / C0 , 1._fp / gamma ) / rho;
    }


    // Computes a hydrostatic background density and potential temperature using c constant potential temperature
    // backgrounda for a single vertical location
    YAKL_INLINE static void hydro_const_theta( real z, real grav, real C0, real cp, real p0, real gamma, real rd,
                                               real &r, real &t ) {
      const real theta0 = 300.;  //Background potential temperature
      const real exner0 = 1.;    //Surface-level Exner pressure
      t = theta0;                                       //Potential Temperature at z
      real exner = exner0 - grav * z / (cp * theta0);   //Exner pressure at z
      real p = p0 * std::pow(exner,(cp/rd));            //Pressure at z
      real rt = std::pow((p / C0),(1._fp / gamma));     //rho*theta at z
      r = rt / t;                                       //Density at z
    }


    // Samples a 3-D ellipsoid at a point in space
    YAKL_INLINE static real sample_ellipse_cosine(real amp, real x   , real y   , real z   ,
                                                            real x0  , real y0  , real z0  ,
                                                            real xrad, real yrad, real zrad) {
      //Compute distance from bubble center
      real dist = sqrt( ((x-x0)/xrad)*((x-x0)/xrad) +
                        ((y-y0)/yrad)*((y-y0)/yrad) +
                        ((z-z0)/zrad)*((z-z0)/zrad) ) * M_PI / 2.;
      //If the distance from bubble center is less than the radius, create a cos**2 profile
      if (dist <= M_PI / 2.) {
        return amp * std::pow(cos(dist),2._fp);
      } else {
        return 0.;
      }
    }


    YAKL_INLINE static real saturation_vapor_pressure(real temp) {
      real tc = temp - 273.15;
      return 610.94 * std::exp( 17.625*tc / (243.04+tc) );
    }


    // Compute supercell temperature profile at a vertical location
    YAKL_INLINE static real init_supercell_temperature(real z, real z_0, real z_trop, real z_top,
                                                       real T_0, real T_trop, real T_top) {
      if (z <= z_trop) {
        real lapse = - (T_trop - T_0) / (z_trop - z_0);
        return T_0 - lapse * (z - z_0);
      } else {
        real lapse = - (T_top - T_trop) / (z_top - z_trop);
        return T_trop - lapse * (z - z_trop);
      }
    }


    // Compute supercell dry pressure profile at a vertical location
    YAKL_INLINE static real init_supercell_pressure_dry(real z, real z_0, real z_trop, real z_top,
                                                        real T_0, real T_trop, real T_top,
                                                        real p_0, real R_d, real grav) {
      if (z <= z_trop) {
        real lapse = - (T_trop - T_0) / (z_trop - z_0);
        real T = init_supercell_temperature(z, z_0, z_trop, z_top, T_0, T_trop, T_top);
        return p_0 * pow( T / T_0 , grav/(R_d*lapse) );
      } else {
        // Get pressure at the tropopause
        real lapse = - (T_trop - T_0) / (z_trop - z_0);
        real p_trop = p_0 * pow( T_trop / T_0 , grav/(R_d*lapse) );
        // Get pressure at requested height
        lapse = - (T_top - T_trop) / (z_top - z_trop);
        if (lapse != 0) {
          real T = init_supercell_temperature(z, z_0, z_trop, z_top, T_0, T_trop, T_top);
          return p_trop * pow( T / T_trop , grav/(R_d*lapse) );
        } else {
          return p_trop * exp(-grav*(z-z_trop)/(R_d*T_trop));
        }
      }
    }

    
    // Compute supercell relative humidity profile at a vertical location
    YAKL_INLINE static real init_supercell_relhum(real z, real z_0, real z_trop) {
      if (z <= z_trop) {
        return 1._fp - 0.75_fp * pow(z / z_trop , 1.25_fp );
      } else {
        return 0.25_fp;
      }
    }


    // Computes dry saturation mixing ratio
    YAKL_INLINE static real init_supercell_sat_mix_dry( real press , real T ) {
      return 380/(press) * exp( 17.27_fp * (T-273)/(T-36) );
    }


    // Initialize the class data as well as the state and tracers arrays and convert them back into the coupler state
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      // Set class data from # grid points, grid spacing, domain sizes, whether it's 2-D, and physical constants
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto dx          = coupler.get_dx();
      auto dy          = coupler.get_dy();
      auto dz          = coupler.get_dz();
      auto xlen        = coupler.get_xlen();
      auto ylen        = coupler.get_ylen();
      auto zlen        = coupler.get_zlen();
      auto i_beg       = coupler.get_i_beg();
      auto j_beg       = coupler.get_j_beg();
      auto nx_glob     = coupler.get_nx_glob();
      auto ny_glob     = coupler.get_ny_glob();
      auto sim2d       = coupler.is_sim2d();
      auto num_tracers = coupler.get_num_tracers();
      auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);

      if (! coupler.option_exists("R_d"     )) coupler.set_option<real>("R_d"     ,287.       );
      if (! coupler.option_exists("cp_d"    )) coupler.set_option<real>("cp_d"    ,1003.      );
      if (! coupler.option_exists("R_v"     )) coupler.set_option<real>("R_v"     ,461.       );
      if (! coupler.option_exists("cp_v"    )) coupler.set_option<real>("cp_v"    ,1859       );
      if (! coupler.option_exists("p0"      )) coupler.set_option<real>("p0"      ,1.e5       );
      if (! coupler.option_exists("grav"    )) coupler.set_option<real>("grav"    ,9.81       );
      if (! coupler.option_exists("earthrot")) coupler.set_option<real>("earthrot",7.292115e-5);
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
      coupler.set_option<real>("latitude",0);

      auto &dm = coupler.get_data_manager_readwrite();

      dm.register_and_allocate<real>("density_dry","",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("uvel"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("vvel"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("wvel"       ,"",{nz,ny,nx,nens});
      dm.register_and_allocate<real>("temp"       ,"",{nz,ny,nx,nens});

      sim2d = (coupler.get_ny_glob() == 1);

      R_d   = coupler.get_option<real>("R_d"    );
      R_v   = coupler.get_option<real>("R_v"    );
      cp_d  = coupler.get_option<real>("cp_d"   );
      cp_v  = coupler.get_option<real>("cp_v"   );
      p0    = coupler.get_option<real>("p0"     );
      grav  = coupler.get_option<real>("grav"   );
      kappa = coupler.get_option<real>("kappa_d");
      gamma = coupler.get_option<real>("gamma_d");
      C0    = coupler.get_option<real>("C0"     );

      // Create arrays to determine whether we should add mass for a tracer or whether it should remain non-negative
      num_tracers = coupler.get_num_tracers();
      bool1d tracer_adds_mass("tracer_adds_mass",num_tracers);
      bool1d tracer_positive ("tracer_positive" ,num_tracers);
      int    idWV;

      // Must assign on the host to avoid segfaults
      auto tracer_adds_mass_host = tracer_adds_mass.createHostCopy();
      auto tracer_positive_host  = tracer_positive .createHostCopy();

      auto tracer_names = coupler.get_tracer_names();  // Get a list of tracer names
      for (int tr=0; tr < num_tracers; tr++) {
        std::string tracer_desc;
        bool        tracer_found, positive, adds_mass;
        coupler.get_tracer_info( tracer_names[tr] , tracer_desc, tracer_found , positive , adds_mass);
        tracer_positive_host (tr) = positive;
        tracer_adds_mass_host(tr) = adds_mass;
        if (tracer_names[tr] == "water_vapor") idWV = tr;  // Be sure to track which index belongs to water vapor
      }
      tracer_positive_host .deep_copy_to(tracer_positive );
      tracer_adds_mass_host.deep_copy_to(tracer_adds_mass);

      auto init_data = coupler.get_option<std::string>("init_data");
      out_freq       = coupler.get_option<real       >("out_freq" );

      coupler.set_option<int>("idWV",idWV);

      dm.register_and_allocate<bool>("tracer_adds_mass","",{num_tracers});
      auto dm_tracer_adds_mass = dm.get<bool,1>("tracer_adds_mass");
      tracer_adds_mass.deep_copy_to(dm_tracer_adds_mass);

      dm.register_and_allocate<bool>("tracer_positive","",{num_tracers});
      auto dm_tracer_positive = dm.get<bool,1>("tracer_positive");
      tracer_positive.deep_copy_to(dm_tracer_positive);

      // Set an integer version of the input_data so we can test it inside GPU kernels
      int init_data_int;
      if      (init_data == "thermal"  ) { init_data_int = DATA_THERMAL;   }
      else if (init_data == "supercell") { init_data_int = DATA_SUPERCELL; }
      else if (init_data == "city"     ) { init_data_int = DATA_CITY;      }
      else if (init_data == "building" ) { init_data_int = DATA_BUILDING;  }
      else { endrun("ERROR: Invalid init_data in yaml input file"); }

      coupler.set_option<bool>("use_immersed_boundaries",false);
      dm.register_and_allocate<real>("immersed_proportion","",{nz,ny,nx,nens});
      auto immersed_proportion = dm.get<real,4>("immersed_proportion");
      immersed_proportion = 0;

      etime   = 0;
      num_out = 0;

      // Allocate temp arrays to hold state and tracers before we convert it back to the coupler state
      real5d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real5d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);

      dm.register_and_allocate<real>("hy_dens_cells"      ,"",{nz,nens});
      dm.register_and_allocate<real>("hy_dens_theta_cells","",{nz,nens});
      auto hy_dens_cells       = dm.get<real,2>("hy_dens_cells"      );
      auto hy_dens_theta_cells = dm.get<real,2>("hy_dens_theta_cells");

      if (init_data_int == DATA_SUPERCELL) {

        coupler.set_option<bool>("enable_gravity",true);
        coupler.add_option<int>("bc_x",BC_PERIODIC);
        coupler.add_option<int>("bc_y",BC_PERIODIC);
        coupler.add_option<int>("bc_z",BC_WALL);
        coupler.add_option<real>("latitude",0);
        init_supercell( coupler , state , tracers );

      } else if (init_data_int == DATA_THERMAL) {

        coupler.set_option<bool>("enable_gravity",true);
        coupler.add_option<int>("bc_x",BC_PERIODIC);
        coupler.add_option<int>("bc_y",BC_PERIODIC);
        coupler.add_option<int>("bc_z",BC_WALL    );
        coupler.add_option<real>("latitude",0);
        // Define quadrature weights and points for 3-point rules
        const int nqpoints = 9;
        SArray<real,1,nqpoints> qpoints;
        SArray<real,1,nqpoints> qweights;

        qpoints(0) = 0.112701665379258311482073460022;
        qpoints(1) = 0.500000000000000000000000000000;
        qpoints(2) = 0.887298334620741688517926539980;

        qweights(0) = 0.277777777777777777777777777779;
        qweights(1) = 0.444444444444444444444444444444;
        qweights(2) = 0.277777777777777777777777777779;

        size_t i_beg = coupler.get_i_beg();
        size_t j_beg = coupler.get_j_beg();

        // Use quadrature to initialize state and tracer data
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i,iens) = 0.; }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i,iens) = 0.; }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
                real rho, u, v, w, theta, rho_v, hr, ht;

                if (init_data_int == DATA_THERMAL) {
                  thermal(x,y,z,xlen,ylen,grav,C0,gamma,cp_d,p0,R_d,R_v,rho,u,v,w,theta,rho_v,hr,ht);
                }

                if (sim2d) v = 0;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i,iens) += rho       * wt;
                state(idU,hs+k,hs+j,hs+i,iens) += rho*u     * wt;
                state(idV,hs+k,hs+j,hs+i,iens) += rho*v     * wt;
                state(idW,hs+k,hs+j,hs+i,iens) += rho*w     * wt;
                state(idT,hs+k,hs+j,hs+i,iens) += rho*theta * wt;
                for (int tr=0; tr < num_tracers; tr++) {
                  if (tr == idWV) { tracers(tr,hs+k,hs+j,hs+i,iens) += rho_v * wt; }
                  else            { tracers(tr,hs+k,hs+j,hs+i,iens) += 0     * wt; }
                }
                if (i == 0 && ii == 0 && j == 0 && jj == 0) {
                  hy_dens_cells      (k,iens) = hr;
                  hy_dens_theta_cells(k,iens) = hr*ht;
                }
              }
            }
          }
        });

      } else if (init_data_int == DATA_CITY) {

        coupler.add_option<int>("bc_x",BC_PERIODIC);
        coupler.add_option<int>("bc_y",BC_PERIODIC);
        coupler.add_option<int>("bc_z",BC_WALL    );
        coupler.set_option<bool>("use_immersed_boundaries",true);
        immersed_proportion = 0;

        real height_mean = 60;
        real height_std  = 10;

        int building_length = 30;
        int cells_per_building = (int) std::round(building_length / dx);
        int buildings_pad = 20;
        int nblocks_x = (static_cast<int>(xlen)/building_length - 2*buildings_pad)/3;
        int nblocks_y = (static_cast<int>(ylen)/building_length - 2*buildings_pad)/9;
        int nbuildings_x = nblocks_x * 3;
        int nbuildings_y = nblocks_y * 9;

        realHost2d building_heights_host("building_heights",nbuildings_y,nbuildings_x);
        if (coupler.is_mainproc()) {
          std::mt19937 gen{17};
          std::normal_distribution<> d{height_mean, height_std};
          for (int j=0; j < nbuildings_y; j++) {
            for (int i=0; i < nbuildings_x; i++) {
              building_heights_host(j,i) = d(gen);
            }
          }
        }
        auto type = coupler.get_mpi_data_type();
        MPI_Bcast( building_heights_host.data() , building_heights_host.size() , type , 0 , MPI_COMM_WORLD);
        auto building_heights = building_heights_host.createDeviceCopy();

        // Define quadrature weights and points for 3-point rules
        const int nqpoints = 9;
        SArray<real,1,nqpoints> qpoints;
        SArray<real,1,nqpoints> qweights;

        TransformMatrices::get_gll_points (qpoints );
        TransformMatrices::get_gll_weights(qweights);

        // Use quadrature to initialize state and tracer data
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i,iens) = 0.; }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i,iens) = 0.; }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
                real rho, u, v, w, theta, rho_v, hr, ht;

                if (enable_gravity) {
                  hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);
                } else {
                  hr = 1.15;
                  ht = 300;
                }

                rho   = hr;
                u     = 20;
                v     = 0;
                w     = 0;
                theta = ht;
                rho_v = 0;

                if (sim2d) v = 0;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i,iens) += rho       * wt;
                state(idU,hs+k,hs+j,hs+i,iens) += rho*u     * wt;
                state(idV,hs+k,hs+j,hs+i,iens) += rho*v     * wt;
                state(idW,hs+k,hs+j,hs+i,iens) += rho*w     * wt;
                state(idT,hs+k,hs+j,hs+i,iens) += rho*theta * wt;
                for (int tr=0; tr < num_tracers; tr++) {
                  if (tr == idWV) { tracers(tr,hs+k,hs+j,hs+i,iens) += rho_v * wt; }
                  else            { tracers(tr,hs+k,hs+j,hs+i,iens) += 0     * wt; }
                }
                if (i == 0 && ii == 0 && j == 0 && jj == 0) {
                  hy_dens_cells      (k,iens) = hr;
                  hy_dens_theta_cells(k,iens) = hr*ht;
                }
              }
            }
          }
          int inorm = (static_cast<int>(i_beg)+i)/cells_per_building - buildings_pad;
          int jnorm = (static_cast<int>(j_beg)+j)/cells_per_building - buildings_pad;
          if ( ( inorm >= 0 && inorm < nblocks_x*3 && inorm%3 < 2 ) &&
               ( jnorm >= 0 && jnorm < nblocks_y*9 && jnorm%9 < 8 ) ) {
            if ( k <= std::ceil( building_heights(jnorm,inorm) / dz ) ) {
              immersed_proportion(k,j,i,iens) = 1;
              state(idU,hs+k,hs+j,hs+i,iens) = 0;
              state(idV,hs+k,hs+j,hs+i,iens) = 0;
              state(idW,hs+k,hs+j,hs+i,iens) = 0;
            }
          }
        });

      } else if (init_data_int == DATA_BUILDING) {

        coupler.add_option<int>("bc_x",BC_PERIODIC);
        coupler.add_option<int>("bc_y",BC_PERIODIC);
        coupler.add_option<int>("bc_z",BC_WALL    );
        coupler.set_option<bool>("use_immersed_boundaries",true);
        immersed_proportion = 0;

        // Define quadrature weights and points for 3-point rules
        const int nqpoints = 9;
        SArray<real,1,nqpoints> qpoints;
        SArray<real,1,nqpoints> qweights;

        TransformMatrices::get_gll_points (qpoints );
        TransformMatrices::get_gll_weights(qweights);

        // Use quadrature to initialize state and tracer data
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          for (int l=0; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i,iens) = 0.; }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i,iens) = 0.; }
          //Use Gauss-Legendre quadrature
          for (int kk=0; kk<nqpoints; kk++) {
            for (int jj=0; jj<nqpoints; jj++) {
              for (int ii=0; ii<nqpoints; ii++) {
                real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
                real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
                real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
                real rho, u, v, w, theta, rho_v, hr, ht;

                if (enable_gravity) {
                  hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht);
                } else {
                  hr = 1.15;
                  ht = 300;
                }

                rho   = hr;
                u     = 20;
                v     = 0;
                w     = 0;
                theta = ht;
                rho_v = 0;

                if (sim2d) v = 0;

                real wt = qweights(ii)*qweights(jj)*qweights(kk);
                state(idR,hs+k,hs+j,hs+i,iens) += rho       * wt;
                state(idU,hs+k,hs+j,hs+i,iens) += rho*u     * wt;
                state(idV,hs+k,hs+j,hs+i,iens) += rho*v     * wt;
                state(idW,hs+k,hs+j,hs+i,iens) += rho*w     * wt;
                state(idT,hs+k,hs+j,hs+i,iens) += rho*theta * wt;
                for (int tr=0; tr < num_tracers; tr++) {
                  if (tr == idWV) { tracers(tr,hs+k,hs+j,hs+i,iens) += rho_v * wt; }
                  else            { tracers(tr,hs+k,hs+j,hs+i,iens) += 0     * wt; }
                }
                if (i == 0 && ii == 0 && j == 0 && jj == 0) {
                  hy_dens_cells      (k,iens) = hr;
                  hy_dens_theta_cells(k,iens) = hr*ht;
                }
              }
            }
          }
          real x0 = 0.3*nx_glob;
          real y0 = 0.5*ny_glob;
          real xr = 0.05*ny_glob;
          real yr = 0.05*ny_glob;
          if ( std::abs(i_beg+i-x0) <= xr && std::abs(j_beg+j-y0) <= yr && k <= 0.2*nz ) {
            immersed_proportion(k,j,i,iens) = 1;
            state(idU,hs+k,hs+j,hs+i,iens) = 0;
            state(idV,hs+k,hs+j,hs+i,iens) = 0;
            state(idW,hs+k,hs+j,hs+i,iens) = 0;
          }
        });

      }

      if (enable_gravity) {
        // Compute forcing due to pressure gradient only in the vertical direction
        coupler.set_option<bool>("save_pressure_z",true);
        dm.register_and_allocate<real>("pressure_z","",{nz+1,ny,nx,nens});
        dm.register_and_allocate<real>("pressure_mult","",{nz+1,nens});
        auto pressure_z = dm.get<real,4>("pressure_z");
        auto pressure_mult = dm.get<real,2>("pressure_mult");
        pressure_mult = 1;
        real dt_dummy = 1.;
        real5d state_tend  ("state_tend"  ,num_state  ,nz,ny,nx,nens);
        real5d tracers_tend("tracers_tend",num_tracers,nz,ny,nx,nens);
        compute_tendencies( coupler , state , state_tend , tracers , tracers_tend , dt_dummy , DIR_Z );
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
        using yakl::componentwise::operator-;
        using yakl::componentwise::operator/;
        std::cout << std::scientific << (pressure_mult_host-1);
      } else {
        dm.register_and_allocate<real>("pressure_mult","",{nz+1,nens});
        auto pressure_mult = dm.get<real,2>("pressure_mult");
        pressure_mult = 1;
      }

      // Convert the initialized state and tracers arrays back to the coupler state
      convert_dynamics_to_coupler( coupler , state , tracers );

      // Output the initial state
      if (out_freq >= 0. ) output( coupler , etime );
    }


    // Initialize the supercell test case
    void init_supercell( core::Coupler &coupler , real5d &state , real5d &tracers ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      real constexpr z_0    = 0;
      real constexpr z_trop = 12000;
      real constexpr T_0    = 300;
      real constexpr T_trop = 213;
      real constexpr T_top  = 213;
      real constexpr p_0    = 100000;

      int constexpr nqpoints = 9;
      SArray<real,1,nqpoints> gll_pts, gll_wts;
      TransformMatrices::get_gll_points (gll_pts);
      TransformMatrices::get_gll_weights(gll_wts);

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto dx          = coupler.get_dx();
      auto dy          = coupler.get_dy();
      auto dz          = coupler.get_dz();
      auto xlen        = coupler.get_xlen();
      auto ylen        = coupler.get_ylen();
      auto sim2d       = coupler.is_sim2d();
      auto R_d         = coupler.get_option<real>("R_d"    );
      auto R_v         = coupler.get_option<real>("R_v"    );
      auto grav        = coupler.get_option<real>("grav"   );
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0"     );
      auto idWV        = coupler.get_option<int >("idWV");
      auto num_tracers = coupler.get_num_tracers();
      auto i_beg       = coupler.get_i_beg();
      auto j_beg       = coupler.get_j_beg();

      auto hy_dens_cells       = coupler.get_data_manager_readwrite().get<real,2>("hy_dens_cells"      );
      auto hy_dens_theta_cells = coupler.get_data_manager_readwrite().get<real,2>("hy_dens_theta_cells");
      real2d hy_dens_edges      ("hy_dens_edges"      ,nz+1,nens);
      real2d hy_dens_theta_edges("hy_dens_theta_edges",nz+1,nens);

      // Temporary arrays used to compute the initial state for high-CAPE supercell conditions
      real3d quad_temp       ("quad_temp"       ,nz,nqpoints-1,nqpoints);
      real2d hyDensGLL       ("hyDensGLL"       ,nz,nqpoints);
      real2d hyDensThetaGLL  ("hyDensThetaGLL"  ,nz,nqpoints);
      real2d hyDensVapGLL    ("hyDensVapGLL"    ,nz,nqpoints);
      real2d hyPressureGLL   ("hyPressureGLL"   ,nz,nqpoints);
      real1d hyDensCells     ("hyDensCells"     ,nz);
      real1d hyDensThetaCells("hyDensThetaCells",nz);

      real ztop = coupler.get_zlen();

      // Compute quadrature term to integrate to get pressure at GLL points
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,nqpoints-1,nqpoints) ,
                    YAKL_LAMBDA (int k, int kk, int kkk) {
        // Middle of this cell
        real cellmid   = (k+0.5_fp) * dz;
        // Bottom, top, and middle of the space between these two nqpoints GLL points
        real nqpoints_b    = cellmid + gll_pts(kk  )*dz;
        real nqpoints_t    = cellmid + gll_pts(kk+1)*dz;
        real nqpoints_m    = 0.5_fp * (nqpoints_b + nqpoints_t);
        // Compute grid spacing between these nqpoints GLL points
        real nqpoints_dz   = dz * ( gll_pts(kk+1) - gll_pts(kk) );
        // Compute the locate of this GLL point within the nqpoints GLL points
        real zloc      = nqpoints_m + nqpoints_dz * gll_pts(kkk);
        // Compute full density at this location
        real temp      = init_supercell_temperature (zloc, z_0, z_trop, ztop, T_0, T_trop, T_top);
        real press_dry = init_supercell_pressure_dry(zloc, z_0, z_trop, ztop, T_0, T_trop, T_top, p_0, R_d, grav);
        real qvs       = init_supercell_sat_mix_dry(press_dry, temp);
        real relhum    = init_supercell_relhum(zloc, z_0, z_trop);
        if (relhum * qvs > 0.014_fp) relhum = 0.014_fp / qvs;
        real qv        = std::min( 0.014_fp , qvs*relhum );
        quad_temp(k,kk,kkk) = -(1+qv)*grav/(R_d+qv*R_v)/temp;
      });

      // Compute pressure at GLL points
      parallel_for( YAKL_AUTO_LABEL() , 1 , YAKL_LAMBDA (int dummy) {
        hyPressureGLL(0,0) = p_0;
        for (int k=0; k < nz; k++) {
          for (int kk=0; kk < nqpoints-1; kk++) {
            real tot = 0;
            for (int kkk=0; kkk < nqpoints; kkk++) {
              tot += quad_temp(k,kk,kkk) * gll_wts(kkk);
            }
            tot *= dz * ( gll_pts(kk+1) - gll_pts(kk) );
            hyPressureGLL(k,kk+1) = hyPressureGLL(k,kk) * exp( tot );
            if (kk == nqpoints-2 && k < nz-1) {
              hyPressureGLL(k+1,0) = hyPressureGLL(k,nqpoints-1);
            }
          }
        }
      });

      // Compute hydrostatic background state at GLL points
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,nqpoints) , YAKL_LAMBDA (int k, int kk) {
        real zloc = (k+0.5_fp)*dz + gll_pts(kk)*dz;
        real temp       = init_supercell_temperature (zloc, z_0, z_trop, ztop, T_0, T_trop, T_top);
        real press_tmp  = init_supercell_pressure_dry(zloc, z_0, z_trop, ztop, T_0, T_trop, T_top, p_0, R_d, grav);
        real qvs        = init_supercell_sat_mix_dry(press_tmp, temp);
        real relhum     = init_supercell_relhum(zloc, z_0, z_trop);
        if (relhum * qvs > 0.014_fp) relhum = 0.014_fp / qvs;
        real qv         = std::min( 0.014_fp , qvs*relhum );
        real press      = hyPressureGLL(k,kk);
        real dens_dry   = press / (R_d+qv*R_v) / temp;
        real dens_vap   = qv * dens_dry;
        real dens       = dens_dry + dens_vap;
        real dens_theta = pow( press / C0 , 1._fp / gamma );
        hyDensGLL     (k,kk) = dens;
        hyDensThetaGLL(k,kk) = dens_theta;
        hyDensVapGLL  (k,kk) = dens_vap;
        if (kk == 0) {
          for (int iens=0; iens < nens; iens++) {
            hy_dens_edges      (k,iens) = dens;
            hy_dens_theta_edges(k,iens) = dens_theta;
          }
        }
        if (k == nz-1 && kk == nqpoints-1) {
          for (int iens=0; iens < nens; iens++) {
            hy_dens_edges      (k+1,iens) = dens;
            hy_dens_theta_edges(k+1,iens) = dens_theta;
          }
        }
      });

      // Compute hydrostatic background state over cells
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<1>(nz) , YAKL_LAMBDA (int k) {
        real press_tot      = 0;
        real dens_tot       = 0;
        real dens_vap_tot   = 0;
        real dens_theta_tot = 0;
        for (int kk=0; kk < nqpoints; kk++) {
          press_tot      += hyPressureGLL (k,kk) * gll_wts(kk);
          dens_tot       += hyDensGLL     (k,kk) * gll_wts(kk);
          dens_vap_tot   += hyDensVapGLL  (k,kk) * gll_wts(kk);
          dens_theta_tot += hyDensThetaGLL(k,kk) * gll_wts(kk);
        }
        real press      = press_tot;
        real dens       = dens_tot;
        real dens_vap   = dens_vap_tot;
        real dens_theta = dens_theta_tot;
        real dens_dry   = dens - dens_vap;
        real R          = dens_dry / dens * R_d + dens_vap / dens * R_v;
        real temp       = press / (dens * R);
        real qv         = dens_vap / dens_dry;
        real zloc       = (k+0.5_fp)*dz;
        real press_tmp  = init_supercell_pressure_dry(zloc, z_0, z_trop, ztop, T_0, T_trop, T_top, p_0, R_d, grav);
        real qvs        = init_supercell_sat_mix_dry(press_tmp, temp);
        real relhum     = qv / qvs;
        real T          = temp - 273;
        real a          = 17.27;
        real b          = 237.7;
        real tdew       = b * ( a*T / (b + T) + log(relhum) ) / ( a - ( a*T / (b+T) + log(relhum) ) );
        // These are used in the rest of the model
        for (int iens=0; iens < nens; iens++) {
          hy_dens_cells      (k,iens) = dens;
          hy_dens_theta_cells(k,iens) = dens_theta;
        }
      });

      // Initialize the state
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        state(idR,hs+k,hs+j,hs+i,iens) = 0;
        state(idU,hs+k,hs+j,hs+i,iens) = 0;
        state(idV,hs+k,hs+j,hs+i,iens) = 0;
        state(idW,hs+k,hs+j,hs+i,iens) = 0;
        state(idT,hs+k,hs+j,hs+i,iens) = 0;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) = 0; }
        for (int kk=0; kk < nqpoints; kk++) {
          for (int jj=0; jj < nqpoints; jj++) {
            for (int ii=0; ii < nqpoints; ii++) {
              real xloc = (i+i_beg+0.5_fp)*dx + gll_pts(ii)*dx;
              real yloc = (j+j_beg+0.5_fp)*dy + gll_pts(jj)*dy;
              real zloc = (k      +0.5_fp)*dz + gll_pts(kk)*dz;

              if (sim2d) yloc = ylen/2;

              real dens = hyDensGLL(k,kk);

              real uvel;
              real constexpr zs = 5000;
              real constexpr us = 30;
              real constexpr uc = 15;
              if (zloc < zs) {
                uvel = us * (zloc / zs) - uc;
              } else {
                uvel = us - uc;
              }

              real vvel       = 0;
              real wvel       = 0;
              real dens_vap   = hyDensVapGLL  (k,kk);
              real dens_theta = hyDensThetaGLL(k,kk);

              real factor = gll_wts(ii) * gll_wts(jj) * gll_wts(kk);
              state  (idR ,hs+k,hs+j,hs+i,iens) += dens        * factor;
              state  (idU ,hs+k,hs+j,hs+i,iens) += dens * uvel * factor;
              state  (idV ,hs+k,hs+j,hs+i,iens) += dens * vvel * factor;
              state  (idW ,hs+k,hs+j,hs+i,iens) += dens * wvel * factor;
              state  (idT ,hs+k,hs+j,hs+i,iens) += dens_theta  * factor;
              tracers(idWV,hs+k,hs+j,hs+i,iens) += dens_vap    * factor;
            }
          }
        }
      });
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
      auto idWV        = coupler.get_option<int >("idWV");
      auto num_tracers = coupler.get_num_tracers();

      auto &dm = coupler.get_data_manager_readwrite();

      // Get state from the coupler
      auto dm_rho_d = dm.get<real,4>("density_dry");
      auto dm_uvel  = dm.get<real,4>("uvel"       );
      auto dm_vvel  = dm.get<real,4>("vvel"       );
      auto dm_wvel  = dm.get<real,4>("wvel"       );
      auto dm_temp  = dm.get<real,4>("temp"       );
      auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");

      // Get tracers from the coupler
      core::MultiField<real,4> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) {
        dm_tracers.add_field( dm.get<real,4>(tracer_names[tr]) );
      }

      // Convert from state and tracers arrays to the coupler's data
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho   = state(idR,hs+k,hs+j,hs+i,iens);
        real u     = state(idU,hs+k,hs+j,hs+i,iens) / rho;
        real v     = state(idV,hs+k,hs+j,hs+i,iens) / rho;
        real w     = state(idW,hs+k,hs+j,hs+i,iens) / rho;
        real theta = state(idT,hs+k,hs+j,hs+i,iens) / rho;
        real press = C0 * pow( rho*theta , gamma );

        real rho_v = tracers(idWV,hs+k,hs+j,hs+i,iens);
        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,hs+j,hs+i,iens);
        }
        real temp = press / ( rho_d * R_d + rho_v * R_v );

        dm_rho_d(k,j,i,iens) = rho_d;
        dm_uvel (k,j,i,iens) = u;
        dm_vvel (k,j,i,iens) = v;
        dm_wvel (k,j,i,iens) = w;
        dm_temp (k,j,i,iens) = temp;
        for (int tr=0; tr < num_tracers; tr++) {
          dm_tracers(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens);
        }
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
      auto idWV        = coupler.get_option<int >("idWV");
      auto num_tracers = coupler.get_num_tracers();

      auto &dm = coupler.get_data_manager_readonly();

      // Get the coupler's state (as const because it's read-only)
      auto dm_rho_d = dm.get<real const,4>("density_dry");
      auto dm_uvel  = dm.get<real const,4>("uvel"       );
      auto dm_vvel  = dm.get<real const,4>("vvel"       );
      auto dm_wvel  = dm.get<real const,4>("wvel"       );
      auto dm_temp  = dm.get<real const,4>("temp"       );
      auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");

      // Get the coupler's tracers (as const because it's read-only)
      core::MultiField<real const,4> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) {
        dm_tracers.add_field( dm.get<real const,4>(tracer_names[tr]) );
      }

      // Convert from the coupler's state to the dycore's state and tracers arrays
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho_d = dm_rho_d(k,j,i,iens);
        real u     = dm_uvel (k,j,i,iens);
        real v     = dm_vvel (k,j,i,iens);
        real w     = dm_wvel (k,j,i,iens);
        real temp  = dm_temp (k,j,i,iens);
        real rho_v = dm_tracers(idWV,k,j,i,iens);
        real press = rho_d * R_d * temp + rho_v * R_v * temp;

        real rho = rho_d;
        for (int tr=0; tr < num_tracers; tr++) {
          if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i,iens);
        }
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;

        state(idR,hs+k,hs+j,hs+i,iens) = rho;
        state(idU,hs+k,hs+j,hs+i,iens) = rho * u;
        state(idV,hs+k,hs+j,hs+i,iens) = rho * v;
        state(idW,hs+k,hs+j,hs+i,iens) = rho * w;
        state(idT,hs+k,hs+j,hs+i,iens) = rho * theta;
        for (int tr=0; tr < num_tracers; tr++) {
          tracers(tr,hs+k,hs+j,hs+i,iens) = dm_tracers(tr,k,j,i,iens);
        }
      });
    }


    // Perform file output
    void output( core::Coupler const &coupler , real etime ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      yakl::timer_start("output");

      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto dx          = coupler.get_dx();
      auto dy          = coupler.get_dy();
      auto dz          = coupler.get_dz();
      auto num_tracers = coupler.get_num_tracers();
      int i_beg = coupler.get_i_beg();
      int j_beg = coupler.get_j_beg();
      int iens = 0;

      if (coupler.get_option<bool>("file_per_process",false)) {

        yakl::SimpleNetCDF nc;
        MPI_Offset ulIndex = 0; // Unlimited dimension index to place this data at

        
        std::stringstream fname;
        fname << coupler.get_option<std::string>("out_prefix") << "_" << std::setw(8) << std::setfill('0')
              << coupler.get_myrank() << ".nc";

        if (etime == 0) {
          nc.create( fname.str() , yakl::NETCDF_MODE_REPLACE );

          nc.createDim( "x" , coupler.get_nx() );
          nc.createDim( "y" , coupler.get_ny() );
          nc.createDim( "z" , nz );
          nc.createDim( "t" );

          // x-coordinate
          real1d xloc("xloc",nx);
          parallel_for( YAKL_AUTO_LABEL() , nx , YAKL_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
          nc.write( xloc , "x" , {"x"} );

          // y-coordinate
          real1d yloc("yloc",ny);
          parallel_for( YAKL_AUTO_LABEL() , ny , YAKL_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
          nc.write( yloc , "y" , {"y"} );

          // z-coordinate
          real1d zloc("zloc",nz);
          parallel_for( YAKL_AUTO_LABEL() , nz , YAKL_LAMBDA (int k) { zloc(k) = (k      +0.5)*dz; });
          nc.write( zloc , "z" , {"z"} );
          nc.write1( 0._fp , "t" , 0 , "t" );

        } else {

          nc.open( fname.str() , yakl::NETCDF_MODE_WRITE );
          ulIndex = nc.getDimSize("t");

          // Write the elapsed time
          nc.write1(etime,"t",ulIndex,"t");

        }

        std::vector<std::string> varnames(num_state+num_tracers);
        varnames[0] = "density_dry";
        varnames[1] = "uvel";
        varnames[2] = "vvel";
        varnames[3] = "wvel";
        varnames[4] = "temp";
        auto tracer_names = coupler.get_tracer_names();
        for (int tr=0; tr < num_tracers; tr++) { varnames[num_state+tr] = tracer_names[tr]; }

        auto &dm = coupler.get_data_manager_readonly();
        real3d data("data",nz,ny,nx);
        for (int i=0; i < varnames.size(); i++) {
          auto var = dm.get<real const,4>(varnames[i]);
          parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) { data(k,j,i) = var(k,j,i,iens); });
          nc.write1(data,varnames[i],{"z","y","x"},ulIndex,"t");
        }

        nc.close();

      } else { // if file_per_process

        yakl::SimplePNetCDF nc;
        MPI_Offset ulIndex = 0; // Unlimited dimension index to place this data at

        std::stringstream fname;
        fname << coupler.get_option<std::string>("out_prefix") << ".nc";
        if (etime == 0) {
          nc.create(fname.str() , NC_CLOBBER | NC_64BIT_DATA);

          nc.create_dim( "x" , coupler.get_nx_glob() );
          nc.create_dim( "y" , coupler.get_ny_glob() );
          nc.create_dim( "z" , nz );
          nc.create_unlim_dim( "t" );

          nc.create_var<real>( "x" , {"x"} );
          nc.create_var<real>( "y" , {"y"} );
          nc.create_var<real>( "z" , {"z"} );
          nc.create_var<real>( "t" , {"t"} );
          nc.create_var<real>( "density_dry" , {"t","z","y","x"} );
          nc.create_var<real>( "uvel"        , {"t","z","y","x"} );
          nc.create_var<real>( "vvel"        , {"t","z","y","x"} );
          nc.create_var<real>( "wvel"        , {"t","z","y","x"} );
          nc.create_var<real>( "temp"        , {"t","z","y","x"} );
          auto tracer_names = coupler.get_tracer_names();
          for (int tr = 0; tr < num_tracers; tr++) {
            nc.create_var<real>( tracer_names[tr] , {"t","z","y","x"} );
          }

          nc.enddef();

          // x-coordinate
          real1d xloc("xloc",nx);
          parallel_for( YAKL_AUTO_LABEL() , nx , YAKL_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
          nc.write_all( xloc.createHostCopy() , "x" , {i_beg} );

          // y-coordinate
          real1d yloc("yloc",ny);
          parallel_for( YAKL_AUTO_LABEL() , ny , YAKL_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
          nc.write_all( yloc.createHostCopy() , "y" , {j_beg} );

          // z-coordinate
          real1d zloc("zloc",nz);
          parallel_for( YAKL_AUTO_LABEL() , nz , YAKL_LAMBDA (int k) { zloc(k) = (k      +0.5)*dz; });
          nc.begin_indep_data();
          if (coupler.is_mainproc()) {
            nc.write( zloc.createHostCopy() , "z" );
            nc.write1( 0._fp , "t" , 0 , "t" );
          }
          nc.end_indep_data();

        } else {

          nc.open(fname.str());
          ulIndex = nc.get_dim_size("t");

          // Write the elapsed time
          nc.begin_indep_data();
          if (coupler.is_mainproc()) {
            nc.write1(etime,"t",ulIndex,"t");
          }
          nc.end_indep_data();

        }

        std::vector<std::string> varnames(num_state+num_tracers);
        varnames[0] = "density_dry";
        varnames[1] = "uvel";
        varnames[2] = "vvel";
        varnames[3] = "wvel";
        varnames[4] = "temp";
        auto tracer_names = coupler.get_tracer_names();
        for (int tr=0; tr < num_tracers; tr++) { varnames[num_state+tr] = tracer_names[tr]; }

        auto &dm = coupler.get_data_manager_readonly();
        real3d data("data",nz,ny,nx);
        for (int i=0; i < varnames.size(); i++) {
          auto var = dm.get<real const,4>(varnames[i]);
          parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) { data(k,j,i) = var(k,j,i,iens); });
          nc.write1_all(data.createHostCopy(),varnames[i],ulIndex,{0,j_beg,i_beg},"t");
        }

        nc.close();

      } // if file_per_process

      yakl::timer_stop("output");
    }


  };

}


