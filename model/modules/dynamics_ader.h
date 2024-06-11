
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
      yakl::index_t static constexpr ord  = 9;
      yakl::index_t static constexpr ngll = 4;
      yakl::index_t static constexpr tord = 3;
    #else
      yakl::index_t static constexpr ord = PORTURB_ORD;
    #endif
    int static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")
    int static constexpr num_state = 5;   // Number of state variables
    // IDs for the variables in the state vector
    int static constexpr idR = 0;  // Density
    int static constexpr idU = 1;  // u-momentum
    int static constexpr idV = 2;  // v-momentum
    int static constexpr idW = 3;  // w-momentum
    int static constexpr idT = 4;  // Density * potential temperature

    int static constexpr DIR_X = 0;
    int static constexpr DIR_Y = 1;
    int static constexpr DIR_Z = 2;

    real static constexpr cs   = 350.;
    real static constexpr cs2  = cs*cs;
    real static constexpr r_cs = 1./cs;


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
    // real compute_time_step( core::Coupler const &coupler ) const {
    //   auto dx = coupler.get_dx();
    //   auto dy = coupler.get_dy();
    //   auto dz = coupler.get_dz();
    //   real constexpr maxwave = 350 + 100;
    //   real cfl = 0.40;
    //   return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
    // }
    real compute_time_step( core::Coupler const &coupler ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx = coupler.get_nx();
      auto ny = coupler.get_ny();
      auto nz = coupler.get_nz();
      auto dx = coupler.get_dx();
      auto dy = coupler.get_dy();
      auto dz = coupler.get_dz();
      auto R_d = coupler.get_option<real>("R_d");
      auto gamma = coupler.get_option<real>("gamma_d");
      auto &dm = coupler.get_data_manager_readonly();
      auto rho_d = dm.get<real const,3>("density_dry");
      auto uvel  = dm.get<real const,3>("uvel"       );
      auto vvel  = dm.get<real const,3>("vvel"       );
      auto wvel  = dm.get<real const,3>("wvel"       );
      auto temp  = dm.get<real const,3>("temp"       );
      real3d dt3d("dt3d",nz,ny,nx);
      real cfl = 0.80;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real r = rho_d(k,j,i);
        real u = uvel (k,j,i);
        real v = vvel (k,j,i);
        real w = wvel (k,j,i);
        real T = temp (k,j,i);
        real p = r*R_d*T;
        real cs = std::sqrt(gamma*p/r);
        real dtx = cfl*dx/(std::abs(u)+cs);
        real dty = cfl*dy/(std::abs(v)+cs);
        real dtz = cfl*dz/(std::abs(w)+cs);
        dt3d(k,j,i) = std::min(std::min(dtx,dty),dtz);
      });
      real maxwave = yakl::intrinsics::minval(dt3d);
      return coupler.get_parallel_comm().all_reduce( maxwave , MPI_MIN );
    }


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
      ncycles = 1;
      dt_dyn = dt_phys / ncycles;
      auto dim_switch = coupler.get_option<bool>("dycore_dim_switch",true);
      for (int icycle = 0; icycle < ncycles; icycle++) {
        if (dim_switch) {
          time_step_ader(coupler,state,tracers,dt_dyn,DIR_X);
          time_step_ader(coupler,state,tracers,dt_dyn,DIR_Y);
          time_step_ader(coupler,state,tracers,dt_dyn,DIR_Z);
        } else {
          time_step_ader(coupler,state,tracers,dt_dyn,DIR_Z);
          time_step_ader(coupler,state,tracers,dt_dyn,DIR_Y);
          time_step_ader(coupler,state,tracers,dt_dyn,DIR_X);
        }
        dim_switch = ! dim_switch;
      }
      coupler.set_option<bool>("dycore_dim_switch",dim_switch);
      convert_dynamics_to_coupler( coupler , state , tracers );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step");
      #endif
    }


    // CFL 0.45 (Differs from paper, but this is the true value for this high-order FV scheme)
    // Third-order, three-stage SSPRK method
    // https://link.springer.com/content/pdf/10.1007/s10915-008-9239-z.pdf
    void time_step_ader( core::Coupler & coupler ,
                         real4d const  & state   ,
                         real4d const  & tracers ,
                         real            dt_dyn  ,
                         int             dir     ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("time_step_ader");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto num_tracers = coupler.get_num_tracers();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto &dm         = coupler.get_data_manager_readonly();
      auto tracer_positive = dm.get<bool const,1>("tracer_positive");
      real4d state_tend  ("state_tend"  ,num_state  ,nz     ,ny     ,nx     );
      real4d tracers_tend("tracers_tend",num_tracers,nz     ,ny     ,nx     );
      enforce_immersed_boundaries( coupler , state , tracers , dt_dyn/2 );
      if (dir == DIR_X) compute_tendencies_x(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
      if (dir == DIR_Y) compute_tendencies_y(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
      if (dir == DIR_Z) compute_tendencies_z(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(num_state+num_tracers,nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state  (l,hs+k,hs+j,hs+i) = state  (l,hs+k,hs+j,hs+i) + dt_dyn * state_tend  (l,k,j,i);
        } else {
          l -= num_state;
          tracers(l,hs+k,hs+j,hs+i) = tracers(l,hs+k,hs+j,hs+i) + dt_dyn * tracers_tend(l,k,j,i);
          if (tracer_positive(l)) tracers(l,hs+k,hs+j,hs+i) = std::max( 0._fp , tracers(l,hs+k,hs+j,hs+i) );
        }
      });
      enforce_immersed_boundaries( coupler , state , tracers , dt_dyn/2 );
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("time_step_ader");
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
      auto hy_dens_cells  = dm.get<real const,1>("hy_dens_cells" ); // Hydrostatic density
      auto hy_theta_cells = dm.get<real const,1>("hy_theta_cells"); // Hydrostatic potential temperature
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



    void compute_tendencies_x( core::Coupler       & coupler      ,
                               real4d        const & state        ,
                               real4d        const & state_tend   ,
                               real4d        const & tracers      ,
                               real4d        const & tracers_tend ,
                               real                  dt           ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("compute_tendencies");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      using yakl::intrinsics::matmul_cr;
      using yakl::COLON;
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
      auto  hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"            ); // Hydrostatic density
      auto  hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"           ); // Hydrostatic potential temperature
      auto  hy_dens_edges     = dm.get<real const,1>("hy_dens_edges"            ); // Hydrostatic density
      auto  hy_theta_edges    = dm.get<real const,1>("hy_theta_edges"           ); // Hydrostatic potential temperature
      auto  hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells"        ); // Hydrostatic pressure
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real r_dz = 1./dz; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

      SArray<real,2,ngll,ngll> g2d2g;
      {
        SArray<real,2,ngll,ngll> g2c;
        SArray<real,2,ngll,ngll> c2g;
        SArray<real,2,ngll,ngll> c2d;
        TransformMatrices::gll_to_coefs  (g2c);
        TransformMatrices::coefs_to_gll  (c2g);
        TransformMatrices::coefs_to_deriv(c2d);
        g2d2g = matmul_cr( c2g , matmul_cr( c2d , g2c ) );
      }
      SArray<real,2,ord,ngll> s2g;
      {
        SArray<real,2,ord,ord > s2c;
        SArray<real,2,ord,ngll> c2g;
        TransformMatrices::sten_to_coefs     (s2c);
        TransformMatrices::coefs_to_gll_lower(c2g);
        s2g = matmul_cr( c2g , s2c );
      }

      real3d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs); // Holds pressure perturbation

      // Compute pressure
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        pressure(hs+k,hs+j,hs+i) = C0*std::pow(state(idT,hs+k,hs+j,hs+i),gamma) - hy_pressure_cells(hs+k);
        for (int l=1; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) /= state(idR,hs+k,hs+j,hs+i); }
        for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) /= state(idR,hs+k,hs+j,hs+i); }
      });

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      {
        core::MultiField<real,3> fields;
        for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<3>(l,0,0,0) ); }
        for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<3>(l,0,0,0) ); }
        fields.add_field( pressure );
        if (ord > 1) coupler.halo_exchange_x( fields , hs );
      }

      // Create arrays to hold cell gll interpolations
      real5d state_gll   ("state_gll"   ,num_state  ,ngll,nz,ny,nx);
      real4d pressure_gll("pressure_gll"            ,ngll,nz,ny,nx);
      real5d tracers_gll ("tracers_gll" ,num_tracers,ngll,nz,ny,nx);
      
      // Aggregate fields for interpolation
      core::MultiField<real,3> fields;
      core::MultiField<real,4> gll_fields;
      int idP = 5;
      for (int l=0; l < num_state; l++) {
        fields    .add_field(state    .slice<3>(l,      COLON,COLON,COLON));
        gll_fields.add_field(state_gll.slice<4>(l,COLON,COLON,COLON,COLON));
      }
      fields    .add_field(pressure    .slice<3>(      COLON,COLON,COLON));
      gll_fields.add_field(pressure_gll.slice<4>(COLON,COLON,COLON,COLON));
      for (int l=0; l < num_tracers; l++) {
        fields    .add_field(tracers    .slice<3>(l      ,COLON,COLON,COLON));
        gll_fields.add_field(tracers_gll.slice<4>(l,COLON,COLON,COLON,COLON));
      }

      typedef limiter::WenoLimiter<real,ord> Limiter;

      // Interpolation to GLL points
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(fields.size(),nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        for (int ii=0; ii<ord; ii++) {
          immersed(ii) = immersed_prop(hs+k,hs+j,i+ii) > 0;
          stencil (ii) = fields     (l,hs+k,hs+j,i+ii);
        }
        if (l == idV || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        SArray<real,1,ngll> gll;
        Limiter::compute_limited_gll( stencil , gll , { ! any_immersed(k,j,i) , immersed(hs-1) , immersed(hs+1) } );
        for (int ii=0; ii < ngll; ii++) { gll_fields(l,ii,k,j,i) = gll(ii); }
      });

      // Create arrays to hold cell interface interpolations
      real5d state_limits   ("state_limits"   ,num_state  ,2,nz,ny,nx+1);
      real4d pressure_limits("pressure_limits"            ,2,nz,ny,nx+1);
      real5d tracers_limits ("tracers_limits" ,num_tracers,2,nz,ny,nx+1);

      // ADER Time averaging
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        SArray<real,2,tord,ngll> r_DTs, ru_DTs, rv_DTs, rw_DTs, rt_DTs, ruu_DTs, ruv_DTs, ruw_DTs, rut_DTs, rt_gamma_DTs;
        for (int ii=0; ii < ngll; ii++) {
          real r = state_gll(idR,ii,k,j,i);
          real u = state_gll(idU,ii,k,j,i);
          real v = state_gll(idV,ii,k,j,i);
          real w = state_gll(idW,ii,k,j,i);
          real t = state_gll(idT,ii,k,j,i);
          r_DTs       (0,ii) = r;
          ru_DTs      (0,ii) = r*u;
          rv_DTs      (0,ii) = r*v;
          rw_DTs      (0,ii) = r*w;
          rt_DTs      (0,ii) = r*t;
          ruu_DTs     (0,ii) = r*u*u;
          ruv_DTs     (0,ii) = r*u*v;
          ruw_DTs     (0,ii) = r*u*w;
          rut_DTs     (0,ii) = r*u*t;
          rt_gamma_DTs(0,ii) = std::pow(r*t,gamma);
        }
        diffTransformEulerConsX( r_DTs, ru_DTs, rv_DTs, rw_DTs, rt_DTs, ruu_DTs, ruv_DTs, ruw_DTs, rut_DTs,
                                 rt_gamma_DTs, g2d2g , C0 , gamma , dx );
        SArray<real,1,ngll> r_tavg, ru_tavg;
        compute_timeAvg( r_DTs  , r_tavg  , dt );
        compute_timeAvg( ru_DTs , ru_tavg , dt );
        compute_timeAvg( rv_DTs           , dt );
        compute_timeAvg( rw_DTs           , dt );
        compute_timeAvg( rt_DTs           , dt );
        state_limits(idR,1,k,j,i  ) = r_tavg  (0     );
        state_limits(idU,1,k,j,i  ) = ru_tavg (0     );
        state_limits(idV,1,k,j,i  ) = rv_DTs(0,0     );
        state_limits(idW,1,k,j,i  ) = rw_DTs(0,0     );
        state_limits(idT,1,k,j,i  ) = rt_DTs(0,0     );
        state_limits(idR,0,k,j,i+1) = r_tavg  (ngll-1);
        state_limits(idU,0,k,j,i+1) = ru_tavg (ngll-1);
        state_limits(idV,0,k,j,i+1) = rv_DTs(0,ngll-1);
        state_limits(idW,0,k,j,i+1) = rw_DTs(0,ngll-1);
        state_limits(idT,0,k,j,i+1) = rt_DTs(0,ngll-1);

        for (int tr=0; tr < num_tracers; tr++) {
          SArray<real,2,tord,ngll> rt_DTs, rut_DTs;
          for (int ii=0; ii < ngll; ii++) {
            real r = state_gll  (idR,ii,k,j,i);
            real u = state_gll  (idU,ii,k,j,i);
            real t = tracers_gll(tr ,ii,k,j,i);
            rt_DTs      (0,ii) = r*t;
            rut_DTs     (0,ii) = r*u*t;
          }
          diffTransformTracer( r_DTs , ru_DTs , rt_DTs , rut_DTs , g2d2g , dx );
          compute_timeAvg( rt_DTs , dt );
          tracers_limits(tr,1,k,j,i  ) = rt_DTs(0,0     );
          tracers_limits(tr,0,k,j,i+1) = rt_DTs(0,ngll-1);
        }
      });

      // Perform periodic horizontal exchange of cell-edge data, and implement vertical boundary conditions
      edge_exchange_x( coupler , state_limits , tracers_limits , pressure_limits );

      real4d state_flux  ("state_flux"  ,num_state  ,nz,ny,nx+1);
      real4d tracers_flux("tracers_flux",num_tracers,nz,ny,nx+1);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      // Speed of sound and its reciprocal. Using a constant speed of sound for upwinding
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx+1) , YAKL_LAMBDA (int k, int j, int i) {
        // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
        real r_L  = state_limits(idR,0,k,j,i);   real r_R  = state_limits(idR,1,k,j,i);
        real ru_L = state_limits(idU,0,k,j,i);   real ru_R = state_limits(idU,1,k,j,i);
        real rt_L = state_limits(idT,0,k,j,i);   real rt_R = state_limits(idT,1,k,j,i);
        real p_L  = C0*std::pow(rt_L,gamma)  ;   real p_R  = C0*std::pow(rt_R,gamma)  ;
        real p_upw  = 0.5_fp*(p_L  + p_R  - cs*(ru_R-ru_L)   );
        real ru_upw = 0.5_fp*(ru_L + ru_R -    (p_R -p_L )/cs);
        // Advectively upwind everything else
        int ind = ru_upw > 0 ? 0 : 1;
        real r_r = 1._fp / state_limits(idR,ind,k,j,i);
        state_flux(idR,k,j,i) = ru_upw;
        state_flux(idU,k,j,i) = ru_upw*state_limits(idU,ind,k,j,i)*r_r + p_upw;
        state_flux(idV,k,j,i) = ru_upw*state_limits(idV,ind,k,j,i)*r_r;
        state_flux(idW,k,j,i) = ru_upw*state_limits(idW,ind,k,j,i)*r_r;
        state_flux(idT,k,j,i) = ru_upw*state_limits(idT,ind,k,j,i)*r_r;
        for (int tr=0; tr < num_tracers; tr++) {
          tracers_flux(tr,k,j,i) = ru_upw*tracers_limits(tr,ind,k,j,i)*r_r;
        }
        if (i < nx) {
          for (int l=1; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i); }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i); }
        }
      });

      // Compute tendencies as the flux divergence + gravity source term + coriolis
      int mx = std::max(num_state,num_tracers);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(mx,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tend(l,k,j,i) = -( state_flux(l,k,j,i+1) - state_flux(l,k,j,i) ) * r_dx;
          if (l == idV && sim2d) state_tend(l,k,j,i) = 0;
          if (latitude != 0 && !sim2d && l == idU) state_tend(l,k,j,i) += fcor*state(idV,hs+k,hs+j,hs+i);
        }
        if (l < num_tracers) { tracers_tend(l,k,j,i) = -( tracers_flux(l,k,j,i+1) - tracers_flux(l,k,j,i) ) * r_dx; }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("compute_tendencies");
      #endif
    }



    void compute_tendencies_y( core::Coupler       & coupler      ,
                               real4d        const & state        ,
                               real4d        const & state_tend   ,
                               real4d        const & tracers      ,
                               real4d        const & tracers_tend ,
                               real                  dt           ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("compute_tendencies");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      using yakl::intrinsics::matmul_cr;
      using yakl::COLON;
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
      auto  hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"            ); // Hydrostatic density
      auto  hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"           ); // Hydrostatic potential temperature
      auto  hy_dens_edges     = dm.get<real const,1>("hy_dens_edges"            ); // Hydrostatic density
      auto  hy_theta_edges    = dm.get<real const,1>("hy_theta_edges"           ); // Hydrostatic potential temperature
      auto  hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells"        ); // Hydrostatic pressure
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real r_dz = 1./dz; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

      SArray<real,2,ngll,ngll> g2d2g;
      {
        SArray<real,2,ngll,ngll> g2c;
        SArray<real,2,ngll,ngll> c2g;
        SArray<real,2,ngll,ngll> c2d;
        using yakl::intrinsics::matmul_cr;
        TransformMatrices::gll_to_coefs  (g2c);
        TransformMatrices::coefs_to_gll  (c2g);
        TransformMatrices::coefs_to_deriv(c2d);
        g2d2g = matmul_cr( c2g , matmul_cr( c2d , g2c ) );
      }
      SArray<real,2,ord,ngll> s2g;
      {
        SArray<real,2,ord,ord > s2c;
        SArray<real,2,ord,ngll> c2g;
        TransformMatrices::sten_to_coefs     (s2c);
        TransformMatrices::coefs_to_gll_lower(c2g);
        s2g = matmul_cr( c2g , s2c );
      }

      real3d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs); // Holds pressure perturbation

      // Compute pressure
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        pressure(hs+k,hs+j,hs+i) = C0*std::pow(state(idT,hs+k,hs+j,hs+i),gamma) - hy_pressure_cells(hs+k);
        for (int l=1; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) /= state(idR,hs+k,hs+j,hs+i); }
        for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) /= state(idR,hs+k,hs+j,hs+i); }
      });

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      {
        core::MultiField<real,3> fields;
        for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<3>(l,0,0,0) ); }
        for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<3>(l,0,0,0) ); }
        fields.add_field( pressure );
        if (ord > 1) coupler.halo_exchange_y( fields , hs );
      }

      // Create arrays to hold cell gll interpolations
      real5d state_gll   ("state_gll"   ,num_state  ,ngll,nz,ny,nx);
      real4d pressure_gll("pressure_gll"            ,ngll,nz,ny,nx);
      real5d tracers_gll ("tracers_gll" ,num_tracers,ngll,nz,ny,nx);
      
      // Aggregate fields for interpolation
      core::MultiField<real,3> fields;
      core::MultiField<real,4> gll_fields;
      int idP = 5;
      for (int l=0; l < num_state; l++) {
        fields    .add_field(state    .slice<3>(l,      COLON,COLON,COLON));
        gll_fields.add_field(state_gll.slice<4>(l,COLON,COLON,COLON,COLON));
      }
      fields    .add_field(pressure    .slice<3>(      COLON,COLON,COLON));
      gll_fields.add_field(pressure_gll.slice<4>(COLON,COLON,COLON,COLON));
      for (int l=0; l < num_tracers; l++) {
        fields    .add_field(tracers    .slice<3>(l      ,COLON,COLON,COLON));
        gll_fields.add_field(tracers_gll.slice<4>(l,COLON,COLON,COLON,COLON));
      }

      typedef limiter::WenoLimiter<real,ord> Limiter;

      // Interpolation to GLL points
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(fields.size(),nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        for (int jj=0; jj<ord; jj++) {
          immersed(jj) = immersed_prop(hs+k,j+jj,hs+i) > 0;
          stencil (jj) = fields     (l,hs+k,j+jj,hs+i);
        }
        if (l == idU || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        SArray<real,1,ngll> gll;
        Limiter::compute_limited_gll( stencil , gll , { ! any_immersed(k,j,i) , immersed(hs-1) , immersed(hs+1) } );
        for (int jj=0; jj < ngll; jj++) { gll_fields(l,jj,k,j,i) = gll(jj); }
      });

      // Create arrays to hold cell interface interpolations
      real5d state_limits   ("state_limits"   ,num_state  ,2,nz,ny+1,nx);
      real4d pressure_limits("pressure_limits"            ,2,nz,ny+1,nx);
      real5d tracers_limits ("tracers_limits" ,num_tracers,2,nz,ny+1,nx);

      // ADER Time averaging
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        SArray<real,2,tord,ngll> r_DTs, ru_DTs, rv_DTs, rw_DTs, rt_DTs, rvu_DTs, rvv_DTs, rvw_DTs, rvt_DTs, rt_gamma_DTs;
        for (int jj=0; jj < ngll; jj++) {
          real r = state_gll(idR,jj,k,j,i);
          real u = state_gll(idU,jj,k,j,i);
          real v = state_gll(idV,jj,k,j,i);
          real w = state_gll(idW,jj,k,j,i);
          real t = state_gll(idT,jj,k,j,i);
          r_DTs       (0,jj) = r;
          ru_DTs      (0,jj) = r*u;
          rv_DTs      (0,jj) = r*v;
          rw_DTs      (0,jj) = r*w;
          rt_DTs      (0,jj) = r*t;
          rvu_DTs     (0,jj) = r*v*u;
          rvv_DTs     (0,jj) = r*v*v;
          rvw_DTs     (0,jj) = r*v*w;
          rvt_DTs     (0,jj) = r*v*t;
          rt_gamma_DTs(0,jj) = std::pow(r*t,gamma);
        }
        diffTransformEulerConsY( r_DTs, ru_DTs, rv_DTs, rw_DTs, rt_DTs, rvu_DTs, rvv_DTs, rvw_DTs, rvt_DTs,
                                 rt_gamma_DTs, g2d2g , C0 , gamma , dy );
        SArray<real,1,ngll> r_tavg, rv_tavg;
        compute_timeAvg( r_DTs  , r_tavg  , dt );
        compute_timeAvg( ru_DTs           , dt );
        compute_timeAvg( rv_DTs , rv_tavg , dt );
        compute_timeAvg( rw_DTs           , dt );
        compute_timeAvg( rt_DTs           , dt );
        state_limits(idR,1,k,j  ,i) = r_tavg  (0     );
        state_limits(idU,1,k,j  ,i) = ru_DTs(0,0     );
        state_limits(idV,1,k,j  ,i) = rv_tavg (0     );
        state_limits(idW,1,k,j  ,i) = rw_DTs(0,0     );
        state_limits(idT,1,k,j  ,i) = rt_DTs(0,0     );
        state_limits(idR,0,k,j+1,i) = r_tavg  (ngll-1);
        state_limits(idU,0,k,j+1,i) = ru_DTs(0,ngll-1);
        state_limits(idV,0,k,j+1,i) = rv_tavg (ngll-1);
        state_limits(idW,0,k,j+1,i) = rw_DTs(0,ngll-1);
        state_limits(idT,0,k,j+1,i) = rt_DTs(0,ngll-1);

        for (int tr=0; tr < num_tracers; tr++) {
          SArray<real,2,tord,ngll> rt_DTs, rvt_DTs;
          for (int jj=0; jj < ngll; jj++) {
            real r = state_gll  (idR,jj,k,j,i);
            real v = state_gll  (idV,jj,k,j,i);
            real t = tracers_gll(tr ,jj,k,j,i);
            rt_DTs      (0,jj) = r*t;
            rvt_DTs     (0,jj) = r*v*t;
          }
          diffTransformTracer( r_DTs , rv_DTs , rt_DTs , rvt_DTs , g2d2g , dy );
          compute_timeAvg( rt_DTs , dt );
          tracers_limits(tr,1,k,j  ,i) = rt_DTs(0,0     );
          tracers_limits(tr,0,k,j+1,i) = rt_DTs(0,ngll-1);
        }
      });

      // Perform periodic horizontal exchange of cell-edge data, and implement vertical boundary conditions
      edge_exchange_y( coupler , state_limits , tracers_limits , pressure_limits );

      real4d state_flux  ("state_flux"  ,num_state  ,nz,ny+1,nx);
      real4d tracers_flux("tracers_flux",num_tracers,nz,ny+1,nx);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      // Speed of sound and its reciprocal. Using a constant speed of sound for upwinding
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny+1,nx) , YAKL_LAMBDA (int k, int j, int i) {
        // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
        real r_L  = state_limits   (idR,0,k,j,i);   real r_R  = state_limits   (idR,1,k,j,i);
        real rv_L = state_limits   (idV,0,k,j,i);   real rv_R = state_limits   (idV,1,k,j,i);
        real rt_L = state_limits   (idT,0,k,j,i);   real rt_R = state_limits   (idT,1,k,j,i);
        real p_L  = C0*std::pow(rt_L,gamma)     ;   real p_R  = C0*std::pow(rt_R,gamma)     ;
        real p_upw  = 0.5_fp*(p_L  + p_R  - cs*(rv_R-rv_L)   );
        real rv_upw = 0.5_fp*(rv_L + rv_R -    (p_R -p_L )/cs);
        // Advectively upwind everything else
        int ind = rv_upw > 0 ? 0 : 1;
        real r_r = 1._fp / state_limits(idR,ind,k,j,i);
        state_flux(idR,k,j,i) = rv_upw;
        state_flux(idU,k,j,i) = rv_upw*state_limits(idU,ind,k,j,i)*r_r;
        state_flux(idV,k,j,i) = rv_upw*state_limits(idV,ind,k,j,i)*r_r + p_upw;
        state_flux(idW,k,j,i) = rv_upw*state_limits(idW,ind,k,j,i)*r_r;
        state_flux(idT,k,j,i) = rv_upw*state_limits(idT,ind,k,j,i)*r_r;
        for (int tr=0; tr < num_tracers; tr++) {
          tracers_flux(tr,k,j,i) = rv_upw*tracers_limits(tr,ind,k,j,i)*r_r;
        }
        if (j < ny) {
          for (int l=1; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i); }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i); }
        }
      });

      // Compute tendencies as the flux divergence + gravity source term + coriolis
      int mx = std::max(num_state,num_tracers);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(mx,nz,ny,nx) , YAKL_LAMBDA (int l, int k, int j, int i) {
        if (l < num_state) {
          state_tend(l,k,j,i) = -( state_flux(l,k,j+1,i) - state_flux(l,k,j,i) ) * r_dy;
          if (l == idV && sim2d) state_tend(l,k,j,i) = 0;
          if (latitude != 0 && !sim2d && l == idV) state_tend(l,k,j,i) -= fcor*state(idU,hs+k,hs+j,hs+i);
        }
        if (l < num_tracers) {
          tracers_tend(l,k,j,i) = -( tracers_flux(l,k,j+1,i) - tracers_flux(l,k,j,i) ) * r_dy;
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("compute_tendencies");
      #endif
    }



    // Compute semi-discrete tendencies in x, y, and z directions
    // Fully split in dimensions, and coupled together inside RK stages
    // dt is not used at the moment
    void compute_tendencies_z( core::Coupler       & coupler      ,
                               real4d        const & state        ,
                               real4d        const & state_tend   ,
                               real4d        const & tracers      ,
                               real4d        const & tracers_tend ,
                               real                  dt           ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("compute_tendencies");
      #endif
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      using yakl::intrinsics::matmul_cr;
      using yakl::COLON;
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
      auto  hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"            ); // Hydrostatic density
      auto  hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"           ); // Hydrostatic potential temperature
      auto  hy_dens_edges     = dm.get<real const,1>("hy_dens_edges"            ); // Hydrostatic density
      auto  hy_theta_edges    = dm.get<real const,1>("hy_theta_edges"           ); // Hydrostatic potential temperature
      auto  hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells"        ); // Hydrostatic pressure
      auto  hy_pressure_edges = dm.get<real const,1>("hy_pressure_edges"        ); // Hydrostatic pressure
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real r_dz = 1./dz; // reciprocal of grid spacing
      real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

      SArray<real,2,ngll,ngll> g2d2g;
      {
        SArray<real,2,ngll,ngll> g2c;
        SArray<real,2,ngll,ngll> c2g;
        SArray<real,2,ngll,ngll> c2d;
        using yakl::intrinsics::matmul_cr;
        TransformMatrices::gll_to_coefs  (g2c);
        TransformMatrices::coefs_to_gll  (c2g);
        TransformMatrices::coefs_to_deriv(c2d);
        g2d2g = matmul_cr( c2g , matmul_cr( c2d , g2c ) );
      }
      SArray<real,2,ord,ngll> s2g;
      {
        SArray<real,2,ord,ord > s2c;
        SArray<real,2,ord,ngll> c2g;
        TransformMatrices::sten_to_coefs     (s2c);
        TransformMatrices::coefs_to_gll_lower(c2g);
        s2g = matmul_cr( c2g , s2c );
      }

      SArray<real,1,ngll> gll_wts;
      TransformMatrices::get_gll_weights(gll_wts);

      real3d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs); // Holds pressure perturbation

      // Compute pressure
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        pressure(hs+k,hs+j,hs+i) = C0*std::pow(state(idT,hs+k,hs+j,hs+i),gamma) - hy_pressure_cells(hs+k);
        for (int l=1; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) /= state(idR,hs+k,hs+j,hs+i); }
        for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) /= state(idR,hs+k,hs+j,hs+i); }
      });

      halo_boundary_conditions( coupler , state , tracers , pressure );

      // Create arrays to hold cell gll interpolations
      real5d state_gll   ("state_gll"   ,num_state  ,ngll,nz,ny,nx);
      real4d pressure_gll("pressure_gll"            ,ngll,nz,ny,nx);
      real5d tracers_gll ("tracers_gll" ,num_tracers,ngll,nz,ny,nx);
      
      // Aggregate fields for interpolation
      core::MultiField<real,3> fields;
      core::MultiField<real,4> gll_fields;
      int idP = 5;
      for (int l=0; l < num_state; l++) {
        fields    .add_field(state    .slice<3>(l,      COLON,COLON,COLON));
        gll_fields.add_field(state_gll.slice<4>(l,COLON,COLON,COLON,COLON));
      }
      fields    .add_field(pressure    .slice<3>(      COLON,COLON,COLON));
      gll_fields.add_field(pressure_gll.slice<4>(COLON,COLON,COLON,COLON));
      for (int l=0; l < num_tracers; l++) {
        fields    .add_field(tracers    .slice<3>(l      ,COLON,COLON,COLON));
        gll_fields.add_field(tracers_gll.slice<4>(l,COLON,COLON,COLON,COLON));
      }

      typedef limiter::WenoLimiter<real,ord> Limiter;

      // Interpolation to GLL points
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(fields.size(),nz,ny,nx) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i) {
        SArray<real,1,ord> stencil;
        SArray<bool,1,ord> immersed;
        for (int kk=0; kk<ord; kk++) {
          immersed(kk) = immersed_prop(k+kk,hs+j,hs+i) > 0;
          stencil (kk) = fields     (l,k+kk,hs+j,hs+i);
        }
        if (l == idU || l == idV || l == idP) modify_stencil_immersed_der0( stencil , immersed );
        SArray<real,1,ngll> gll;
        Limiter::compute_limited_gll( stencil , gll , { ! any_immersed(k,j,i) , immersed(hs-1) , immersed(hs+1) } );
        for (int kk=0; kk < ngll; kk++) { gll_fields(l,kk,k,j,i) = gll(kk); }
      });

      // Create arrays to hold cell interface interpolations
      real5d state_limits   ("state_limits"   ,num_state  ,2,nz+1,ny,nx);
      real4d pressure_limits("pressure_limits"            ,2,nz+1,ny,nx);
      real5d tracers_limits ("tracers_limits" ,num_tracers,2,nz+1,ny,nx);

      // ADER Time averaging
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        SArray<real,2,tord,ngll> r_DTs, ru_DTs, rv_DTs, rw_DTs, rt_DTs, rwu_DTs, rwv_DTs, rww_DTs, rwt_DTs, rt_gamma_DTs;
        for (int kk=0; kk < ngll; kk++) {
          real r = state_gll(idR,kk,k,j,i);
          real u = state_gll(idU,kk,k,j,i);
          real v = state_gll(idV,kk,k,j,i);
          real w = state_gll(idW,kk,k,j,i);
          real t = state_gll(idT,kk,k,j,i);
          r_DTs       (0,kk) = r;
          ru_DTs      (0,kk) = r*u;
          rv_DTs      (0,kk) = r*v;
          rw_DTs      (0,kk) = r*w;
          rt_DTs      (0,kk) = r*t;
          rwu_DTs     (0,kk) = r*w*u;
          rwv_DTs     (0,kk) = r*w*v;
          rww_DTs     (0,kk) = r*w*w;
          rwt_DTs     (0,kk) = r*w*t;
          rt_gamma_DTs(0,kk) = std::pow(r*t,gamma);
        }
        diffTransformEulerConsZ( r_DTs, ru_DTs, rv_DTs, rw_DTs, rt_DTs, rwu_DTs, rwv_DTs, rww_DTs, rwt_DTs,
                                 rt_gamma_DTs, g2d2g , C0 , gamma , grav , dz );
        SArray<real,1,ngll> r_tavg, rw_tavg;
        compute_timeAvg( r_DTs  , r_tavg  , dt );
        compute_timeAvg( ru_DTs           , dt );
        compute_timeAvg( rv_DTs           , dt );
        compute_timeAvg( rw_DTs , rw_tavg , dt );
        compute_timeAvg( rt_DTs           , dt );
        state_limits(idR,1,k  ,j,i) = r_tavg  (0     );
        state_limits(idU,1,k  ,j,i) = ru_DTs(0,0     );
        state_limits(idV,1,k  ,j,i) = rv_DTs(0,0     );
        state_limits(idW,1,k  ,j,i) = rw_tavg (0     );
        state_limits(idT,1,k  ,j,i) = rt_DTs(0,0     );
        state_limits(idR,0,k+1,j,i) = r_tavg  (ngll-1);
        state_limits(idU,0,k+1,j,i) = ru_DTs(0,ngll-1);
        state_limits(idV,0,k+1,j,i) = rv_DTs(0,ngll-1);
        state_limits(idW,0,k+1,j,i) = rw_tavg (ngll-1);
        state_limits(idT,0,k+1,j,i) = rt_DTs(0,ngll-1);

        real r_avg = 0;
        for (int kk = 0; kk < ngll; kk++) { r_avg += r_tavg(kk)*gll_wts(kk); }
        state_tend(idW,k,j,i) = -grav*(r_avg - hy_dens_cells(hs+k));

        for (int tr=0; tr < num_tracers; tr++) {
          SArray<real,2,tord,ngll> rt_DTs, rwt_DTs;
          for (int kk=0; kk < ngll; kk++) {
            real r = state_gll  (idR,kk,k,j,i);
            real w = state_gll  (idW,kk,k,j,i);
            real t = tracers_gll(tr ,kk,k,j,i);
            rt_DTs      (0,kk) = r*t;
            rwt_DTs     (0,kk) = r*w*t;
          }
          diffTransformTracer( r_DTs , rw_DTs , rt_DTs , rwt_DTs , g2d2g , dz );
          compute_timeAvg( rt_DTs , dt );
          tracers_limits(tr,1,k  ,j,i) = rt_DTs(0,0     );
          tracers_limits(tr,0,k+1,j,i) = rt_DTs(0,ngll-1);
        }
      });


      // Perform periodic horizontal exchange of cell-edge data, and implement vertical boundary conditions
      edge_exchange_z( coupler , state_limits , tracers_limits , pressure_limits );

      real4d state_flux  ("state_flux"  ,num_state  ,nz+1,ny,nx);
      real4d tracers_flux("tracers_flux",num_tracers,nz+1,ny,nx);

      // Use upwind Riemann solver to reconcile discontinuous limits of state and tracers at each cell edges
      // Speed of sound and its reciprocal. Using a constant speed of sound for upwinding
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
        real r_L  = state_limits   (idR,0,k,j,i);   real r_R  = state_limits   (idR,1,k,j,i);
        real rw_L = state_limits   (idW,0,k,j,i);   real rw_R = state_limits   (idW,1,k,j,i);
        real rt_L = state_limits   (idT,0,k,j,i);   real rt_R = state_limits   (idT,1,k,j,i);
        real p_L  = C0*std::pow(rt_L,gamma)     ;   real p_R  = C0*std::pow(rt_R,gamma)     ;
        real p_upw  = 0.5_fp*(p_L  + p_R  - cs*(rw_R-rw_L)   );
        real rw_upw = 0.5_fp*(rw_L + rw_R -    (p_R -p_L )/cs);
        // Advectively upwind everything else
        int ind = rw_upw > 0 ? 0 : 1;
        real r_r = 1._fp / state_limits(idR,ind,k,j,i);
        state_flux(idR,k,j,i) = rw_upw;
        state_flux(idU,k,j,i) = rw_upw*state_limits(idU,ind,k,j,i)*r_r;
        state_flux(idV,k,j,i) = rw_upw*state_limits(idV,ind,k,j,i)*r_r;
        state_flux(idW,k,j,i) = rw_upw*state_limits(idW,ind,k,j,i)*r_r + p_upw - hy_pressure_edges(k);
        state_flux(idT,k,j,i) = rw_upw*state_limits(idT,ind,k,j,i)*r_r;
        for (int tr=0; tr < num_tracers; tr++) {
          tracers_flux(tr,k,j,i) = rw_upw*tracers_limits(tr,ind,k,j,i)*r_r;
        }
        if (k < nz) {
          for (int l=1; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i); }
          for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i); }
        }
      });

      // Compute tendencies as the flux divergence + gravity source term + coriolis
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        state_tend(idR,k,j,i)  = -( state_flux(idR,k+1,j,i) - state_flux(idR,k,j,i) ) * r_dz;
        state_tend(idU,k,j,i)  = -( state_flux(idU,k+1,j,i) - state_flux(idU,k,j,i) ) * r_dz;
        state_tend(idV,k,j,i)  = -( state_flux(idV,k+1,j,i) - state_flux(idV,k,j,i) ) * r_dz;
        state_tend(idW,k,j,i) += -( state_flux(idW,k+1,j,i) - state_flux(idW,k,j,i) ) * r_dz;
        state_tend(idT,k,j,i)  = -( state_flux(idT,k+1,j,i) - state_flux(idT,k,j,i) ) * r_dz;
        for (int l = 0; l < num_tracers; l++) {
          tracers_tend(l,k,j,i) = -( tracers_flux(l,k+1,j,i) - tracers_flux(l,k,j,i) ) * r_dz;
        }
      });
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("compute_tendencies");
      #endif
    }



    void halo_boundary_conditions( core::Coupler const & coupler  ,
                                   real4d       const & state    ,
                                   real4d       const & tracers  ,
                                   real3d       const & pressure ) const {
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
      auto hy_dens_cells   = dm.get<real const,1>("hy_dens_cells" );
      auto hy_theta_cells  = dm.get<real const,1>("hy_theta_cells");
      auto surface_temp    = dm.get<real const,2>("surface_temp");

      // z-direction BC's
      if (bc_z == "solid_wall") {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny,nx) , YAKL_LAMBDA (int kk, int j, int i) {
          state(idR,kk,hs+j,hs+i) = hy_dens_cells(kk);
          state(idU,kk,hs+j,hs+i) = state(idU,hs+0,hs+j,hs+i);
          state(idV,kk,hs+j,hs+i) = state(idV,hs+0,hs+j,hs+i);
          state(idW,kk,hs+j,hs+i) = 0;
          state(idT,kk,hs+j,hs+i) = state(idT,hs+0,hs+j,hs+i);
          pressure( kk,hs+j,hs+i) = pressure (hs+0,hs+j,hs+i);
          state(idR,hs+nz+kk,hs+j,hs+i) = hy_dens_cells(hs+nz+kk);
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



    void edge_exchange_x( core::Coupler const & coupler           ,
                          real5d       const & state_limits_x    ,
                          real5d       const & tracers_limits_x  ,
                          real4d       const & pressure_limits_x ) const {
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_start("edge_exchange_x");
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
      auto hy_dens_edges  = dm.get<real const,1>("hy_dens_edges" );
      auto hy_theta_edges = dm.get<real const,1>("hy_theta_edges");
      auto surface_temp   = dm.get<real const,2>("surface_temp"  );
      int npack = num_state + num_tracers+1;

      // x-exchange
      {
        real3d edge_send_buf_W("edge_send_buf_W",npack,nz,ny);
        real3d edge_send_buf_E("edge_send_buf_E",npack,nz,ny);
        real3d edge_recv_buf_W("edge_recv_buf_W",npack,nz,ny);
        real3d edge_recv_buf_E("edge_recv_buf_E",npack,nz,ny);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
          if        (v < num_state) {
            edge_send_buf_W(v,k,j) = state_limits_x  (v          ,1,k,j,0 );
            edge_send_buf_E(v,k,j) = state_limits_x  (v          ,0,k,j,nx);
          } else if (v < num_state + num_tracers) {                    
            edge_send_buf_W(v,k,j) = tracers_limits_x(v-num_state,1,k,j,0 );
            edge_send_buf_E(v,k,j) = tracers_limits_x(v-num_state,0,k,j,nx);
          } else {
            edge_send_buf_W(v,k,j) = pressure_limits_x(1,k,j,0 );
            edge_send_buf_E(v,k,j) = pressure_limits_x(0,k,j,nx);
          }
        });
        coupler.get_parallel_comm().send_receive<real,3>( { {edge_recv_buf_W,neigh(1,0),4} , {edge_recv_buf_E,neigh(1,2),5} } ,
                                                          { {edge_send_buf_W,neigh(1,0),5} , {edge_send_buf_E,neigh(1,2),4} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,ny) , YAKL_LAMBDA (int v, int k, int j) {
          if        (v < num_state) {
            state_limits_x  (v          ,0,k,j,0 ) = edge_recv_buf_W(v,k,j);
            state_limits_x  (v          ,1,k,j,nx) = edge_recv_buf_E(v,k,j);
          } else if (v < num_state + num_tracers) {
            tracers_limits_x(v-num_state,0,k,j,0 ) = edge_recv_buf_W(v,k,j);
            tracers_limits_x(v-num_state,1,k,j,nx) = edge_recv_buf_E(v,k,j);
          } else {
            pressure_limits_x(0,k,j,0 ) = edge_recv_buf_W(v,k,j);
            pressure_limits_x(1,k,j,nx) = edge_recv_buf_E(v,k,j);
          }
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("edge_exchange_x");
      #endif
    }



    void edge_exchange_y( core::Coupler const & coupler          ,
                          real5d       const & state_limits_y    ,
                          real5d       const & tracers_limits_y  ,
                          real4d       const & pressure_limits_y ) const {
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
      auto hy_dens_edges  = dm.get<real const,1>("hy_dens_edges" );
      auto hy_theta_edges = dm.get<real const,1>("hy_theta_edges");
      auto surface_temp   = dm.get<real const,2>("surface_temp"  );
      int npack = num_state + num_tracers+1;

      // y-direction exchange
      {
        real3d edge_send_buf_S("edge_send_buf_S",npack,nz,nx);
        real3d edge_send_buf_N("edge_send_buf_N",npack,nz,nx);
        real3d edge_recv_buf_S("edge_recv_buf_S",npack,nz,nx);
        real3d edge_recv_buf_N("edge_recv_buf_N",npack,nz,nx);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if        (v < num_state) {
            edge_send_buf_S(v,k,i) = state_limits_y  (v          ,1,k,0 ,i);
            edge_send_buf_N(v,k,i) = state_limits_y  (v          ,0,k,ny,i);
          } else if (v < num_state + num_tracers) {                    
            edge_send_buf_S(v,k,i) = tracers_limits_y(v-num_state,1,k,0 ,i);
            edge_send_buf_N(v,k,i) = tracers_limits_y(v-num_state,0,k,ny,i);
          } else {
            edge_send_buf_S(v,k,i) = pressure_limits_y(1,k,0 ,i);
            edge_send_buf_N(v,k,i) = pressure_limits_y(0,k,ny,i);
          }
        });
        coupler.get_parallel_comm().send_receive<real,3>( { {edge_recv_buf_S,neigh(0,1),6} , {edge_recv_buf_N,neigh(2,1),7} } ,
                                                           { {edge_send_buf_S,neigh(0,1),7} , {edge_send_buf_N,neigh(2,1),6} } );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(npack,nz,nx) , YAKL_LAMBDA (int v, int k, int i) {
          if        (v < num_state) {
            state_limits_y  (v          ,0,k,0 ,i) = edge_recv_buf_S(v,k,i);
            state_limits_y  (v          ,1,k,ny,i) = edge_recv_buf_N(v,k,i);
          } else if (v < num_state + num_tracers) {
            tracers_limits_y(v-num_state,0,k,0 ,i) = edge_recv_buf_S(v,k,i);
            tracers_limits_y(v-num_state,1,k,ny,i) = edge_recv_buf_N(v,k,i);
          } else {
            pressure_limits_y(0,k,0 ,i) = edge_recv_buf_S(v,k,i);
            pressure_limits_y(1,k,ny,i) = edge_recv_buf_N(v,k,i);
          }
        });
      }
      #ifdef YAKL_AUTO_PROFILE
        yakl::timer_stop("edge_exchange");
      #endif
    }



    void edge_exchange_z( core::Coupler const & coupler          ,
                          real5d       const & state_limits_z    ,
                          real5d       const & tracers_limits_z  ,
                          real4d       const & pressure_limits_z ) const {
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
      auto hy_dens_edges  = dm.get<real const,1>("hy_dens_edges" );
      auto hy_theta_edges = dm.get<real const,1>("hy_theta_edges");
      auto surface_temp   = dm.get<real const,2>("surface_temp"  );
      int npack = num_state + num_tracers+1;

      if (bc_z == "solid_wall") {
        // z-direction BC's
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
          // Dirichlet
          state_limits_z(idW,0,0 ,j,i) = 0;
          state_limits_z(idW,1,0 ,j,i) = 0;
          state_limits_z(idW,0,nz,j,i) = 0;
          state_limits_z(idW,1,nz,j,i) = 0;
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(l,0,0 ,j,i) = 0;
            tracers_limits_z(l,1,0 ,j,i) = 0;
            tracers_limits_z(l,0,nz,j,i) = 0;
            tracers_limits_z(l,1,nz,j,i) = 0;
          }
          // Neumann
          state_limits_z   (idR,0,0 ,j,i) = state_limits_z   (idR,1,0 ,j,i);
          state_limits_z   (idU,0,0 ,j,i) = state_limits_z   (idU,1,0 ,j,i);
          state_limits_z   (idV,0,0 ,j,i) = state_limits_z   (idV,1,0 ,j,i);
          state_limits_z   (idT,0,0 ,j,i) = state_limits_z   (idT,1,0 ,j,i);
          pressure_limits_z(    0,0 ,j,i) = pressure_limits_z(    1,0 ,j,i);
          state_limits_z   (idR,1,nz,j,i) = state_limits_z   (idR,0,nz,j,i);
          state_limits_z   (idU,1,nz,j,i) = state_limits_z   (idU,0,nz,j,i);
          state_limits_z   (idV,1,nz,j,i) = state_limits_z   (idV,0,nz,j,i);
          state_limits_z   (idT,1,nz,j,i) = state_limits_z   (idT,0,nz,j,i);
          pressure_limits_z(    1,nz,j,i) = pressure_limits_z(    0,nz,j,i);
        });
      } else if (bc_z == "periodic") {
        // z-direction BC's
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , YAKL_LAMBDA (int j, int i) {
          state_limits_z   (idR,0,0 ,j,i) = state_limits_z   (idR,0,nz,j,i);
          state_limits_z   (idU,0,0 ,j,i) = state_limits_z   (idU,0,nz,j,i);
          state_limits_z   (idV,0,0 ,j,i) = state_limits_z   (idV,0,nz,j,i);
          state_limits_z   (idW,0,0 ,j,i) = state_limits_z   (idW,0,nz,j,i);
          state_limits_z   (idT,0,0 ,j,i) = state_limits_z   (idT,0,nz,j,i);
          pressure_limits_z(    0,0 ,j,i) = pressure_limits_z(    0,nz,j,i);
          state_limits_z   (idR,1,nz,j,i) = state_limits_z   (idR,1,0 ,j,i);
          state_limits_z   (idU,1,nz,j,i) = state_limits_z   (idU,1,0 ,j,i);
          state_limits_z   (idV,1,nz,j,i) = state_limits_z   (idV,1,0 ,j,i);
          state_limits_z   (idW,1,nz,j,i) = state_limits_z   (idW,1,0 ,j,i);
          state_limits_z   (idT,1,nz,j,i) = state_limits_z   (idT,1,0 ,j,i);
          pressure_limits_z(    1,nz,j,i) = pressure_limits_z(    1,0 ,j,i);
          for (int l=0; l < num_tracers; l++) {
            tracers_limits_z(l,0,0 ,j,i) = tracers_limits_z(l,0,nz,j,i);
            tracers_limits_z(l,1,nz,j,i) = tracers_limits_z(l,1,0 ,j,i);
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
      dm.register_and_allocate<real>("hy_dens_cells"    ,"",{nz+2*hs});
      dm.register_and_allocate<real>("hy_theta_cells"   ,"",{nz+2*hs});
      dm.register_and_allocate<real>("hy_pressure_cells","",{nz+2*hs});
      auto r = dm.get<real,1>("hy_dens_cells"    );    r = 0;
      auto t = dm.get<real,1>("hy_theta_cells"   );    t = 0;
      auto p = dm.get<real,1>("hy_pressure_cells");    p = 0;
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
            hy_pressure_edges(k) = -1./12.*hy_pressure_cells(hs+k-2) +
                                    7./12.*hy_pressure_cells(hs+k-1) +
                                    7./12.*hy_pressure_cells(hs+k  ) +
                                   -1./12.*hy_pressure_cells(hs+k+1);
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
        nc.create_var<real>( "theta_pert"        , {"z","y","x"});
        nc.create_var<real>( "pressure_pert"     , {"z","y","x"});
        nc.create_var<real>( "density_pert"      , {"z","y","x"});
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
        auto hy_dens_cells = dm.get<real const,1>("hy_dens_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = state(idR,hs+k,hs+j,hs+i) - hy_dens_cells(hs+k);
        });
        nc.write_all(data,"density_pert",start_3d);
        auto hy_theta_cells = dm.get<real const,1>("hy_theta_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = state(idT,hs+k,hs+j,hs+i) / state(idR,hs+k,hs+j,hs+i) - hy_theta_cells(hs+k);
        });
        nc.write_all(data,"theta_pert",start_3d);
        auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells");
        yakl::c::parallel_for( yakl::c::Bounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
          data(k,j,i) = C0 * std::pow( state(idT,hs+k,hs+j,hs+i) , gamma ) - hy_pressure_cells(hs+k);
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



    template <yakl::index_t nAder, yakl::index_t ngll>
    YAKL_INLINE void static diffTransformEulerConsX( SArray<real,2,nAder,ngll> &r  ,
                                                     SArray<real,2,nAder,ngll> &ru ,
                                                     SArray<real,2,nAder,ngll> &rv ,
                                                     SArray<real,2,nAder,ngll> &rw ,
                                                     SArray<real,2,nAder,ngll> &rt ,
                                                     SArray<real,2,nAder,ngll> &ruu ,
                                                     SArray<real,2,nAder,ngll> &ruv ,
                                                     SArray<real,2,nAder,ngll> &ruw ,
                                                     SArray<real,2,nAder,ngll> &rut ,
                                                     SArray<real,2,nAder,ngll> &rt_gamma ,
                                                     SArray<real,2,ngll,ngll> const &deriv ,
                                                     real C0 ,
                                                     real GAMMA ,
                                                     real dx ) {
      // zero out the non-linear DTs
      for (int kt=1; kt < nAder; kt++) {
        for (int ii=0; ii < ngll; ii++) {
          ruu     (kt,ii) = 0;
          ruv     (kt,ii) = 0;
          ruw     (kt,ii) = 0;
          rut     (kt,ii) = 0;
          rt_gamma(kt,ii) = 0;
        }
      }

      // Loop over the time derivatives
      for (int kt=0; kt<nAder-1; kt++) {
        // Compute the state at the next time level
        for (int ii=0; ii<ngll; ii++) {
          real df1_dx = 0;
          real df2_dx = 0;
          real df3_dx = 0;
          real df4_dx = 0;
          real df5_dx = 0;
          for (int s=0; s<ngll; s++) {
            df1_dx += deriv(s,ii) * ( ru (kt,s) );
            if (kt == 0) { df2_dx += deriv(s,ii) * ( ruu(kt,s) + C0*rt_gamma(kt,s)   ); }
            else         { df2_dx += deriv(s,ii) * ( ruu(kt,s) + C0*rt_gamma(kt,s)/2 ); }
            df3_dx += deriv(s,ii) * ( ruv(kt,s) );
            df4_dx += deriv(s,ii) * ( ruw(kt,s) );
            df5_dx += deriv(s,ii) * ( rut(kt,s) );
          }
          r (kt+1,ii) = -df1_dx/dx/(kt+1._fp);
          ru(kt+1,ii) = -df2_dx/dx/(kt+1._fp);
          rv(kt+1,ii) = -df3_dx/dx/(kt+1._fp);
          rw(kt+1,ii) = -df4_dx/dx/(kt+1._fp);
          rt(kt+1,ii) = -df5_dx/dx/(kt+1._fp);
        }

        // Compute ru* at the next time level
        for (int ii=0; ii<ngll; ii++) {
          real tot_ruu = 0;
          real tot_ruv = 0;
          real tot_ruw = 0;
          real tot_rut = 0;
          for (int ir=0; ir<=kt+1; ir++) {
            tot_ruu += ru(ir,ii) * ru(kt+1-ir,ii) - r(ir,ii) * ruu(kt+1-ir,ii);
            tot_ruv += ru(ir,ii) * rv(kt+1-ir,ii) - r(ir,ii) * ruv(kt+1-ir,ii);
            tot_ruw += ru(ir,ii) * rw(kt+1-ir,ii) - r(ir,ii) * ruw(kt+1-ir,ii);
            tot_rut += ru(ir,ii) * rt(kt+1-ir,ii) - r(ir,ii) * rut(kt+1-ir,ii);
          }
          ruu(kt+1,ii) = tot_ruu / r(0,ii);
          ruv(kt+1,ii) = tot_ruv / r(0,ii);
          ruw(kt+1,ii) = tot_ruw / r(0,ii);
          rut(kt+1,ii) = tot_rut / r(0,ii);

          // Compute rt_gamma at the next time level
          real tot_rt_gamma = 0;
          for (int ir=0; ir<=kt; ir++) {
            tot_rt_gamma += (kt+1._fp -ir) * ( GAMMA*rt_gamma(ir,ii)*rt(kt+1-ir,ii) - rt(ir,ii)*rt_gamma(kt+1-ir,ii) );
          }
          rt_gamma(kt+1,ii) = ( GAMMA*rt_gamma(0,ii)*rt(kt+1,ii) + tot_rt_gamma / (kt+1._fp) ) / rt(0,ii);
        }
      }

      // Fix the rt_gamma
      for (int kt=1; kt < nAder; kt++) {
        for (int ii=0; ii < ngll; ii++) {
          rt_gamma(kt,ii) /= 2;
        }
      }
    }



    template <yakl::index_t nAder, yakl::index_t ngll>
    YAKL_INLINE static void diffTransformEulerConsY( SArray<real,2,nAder,ngll> &r  ,
                                                     SArray<real,2,nAder,ngll> &ru ,
                                                     SArray<real,2,nAder,ngll> &rv ,
                                                     SArray<real,2,nAder,ngll> &rw ,
                                                     SArray<real,2,nAder,ngll> &rt ,
                                                     SArray<real,2,nAder,ngll> &rvu ,
                                                     SArray<real,2,nAder,ngll> &rvv ,
                                                     SArray<real,2,nAder,ngll> &rvw ,
                                                     SArray<real,2,nAder,ngll> &rvt ,
                                                     SArray<real,2,nAder,ngll> &rt_gamma ,
                                                     SArray<real,2,ngll,ngll> const &deriv , 
                                                     real C0 ,
                                                     real GAMMA ,
                                                     real dy ) {
      // zero out the non-linear DTs
      for (int kt=1; kt < nAder; kt++) {
        for (int ii=0; ii < ngll; ii++) {
          rvu     (kt,ii) = 0;
          rvv     (kt,ii) = 0;
          rvw     (kt,ii) = 0;
          rvt     (kt,ii) = 0;
          rt_gamma(kt,ii) = 0;
        }
      }

      // Loop over the time derivatives
      for (int kt=0; kt<nAder-1; kt++) {
        // Compute the state at the next time level
        for (int ii=0; ii<ngll; ii++) {
          real drv_dy    = 0;
          real drvu_dy   = 0;
          real drvv_p_dy = 0;
          real drvw_dy   = 0;
          real drvt_dy   = 0;
          for (int s=0; s<ngll; s++) {
            drv_dy    += deriv(s,ii) * rv(kt,s);
            drvu_dy   += deriv(s,ii) * rvu(kt,s);
            if (kt == 0) { drvv_p_dy += deriv(s,ii) * ( rvv(kt,s) + C0*rt_gamma(kt,s)   ); }
            else         { drvv_p_dy += deriv(s,ii) * ( rvv(kt,s) + C0*rt_gamma(kt,s)/2 ); }
            drvw_dy   += deriv(s,ii) * rvw(kt,s);
            drvt_dy   += deriv(s,ii) * rvt(kt,s);
          }
          r (kt+1,ii) = -drv_dy   /dy/(kt+1);
          ru(kt+1,ii) = -drvu_dy  /dy/(kt+1);
          rv(kt+1,ii) = -drvv_p_dy/dy/(kt+1);
          rw(kt+1,ii) = -drvw_dy  /dy/(kt+1);
          rt(kt+1,ii) = -drvt_dy  /dy/(kt+1);
        }

        // Compute ru* at the next time level
        for (int ii=0; ii<ngll; ii++) {
          // Compute the non-linear differential transforms
          real tot_rvu = 0;
          real tot_rvv = 0;
          real tot_rvw = 0;
          real tot_rvt = 0;
          for (int l=0; l<=kt+1; l++) {
            tot_rvu += rv(l,ii) * ru(kt+1-l,ii) - r(l,ii) * rvu(kt+1-l,ii);
            tot_rvv += rv(l,ii) * rv(kt+1-l,ii) - r(l,ii) * rvv(kt+1-l,ii);
            tot_rvw += rv(l,ii) * rw(kt+1-l,ii) - r(l,ii) * rvw(kt+1-l,ii);
            tot_rvt += rv(l,ii) * rt(kt+1-l,ii) - r(l,ii) * rvt(kt+1-l,ii);
          }
          rvu(kt+1,ii) = tot_rvu / r(0,ii);
          rvv(kt+1,ii) = tot_rvv / r(0,ii);
          rvw(kt+1,ii) = tot_rvw / r(0,ii);
          rvt(kt+1,ii) = tot_rvt / r(0,ii);

          // Compute rt_gamma at the next time level
          real tot_rt_gamma = 0;
          for (int l=0; l<=kt; l++) {
            tot_rt_gamma += (kt+1._fp -l) * ( GAMMA*rt_gamma(l,ii)*rt(kt+1-l,ii) - rt(l,ii)*rt_gamma(kt+1-l,ii) );
          }
          rt_gamma(kt+1,ii) = ( GAMMA*rt_gamma(0,ii)*rt(kt+1,ii) + tot_rt_gamma / (kt+1._fp) ) / rt(0,ii);
        }
      }

      // Fix the rt_gamma
      for (int kt=1; kt < nAder; kt++) {
        for (int ii=0; ii < ngll; ii++) {
          rt_gamma(kt,ii) /= 2;
        }
      }
    }



    template <yakl::index_t nAder, yakl::index_t ngll>
    YAKL_INLINE static void diffTransformEulerConsZ( SArray<real,2,nAder,ngll> &r  ,
                                                     SArray<real,2,nAder,ngll> &ru ,
                                                     SArray<real,2,nAder,ngll> &rv ,
                                                     SArray<real,2,nAder,ngll> &rw ,
                                                     SArray<real,2,nAder,ngll> &rt ,
                                                     SArray<real,2,nAder,ngll> &rwu ,
                                                     SArray<real,2,nAder,ngll> &rwv ,
                                                     SArray<real,2,nAder,ngll> &rww ,
                                                     SArray<real,2,nAder,ngll> &rwt ,
                                                     SArray<real,2,nAder,ngll> &rt_gamma ,
                                                     SArray<real,2,ngll,ngll> const &deriv , 
                                                     real C0 ,
                                                     real GAMMA ,
                                                     real GRAV  ,
                                                     real dz ) {
      // zero out the non-linear DTs
      for (int kt=1; kt < nAder; kt++) {
        for (int ii=0; ii < ngll; ii++) {
          rwu     (kt,ii) = 0;
          rwv     (kt,ii) = 0;
          rww     (kt,ii) = 0;
          rwt     (kt,ii) = 0;
          rt_gamma(kt,ii) = 0;
        }
      }

      // Loop over the time derivatives
      for (int kt=0; kt<nAder-1; kt++) {
        // Compute the state at the next time level
        for (int ii=0; ii<ngll; ii++) {
          real drw_dz    = 0;
          real drwu_dz   = 0;
          real drwv_dz   = 0;
          real drww_p_dz = 0;
          real drwt_dz   = 0;
          for (int s=0; s<ngll; s++) {
            drw_dz    += deriv(s,ii) * rw(kt,s);
            drwu_dz   += deriv(s,ii) * rwu(kt,s);
            drwv_dz   += deriv(s,ii) * rwv(kt,s);
            if (kt == 0) { drww_p_dz += deriv(s,ii) * ( rww(kt,s) + C0*rt_gamma(kt,s)  ); }
            else         { drww_p_dz += deriv(s,ii) * ( rww(kt,s) + C0*rt_gamma(kt,s)/2); }
            drwt_dz   += deriv(s,ii) * rwt(kt,s);
          }
          r (kt+1,ii) = -drw_dz   /dz/(kt+1);
          ru(kt+1,ii) = -drwu_dz  /dz/(kt+1);
          rv(kt+1,ii) = -drwv_dz  /dz/(kt+1);
          rw(kt+1,ii) = -drww_p_dz/dz/(kt+1) -r(kt,ii)*GRAV/(kt+1);
          rt(kt+1,ii) = -drwt_dz  /dz/(kt+1);
        }

        // Compute ru* at the next time level
        for (int ii=0; ii<ngll; ii++) {
          // Compute the non-linear differential transforms
          real tot_rwu = 0;
          real tot_rwv = 0;
          real tot_rww = 0;
          real tot_rwt = 0;
          for (int l=0; l<=kt+1; l++) {
            tot_rwu += rw(l,ii) * ru(kt+1-l,ii) - r(l,ii) * rwu(kt+1-l,ii);
            tot_rwv += rw(l,ii) * rv(kt+1-l,ii) - r(l,ii) * rwv(kt+1-l,ii);
            tot_rww += rw(l,ii) * rw(kt+1-l,ii) - r(l,ii) * rww(kt+1-l,ii);
            tot_rwt += rw(l,ii) * rt(kt+1-l,ii) - r(l,ii) * rwt(kt+1-l,ii);
          }
          rwu(kt+1,ii) = tot_rwu / r(0,ii);
          rwv(kt+1,ii) = tot_rwv / r(0,ii);
          rww(kt+1,ii) = tot_rww / r(0,ii);
          rwt(kt+1,ii) = tot_rwt / r(0,ii);

          // Compute rt_gamma at the next time level
          real tot_rt_gamma = 0;
          for (int l=0; l<=kt; l++) {
            tot_rt_gamma += (kt+1-l) * ( GAMMA*rt_gamma(l,ii)*rt(kt+1-l,ii) - rt(l,ii)*rt_gamma(kt+1-l,ii) );
          }
          rt_gamma(kt+1,ii) = ( GAMMA*rt_gamma(0,ii)*rt(kt+1,ii) + tot_rt_gamma / (kt+1) ) / rt(0,ii);
        }
      }

      // Fix the rt_gamma
      for (int kt=1; kt < nAder; kt++) {
        for (int ii=0; ii < ngll; ii++) {
          rt_gamma(kt,ii) /= 2;
        }
      }
    }



    template <yakl::index_t nAder, yakl::index_t ngll>
    YAKL_INLINE static void diffTransformTracer( SArray<real,2,nAder,ngll> const &r  ,
                                                 SArray<real,2,nAder,ngll> const &ru ,
                                                 SArray<real,2,nAder,ngll> &rt ,
                                                 SArray<real,2,nAder,ngll> &rut ,
                                                 SArray<real,2,ngll,ngll> const &deriv , 
                                                 real dx ) {
      // zero out the non-linear DT
      for (int kt=1; kt < nAder; kt++) {
        for (int ii=0; ii < ngll; ii++) {
          rut(kt,ii) = 0;
        }
      }
      // Loop over the time derivatives
      for (int kt=0; kt<nAder-1; kt++) {
        // Compute the rho*tracer at the next time level
        for (int ii=0; ii<ngll; ii++) {
          real df_dx = 0;
          for (int s=0; s<ngll; s++) {
            df_dx += deriv(s,ii) * rut(kt,s);
          }
          rt(kt+1,ii) = -df_dx/dx/(kt+1._fp);
        }
        // Compute rut at the next time level
        for (int ii=0; ii<ngll; ii++) {
          real tot_rut = 0;
          for (int ir=0; ir<=kt+1; ir++) {
            tot_rut += ru(ir,ii) * rt(kt+1-ir,ii) - r(ir,ii) * rut(kt+1-ir,ii);
          }
          rut(kt+1,ii) = tot_rut / r(0,ii);
        }
      }
    }



    template <yakl::index_t nAder, yakl::index_t ngll>
    YAKL_INLINE static void compute_timeAvg( SArray<real,2,nAder,ngll> &dts , real dt ) {
      real dtmult = dt;
      for (int kt=1; kt<nAder; kt++) {
        for (int ii=0; ii<ngll; ii++) {
          dts(0,ii) += dts(kt,ii) * dtmult / (kt+1._fp);
        }
        dtmult *= dt;
      }
    }



    template <yakl::index_t nAder, yakl::index_t ngll>
    YAKL_INLINE static void compute_timeAvg( SArray<real,2,nAder,ngll> const &dts , SArray<real,1,ngll> &tavg , real dt ) {
      for (int ii=0; ii<ngll; ii++) {
        tavg(ii) = dts(0,ii);
      }
      real dtmult = dt;
      for (int kt=1; kt<nAder; kt++) {
        for (int ii=0; ii<ngll; ii++) {
          tavg(ii) += dts(kt,ii) * dtmult / (kt+1._fp);
        }
        dtmult *= dt;
      }
    }


  };

}

