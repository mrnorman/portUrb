
#include "dynamics_rk.h"

namespace modules {

  Dynamics_Euler_Stratified_WenoFV::Dynamics_Euler_Stratified_WenoFV() { }



  // Use CFL criterion to determine the time step. Currently hardwired
  real Dynamics_Euler_Stratified_WenoFV::compute_time_step( core::Coupler const &coupler ) const {
    auto dx = coupler.get_dx();
    auto dy = coupler.get_dy();
    auto dz = coupler.get_dz();
    real constexpr maxwave = 350 + 100;
    real cfl = 0.70;
    return cfl * std::min( std::min( dx , dy ) , dz ) / maxwave;
  }
  // real Dynamics_Euler_Stratified_WenoFV::compute_time_step( core::Coupler const &coupler ) const {
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
  //   real cfl = 0.70;
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
  void Dynamics_Euler_Stratified_WenoFV::time_step(core::Coupler &coupler, real dt_phys) const {
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
    convert_coupler_to_dynamics( coupler , state , tracers );
    real dt_dyn = compute_time_step( coupler );
    int ncycles = (int) std::ceil( dt_phys / dt_dyn );
    dt_dyn = dt_phys / ncycles;
    for (int icycle = 0; icycle < ncycles; icycle++) { time_step_rk_3_3(coupler,state,tracers,dt_dyn); }
    convert_dynamics_to_coupler( coupler , state , tracers );
    #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("time_step");
    #endif
  }



  // CFL 0.45 (Differs from paper, but this is the true value for this high-order FV scheme)
  // Third-order, three-stage SSPRK method
  // https://link.springer.com/content/pdf/10.1007/s10915-008-9239-z.pdf
  void Dynamics_Euler_Stratified_WenoFV::time_step_rk_3_3( core::Coupler & coupler ,
                                                           real4d const  & state   ,
                                                           real4d const  & tracers ,
                                                           real            dt_dyn  ) const {
    #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
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
    compute_tendencies(coupler,state,state_tend,tracers,tracers_tend,dt_dyn);
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
    //////////////
    // Stage 2
    //////////////
    compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,dt_dyn/4.);
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
    //////////////
    // Stage 3
    //////////////
    compute_tendencies(coupler,state_tmp,state_tend,tracers_tmp,tracers_tend,2.*dt_dyn/3.);
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
      yakl::timer_stop("time_step_rk_3_3");
    #endif
  }



  void Dynamics_Euler_Stratified_WenoFV::enforce_immersed_boundaries( core::Coupler const & coupler ,
                                                                      real4d        const & state   ,
                                                                      real4d        const & tracers ,
                                                                      real                  dt      ) const {
    #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
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
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("enforce_immersed_boundaries");
    #endif
  }



  void Dynamics_Euler_Stratified_WenoFV::halo_boundary_conditions( core::Coupler const & coupler  ,
                                                                   real4d        const & state    ,
                                                                   real4d        const & tracers  ,
                                                                   real3d        const & pressure ) const {
    #ifdef YAKL_AUTO_PROFILE
      coupler.get_parallel_comm().barrier();
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
        state(idT,hs+nz+kk,hs+j,hs+i) = hy_theta_cells(hs+nz+kk);
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
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("halo_boundary_conditions");
    #endif
  }



  // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
  void Dynamics_Euler_Stratified_WenoFV::convert_dynamics_to_coupler( core::Coupler &coupler ,
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
    auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
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
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("convert_dynamics_to_coupler");
    #endif
  }



  // Convert coupler's data to state and tracers arrays
  void Dynamics_Euler_Stratified_WenoFV::convert_coupler_to_dynamics( core::Coupler const &coupler ,
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
    auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
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
      coupler.get_parallel_comm().barrier();
      yakl::timer_stop("convert_coupler_to_dynamics");
    #endif
  }

}


