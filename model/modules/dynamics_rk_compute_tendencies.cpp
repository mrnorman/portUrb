
#include "dynamics_rk.h"

namespace modules {

  void Dynamics_Euler_Stratified_WenoFV::compute_tendencies( core::Coupler       & coupler      ,
                                                             real4d        const & state        ,
                                                             real4d        const & state_tend   ,
                                                             real4d        const & tracers      ,
                                                             real4d        const & tracers_tend ,
                                                             real                  dt           ) const {
    #ifdef YAKL_AUTO_PROFILE
      MPI_Barrier(MPI_COMM_WORLD);
      yakl::timer_start("compute_tendencies");
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
    auto tracer_positive   = dm.get<bool const,1>("tracer_positive"          ); // Is a tracer positive-definite?
    auto immersed_prop     = dm.get<real const,3>("dycore_immersed_proportion_halos"); // Immersed Proportion
    auto any_immersed      = dm.get<bool const,3>("dycore_any_immersed"      ); // Are any immersed in 3-D halo?
    auto hy_dens_cells     = dm.get<real const,1>("hy_dens_cells"            ); // Hydrostatic density
    auto hy_theta_cells    = dm.get<real const,1>("hy_theta_cells"           ); // Hydrostatic potential temperature
    auto hy_dens_edges     = dm.get<real const,1>("hy_dens_edges"            ); // Hydrostatic density
    auto hy_theta_edges    = dm.get<real const,1>("hy_theta_edges"           ); // Hydrostatic potential temperature
    auto hy_pressure_cells = dm.get<real const,1>("hy_pressure_cells"        ); // Hydrostatic pressure
    // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
    // These matrices will be in column-row format. That performed better than row-column format in performance tests
    real r_dx = 1./dx; // reciprocal of grid spacing
    real r_dy = 1./dy; // reciprocal of grid spacing
    real r_dz = 1./dz; // reciprocal of grid spacing
    real fcor = 2*7.2921e-5*std::sin(latitude/180*M_PI);      // For coriolis: 2*Omega*sin(latitude)

    // Compute pressure
    real3d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs); // Holds pressure perturbation
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      pressure(hs+k,hs+j,hs+i) = C0*std::pow(state(idT,hs+k,hs+j,hs+i),gamma) - hy_pressure_cells(hs+k);
      real r_r = 1._fp / state(idR,hs+k,hs+j,hs+i);
      for (int l=1; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) *= r_r; }
      for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) *= r_r; }
    });

    // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
    {
      core::MultiField<real,3> fields;
      for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<3>(l,0,0,0) ); }
      for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<3>(l,0,0,0) ); }
      fields.add_field( pressure );
      if (ord > 1) coupler.halo_exchange( fields , hs );
      halo_boundary_conditions( coupler , state , tracers , pressure );
    }

    typedef limiter::WenoLimiter<ord> Limiter;
    Limiter lim;

    // Create arrays to hold cell interface interpolations
    real5d state_limits_x   ("state_limits_x"   ,2,num_state  ,nz,ny,nx+1);
    real5d state_limits_y   ("state_limits_y"   ,2,num_state  ,nz,ny+1,nx);
    real5d state_limits_z   ("state_limits_z"   ,2,num_state  ,nz+1,ny,nx);
    real4d pressure_limits_x("pressure_limits_x",2            ,nz,ny,nx+1);
    real4d pressure_limits_y("pressure_limits_y",2            ,nz,ny+1,nx);
    real4d pressure_limits_z("pressure_limits_z",2            ,nz+1,ny,nx);
    real5d tracers_limits_x ("tracers_limits_x" ,2,num_tracers,nz,ny,nx+1);
    real5d tracers_limits_y ("tracers_limits_y" ,2,num_tracers,nz,ny+1,nx);
    real5d tracers_limits_z ("tracers_limits_z" ,2,num_tracers,nz+1,ny,nx);
    
    // Aggregate fields for interpolation
    core::MultiField<real,3> fields, lim_x_L, lim_x_R, lim_y_L, lim_y_R, lim_z_L, lim_z_R;
    int idP = 5;
    for (int l=0; l < num_state; l++) {
      fields .add_field(state         .slice<3>(  l,0,0,0));
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
      fields .add_field(tracers         .slice<3>(  l,0,0,0));
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
      SArray<real,1,ord> stencil;
      SArray<bool,1,ord> immersed;
      for (int ii=0; ii<ord; ii++) {
        immersed(ii) = immersed_prop(hs+k,hs+j,i+ii);
        stencil (ii) = fields     (l,hs+k,hs+j,i+ii);
      }
      if (l == idV || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
      bool map = true; // ! any_immersed(k,j,i);
      Limiter::compute_limited_edges( stencil , lim_x_R(l,k,j,i) , lim_x_L(l,k,j,i+1) , 
                                      { map , immersed(hs-1) , immersed(hs+1)} );
    });

    // Y-direction interpolation
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(fields.size(),nz,ny,nx) ,
                                      YAKL_LAMBDA (int l, int k, int j, int i) {
      SArray<real,1,ord> stencil;
      SArray<bool,1,ord> immersed;
      for (int jj=0; jj<ord; jj++) {
        immersed(jj) = immersed_prop(hs+k,j+jj,hs+i);
        stencil (jj) = fields     (l,hs+k,j+jj,hs+i);
      }
      if (l == idU || l == idW || l == idP) modify_stencil_immersed_der0( stencil , immersed );
      bool map = true; // ! any_immersed(k,j,i);
      Limiter::compute_limited_edges( stencil , lim_y_R(l,k,j,i) , lim_y_L(l,k,j+1,i) , 
                                      { map , immersed(hs-1) , immersed(hs+1)} );
    });

    // Z-direction interpolation
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(fields.size(),nz,ny,nx) ,
                                      YAKL_LAMBDA (int l, int k, int j, int i) {
      SArray<real,1,ord> stencil;
      SArray<bool,1,ord> immersed;
      for (int kk=0; kk<ord; kk++) {
        immersed(kk) = immersed_prop(k+kk,hs+j,hs+i);
        stencil (kk) = fields     (l,k+kk,hs+j,hs+i);
      }
      if (l == idU || l == idV || l == idP) modify_stencil_immersed_der0( stencil , immersed );
      bool map = true; // ! any_immersed(k,j,i);
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
    real constexpr cs   = 350.;
    real constexpr r_cs = 1./cs;
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz+1,ny+1,nx+1) , YAKL_LAMBDA (int k, int j, int i) {
      if (j < ny && k < nz) {
        // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
        real r_L  = state_limits_x  (0,idR,k,j,i)    ;   real r_R  = state_limits_x   (1,idR,k,j,i)    ;
        real ru_L = state_limits_x  (0,idU,k,j,i)*r_L;   real ru_R = state_limits_x   (1,idU,k,j,i)*r_R;
        real p_L  = pressure_limits_x(0   ,k,j,i)    ;   real p_R  = pressure_limits_x(1    ,k,j,i)    ;
        real w1 = 0.5_fp * (p_R-cs*ru_R);
        real w2 = 0.5_fp * (p_L+cs*ru_L);
        real p_upw  = w1 + w2;
        real ru_upw = (w2-w1)*r_cs;
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
        // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
        real r_L  = state_limits_y  (0,idR,k,j,i)    ;   real r_R  = state_limits_y   (1,idR,k,j,i)    ;
        real rv_L = state_limits_y  (0,idV,k,j,i)*r_L;   real rv_R = state_limits_y   (1,idV,k,j,i)*r_R;
        real p_L  = pressure_limits_y(0   ,k,j,i)    ;   real p_R  = pressure_limits_y(1    ,k,j,i)    ;
        real w1 = 0.5_fp * (p_R-cs*rv_R);
        real w2 = 0.5_fp * (p_L+cs*rv_L);
        real p_upw  = w1 + w2;
        real rv_upw = (w2-w1)*r_cs;
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
        // Acoustically upwind mass flux and pressure, linear constant Jacobian diagonalization
        real r_L  = state_limits_z  (0,idR,k,j,i)    ;   real r_R  = state_limits_z   (1,idR,k,j,i)    ;
        real rw_L = state_limits_z  (0,idW,k,j,i)*r_L;   real rw_R = state_limits_z   (1,idW,k,j,i)*r_R;
        real p_L  = pressure_limits_z(0   ,k,j,i)    ;   real p_R  = pressure_limits_z(1    ,k,j,i)    ;
        real w1 = 0.5_fp * (p_R-cs*rw_R);
        real w2 = 0.5_fp * (p_L+cs*rw_L);
        real p_upw  = w1 + w2;
        real rw_upw = (w2-w1)*r_cs;
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
      if (i < nx && j < ny && k < nz) {
        for (int l=1; l < num_state  ; l++) { state  (l,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i); }
        for (int l=0; l < num_tracers; l++) { tracers(l,hs+k,hs+j,hs+i) *= state(idR,hs+k,hs+j,hs+i); }
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
      MPI_Barrier(MPI_COMM_WORLD);
      yakl::timer_stop("compute_tendencies");
    #endif
  }

}


