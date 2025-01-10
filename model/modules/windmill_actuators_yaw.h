
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "Betti_simplified.h"

namespace modules {

  // Uses disk actuators to represent wind turbines in an LES model by applying friction terms to horizontal
  //   velocities and adding a portion of the thrust not generating power to TKE.
  struct WindmillActuators {


    // Stores information needed to imprint a turbine actuator disk onto the grid.
    struct RefTurbine {
      // Reference wind turbine (RWT) tables
      realHost1d velmag_host;      // Velocity magnitude at infinity (m/s)
      realHost1d thrust_coef_host; // Thrust coefficient             (dimensionless)
      realHost1d power_coef_host;  // Power coefficient              (dimensionless)
      realHost1d power_host;       // Power generation               (MW)
      realHost1d rotation_host;    // Rotation speed                 (radians / sec)
      // Turbine properties
      real       hub_height;       // Hub height                     (m)
      real       blade_radius;     // Blade radius                   (m)
      real       max_yaw_speed;    // Angular active yawing speed    (radians / sec)
      real       base_diameter;    // Average diameter of the base   (m)
      void init( std::string fname , real dx , real dy , real dz ) {
        YAML::Node config = YAML::LoadFile( fname );
        if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
        auto velmag_vec      = config["velocity_magnitude"].as<std::vector<real>>();
        auto thrust_coef_vec = config["thrust_coef"       ].as<std::vector<real>>();
        auto power_coef_vec  = config["power_coef"        ].as<std::vector<real>>();
        auto power_vec       = config["power_megawatts"   ].as<std::vector<real>>();
        bool do_blades       = false;
        if ( config["rotation_rpm"] ) do_blades = true;
        auto rotation_vec = do_blades ? config["rotation_rpm"].as<std::vector<real>>() : std::vector<real>();
        // Allocate YAKL arrays to ensure the data is contiguous and to load into the data manager later
        velmag_host      = realHost1d("velmag"     ,velmag_vec     .size());
        thrust_coef_host = realHost1d("thrust_coef",thrust_coef_vec.size());
        power_coef_host  = realHost1d("power_coef" ,power_coef_vec .size());
        power_host       = realHost1d("power"      ,power_vec      .size());
        if (do_blades) rotation_host = realHost1d("rotation",rotation_vec.size());
        // Make sure the sizes match
        if ( velmag_host.size() != thrust_coef_host.size() ||
             velmag_host.size() != power_coef_host .size() ||
             velmag_host.size() != power_host      .size() ||
             (do_blades && (velmag_host.size() != rotation_host.size())) ) {
          Kokkos::abort("ERROR: turbine arrays not all the same size");
        }
        // Move from std::vectors into YAKL arrays
        for (int i=0; i < velmag_host.size(); i++) {
          velmag_host     (i) = velmag_vec     .at(i);
          thrust_coef_host(i) = thrust_coef_vec.at(i);
          power_coef_host (i) = power_coef_vec .at(i);
          power_host      (i) = power_vec      .at(i);
          if (do_blades) rotation_host(i) = rotation_vec.at(i)*2*M_PI/60; // Convert from rpm to radians/sec
        }
        // Copy from host to device and set other parameters
        this->hub_height    = config["hub_height"   ].as<real>();
        this->blade_radius  = config["blade_radius" ].as<real>();
        this->max_yaw_speed = config["max_yaw_speed"].as<real>(0.5)/180.*M_PI; // Convert from deg/sec to rad/sec
        this->base_diameter = config["base_diameter"].as<real>(9);
      }
    };


    // Yaw will change as if it were an active yaw system that moves at a certain max speed. It will react
    //   to some time average of the wind velocities. The operator() outputs the new yaw angle in radians.
    struct YawTend {
      real tau, uavg, vavg;
      YawTend( real tau_in=60 , real uavg_in=0, real vavg_in=0 ) { tau=tau_in; uavg=uavg_in; vavg=vavg_in; }
      real operator() ( real uvel , real vvel , real dt , real yaw , real max_yaw_speed ) {
        // Update the moving average by weighting according using time scale as inertia
        uavg = (tau-dt)/tau*uavg + dt/tau*uvel;
        vavg = (tau-dt)/tau*vavg + dt/tau*vvel;
        // atan2 gives [-pi,pi] with zero representing moving toward the east
        // But we're using a coordinate system rotated by pi such that zero faces west.
        // That is, we're using an "upwind" coordinate system
        real dir_upwind = std::atan2(vavg,uavg);
        // Compute difference between time-averaged upwind direction and current yaw
        real diff = dir_upwind - yaw;
        if (diff >  M_PI) diff -= 2*M_PI;
        if (diff < -M_PI) diff += 2*M_PI;
        // Limit to the max yaw speed of the turbine
        real tend = diff / dt;
        if (tend > 0) { tend = std::min(  max_yaw_speed , tend ); }
        else          { tend = std::max( -max_yaw_speed , tend ); }
        // Return the new yaw angle
        return yaw+dt*tend;
      }
    };


    // Holds information about a turbine (location, reference_type, yaw, etc)
    struct Turbine {
      bool                    active;            // Whether this turbine affects this MPI task
      real                    base_loc_x;        // x location of the tower base
      real                    base_loc_y;        // y location of the tower base
      std::vector<real>       power_trace;       // Time trace of power generation
      std::vector<real>       yaw_trace;         // Time trace of yaw of the turbine
      std::vector<real>       u_samp_trace;      // Time trace of disk-integrated inflow u velocity
      std::vector<real>       v_samp_trace;      // Time trace of disk-integrated inflow v velocity
      std::vector<real>       mag195_trace;      // Time trace of disk-integrated 19.5m infoat velocity
      std::vector<real>       betti_trace;       // Time trace of floating motions perturbations
      std::vector<real>       cp_trace;          // Time trace of coefficient of power
      std::vector<real>       ct_trace;          // Time trace of coefficient of thrust
      real                    u_samp_inertial;   // Intertial inflow u-velocity normal to the turbine plane
      real                    v_samp_inertial;   // Intertial inflow u-velocity normal to the turbine plane
      real                    yaw_angle;         // Current yaw angle (radians counter-clockwise from facing west)
      real                    rot_angle;         // Current rotation angle (radians)
      YawTend                 yaw_tend;          // Functor to compute the change in yaw
      RefTurbine              ref_turbine;       // The reference turbine to use for this turbine
      core::ParallelComm      par_comm;          // MPI communicator for this turbine
      int                     nranks;            // Number of MPI ranks involved with this turbine
      int                     sub_rankid;        // My process's rank ID in the sub communicator
      int                     owning_sub_rankid; // Subcommunicator rank ID of the owner of this turbine
      bool                    apply_thrust;      // Whether to apply the thrust to the simulation or not
      Floating_motions_betti  floating_motions;  // Class to handle floating motions due to waves, thrust, etc
    };


    struct TurbineGroup {
      std::vector<Turbine> turbines;
      void add_turbine( core::Coupler       & coupler     ,
                        real                  base_loc_x  ,
                        real                  base_loc_y  ,
                        RefTurbine    const & ref_turbine ,
                        bool                  apply_thrust = true ) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;
        auto i_beg  = coupler.get_i_beg();
        auto j_beg  = coupler.get_j_beg();
        auto nx     = coupler.get_nx();
        auto ny     = coupler.get_ny();
        auto nz     = coupler.get_nz();
        auto dx     = coupler.get_dx();
        auto dy     = coupler.get_dy();
        auto dz     = coupler.get_dz();
        auto myrank = coupler.get_myrank();
        auto imm    = coupler.get_data_manager_readwrite().get<real,3>("immersed_proportion");
        // bounds of this MPI task's domain
        real dom_x1  = (i_beg+0 )*dx;
        real dom_x2  = (i_beg+nx)*dx;
        real dom_y1  = (j_beg+0 )*dy;
        real dom_y2  = (j_beg+ny)*dy;
        // Rectangular bounds of this turbine's potential influence
        real turb_x1 = base_loc_x-6*ref_turbine.blade_radius-6*std::sqrt(dx*dy);
        real turb_x2 = base_loc_x+6*ref_turbine.blade_radius+6*std::sqrt(dx*dy);
        real turb_y1 = base_loc_y-6*ref_turbine.blade_radius-6*std::sqrt(dx*dy);
        real turb_y2 = base_loc_y+6*ref_turbine.blade_radius+6*std::sqrt(dx*dy);
        // Determine if the two domains overlap
        bool active = !( turb_x1 > dom_x2 || // Turbine's to the right
                         turb_x2 < dom_x1 || // Turbine's to the left
                         turb_y1 > dom_y2 || // Turbine's above
                         turb_y2 < dom_y1 ); // Turbine's below
        std::random_device rd{};
        Turbine loc;
        loc.active          = active;
        loc.base_loc_x      = base_loc_x;
        loc.base_loc_y      = base_loc_y;
        loc.yaw_angle       = coupler.get_option<real>("turbine_initial_yaw",0);
        loc.rot_angle       = 0.;
        loc.yaw_tend        = YawTend();
        loc.ref_turbine     = ref_turbine;
        loc.u_samp_inertial = 0;
        loc.v_samp_inertial = 0;
        loc.apply_thrust    = apply_thrust;
        // loc.floating_motions.init("./inputs/Betti_NREL_5MW.nc");
        loc.floating_motions.init();
        loc.par_comm.create( active , coupler.get_parallel_comm().get_mpi_comm() );
        if (active) {
          // Get subcommunicator size and rank id
          loc.nranks     = loc.par_comm.get_size();
          loc.sub_rankid = loc.par_comm.get_rank_id();
          // Determine if I "own" the turbine (if the hub's in my domain)
          bool owner = base_loc_x >= i_beg*dx && base_loc_x < (i_beg+nx)*dx &&
                       base_loc_y >= j_beg*dy && base_loc_y < (j_beg+ny)*dy ;
          // Gather who owns the turbine, so yaw angles can be broadcast later
          if ( loc.nranks == 1) {
            loc.owning_sub_rankid = 0;
          } else {
            bool owner = base_loc_x >= i_beg*dx && base_loc_x < (i_beg+nx)*dx &&
                         base_loc_y >= j_beg*dy && base_loc_y < (j_beg+ny)*dy ;
            auto owner_arr = loc.par_comm.all_gather( owner );
            for (int i=0; i < loc.nranks; i++) { if (owner_arr(i)) loc.owning_sub_rankid = i; }
          }
        } else {
          // Don't want comparisons to give true for any of these
          loc.nranks = -1;
          loc.sub_rankid = -2;
          loc.owning_sub_rankid = -3;
        }
        // Add the turbine
        turbines.push_back(loc);
        // // Add the base to immersed_proportion
        // int N = 10;
        // parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        //   int count = 0;
        //   for (int kk=0; kk < N; kk++) {
        //     for (int jj=0; jj < N; jj++) {
        //       for (int ii=0; ii < N; ii++) {
        //         int x = (i_beg+i)*dx + ii*dx/(N-1);
        //         int y = (j_beg+j)*dy + jj*dy/(N-1);
        //         int z = (      k)*dz + kk*dz/(N-1);
        //         auto bx  = base_loc_x;
        //         auto by  = base_loc_y;
        //         auto rad = ref_turbine.base_diameter/2;
        //         auto h   = ref_turbine.hub_height;
        //         if ( (x-bx)*(x-bx) + (y-by)*(y-by) <= rad*rad  && z <= h ) count++;
        //       }
        //     }
        //   }
        //   // Express the base as an immersed boundary
        //   imm(k,j,i) += static_cast<real>(count)/(N*N*N);
        // });
      }
    };


    // Sagemath code producing the function used in DefaultThrustShape
    // def c_scalar(val,coeflab) :
    //     import re
    //     s = str(val).replace(' ','')
    //     s = re.sub("([a-zA-Z0-9_]*)\\^2","(\\1*\\1)",s,0,re.DOTALL)
    //     s = re.sub("([a-zA-Z0-9_]*)\\^3","(\\1*\\1*\\1)",s,0,re.DOTALL)
    //     return s
    // def coefs_1d(N,N0,lab) :
    //     return vector([ var(lab+'%s'%i) for i in range(N0,N0+N) ])
    // def poly_1d(N,coefs) :
    //     return sum( vector([ coefs[i]*x^i for i in range(N) ]) )
    // var('x2,x3,a')
    // coefs = coefs_1d(3,0,'a')
    // p = poly_1d(3,coefs)
    // constr = vector([p.subs(x=0),p.subs(x=x2),p.diff(x).subs(x=x2)])
    // p1 = poly_1d(3,(jacobian(constr,coefs)^-1)*vector([0,1,0]))
    // coefs = coefs_1d(4,0,'a')
    // p = poly_1d(4,coefs)
    // constr = vector([p.subs(x=x2),p.diff(x).subs(x=x2),p.subs(x=x3),p.diff(x).subs(x=x3)])
    // p2 = poly_1d(4,(jacobian(constr,coefs)^-1)*vector([1,0,0,0]))
    // print("p1 = pow(",c_scalar(p1.simplify_full(),'none'),", a );")
    // print("p2 = ",c_scalar(p2.simplify_full(),'none'),";")
    // x2 = 0.9;    x3 = 1;    a = 0.5
    // ( plot(p1.subs(x2=x2)^a,x,0 ,x2) + plot(p2.subs(x2=x2,x3=x3),x,x2,x3) ).show()
    struct DefaultThrustShape {
      KOKKOS_INLINE_FUNCTION real operator() ( real r , real x2 = 0.9 , real x3 = 1.0 , real a = 0.5 ) const {
        using std::pow;
        real x = r;
        if (r < x2) return pow(-1.0*((x*x)-2*x*x2)/(x2*x2),a);
        if (r < x3) return -1.0*(2*(x*x*x)-3*(x*x)*x2-3*x2*(x3*x3)+(x3*x3*x3)-3*((x*x)-2*x*x2)*x3)/((x2*x2*x2)-3*(x2*x2)*x3+3*x2*(x3*x3)-(x3*x3*x3));
        return 0;
      }
    };


    struct DefaultProjectionShape1D {
      KOKKOS_INLINE_FUNCTION real operator() ( real x , real xr , int p = 2 ) const {
        real term = 1-(x/xr)*(x/xr);
        if (term <= 0) return 0;
        real term_p = term;
        for (int i = 0; i < p-1; i++) { term_p *= term; }
        return term_p;
      }
    };


    struct DefaultProjectionShape2D {
      KOKKOS_INLINE_FUNCTION real operator() ( real x , real y , real xr , real yr , int p = 2 ) const {
        real term = 1-(x/xr)*(x/xr)-(y/yr)*(y/yr);
        if (term <= 0) return 0;
        real term_p = term;
        for (int i = 0; i < p-1; i++) { term_p *= term; }
        return term_p;
      }
    };


    // Class data members
    TurbineGroup  turbine_group;
    int           trace_size;
    int           sample_counter;


    void init( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx      = coupler.get_nx  ();
      auto ny      = coupler.get_ny  ();
      auto nz      = coupler.get_nz  ();
      auto dx      = coupler.get_dx  ();
      auto dy      = coupler.get_dy  ();
      auto dz      = coupler.get_dz  ();
      auto xlen    = coupler.get_xlen();
      auto ylen    = coupler.get_ylen();
      auto i_beg   = coupler.get_i_beg();
      auto j_beg   = coupler.get_j_beg();
      auto nx_glob = coupler.get_nx_glob();
      auto ny_glob = coupler.get_ny_glob();
      auto myrank  = coupler.get_myrank();
      auto &dm     = coupler.get_data_manager_readwrite();

      trace_size = 0;
      
      RefTurbine ref_turbine;
      ref_turbine.init( coupler.get_option<std::string>("turbine_file") , dx , dy , dz );

      int num_x = (int) std::round( xlen / 10 / (2*ref_turbine.blade_radius) );
      int num_y = (int) std::round( ylen / 10 / (2*ref_turbine.blade_radius) );
      real xinc = xlen/num_x;
      real yinc = ylen/num_y;
      // Determine the x and y bounds of this MPI task's domain
      if (coupler.option_exists("turbine_x_locs") && coupler.option_exists("turbine_y_locs")) {
        auto x_locs = coupler.get_option<std::vector<real>>("turbine_x_locs");
        auto y_locs = coupler.get_option<std::vector<real>>("turbine_y_locs");
        std::vector<bool> apply_thrust;
        apply_thrust.assign(x_locs.size(),true);
        if (coupler.option_exists("turbine_apply_thrust")) {
          apply_thrust = coupler.get_option<std::vector<bool>>("turbine_apply_thrust");
        }
        for (int iturb = 0; iturb < x_locs.size(); iturb++) {
          turbine_group.add_turbine( coupler , x_locs.at(iturb) , y_locs.at(iturb) , ref_turbine ,
                                     apply_thrust.at(iturb) );
        }
      } else {
        for (real y = yinc/2; y < ylen; y += yinc) {
          for (real x = xinc/2; x < xlen; x += xinc) {
            turbine_group.add_turbine( coupler , x , y , ref_turbine );
          }
        }
      }

      dm.register_and_allocate<real>("windmill_proj_weight","",{nz,ny,nx});
      coupler.register_output_variable<real>( "windmill_proj_weight" , core::Coupler::DIMS_3D );
      dm.register_and_allocate<real>("windmill_samp_weight","",{nz,ny,nx});
      coupler.register_output_variable<real>( "windmill_samp_weight" , core::Coupler::DIMS_3D );
      // Create an output module in the coupler to dump the windmill portions and the power trace from task zero
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        if (trace_size > 0) {
          nc.redef();
          nc.create_dim( "num_time_steps" , trace_size );
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            std::string pow_vname    = std::string("power_trace_turb_" ) + std::to_string(iturb);
            std::string yaw_vname    = std::string("yaw_trace_turb_"   ) + std::to_string(iturb);
            std::string u_samp_vname = std::string("u_samp_trace_turb_") + std::to_string(iturb);
            std::string v_samp_vname = std::string("v_samp_trace_turb_") + std::to_string(iturb);
            std::string mag195_vname = std::string("mag195_trace_turb_") + std::to_string(iturb);
            std::string betti_vname  = std::string("betti_trace_turb_" ) + std::to_string(iturb);
            std::string cp_vname     = std::string("cp_trace_turb_"    ) + std::to_string(iturb);
            std::string ct_vname     = std::string("ct_trace_turb_"    ) + std::to_string(iturb);
            nc.create_var<real>( pow_vname    , {"num_time_steps"} );
            nc.create_var<real>( yaw_vname    , {"num_time_steps"} );
            nc.create_var<real>( u_samp_vname , {"num_time_steps"} );
            nc.create_var<real>( v_samp_vname , {"num_time_steps"} );
            nc.create_var<real>( mag195_vname , {"num_time_steps"} );
            nc.create_var<real>( betti_vname  , {"num_time_steps"} );
            nc.create_var<real>( cp_vname     , {"num_time_steps"} );
            nc.create_var<real>( ct_vname     , {"num_time_steps"} );
          }
          nc.enddef();
          nc.begin_indep_data();
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            auto &turbine = turbine_group.turbines.at(iturb);
            if (turbine.active && turbine.sub_rankid == turbine.owning_sub_rankid) {
              realHost1d power_arr ("power_arr" ,trace_size);
              realHost1d yaw_arr   ("yaw_arr"   ,trace_size);
              realHost1d u_samp_arr("u_samp_arr",trace_size);
              realHost1d v_samp_arr("v_samp_arr",trace_size);
              realHost1d mag195_arr("mag195_arr",trace_size);
              realHost1d betti_arr ("betti_arr" ,trace_size);
              realHost1d cp_arr    ("cp_arr"    ,trace_size);
              realHost1d ct_arr    ("ct_arr"    ,trace_size);
              for (int i=0; i < trace_size; i++) { power_arr (i) = turbine.power_trace .at(i); }
              for (int i=0; i < trace_size; i++) { yaw_arr   (i) = turbine.yaw_trace   .at(i)/M_PI*180; }
              for (int i=0; i < trace_size; i++) { u_samp_arr(i) = turbine.u_samp_trace.at(i); }
              for (int i=0; i < trace_size; i++) { v_samp_arr(i) = turbine.v_samp_trace.at(i); }
              for (int i=0; i < trace_size; i++) { mag195_arr(i) = turbine.mag195_trace.at(i); }
              for (int i=0; i < trace_size; i++) { betti_arr (i) = turbine.betti_trace .at(i); }
              for (int i=0; i < trace_size; i++) { cp_arr    (i) = turbine.cp_trace    .at(i); }
              for (int i=0; i < trace_size; i++) { ct_arr    (i) = turbine.ct_trace    .at(i); }
              std::string pow_vname    = std::string("power_trace_turb_" ) + std::to_string(iturb);
              std::string yaw_vname    = std::string("yaw_trace_turb_"   ) + std::to_string(iturb);
              std::string u_samp_vname = std::string("u_samp_trace_turb_") + std::to_string(iturb);
              std::string v_samp_vname = std::string("v_samp_trace_turb_") + std::to_string(iturb);
              std::string mag195_vname = std::string("mag195_trace_turb_") + std::to_string(iturb);
              std::string betti_vname  = std::string("betti_trace_turb_" ) + std::to_string(iturb);
              std::string cp_vname     = std::string("cp_trace_turb_"    ) + std::to_string(iturb);
              std::string ct_vname     = std::string("ct_trace_turb_"    ) + std::to_string(iturb);
              nc.write( power_arr  , pow_vname    );
              nc.write( yaw_arr    , yaw_vname    );
              nc.write( u_samp_arr , u_samp_vname );
              nc.write( v_samp_arr , v_samp_vname );
              nc.write( mag195_arr , mag195_vname );
              nc.write( betti_arr  , betti_vname  );
              nc.write( cp_arr     , cp_vname     );
              nc.write( ct_arr     , ct_vname     );
            }
            coupler.get_parallel_comm().barrier();
            turbine.power_trace   .clear();
            turbine.yaw_trace     .clear();
            turbine.u_samp_trace  .clear();
            turbine.v_samp_trace  .clear();
            turbine.mag195_trace  .clear();
            turbine.betti_trace   .clear();
            turbine.cp_trace      .clear();
            turbine.ct_trace      .clear();
          }
          nc.end_indep_data();
        }
        trace_size = 0;
      });
      sample_counter = 0;
    }


    void apply( core::Coupler & coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx              = coupler.get_nx   ();
      auto ny              = coupler.get_ny   ();
      auto nz              = coupler.get_nz   ();
      auto dx              = coupler.get_dx   ();
      auto dy              = coupler.get_dy   ();
      auto dz              = coupler.get_dz   ();
      auto zlen            = coupler.get_zlen();
      auto i_beg           = coupler.get_i_beg();
      auto j_beg           = coupler.get_j_beg();
      auto myrank          = coupler.get_myrank();
      auto &dm             = coupler.get_data_manager_readwrite();
      auto rho_d           = dm.get<real const,3>("density_dry"  );
      auto uvel            = dm.get<real      ,3>("uvel"         );
      auto vvel            = dm.get<real      ,3>("vvel"         );
      auto tke             = dm.get<real      ,3>("TKE"          );
      auto proj_weight_tot = dm.get<real      ,3>("windmill_proj_weight");
      auto samp_weight_tot = dm.get<real      ,3>("windmill_samp_weight");
      auto thrust_shape = DefaultThrustShape();
      auto proj_shape_1d = DefaultProjectionShape1D();
      auto proj_shape_2d = DefaultProjectionShape2D();

      real3d tend_u  ("tend_u"  ,nz,ny,nx);
      real3d tend_v  ("tend_v"  ,nz,ny,nx);
      real3d tend_tke("tend_tke",nz,ny,nx);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        tend_u         (k,j,i) = 0;
        tend_v         (k,j,i) = 0;
        tend_tke       (k,j,i) = 0;
        proj_weight_tot(k,j,i) = 0;
        samp_weight_tot(k,j,i) = 0;
      });

      for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++) {
        auto &turbine = turbine_group.turbines.at(iturb);
        if (turbine.active) {
          ///////////////////////////////////////////////////
          // Sampling of turbine disk
          ///////////////////////////////////////////////////
          // Pre-compute rotation matrix terms
          real cos_yaw = std::cos(turbine.yaw_angle);
          real sin_yaw = std::sin(turbine.yaw_angle);
          // These are the global extents of this MPI task's domain
          real dom_x1 = (i_beg+0 )*dx;
          real dom_x2 = (i_beg+nx)*dx;
          real dom_y1 = (j_beg+0 )*dy;
          real dom_y2 = (j_beg+ny)*dy;
          // Use monte carlo to compute proportion of the turbine in each cell
          // Get reference data for later computations
          real rad             = turbine.ref_turbine.blade_radius    ; // Radius of the blade plane
          real hub_height      = turbine.ref_turbine.hub_height      ; // height of the hub
          real base_x          = turbine.base_loc_x;
          real base_y          = turbine.base_loc_y;
          auto ref_velmag      = turbine.ref_turbine.velmag_host     ; // For interpolation
          auto ref_thrust_coef = turbine.ref_turbine.thrust_coef_host; // For interpolation
          auto ref_power_coef  = turbine.ref_turbine.power_coef_host ; // For interpolation
          auto ref_power       = turbine.ref_turbine.power_host      ; // For interpolation
          auto ref_rotation    = turbine.ref_turbine.rotation_host   ; // For interpolation
          bool do_blades       = ref_rotation.initialized();
          // Zero out disk weights for projection and sampling
          // Compute 19.5m horizontal wind magnitude for floating platform motions
          // Compute wind direction and offset for upstream sampling to get freestream velocities
          real3d disk_weight_proj("disk_weight_proj",nz,ny,nx);
          real3d disk_weight_samp("disk_weight_samp",nz,ny,nx);
          real3d uvel_3d         ("uvel_3d"         ,nz,ny,nx);
          real3d vvel_3d         ("vvel_3d"         ,nz,ny,nx);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
            disk_weight_proj(k,j,i) = 0;
            disk_weight_samp(k,j,i) = 0;
            real x = (i_beg+i+0.5_fp)*dx;
            real y = (j_beg+j+0.5_fp)*dy;
            real z = (      k+0.5_fp)*dz;
            if ( z >= hub_height-rad && z <= hub_height+rad &&
                 y >= base_y    -rad && y <= base_y    +rad &&
                 x >= base_x    -rad && x <= base_x    +rad ) {
              uvel_3d(k,j,i) = uvel(k,j,i);
              vvel_3d(k,j,i) = vvel(k,j,i);
            } else {
              uvel_3d(k,j,i) = 0;
              vvel_3d(k,j,i) = 0;
            }
          });
          yakl::SArray<real,1,2> weights_tot;
          weights_tot(0) = yakl::intrinsics::sum(uvel_3d);
          weights_tot(1) = yakl::intrinsics::sum(vvel_3d);
          weights_tot = turbine.par_comm.all_reduce( weights_tot , MPI_SUM , "windmill_Allreduce1" );
          real upstream_uvel = weights_tot(0);
          real upstream_vvel = weights_tot(1);
          real upstream_dir;
          if (coupler.option_exists("turbine_upstream_dir")) {
            upstream_dir = coupler.get_option<real>("turbine_upstream_dir");
          } else {
            upstream_dir = std::atan2( upstream_vvel , upstream_uvel );  // theta=tan^-1(v/u)
          }
          real upstream_x_offset = -5*rad*std::cos(upstream_dir);
          real upstream_y_offset = -5*rad*std::sin(upstream_dir);
          // Compute and sum weights for disk projection and upstream sampling projection
          real2d umag_19_5m_2d("umag_19_5m_2d",ny,nx);
          {
            real xr = 5*dx;
            int num_x = std::round(xr*2);
            int num_y = std::round(rad*2);
            int num_z = std::round(rad*2);
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_z,num_y,num_x) , KOKKOS_LAMBDA (int k, int j, int i) {
              real x = -xr  + (2*xr *i)/(num_x-1);
              real y = -rad + (2*rad*j)/(num_y-1);
              real z = -rad + (2*rad*k)/(num_z-1);
              real rloc = std::sqrt(y*y+z*z);
              if (rloc <= rad) {
                real proj1d = proj_shape_1d(x,xr);
                // Now rotate x and y according to the yaw angle, and translate to base location
                real xp = base_x     + cos_yaw*x - sin_yaw*y;
                real yp = base_y     + sin_yaw*x + cos_yaw*y;
                real zp = hub_height + z;
                // if it's in this task's domain, then increment the appropriate cell count atomically
                int ti = static_cast<int>(std::round(xp/dx-0.5-i_beg));
                int tj = static_cast<int>(std::round(yp/dy-0.5-j_beg));
                int tk = static_cast<int>(std::round(zp/dz-0.5      ));
                if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
                  Kokkos::atomic_add( &disk_weight_proj(tk,tj,ti) , thrust_shape(rloc/rad)*proj1d );
                }
                xp += upstream_x_offset;
                yp += upstream_y_offset;
                ti = static_cast<int>(std::round(xp/dx-0.5-i_beg));
                tj = static_cast<int>(std::round(yp/dy-0.5-j_beg));
                tk = static_cast<int>(std::round(zp/dz-0.5      ));
                if ( ti >= 0 && ti < nx && tj >= 0 && tj < ny && tk >= 0 && tk < nz) {
                  Kokkos::atomic_add( &disk_weight_samp(tk,tj,ti) , thrust_shape(rloc/rad)*proj1d );
                }
              }
            });
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(ny,nx) , KOKKOS_LAMBDA (int j, int i) {
              real x = (i_beg+i+0.5_fp)*dx;
              real y = (j_beg+j+0.5_fp)*dy;
              if (std::abs(x-(base_x+upstream_x_offset)) <= rad && std::abs(y-(base_y+upstream_y_offset)) <= rad) {
                int k19_5 = std::max( 0._fp , std::round(19.5/dz-0.5) );
                real u = uvel(k19_5,j,i);
                real v = vvel(k19_5,j,i);
                umag_19_5m_2d(j,i) = std::sqrt(u*u + v*v);
              } else {
                umag_19_5m_2d(j,i) = 0;
              }
            });
          }
          using yakl::componentwise::operator>;
          yakl::SArray<real,1,4> weights_tot2;
          weights_tot2(0) = yakl::intrinsics::sum(umag_19_5m_2d);
          weights_tot2(1) = (real) yakl::intrinsics::count(umag_19_5m_2d > 0._fp);
          weights_tot2(2) = yakl::intrinsics::sum(disk_weight_proj);
          weights_tot2(3) = yakl::intrinsics::sum(disk_weight_samp);
          weights_tot2 = turbine.par_comm.all_reduce( weights_tot2 , MPI_SUM , "windmill_Allreduce1" );
          real umag_19_5m    = weights_tot2(0) / weights_tot2(1);
          real disk_proj_tot = weights_tot2(2);
          real disk_samp_tot = weights_tot2(3);
          turbine.mag195_trace.push_back( umag_19_5m );
          ///////////////////////////////////////////////////
          // Aggregation of disk integrals
          ///////////////////////////////////////////////////
          // Normalize disk weights for projection and upstream sampling
          // Aggregate disk-averaged wind velocities in upstream sampling region
          real3d samp_u("samp_u",nz,ny,nx);
          real3d samp_v("samp_v",nz,ny,nx);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
            if (disk_weight_proj(k,j,i) > 0) {
              disk_weight_proj(k,j,i) /= disk_proj_tot;
              proj_weight_tot (k,j,i) += disk_weight_proj(k,j,i);
            }
            if (disk_weight_samp(k,j,i) > 0) {
              disk_weight_samp(k,j,i) /= disk_samp_tot;
              samp_weight_tot (k,j,i) += disk_weight_samp(k,j,i);
              samp_u          (k,j,i)  = disk_weight_samp(k,j,i)*uvel(k,j,i);
              samp_v          (k,j,i)  = disk_weight_samp(k,j,i)*vvel(k,j,i);
            } else {
              samp_u          (k,j,i) = 0;
              samp_v          (k,j,i) = 0;
            }
          });
          SArray<real,1,2> sums;
          sums(0) = yakl::intrinsics::sum( samp_u );
          sums(1) = yakl::intrinsics::sum( samp_v );
          sums = turbine.par_comm.all_reduce( sums , MPI_SUM , "windmill_Allreduce2" );
          turbine.u_samp_trace.push_back( sums(0) );
          turbine.v_samp_trace.push_back( sums(1) );
          // Compute instantaneous wind magnitude
          real instant_u0   = sums(0)*cos_yaw;  // instantaneous u-velocity normal to the turbine plane
          real instant_v0   = sums(1)*sin_yaw;  // instantaneous v-velocity normal to the turbine plane
          real instant_mag0 = std::max( 0._fp , instant_u0 + instant_v0 );
          // Compute inertial wind magnitude
          real inertial_u0   = turbine.u_samp_inertial;  // inertial u-velocity normal to the turbine plane
          real inertial_v0   = turbine.v_samp_inertial;  // inertial v-velocity normal to the turbine plane
          real inertial_mag0 = std::max( 0._fp , inertial_u0 + inertial_v0 );
          ///////////////////////////////////////////////////
          // Computation of disk properties
          ///////////////////////////////////////////////////
          real C_T       = interp( ref_velmag , ref_thrust_coef , inertial_mag0 ); // Interpolate thrust coefficient
          real C_P       = interp( ref_velmag , ref_power_coef  , inertial_mag0 ); // Interpolate power coefficient
          real pwr       = interp( ref_velmag , ref_power       , inertial_mag0 ); // Interpolate power generation
          real rot_speed = do_blades ? interp( ref_velmag , ref_rotation , inertial_mag0 ) : 0; // Interpolate rot speed
          if (inertial_mag0 > 1.e-10) {
            if ( ! coupler.get_option<bool>("turbine_orig_C_T",false) ) {
              real a = std::max( 0._fp , std::min( 1._fp , 1 - C_P / (C_T+1.e-10) ) );
              C_T    = 4*a*(1-a);
            }
            C_P    = std::min( C_T , pwr*1.e6/(0.5*1.2*M_PI*rad*rad*inertial_mag0*inertial_mag0*inertial_mag0) );
          } else {
            C_T = 0;
            C_P = 0;
          }
          //////////////////////////////////////////////////////////////////
          // Application of floating turbine motion perturbation
          //////////////////////////////////////////////////////////////////
          if (coupler.get_option<bool>("turbine_floating_motions",false)) {
            real betti_pert;
            if (coupler.get_option<bool>( "turbine_floating_sine"  , false )) {
              auto amp   = coupler.get_option<real>( "turbine_floating_sine_amp"  );
              auto freq  = coupler.get_option<real>( "turbine_floating_sine_freq" );
              auto etime = coupler.get_option<real>( "elapsed_time"               );
              betti_pert = freq*amp*std::cos(freq*etime);
            } else {
              betti_pert = turbine.floating_motions.time_step( dt , instant_mag0 , umag_19_5m , C_T );
            }
            turbine.betti_trace.push_back( betti_pert );
            real mult = 1;
            if ( instant_mag0 > 1.e-10 ) mult = std::max(0._fp,instant_mag0+betti_pert)/instant_mag0;
            instant_mag0 *= mult;
            instant_u0   *= mult;
            instant_v0   *= mult;
          } else {
            turbine.betti_trace.push_back( 0 );
          }
          // Compute inertial u and v at sampling disk
          real inertial_tau = 30;
          turbine.u_samp_inertial = instant_u0*dt/inertial_tau + (inertial_tau-dt)/inertial_tau*turbine.u_samp_inertial;
          turbine.v_samp_inertial = instant_v0*dt/inertial_tau + (inertial_tau-dt)/inertial_tau*turbine.v_samp_inertial;
          // Keep track of the turbine yaw angle and the power production for this time step
          turbine.yaw_trace   .push_back( turbine.yaw_angle );
          turbine.power_trace .push_back( pwr               );
          turbine.cp_trace    .push_back( C_P               );
          turbine.ct_trace    .push_back( C_T               );
          // This is needed to compute the thrust force based on windmill proportion in each cell
          real turb_factor = M_PI*rad*rad/(dx*dy*dz);
          // Fraction of thrust that didn't generate power to send into TKE
          real f_TKE = 0.25_fp; // Recommended by Archer et al., 2020, MWR "Two corrections TKE ..."
          real C_TKE = f_TKE * (C_T - C_P);
          ///////////////////////////////////////////////////
          // Application of disk onto tendencies
          ///////////////////////////////////////////////////
          if (turbine.apply_thrust) {
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
              if (disk_weight_proj(k,j,i) > 0) {
                real wt = disk_weight_proj(k,j,i)*turb_factor;
                tend_u  (k,j,i) += -0.5_fp             *C_T  *instant_mag0*instant_mag0*cos_yaw     *wt;
                tend_v  (k,j,i) += -0.5_fp             *C_T  *instant_mag0*instant_mag0*sin_yaw     *wt;
                tend_tke(k,j,i) +=  0.5_fp*rho_d(k,j,i)*C_TKE*instant_mag0*instant_mag0*instant_mag0*wt;
              }
            });
          }
          ///////////////////////////////////////////////////
          // Update the disk's yaw angle and rot angle
          ///////////////////////////////////////////////////
          // Using only the hub cell's velocity leads to odd behavior. I'm going to use the disk-averaged
          // u and v velocity instead (note it's *not* normal u an v velocity but just plain u and v)
          real max_yaw_speed = turbine.ref_turbine.max_yaw_speed;
          if (! coupler.get_option<bool>("turbine_fixed_yaw",false)) {
            turbine.yaw_angle = turbine.yaw_tend( upstream_uvel , upstream_vvel , dt , turbine.yaw_angle , max_yaw_speed );
          }
          turbine.rot_angle -= rot_speed*dt;
          if (turbine.rot_angle < -2*M_PI) turbine.rot_angle += 2*M_PI;
        } // if (turbine.active)
      } // for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++)

      ///////////////////////////////////////////////////
      // Application of tendencies onto model variables
      ///////////////////////////////////////////////////
      // Update velocities and TKE based on tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        uvel(k,j,i) += dt * tend_u  (k,j,i);
        vvel(k,j,i) += dt * tend_v  (k,j,i);
        tke (k,j,i) += dt * tend_tke(k,j,i);
      });

      // So all tasks know how large the trace is. Makes PNetCDF output easier to manage
      trace_size++;
    }


    // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
    real interp( realHost1d const &ref_umag , realHost1d const &ref_var , real umag ) {
      int imax = ref_umag.extent(0)-1; // Max index for the table
      // If umag exceeds the bounds of the reference data, the turbine is idle and producing no power
      if ( umag < ref_umag(0) || umag > ref_umag(imax) ) return 0;
      // Find the index such that umag lies between ref_umag(i) and ref_umag(i+1)
      int i = 0;
      // Increment past the cell it needs to be in (unless it stops at cell zero)
      while (umag > ref_umag(i)) { i++; }
      // Decrement to make it correct if not task zero
      if (i > 0) i--;
      // Linear interpolation: higher weight for left if it's closer to left
      real fac = (ref_umag(i+1) - umag) / (ref_umag(i+1)-ref_umag(i));
      return fac*ref_var(i) + (1-fac)*ref_var(i+1);
    }

  };

}


