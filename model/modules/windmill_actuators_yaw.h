
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "Betti_simplified.h"

namespace modules {

  // Uses simple disk actuators to represent wind turbines in an LES model by applying friction terms to horizontal
  //   velocities and adding a portion of the thrust not generating power to TKE.
  struct WindmillActuators {


    // Stores information needed to imprint a turbine actuator disk onto the grid. The base location will
    //   sit in the center cell, and there will be halo_x * halo_y on either side of the base cell
    struct RefTurbine {
      realHost1d velmag_host;      // Velocity magnitude at infinity (m/s)
      realHost1d thrust_coef_host; // Thrust coefficient             (dimensionless)
      realHost1d power_coef_host;  // Power coefficient              (dimensionless)
      realHost1d power_host;       // Power generation               (MW)
      realHost1d rotation_host;    // Rotation speed                 (radians / sec)
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
          yakl::yakl_throw("ERROR: turbine arrays not all the same size");
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


    struct Turbine {
      bool                    active;         // Whether this turbine affects this MPI task
      real                    base_loc_x;     // x location of the tower base
      real                    base_loc_y;     // y location of the tower base
      std::vector<real>       power_trace;    // Time trace of power generation
      std::vector<real>       yaw_trace;      // Time trace of yaw of the turbine
      std::vector<real>       mag_trace;      // Time trace of disk-integrated velocity
      std::vector<real>       normmag_trace;  // Time trace of disk-integrated normal velocity
      std::vector<real>       normmag0_trace; // Time trace of disk-integrated free-stream normal velocity used for look-ups
      std::vector<real>       betti_trace;    // Time trace of floating motions perturbations
      std::vector<real>       cp_trace;       // Time trace of coefficient of power
      std::vector<real>       ct_trace;       // Time trace of coefficient of thrust
      real                    mag0_inertial;  // Intertial freestream normal wind magnitude (for power generation)
      real                    yaw_angle;      // Current yaw angle   (radians going counter-clockwise from facing west)
      real                    rot_angle;      // Current rotation angle (radians)
      YawTend                 yaw_tend;       // Functor to compute the change in yaw
      RefTurbine              ref_turbine;    // The reference turbine to use for this turbine
      core::ParallelComm      par_comm;       // MPI communicator for this turbine
      int                     nranks;         // Number of MPI ranks involved with this turbine
      int                     sub_rankid;     // My process's rank ID in the sub communicator
      int                     owning_sub_rankid; // Subcommunicator rank ID of the owner of this turbine
      Floating_motions_betti  floating_motions;
    };


    struct TurbineGroup {
      std::vector<Turbine> turbines;
      void add_turbine( core::Coupler       & coupler     ,
                        real                  base_loc_x  ,
                        real                  base_loc_y  ,
                        RefTurbine    const & ref_turbine ) {
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
        real turb_x1 = base_loc_x-ref_turbine.blade_radius-6*std::sqrt(dx*dy);
        real turb_x2 = base_loc_x+ref_turbine.blade_radius+6*std::sqrt(dx*dy);
        real turb_y1 = base_loc_y-ref_turbine.blade_radius-6*std::sqrt(dx*dy);
        real turb_y2 = base_loc_y+ref_turbine.blade_radius+6*std::sqrt(dx*dy);
        // Determine if the two domains overlap
        bool active = !( turb_x1 > dom_x2 || // Turbine's to the right
                         turb_x2 < dom_x1 || // Turbine's to the left
                         turb_y1 > dom_y2 || // Turbine's above
                         turb_y2 < dom_y1 ); // Turbine's below
        std::random_device rd{};
        Turbine loc;
        loc.active        = active;
        loc.base_loc_x    = base_loc_x;
        loc.base_loc_y    = base_loc_y;
        loc.yaw_angle     = coupler.get_option<real>("turbine_initial_yaw",0);
        loc.rot_angle     = 0.;
        loc.yaw_tend      = YawTend();
        loc.ref_turbine   = ref_turbine;
        loc.mag0_inertial = 0;
        loc.floating_motions.init( loc.ref_turbine.velmag_host      ,
                                   loc.ref_turbine.power_coef_host  ,
                                   loc.ref_turbine.thrust_coef_host );
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
        // parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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


    // IMPORTANT: It looks like c cannot be zero or bad things happen. So make it close to zero if you want zero
    struct DefaultThrustShape {
      YAKL_INLINE real operator() ( real r , real a = 4 , real b = -8 , real c = 0.01 ) const {
        using std::pow;
        if ( r > 1 ) return 0;
        // ((2*(c^a-2*c^b)*x^(2*a)-(c^a-4*c^b)*x^a)/c^b)^(1/(a-b))
        return pow( (2*(pow(c,a)-2*pow(c,b))*pow(r,(2*a))-(pow(c,a)-4*pow(c,b))*pow(r,a))/pow(c,b) , 1/(a-b) );
      }
    };


    struct DefaultProjectionShape1D {
      YAKL_INLINE real operator() ( real x , real xr , int p = 2 ) const {
        real term = 1-(x/xr)*(x/xr);
        if (term <= 0) return 0;
        real term_p = term;
        for (int i = 0; i < p-1; i++) { term_p *= term; }
        return term_p;
      }
    };


    struct DefaultProjectionShape2D {
      YAKL_INLINE real operator() ( real x , real y , real xr , real yr , int p = 2 ) const {
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
    real          etime;
    std::mt19937  gen;


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
        for (int iturb = 0; iturb < x_locs.size(); iturb++) {
          turbine_group.add_turbine( coupler , x_locs.at(iturb) , y_locs.at(iturb) , ref_turbine );
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
            std::string pow_vname      = std::string("power_trace_turb_"   ) + std::to_string(iturb);
            std::string yaw_vname      = std::string("yaw_trace_turb_"     ) + std::to_string(iturb);
            std::string mag_vname      = std::string("mag_trace_turb_"     ) + std::to_string(iturb);
            std::string normmag_vname  = std::string("normmag_trace_turb_" ) + std::to_string(iturb);
            std::string normmag0_vname = std::string("normmag0_trace_turb_") + std::to_string(iturb);
            std::string betti_vname    = std::string("betti_trace_turb_"   ) + std::to_string(iturb);
            std::string cp_vname       = std::string("cp_trace_turb_"      ) + std::to_string(iturb);
            std::string ct_vname       = std::string("ct_trace_turb_"      ) + std::to_string(iturb);
            nc.create_var<real>( pow_vname      , {"num_time_steps"} );
            nc.create_var<real>( yaw_vname      , {"num_time_steps"} );
            nc.create_var<real>( mag_vname      , {"num_time_steps"} );
            nc.create_var<real>( normmag_vname  , {"num_time_steps"} );
            nc.create_var<real>( normmag0_vname , {"num_time_steps"} );
            nc.create_var<real>( betti_vname    , {"num_time_steps"} );
            nc.create_var<real>( cp_vname       , {"num_time_steps"} );
            nc.create_var<real>( ct_vname       , {"num_time_steps"} );
          }
          nc.enddef();
          nc.begin_indep_data();
          for (int iturb=0; iturb < turbine_group.turbines.size(); iturb++) {
            auto &turbine = turbine_group.turbines.at(iturb);
            if (turbine.active && turbine.sub_rankid == turbine.owning_sub_rankid) {
              realHost1d power_arr   ("power_arr"   ,trace_size);
              realHost1d yaw_arr     ("yaw_arr"     ,trace_size);
              realHost1d mag_arr     ("mag_arr"     ,trace_size);
              realHost1d normmag_arr ("normmag_arr" ,trace_size);
              realHost1d normmag0_arr("normmag0_arr",trace_size);
              realHost1d betti_arr   ("betti_arr"   ,trace_size);
              realHost1d cp_arr      ("cp_arr"      ,trace_size);
              realHost1d ct_arr      ("ct_arr"      ,trace_size);
              for (int i=0; i < trace_size; i++) { power_arr   (i) = turbine.power_trace   .at(i); }
              for (int i=0; i < trace_size; i++) { yaw_arr     (i) = turbine.yaw_trace     .at(i)/M_PI*180; }
              for (int i=0; i < trace_size; i++) { mag_arr     (i) = turbine.mag_trace     .at(i); }
              for (int i=0; i < trace_size; i++) { normmag_arr (i) = turbine.normmag_trace .at(i); }
              for (int i=0; i < trace_size; i++) { normmag0_arr(i) = turbine.normmag0_trace.at(i); }
              for (int i=0; i < trace_size; i++) { betti_arr   (i) = turbine.betti_trace   .at(i); }
              for (int i=0; i < trace_size; i++) { cp_arr      (i) = turbine.cp_trace      .at(i); }
              for (int i=0; i < trace_size; i++) { ct_arr      (i) = turbine.ct_trace      .at(i); }
              std::string pow_vname      = std::string("power_trace_turb_"   ) + std::to_string(iturb);
              std::string yaw_vname      = std::string("yaw_trace_turb_"     ) + std::to_string(iturb);
              std::string mag_vname      = std::string("mag_trace_turb_"     ) + std::to_string(iturb);
              std::string normmag_vname  = std::string("normmag_trace_turb_" ) + std::to_string(iturb);
              std::string normmag0_vname = std::string("normmag0_trace_turb_") + std::to_string(iturb);
              std::string betti_vname    = std::string("betti_trace_turb_"   ) + std::to_string(iturb);
              std::string cp_vname       = std::string("cp_trace_turb_"      ) + std::to_string(iturb);
              std::string ct_vname       = std::string("ct_trace_turb_"      ) + std::to_string(iturb);
              nc.write( power_arr    , pow_vname      );
              nc.write( yaw_arr      , yaw_vname      );
              nc.write( mag_arr      , mag_vname      );
              nc.write( normmag_arr  , normmag_vname  );
              nc.write( normmag0_arr , normmag0_vname );
              nc.write( betti_arr    , betti_vname    );
              nc.write( cp_arr       , cp_vname       );
              nc.write( ct_arr       , ct_vname       );
            }
            coupler.get_parallel_comm().barrier();
            turbine.power_trace   .clear();
            turbine.yaw_trace     .clear();
            turbine.mag_trace     .clear();
            turbine.normmag_trace .clear();
            turbine.normmag0_trace.clear();
            turbine.betti_trace   .clear();
            turbine.cp_trace      .clear();
            turbine.ct_trace      .clear();
          }
          nc.end_indep_data();
        }
        trace_size = 0;
      });

      etime = 0;

      std::random_device rd{};
      gen = std::mt19937{rd()};
    }


    template < class THRUST_SHAPE  = DefaultThrustShape       ,
               class PROJ_SHAPE_1D = DefaultProjectionShape1D ,
               class PROJ_SHAPE_2D = DefaultProjectionShape2D >
    void apply( core::Coupler      & coupler                                     ,
                real                 dt                                          ,
                THRUST_SHAPE  const & thrust_shape  = DefaultThrustShape()       ,
                PROJ_SHAPE_1D const & proj_shape_1d = DefaultProjectionShape1D() ,
                PROJ_SHAPE_2D const & proj_shape_2d = DefaultProjectionShape2D() ) {
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

      real3d tend_u  ("tend_u"  ,nz,ny,nx);
      real3d tend_v  ("tend_v"  ,nz,ny,nx);
      real3d tend_tke("tend_tke",nz,ny,nx);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
          real proj_rad = std::max( 5._fp , 5*std::sqrt(dx*dy) ); // Arbitrarily limit to 5m for now
          real3d disk_weight_proj("disk_weight_proj",nz,ny,nx);
          real3d disk_weight_samp("disk_weight_samp",nz,ny,nx);
          real2d umag_19_5m_2d("umag_19_5m_2d"   ,ny,nx);
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
            disk_weight_proj(k,j,i) = 0;
            disk_weight_samp(k,j,i) = 0;
            if (k == 0) {
              real x = (i_beg+i+0.5)*dx;
              real y = (j_beg+j+0.5)*dy;
              if (std::abs(x-base_x) <= rad && std::abs(y-base_y) <= rad) {
                int k19_5 = std::max( 0._fp , std::round(19.5/dz-0.5) );
                real u = uvel(k19_5,j,i);
                real v = vvel(k19_5,j,i);
                umag_19_5m_2d(j,i) = std::sqrt(u*u + v*v);
              } else {
                umag_19_5m_2d(j,i) = 0;
              }
            }
          });
          {
            real xr = 5*dx;
            int nper  = 10;
            int num_x = (int) std::ceil(xr*2 /dx*nper);
            int num_y = (int) std::ceil(rad*2/dy*nper);
            int num_z = (int) std::ceil(rad*2/dz*nper);
            parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_z,num_y,num_x) , YAKL_LAMBDA (int k, int j, int i) {
              real x = -xr  + (2*xr *i)/(num_x-1);
              real y = -rad + (2*rad*j)/(num_y-1);
              real z = -rad + (2*rad*k)/(num_z-1);
              real proj1d = proj_shape_1d(x,xr);
              // Now rotate x and y according to the yaw angle, and translate to base location
              real xp = base_x     + cos_yaw*x - sin_yaw*y;
              real yp = base_y     + sin_yaw*x + cos_yaw*y;
              real zp = hub_height + z;
              // if it's in this task's domain, then increment the appropriate cell count atomically
              if (xp >= dom_x1 && xp < dom_x2 && yp >= dom_y1 && yp < dom_y2 ) {
                int  i = static_cast<int>(std::floor((xp-dom_x1)/dx));
                int  j = static_cast<int>(std::floor((yp-dom_y1)/dy));
                int  k = static_cast<int>(std::floor((zp       )/dz));
                real rloc = std::sqrt(y*y+z*z);
                if (rloc <= rad) {
                  yakl::atomicAdd( disk_weight_proj(k,j,i) , thrust_shape(rloc/rad)*proj1d );
                  if (x > dx && x < 2*dx) yakl::atomicAdd( disk_weight_samp(k,j,i) , thrust_shape(rloc/rad) );
                }
              }
            });
          }
          using yakl::componentwise::operator>;
          yakl::SArray<real,1,4> weights_tot;
          weights_tot(0) = yakl::intrinsics::sum(disk_weight_proj);
          weights_tot(1) = yakl::intrinsics::sum(disk_weight_samp);
          weights_tot(2) = yakl::intrinsics::sum(umag_19_5m_2d   );
          weights_tot(3) = (real) yakl::intrinsics::count(umag_19_5m_2d > 0._fp);
          weights_tot = turbine.par_comm.all_reduce( weights_tot , MPI_SUM , "windmill_Allreduce1" );
          real disk_proj_tot = weights_tot(0);
          real disk_samp_tot = weights_tot(1);
          real umag_19_5m    = weights_tot(2) / weights_tot(3);
          ///////////////////////////////////////////////////
          // Aggregation of disk integrals
          ///////////////////////////////////////////////////
          // Aggregate disk-averaged quantites and the proportion of the turbine in each cell
          real3d disk_u("disk_u",nz,ny,nx);
          real3d disk_v("disk_v",nz,ny,nx);
          // Sum up weighted normal wind magnitude over the disk by proportion in each cell for this MPI task
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
            if (disk_weight_samp(k,j,i) > 0) {
              disk_weight_samp(k,j,i) /= disk_samp_tot;
              samp_weight_tot (k,j,i) += disk_weight_samp(k,j,i);
              disk_u          (k,j,i)  = disk_weight_samp(k,j,i)*uvel(k,j,i);
              disk_v          (k,j,i)  = disk_weight_samp(k,j,i)*vvel(k,j,i);
            } else {
              disk_u          (k,j,i) = 0;
              disk_v          (k,j,i) = 0;
            }
            if (disk_weight_proj(k,j,i) > 0) {
              disk_weight_proj(k,j,i) /= disk_proj_tot;
              proj_weight_tot (k,j,i) += disk_weight_proj(k,j,i);
            }
          });
          SArray<real,1,2> sums;
          sums(0) = yakl::intrinsics::sum( disk_u );
          sums(1) = yakl::intrinsics::sum( disk_v );
          sums = turbine.par_comm.all_reduce( sums , MPI_SUM , "windmill_Allreduce2" );
          real glob_u    = sums(0);
          real glob_v    = sums(1);
          real glob_unorm = glob_u*cos_yaw;
          real glob_vnorm = glob_v*sin_yaw;
          real glob_mag   = std::sqrt(glob_unorm*glob_unorm + glob_vnorm*glob_vnorm);
          turbine.mag_trace    .push_back(std::sqrt(glob_u*glob_u+glob_v*glob_v));
          turbine.normmag_trace.push_back(glob_mag);
          //////////////////////////////////////////////////////////////////
          // Computation axial induction factor and freestream velocities
          //////////////////////////////////////////////////////////////////
          real a = 0;
          for (int iter = 0; iter < 100; iter++) {
            real C_T = std::min( 1._fp , interp( ref_velmag , ref_thrust_coef , glob_mag/(1-a) ) );
            a        = 0.5_fp * ( 1 - std::sqrt(1-C_T) );                       // From 1-D momentum theory
          }
          real mag0 = glob_mag  /(1-a);  // wind magintude at infinity
          real u0   = glob_unorm/(1-a);  // u-velocity at infinity
          real v0   = glob_vnorm/(1-a);  // v-velocity at infinity
          //////////////////////////////////////////////////////////////////
          // Application of floating turbine motion perturbation
          //////////////////////////////////////////////////////////////////
          turbine.normmag0_trace.push_back( mag0 );
          if (coupler.get_option<bool>("turbine_floating_motions",false)) {
            real betti_pert = turbine.floating_motions.time_step( dt , mag0 , umag_19_5m );
            turbine.betti_trace.push_back( betti_pert );
            real mult = 1;
            if ( mag0 > 1.e-10 ) mult = std::max(0._fp,mag0+betti_pert)/mag0;
            mag0 *= mult;
            u0   *= mult;
            v0   *= mult;
          } else {
            turbine.betti_trace.push_back( 0 );
          }
          ///////////////////////////////////////////////////
          // Computation of disk properties
          ///////////////////////////////////////////////////
          // Using induction factor, interpolate power coefficient and power for normal wind magnitude at infinity
          real C_T       = interp( ref_velmag , ref_thrust_coef , mag0 ); // Interpolate power coef
          real inertial_tau = 30;
          turbine.mag0_inertial = mag0*dt/inertial_tau + (inertial_tau-dt)/inertial_tau*turbine.mag0_inertial;
          real C_P       = interp( ref_velmag , ref_power_coef  , turbine.mag0_inertial ); // Interpolate power coef
          real pwr       = interp( ref_velmag , ref_power       , turbine.mag0_inertial ); // Interpolate power
          real rot_speed = do_blades ? interp( ref_velmag , ref_rotation , turbine.mag0_inertial ) : 0; // Interpolate rotation speed
          // Keep track of the turbine yaw angle and the power production for this time step
          turbine.yaw_trace  .push_back( turbine.yaw_angle );
          turbine.power_trace.push_back( pwr               );
          turbine.cp_trace   .push_back( C_P               );
          turbine.ct_trace   .push_back( C_T               );
          // This is needed to compute the thrust force based on windmill proportion in each cell
          real turb_factor = M_PI*rad*rad/(dx*dy*dz);
          // Fraction of thrust that didn't generate power to send into TKE
          real f_TKE = 0.25_fp; // Recommended by Archer et al., 2020, MWR "Two corrections TKE ..."
          real C_TKE = f_TKE * (C_T - C_P);
          ///////////////////////////////////////////////////
          // Application of disk onto tendencies
          ///////////////////////////////////////////////////
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
            if (disk_weight_proj(k,j,i) > 0) {
              tend_u  (k,j,i) += -0.5_fp             *C_T  *mag0*u0       *disk_weight_proj(k,j,i)*turb_factor;
              tend_v  (k,j,i) += -0.5_fp             *C_T  *mag0*v0       *disk_weight_proj(k,j,i)*turb_factor;
              tend_tke(k,j,i) +=  0.5_fp*rho_d(k,j,i)*C_TKE*mag0*mag0*mag0*disk_weight_proj(k,j,i)*turb_factor;
            }
          });
          ///////////////////////////////////////////////////
          // Update the disk's yaw angle and rot angle
          ///////////////////////////////////////////////////
          // Using only the hub cell's velocity leads to odd behavior. I'm going to use the disk-averaged
          // u and v velocity instead (note it's *not* normal u an v velocity but just plain u and v)
          real max_yaw_speed = turbine.ref_turbine.max_yaw_speed;
          if (! coupler.get_option<bool>("turbine_fixed_yaw",false)) {
            turbine.yaw_angle = turbine.yaw_tend( glob_u , glob_v , dt , turbine.yaw_angle , max_yaw_speed );
          }
          turbine.rot_angle -= rot_speed*dt;
          if (turbine.rot_angle < -2*M_PI) turbine.rot_angle += 2*M_PI;
        } // if (turbine.active)
      } // for (int iturb = 0; iturb < turbine_group.turbines.size(); iturb++)

      ///////////////////////////////////////////////////
      // Application of tendencies onto model variables
      ///////////////////////////////////////////////////
      // Update velocities and TKE based on tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        uvel(k,j,i) += dt * tend_u  (k,j,i);
        vvel(k,j,i) += dt * tend_v  (k,j,i);
        tke (k,j,i) += dt * tend_tke(k,j,i);
      });

      // So all tasks know how large the trace is. Makes PNetCDF output easier to manage
      trace_size++;
      etime += dt;
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


