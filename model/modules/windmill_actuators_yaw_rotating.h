
#pragma once

#include "main_header.h"
#include "coupler.h"

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
      real1d     velmag;           // Velocity magnitude at infinity (m/s)
      real1d     thrust_coef;      // Thrust coefficient             (dimensionless)
      real1d     power_coef;       // Power coefficient              (dimensionless)
      real1d     power;            // Power generation               (MW)
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
        // Allocate YAKL arrays to ensure the data is contiguous and to load into the data manager later
        velmag_host      = realHost1d("velmag"     ,velmag_vec     .size());
        thrust_coef_host = realHost1d("thrust_coef",thrust_coef_vec.size());
        power_coef_host  = realHost1d("power_coef" ,power_coef_vec .size());
        power_host       = realHost1d("power"      ,power_vec      .size());
        // Make sure the sizes match
        if ( velmag_host.size() != thrust_coef_host.size() ||
             velmag_host.size() != power_coef_host .size() ||
             velmag_host.size() != power_host      .size() ) {
          yakl::yakl_throw("ERROR: turbine arrays not all the same size");
        }
        // Move from std::vectors into YAKL arrays
        for (int i=0; i < velmag_host.size(); i++) {
          velmag_host     (i) = velmag_vec     [i];
          thrust_coef_host(i) = thrust_coef_vec[i];
          power_coef_host (i) = power_coef_vec [i];
          power_host      (i) = power_vec      [i];
        }
        // Copy from host to device and set other parameters
        this->velmag        = velmag_host     .createDeviceCopy();
        this->thrust_coef   = thrust_coef_host.createDeviceCopy();
        this->power_coef    = power_coef_host .createDeviceCopy();
        this->power         = power_host      .createDeviceCopy();
        this->hub_height    = config["hub_height"   ].as<real>();
        this->blade_radius  = config["blade_radius" ].as<real>();
        this->max_yaw_speed = config["max_yaw_speed"].as<real>(0.5)/180.*M_PI;
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
      int                turbine_id;  // Global turbine ID
      bool               active;      // Whether this turbine affects this MPI task
      real               base_loc_x;  // x location of the tower base
      real               base_loc_y;  // y location of the tower base
      std::vector<real>  power_trace; // Time trace of power generation
      std::vector<real>  yaw_trace;   // Time trace of yaw of the turbine
      real               yaw_angle;   // Current yaw angle   (radians going counter-clockwise from facing west)
      YawTend            yaw_tend;    // Functor to compute the change in yaw
      RefTurbine         ref_turbine; // The reference turbine to use for this turbine
      MPI_Comm           mpi_comm;    // MPI communicator for this turbine
      int                nranks;      // Number of MPI ranks involved with this turbine
      int                sub_rankid;  // My process's rank ID in the sub communicator
      int                owning_sub_rankid; // Subcommunicator rank ID of the owner of this turbine
    };


    template <yakl::index_t MAX_TURBINES=200>
    struct TurbineGroup {
      yakl::SArray<Turbine,1,MAX_TURBINES> turbines;
      int num_turbines;
      TurbineGroup() { num_turbines = 0; }
      void add_turbine( core::Coupler       & coupler     ,
                        real                  base_loc_x  ,
                        real                  base_loc_y  ,
                        RefTurbine    const & ref_turbine ) {
        using yakl::c::parallel_for;
        using yakl::c::SimpleBounds;
        auto i_beg  = coupler.get_i_beg();
        auto j_beg  = coupler.get_j_beg();
        auto nens   = coupler.get_nens();
        auto nx     = coupler.get_nx();
        auto ny     = coupler.get_ny();
        auto nz     = coupler.get_nz();
        auto dx     = coupler.get_dx();
        auto dy     = coupler.get_dy();
        auto dz     = coupler.get_dz();
        auto myrank = coupler.get_myrank();
        auto imm    = coupler.get_data_manager_readwrite().get<real,4>("immersed_proportion");
        // bounds of this MPI task's domain
        real dom_x1  = (i_beg+0 )*dx;
        real dom_x2  = (i_beg+nx)*dx;
        real dom_y1  = (j_beg+0 )*dy;
        real dom_y2  = (j_beg+ny)*dy;
        // Rectangular bounds of this turbine's potential influence
        real turb_x1 = base_loc_x-ref_turbine.blade_radius;
        real turb_x2 = base_loc_x+ref_turbine.blade_radius;
        real turb_y1 = base_loc_y-ref_turbine.blade_radius;
        real turb_y2 = base_loc_y+ref_turbine.blade_radius;
        // Determine if the two domains overlap
        bool active = !( turb_x1 > dom_x2 || // Turbine's to the right
                         turb_x2 < dom_x1 || // Turbine's to the left
                         turb_y1 > dom_y2 || // Turbine's above
                         turb_y2 < dom_y1 ); // Turbine's below
        Turbine loc;
        loc.turbine_id  = num_turbines;
        loc.active      = active;
        loc.base_loc_x  = base_loc_x;
        loc.base_loc_y  = base_loc_y;
        loc.yaw_angle   = 0.;
        loc.yaw_tend    = YawTend();
        loc.ref_turbine = ref_turbine;
        // Get the sub-communicator for tasks this turbine could affect
        MPI_Comm_split( MPI_COMM_WORLD , active ? 1 : 0 , myrank , &(loc.mpi_comm) );
        if (active) {
          // Get subcommunicator size and rank id
          MPI_Comm_size( loc.mpi_comm , &(loc.nranks    ) );
          MPI_Comm_rank( loc.mpi_comm , &(loc.sub_rankid) );
          // Determine if I "own" the turbine (if the hub's in my domain)
          bool owner = base_loc_x >= i_beg*dx && base_loc_x < (i_beg+nx)*dx &&
                       base_loc_y >= j_beg*dy && base_loc_y < (j_beg+ny)*dy ;
          // Gather who owns the turbine, so yaw angles can be broadcast later
          if ( loc.nranks == 1) {
            loc.owning_sub_rankid = 0;
          } else {
            boolHost1d owner_arr("owner_arr",loc.nranks);
            bool owner = base_loc_x >= i_beg*dx && base_loc_x < (i_beg+nx)*dx &&
                         base_loc_y >= j_beg*dy && base_loc_y < (j_beg+ny)*dy ;
            MPI_Allgather( &owner , 1 , MPI_C_BOOL , owner_arr.data() , 1 , MPI_C_BOOL , loc.mpi_comm );
            for (int i=0; i < loc.nranks; i++) { if (owner_arr(i)) loc.owning_sub_rankid = i; }
          }
        } else {
          // Don't want comparisons to give true for any of these
          loc.nranks = -1;
          loc.sub_rankid = -2;
          loc.owning_sub_rankid = -3;
        }
        // Add the turbine
        turbines(num_turbines) = loc;
        // Add the base to immersed_proportion
        int N = 10;
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                          YAKL_LAMBDA (int k, int j, int i, int iens) {
          int count = 0;
          for (int kk=0; kk < N; kk++) {
            for (int jj=0; jj < N; jj++) {
              for (int ii=0; ii < N; ii++) {
                int x = (i_beg+i)*dx + ii*dx/(N-1);
                int y = (j_beg+j)*dy + jj*dy/(N-1);
                int z = (      k)*dz + kk*dz/(N-1);
                auto bx  = base_loc_x;
                auto by  = base_loc_y;
                auto rad = ref_turbine.base_diameter;
                auto h   = ref_turbine.hub_height;
                if ( (x-bx)*(x-bx) + (y-by)*(y-by) <= rad*rad  && z <= h ) count++;
              }
            }
          }
          imm(k,j,i,iens) += static_cast<real>(count)/(N*N*N);
        });
        // Increment the turbine counter
        num_turbines++;
      }
    };


    // Class data members
    TurbineGroup<>  turbine_group;
    int             trace_size;


    void init( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx      = coupler.get_nx  ();
      auto ny      = coupler.get_ny  ();
      auto nz      = coupler.get_nz  ();
      auto nens    = coupler.get_nens();
      auto dx      = coupler.get_dx  ();
      auto dy      = coupler.get_dy  ();
      auto dz      = coupler.get_dz  ();
      auto xlen    = coupler.get_xlen();
      auto ylen    = coupler.get_ylen();
      auto i_beg   = coupler.get_i_beg();
      auto j_beg   = coupler.get_j_beg();
      auto nx_glob = coupler.get_nx_glob();
      auto ny_glob = coupler.get_ny_glob();
      auto dtype   = coupler.get_mpi_data_type();
      auto myrank  = coupler.get_myrank();
      auto &dm     = coupler.get_data_manager_readwrite();

      trace_size = 0;
      
      RefTurbine ref_turbine;
      ref_turbine.init( coupler.get_option<std::string>("turbine_file") , dx , dy , dz );

      // Increment turbines in terms of 10 diameters in each direction
      real xinc = ref_turbine.blade_radius*2*10;
      real yinc = ref_turbine.blade_radius*2*10;
      // Determine the x and y bounds of this MPI task's domain
      for (real y = yinc; y < ylen-yinc; y += yinc) {
        for (real x = xinc; x < xlen-xinc; x += xinc) {
          turbine_group.add_turbine( coupler , x , y , ref_turbine );
        }
      }

      dm.register_and_allocate<real>("windmill_prop","",{nz,ny,nx,nens});
      coupler.register_output_variable<real>( "windmill_prop" , core::Coupler::DIMS_3D );
      // Create an output module in the coupler to dump the windmill portions and the power trace from task zero
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        if (trace_size > 0) {
          nc.redef();
          nc.create_dim( "num_time_steps" , trace_size );
          for (int iturb=0; iturb < turbine_group.num_turbines; iturb++) {
            std::string pow_vname = std::string("power_trace_turb_") + std::to_string(iturb);
            std::string yaw_vname = std::string("yaw_trace_turb_"  ) + std::to_string(iturb);
            nc.create_var<real>( pow_vname , {"num_time_steps"} );
            nc.create_var<real>( yaw_vname , {"num_time_steps"} );
          }
          nc.enddef();
          nc.begin_indep_data();
          for (int iturb=0; iturb < turbine_group.num_turbines; iturb++) {
            auto &turbine = turbine_group.turbines(iturb);
            if (turbine.active && turbine.sub_rankid == turbine.owning_sub_rankid) {
              realHost1d power_arr("power_arr",trace_size);
              realHost1d yaw_arr  ("yaw_arr"  ,trace_size);
              for (int i=0; i < trace_size; i++) { power_arr(i) = turbine.power_trace[i]; }
              for (int i=0; i < trace_size; i++) { yaw_arr  (i) = turbine.yaw_trace  [i]/M_PI*180; }
              std::string pow_vname = std::string("power_trace_turb_") + std::to_string(iturb);
              std::string yaw_vname = std::string("yaw_trace_turb_"  ) + std::to_string(iturb);
              nc.write( power_arr , pow_vname );
              nc.write( yaw_arr   , yaw_vname );
            }
            MPI_Barrier(MPI_COMM_WORLD);
          }
          nc.end_indep_data();
        }
      });
    }


    void apply( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx            = coupler.get_nx   ();
      auto ny            = coupler.get_ny   ();
      auto nz            = coupler.get_nz   ();
      auto nens          = coupler.get_nens ();
      auto dx            = coupler.get_dx   ();
      auto dy            = coupler.get_dy   ();
      auto dz            = coupler.get_dz   ();
      auto i_beg         = coupler.get_i_beg();
      auto j_beg         = coupler.get_j_beg();
      auto dtype         = coupler.get_mpi_data_type();
      auto myrank        = coupler.get_myrank();
      auto &dm           = coupler.get_data_manager_readwrite();
      auto rho_d         = dm.get<real const,4>("density_dry");
      auto uvel          = dm.get<real      ,4>("uvel"       );
      auto vvel          = dm.get<real      ,4>("vvel"       );
      auto wvel          = dm.get<real      ,4>("wvel"       );
      auto tke           = dm.get<real      ,4>("TKE"        );
      auto turb_prop_tot = dm.get<real      ,4>("windmill_prop");

      real4d tend_u  ("tend_u"  ,nz,ny,nx,nens);
      real4d tend_v  ("tend_v"  ,nz,ny,nx,nens);
      real4d tend_w  ("tend_w"  ,nz,ny,nx,nens);
      real4d tend_tke("tend_tke",nz,ny,nx,nens);
      tend_u   = 0;
      tend_v   = 0;
      tend_w   = 0;
      tend_tke = 0;
      turb_prop_tot = 0;

      for (int iturb = 0; iturb < turbine_group.num_turbines; iturb++) {
        auto &turbine = turbine_group.turbines(iturb);
        if (turbine.active) {
          ///////////////////////////////////////////////////
          // Monte Carlo sampling of turbine disk
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
          real turb_area = M_PI*rad*rad;          // Area of the tubine
          real grid_area = std::sqrt(dx*dy)*dz;   // Area of this grid cell in the vertical plane
          int  ncells = static_cast<int>(std::ceil(turb_area/grid_area)); // number of cells in the turbine area
          int  points_per_cell = 1000;               // Desired average number of samples per cell
          int  npoints = ncells * points_per_cell;   // Total samples to randomly select
          real4d mc_weight("mc_weight",nz,ny,nx,nens);   // Hit count for each cell
          mc_weight = 0;
          // Using yakl::Random with a common seed for all MPI tasks to ensure we get the exact same values per task
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(npoints,nens) , YAKL_LAMBDA (int ipoint, int iens) {
            // Randomly sample the radius and angle of a circle at [0,0,0] facing westward
            yakl::Random rand(ipoint);
            // Polar coordinates
            real theta = rand.genFP<real>()*2*static_cast<real>(M_PI);
            real r     = std::sqrt( rand.genFP<real>() );
            // Transorm to cartesian coordinates
            real x     = 0;
            real y     = rad*r*std::cos(theta);
            real z     = rad*r*std::sin(theta);
            // Now rotate x and y according to the yaw angle, and translate to base location
            real xp = base_x + cos_yaw*x - sin_yaw*y;
            real yp = base_y + sin_yaw*x + cos_yaw*y;
            real zp = hub_height + z;
            // if it's in this task's domain, then increment the appropriate cell count atomically
            if (xp >= dom_x1 && xp < dom_x2 && yp >= dom_y1 && yp < dom_y2 ) {
              int i = static_cast<int>(std::floor((xp-dom_x1)/dx));
              int j = static_cast<int>(std::floor((yp-dom_y1)/dy));
              int k = static_cast<int>(std::floor((zp       )/dz));
              // Calculate distance from center
              real r2 = r*r;
              real r4 = r2*r2;
              real r8 = r4*r4;
              real weight = -0.39_fp*r8 + 3.95_fp*r4; 
              weight = std::max(0._fp,std::min(1._fp,weight));
              weight = std::pow( weight , 0.25_fp );
              yakl::atomicAdd( mc_weight(k,j,i,iens) , weight );
            }
          });
          real weight_tot_loc = yakl::intrinsics::sum(mc_weight);
          real weight_tot;
          if (turbine.nranks == 1) {
            weight_tot = weight_tot_loc;
          } else {
            MPI_Allreduce( &weight_tot_loc , &weight_tot , 1 , dtype , MPI_SUM , turbine.mpi_comm );
          }
          ///////////////////////////////////////////////////
          // Aggregation of disk integrals
          ///////////////////////////////////////////////////
          // Aggregate disk-averaged quantites and the proportion of the turbine in each cell
          real4d turb_prop ("turb_prop" ,nz,ny,nx,nens);
          real4d disk_mag  ("disk_mag"  ,nz,ny,nx,nens);
          real4d disk_u    ("disk_u"    ,nz,ny,nx,nens);
          real4d disk_v    ("disk_v"    ,nz,ny,nx,nens);
          real4d disk_unorm("disk_unorm",nz,ny,nx,nens);
          real4d disk_vnorm("disk_vnorm",nz,ny,nx,nens);
          // Sum up weighted normal wind magnitude over the disk by proportion in each cell for this MPI task
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            turb_prop(k,j,i,iens) = mc_weight(k,j,i,iens)/weight_tot;
            turb_prop_tot(k,j,i,iens) += turb_prop(k,j,i,iens);
            real u = uvel(k,j,i,iens)*cos_yaw;
            real v = vvel(k,j,i,iens)*sin_yaw;
            disk_u    (k,j,i,iens) = turb_prop(k,j,i,iens)*uvel(k,j,i,iens);
            disk_v    (k,j,i,iens) = turb_prop(k,j,i,iens)*vvel(k,j,i,iens);
            disk_mag  (k,j,i,iens) = turb_prop(k,j,i,iens)*std::sqrt(u*u+v*v);
            disk_unorm(k,j,i,iens) = turb_prop(k,j,i,iens)*u;
            disk_vnorm(k,j,i,iens) = turb_prop(k,j,i,iens)*v;
          });
          // Calculate local sums
          SArray<real,1,5> sum_loc, sum_glob;
          sum_loc(0) = yakl::intrinsics::sum( disk_u    );
          sum_loc(1) = yakl::intrinsics::sum( disk_v    );
          sum_loc(2) = yakl::intrinsics::sum( disk_mag  );
          sum_loc(3) = yakl::intrinsics::sum( disk_unorm);
          sum_loc(4) = yakl::intrinsics::sum( disk_vnorm);
          // Calculate global sums
          real glob_u, glob_v, glob_mag, glob_unorm, glob_vnorm;
          if (turbine.nranks == 1) {
            glob_u     = sum_loc(0);
            glob_v     = sum_loc(1);
            glob_mag   = sum_loc(2);
            glob_unorm = sum_loc(3);
            glob_vnorm = sum_loc(4);
          } else {
            MPI_Allreduce( sum_loc.data() , sum_glob.data() , sum_loc.size() , dtype , MPI_SUM , turbine.mpi_comm );
            glob_u     = sum_glob(0);
            glob_v     = sum_glob(1);
            glob_mag   = sum_glob(2);
            glob_unorm = sum_glob(3);
            glob_vnorm = sum_glob(4);
          }
          ///////////////////////////////////////////////////
          // Computation of disk properties
          ///////////////////////////////////////////////////
          // Iterate out the induction factor and thrust coefficient, which depend on each other
          real a = 0.3; // Starting guess for axial induction factor based on ... chatGPT. Yeah, I know
          real C_T;
          for (int iter = 0; iter < 100; iter++) {
            C_T = interp( ref_velmag , ref_thrust_coef , glob_mag/(1-a) ); // Interpolate thrust coefficient
            a   = 0.5_fp * ( 1 - std::sqrt(1-C_T) );                       // From 1-D momentum theory
          }
          // Using induction factor, interpolate power coefficient and power for normal wind magnitude at infinity
          real C_P  = interp( ref_velmag , ref_power_coef , glob_mag/(1-a) ); // Interpolate power coef
          real pwr  = interp( ref_velmag , ref_power      , glob_mag/(1-a) ); // Interpolate power
          real mag0 = glob_mag/(1-a);                                         // wind magintude at infinity
          // Keep track of the turbine yaw angle and the power production for this time step
          turbine.yaw_trace  .push_back( turbine.yaw_angle );
          turbine.power_trace.push_back( pwr               );
          // This is needed to compute the thrust force based on windmill proportion in each cell
          real turb_factor = M_PI*rad*rad/(dx*dy*dz);
          real f_TKE = 0.25_fp; // Recommended by Archer et al., 2020, MWR "Two corrections TKE ..."
          real C_TKE   = f_TKE * (C_T - C_P);       // Proportion out some of the unused energy to go into TKE
          real u0      = glob_unorm/(1-a);          // u-velocity at infinity
          real v0      = glob_vnorm/(1-a);          // v-velocity at infinity
          real f_rot = 0.25_fp;
          real C_rot = f_rot * (C_T - C_P);
          ///////////////////////////////////////////////////
          // Application of disk onto tendencies
          ///////////////////////////////////////////////////
          parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                            YAKL_LAMBDA (int k, int j, int i, int iens) {
            if (turb_prop(k,j,i,iens) > 0) {
              real r       = rho_d(k,j,i,iens);         // Needed for tendency on mass-weighted TKE tracer
              tend_u  (k,j,i,iens) += -0.5_fp  *C_T  *mag0*u0       *turb_prop(k,j,i,iens)*turb_factor;
              tend_v  (k,j,i,iens) += -0.5_fp  *C_T  *mag0*v0       *turb_prop(k,j,i,iens)*turb_factor;
              tend_tke(k,j,i,iens) +=  0.5_fp*r*C_TKE*mag0*mag0*mag0*turb_prop(k,j,i,iens)*turb_factor;
              // Add rotation
              real x = (i_beg+i+0.5_fp)*dx;
              real y = (j_beg+j+0.5_fp)*dy;
              real z = (      k+0.5_fp)*dz;
              real angle_rot = std::atan2(z-hub_height,y-base_y) + M_PI*0.5_fp;
              real urot  = 0;
              real vrot  = std::cos(angle_rot);
              real wrot  = std::sin(angle_rot);
              real urotp = cos_yaw*urot - sin_yaw*vrot;
              real vrotp = sin_yaw*urot + cos_yaw*vrot;
              real wrotp = wrot;
              tend_u(k,j,i,iens) += 0.5_fp*C_rot*mag0*urotp*turb_prop(k,j,i,iens)*turb_factor;
              tend_v(k,j,i,iens) += 0.5_fp*C_rot*mag0*vrotp*turb_prop(k,j,i,iens)*turb_factor;
              tend_w(k,j,i,iens) += 0.5_fp*C_rot*mag0*wrotp*turb_prop(k,j,i,iens)*turb_factor;
            }
          });
          ///////////////////////////////////////////////////
          // Update the disk's yaw angle
          ///////////////////////////////////////////////////
          // If this cell contains the turbine hub, update the turbine's yaw angle based on hub wind velocity
          // if (turbine.sub_rankid == turbine.owning_sub_rankid) {
          //   real hub_height = turbine.ref_turbine.hub_height;
          //   int i = static_cast<int>(std::floor((turbine.base_loc_x-dom_x1)/dx));
          //   int j = static_cast<int>(std::floor((turbine.base_loc_y-dom_y1)/dy));
          //   int k = static_cast<int>(std::floor((hub_height               )/dz));
          //   real2d vel("vel",2,nens);
          //   parallel_for( YAKL_AUTO_LABEL() , nens , YAKL_LAMBDA (int iens) {
          //     vel(0,iens) = uvel(k,j,i,iens);
          //     vel(1,iens) = vvel(k,j,i,iens);
          //   });
          //   auto vel_host = vel.createHostCopy();
          //   real max_yaw_speed = turbine.ref_turbine.max_yaw_speed;
          //   real uvel = vel_host(0,0);
          //   real vvel = vel_host(1,0);
          //   turbine.yaw_angle = turbine.yaw_tend( uvel , vvel , dt , turbine.yaw_angle , max_yaw_speed );
          // }
          // // Broadcast the hub-owning new yaw angle to the other processes on this task
          // if (turbine.nranks > 1) {
          //   MPI_Bcast( &(turbine.yaw_angle) , 1 , dtype , turbine.owning_sub_rankid, turbine.mpi_comm );
          // }

          // Resolving the base leads to odd behavior at the hub grid cell. I'm going to use the disk-averaged
          // u and v velocity instead (note it's *not* normal u an v velocity but just plain u and v)
          real max_yaw_speed = turbine.ref_turbine.max_yaw_speed;
          turbine.yaw_angle = turbine.yaw_tend( glob_u , glob_v , dt , turbine.yaw_angle , max_yaw_speed );
        } // if (turbine.active)
      } // for (int iturb = 0; iturb < turbine_group.num_turbines; iturb++)

      ///////////////////////////////////////////////////
      // Application of tendencies onto model variables
      ///////////////////////////////////////////////////
      // Update velocities and TKE based on tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        uvel(k,j,i,iens) += dt * tend_u  (k,j,i,iens);
        vvel(k,j,i,iens) += dt * tend_v  (k,j,i,iens);
        wvel(k,j,i,iens) += dt * tend_w  (k,j,i,iens);
        tke (k,j,i,iens) += dt * tend_tke(k,j,i,iens);
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
