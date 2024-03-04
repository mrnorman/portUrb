
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  // Saving some sagemath code that correctly yaws and pitches the east-facing disk with rotation matrices
  // It also correctly samples evenly within a circle
  //
  // N = 1000
  // yaw   =  60/180*pi
  // pitch = -30/180*pi
  // Ry    = Matrix(3,3,[cos(pitch),0,sin(pitch),0,1,0,-sin(pitch),0,cos(pitch)])
  // Rz    = Matrix(3,3,[cos(yaw),-sin(yaw),0,sin(yaw),cos(yaw),0,0,0,1])
  // pnts = [[0. for i in range(3)] for j in range(N)]
  // for i in range(N) :
  //     theta = random()*2*pi
  //     r     = sqrt(random())
  //     pnt   = [0,r*cos(theta),r*sin(theta)]
  //     pnts[i] = (Rz*Ry*vector(pnt)).list()
  // point3d(pnts,size=2).show()

  // Uses simple disk actuators to represent wind turbines in an LES model by applying friction terms to horizontal
  //   velocities and adding a portion of the thrust not generating power to TKE. Adapted from the following paper:
  // https://egusphere.copernicus.org/preprints/2023/egusphere-2023-491/egusphere-2023-491.pdf
  struct WindmillActuators {


    struct RefTurbine {
      real1d velmag;        // Velocity magnitude at infinity (m/s)
      real1d thrust_coef;   // Thrust coefficient             (dimensionless)
      real1d power_coef;    // Power coefficient              (dimensionless)
      real1d power;         // Power generation               (MW)
      real   hub_height;    // Hub height                     (m)
      real   blade_radius;  // Blade radius                   (m)
      real   max_yaw_speed; // Angular active yawing speed    (radians / sec)
      int    turb_nx;       // Number of cells in the x-direction this turbine could possibly interact with
      int    turb_ny;       // Number of cells in the y-direction this turbine could possibly interact with
      int    turb_nz;       // Number of cells in the z-direction this turbine could possibly interact with
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
        this->turb_nx       = static_cast<int>(std::ceil(blade_radius/dx))*2+1;
        this->turb_ny       = static_cast<int>(std::ceil(blade_radius/dy))*2+1;
        this->turb_nz       = static_cast<int>(std::ceil((hub_height+blade_radius)/dz));
      }
    };


    // Yaw will change as if it were an active yaw system that moves at a certain max speed. It will react
    //   to some time average of the wind velocities. The operator() outputs the yaw angle tendency in 
    //   radians per second.
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
        dir_upwind = std::atan2(vavg,uavg);
        // If the upwind direction and current yaw angle are of different sign, we've hit the sign change
        //    discontinuity in describing the angle. So make the negative value positive in this case
        if (dir_upwind < 0 && yaw > 0) dir_upwind += 2*M_PI;
        if (dir_upwind > 0 && yaw < 0) yaw        += 2*M_PI;
        real tend = (dir_upwind - yaw) / dt;
        if (tend > 0) { return std::min(  max_yaw_speed , tend ); }
        else          { return std::max( -max_yaw_speed , tend ); }
      }
    };


    struct Turbine {
      int                icenter;     // i-index for this MPI task fo the turbine center
      int                jcenter;     // j-index for this MPI task fo the turbine center
      real               base_loc_x;  // x location of the tower base
      real               base_loc_y;  // y location of the tower base
      real3d             tend_u;      // u-velocity friction tendency        (turb_nz,turb_ny,turb_nx)
      real3d             tend_v;      // v-velocity friction tendency        (turb_nz,turb_ny,turb_nx)
      real3d             tend_tke;    // Mass-weighted TKE tendency          (turb_nz,turb_ny,turb_nx)
      std::vector<real>  power_trace; // Time trace of power generation
      std::vector<real>  yaw_trace;   // Time trace of yaw of the turbine
      real               yaw_angle;   // Current yaw angle   (radians going counter-clockwise from facing west)
      YawTend            yaw_tend;    // Functor to compute the change in yaw
      RefTurbine         ref_turbine; // The reference turbine to use for this turbine
    };


    template <yakl::index_t MAX_TURBINES=200>
    struct TurbineGroup {
      yakl::SArray<Turbine,1,MAX_TURBINES> turbines;
      int num_turbines;
      TurbineGroup() { num_turbines = 0; }
      void add_turbine( real base_loc_x , real base_loc_y , int icenter , int jcenter ,
                        RefTurbine const &ref_turbine ) {
        int turb_nx = ref_turbine.turb_nx;
        int turb_ny = ref_turbine.turb_ny;
        int turb_nz = ref_turbine.turb_nz;
        Turbine loc;
        loc.icenter     = icenter;
        loc.jcenter     = jcenter;
        loc.base_loc_x  = base_loc_x;
        loc.base_loc_y  = base_loc_y;
        loc.tend_u      = real3d("turbine_tend_u"  ,turb_nz,turb_ny,turb_nx);
        loc.tend_v      = real3d("turbine_tend_v"  ,turb_nz,turb_ny,turb_nx);
        loc.tend_tke    = real3d("turbine_tend_tke",turb_nz,turb_ny,turb_nx);
        loc.yaw_angle   = 0.;
        loc.yaw_tend    = YawTend();
        loc.ref_turbine = ref_turbine;
        num_turbines++;
      }
    }


    // Class data members
    TurbineGroup  turbine_group;
    int           halo_x;
    int           halo_y;


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
      auto &dm     = coupler.get_data_manager_readwrite();
      
      RefTurbine ref_turbine;
      ref_turbine.init( coupler.get_option<std::string>("turbine_file") , dx , dy , dz );

      // Increment turbines in terms of 10 diameters in each direction
      real xinc = ref_turbine.blade_radius*2*10/dx;
      real yinc = ref_turbine.blade_radius*2*10/dy;
      // Determine the x and y bounds of this MPI task's domain
      real dom_x1  = (i_beg+0   )*dx;
      real dom_x2  = (i_beg+nx-1)*dx;
      real dom_y1  = (j_beg+0   )*dy;
      real dom_y2  = (j_beg+ny-1)*dy;
      for (real y = yinc; j < ylen-yinc; j += yinc) {
        for (real x = xinc; i < xlen-xinc; i += xinc) {
          // Determine this turbine's domain of influence
          real turb_x1 = x-ref_turbine.blade_radius;
          real turb_x2 = x+ref_turbine.blade_radius;
          real turb_y1 = y-ref_turbine.blade_radius;
          real turb_y2 = y+ref_turbine.blade_radius;
          // If the turbine's domain of influence overlaps with this MPI task's domain, then add it to this task
          // https://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other
          if ( turb_x1 < dom_x2 && turb_x2 > dom_x1 && turb_y2 > dom_y1 && turb_y1 < dom_y2 ) {
            turbine_group.add_turbine( x , y , std::floor(x/dx)-i_beg , std::floor(y/dy)-j_beg , ref_turbine );
          }
        }
      }
      // This is the number of halo cells we must communicate to ensure we can resolve all turbines that influence
      //   this particular MPI task. It's equal to the number of cells it takes to span the diameter of a blade plane
      this->halo_x = static_cast<int>(std::ceil(ref_turbine.blade_radius*2/dx));
      this->halo_y = static_cast<int>(std::ceil(ref_turbine.blade_radius*2/dy));

      // // Create an output module in the coupler to dump the windmill portions and the power trace from task zero
      // coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
      //   nc.redef();
      //   nc.create_var<real>( "windmill_prop" , {"z","y","x","ens"});
      //   nc.enddef();
      //   if (num_traces > 0) {
      //     nc.redef();
      //     auto nens = coupler.get_nens();
      //     nc.create_dim( "num_time_steps" , num_traces );
      //     nc.create_var<real>( "power_trace" , {"num_time_steps","ens"} );
      //     nc.enddef();
      //     nc.begin_indep_data();
      //     if (coupler.get_myrank() == 0) {
      //       nc.write( power_trace.subset_slowest_dimension(0,num_traces-1) , "power_trace" );
      //     }
      //     nc.end_indep_data();
      //   }
      //   auto i_beg = coupler.get_i_beg();
      //   auto j_beg = coupler.get_j_beg();
      //   std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg,0};
      //   nc.write_all(coupler.get_data_manager_readonly().get<real const,4>("windmill_prop"),"windmill_prop",start_3d);
      // });
    }


    void apply( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx             = coupler.get_nx  ();
      auto ny             = coupler.get_ny  ();
      auto nz             = coupler.get_nz  ();
      auto nens           = coupler.get_nens();
      auto dx             = coupler.get_dx  ();
      auto dy             = coupler.get_dy  ();
      auto dz             = coupler.get_dz  ();
      auto &dm            = coupler.get_data_manager_readwrite();
      auto windmill_prop  = dm.get<real const,4>("windmill_prop");
      auto rho_d          = dm.get<real const,4>("density_dry");
      auto uvel           = dm.get<real      ,4>("uvel"       );
      auto vvel           = dm.get<real      ,4>("vvel"       );
      auto tke            = dm.get<real      ,4>("TKE"        );
      auto tend_u         = uvel.createDeviceObject(); // Create device arrays of the same size to hold tendencies
      auto tend_v         = vvel.createDeviceObject();
      auto tend_tke       = tke .createDeviceObject();
      real4d windmill_power("windmill_power",nens,nz,ny,nx); // To hold per-cell power generation for a later summation
      real rad = coupler.get_option<real>("blade_radius");

      // Grab reference turbine performance data from the coupler's data manager
      auto ref_umag     = coupler.get_data_manager_readonly().get<real const,1>("turbine_velmag"     );
      auto ref_thr_coef = coupler.get_data_manager_readonly().get<real const,1>("turbine_thrust_coef");
      auto ref_pow_coef = coupler.get_data_manager_readonly().get<real const,1>("turbine_power_coef" );
      auto ref_pow      = coupler.get_data_manager_readonly().get<real const,1>("turbine_power"      );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        real r     = rho_d(k,j,i,iens); // Needed for tendency on mass-weighted TKE tracer
        real u     = uvel (k,j,i,iens); // Velocities in the coupler are not mass-weighted
        real v     = vvel (k,j,i,iens);
        real mag   = std::sqrt(u*u + v*v); // Horizontal wind only implies disk is not pitching back and forth
        real C_T;
        real a = 0.3; // Starting guess for axial induction factor based on ... chatGPT. Yeah, I know
        // Iterate to ensure we get a converged estimate of C_T and a since they depend on each other
        for (int iter = 0; iter < 10; iter++) {
          C_T = interp( ref_umag , ref_thr_coef , mag/(1-a) );  // Interpolate thrust coefficient based on u_infinity
          // Below is from https://egusphere.copernicus.org/preprints/2023/egusphere-2023-491/egusphere-2023-491.pdf
          // From 1-D momentum theory
          a   = 0.5_fp * ( 1 - std::sqrt(1-C_T) );
        }
        real C_P   = interp( ref_umag , ref_pow_coef , mag/(1-a) ); // Interpolate power coefficient based on u_infinity
        real pwr   = interp( ref_umag , ref_pow      , mag/(1-a) ); // Interpolate power based on u_infinity
        real f_TKE = 0.25_fp;              // Recommended by Archer et al., 2020, MWR "Two corrections for TKE ..."
        real C_TKE = f_TKE * (C_T - C_P);  // Proportion out some of the unused energy to go into TKE
        real mag0  = mag/(1-a);            // wind magintude at infinity
        real u0    = u  /(1-a);            // u-velocity at infinity
        real v0    = v  /(1-a);            // v-velocity at infinity
        // These friction terms are from Bui et al preprint listed above
        tend_u        (k,j,i,iens) = -0.5_fp  *C_T  *mag0*u0       *windmill_prop(k,j,i,iens);
        tend_v        (k,j,i,iens) = -0.5_fp  *C_T  *mag0*v0       *windmill_prop(k,j,i,iens);
        tend_tke      (k,j,i,iens) =  0.5_fp*r*C_TKE*mag0*mag0*mag0*windmill_prop(k,j,i,iens);
        // Power has ensemble as slowest varying dimension to make it easier to do per-ensemble reductions
        // windmill_prop to be a true proportion of a single actuator disks's area needs to be divided by
        //   total disk area and multiplied by grid volume. Then it can be a multiplier to power that was
        //   interpolated from the u_infinity speed from the turbine performance table.
        windmill_power(iens,k,j,i) = pwr/(M_PI*rad*rad)*dx*dy*dz   *windmill_prop(k,j,i,iens);
      });
      // Save the total power produced on the grid for each ensemble
      realHost1d power_loc("power_loc",nens);
      // Perform a per-ensemble local reduction
      for (int iens = 0; iens < nens; iens++) {
        power_loc(iens) = yakl::intrinsics::sum(windmill_power.slice<3>(iens,0,0,0));
      }
      // Do an MPI Reduce to task zero of the global sum of total power produced
      auto power_glob = power_loc.createHostObject();
      auto dtype = coupler.get_mpi_data_type();
      MPI_Reduce( power_loc.data() , power_glob.data() , nens , dtype , MPI_SUM , 0 , MPI_COMM_WORLD );
      // Store task zero's global power output in partially filled array and increment the count in that array
      if (coupler.get_myrank() == 0) {
        for (int iens=0; iens < nens; iens++) {
          power_trace(num_traces,iens) = power_glob(iens);
          num_traces++;
        }
      }
      // Update velocities and TKE based on tendencies
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        uvel(k,j,i,iens) += dt * tend_u  (k,j,i,iens);
        vvel(k,j,i,iens) += dt * tend_v  (k,j,i,iens);
        tke (k,j,i,iens) += dt * tend_tke(k,j,i,iens);
      });
    }


    // Linear interpolation in a reference variable based on u_infinity and reference u_infinity
    YAKL_INLINE static real interp( realConst1d const &ref_umag , realConst1d const &ref_var , real umag ) {
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
