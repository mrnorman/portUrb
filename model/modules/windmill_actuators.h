
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  struct WindmillActuators {

    int static constexpr MAX_TRACES = 1e7;
    realHost2d power_trace;
    int num_traces;

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
      YAML::Node config = YAML::LoadFile( coupler.get_option<std::string>("turbine_file") );
      if ( !config ) { endrun("ERROR: Invalid turbine input file"); }
      coupler.set_option<real>( "hub_height"   , config["hub_height"  ].as<real>() );
      coupler.set_option<real>( "blade_radius" , config["blade_radius"].as<real>() );
      auto velmag_vec      = config["velocity_magnitude"].as<std::vector<real>>();
      auto thrust_coef_vec = config["thrust_coef"       ].as<std::vector<real>>();
      auto power_coef_vec  = config["power_coef"        ].as<std::vector<real>>();
      auto power_vec       = config["power_megawatts"   ].as<std::vector<real>>();
      realHost1d velmag_host     ("velmag"     ,velmag_vec     .size());
      realHost1d thrust_coef_host("thrust_coef",thrust_coef_vec.size());
      realHost1d power_coef_host ("power_coef" ,power_coef_vec .size());
      realHost1d power_host      ("power"      ,power_vec      .size());
      if ( velmag_host.size() != thrust_coef_host.size() ||
           velmag_host.size() != power_coef_host .size() ||
           velmag_host.size() != power_host      .size() ) {
        yakl::yakl_throw("ERROR: turbine arrays not all the same size");
      }
      for (int i=0; i < velmag_host.size(); i++) {
        velmag_host     (i) = velmag_vec     [i];
        thrust_coef_host(i) = thrust_coef_vec[i];
        power_coef_host (i) = power_coef_vec [i];
        power_host      (i) = power_vec      [i];
      }
      dm.register_and_allocate<real>( "turbine_velmag"      , "" , {static_cast<int>(velmag_host.size())} , {"turb_points"} );
      dm.register_and_allocate<real>( "turbine_thrust_coef" , "" , {static_cast<int>(velmag_host.size())} , {"turb_points"} );
      dm.register_and_allocate<real>( "turbine_power_coef"  , "" , {static_cast<int>(velmag_host.size())} , {"turb_points"} );
      dm.register_and_allocate<real>( "turbine_power"       , "" , {static_cast<int>(velmag_host.size())} , {"turb_points"} );
      auto velmag      = dm.get<real,1>("turbine_velmag"     );
      auto thrust_coef = dm.get<real,1>("turbine_thrust_coef");
      auto power_coef  = dm.get<real,1>("turbine_power_coef" );
      auto power       = dm.get<real,1>("turbine_power"      );
      velmag_host     .deep_copy_to(velmag     );
      thrust_coef_host.deep_copy_to(thrust_coef);
      power_coef_host .deep_copy_to(power_coef );
      power_host      .deep_copy_to(power      );
      yakl::fence();
      std::cout << velmag      << std::endl;
      std::cout << thrust_coef << std::endl;
      std::cout << power_coef  << std::endl;
      std::cout << power       << std::endl;
      dm.register_and_allocate<real>( "windmill_prop" , "" , {nz,ny,nx,nens} , {"z","y","x","nens"} );
      auto windmill_prop = dm.get<real,4>("windmill_prop");
      windmill_prop = 0;
      real z0  = coupler.get_option<real>("hub_height");
      real rad = coupler.get_option<real>("blade_radius");
      int ny_t = std::ceil(rad*2/dy);
      int nz_t = std::ceil((z0+rad)/dz);
      real y0 = ny_t*dy/2;
      int constexpr N = 1000;
      real ddy = dy / N;
      real ddz = dz / N;
      realHost2d templ_host("templ",nz_t,ny_t);
      size_t count_total = 0;
      for (int k=0; k < nz_t; k++) {
        for (int j=0; j < ny_t; j++) {
          int count = 0;
          for (int kk=0; kk < N; kk++) {
            for (int jj=0; jj < N; jj++) {
              real z = k*dz + kk*ddz;
              real y = j*dy + jj*ddy;
              if (std::sqrt((z-z0)*(z-z0)+(y-y0)*(y-y0))/rad < 1) { count++; count_total++; }
            }
          }
          templ_host(k,j) = static_cast<double>(count);
        }
      }
      for (int k=0; k < nz_t; k++) {
        for (int j=0; j < ny_t; j++) {
          templ_host(k,j) = M_PI*rad*rad*templ_host(k,j)/static_cast<double>(count_total)/(dx*dy*dz);
        }
      }
      auto templ = templ_host.createDeviceCopy();
      std::vector<int> xind_vec ;
      std::vector<int> yind0_vec;
      int xinc = (int) std::ceil(rad*2*10/dx);
      int yinc = (int) std::ceil(rad*2*5 /dy);
      for (int i=xinc; i < nx_glob; i += xinc) { xind_vec .push_back(i); }
      for (int j=yinc; j < ny_glob; j += yinc) { yind0_vec.push_back(j); }
      intHost1d xind_host ("wf_xind" ,xind_vec .size());
      intHost1d yind0_host("wf_yind0",yind0_vec.size());
      for (int i=0; i < xind_vec .size(); i++) { xind_host (i) = xind_vec [i]; }
      for (int j=0; j < yind0_vec.size(); j++) { yind0_host(j) = yind0_vec[j]; }
      auto xind  = xind_host .createDeviceCopy();
      auto yind0 = yind0_host.createDeviceCopy();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz_t,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        // Loop over windmills
        for (int jj=0; jj < yind0.size(); jj++) {
          for (int ii=0; ii < xind .size(); ii++) {
            // If we're at the right index for a given windmill, then copy in the template
            if (i_beg+i == xind(ii) && j_beg+j >= yind0(jj) && j_beg+j <= yind0(jj)+ny_t-1) {
              windmill_prop(k,j,i,iens) = templ(k,j_beg+j-yind0(jj));
            }
          }
        }
      });
      if (coupler.get_myrank() == 0) {
        power_trace = realHost2d("power_trace",MAX_TRACES,nens);
        num_traces = 0;
      }
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        nc.redef();
        nc.create_var<real>( "windmill_prop" , {"z","y","x","ens"});
        nc.enddef();
        if (num_traces > 0) {
          nc.redef();
          auto nens = coupler.get_nens();
          nc.create_dim( "num_time_steps" , num_traces );
          nc.create_var<real>( "power_trace" , {"num_time_steps","ens"} );
          nc.enddef();
          nc.begin_indep_data();
          if (coupler.get_myrank() == 0) {
            nc.write( power_trace.subset_slowest_dimension(0,num_traces-1) , "power_trace" );
          }
          nc.end_indep_data();
        }
        auto i_beg = coupler.get_i_beg();
        auto j_beg = coupler.get_j_beg();
        std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg,0};
        nc.write_all(coupler.get_data_manager_readonly().get<real const,4>("windmill_prop"),"windmill_prop",start_3d);
      });
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
      auto wvel           = dm.get<real      ,4>("wvel"       );
      auto tke            = dm.get<real      ,4>("TKE"        );
      auto tend_u         = uvel.createDeviceObject();
      auto tend_v         = vvel.createDeviceObject();
      auto tend_w         = wvel.createDeviceObject();
      auto tend_tke       = tke .createDeviceObject();
      real4d windmill_power("windmill_power",nens,nz,ny,nx);
      real rad = coupler.get_option<real>("blade_radius");

      auto ref_umag     = coupler.get_data_manager_readonly().get<real const,1>("turbine_velmag"     );
      auto ref_thr_coef = coupler.get_data_manager_readonly().get<real const,1>("turbine_thrust_coef");
      auto ref_pow_coef = coupler.get_data_manager_readonly().get<real const,1>("turbine_power_coef" );
      auto ref_pow      = coupler.get_data_manager_readonly().get<real const,1>("turbine_power"      );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        real r     = rho_d(k,j,i,iens);
        real u     = uvel (k,j,i,iens);
        real v     = vvel (k,j,i,iens);
        real w     = wvel (k,j,i,iens);
        real mag   = std::sqrt(u*u + v*v + w*w);
        real C_T;
        real a = 0.3;
        // Iterate to ensure we get a converged estimate of C_T and a since they depend on each other
        for (int iter = 0; iter < 10; iter++) {
          C_T = interp( ref_umag , ref_thr_coef , mag/(1-a) );
          a   = 0.5_fp * ( 1 - std::sqrt(1-C_T) );
        }
        real C_P   = interp( ref_umag , ref_pow_coef , mag/(1-a) );
        real pwr   = interp( ref_umag , ref_pow      , mag/(1-a) );
        real f_TKE = 0.25_fp; // Recommended by Archer et al., 2020, MWR "Two corrections for TKE ..."
        real C_TKE = f_TKE * (C_T - C_P);
        real mag0  = mag/(1-a);
        real u0    = u  /(1-a);
        real v0    = v  /(1-a);
        real w0    = w  /(1-a);
        tend_u        (k,j,i,iens) = -0.5_fp  *C_T  *mag0*u0       *windmill_prop(k,j,i,iens);
        tend_v        (k,j,i,iens) = -0.5_fp  *C_T  *mag0*v0       *windmill_prop(k,j,i,iens);
        tend_w        (k,j,i,iens) = -0.5_fp  *C_T  *mag0*w0       *windmill_prop(k,j,i,iens);
        tend_tke      (k,j,i,iens) =  0.5_fp*r*C_TKE*mag0*mag0*mag0*windmill_prop(k,j,i,iens);
        windmill_power(iens,k,j,i) = pwr/(M_PI*rad*rad)*dx*dy*dz   *windmill_prop(k,j,i,iens);
      });

      // Save the total power produced on the grid
      realHost1d power_loc("power_loc",nens);
      for (int iens = 0; iens < nens; iens++) {
        power_loc(iens) = yakl::intrinsics::sum(windmill_power.slice<3>(iens,0,0,0));
      }
      auto power_glob = power_loc.createHostObject();
      auto dtype = coupler.get_mpi_data_type();
      MPI_Reduce( power_loc.data() , power_glob.data() , nens , dtype , MPI_SUM , 0 , MPI_COMM_WORLD );
      if (coupler.get_myrank() == 0) {
        for (int iens=0; iens < nens; iens++) {
          power_trace(num_traces,iens) = power_glob(iens);
          num_traces++;
        }
      }

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        uvel(k,j,i,iens) += dt * tend_u  (k,j,i,iens);
        vvel(k,j,i,iens) += dt * tend_v  (k,j,i,iens);
        wvel(k,j,i,iens) += dt * tend_w  (k,j,i,iens);
        tke (k,j,i,iens) += dt * tend_tke(k,j,i,iens);
      });
    }


    YAKL_INLINE static real interp( realConst1d const &ref_umag , realConst1d const &ref_var , real umag ) {
      int imax = ref_umag.extent(0)-1; // Max index for the table
      // If umag exceeds the bounds of the reference data, the turbine is idle and producing no power
      if ( umag < ref_umag(0) || umag > ref_umag(imax) ) return 0;
      // Find the index such that umag lies between ref_umag(i) and ref_umag(i+1)
      int i = 0;
      while (umag > ref_umag(i)) { i++; }
      if (i > 0) i--;
      // Linear interpolation: higher weight for left if it's closer to left
      real fac = (ref_umag(i+1) - umag) / (ref_umag(i+1)-ref_umag(i));
      return fac*ref_var(i) + (1-fac)*ref_var(i+1);
    }

  };

}

