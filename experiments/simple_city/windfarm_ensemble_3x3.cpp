
#include "coupler.h"
#include "dynamics_rk.h"
#include "time_averager.h"
#include "sc_init.h"
#include "les_closure.h"
#include "windmill_actuators_yaw.h"
#include "surface_flux.h"
#include "perturb_temperature.h"
#include "sponge_layer.h"
#include "uniform_pg_wind_forcing.h"
#include "Ensembler.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

    coupler.set_option<std::string>("ensemble_stdout","ensemble" );
    coupler.set_option<std::string>("out_prefix"     ,"turbulent_3x3");

    // This holds all of the model's variables, dimension sizes, and options
    core::Ensembler ensembler;

    // Add wind dimension
    {
      auto func_nranks  = [=] (int ind) { return 1; };
      auto func_coupler = [=] (int ind, core::Coupler &coupler) {
        real wind = (ind+1)*3-1;
        coupler.set_option<real>("hub_height_wind_mag",wind);
        ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("wind-")+std::to_string(wind));
        ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("wind-")+std::to_string(wind));
      };
      ensembler.register_dimension( 9 , func_nranks , func_coupler );
    }

    // Add floating dimension
    {
      auto func_nranks  = [=] (int ind) { return 1; };
      auto func_coupler = [=] (int ind, core::Coupler &coupler) {
        if (ind == 0) {
          coupler.set_option<bool>( "turbine_floating_motions" , false );
          ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("fixed-"));
          ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("fixed-"));
        } else {
          coupler.set_option<bool>( "turbine_floating_motions" , true );
          ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("floating-"));
          ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("floating-"));
        }
      };
      ensembler.register_dimension( 2 , func_nranks , func_coupler );
    }
    // coupler.set_option<bool>( "turbine_floating_motions" , true );

    auto par_comm = ensembler.create_coupler_comm( coupler , 6 , MPI_COMM_WORLD );
    // auto par_comm = ensembler.create_coupler_comm( coupler , 12 , MPI_COMM_WORLD );

    auto ostr = std::ofstream(coupler.get_option<std::string>("ensemble_stdout")+std::string(".out"));
    std::cout.rdbuf(ostr.rdbuf());
    std::cerr.rdbuf(ostr.rdbuf());

    if (par_comm.valid()) {
      yakl::timer_start("main");

      std::cout << "Ensemble memeber using an initial hub wind speed of ["
                << coupler.get_option<real>("hub_height_wind_mag")
                << "] m/s" << std::endl;
      real        sim_time          = 3600*12+1;
      int         nx_glob           = 378;
      int         ny_glob           = 378;
      int         nz                = 60;
      real        xlen              = 3780;
      real        ylen              = 3780;
      real        zlen              = 600;
      real        dtphys_in         = 0.;  // Dycore determined time step size
      int         dyn_cycle         = 10;
      std::string init_data         = "ABL_neutral2";
      real        out_freq          = 1800;
      real        inform_freq       = 10;
      std::string out_prefix        = coupler.get_option<std::string>("out_prefix");
      std::string restart_file      = "";
      coupler.set_option<std::string      >( "init_data"           , init_data         );
      coupler.set_option<real             >( "out_freq"            , out_freq          );
      coupler.set_option<std::string      >( "restart_file"        , restart_file      );
      coupler.set_option<real             >( "latitude"            , 0.                );
      coupler.set_option<real             >( "roughness"           , 0.0002            );
      coupler.set_option<std::string      >( "turbine_file"        , "./inputs/NREL_5MW_126_RWT.yaml" );
      coupler.set_option<bool             >( "turbine_do_blades"   , false );
      coupler.set_option<real             >( "turbine_initial_yaw" , 0     );
      coupler.set_option<bool             >( "turbine_fixed_yaw"   , false );
      coupler.set_option<bool             >( "weno_all"            , true  );
      coupler.set_option<std::vector<real>>("turbine_x_locs",{1.*xlen/6.,
                                                              3.*xlen/6.,
                                                              5.*xlen/6.,
                                                              1.*xlen/6.,
                                                              3.*xlen/6.,
                                                              5.*xlen/6.,
                                                              1.*xlen/6.,
                                                              3.*xlen/6.,
                                                              5.*xlen/6.});

      coupler.set_option<std::vector<real>>("turbine_y_locs",{1.*ylen/6.,
                                                              1.*ylen/6.,
                                                              1.*ylen/6.,
                                                              3.*ylen/6.,
                                                              3.*ylen/6.,
                                                              3.*ylen/6.,
                                                              5.*ylen/6.,
                                                              5.*ylen/6.,
                                                              5.*ylen/6.});
      coupler.set_option<std::vector<bool>>("turbine_apply_thrust",{true,true,true,true,true,true,true,true,true});

      // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
      //                   (6+) tracer masses (*not* mixing ratios!); and Option elapsed_time init to zero
      coupler.distribute_mpi_and_allocate_coupled_state( par_comm , nz, ny_glob, nx_glob);

      // Just tells the coupler how big the domain is in each dimensions
      coupler.set_grid( xlen , ylen , zlen );

      // No microphysics specified, so create a water_vapor tracer required by the dycore
      coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
      coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

      // Classes that can work on multiple couplers without issue (no internal state)
      modules::LES_Closure                       les_closure;
      modules::Dynamics_Euler_Stratified_WenoFV  dycore;

      // Classes working on coupler
      custom_modules::Time_Averager              time_averager;
      modules::WindmillActuators                 windmills;

      // Run the initialization modules on coupler
      custom_modules::sc_init     ( coupler );
      les_closure       .init     ( coupler );
      dycore            .init     ( coupler );
      windmills         .init     ( coupler );
      time_averager     .init     ( coupler );
      modules::perturb_temperature( coupler , nz);

      // Get elapsed time (zero), and create counters for output and informing the user in stdout
      real etime = coupler.get_option<real>("elapsed_time");
      core::Counter output_counter( out_freq    , etime );
      core::Counter inform_counter( inform_freq , etime );

      // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
      if (restart_file != "" && restart_file != "null") {
        coupler.overwrite_with_restart();
        etime = coupler.get_option<real>("elapsed_time");
        output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
        inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
      } else {
        if (out_freq >= 0) coupler.write_output_file( out_prefix );
      }

      // Begin main simulation loop over time steps
      real dt = dtphys_in;
      while (etime < sim_time) {
        // If dt <= 0, then set it to the dynamical core's max stable time step
        if (dtphys_in <= 0.) { dt = dycore.compute_time_step(coupler)*dyn_cycle; }
        // If we're about to go past the final time, then limit to time step to exactly hit the final time
        if (etime + dt > sim_time) { dt = sim_time - etime; }

        // Run modules
        {
          using core::Coupler;
          using modules::uniform_pg_wind_forcing_height;
          real h = 90;
          real u = coupler.get_option<real>("hub_height_wind_mag");
          real v = 0;
          coupler.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_height(c,dt,h,u,v); } , "pg_forcing"        );
          coupler.run_module( [&] (Coupler &c) { dycore.time_step              (c,dt);       } , "dycore"            );
          coupler.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes (c,dt);       } , "surface_fluxes"    );
          coupler.run_module( [&] (Coupler &c) { windmills.apply               (c,dt);       } , "windmillactuators" );
          coupler.run_module( [&] (Coupler &c) { les_closure.apply             (c,dt);       } , "les_closure"       );
          coupler.run_module( [&] (Coupler &c) { time_averager.accumulate      (c,dt);       } , "time_averager"     );
        }

        // Update time step
        etime += dt; // Advance elapsed time
        coupler.set_option<real>("elapsed_time",etime);
        if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
          coupler.inform_user();
          inform_counter.reset();
        }
        if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
          coupler.write_output_file( out_prefix , true );
          time_averager.reset(coupler);
          output_counter.reset();
        }
      } // End main simulation loop
      yakl::timer_stop("main");
    } // if (par_comm.valid())
  }
  yakl::finalize();
  MPI_Finalize();
}

