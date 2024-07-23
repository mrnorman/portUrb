
#include "coupler.h"
#include "dynamics_rk.h"
#include "time_averager.h"
#include "sc_init.h"
#include "les_closure.h"
#include "windmill_actuators_yaw.h"
#include "surface_flux.h"
#include "uniform_pg_wind_forcing.h"
#include "perturb_temperature.h"
#include "precursor_sponge.h"
#include "sponge_layer.h"
#include "Ensembler.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    // Read the YAML input file for variables pertinent to running the driver
    if (argc <= 1) { endrun("ERROR: Must pass the input YAML filename as a parameter"); }
    std::string inFile(argv[1]);
    YAML::Node config = YAML::LoadFile(inFile);
    if ( !config ) { endrun("ERROR: Invalid YAML input file"); }
    // Required YAML entries
    auto sim_time     = config["sim_time"    ].as<real       >();
    auto zlen         = config["zlen"        ].as<real       >();
    auto dtphys_in    = config["dt_phys"     ].as<real       >();
    auto dyn_cycle    = config["dyn_cycle"   ].as<int        >(1);
    auto init_data    = config["init_data"   ].as<std::string>();
    auto out_freq     = config["out_freq"    ].as<real       >(sim_time/10. );
    auto inform_freq  = config["inform_freq" ].as<real       >(sim_time/100.);
    auto out_prefix   = config["out_prefix"  ].as<std::string>("test"       );
    auto restart_file = config["restart_file"].as<std::string>(""           );

    // Things the coupler might need to know about
    core::Coupler coupler;
    coupler.set_option<std::string>( "standalone_input_file"  , inFile            );
    coupler.set_option<std::string>( "out_prefix"             , out_prefix        );
    coupler.set_option<std::string>( "init_data"              , init_data         );
    coupler.set_option<real       >( "out_freq"               , out_freq          );
    coupler.set_option<std::string>( "restart_file"           , restart_file      );
    coupler.set_option<real       >( "roughness"              , config["roughness"   ].as<real       >(0.1) );
    coupler.set_option<std::string>( "turbine_file"           , config["turbine_file"].as<std::string>()    );

    // This holds all of the model's variables, dimension sizes, and options
    core::Ensembler ensembler;

    // Add wind dimension
    ensembler.register_dimension( 3 , [=] (int ind, core::Coupler &coupler) {
      // Set the wind
      real wind = 0;
      if      (ind == 0) { wind =  3; }
      else if (ind == 1) { wind = 15; }
      else if (ind == 2) { wind = 27; }
      coupler.set_option<real>("hub_height_wind_mag",wind);
      ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("wind-")+std::to_string(wind));
      ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("wind-")+std::to_string(wind));
      ensembler.append_coupler_string(coupler,"restart_file"   ,std::string("wind-")+std::to_string(wind));
      return 1;
    });
    // Add turbine distance dimension, each with its own multiplier to the total ranks for that ensemble
    ensembler.register_dimension( 2 , [=] (int ind, core::Coupler &coupler) {
      int rank_mult;
      int diams;
      if      (ind == 0) { diams =  5; rank_mult = 1*1; }
      else if (ind == 1) { diams = 10; rank_mult = 2*2; }
      coupler.set_option<real>("turbine_diameters_apart", (real) diams);
      ensembler.append_coupler_string(coupler,"ensemble_stdout",std::string("diams-")+std::to_string(diams));
      ensembler.append_coupler_string(coupler,"out_prefix"     ,std::string("diams-")+std::to_string(diams));
      ensembler.append_coupler_string(coupler,"restart_file"   ,std::string("diams-")+std::to_string(diams));
      return rank_mult;
    });

    auto par_comm = ensembler.create_coupler_comm( coupler , 4 , MPI_COMM_WORLD );

    auto ostr = std::ofstream(coupler.get_option<std::string>("ensemble_stdout")+std::string(".out"));
    std::cout.rdbuf(ostr.rdbuf());
    std::cerr.rdbuf(ostr.rdbuf());

    if (par_comm.valid()) {
      yakl::timer_start("main");

      std::cout << "Ensemble memeber using a separation of ["
                << coupler.get_option<real>("turbine_diameters_apart")
                << "] turbine diameters" << std::endl;
      std::cout << "Ensemble memeber using an initial hub wind speed of ["
                << coupler.get_option<real>("hub_height_wind_mag")
                << "] m/s" << std::endl;

      YAML::Node turb = YAML::LoadFile(coupler.get_option<std::string>("turbine_file"));
      real turb_diam = turb["blade_radius"].as<real>()*2;
      coupler.set_option<real>("turbine_hub_height",turb["hub_height"].as<real>());
      real xlen = coupler.get_option<real>("turbine_diameters_apart") * turb_diam * 3;
      real ylen = coupler.get_option<real>("turbine_diameters_apart") * turb_diam * 3;
      int nx_glob = (int) std::ceil( xlen / 10. );
      int ny_glob = (int) std::ceil( ylen / 10. );
      int nz      = (int) std::ceil( zlen / 10. );

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

      coupler.distribute_mpi_and_allocate_coupled_state( par_comm , nz , ny_glob , nx_glob );

      coupler.set_grid( xlen , ylen , zlen );

      // No microphysics specified, so create a water_vapor tracer required by the dycore
      coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
      coupler.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

      // Classes that can work on multiple couplers without issue (no internal state)
      modules::LES_Closure                       les_closure;
      modules::Dynamics_Euler_Stratified_WenoFV  dycore;
      custom_modules::Time_Averager              time_averager;
      modules::WindmillActuators                 windmills;

      // Run the initialization modules on coupler
      custom_modules::sc_init     ( coupler );
      les_closure  .init          ( coupler );
      dycore       .init          ( coupler );
      modules::perturb_temperature( coupler , nz);
      windmills    .init          ( coupler );
      time_averager.init          ( coupler );

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
          using modules::uniform_pg_wind_forcing;
          coupler.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing      (c,dt); } , "pg_forcing"        );
          coupler.run_module( [&] (Coupler &c) { dycore.time_step             (c,dt); } , "dycore"            );
          coupler.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes(c,dt); } , "surface_fluxes"    );
          coupler.run_module( [&] (Coupler &c) { windmills.apply              (c,dt); } , "windmillactuators" );
          coupler.run_module( [&] (Coupler &c) { les_closure.apply            (c,dt); } , "les_closure"       );
          coupler.run_module( [&] (Coupler &c) { time_averager.accumulate     (c,dt); } , "time_averager"     );
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

