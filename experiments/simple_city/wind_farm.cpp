
#include "coupler.h"
#include "dynamics_rk.h"
#include "time_averager.h"
#include "sc_init.h"
#include "les_closure.h"
#include "windmill_actuators_yaw.h"
#include "surface_flux.h"
#include "perturb_temperature.h"
#include "precursor_sponge.h"
#include "sponge_layer.h"
#include "uniform_pg_wind_forcing.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler_main;
    core::Coupler coupler_prec;

    // Read the YAML input file for variables pertinent to running the driver
    if (argc <= 1) { endrun("ERROR: Must pass the input YAML filename as a parameter"); }
    std::string inFile(argv[1]);
    YAML::Node config = YAML::LoadFile(inFile);
    if ( !config ) { endrun("ERROR: Invalid YAML input file"); }
    // Required YAML entries
    auto sim_time     = config["sim_time"    ].as<real       >();
    auto nx_glob      = config["nx_glob"     ].as<int        >();
    auto ny_glob      = config["ny_glob"     ].as<int        >();
    auto nz           = config["nz"          ].as<int        >();
    auto xlen         = config["xlen"        ].as<real       >();
    auto ylen         = config["ylen"        ].as<real       >();
    auto zlen         = config["zlen"        ].as<real       >();
    auto dtphys_in    = config["dt_phys"     ].as<real       >();
    auto dyn_cycle    = config["dyn_cycle"   ].as<int        >(1);
    auto init_data    = config["init_data"   ].as<std::string>();
    // Optional YAML entries
    auto out_freq          = config["out_freq"              ].as<real       >(sim_time/10. );
    auto inform_freq       = config["inform_freq"           ].as<real       >(sim_time/100.);
    auto out_prefix        = config["out_prefix"            ].as<std::string>("test"       );
    auto out_prefix_prec   = config["out_prefix_precursor"  ].as<std::string>(out_prefix+std::string("_precursor"));
    auto restart_file      = config["restart_file"          ].as<std::string>(""           );
    auto restart_file_prec = config["restart_file_precursor"].as<std::string>(""           );
    auto run_main          = config["run_main"              ].as<bool       >(true         );

    // Things the coupler might need to know about
    coupler_main.set_option<std::string>( "standalone_input_file"  , inFile            );
    coupler_main.set_option<std::string>( "out_prefix"             , out_prefix        );
    coupler_main.set_option<std::string>( "init_data"              , init_data         );
    coupler_main.set_option<real       >( "out_freq"               , out_freq          );
    coupler_main.set_option<std::string>( "restart_file"           , restart_file      );
    coupler_main.set_option<std::string>( "restart_file_precursor" , restart_file_prec );
    coupler_main.set_option<real       >( "latitude"               , config["latitude"    ].as<real       >(0  ) );
    coupler_main.set_option<real       >( "roughness"              , config["roughness"   ].as<real       >(0.1) );
    coupler_main.set_option<std::string>( "turbine_file"           , config["turbine_file"].as<std::string>()    );
    coupler_main.set_option<bool       >( "turbine_do_blades"        , false );
    coupler_main.set_option<real       >( "turbine_initial_yaw"      , 0     );
    coupler_main.set_option<bool       >( "turbine_fixed_yaw"        , true  );
    coupler_main.set_option<bool       >( "turbine_floating_motions" , false );
    coupler_main.set_option<bool       >( "weno_all"                 , true  );

    coupler_main.set_option<std::vector<real>>("turbine_x_locs",{0.3_fp*xlen});
    coupler_main.set_option<std::vector<real>>("turbine_y_locs",{0.5_fp*ylen});

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!); and Option elapsed_time init to zero
    coupler_main.distribute_mpi_and_allocate_coupled_state( core::ParallelComm(MPI_COMM_WORLD) , nz, ny_glob, nx_glob);

    // Just tells the coupler how big the domain is in each dimensions
    coupler_main.set_grid( xlen , ylen , zlen );

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler_main.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler_main.get_data_manager_readwrite().get<real,3>("water_vapor") = 0;

    // Classes that can work on multiple couplers without issue (no internal state)
    modules::LES_Closure                       les_closure;
    modules::Dynamics_Euler_Stratified_WenoFV  dycore;

    // Classes working on coupler_main
    custom_modules::Time_Averager              time_averager_main;
    custom_modules::Time_Averager              time_averager_prec;
    modules::WindmillActuators                 windmills;

    // Run the initialization modules on coupler_main
    custom_modules::sc_init     ( coupler_main );
    les_closure  .init          ( coupler_main );
    dycore       .init          ( coupler_main );
    modules::perturb_temperature( coupler_main , nz);

    /////////////////////////////////////////////////////////////////////////
    // Everything previous to this is now replicated in coupler_precursor
    // From here out, the will be treated separately
    coupler_main.clone_into(coupler_prec);
    /////////////////////////////////////////////////////////////////////////

    windmills         .init( coupler_main );
    time_averager_main.init( coupler_main );

    time_averager_prec.init ( coupler_prec );

    // Get elapsed time (zero), and create counters for output and informing the user in stdout
    real etime = coupler_main.get_option<real>("elapsed_time");
    core::Counter output_counter( out_freq    , etime );
    core::Counter inform_counter( inform_freq , etime );

    // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
    if (restart_file != "" && restart_file != "null") {
      coupler_main.overwrite_with_restart();
      etime = coupler_main.get_option<real>("elapsed_time");
      output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
      inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
    } else {
      if (out_freq >= 0 && run_main) coupler_main.write_output_file( out_prefix );
    }

    // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
    if (restart_file_prec != "" && restart_file_prec != "null") {
      coupler_prec.set_option<std::string>("restart_file",restart_file_prec);
      coupler_prec.overwrite_with_restart();
      auto &dm_prec = coupler_prec.get_data_manager_readonly();
      auto &dm_main = coupler_main     .get_data_manager_readwrite();
      dm_prec.get<real const,3>("density_dry").deep_copy_to(dm_main.get<real,3>("density_dry"));
      dm_prec.get<real const,3>("uvel"       ).deep_copy_to(dm_main.get<real,3>("uvel"       ));
      dm_prec.get<real const,3>("vvel"       ).deep_copy_to(dm_main.get<real,3>("vvel"       ));
      dm_prec.get<real const,3>("wvel"       ).deep_copy_to(dm_main.get<real,3>("wvel"       ));
      dm_prec.get<real const,3>("temp"       ).deep_copy_to(dm_main.get<real,3>("temp"       ));
      dm_prec.get<real const,3>("TKE"        ).deep_copy_to(dm_main.get<real,3>("TKE"        ));
    } else {
      if (out_freq >= 0) coupler_prec.write_output_file( out_prefix_prec );
    }

    // Begin main simulation loop over time steps
    real dt = dtphys_in;
    while (etime < sim_time) {
      // If dt <= 0, then set it to the dynamical core's max stable time step
      if (dtphys_in <= 0.) { dt = dycore.compute_time_step(coupler_main)*dyn_cycle; }
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dt > sim_time) { dt = sim_time - etime; }

      // Run modules
      {
        using core::Coupler;
        if (run_main) {
          custom_modules::precursor_sponge( coupler_main , coupler_prec , dt , dt*100 ,
                                            {"density_dry","uvel","vvel","wvel","temp"} ,
                                            (int) (0.1*nx_glob) , (int) (0.1*nx_glob) ,
                                            (int) (0.1*ny_glob) , (int) (0.1*ny_glob) );
          // coupler_prec.run_module( [&] (Coupler &c) { modules::sponge_layer        (c,dt,dt*100,10); } , "sponge"            );
          coupler_main.run_module( [&] (Coupler &c) { dycore.time_step             (c,dt);           } , "dycore"            );
          coupler_main.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes(c,dt);           } , "surface_fluxes"    );
          coupler_main.run_module( [&] (Coupler &c) { windmills.apply              (c,dt);           } , "windmillactuators" );
          coupler_main.run_module( [&] (Coupler &c) { les_closure.apply            (c,dt);           } , "les_closure"       );
          coupler_main.run_module( [&] (Coupler &c) { time_averager_main.accumulate(c,dt);           } , "time_averager"     );
        }

        using modules::uniform_pg_wind_forcing_height;
        real h = 89;
        real u = 6.27*std::cos(4.33/180*M_PI);
        real v = 6.27*std::sin(4.33/180*M_PI);
        coupler_prec.run_module( [&] (Coupler &c) { uniform_pg_wind_forcing_height(c,dt,h,u,v);     } , "pg_forcing"     );
        // coupler_prec.run_module( [&] (Coupler &c) { modules::sponge_layer         (c,dt,dt*100,10); } , "sponge"         );
        coupler_prec.run_module( [&] (Coupler &c) { dycore.time_step              (c,dt);           } , "dycore"         );
        coupler_prec.run_module( [&] (Coupler &c) { modules::apply_surface_fluxes (c,dt);           } , "surface_fluxes" );
        coupler_prec.run_module( [&] (Coupler &c) { les_closure.apply             (c,dt);           } , "les_closure"    );
        coupler_prec.run_module( [&] (Coupler &c) { time_averager_prec.accumulate (c,dt);           } , "time_averager"  );
      }

      // Update time step
      etime += dt; // Advance elapsed time
      coupler_main.set_option<real>("elapsed_time",etime);
      coupler_prec.set_option<real>("elapsed_time",etime);
      if (inform_freq >= 0. && inform_counter.update_and_check(dt)) {
        if (run_main) { coupler_main.inform_user(); }
        else          { coupler_prec.inform_user(); }
        inform_counter.reset();
      }
      if (out_freq    >= 0. && output_counter.update_and_check(dt)) {
        if (run_main) coupler_main.write_output_file( out_prefix , true );
        coupler_prec.write_output_file( out_prefix_prec , true );
        time_averager_main.reset(coupler_main);
        time_averager_prec.reset(coupler_prec);
        output_counter.reset();
      }
    } // End main simulation loop

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}

