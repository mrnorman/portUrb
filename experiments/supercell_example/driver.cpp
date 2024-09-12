
#include "coupler.h"
#include "dynamics_rk.h"
#include "time_averager.h"
#include "sc_init.h"
#include "les_closure.h"
#include "microphysics_kessler.h"
#include "surface_flux.h"
#include "sponge_layer.h"
#include "perturb_temperature.h"
#include "column_nudging.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

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

    // Optional YAML entries
    auto out_freq     = config["out_freq"    ].as<real       >(sim_time/10. );
    auto inform_freq  = config["inform_freq" ].as<real       >(sim_time/100.);
    auto out_prefix   = config["out_prefix"  ].as<std::string>("test"       );
    auto is_restart   = config["is_restart"  ].as<bool       >(false        );
    auto restart_file = config["restart_file"].as<std::string>(""           );
    auto roughness    = config["roughness"   ].as<real       >(0.1          );
    auto use_weno     = config["use_weno"    ].as<bool       >(true         );

    // Things the coupler might need to know about
    coupler.set_option<std::string>( "out_prefix"   , out_prefix   );
    coupler.set_option<real       >( "out_freq"     , out_freq     );
    coupler.set_option<bool       >( "is_restart"   , is_restart   );
    coupler.set_option<bool       >( "use_weno"     , use_weno     );
    coupler.set_option<std::string>( "restart_file" , restart_file );
    coupler.set_option<real       >( "roughness"    , roughness    );
    coupler.set_option<real       >( "kinematic_viscosity" , config["kinematic_viscosity"].as<real>(0) );
    coupler.set_option<bool       >( "dns" , config["dns"].as<real>(false) );

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!)
    coupler.distribute_mpi_and_allocate_coupled_state( core::ParallelComm(MPI_COMM_WORLD) , nz, ny_glob, nx_glob);

    // Just tells the coupler how big the domain is in each dimensions
    coupler.set_grid( xlen , ylen , zlen );

    // This is for the dycore to pull out to determine how to do idealized test cases
    coupler.set_option<std::string>( "standalone_input_file" , inFile );

    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    custom_modules::Time_Averager              time_averager;
    modules::LES_Closure                       les_closure;
    modules::ColumnNudger                      column_nudger;
    modules::Microphysics_Kessler              micro;

    // Run the initialization modules
    micro        .init          ( coupler );
    custom_modules::sc_init     ( coupler );
    les_closure  .init          ( coupler );
    dycore       .init          ( coupler ); // Dycore should initialize its own state here
    column_nudger.set_column    ( coupler , {"temp","density_dry"} );
    time_averager.init          ( coupler );
    modules::perturb_temperature( coupler , true , false );

    real etime = coupler.get_option<real>("elapsed_time");
    core::Counter output_counter( out_freq    , etime );
    core::Counter inform_counter( inform_freq , etime );

    // if restart, overwrite with restart data, and set the counters appropriately. Otherwise, write initial output
    if (is_restart) {
      coupler.overwrite_with_restart();
      etime = coupler.get_option<real>("elapsed_time");
      output_counter = core::Counter( out_freq    , etime-((int)(etime/out_freq   ))*out_freq    );
      inform_counter = core::Counter( inform_freq , etime-((int)(etime/inform_freq))*inform_freq );
    } else {
      coupler.write_output_file( out_prefix );
    }

    // Begin main simulation loop over time steps
    real dt = dtphys_in;
    yakl::fence();
    auto tm = std::chrono::high_resolution_clock::now();
    while (etime < sim_time*(1+1.e-8)) {
      // If dt <= 0, then set it to the dynamical core's max stable time step
      if (dtphys_in <= 0.) { dt = dycore.compute_time_step(coupler)*dyn_cycle; }
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dt > sim_time*(1+1.e-8)) { dt = std::max(0.,sim_time*(1+1.e-8) - etime); }

      // Run modules
      {
        using core::Coupler;
        auto run_nudger    = [&] (Coupler &coupler) { column_nudger.nudge_to_column(coupler,dt,dt*10); };
        auto run_sponge    = [&] (Coupler &coupler) { modules::sponge_layer(coupler,dt,dt*10,0.1);     };
        auto run_micro     = [&] (Coupler &coupler) { micro.time_step              (coupler,dt);       };
        auto run_dycore    = [&] (Coupler &coupler) { dycore.time_step     (coupler,dt);               };
        auto run_surf_flux = [&] (Coupler &coupler) { modules::apply_surface_fluxes(coupler,dt);       };
        auto run_les       = [&] (Coupler &coupler) { les_closure.apply            (coupler,dt);       };
        auto run_tavg      = [&] (Coupler &coupler) { time_averager.accumulate     (coupler,dt);       };
        coupler.run_module( run_nudger    , "column_nudger"  );
        coupler.run_module( run_sponge    , "sponge_layer"   );
        coupler.run_module( run_micro     , "microphysics"   );
        coupler.run_module( run_dycore    , "dycore"         );
        coupler.run_module( run_surf_flux , "surface_fluxes" );
        coupler.run_module( run_les       , "les_closure"    );
        coupler.run_module( run_tavg      , "time_averager"  );
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
        output_counter.reset();
      }
    } // End main simulation loop

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}


