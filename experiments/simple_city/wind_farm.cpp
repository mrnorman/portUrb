
#include "coupler.h"
#include "dynamics_rk.h"
#include "time_averager.h"
#include "sc_init.h"
#include "les_closure.h"
#include "windmill_actuators_yaw.h"
#include "surface_flux.h"
#include "column_nudging.h"
#include "perturb_temperature.h"
#include "EdgeSponge.h"
#include "sponge_layer.h"

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
    auto init_data    = config["init_data"   ].as<std::string>();
    // Optional YAML entries
    auto nens         = config["nens"        ].as<int        >(1            );
    auto out_freq     = config["out_freq"    ].as<real       >(sim_time/10. );
    auto inform_freq  = config["inform_freq" ].as<real       >(sim_time/100.);
    auto out_prefix   = config["out_prefix"  ].as<std::string>("test"       );
    auto is_restart   = config["is_restart"  ].as<bool       >(false        );

    // Things the coupler might need to know about
    coupler.set_option<std::string>( "standalone_input_file" , inFile       );
    coupler.set_option<std::string>( "out_prefix"            , out_prefix   );
    coupler.set_option<std::string>( "init_data"             , init_data    );
    coupler.set_option<real       >( "out_freq"              , out_freq     );
    coupler.set_option<bool       >( "is_restart"            , is_restart   );
    coupler.set_option<std::string>( "restart_file"          , config["restart_file"].as<std::string>("" ) );
    coupler.set_option<real       >( "latitude"              , config["latitude"    ].as<real       >(0  ) );
    coupler.set_option<real       >( "roughness"             , config["roughness"   ].as<real       >(0.1) );
    coupler.set_option<std::string>( "turbine_file"          , config["turbine_file"].as<std::string>()    );

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!); and Option elapsed_time init to zero
    coupler.distribute_mpi_and_allocate_coupled_state(nz, ny_glob, nx_glob, nens);

    // Just tells the coupler how big the domain is in each dimensions
    coupler.set_grid( xlen , ylen , zlen );

    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    custom_modules::Time_Averager              time_averager;
    modules::LES_Closure                       les_closure;
    modules::WindmillActuators                 windmills;
    modules::ColumnNudger                      column_nudger;
    custom_modules::EdgeSponge                 edge_sponge;

    // No microphysics specified, so create a water_vapor tracer required by the dycore
    coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler.get_data_manager_readwrite().get<real,4>("water_vapor") = 0;

    // Run the initialization modules
    custom_modules::sc_init     ( coupler );
    les_closure  .init          ( coupler );
    windmills    .init          ( coupler );
    dycore       .init          ( coupler ); // Dycore should initialize its own state here
    column_nudger.set_column    ( coupler , {"uvel","vvel"} );
    time_averager.init          ( coupler );
    edge_sponge  .init          ( coupler );
    modules::perturb_temperature( coupler , false , true );

    // Get elapsed time (zero), and create counters for output and informing the user in stdout
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
      if (out_freq >= 0) coupler.write_output_file( out_prefix );
    }

    // Begin main simulation loop over time steps
    real dtphys = dtphys_in;
    while (etime < sim_time) {
      // If dtphys <= 0, then set it to the dynamical core's max stable time step
      if (dtphys_in <= 0.) { dtphys = dycore.compute_time_step(coupler)*dyn_cycle; }
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dtphys > sim_time) { dtphys = sim_time - etime; }

      // Run modules
      {
        using core::Coupler;
        coupler.run_module( [&] (Coupler &coupler) { column_nudger.nudge_to_column(coupler,dtphys,dtphys*100); } , "column_nudger"     );
        coupler.run_module( [&] (Coupler &coupler) { dycore.time_step             (coupler,dtphys);            } , "dycore"            );
        coupler.run_module( [&] (Coupler &coupler) { modules::apply_surface_fluxes(coupler,dtphys);            } , "surface_fluxes"    );
        coupler.run_module( [&] (Coupler &coupler) { windmills.apply              (coupler,dtphys);            } , "windmillactuators" );
        coupler.run_module( [&] (Coupler &coupler) { les_closure.apply            (coupler,dtphys);            } , "les_closure"       );
        coupler.run_module( [&] (Coupler &coupler) { time_averager.accumulate     (coupler,dtphys);            } , "time_averager"     );
      }

      // Update time step
      etime += dtphys; // Advance elapsed time
      coupler.set_option<real>("elapsed_time",etime);
      if (inform_freq >= 0. && inform_counter.update_and_check(dtphys)) {
        coupler.inform_user();
        inform_counter.reset();
      }
      if (out_freq    >= 0. && output_counter.update_and_check(dtphys)) {
        coupler.write_output_file( out_prefix , true );
        output_counter.reset();
      }
    } // End main simulation loop

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}

