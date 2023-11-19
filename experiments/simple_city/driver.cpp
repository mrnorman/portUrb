
#include "coupler.h"
#include "dynamics_ader.h"
#include "horizontal_sponge.h"
#include "time_averager.h"
#include "sponge_layer.h"
#include "sc_init.h"
#include "sc_output.h"

int main(int argc, char** argv) {
  MPI_Init( &argc , &argv );
  yakl::init();
  {
    using yakl::intrinsics::abs;
    using yakl::intrinsics::maxval;
    yakl::timer_start("main");

    // This holds all of the model's variables, dimension sizes, and options
    core::Coupler coupler;

    // Read the YAML input file for variables pertinent to running the driver
    if (argc <= 1) { endrun("ERROR: Must pass the input YAML filename as a parameter"); }
    std::string inFile(argv[1]);
    YAML::Node config = YAML::LoadFile(inFile);
    if ( !config            ) { endrun("ERROR: Invalid YAML input file"); }
    auto sim_time  = config["sim_time"].as<real>();
    auto nens      = config["nens"    ].as<int>();
    auto nx_glob   = config["nx_glob" ].as<size_t>();
    auto ny_glob   = config["ny_glob" ].as<size_t>();
    auto nz        = config["nz"      ].as<int>();
    auto xlen      = config["xlen"    ].as<real>();
    auto ylen      = config["ylen"    ].as<real>();
    auto zlen      = config["zlen"    ].as<real>();
    auto dtphys_in = config["dt_phys" ].as<real>();
    auto out_freq  = config["out_freq"].as<real>();

    coupler.set_option<std::string>( "out_prefix"      , config["out_prefix"      ].as<std::string>() );
    coupler.set_option<std::string>( "init_data"       , config["init_data"       ].as<std::string>() );
    coupler.set_option<real       >( "out_freq"        , config["out_freq"        ].as<real       >() );
    coupler.set_option<bool       >( "enable_gravity"  , config["enable_gravity"  ].as<bool       >(true));
    coupler.set_option<bool       >( "file_per_process", config["file_per_process"].as<bool       >(false));

    // Coupler state is: (1) dry density;  (2) u-velocity;  (3) v-velocity;  (4) w-velocity;  (5) temperature
    //                   (6+) tracer masses (*not* mixing ratios!)
    coupler.distribute_mpi_and_allocate_coupled_state(nz, ny_glob, nx_glob, nens);

    // Just tells the coupler how big the domain is in each dimensions
    coupler.set_grid( xlen , ylen , zlen );

    // This is for the dycore to pull out to determine how to do idealized test cases
    coupler.set_option<std::string>( "standalone_input_file" , inFile );

    // They dynamical core "dycore" integrates the Euler equations and performans transport of tracers
    modules::Dynamics_Euler_Stratified_WenoFV  dycore;
    custom_modules::Horizontal_Sponge          horiz_sponge;
    custom_modules::Time_Averager              time_averager;

    coupler.add_tracer("water_vapor","water_vapor",true,true ,true);
    coupler.add_tracer("pollution1" , ""          ,true,false,true);
    coupler.get_data_manager_readwrite().get<real,4>("water_vapor") = 0;
    coupler.get_data_manager_readwrite().get<real,4>("pollution1" ) = 0;

    real etime = 0;   // Elapsed time
    int num_out = 0;
    int file_counter = 0;

    custom_modules::sc_init  ( coupler );
    custom_modules::sc_output( coupler , etime , file_counter );

    // Run the initialization modules
    dycore       .init( coupler ); // Dycore should initialize its own state here
    horiz_sponge .init( coupler , 10 , 1. );
    time_averager.init( coupler );

    real dtphys = dtphys_in;
    while (etime < sim_time) {
      // If dtphys <= 0, then set it to the dynamical core's max stable time step
      if (dtphys_in <= 0.) { dtphys = dycore.compute_time_step(coupler); }
      // If we're about to go past the final time, then limit to time step to exactly hit the final time
      if (etime + dtphys > sim_time) { dtphys = sim_time - etime; }

      horiz_sponge.apply      ( coupler , dtphys , true , true , false , false );
      dycore.time_step        ( coupler , dtphys );  // Move the flow forward according to the Euler equations
      modules::sponge_layer   ( coupler , dtphys , 1 );
      time_averager.accumulate( coupler , dtphys );

      etime += dtphys; // Advance elapsed time

      if (out_freq >= 0. && etime / out_freq >= num_out+1) {
        custom_modules::sc_output( coupler , etime , file_counter );
        num_out++;
        // Let the user know what the max vertical velocity is to ensure the model hasn't crashed
        auto &dm = coupler.get_data_manager_readonly();
        auto u = dm.get_collapsed<real const>("uvel");
        auto v = dm.get_collapsed<real const>("vvel");
        auto w = dm.get_collapsed<real const>("wvel");
        auto mag = u.createDeviceObject();
        yakl::c::parallel_for( YAKL_AUTO_LABEL() , mag.size() , YAKL_LAMBDA (int i) {
          mag(i) = std::sqrt( u(i)*u(i) + v(i)*v(i) + w(i)*w(i) );
        });
        real wind_mag_loc = yakl::intrinsics::maxval(mag);
        real wind_mag;
        auto mpi_data_type = coupler.get_mpi_data_type();
        MPI_Reduce( &wind_mag_loc , &wind_mag , 1 , mpi_data_type , MPI_MAX , 0 , MPI_COMM_WORLD );
        if (coupler.is_mainproc()) {
          std::cout << "Etime , dtphys, wind_mag: " << std::scientific << std::setw(10) << etime    << " , " 
                                                    << std::scientific << std::setw(10) << dtphys   << " , "
                                                    << std::scientific << std::setw(10) << wind_mag << std::endl;
        }
      }
    }

    time_averager.finalize( coupler );

    yakl::timer_stop("main");
  }
  yakl::finalize();
  MPI_Finalize();
}


