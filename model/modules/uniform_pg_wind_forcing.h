
#include "main_header.h"

namespace modules {

  inline void uniform_pg_wind_forcing( core::Coupler &coupler , real dt , real tau = 60 , size_t update_cycles = 20 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nz      = coupler.get_nz();
    auto ny      = coupler.get_ny();
    auto nx      = coupler.get_nx();
    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto &dm     = coupler.get_data_manager_readwrite();
    auto uvel    = dm.get<real,3>("uvel");
    auto vvel    = dm.get<real,3>("vvel");
    auto counter = coupler.get_option<size_t>("uniform_pg_wind_forcing_counter",0);
    if (counter%update_cycles == 0) {
      SArray<real,1,2> u_v;
      u_v(0) = yakl::intrinsics::sum(uvel);
      u_v(1) = yakl::intrinsics::sum(vvel);
      u_v = coupler.get_parallel_comm().all_reduce( u_v , MPI_SUM , "uniform_pg_allreduce" );
      coupler.set_option<real>("uniform_pg_wind_forcing_u",u_v(0)/(nz*ny_glob*nx_glob));
      coupler.set_option<real>("uniform_pg_wind_forcing_v",u_v(1)/(nz*ny_glob*nx_glob));
      if (counter == 0) {
        coupler.set_option<real>("uniform_pg_wind_forcing_u0",u_v(0)/(nz*ny_glob*nx_glob));
        coupler.set_option<real>("uniform_pg_wind_forcing_v0",u_v(1)/(nz*ny_glob*nx_glob));
      }
    }
    auto u0 = coupler.get_option<real>("uniform_pg_wind_forcing_u0");
    auto v0 = coupler.get_option<real>("uniform_pg_wind_forcing_v0");
    auto u  = coupler.get_option<real>("uniform_pg_wind_forcing_u");
    auto v  = coupler.get_option<real>("uniform_pg_wind_forcing_v");
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += dt/tau*(u0-u);
      vvel(k,j,i) += dt/tau*(v0-v);
    });
    counter++;
    coupler.set_option<size_t>("uniform_pg_wind_forcing_counter",counter);
  }

}

