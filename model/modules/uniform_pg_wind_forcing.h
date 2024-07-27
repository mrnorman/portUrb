
#include "main_header.h"

namespace modules {

  inline void uniform_pg_wind_forcing( core::Coupler &coupler , real dt , real tau = 10 , size_t update_cycles = 1 ) {
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


  inline void uniform_pg_wind_forcing_height( core::Coupler &coupler , real dt , real height , real u0 , real v0 ,
                                              real tau = 10 ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto dz      = coupler.get_dz();
    auto nz      = coupler.get_nz();
    auto ny      = coupler.get_ny();
    auto nx      = coupler.get_nx();
    auto ny_glob = coupler.get_ny_glob();
    auto nx_glob = coupler.get_nx_glob();
    auto &dm     = coupler.get_data_manager_readwrite();
    auto uvel    = dm.get<real,3>("uvel");
    auto vvel    = dm.get<real,3>("vvel");
    auto counter = coupler.get_option<size_t>("uniform_pg_wind_forcing_counter",0);
    int  k       = (int) std::round(height/dz-0.5);
    SArray<real,1,2> u_v;
    u_v(0) = yakl::intrinsics::sum(uvel.slice<2>(k,0,0));
    u_v(1) = yakl::intrinsics::sum(vvel.slice<2>(k,0,0));
    u_v = coupler.get_parallel_comm().all_reduce( u_v , MPI_SUM , "uniform_pg_allreduce" );
    real u = u_v(0)/(ny_glob*nx_glob);
    real v = u_v(1)/(ny_glob*nx_glob);
    real u_forcing = dt / tau*(u0-u);
    real v_forcing = dt / tau*(v0-v);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      uvel(k,j,i) += u_forcing;
      vvel(k,j,i) += v_forcing;
    });
    counter++;
    coupler.set_option<size_t>("uniform_pg_wind_forcing_counter",counter);
  }

}

