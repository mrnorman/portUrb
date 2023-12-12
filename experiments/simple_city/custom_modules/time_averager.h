#pragma once

#include "coupler.h"

namespace custom_modules {
  
  struct Time_Averager {
    real etime;
    int  counter;

    void init( core::Coupler &coupler ) {
      auto nens         = coupler.get_nens();
      auto nx           = coupler.get_nx();
      auto ny           = coupler.get_ny();
      auto nz           = coupler.get_nz();
      auto &dm          = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<real>("avg_u","",{nz,ny,nx,nens});    dm.get<real,4>("avg_u") = 0;
      dm.register_and_allocate<real>("avg_v","",{nz,ny,nx,nens});    dm.get<real,4>("avg_v") = 0;
      dm.register_and_allocate<real>("avg_w","",{nz,ny,nx,nens});    dm.get<real,4>("avg_w") = 0;
      dm.register_and_allocate<real>("tke"  ,"",{nz,ny,nx,nens});    dm.get<real,4>("tke"  ) = 0;
      etime   = 0.;
      counter = 1;
    }

    void accumulate( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens = coupler.get_nens();
      auto nx   = coupler.get_nx();
      auto ny   = coupler.get_ny();
      auto nz   = coupler.get_nz();
      auto uvel  = coupler.get_data_manager_readonly ().get<real const,4>("uvel" );
      auto vvel  = coupler.get_data_manager_readonly ().get<real const,4>("vvel" );
      auto wvel  = coupler.get_data_manager_readonly ().get<real const,4>("wvel" );
      auto avg_u = coupler.get_data_manager_readwrite().get<real      ,4>("avg_u");
      auto avg_v = coupler.get_data_manager_readwrite().get<real      ,4>("avg_v");
      auto avg_w = coupler.get_data_manager_readwrite().get<real      ,4>("avg_w");
      auto tke   = coupler.get_data_manager_readwrite().get<real      ,4>("tke"  );
      double inertia = etime / (etime + dt);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        avg_u(k,j,i,iens) = inertia * avg_u(k,j,i,iens) + (1-inertia) * uvel(k,j,i,iens);
        avg_v(k,j,i,iens) = inertia * avg_v(k,j,i,iens) + (1-inertia) * vvel(k,j,i,iens);
        avg_w(k,j,i,iens) = inertia * avg_w(k,j,i,iens) + (1-inertia) * wvel(k,j,i,iens);
        real up = uvel(k,j,i,iens) - avg_u(k,j,i,iens);
        real vp = vvel(k,j,i,iens) - avg_v(k,j,i,iens);
        real wp = wvel(k,j,i,iens) - avg_w(k,j,i,iens);
        tke(k,j,i,iens) = inertia * tke(k,j,i,iens) + (1-inertia) * 0.5_fp*(up*up + vp*vp + wp*wp);
      });
      etime += dt;
    }

    void dump( core::Coupler &coupler ) {
      int  nx_glob = coupler.get_nx_glob();
      int  ny_glob = coupler.get_ny_glob();
      auto nens    = coupler.get_nens();
      auto nz      = coupler.get_nz();
      int  i_beg   = coupler.get_i_beg();
      int  j_beg   = coupler.get_j_beg();
      auto &dm     = coupler.get_data_manager_readonly();
      yakl::SimplePNetCDF nc;
      MPI_Info info;
      MPI_Info_create( &info );
      MPI_Info_set( info , "romio_no_indep_rw"    , "true"    );
      MPI_Info_set( info , "nc_header_align_size" , "1048576" );
      MPI_Info_set( info , "nc_var_align_size"    , "1048576" );
      std::stringstream fname;
      fname << coupler.get_option<std::string>("out_prefix") << "_tavg_" << std::setw(8) << std::setfill('0')
            << counter << ".nc";
      nc.create( fname.str() , NC_CLOBBER | NC_64BIT_DATA , info );
      nc.create_dim( "ens" , nens    );
      nc.create_dim( "x"   , nx_glob );
      nc.create_dim( "y"   , ny_glob );
      nc.create_dim( "z"   , nz      );
      nc.create_var<real>( "avg_u" , {"z","y","x","ens"} );
      nc.create_var<real>( "avg_v" , {"z","y","x","ens"} );
      nc.create_var<real>( "avg_w" , {"z","y","x","ens"} );
      nc.create_var<real>( "tke"   , {"z","y","x","ens"} );
      nc.enddef();
      nc.write_all( dm.get<real const,4>("avg_u").createHostCopy() , "avg_u" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("avg_v").createHostCopy() , "avg_v" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("avg_w").createHostCopy() , "avg_w" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("tke"  ).createHostCopy() , "tke"   , {0,j_beg,i_beg,0} );
      nc.close();
      counter++;
    }
  };

}


