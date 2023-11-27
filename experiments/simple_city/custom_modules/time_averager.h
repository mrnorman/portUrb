#pragma once

#include "coupler.h"

namespace custom_modules {
  
  struct Time_Averager {
    real etime;

    void init( core::Coupler &coupler ) {
      auto nens         = coupler.get_nens();
      auto nx           = coupler.get_nx();
      auto ny           = coupler.get_ny();
      auto nz           = coupler.get_nz();
      auto &dm          = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<real>("min_uvel","",{nz,ny,nx,nens});    dm.get<real,4>("min_uvel") = 1.e10;
      dm.register_and_allocate<real>("min_vvel","",{nz,ny,nx,nens});    dm.get<real,4>("min_vvel") = 1.e10;
      dm.register_and_allocate<real>("min_wvel","",{nz,ny,nx,nens});    dm.get<real,4>("min_wvel") = 1.e10;
      dm.register_and_allocate<real>("avg_uvel","",{nz,ny,nx,nens});    dm.get<real,4>("avg_uvel") = 0;
      dm.register_and_allocate<real>("avg_vvel","",{nz,ny,nx,nens});    dm.get<real,4>("avg_vvel") = 0;
      dm.register_and_allocate<real>("avg_wvel","",{nz,ny,nx,nens});    dm.get<real,4>("avg_wvel") = 0;
      dm.register_and_allocate<real>("max_uvel","",{nz,ny,nx,nens});    dm.get<real,4>("max_uvel") = -1.e10;
      dm.register_and_allocate<real>("max_vvel","",{nz,ny,nx,nens});    dm.get<real,4>("max_vvel") = -1.e10;
      dm.register_and_allocate<real>("max_wvel","",{nz,ny,nx,nens});    dm.get<real,4>("max_wvel") = -1.e10;
      etime = 0.;
    }

    void accumulate( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens = coupler.get_nens();
      auto nx   = coupler.get_nx();
      auto ny   = coupler.get_ny();
      auto nz   = coupler.get_nz();
      auto uvel     = coupler.get_data_manager_readonly ().get<real const,4>("uvel"    );
      auto vvel     = coupler.get_data_manager_readonly ().get<real const,4>("vvel"    );
      auto wvel     = coupler.get_data_manager_readonly ().get<real const,4>("wvel"    );
      auto min_uvel = coupler.get_data_manager_readwrite().get<real      ,4>("min_uvel");
      auto min_vvel = coupler.get_data_manager_readwrite().get<real      ,4>("min_vvel");
      auto min_wvel = coupler.get_data_manager_readwrite().get<real      ,4>("min_wvel");
      auto avg_uvel = coupler.get_data_manager_readwrite().get<real      ,4>("avg_uvel");
      auto avg_vvel = coupler.get_data_manager_readwrite().get<real      ,4>("avg_vvel");
      auto avg_wvel = coupler.get_data_manager_readwrite().get<real      ,4>("avg_wvel");
      auto max_uvel = coupler.get_data_manager_readwrite().get<real      ,4>("max_uvel");
      auto max_vvel = coupler.get_data_manager_readwrite().get<real      ,4>("max_vvel");
      auto max_wvel = coupler.get_data_manager_readwrite().get<real      ,4>("max_wvel");
      double inertia = etime / (etime + dt);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        min_uvel(k,j,i,iens) = std::min( min_uvel(k,j,i,iens) , uvel(k,j,i,iens) );
        min_vvel(k,j,i,iens) = std::min( min_vvel(k,j,i,iens) , vvel(k,j,i,iens) );
        min_wvel(k,j,i,iens) = std::min( min_wvel(k,j,i,iens) , wvel(k,j,i,iens) );
        avg_uvel(k,j,i,iens) = inertia * avg_uvel(k,j,i,iens) + (1-inertia) * uvel(k,j,i,iens);
        avg_vvel(k,j,i,iens) = inertia * avg_vvel(k,j,i,iens) + (1-inertia) * vvel(k,j,i,iens);
        avg_wvel(k,j,i,iens) = inertia * avg_wvel(k,j,i,iens) + (1-inertia) * wvel(k,j,i,iens);
        max_uvel(k,j,i,iens) = std::max( max_uvel(k,j,i,iens) , uvel(k,j,i,iens) );
        max_vvel(k,j,i,iens) = std::max( max_vvel(k,j,i,iens) , vvel(k,j,i,iens) );
        max_wvel(k,j,i,iens) = std::max( max_wvel(k,j,i,iens) , wvel(k,j,i,iens) );
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
      nc.create( "time_averaged_fields.nc" , NC_CLOBBER | NC_64BIT_DATA , info );
      nc.create_dim( "ens" , nens );
      nc.create_dim( "x"   , nx_glob );
      nc.create_dim( "y"   , ny_glob );
      nc.create_dim( "z"   , nz );
      nc.create_var<real>( "min_uvel" , {"z","y","x","ens"} );
      nc.create_var<real>( "min_vvel" , {"z","y","x","ens"} );
      nc.create_var<real>( "min_wvel" , {"z","y","x","ens"} );
      nc.create_var<real>( "avg_uvel" , {"z","y","x","ens"} );
      nc.create_var<real>( "avg_vvel" , {"z","y","x","ens"} );
      nc.create_var<real>( "avg_wvel" , {"z","y","x","ens"} );
      nc.create_var<real>( "max_uvel" , {"z","y","x","ens"} );
      nc.create_var<real>( "max_vvel" , {"z","y","x","ens"} );
      nc.create_var<real>( "max_wvel" , {"z","y","x","ens"} );
      nc.enddef();
      nc.write_all( dm.get<real const,4>("min_uvel").createHostCopy() , "min_uvel" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("min_vvel").createHostCopy() , "min_vvel" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("min_wvel").createHostCopy() , "min_wvel" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("avg_uvel").createHostCopy() , "avg_uvel" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("avg_vvel").createHostCopy() , "avg_vvel" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("avg_wvel").createHostCopy() , "avg_wvel" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("max_uvel").createHostCopy() , "max_uvel" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("max_vvel").createHostCopy() , "max_vvel" , {0,j_beg,i_beg,0} );
      nc.write_all( dm.get<real const,4>("max_wvel").createHostCopy() , "max_wvel" , {0,j_beg,i_beg,0} );
      nc.close();
    }
  };

}


