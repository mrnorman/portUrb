
#pragma once

#include "main_header.h"

namespace custom_modules {

  inline void sc_output( core::Coupler &coupler , real etime , int &file_counter ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    yakl::timer_start("output");

    auto nens        = coupler.get_nens();
    auto nx          = coupler.get_nx();
    auto ny          = coupler.get_ny();
    auto nz          = coupler.get_nz();
    auto dx          = coupler.get_dx();
    auto dy          = coupler.get_dy();
    auto dz          = coupler.get_dz();
    auto num_tracers = coupler.get_num_tracers();
    auto C0          = coupler.get_option<real>("C0");
    auto R_d         = coupler.get_option<real>("R_d");
    auto gamma       = coupler.get_option<real>("gamma_d");
    int i_beg        = coupler.get_i_beg();
    int j_beg        = coupler.get_j_beg();
    int iens = 0;

    yakl::SimplePNetCDF nc;
    MPI_Offset ulIndex = 0; // Unlimited dimension index to place this data at

    std::stringstream fname;
    fname << coupler.get_option<std::string>("out_prefix") << "_" << std::setw(8) << std::setfill('0')
          << file_counter << ".nc";

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_no_indep_rw",    "true");
    MPI_Info_set(info, "nc_header_align_size", "1048576");
    MPI_Info_set(info, "nc_var_align_size",    "1048576");

    nc.create(fname.str() , NC_CLOBBER | NC_64BIT_DATA , info );

    nc.create_dim( "x" , coupler.get_nx_glob() );
    nc.create_dim( "y" , coupler.get_ny_glob() );
    nc.create_dim( "z" , nz );
    nc.create_unlim_dim( "t" );

    nc.create_var<real>( "x" , {"x"} );
    nc.create_var<real>( "y" , {"y"} );
    nc.create_var<real>( "z" , {"z"} );
    nc.create_var<real>( "t" , {"t"} );
    nc.create_var<real>( "density_dry" , {"t","z","y","x"} );
    nc.create_var<real>( "uvel"        , {"t","z","y","x"} );
    nc.create_var<real>( "vvel"        , {"t","z","y","x"} );
    nc.create_var<real>( "wvel"        , {"t","z","y","x"} );
    nc.create_var<real>( "temperature" , {"t","z","y","x"} );
    nc.create_var<real>( "theta"       , {"t","z","y","x"} );
    nc.create_var<real>( "immersed"    , {"t","z","y","x"} );
    auto tracer_names = coupler.get_tracer_names();
    for (int tr = 0; tr < num_tracers; tr++) { nc.create_var<real>( tracer_names[tr] , {"t","z","y","x"} ); }

    nc.enddef();

    // x-coordinate
    real1d xloc("xloc",nx);
    parallel_for( YAKL_AUTO_LABEL() , nx , YAKL_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
    nc.write_all( xloc.createHostCopy() , "x" , {i_beg} );

    // y-coordinate
    real1d yloc("yloc",ny);
    parallel_for( YAKL_AUTO_LABEL() , ny , YAKL_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
    nc.write_all( yloc.createHostCopy() , "y" , {j_beg} );

    // z-coordinate
    real1d zloc("zloc",nz);
    parallel_for( YAKL_AUTO_LABEL() , nz , YAKL_LAMBDA (int k) { zloc(k) = (k      +0.5)*dz; });
    nc.begin_indep_data();
    if (coupler.is_mainproc()) {
      nc.write( zloc.createHostCopy() , "z" );
      nc.write1( 0._fp , "t" , 0 , "t" );
    }
    nc.end_indep_data();

    auto &dm = coupler.get_data_manager_readonly();
    real3d data("data",nz,ny,nx);

    auto immersed_proportion = dm.get<real const,4>("immersed_proportion");
    parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      data(k,j,i) = immersed_proportion(k,j,i,iens);
    });
    nc.write1_all(data.createHostCopy(),"immersed",ulIndex,{0,j_beg,i_beg},"t");

    {
      auto var           = dm.get<real const,4>("density_dry");
      parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        data(k,j,i) = var(k,j,i,iens);
      });
      nc.write1_all(data.createHostCopy(),"density_dry",ulIndex,{0,j_beg,i_beg},"t");
    }
    {
      auto var = dm.get<real const,4>("uvel");
      parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        data(k,j,i) = var(k,j,i,iens);
      });
      nc.write1_all(data.createHostCopy(),"uvel",ulIndex,{0,j_beg,i_beg},"t");
    }
    {
      auto var = dm.get<real const,4>("vvel");
      parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        data(k,j,i) = var(k,j,i,iens);
      });
      nc.write1_all(data.createHostCopy(),"vvel",ulIndex,{0,j_beg,i_beg},"t");
    }
    {
      auto var = dm.get<real const,4>("wvel");
      parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        data(k,j,i) = var(k,j,i,iens);
      });
      nc.write1_all(data.createHostCopy(),"wvel",ulIndex,{0,j_beg,i_beg},"t");
    }
    {
      auto var                 = dm.get<real const,4>("temp"               );
      parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        data(k,j,i) = var(k,j,i,iens);
      });
      nc.write1_all(data.createHostCopy(),"temperature",ulIndex,{0,j_beg,i_beg},"t");
    }

    for (int i=0; i < tracer_names.size(); i++) {
      auto var = dm.get<real const,4>(tracer_names[i]);
      parallel_for( SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) { data(k,j,i) = var(k,j,i,iens); });
      nc.write1_all(data.createHostCopy(),tracer_names[i],ulIndex,{0,j_beg,i_beg},"t");
    }

    nc.close();

    file_counter++;

    yakl::timer_stop("output");
  }

}


