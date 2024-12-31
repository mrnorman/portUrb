#pragma once

#include "coupler.h"

namespace custom_modules {


  inline void vort_write_file( core::Coupler &coupler , float3d const &var , std::string vname , std::string fname ) {
    using yakl::c::parallel_for;
    auto nx    = coupler.get_nx();
    auto ny    = coupler.get_ny();
    auto nz    = coupler.get_nz();
    auto dx    = coupler.get_dx();
    auto dy    = coupler.get_dy();
    auto dz    = coupler.get_dz();
    int  i_beg = coupler.get_i_beg();
    int  j_beg = coupler.get_j_beg();
    yakl::SimplePNetCDF nc(coupler.get_parallel_comm().get_mpi_comm());
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_no_indep_rw",    "true");
    MPI_Info_set(info, "nc_header_align_size", "1048576");
    MPI_Info_set(info, "nc_var_align_size",    "1048576");
    nc.create(fname , NC_CLOBBER | NC_64BIT_DATA , info );
    nc.create_dim( "x" , coupler.get_nx_glob() );
    nc.create_dim( "y" , coupler.get_ny_glob() );
    nc.create_dim( "z" , nz );
    nc.create_var<float>( "x" , {"x"} );
    nc.create_var<float>( "y" , {"y"} );
    nc.create_var<float>( "z" , {"z"} );
    nc.create_var<float>( vname , {"z","y","x"} );
    nc.enddef();
    // x-coordinate
    float1d xloc("xloc",nx);
    parallel_for( YAKL_AUTO_LABEL() , nx , KOKKOS_LAMBDA (int i) { xloc(i) = (i+i_beg+0.5)*dx; });
    nc.write_all( xloc , "x" , {i_beg} );
    // y-coordinate
    float1d yloc("yloc",ny);
    parallel_for( YAKL_AUTO_LABEL() , ny , KOKKOS_LAMBDA (int j) { yloc(j) = (j+j_beg+0.5)*dy; });
    nc.write_all( yloc , "y" , {j_beg} );
    // z-coordinate
    float1d zloc("zloc",nz);
    parallel_for( YAKL_AUTO_LABEL() , nz , KOKKOS_LAMBDA (int k) { zloc(k) = (k      +0.5)*dz; });
    nc.begin_indep_data();
    if (coupler.is_mainproc()) nc.write( zloc , "z" );
    nc.end_indep_data();
    std::vector<MPI_Offset> start_3d = {0,j_beg,i_beg};
    nc.write_all(var,vname,start_3d);
    nc.close();
  }

  
  inline void dump_vorticity( core::Coupler &coupler ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    if (! coupler.option_exists("vorticity_counter")) coupler.set_option<int>("vorticity_counter",0);
    auto counter = coupler.get_option<int>("vorticity_counter");
    auto &dm     = coupler.get_data_manager_readonly();
    auto nx      = coupler.get_nx();
    auto ny      = coupler.get_ny();
    auto nz      = coupler.get_nz();
    auto dx      = coupler.get_dx();
    auto dy      = coupler.get_dy();
    auto dz      = coupler.get_dz();
    auto dm_u    = dm.get<real const,3>("uvel");
    auto dm_v    = dm.get<real const,3>("vvel");
    auto dm_w    = dm.get<real const,3>("wvel");
    if (counter == 0) {
      auto imm = dm.get<real const,3>("immersed_proportion").as<float>();
      vort_write_file( coupler , imm , "immersed_proportion" , "vortmag_dump_immersed.nc" );
    }
    core::MultiField<real const,3> fields;
    fields.add_field( dm_u );
    fields.add_field( dm_v );
    fields.add_field( dm_w );
    int constexpr hs = 1;
    auto fields_halos = coupler.create_and_exchange_halos( fields , hs );
    auto u = fields_halos.get_field(0);
    auto v = fields_halos.get_field(1);
    auto w = fields_halos.get_field(2);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      u(0    ,hs+j,hs+i) = 0;
      v(0    ,hs+j,hs+i) = 0;
      w(0    ,hs+j,hs+i) = 0;
      u(hs+nz,hs+j,hs+i) = u(hs+nz-1,hs+j,hs+i);
      v(hs+nz,hs+j,hs+i) = v(hs+nz-1,hs+j,hs+i);
      w(hs+nz,hs+j,hs+i) = w(hs+nz-1,hs+j,hs+i);
    });
    float3d vortmag("vortmag",nz,ny,nx);
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
      float dw_dy = (w(hs+k  ,hs+j+1,hs+i  )-w(hs+k  ,hs+j-1,hs+i  ))/(2*dy);
      float dv_dz = (v(hs+k+1,hs+j  ,hs+i  )-v(hs+k-1,hs+j  ,hs+i  ))/(2*dz);
      float du_dz = (u(hs+k+1,hs+j  ,hs+i  )-u(hs+k-1,hs+j  ,hs+i  ))/(2*dz);
      float dw_dx = (w(hs+k  ,hs+j  ,hs+i+1)-w(hs+k  ,hs+j  ,hs+i-1))/(2*dx);
      float dv_dx = (v(hs+k  ,hs+j  ,hs+i+1)-v(hs+k  ,hs+j  ,hs+i-1))/(2*dx);
      float du_dy = (u(hs+k  ,hs+j+1,hs+i  )-u(hs+k  ,hs+j-1,hs+i  ))/(2*dy);
      float vx = dw_dy-dv_dz;
      float vy = du_dz-dw_dx;
      float vz = dv_dx-du_dy;
      vortmag(k,j,i) = std::sqrt( vx*vx + vy*vy + vz*vz );
    });
    std::stringstream fname;
    fname << "vortmag_dump_" << std::setw(8) << std::setfill('0') << counter << ".nc";
    vort_write_file( coupler , vortmag , "vortmag" , fname.str() );
    coupler.set_option<int>("vorticity_counter",counter+1);
  }

}

