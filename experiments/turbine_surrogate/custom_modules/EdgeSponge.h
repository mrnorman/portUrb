
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  struct EdgeSponge {

    std::vector<std::string> varnames;
    real3d                   col_avg;


    void init( core::Coupler &coupler , std::vector<std::string> names ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      varnames = names;
      auto nx_glob    = coupler.get_nx_glob();
      auto ny_glob    = coupler.get_ny_glob();
      auto nens       = coupler.get_nens();
      auto nx         = coupler.get_nx();
      auto ny         = coupler.get_ny();
      auto nz         = coupler.get_nz();
      auto num_fields = varnames.size();
      auto &dm        = coupler.get_data_manager_readonly();
      core::MultiField<real const,4> fields;
      for (int i=0; i < num_fields; i++) { fields.add_field( dm.get<real const,4>(varnames[i]) ); }
      col_avg = real3d("col_avg",num_fields,nz,nens);
      col_avg = 0;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        yakl::atomicAdd( col_avg(l,k,iens) , fields(l,k,j,i,iens) );
      });
      #ifdef MW_GPU_AWARE_MPI
        auto column_total = col_avg.createDeviceObject();
        yakl::fence();
        yakl::timer_start("column_nudging_Allreduce");
        MPI_Allreduce( col_avg.data() , column_total.data() , column_total.size() ,
                       coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );
        yakl::timer_stop("column_nudging_Allreduce");
      #else
        yakl::timer_start("column_nudging_Allreduce");
        auto column_total_host = col_avg.createHostObject();
        MPI_Allreduce( col_avg.createHostCopy().data() , column_total_host.data() , column_total_host.size() ,
                       coupler.get_mpi_data_type() , MPI_SUM , MPI_COMM_WORLD );
        auto column_total = column_total_host.createDeviceCopy();
        yakl::timer_stop("column_nudging_Allreduce");
      #endif
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_fields,nz,nens) , YAKL_LAMBDA (int l, int k, int iens) {
        col_avg(l,k,iens) = column_total(l,k,iens) / (nx_glob*ny_glob);
      });
    }


    template <class T>
    void override_var( std::string name , T val ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nz   = col_avg.extent(1);
      auto nens = col_avg.extent(2);
      int varid = -1;
      for (int l=0; l < varnames.size(); l++) {
        if (varnames[l] == name) { varid = l; break; }
      }
      if (varid > 0) {
        YAKL_SCOPE( col_avg , this->col_avg );
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,nens) , YAKL_LAMBDA (int k, int iens) {
          col_avg(varid,k,iens) = val;
        });
      }
    }


    void apply( core::Coupler & coupler    ,
                real            dt         ,
                real            time_scale ,
                int             cells_x1=0 ,
                int             cells_x2=0 ,
                int             cells_y1=0 ,
                int             cells_y2=0  ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens       = coupler.get_nens();
      auto nx         = coupler.get_nx();
      auto ny         = coupler.get_ny();
      auto nz         = coupler.get_nz();
      auto nx_glob    = coupler.get_nx_glob();
      auto ny_glob    = coupler.get_ny_glob();
      auto dtype      = coupler.get_mpi_data_type();
      auto &dm        = coupler.get_data_manager_readwrite();
      int  num_fields = varnames.size();
      core::MultiField<real,4> fields;
      for (int i=0; i < num_fields; i++) { fields.add_field( dm.get<real,4>(varnames[i]) ); }
      real time_factor = dt / time_scale;
      YAKL_SCOPE( col_avg , this->col_avg );

      if (coupler.get_px() == 0 && cells_x1 > 0) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz,ny,cells_x1,nens) ,
                                          YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
          real xloc   = i / (cells_x1-1._fp);
          real weight = (cos(M_PI*xloc)+1)/2;
          weight *= time_factor;
          fields(l,k,j,i,iens) = weight*col_avg(l,k,iens) + (1-weight)*fields(l,k,j,i,iens);
        });
      }
      if (coupler.get_px() == coupler.get_nproc_x()-1 && cells_x2 > 0) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz,ny,cells_x2,nens) ,
                                          YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
          real xloc   = i / (cells_x2-1._fp);
          real weight = (cos(M_PI*xloc)+1)/2;
          weight *= time_factor;
          fields(l,k,j,nx-1-i,iens) = weight*col_avg(l,k,iens) + (1-weight)*fields(l,k,j,nx-1-i,iens);
        });
      }
      if (coupler.get_py() == 0 && cells_y1 > 0) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz,cells_y1,nx,nens) ,
                                          YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
          real yloc   = j / (cells_y1-1._fp);
          real weight = (cos(M_PI*yloc)+1)/2;
          weight *= time_factor;
          fields(l,k,j,i,iens) = weight*col_avg(l,k,iens) + (1-weight)*fields(l,k,j,i,iens);
        });
      }
      if (coupler.get_py() == coupler.get_nproc_y()-1 && cells_y2 > 0) {
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(num_fields,nz,cells_y2,nx,nens) ,
                                          YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
          real yloc   = j / (cells_y2-1._fp);
          real weight = (cos(M_PI*yloc)+1)/2;
          weight *= time_factor;
          fields(l,k,ny-1-j,i,iens) = weight*col_avg(l,k,iens) + (1-weight)*fields(l,k,ny-1-j,i,iens);
        });
      }
    }


  };

}


