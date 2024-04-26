
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  inline void precursor_sponge( core::Coupler            & coupler_main      ,
                                core::Coupler      const & coupler_precursor ,
                                real                       dt                ,
                                real                       time_scale        ,
                                std::vector<std::string>   vnames            ,
                                int                        cells_x1 = 0      ,
                                int                        cells_x2 = 0      ,
                                int                        cells_y1 = 0      ,
                                int                        cells_y2 = 0      ,
                                int                        cells_z2 = 0      ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nens          = coupler_main     .get_nens();
    auto nx            = coupler_main     .get_nx();
    auto ny            = coupler_main     .get_ny();
    auto nz            = coupler_main     .get_nz();
    auto i_beg         = coupler_main     .get_i_beg();
    auto j_beg         = coupler_main     .get_j_beg();
    auto nx_glob       = coupler_main     .get_nx_glob();
    auto ny_glob       = coupler_main     .get_ny_glob();
    auto &dm_main      = coupler_main     .get_data_manager_readwrite();
    auto &dm_precursor = coupler_precursor.get_data_manager_readonly();
    core::MultiField<real      ,4> fields_main;
    core::MultiField<real const,4> fields_precursor;
    int numvars = vnames.size();
    for (int i=0; i < numvars; i++) {
      fields_main     .add_field( dm_main     .get<real      ,4>(vnames[i]) );
      fields_precursor.add_field( dm_precursor.get<real const,4>(vnames[i]) );
    }

    real time_factor = dt / time_scale;

    if (cells_x1 > 0) {
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(numvars,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        if (i_beg+i < cells_x1) {
          real xloc   = (i_beg+i)/(cells_x1-1._fp);
          real weight = (std::cos(M_PI*xloc)+1)/2 * time_factor;
          fields_main(l,k,j,i,iens) = weight*fields_precursor(l,k,j,i,iens) + (1-weight)*fields_main(l,k,j,i,iens);
        }
      });
    }
    if (cells_x2 > 0) {
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(numvars,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        if (nx_glob-1-(i_beg+i) < cells_x2) {
          real xloc   = (nx_glob-1-(i_beg+i))/(cells_x2-1._fp);
          real weight = (std::cos(M_PI*xloc)+1)/2 * time_factor;
          fields_main(l,k,j,i,iens) = weight*fields_precursor(l,k,j,i,iens) + (1-weight)*fields_main(l,k,j,i,iens);
        }
      });
    }
    if (cells_y1 > 0) {
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(numvars,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        if (j_beg+j < cells_y1) {
          real yloc   = (j_beg+j)/(cells_y1-1._fp);
          real weight = (std::cos(M_PI*yloc)+1)/2 * time_factor;
          fields_main(l,k,j,i,iens) = weight*fields_precursor(l,k,j,i,iens) + (1-weight)*fields_main(l,k,j,i,iens);
        }
      });
    }
    if (cells_y2 > 0) {
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(numvars,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        if (ny_glob-1-(j_beg+j) < cells_y2) {
          real yloc   = (ny_glob-1-(j_beg+j))/(cells_y2-1._fp);
          real weight = (std::cos(M_PI*yloc)+1)/2 * time_factor;
          fields_main(l,k,j,i,iens) = weight*fields_precursor(l,k,j,i,iens) + (1-weight)*fields_main(l,k,j,i,iens);
        }
      });
    }
    if (cells_z2 > 0) {
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(numvars,nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int l, int k, int j, int i, int iens) {
        if (nz-1-k < cells_z2) {
          real zloc   = (nz-1-k)/(cells_z2-1._fp);
          real weight = (std::cos(M_PI*zloc)+1)/2 * time_factor;
          fields_main(l,k,j,i,iens) = weight*fields_precursor(l,k,j,i,iens) + (1-weight)*fields_main(l,k,j,i,iens);
        }
      });
    }
  }
}


