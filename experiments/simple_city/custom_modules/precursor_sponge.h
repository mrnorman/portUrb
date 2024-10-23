
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  inline void precursor_sponge( core::Coupler            & coupler_main      ,
                                core::Coupler      const & coupler_precursor ,
                                std::vector<std::string>   vnames            ,
                                int                        cells_x1 = 0      ,
                                int                        cells_x2 = 0      ,
                                int                        cells_y1 = 0      ,
                                int                        cells_y2 = 0      ) {
    using yikl::parallel_for;
    using yikl::SimpleBounds;
    auto nx            = coupler_main     .get_nx();
    auto ny            = coupler_main     .get_ny();
    auto nz            = coupler_main     .get_nz();
    auto i_beg         = coupler_main     .get_i_beg();
    auto j_beg         = coupler_main     .get_j_beg();
    auto nx_glob       = coupler_main     .get_nx_glob();
    auto ny_glob       = coupler_main     .get_ny_glob();
    auto &dm_main      = coupler_main     .get_data_manager_readwrite();
    auto &dm_precursor = coupler_precursor.get_data_manager_readonly();
    core::MultiField<real      ,3> fields_main;
    core::MultiField<real const,3> fields_precursor;
    int numvars = vnames.size();
    for (int i=0; i < numvars; i++) {
      fields_main     .add_field( dm_main     .get<real      ,3>(vnames.at(i)) );
      fields_precursor.add_field( dm_precursor.get<real const,3>(vnames.at(i)) );
    }

    real p = 5;

    if (cells_x1 > 0) {
      real i1 = 0;
      real i2 = cells_x1;
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (i_beg+i < cells_x1) {
          real weight = std::pow((i2-(i_beg+i))/(i2-i1),p);
          if (i_beg+i == 0) weight = 1;
          fields_main(l,k,j,i) = weight*fields_precursor(l,k,j,i) + (1-weight)*fields_main(l,k,j,i);
        }
      });
    }
    if (cells_x2 > 0) {
      real i1 = nx_glob-cells_x2;
      real i2 = nx_glob-1;
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (nx_glob-1-(i_beg+i) < cells_x2) {
          real weight = std::pow(((i_beg+i)-i1)/(i2-i1),p);
          if (i_beg+i == nx_glob-1) weight = 1;
          fields_main(l,k,j,i) = weight*fields_precursor(l,k,j,i) + (1-weight)*fields_main(l,k,j,i);
        }
      });
    }
    if (cells_y1 > 0) {
      real j1 = 0;
      real j2 = cells_y1;
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (j_beg+j < cells_y1) {
          real weight = std::pow((j2-(j_beg+j))/(j2-j1),p);
          if (j_beg+j == 0) weight = 1;
          fields_main(l,k,j,i) = weight*fields_precursor(l,k,j,i) + (1-weight)*fields_main(l,k,j,i);
        }
      });
    }
    if (cells_y2 > 0) {
      real j1 = ny_glob-cells_y2;
      real j2 = ny_glob-1;
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<4>(numvars,nz,ny,nx) , KOKKOS_LAMBDA (int l, int k, int j, int i) {
        if (ny_glob-1-(j_beg+j) < cells_y2) {
          real weight = std::pow(((j_beg+j)-j1)/(j2-j1),p);
          if (j_beg+j == ny_glob-1) weight = 1;
          fields_main(l,k,j,i) = weight*fields_precursor(l,k,j,i) + (1-weight)*fields_main(l,k,j,i);
        }
      });
    }
  }
}


