#pragma once

#include "coupler.h"

namespace custom_modules {
  
  inline void surface_cooling( core::Coupler &coupler , real dt ) {
    using yikl::parallel_for;
    using yikl::SimpleBounds;
    if (coupler.option_exists("standalone_input_file")) {
      YAML::Node config = YAML::LoadFile(coupler.get_option<std::string>("standalone_input_file"));
      real rate = config["surface_cooling_rate"].as<real>(0.);
      auto sfc_temp       = coupler.get_data_manager_readwrite().get<real,2>("surface_temp"       );
      auto sfc_temp_halos = coupler.get_data_manager_readwrite().get<real,2>("surface_temp_halos" );
      auto sfc_imm_temp   = coupler.get_data_manager_readwrite().get<real,3>("immersed_temp_halos").slice<2>(0,0,0);
      int nx = coupler.get_nx();
      int ny = coupler.get_ny();
      int hs = (sfc_temp_halos.extent(0)-ny)/2;
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<2>(ny+2*hs,nx+2*hs) , KOKKOS_LAMBDA (int j, int i) {
        if (j < ny && i < nx) sfc_temp(j,i) -= dt*rate/3600;
        sfc_temp_halos(j,i) -= dt*rate/3600;
        sfc_imm_temp  (j,i) -= dt*rate/3600;
      });
    }
  }

}


