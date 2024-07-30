#pragma once

#include "coupler.h"

namespace custom_modules {
  
  struct Time_Averager {

    real etime;

    void init( core::Coupler &coupler ) {
      auto nx   = coupler.get_nx();
      auto ny   = coupler.get_ny();
      auto nz   = coupler.get_nz();
      auto &dm  = coupler.get_data_manager_readwrite();
      etime = 0;
      dm.register_and_allocate<real>("avg_u"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_u"  ) = 0;
      dm.register_and_allocate<real>("avg_v"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_v"  ) = 0;
      dm.register_and_allocate<real>("avg_w"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_w"  ) = 0;
      dm.register_and_allocate<real>("avg_tke","",{nz,ny,nx});    dm.get<real,3>("avg_tke") = 0;
      coupler.register_output_variable<real>( "avg_u"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_v"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_w"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_tke" , core::Coupler::DIMS_3D );
    }


    void reset( core::Coupler &coupler ) {
      etime = 0;
      auto &dm  = coupler.get_data_manager_readwrite();
      dm.get<real,3>("avg_u"  ) = 0;
      dm.get<real,3>("avg_v"  ) = 0;
      dm.get<real,3>("avg_w"  ) = 0;
      dm.get<real,3>("avg_tke") = 0;
    }

    void accumulate( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx      = coupler.get_nx();
      auto ny      = coupler.get_ny();
      auto nz      = coupler.get_nz();
      auto &dm     = coupler.get_data_manager_readwrite();
      auto uvel    = dm.get<real const,3>("uvel"   );
      auto vvel    = dm.get<real const,3>("vvel"   );
      auto wvel    = dm.get<real const,3>("wvel"   );
      auto tke     = dm.get<real const,3>("TKE"    );
      auto avg_u   = dm.get<real      ,3>("avg_u"  );
      auto avg_v   = dm.get<real      ,3>("avg_v"  );
      auto avg_w   = dm.get<real      ,3>("avg_w"  );
      auto avg_tke = dm.get<real      ,3>("avg_tke");
      double inertia = etime / (etime + dt);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        avg_u  (k,j,i) = inertia * avg_u  (k,j,i) + (1-inertia) * uvel(k,j,i);
        avg_v  (k,j,i) = inertia * avg_v  (k,j,i) + (1-inertia) * vvel(k,j,i);
        avg_w  (k,j,i) = inertia * avg_w  (k,j,i) + (1-inertia) * wvel(k,j,i);
        avg_tke(k,j,i) = inertia * avg_tke(k,j,i) + (1-inertia) * tke (k,j,i);
      });
      etime += dt;
    }
  };

}


