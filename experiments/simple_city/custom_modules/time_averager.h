#pragma once

#include "coupler.h"

namespace custom_modules {
  
  struct Time_Averager {

    void init( core::Coupler &coupler ) {
      auto nx   = coupler.get_nx();
      auto ny   = coupler.get_ny();
      auto nz   = coupler.get_nz();
      auto &dm  = coupler.get_data_manager_readwrite();
      coupler.set_option<real>("time_averager_etime",0);
      dm.register_and_allocate<real>("avg_u"      ,"",{nz,ny,nx});    dm.get<real,3>("avg_u"      ) = 0;
      dm.register_and_allocate<real>("avg_v"      ,"",{nz,ny,nx});    dm.get<real,3>("avg_v"      ) = 0;
      dm.register_and_allocate<real>("avg_w"      ,"",{nz,ny,nx});    dm.get<real,3>("avg_w"      ) = 0;
      dm.register_and_allocate<real>("avg_tke"    ,"",{nz,ny,nx});    dm.get<real,3>("avg_tke"    ) = 0;
      dm.register_and_allocate<real>("avg_up_up"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_up_up"  ) = 0;
      dm.register_and_allocate<real>("avg_up_vp"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_up_vp"  ) = 0;
      dm.register_and_allocate<real>("avg_up_wp"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_up_wp"  ) = 0;
      dm.register_and_allocate<real>("avg_vp_vp"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_vp_vp"  ) = 0;
      dm.register_and_allocate<real>("avg_vp_wp"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_vp_wp"  ) = 0;
      dm.register_and_allocate<real>("avg_wp_wp"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_wp_wp"  ) = 0;
      dm.register_and_allocate<real>("long_avg_u" ,"",{nz,ny,nx});    dm.get<real,3>("long_avg_u" ) = 0;
      dm.register_and_allocate<real>("long_avg_v" ,"",{nz,ny,nx});    dm.get<real,3>("long_avg_v" ) = 0;
      dm.register_and_allocate<real>("long_avg_w" ,"",{nz,ny,nx});    dm.get<real,3>("long_avg_w" ) = 0;
      coupler.register_output_variable<real>( "avg_u"       , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_v"       , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_w"       , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_tke"     , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_up_up"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_up_vp"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_up_wp"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_vp_vp"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_vp_wp"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_wp_wp"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "long_avg_u"  , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "long_avg_v"  , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "long_avg_w"  , core::Coupler::DIMS_3D );
    }


    void reset( core::Coupler &coupler ) {
      coupler.set_option<real>("time_averager_etime",0);
      auto &dm  = coupler.get_data_manager_readwrite();
      dm.get<real,3>("avg_u"      ) = 0;
      dm.get<real,3>("avg_v"      ) = 0;
      dm.get<real,3>("avg_w"      ) = 0;
      dm.get<real,3>("avg_tke"    ) = 0;
      dm.get<real,3>("avg_up_up"  ) = 0;
      dm.get<real,3>("avg_up_vp"  ) = 0;
      dm.get<real,3>("avg_up_wp"  ) = 0;
      dm.get<real,3>("avg_vp_vp"  ) = 0;
      dm.get<real,3>("avg_vp_wp"  ) = 0;
      dm.get<real,3>("avg_wp_wp"  ) = 0;
    }

    void accumulate( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx_glob = coupler.get_nx_glob();
      auto ny_glob = coupler.get_ny_glob();
      auto nx      = coupler.get_nx();
      auto ny      = coupler.get_ny();
      auto nz      = coupler.get_nz();
      auto &dm     = coupler.get_data_manager_readwrite();
      auto uvel        = dm.get<real const,3>("uvel"       );
      auto vvel        = dm.get<real const,3>("vvel"       );
      auto wvel        = dm.get<real const,3>("wvel"       );
      auto tke         = dm.get<real const,3>("TKE"        );
      auto avg_u       = dm.get<real      ,3>("avg_u"      );
      auto avg_v       = dm.get<real      ,3>("avg_v"      );
      auto avg_w       = dm.get<real      ,3>("avg_w"      );
      auto long_avg_u  = dm.get<real      ,3>("long_avg_u" );
      auto long_avg_v  = dm.get<real      ,3>("long_avg_v" );
      auto long_avg_w  = dm.get<real      ,3>("long_avg_w" );
      auto avg_tke     = dm.get<real      ,3>("avg_tke"    );
      auto avg_up_up   = dm.get<real      ,3>("avg_up_up"  );
      auto avg_up_vp   = dm.get<real      ,3>("avg_up_vp"  );
      auto avg_up_wp   = dm.get<real      ,3>("avg_up_wp"  );
      auto avg_vp_vp   = dm.get<real      ,3>("avg_vp_vp"  );
      auto avg_vp_wp   = dm.get<real      ,3>("avg_vp_wp"  );
      auto avg_wp_wp   = dm.get<real      ,3>("avg_wp_wp"  );
      auto etime       = coupler.get_option<real>("time_averager_etime");
      real inertia = etime / (etime + dt);
      real etime_long = coupler.get_option<real>("elapsed_time");
      real inertia_long = etime_long / (etime_long + dt);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        long_avg_u(k,j,i) = inertia_long * long_avg_u(k,j,i) + (1-inertia_long) * uvel(k,j,i);
        long_avg_v(k,j,i) = inertia_long * long_avg_v(k,j,i) + (1-inertia_long) * vvel(k,j,i);
        long_avg_w(k,j,i) = inertia_long * long_avg_w(k,j,i) + (1-inertia_long) * wvel(k,j,i);
        real up = uvel(k,j,i) - long_avg_u(k,j,i);
        real vp = vvel(k,j,i) - long_avg_v(k,j,i);
        real wp = wvel(k,j,i) - long_avg_w(k,j,i);
        avg_u     (k,j,i) = inertia      * avg_u     (k,j,i) + (1-inertia     ) * uvel(k,j,i);
        avg_v     (k,j,i) = inertia      * avg_v     (k,j,i) + (1-inertia     ) * vvel(k,j,i);
        avg_w     (k,j,i) = inertia      * avg_w     (k,j,i) + (1-inertia     ) * wvel(k,j,i);
        avg_tke   (k,j,i) = inertia      * avg_tke   (k,j,i) + (1-inertia     ) * tke (k,j,i);
        avg_up_up (k,j,i) = inertia      * avg_up_up (k,j,i) + (1-inertia     ) * up*up;
        avg_up_vp (k,j,i) = inertia      * avg_up_vp (k,j,i) + (1-inertia     ) * up*vp;
        avg_up_wp (k,j,i) = inertia      * avg_up_wp (k,j,i) + (1-inertia     ) * up*wp;
        avg_vp_vp (k,j,i) = inertia      * avg_vp_vp (k,j,i) + (1-inertia     ) * vp*vp;
        avg_vp_wp (k,j,i) = inertia      * avg_vp_wp (k,j,i) + (1-inertia     ) * vp*wp;
        avg_wp_wp (k,j,i) = inertia      * avg_wp_wp (k,j,i) + (1-inertia     ) * wp*wp;
      });
      coupler.set_option<real>("time_averager_etime",etime+dt);
    }
  };

}


