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
      dm.register_and_allocate<real>("avg_u"    ,"",{nz,ny,nx});    dm.get<real,3>("avg_u"    ) = 0;
      dm.register_and_allocate<real>("avg_v"    ,"",{nz,ny,nx});    dm.get<real,3>("avg_v"    ) = 0;
      dm.register_and_allocate<real>("avg_w"    ,"",{nz,ny,nx});    dm.get<real,3>("avg_w"    ) = 0;
      dm.register_and_allocate<real>("avg_tke"  ,"",{nz,ny,nx});    dm.get<real,3>("avg_tke"  ) = 0;
      dm.register_and_allocate<real>("avg_up_up","",{nz,ny,nx});    dm.get<real,3>("avg_up_up") = 0;
      dm.register_and_allocate<real>("avg_up_vp","",{nz,ny,nx});    dm.get<real,3>("avg_up_vp") = 0;
      dm.register_and_allocate<real>("avg_up_wp","",{nz,ny,nx});    dm.get<real,3>("avg_up_wp") = 0;
      dm.register_and_allocate<real>("avg_vp_vp","",{nz,ny,nx});    dm.get<real,3>("avg_vp_vp") = 0;
      dm.register_and_allocate<real>("avg_vp_wp","",{nz,ny,nx});    dm.get<real,3>("avg_vp_wp") = 0;
      dm.register_and_allocate<real>("avg_wp_wp","",{nz,ny,nx});    dm.get<real,3>("avg_wp_wp") = 0;
      coupler.register_output_variable<real>( "avg_u"     , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_v"     , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_w"     , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_tke"   , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_up_up" , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_up_vp" , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_up_wp" , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_vp_vp" , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_vp_wp" , core::Coupler::DIMS_3D );
      coupler.register_output_variable<real>( "avg_wp_wp" , core::Coupler::DIMS_3D );
    }


    void reset( core::Coupler &coupler ) {
      etime = 0;
      auto &dm  = coupler.get_data_manager_readwrite();
      dm.get<real,3>("avg_u"    ) = 0;
      dm.get<real,3>("avg_v"    ) = 0;
      dm.get<real,3>("avg_w"    ) = 0;
      dm.get<real,3>("avg_tke"  ) = 0;
      dm.get<real,3>("avg_up_up") = 0;
      dm.get<real,3>("avg_up_vp") = 0;
      dm.get<real,3>("avg_up_wp") = 0;
      dm.get<real,3>("avg_vp_vp") = 0;
      dm.get<real,3>("avg_vp_wp") = 0;
      dm.get<real,3>("avg_wp_wp") = 0;
    }

    void accumulate( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx      = coupler.get_nx();
      auto ny      = coupler.get_ny();
      auto nz      = coupler.get_nz();
      auto uvel      = coupler.get_data_manager_readonly ().get<real const,3>("uvel"     );
      auto vvel      = coupler.get_data_manager_readonly ().get<real const,3>("vvel"     );
      auto wvel      = coupler.get_data_manager_readonly ().get<real const,3>("wvel"     );
      auto tke       = coupler.get_data_manager_readonly ().get<real const,3>("TKE"      );
      auto avg_u     = coupler.get_data_manager_readwrite().get<real      ,3>("avg_u"    );
      auto avg_v     = coupler.get_data_manager_readwrite().get<real      ,3>("avg_v"    );
      auto avg_w     = coupler.get_data_manager_readwrite().get<real      ,3>("avg_w"    );
      auto avg_tke   = coupler.get_data_manager_readwrite().get<real      ,3>("avg_tke"  );
      auto avg_up_up = coupler.get_data_manager_readwrite().get<real      ,3>("avg_up_up");
      auto avg_up_vp = coupler.get_data_manager_readwrite().get<real      ,3>("avg_up_vp");
      auto avg_up_wp = coupler.get_data_manager_readwrite().get<real      ,3>("avg_up_wp");
      auto avg_vp_vp = coupler.get_data_manager_readwrite().get<real      ,3>("avg_vp_vp");
      auto avg_vp_wp = coupler.get_data_manager_readwrite().get<real      ,3>("avg_vp_wp");
      auto avg_wp_wp = coupler.get_data_manager_readwrite().get<real      ,3>("avg_wp_wp");
      double inertia = etime / (etime + dt);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        avg_u  (k,j,i) = inertia * avg_u  (k,j,i) + (1-inertia) * uvel(k,j,i);
        avg_v  (k,j,i) = inertia * avg_v  (k,j,i) + (1-inertia) * vvel(k,j,i);
        avg_w  (k,j,i) = inertia * avg_w  (k,j,i) + (1-inertia) * wvel(k,j,i);
        avg_tke(k,j,i) = inertia * avg_tke(k,j,i) + (1-inertia) * tke (k,j,i);
        real up = uvel(k,j,i) - avg_u(k,j,i);
        real vp = vvel(k,j,i) - avg_v(k,j,i);
        real wp = wvel(k,j,i) - avg_w(k,j,i);
        avg_up_up(k,j,i) = inertia * avg_up_up(k,j,i) + (1-inertia) * up*up;
        avg_up_vp(k,j,i) = inertia * avg_up_vp(k,j,i) + (1-inertia) * up*vp;
        avg_up_wp(k,j,i) = inertia * avg_up_wp(k,j,i) + (1-inertia) * up*wp;
        avg_vp_vp(k,j,i) = inertia * avg_vp_vp(k,j,i) + (1-inertia) * vp*vp;
        avg_vp_wp(k,j,i) = inertia * avg_vp_wp(k,j,i) + (1-inertia) * vp*wp;
        avg_wp_wp(k,j,i) = inertia * avg_wp_wp(k,j,i) + (1-inertia) * wp*wp;
      });
      etime += dt;
    }
  };

}


