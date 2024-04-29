#pragma once

#include "coupler.h"

namespace custom_modules {
  
  struct Time_Averager {

    void init( core::Coupler &coupler ) {
      auto nens = coupler.get_nens();
      auto nx   = coupler.get_nx();
      auto ny   = coupler.get_ny();
      auto nz   = coupler.get_nz();
      auto &dm  = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<real>("avg_u"    ,"",{nz,ny,nx,nens});    dm.get<real,4>("avg_u"    ) = 0;
      dm.register_and_allocate<real>("avg_v"    ,"",{nz,ny,nx,nens});    dm.get<real,4>("avg_v"    ) = 0;
      dm.register_and_allocate<real>("avg_w"    ,"",{nz,ny,nx,nens});    dm.get<real,4>("avg_w"    ) = 0;
      dm.register_and_allocate<real>("avg_tke"  ,"",{nz,ny,nx,nens});    dm.get<real,4>("avg_tke"  ) = 0;
      dm.register_and_allocate<real>("avg_up_up","",{nz,ny,nx,nens});    dm.get<real,4>("avg_up_up") = 0;
      dm.register_and_allocate<real>("avg_up_vp","",{nz,ny,nx,nens});    dm.get<real,4>("avg_up_vp") = 0;
      dm.register_and_allocate<real>("avg_up_wp","",{nz,ny,nx,nens});    dm.get<real,4>("avg_up_wp") = 0;
      dm.register_and_allocate<real>("avg_vp_vp","",{nz,ny,nx,nens});    dm.get<real,4>("avg_vp_vp") = 0;
      dm.register_and_allocate<real>("avg_vp_wp","",{nz,ny,nx,nens});    dm.get<real,4>("avg_vp_wp") = 0;
      dm.register_and_allocate<real>("avg_wp_wp","",{nz,ny,nx,nens});    dm.get<real,4>("avg_wp_wp") = 0;
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

    void accumulate( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens    = coupler.get_nens();
      auto nx      = coupler.get_nx();
      auto ny      = coupler.get_ny();
      auto nz      = coupler.get_nz();
      auto etime   = coupler.get_option<real>("elapsed_time");
      auto uvel      = coupler.get_data_manager_readonly ().get<real const,4>("uvel"     );
      auto vvel      = coupler.get_data_manager_readonly ().get<real const,4>("vvel"     );
      auto wvel      = coupler.get_data_manager_readonly ().get<real const,4>("wvel"     );
      auto tke       = coupler.get_data_manager_readonly ().get<real const,4>("TKE"      );
      auto avg_u     = coupler.get_data_manager_readwrite().get<real      ,4>("avg_u"    );
      auto avg_v     = coupler.get_data_manager_readwrite().get<real      ,4>("avg_v"    );
      auto avg_w     = coupler.get_data_manager_readwrite().get<real      ,4>("avg_w"    );
      auto avg_tke   = coupler.get_data_manager_readwrite().get<real      ,4>("avg_tke"  );
      auto avg_up_up = coupler.get_data_manager_readwrite().get<real      ,4>("avg_up_up");
      auto avg_up_vp = coupler.get_data_manager_readwrite().get<real      ,4>("avg_up_vp");
      auto avg_up_wp = coupler.get_data_manager_readwrite().get<real      ,4>("avg_up_wp");
      auto avg_vp_vp = coupler.get_data_manager_readwrite().get<real      ,4>("avg_vp_vp");
      auto avg_vp_wp = coupler.get_data_manager_readwrite().get<real      ,4>("avg_vp_wp");
      auto avg_wp_wp = coupler.get_data_manager_readwrite().get<real      ,4>("avg_wp_wp");
      double inertia = etime / (etime + dt);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        avg_u  (k,j,i,iens) = inertia * avg_u  (k,j,i,iens) + (1-inertia) * uvel(k,j,i,iens);
        avg_v  (k,j,i,iens) = inertia * avg_v  (k,j,i,iens) + (1-inertia) * vvel(k,j,i,iens);
        avg_w  (k,j,i,iens) = inertia * avg_w  (k,j,i,iens) + (1-inertia) * wvel(k,j,i,iens);
        avg_tke(k,j,i,iens) = inertia * avg_tke(k,j,i,iens) + (1-inertia) * tke (k,j,i,iens);
        real up = uvel(k,j,i,iens) - avg_u(k,j,i,iens);
        real vp = vvel(k,j,i,iens) - avg_v(k,j,i,iens);
        real wp = wvel(k,j,i,iens) - avg_w(k,j,i,iens);
        avg_up_up(k,j,i,iens) = inertia * avg_up_up(k,j,i,iens) + (1-inertia) * up*up;
        avg_up_vp(k,j,i,iens) = inertia * avg_up_vp(k,j,i,iens) + (1-inertia) * up*vp;
        avg_up_wp(k,j,i,iens) = inertia * avg_up_wp(k,j,i,iens) + (1-inertia) * up*wp;
        avg_vp_vp(k,j,i,iens) = inertia * avg_vp_vp(k,j,i,iens) + (1-inertia) * vp*vp;
        avg_vp_wp(k,j,i,iens) = inertia * avg_vp_wp(k,j,i,iens) + (1-inertia) * vp*wp;
        avg_wp_wp(k,j,i,iens) = inertia * avg_wp_wp(k,j,i,iens) + (1-inertia) * wp*wp;
      });
    }
  };

}


