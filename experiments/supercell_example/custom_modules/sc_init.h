
#pragma once

#include "main_header.h"
#include "profiles.h"

namespace custom_modules {

  inline void sc_init( core::Coupler & coupler ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nx      = coupler.get_nx();
    auto ny      = coupler.get_ny();
    auto nz      = coupler.get_nz();
    auto dx      = coupler.get_dx();
    auto dy      = coupler.get_dy();
    auto dz      = coupler.get_dz();
    auto xlen    = coupler.get_xlen();
    auto ylen    = coupler.get_ylen();
    auto zlen    = coupler.get_zlen();
    auto i_beg   = coupler.get_i_beg();
    auto j_beg   = coupler.get_j_beg();
    auto nx_glob = coupler.get_nx_glob();
    auto ny_glob = coupler.get_ny_glob();
    auto sim2d   = coupler.is_sim2d();
    if (! coupler.option_exists("R_d"     )) coupler.set_option<real>("R_d"     ,287.       );
    if (! coupler.option_exists("cp_d"    )) coupler.set_option<real>("cp_d"    ,1003.      );
    if (! coupler.option_exists("R_v"     )) coupler.set_option<real>("R_v"     ,461.       );
    if (! coupler.option_exists("cp_v"    )) coupler.set_option<real>("cp_v"    ,1859       );
    if (! coupler.option_exists("p0"      )) coupler.set_option<real>("p0"      ,1.e5       );
    if (! coupler.option_exists("grav"    )) coupler.set_option<real>("grav"    ,9.81       );
    auto R_d  = coupler.get_option<real>("R_d" );
    auto cp_d = coupler.get_option<real>("cp_d");
    auto R_v  = coupler.get_option<real>("R_v" );
    auto cp_v = coupler.get_option<real>("cp_v");
    auto p0   = coupler.get_option<real>("p0"  );
    auto grav = coupler.get_option<real>("grav");
    if (! coupler.option_exists("cv_d"   )) coupler.set_option<real>("cv_d"   ,cp_d - R_d );
    auto cv_d = coupler.get_option<real>("cv_d");
    if (! coupler.option_exists("gamma_d")) coupler.set_option<real>("gamma_d",cp_d / cv_d);
    if (! coupler.option_exists("kappa_d")) coupler.set_option<real>("kappa_d",R_d  / cp_d);
    if (! coupler.option_exists("cv_v"   )) coupler.set_option<real>("cv_v"   ,R_v - cp_v );
    auto gamma = coupler.get_option<real>("gamma_d");
    auto kappa = coupler.get_option<real>("kappa_d");
    if (! coupler.option_exists("C0")) coupler.set_option<real>("C0" , pow( R_d * pow( p0 , -kappa ) , gamma ));
    auto C0    = coupler.get_option<real>("C0");
    auto roughness = coupler.get_option<real>("roughness",0.1);
    auto &dm = coupler.get_data_manager_readwrite();
    auto dims3d = {nz,ny,nx};
    auto dims2d = {   ny,nx};
    if (! dm.entry_exists("density_dry"        )) dm.register_and_allocate<real>("density_dry"        ,"",dims3d);
    if (! dm.entry_exists("uvel"               )) dm.register_and_allocate<real>("uvel"               ,"",dims3d);
    if (! dm.entry_exists("vvel"               )) dm.register_and_allocate<real>("vvel"               ,"",dims3d);
    if (! dm.entry_exists("wvel"               )) dm.register_and_allocate<real>("wvel"               ,"",dims3d);
    if (! dm.entry_exists("temp"               )) dm.register_and_allocate<real>("temp"               ,"",dims3d);
    if (! dm.entry_exists("water_vapor"        )) dm.register_and_allocate<real>("water_vapor"        ,"",dims3d);
    if (! dm.entry_exists("immersed_proportion")) dm.register_and_allocate<real>("immersed_proportion","",dims3d);
    if (! dm.entry_exists("immersed_roughness" )) dm.register_and_allocate<real>("immersed_roughness" ,"",dims3d);
    if (! dm.entry_exists("immersed_temp"      )) dm.register_and_allocate<real>("immersed_temp"      ,"",dims3d);
    if (! dm.entry_exists("immersed_khf"       )) dm.register_and_allocate<real>("immersed_khf"       ,"",dims3d);
    if (! coupler.option_exists("idWV")) {
      auto tracer_names = coupler.get_tracer_names();
      int idWV = -1;
      for (int tr=0; tr < tracer_names.size(); tr++) { if (tracer_names[tr] == "water_vapor") idWV = tr; }
      coupler.set_option<int>("idWV",idWV);
    }
    int idWV = coupler.get_option<int>("idWV");
    auto dm_rho_d          = dm.get<real,3>("density_dry"        );
    auto dm_uvel           = dm.get<real,3>("uvel"               );
    auto dm_vvel           = dm.get<real,3>("vvel"               );
    auto dm_wvel           = dm.get<real,3>("wvel"               );
    auto dm_temp           = dm.get<real,3>("temp"               );
    auto dm_rho_v          = dm.get<real,3>("water_vapor"        );
    auto dm_immersed_prop  = dm.get<real,3>("immersed_proportion");
    auto dm_immersed_rough = dm.get<real,3>("immersed_roughness" );
    auto dm_immersed_temp  = dm.get<real,3>("immersed_temp"      );
    auto dm_immersed_khf   = dm.get<real,3>("immersed_khf"       );
    dm_immersed_prop  = 0;
    dm_immersed_rough = roughness;
    dm_immersed_temp  = 0;
    dm_immersed_khf   = 0;
    dm_rho_v          = 0;

    const int nq = 9;
    SArray<real,1,nq> qpoints;
    SArray<real,1,nq> qweights;
    TransformMatrices::get_gll_points (qpoints );
    TransformMatrices::get_gll_weights(qweights);

    coupler.add_option<std::string>("bc_x","periodic");
    coupler.add_option<std::string>("bc_y","periodic");
    coupler.add_option<std::string>("bc_z","solid_wall");
    coupler.add_option<bool       >("enable_gravity",true);

    real constexpr z_0    = 0;
    real constexpr z_trop = 12000;
    real constexpr z_top  = 20000;
    real constexpr T_0    = 300;
    real constexpr T_trop = 213;
    real constexpr T_top  = 213;
    real constexpr p_0    = 100000;

    realHost2d rho_d_gll_host("rho_d",nz,nq);
    realHost2d rho_v_gll_host("rho_v",nz,nq);
    realHost2d press_gll_host("press",nz,nq);
    real pgll = p0;
    for (int k1 = 0; k1 < nz; k1++) {     // Loop over cells
      real z    = (k1+0.5_fp)*dz + qpoints(0)*dz;
      real T    = modules::profiles::init_supercell_temperature (z,z_0,z_trop,z_top,T_0,T_trop,T_top);
      real RH   = modules::profiles::init_supercell_relhum      (z,z_0,z_trop);
      real pdry = modules::profiles::init_supercell_pressure_dry(z,z_0,z_trop,z_top,T_0,T_trop,T_top,p_0,R_d,grav);
      real qsat = modules::profiles::init_supercell_sat_mix_dry (pdry,T);
      real qv   = std::min( 0.014_fp , qsat*RH );
      rho_d_gll_host(k1,0) = pgll / (R_d+qv*R_v) / T  ;
      rho_v_gll_host(k1,0) = qv * rho_d_gll_host(k1,0);
      press_gll_host(k1,0) = pgll                     ;
      for (int k2=1; k2 < nq; k2++) { // Loop over intervals between GLL quadrature points
        real tot_quad = 0;
        real z1 = (k1+0.5_fp)*dz + qpoints(k2-1)*dz;
        real z2 = (k1+0.5_fp)*dz + qpoints(k2  )*dz;
        real dzloc = (z2-z1);
        for (int k3=0; k3 < nq; k3++) {    // Loop over GLL points within this interval
          real z    = z1 + 0.5_fp*dzloc + qpoints(k3)*dzloc;
          real T    = modules::profiles::init_supercell_temperature (z,z_0,z_trop,z_top,T_0,T_trop,T_top);
          real RH   = modules::profiles::init_supercell_relhum      (z,z_0,z_trop);
          real pdry = modules::profiles::init_supercell_pressure_dry(z,z_0,z_trop,z_top,T_0,T_trop,T_top,p_0,R_d,grav);
          real qsat = modules::profiles::init_supercell_sat_mix_dry (pdry,T);
          real qv   = std::min( 0.014_fp , qsat*RH );
          tot_quad -= ((1+qv)*grav/((R_d+qv*R_v)*T))*qweights(k3);
        }
        pgll      = pgll*exp(tot_quad*dzloc);
        real z    = z2;
        real T    = modules::profiles::init_supercell_temperature (z,z_0,z_trop,z_top,T_0,T_trop,T_top);
        real RH   = modules::profiles::init_supercell_relhum      (z,z_0,z_trop);
        real pdry = modules::profiles::init_supercell_pressure_dry(z,z_0,z_trop,z_top,T_0,T_trop,T_top,p_0,R_d,grav);
        real qsat = modules::profiles::init_supercell_sat_mix_dry (pdry,T);
        real qv   = std::min( 0.014_fp , qsat*RH );
        rho_d_gll_host(k1,k2) = pgll / (R_d+qv*R_v) / T   ;
        rho_v_gll_host(k1,k2) = qv * rho_d_gll_host(k1,k2);
        press_gll_host(k1,k2) = pgll                      ;
      }
    }
    auto rho_d_gll = rho_d_gll_host.createDeviceCopy();
    auto rho_v_gll = rho_v_gll_host.createDeviceCopy();
    auto press_gll = press_gll_host.createDeviceCopy();

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      dm_rho_d(k,j,i) = 0;
      dm_uvel (k,j,i) = 0;
      dm_vvel (k,j,i) = 0;
      dm_wvel (k,j,i) = 0;
      dm_temp (k,j,i) = 0;
      dm_rho_v(k,j,i) = 0;
      for (int kk=0; kk < nq; kk++) {
        real zloc = (k+0.5_fp)*dz + qpoints(kk)*dz;
        real constexpr zs = 5000;
        real constexpr us = 30;
        real constexpr uc = 15;
        real uvel  = zloc < zs ? us * (zloc / zs) - uc : us - uc;
        real rho_d = rho_d_gll(k,kk);
        real rho_v = rho_v_gll(k,kk);
        real p     = press_gll(k,kk);
        real T     = p/(rho_d*R_d+rho_v*R_v);
        dm_rho_d(k,j,i) += rho_d*qweights(kk);
        dm_uvel (k,j,i) += uvel *qweights(kk);
        dm_vvel (k,j,i) += 0;
        dm_wvel (k,j,i) += 0;
        dm_temp (k,j,i) += T    *qweights(kk);
        dm_rho_v(k,j,i) += rho_v*qweights(kk);
      }
    });

    core::MultiField<real,3> fields;
    fields.add_field( dm_immersed_prop  );
    fields.add_field( dm_immersed_rough );
    fields.add_field( dm_immersed_temp  );
    fields.add_field( dm_immersed_khf   );
    auto fields_halos = coupler.create_and_exchange_halos( fields , 1 );
    dm.register_and_allocate<real>("immersed_proportion_halos","",{nz+2,ny+2,nx+2},{"z_halo1","y_halo1","x_halo1"});
    dm.register_and_allocate<real>("immersed_roughness_halos" ,"",{nz+2,ny+2,nx+2},{"z_halo1","y_halo1","x_halo1"});
    dm.register_and_allocate<real>("immersed_temp_halos"      ,"",{nz+2,ny+2,nx+2},{"z_halo1","y_halo1","x_halo1"});
    dm.register_and_allocate<real>("immersed_khf_halos"       ,"",{nz+2,ny+2,nx+2},{"z_halo1","y_halo1","x_halo1"});
    fields_halos.get_field(0).deep_copy_to( dm.get<real,3>("immersed_proportion_halos") );
    fields_halos.get_field(1).deep_copy_to( dm.get<real,3>("immersed_roughness_halos" ) );
    fields_halos.get_field(2).deep_copy_to( dm.get<real,3>("immersed_temp_halos"      ) );
    fields_halos.get_field(3).deep_copy_to( dm.get<real,3>("immersed_khf_halos"       ) );
    int hs = 1;
    auto immersed_proportion_halos = dm.get<real,3>("immersed_proportion_halos");
    auto immersed_roughness_halos  = dm.get<real,3>("immersed_roughness_halos" );
    if (coupler.get_option<std::string>("bc_z") == "solid_wall") {
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) ,
                                        YAKL_LAMBDA (int kk, int j, int i) {
        immersed_proportion_halos(      kk,j,i) = 1;
        immersed_proportion_halos(hs+nz+kk,j,i) = 1;
        immersed_roughness_halos (      kk,j,i) = roughness;
        immersed_roughness_halos (hs+nz+kk,j,i) = 0;
      });
    }
  }

}


