
#pragma once

#include "main_header.h"
#include "profiles.h"
#include "coupler.h"
#include "TransformMatrices.h"
#include "hydrostasis.h"
#include <random>

namespace custom_modules {

  inline void sc_perturb( core::Coupler & coupler ) {
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
    if (! dm.entry_exists("surface_roughness"  )) dm.register_and_allocate<real>("surface_roughness"  ,"",dims2d);
    if (! dm.entry_exists("surface_temp"       )) dm.register_and_allocate<real>("surface_temp"       ,"",dims2d);
    if (! dm.entry_exists("surface_khf"        )) dm.register_and_allocate<real>("surface_khf"        ,"",dims2d);
    if (! coupler.option_exists("idWV")) {
      auto tracer_names = coupler.get_tracer_names();
      int idWV = -1;
      for (int tr=0; tr < tracer_names.size(); tr++) { if (tracer_names.at(tr) == "water_vapor") idWV = tr; }
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
    auto dm_surface_rough  = dm.get<real,2>("surface_roughness"  );
    auto dm_surface_temp   = dm.get<real,2>("surface_temp"       );
    auto dm_surface_khf    = dm.get<real,2>("surface_khf"        );
    dm_immersed_prop  = 0;
    dm_immersed_rough = roughness;
    dm_immersed_temp  = 0;
    dm_immersed_khf   = 0;
    dm_surface_rough  = roughness;
    dm_surface_temp   = 0;
    dm_surface_khf    = 0;
    dm_rho_v          = 0;

    const int nqpoints = 9;
    SArray<real,1,nqpoints> qpoints;
    SArray<real,1,nqpoints> qweights;
    TransformMatrices::get_gll_points (qpoints );
    TransformMatrices::get_gll_weights(qweights);

    coupler.add_option<std::string>("bc_x","periodic");
    coupler.add_option<std::string>("bc_y","periodic");
    coupler.add_option<std::string>("bc_z","solid_wall");
    auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);

    if        (coupler.get_option<std::string>("init_data") == "city") {

    } else if (coupler.get_option<std::string>("init_data") == "building") {

    } else if (coupler.get_option<std::string>("init_data") == "buildings_periodic") {

    } else if (coupler.get_option<std::string>("init_data") == "cubes_periodic") {

    } else if (coupler.get_option<std::string>("init_data") == "constant") {

    } else if (coupler.get_option<std::string>("init_data") == "bomex") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        if ((k+0.5_fp)*dz <= 1600) dm_temp (k,j,i) += rand.genFP<real>(-0.1,0.1);
        if ((k+0.5_fp)*dz <= 1600) dm_rho_v(k,j,i) += rand.genFP<real>(-2.5e-5,2.5e-5)*dm_rho_d(k,j,i);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_neutral") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        if ((k+0.5_fp)*dz <= 400) dm_temp(k,j,i) += rand.genFP<real>(-0.25,0.25);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_convective") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        if ((k+0.5_fp)*dz <= 400) dm_temp(k,j,i) += rand.genFP<real>(-0.25,0.25);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_stable") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        if ((k+0.5_fp)*dz <= 50) dm_temp(k,j,i) += rand.genFP<real>(-0.10,0.10);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_stable_bvf") {

    } else if (coupler.get_option<std::string>("init_data") == "ABL_neutral2") {

    } else if (coupler.get_option<std::string>("init_data") == "AWAKEN_neutral") {

    } else if (coupler.get_option<std::string>("init_data") == "supercell") {

      real x0    = xlen / 2;
      real y0    = ylen / 2;
      real z0    = 1500;
      real radx  = 10000;
      real rady  = 10000;
      real radz  = 1500;
      real amp   = 3;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real Tpert = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          for (int jj=0; jj<nqpoints; jj++) {
            for (int ii=0; ii<nqpoints; ii++) {
              real x     = (i_beg+i+0.5)*dx + qpoints(ii)*dx;
              real y     = (j_beg+j+0.5)*dy + qpoints(jj)*dy;
              real z     = (      k+0.5)*dz + qpoints(kk)*dz;
              real xn    = (x-x0)/radx;
              real yn    = (y-y0)/rady;
              real zn    = (z-z0)/radz;
              real rad   = sqrt( xn*xn + yn*yn + zn*zn );
              if (rad <= 1) Tpert += amp * pow( cos(M_PI*rad/2) , 2._fp );
            }
          }
        }
        dm_temp(k,j,i) += Tpert;
      });

    } // if (init_data == ...)

    int hs = 1;
    {
      core::MultiField<real,3> fields;
      fields.add_field( dm_immersed_prop  );
      fields.add_field( dm_immersed_rough );
      fields.add_field( dm_immersed_temp  );
      fields.add_field( dm_immersed_khf   );
      auto fields_halos = coupler.create_and_exchange_halos( fields , hs );
      std::vector<std::string> dim_names = {"z_halo1","y_halo1","x_halo1"};
      dm.register_and_allocate<real>("immersed_proportion_halos","",{nz+2*hs,ny+2*hs,nx+2*hs},dim_names);
      dm.register_and_allocate<real>("immersed_roughness_halos" ,"",{nz+2*hs,ny+2*hs,nx+2*hs},dim_names);
      dm.register_and_allocate<real>("immersed_temp_halos"      ,"",{nz+2*hs,ny+2*hs,nx+2*hs},dim_names);
      dm.register_and_allocate<real>("immersed_khf_halos"       ,"",{nz+2*hs,ny+2*hs,nx+2*hs},dim_names);
      fields_halos.get_field(0).deep_copy_to( dm.get<real,3>("immersed_proportion_halos") );
      fields_halos.get_field(1).deep_copy_to( dm.get<real,3>("immersed_roughness_halos" ) );
      fields_halos.get_field(2).deep_copy_to( dm.get<real,3>("immersed_temp_halos"      ) );
      fields_halos.get_field(3).deep_copy_to( dm.get<real,3>("immersed_khf_halos"       ) );
    }
    {
      core::MultiField<real,2> fields;
      fields.add_field( dm_surface_rough );
      fields.add_field( dm_surface_temp  );
      fields.add_field( dm_surface_khf   );
      auto fields_halos = coupler.create_and_exchange_halos( fields , hs );
      std::vector<std::string> dim_names = {"y_halo1","x_halo1"};
      dm.register_and_allocate<real>("surface_roughness_halos" ,"",{ny+2*hs,nx+2*hs},dim_names);
      dm.register_and_allocate<real>("surface_temp_halos"      ,"",{ny+2*hs,nx+2*hs},dim_names);
      dm.register_and_allocate<real>("surface_khf_halos"       ,"",{ny+2*hs,nx+2*hs},dim_names);
      fields_halos.get_field(0).deep_copy_to( dm.get<real,2>("surface_roughness_halos" ) );
      fields_halos.get_field(1).deep_copy_to( dm.get<real,2>("surface_temp_halos"      ) );
      fields_halos.get_field(2).deep_copy_to( dm.get<real,2>("surface_khf_halos"       ) );
    }

    auto imm_prop  = dm.get<real,3>("immersed_proportion_halos");
    auto imm_rough = dm.get<real,3>("immersed_roughness_halos" );
    auto imm_temp  = dm.get<real,3>("immersed_temp_halos"      );
    auto imm_khf   = dm.get<real,3>("immersed_khf_halos"       );
    auto sfc_rough = dm.get<real,2>("surface_roughness_halos"  );
    auto sfc_temp  = dm.get<real,2>("surface_temp_halos"       );
    auto sfc_khf   = dm.get<real,2>("surface_khf_halos"        );
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) , KOKKOS_LAMBDA (int kk, int j, int i) {
      imm_prop (      kk,j,i) = 1;
      imm_rough(      kk,j,i) = sfc_rough(j,i);
      imm_temp (      kk,j,i) = sfc_temp (j,i);
      imm_khf  (      kk,j,i) = sfc_khf  (j,i);
      imm_prop (hs+nz+kk,j,i) = 0;
      imm_rough(hs+nz+kk,j,i) = 0;
      imm_temp (hs+nz+kk,j,i) = 0;
      imm_khf  (hs+nz+kk,j,i) = 0;
    });
  }

}


