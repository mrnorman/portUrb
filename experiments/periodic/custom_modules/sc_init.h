
#pragma once

#include "main_header.h"
#include "profiles.h"

namespace custom_modules {

  inline void sc_init( core::Coupler & coupler ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nens    = coupler.get_nens();
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

    auto &dm = coupler.get_data_manager_readwrite();
    auto dims3d = {nz,ny,nx,nens};
    auto dims2d = {   ny,nx,nens};
    if (! dm.entry_exists("density_dry"        )) dm.register_and_allocate<real>("density_dry"        ,"",dims3d);
    if (! dm.entry_exists("uvel"               )) dm.register_and_allocate<real>("uvel"               ,"",dims3d);
    if (! dm.entry_exists("vvel"               )) dm.register_and_allocate<real>("vvel"               ,"",dims3d);
    if (! dm.entry_exists("wvel"               )) dm.register_and_allocate<real>("wvel"               ,"",dims3d);
    if (! dm.entry_exists("temp"               )) dm.register_and_allocate<real>("temp"               ,"",dims3d);
    if (! dm.entry_exists("water_vapor"        )) dm.register_and_allocate<real>("water_vapor"        ,"",dims3d);
    if (! dm.entry_exists("immersed_proportion")) dm.register_and_allocate<real>("immersed_proportion","",dims3d);
    if (! coupler.option_exists("idWV")) {
      auto tracer_names = coupler.get_tracer_names();
      int idWV = -1;
      for (int tr=0; tr < tracer_names.size(); tr++) { if (tracer_names[tr] == "water_vapor") idWV = tr; }
      coupler.set_option<int>("idWV",idWV);
    }
    int idWV = coupler.get_option<int>("idWV");

    auto dm_rho_d               = dm.get<real,4>("density_dry"        );
    auto dm_uvel                = dm.get<real,4>("uvel"               );
    auto dm_vvel                = dm.get<real,4>("vvel"               );
    auto dm_wvel                = dm.get<real,4>("wvel"               );
    auto dm_temp                = dm.get<real,4>("temp"               );
    auto dm_rho_v               = dm.get<real,4>("water_vapor"        );
    auto dm_immersed_proportion = dm.get<real,4>("immersed_proportion");
    dm_immersed_proportion = 0;
    dm_rho_v               = 0;

    const int nqpoints = 9;
    SArray<real,1,nqpoints> qpoints;
    SArray<real,1,nqpoints> qweights;
    TransformMatrices::get_gll_points (qpoints );
    TransformMatrices::get_gll_weights(qweights);

    coupler.add_option<std::string>("bc_x","periodic");
    coupler.add_option<std::string>("bc_y","periodic");
    coupler.add_option<std::string>("bc_z","periodic");
    coupler.add_option<bool       >("enable_gravity",false);

    if (coupler.get_option<std::string>("init_data") == "taylor_green") {
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        dm_rho_d(k,j,i,iens) = 0;
        dm_uvel (k,j,i,iens) = 0;
        dm_vvel (k,j,i,iens) = 0;
        dm_wvel (k,j,i,iens) = 0;
        dm_temp (k,j,i,iens) = 0;
        for (int kk=0; kk < nqpoints; kk++) {
        for (int jj=0; jj < nqpoints; jj++) {
        for (int ii=0; ii < nqpoints; ii++) {
          real amp = 20;
          real x   = 2*M_PI*( (i+0.5_fp)*dx + qpoints(ii)*dx )/xlen;
          real y   = 2*M_PI*( (j+0.5_fp)*dy + qpoints(jj)*dy )/ylen;
          real z   = 2*M_PI*( (k+0.5_fp)*dz + qpoints(kk)*dz )/zlen;
          dm_rho_d(k,j,i,iens) += 1                       *qweights(ii)*qweights(jj)*qweights(kk);
          dm_uvel (k,j,i,iens) += amp*cos(x)*sin(y)*sin(z)*qweights(ii)*qweights(jj)*qweights(kk);
          dm_vvel (k,j,i,iens) += amp*sin(x)*cos(y)*sin(z)*qweights(ii)*qweights(jj)*qweights(kk);
          dm_wvel (k,j,i,iens) += amp*sin(x)*sin(y)*cos(z)*qweights(ii)*qweights(jj)*qweights(kk);
          dm_temp (k,j,i,iens) += 300                     *qweights(ii)*qweights(jj)*qweights(kk);
        }
        }
        }
        real ramp = 2;
        yakl::Random rng(k*ny_glob*nx_glob*nens*3+(j_beg+j)*nx_glob*nens*3+(i_beg+i)*nens*3+iens*3);
        dm_uvel(k,j,i,iens) += rng.genFP<real>(-ramp,ramp);
        dm_vvel(k,j,i,iens) += rng.genFP<real>(-ramp,ramp);
        dm_wvel(k,j,i,iens) += rng.genFP<real>(-ramp,ramp);
      });

    }

  }

}

