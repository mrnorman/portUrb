
#pragma once

#include "main_header.h"
#include "profiles.h"

namespace custom_modules {

  inline void sc_init( core::Coupler & coupler ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;
    auto nens        = coupler.get_nens();
    auto nx          = coupler.get_nx();
    auto ny          = coupler.get_ny();
    auto nz          = coupler.get_nz();
    auto dx          = coupler.get_dx();
    auto dy          = coupler.get_dy();
    auto dz          = coupler.get_dz();
    auto xlen        = coupler.get_xlen();
    auto ylen        = coupler.get_ylen();
    auto zlen        = coupler.get_zlen();
    auto i_beg       = coupler.get_i_beg();
    auto j_beg       = coupler.get_j_beg();
    auto nx_glob     = coupler.get_nx_glob();
    auto ny_glob     = coupler.get_ny_glob();
    auto sim2d       = coupler.is_sim2d();
    auto enable_gravity = coupler.get_option<bool>("enable_gravity",true);
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
    if (! dm.entry_exists("density_dry"        )) dm.register_and_allocate<real>("density_dry"        ,"",{nz,ny,nx,nens});
    if (! dm.entry_exists("uvel"               )) dm.register_and_allocate<real>("uvel"               ,"",{nz,ny,nx,nens});
    if (! dm.entry_exists("vvel"               )) dm.register_and_allocate<real>("vvel"               ,"",{nz,ny,nx,nens});
    if (! dm.entry_exists("wvel"               )) dm.register_and_allocate<real>("wvel"               ,"",{nz,ny,nx,nens});
    if (! dm.entry_exists("temp"               )) dm.register_and_allocate<real>("temp"               ,"",{nz,ny,nx,nens});
    if (! dm.entry_exists("water_vapor"        )) dm.register_and_allocate<real>("water_vapor"        ,"",{nz,ny,nx,nens});
    if (! dm.entry_exists("immersed_proportion")) dm.register_and_allocate<real>("immersed_proportion","",{nz,ny,nx,nens});
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

    const int nqpoints = 9;
    SArray<real,1,nqpoints> qpoints;
    SArray<real,1,nqpoints> qweights;
    TransformMatrices::get_gll_points (qpoints );
    TransformMatrices::get_gll_weights(qweights);

    coupler.add_option<std::string>("bc_x","periodic");
    coupler.add_option<std::string>("bc_y","periodic");
    coupler.add_option<std::string>("bc_z","wall"    );
    coupler.set_option<bool>("use_immersed_boundaries",true);

    if (coupler.get_option<std::string>("init_data") == "city") {
      real height_mean = 60;
      real height_std  = 10;

      int building_length = 30;
      int cells_per_building = (int) std::round(building_length / dx);
      int buildings_pad = 20;
      int nblocks_x = (static_cast<int>(xlen)/building_length - 2*buildings_pad)/3;
      int nblocks_y = (static_cast<int>(ylen)/building_length - 2*buildings_pad)/9;
      int nbuildings_x = nblocks_x * 3;
      int nbuildings_y = nblocks_y * 9;

      realHost2d building_heights_host("building_heights",nbuildings_y,nbuildings_x);
      if (coupler.is_mainproc()) {
        std::mt19937 gen{17};
        std::normal_distribution<> d{height_mean, height_std};
        for (int j=0; j < nbuildings_y; j++) {
          for (int i=0; i < nbuildings_x; i++) {
            building_heights_host(j,i) = d(gen);
          }
        }
      }
      auto type = coupler.get_mpi_data_type();
      MPI_Bcast( building_heights_host.data() , building_heights_host.size() , type , 0 , MPI_COMM_WORLD);
      auto building_heights = building_heights_host.createDeviceCopy();

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        dm_rho_d(k,j,i,iens) = 0;
        dm_uvel (k,j,i,iens) = 0;
        dm_vvel (k,j,i,iens) = 0;
        dm_wvel (k,j,i,iens) = 0;
        dm_temp (k,j,i,iens) = 0;
        dm_rho_v(k,j,i,iens) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          for (int jj=0; jj<nqpoints; jj++) {
            for (int ii=0; ii<nqpoints; ii++) {
              real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
              real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
              real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
              real rho, u, v, w, theta, rho_v, hr, ht;
              if (enable_gravity) { modules::profiles::hydro_const_theta(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }
              else                { hr = 1.15;     ht = 300;                              }
              rho   = hr;
              u     = 20;
              v     = 0;
              w     = 0;
              theta = ht;
              rho_v = 0;
              real T = C0*std::pow(rho*theta,gamma)/(rho*R_d);
              if (sim2d) v = 0;
              real wt = qweights(ii)*qweights(jj)*qweights(kk);
              dm_rho_d(k,j,i,iens) += rho   * wt;
              dm_uvel (k,j,i,iens) += u     * wt;
              dm_vvel (k,j,i,iens) += v     * wt;
              dm_wvel (k,j,i,iens) += w     * wt;
              dm_temp (k,j,i,iens) += T     * wt;
              dm_rho_v(k,j,i,iens) += rho_v * wt;
            }
          }
        }
        int inorm = (static_cast<int>(i_beg)+i)/cells_per_building - buildings_pad;
        int jnorm = (static_cast<int>(j_beg)+j)/cells_per_building - buildings_pad;
        if ( ( inorm >= 0 && inorm < nblocks_x*3 && inorm%3 < 2 ) &&
             ( jnorm >= 0 && jnorm < nblocks_y*9 && jnorm%9 < 8 ) ) {
          if ( k <= std::ceil( building_heights(jnorm,inorm) / dz ) ) {
            dm_immersed_proportion(k,j,i,iens) = 1;
            dm_uvel               (k,j,i,iens) = 0;
            dm_vvel               (k,j,i,iens) = 0;
            dm_wvel               (k,j,i,iens) = 0;
          }
        }
      });

    } else if (coupler.get_option<std::string>("init_data") == "building") {

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        dm_rho_d(k,j,i,iens) = 0;
        dm_uvel (k,j,i,iens) = 0;
        dm_vvel (k,j,i,iens) = 0;
        dm_wvel (k,j,i,iens) = 0;
        dm_temp (k,j,i,iens) = 0;
        dm_rho_v(k,j,i,iens) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          for (int jj=0; jj<nqpoints; jj++) {
            for (int ii=0; ii<nqpoints; ii++) {
              real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
              real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
              real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
              real rho, u, v, w, theta, rho_v, hr, ht;
              if (enable_gravity) { modules::profiles::hydro_const_bvf(z,grav,C0,cp_d,p0,gamma,R_d,hr,ht); }
              else                { hr = 1.15;     ht = 300;                              }
              rho   = hr;
              u     = 20;
              v     = 0;
              w     = 0;
              theta = ht;
              rho_v = 0;
              real T = C0*std::pow(rho*theta,gamma)/(rho*R_d);
              if (sim2d) v = 0;
              real wt = qweights(ii)*qweights(jj)*qweights(kk);
              dm_rho_d(k,j,i,iens) += rho   * wt;
              dm_uvel (k,j,i,iens) += u     * wt;
              dm_vvel (k,j,i,iens) += v     * wt;
              dm_wvel (k,j,i,iens) += w     * wt;
              dm_temp (k,j,i,iens) += T     * wt;
              dm_rho_v(k,j,i,iens) += rho_v * wt;
            }
          }
        }
        real x0 = 0.2*nx_glob;
        real y0 = 0.5*ny_glob;
        real xr = 0.05*ny_glob;
        real yr = 0.05*ny_glob;
        if ( std::abs(i_beg+i-x0) <= xr && std::abs(j_beg+j-y0) <= yr && k <= 0.3*nz ) {
          dm_immersed_proportion(k,j,i,iens) = 1;
          dm_uvel               (k,j,i,iens) = 0;
          dm_vvel               (k,j,i,iens) = 0;
          dm_wvel               (k,j,i,iens) = 0;
        }
      });

    } else if (coupler.get_option<std::string>("init_data") == "cube") {

      coupler.set_option<bool>("enable_gravity",false);
      coupler.set_option<real>("grav",0.0_fp);
      coupler.set_option<bool>("use_immersed_boundaries",true);
      coupler.add_option<std::string>("bc_x","periodic");
      coupler.add_option<std::string>("bc_y","periodic");
      coupler.add_option<std::string>("bc_z","periodic");

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        dm_rho_d(k,j,i,iens) = 0;
        dm_uvel (k,j,i,iens) = 0;
        dm_vvel (k,j,i,iens) = 0;
        dm_wvel (k,j,i,iens) = 0;
        dm_temp (k,j,i,iens) = 0;
        dm_rho_v(k,j,i,iens) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          for (int jj=0; jj<nqpoints; jj++) {
            for (int ii=0; ii<nqpoints; ii++) {
              real x = (i+i_beg+0.5)*dx + (qpoints(ii)-0.5)*dx;
              real y = (j+j_beg+0.5)*dy + (qpoints(jj)-0.5)*dy;   if (sim2d) y = ylen/2;
              real z = (k      +0.5)*dz + (qpoints(kk)-0.5)*dz;
              real hr    = 1.15;
              real ht    = 300;
              real rho   = hr;
              real u     = 10;
              real v     = 0;
              real w     = 0;
              real theta = ht;
              real rho_v = 0;
              real T     = C0*std::pow(rho*theta,gamma)/(rho*R_d);
              if (sim2d) v = 0;
              real wt = qweights(ii)*qweights(jj)*qweights(kk);
              dm_rho_d(k,j,i,iens) += rho   * wt;
              dm_uvel (k,j,i,iens) += u     * wt;
              dm_vvel (k,j,i,iens) += v     * wt;
              dm_wvel (k,j,i,iens) += w     * wt;
              dm_temp (k,j,i,iens) += T     * wt;
              dm_rho_v(k,j,i,iens) += rho_v * wt;
            }
          }
        }
        real x1 = 1*nx_glob/10;
        real y1 = 4*ny_glob/10;
        real z1 = 4*nz     /10;
        real x2 = 2*nx_glob/10-1;
        real y2 = 6*ny_glob/10-1;
        real z2 = 6*nz     /10-1;
        if ( i_beg+i >= x1 && i_beg+i <= x2 &&
             j_beg+j >= y1 && j_beg+j <= y2 &&
                   k >= z1 &&       k <= z2 ) {
          dm_immersed_proportion(k,j,i,iens) = 1;
          dm_uvel               (k,j,i,iens) = 0;
          dm_vvel               (k,j,i,iens) = 0;
          dm_wvel               (k,j,i,iens) = 0;
        }
      });
    }

  }

}

