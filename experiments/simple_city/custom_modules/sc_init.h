
#pragma once

#include "main_header.h"
#include "profiles.h"
#include "coupler.h"
#include "TransformMatrices.h"
#include "hydrostasis.h"
#include <random>

namespace custom_modules {

  inline void sc_init( core::Coupler & coupler ) {
    using yikl::parallel_for;
    using yikl::SimpleBounds;
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
      coupler.get_parallel_comm().broadcast( building_heights_host );
      auto building_heights = building_heights_host.createDeviceCopy();

      real constexpr uref       = 10;   // Velocity at hub height
      real constexpr theta0     = 300;
      real constexpr href       = 100;   // Height of hub / center of windmills
      real constexpr von_karman = 0.40;
      real slope = -grav*std::pow( p0 , R_d/cp_d ) / (cp_d*theta0);
      realHost1d press_host("press",nz);
      press_host(0) = std::pow( p0 , R_d/cp_d ) + slope*dz/2;
      for (int k=1; k < nz; k++) { press_host(k) = press_host(k-1) + slope*dz; }
      for (int k=0; k < nz; k++) { press_host(k) = std::pow( press_host(k) , cp_d/R_d ); }
      auto press = press_host.createDeviceCopy();
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real zloc = (k+0.5_fp)*dz;
        real ustar = von_karman * uref / std::log((href+roughness)/roughness);
        real u     = ustar / von_karman * std::log((zloc+roughness)/roughness);
        real p     = press(k);
        real rt    = std::pow( p/C0 , 1._fp/gamma );
        real r     = rt / theta0;
        real T     = p/R_d/r;
        dm_rho_d(k,j,i) = rt / theta0;
        dm_uvel (k,j,i) = u;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = T;
        dm_rho_v(k,j,i) = 0;
        int inorm = (static_cast<int>(i_beg)+i)/cells_per_building - buildings_pad;
        int jnorm = (static_cast<int>(j_beg)+j)/cells_per_building - buildings_pad;
        if ( ( inorm >= 0 && inorm < nblocks_x*3 && inorm%3 < 2 ) &&
             ( jnorm >= 0 && jnorm < nblocks_y*9 && jnorm%9 < 8 ) ) {
          if ( k <= std::ceil( building_heights(jnorm,inorm) / dz ) ) {
            dm_immersed_prop(k,j,i) = 1;
            dm_uvel         (k,j,i) = 0;
            dm_vvel         (k,j,i) = 0;
            dm_wvel         (k,j,i) = 0;
          }
        }
        if (k == 0) dm_surface_temp(j,i) = 300;
      });

    } else if (coupler.get_option<std::string>("init_data") == "building") {

      auto u_g = coupler.get_option<real>("geostrophic_u",10.);
      auto v_g = coupler.get_option<real>("geostrophic_v",0. );
      real constexpr uref       = 10;   // Velocity at hub height
      real constexpr theta0     = 300;
      real constexpr href       = 100;   // Height of hub / center of windmills
      real constexpr von_karman = 0.40;
      real1d press("press",nz);
      if (enable_gravity) {
        real slope = -grav*std::pow( p0 , R_d/cp_d ) / (cp_d*theta0);
        realHost1d press_host("press",nz);
        press_host(0) = std::pow( p0 , R_d/cp_d ) + slope*dz/2;
        for (int k=1; k < nz; k++) { press_host(k) = press_host(k-1) + slope*dz; }
        for (int k=0; k < nz; k++) { press_host(k) = std::pow( press_host(k) , cp_d/R_d ); }
        press = press_host.createDeviceCopy();
      } else {
        press = p0;
      }
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real zloc = (k+0.5_fp)*dz;
        real p     = press(k);
        real rt    = std::pow( p/C0 , 1._fp/gamma );
        real r     = rt / theta0;
        real T     = p/R_d/r;
        dm_rho_d(k,j,i) = rt / theta0;
        dm_uvel (k,j,i) = u_g;
        dm_vvel (k,j,i) = v_g;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = T;
        dm_rho_v(k,j,i) = 0;
        real x0 = 0.2 *nx_glob;
        real y0 = 0.5 *ny_glob;
        real xr = 0.05*ny_glob;
        real yr = 0.05*ny_glob;
        if ( std::abs(i_beg+i-x0) <= xr && std::abs(j_beg+j-y0) <= yr && k <= 0.3*nz ) {
          dm_immersed_prop(k,j,i) = 1;
          dm_uvel         (k,j,i) = 0;
          dm_vvel         (k,j,i) = 0;
          dm_wvel         (k,j,i) = 0;
        }
        if (k == 0) dm_surface_temp(j,i) = 300;
      });

    } else if (coupler.get_option<std::string>("init_data") == "sphere") {

      real constexpr uref       = 10;   // Velocity at hub height
      real constexpr theta0     = 300;
      real constexpr href       = 100;   // Height of hub / center of windmills
      real constexpr von_karman = 0.40;
      real slope = -grav*std::pow( p0 , R_d/cp_d ) / (cp_d*theta0);
      realHost1d press_host("press",nz);
      press_host(0) = std::pow( p0 , R_d/cp_d ) + slope*dz/2;
      for (int k=1; k < nz; k++) { press_host(k) = press_host(k-1) + slope*dz; }
      for (int k=0; k < nz; k++) { press_host(k) = std::pow( press_host(k) , cp_d/R_d ); }
      auto press = press_host.createDeviceCopy();
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real zloc = (k+0.5_fp)*dz;
        real ustar = von_karman * uref / std::log((href+roughness)/roughness);
        real u     = ustar / von_karman * std::log((zloc+roughness)/roughness);
        real p     = press(k);
        real rt    = std::pow( p/C0 , 1._fp/gamma );
        real r     = rt / theta0;
        real T     = p/R_d/r;
        dm_rho_d(k,j,i) = rt / theta0;
        dm_uvel (k,j,i) = u;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = T;
        dm_rho_v(k,j,i) = 0;
        real x0 = 0.5 *xlen;
        real y0 = 0.5 *ylen;
        real z0 = 0.2 *zlen;
        real rad = 0.05*zlen;
        int N = 10;
        int count = 0;
        for (int kk=0; kk < N; kk++) {
          for (int jj=0; jj < N; jj++) {
            for (int ii=0; ii < N; ii++) {
              real x = (i_beg+i)*dx+ii*dx/(N-1);
              real y = (j_beg+j)*dy+jj*dy/(N-1);
              real z = (      k)*dz+kk*dz/(N-1);
              if ( (x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0) <= rad*rad ) count++;
            }
          }
        }
        dm_immersed_prop(k,j,i) = static_cast<real>(count)/(N*N*N);
        dm_uvel         (k,j,i) *= (1-dm_immersed_prop(k,j,i));
        dm_vvel         (k,j,i) *= (1-dm_immersed_prop(k,j,i));
        dm_wvel         (k,j,i) *= (1-dm_immersed_prop(k,j,i));
        if (count == N*N*N) dm_immersed_prop(k,j,i) = 1;
        if (k == 0) dm_surface_temp(j,i) = 300;
      });

    } else if (coupler.get_option<std::string>("init_data") == "constant") {

      coupler.set_option<std::string>("bc_x","precursor");
      coupler.set_option<std::string>("bc_y","precursor");
      coupler.set_option<std::string>("bc_z","solid_wall");
      coupler.set_option<bool>("enable_gravity",false);
      real u  = coupler.get_option<real>( "constant_uvel"  , 10.  );
      real v  = coupler.get_option<real>( "constant_vvel"  , 0.   );
      real w  = 0;
      real T  = coupler.get_option<real>( "constant_temp"  , 300. );
      real p  = coupler.get_option<real>( "constant_press" , 1.e5 );
      real r  = p/(R_d*T);
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = r;
        dm_uvel (k,j,i) = u;
        dm_vvel (k,j,i) = v;
        dm_wvel (k,j,i) = w;
        dm_temp (k,j,i) = T;
        dm_rho_v(k,j,i) = 0;
        if (k == 0) dm_surface_temp(j,i) = T;
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_neutral") {

      auto compute_theta = KOKKOS_LAMBDA (real z) -> real {
        if      (z <  500)            { return 300;                        }
        else if (z >= 500 && z < 650) { return 300+0.08*(z-500);           }
        else                          { return 300+0.08*150+0.003*(z-650); }
      };
      auto pressGLL = modules::integrate_hydrostatic_pressure_gll_theta(compute_theta,nz,zlen,p0,grav,R_d,cp_d).createDeviceCopy();
      auto u_g = coupler.get_option<real>("geostrophic_u",10.);
      auto v_g = coupler.get_option<real>("geostrophic_v",0. );
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = 0;
        dm_uvel (k,j,i) = 0;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = 0;
        dm_rho_v(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          real z         = (k+0.5)*dz + qpoints(kk)*dz;
          real theta     = compute_theta(z);
          real p         = pressGLL(k,kk);
          real rho_theta = std::pow( p/C0 , 1._fp/gamma );
          real rho       = rho_theta / theta;
          real u         = u_g;
          real v         = v_g;
          real w         = 0;
          real T         = p/(rho*R_d);
          real rho_v     = 0;
          real wt = qweights(kk);
          dm_rho_d(k,j,i) += rho   * wt;
          dm_uvel (k,j,i) += u     * wt;
          dm_vvel (k,j,i) += v     * wt;
          dm_wvel (k,j,i) += w     * wt;
          dm_temp (k,j,i) += T     * wt;
          dm_rho_v(k,j,i) += rho_v * wt;
        }
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        if ((k+0.5_fp)*dz <= 400) dm_temp(k,j,i) += rand.genFP<real>(-0.25,0.25);
        if (k == 0) dm_surface_temp(j,i) = 300;
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_convective") {

      auto compute_theta = KOKKOS_LAMBDA (real z) -> real {
        if   (z <  600) { return 309;               }
        else            { return 309+0.004*(z-600); }
      };
      auto pressGLL = modules::integrate_hydrostatic_pressure_gll_theta(compute_theta,nz,zlen,p0,grav,R_d,cp_d).createDeviceCopy();
      auto u_g = coupler.get_option<real>("geostrophic_u",9.);
      auto v_g = coupler.get_option<real>("geostrophic_v",0.);
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = 0;
        dm_uvel (k,j,i) = 0;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = 0;
        dm_rho_v(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          real z         = (k+0.5)*dz + qpoints(kk)*dz;
          real theta     = compute_theta(z);
          real p         = pressGLL(k,kk);
          real rho_theta = std::pow( p/C0 , 1._fp/gamma );
          real rho       = rho_theta / theta;
          real u         = u_g;
          real v         = v_g;
          real w         = 0;
          real T         = p/(rho*R_d);
          real rho_v     = 0;
          real wt = qweights(kk);
          dm_rho_d(k,j,i) += rho   * wt;
          dm_uvel (k,j,i) += u     * wt;
          dm_vvel (k,j,i) += v     * wt;
          dm_wvel (k,j,i) += w     * wt;
          dm_temp (k,j,i) += T     * wt;
          dm_rho_v(k,j,i) += rho_v * wt;
        }
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        if ((k+0.5_fp)*dz <= 400) dm_temp(k,j,i) += rand.genFP<real>(-0.25,0.25);
        if (k == 0) dm_surface_temp(j,i) = 309;
        if (k == 0) dm_surface_khf (j,i) = 0.35/cp_d/dm_rho_d(k,j,i);
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_stable") {

      auto compute_theta = KOKKOS_LAMBDA (real z) -> real {
        if   (z <  100) { return 265;              }
        else            { return 265+0.01*(z-100); }
      };
      auto pressGLL = modules::integrate_hydrostatic_pressure_gll_theta(compute_theta,nz,zlen,p0,grav,R_d,cp_d).createDeviceCopy();
      auto u_g = coupler.get_option<real>("geostrophic_u",8.);
      auto v_g = coupler.get_option<real>("geostrophic_v",0.);
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = 0;
        dm_uvel (k,j,i) = 0;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = 0;
        dm_rho_v(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          real z         = (k+0.5)*dz + qpoints(kk)*dz;
          real theta     = compute_theta(z);
          real p         = pressGLL(k,kk);
          real rho_theta = std::pow( p/C0 , 1._fp/gamma );
          real rho       = rho_theta / theta;
          real u         = u_g;
          real v         = v_g;
          real w         = 0;
          real T         = p/(rho*R_d);
          real rho_v     = 0;
          real wt = qweights(kk);
          dm_rho_d(k,j,i) += rho   * wt;
          dm_uvel (k,j,i) += u     * wt;
          dm_vvel (k,j,i) += v     * wt;
          dm_wvel (k,j,i) += w     * wt;
          dm_temp (k,j,i) += T     * wt;
          dm_rho_v(k,j,i) += rho_v * wt;
        }
        yakl::Random rand(k*ny_glob*nx_glob + (j_beg+j)*nx_glob + (i_beg+i));
        if ((k+0.5_fp)*dz <= 50) dm_temp(k,j,i) += rand.genFP<real>(-0.10,0.10);
        if (k == 0) dm_surface_temp(j,i) = 265;
        if (k == 0) dm_surface_khf (j,i) = 0;
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_stable_bvf") {

      real theta0 = 300;
      auto bvf    = coupler.get_option<real>("bvf_freq",0.01);
      auto compute_theta = KOKKOS_LAMBDA (real z) -> real {
        return theta0*std::exp(bvf*bvf/grav*z);
      };
      auto pressGLL = modules::integrate_hydrostatic_pressure_gll_theta(compute_theta,nz,zlen,p0,grav,R_d,cp_d).createDeviceCopy();
      real uref     = coupler.get_option<real>("hub_height_wind_mag",12); // Velocity at hub height
      real href     = coupler.get_option<real>("turbine_hub_height",90);  // Height of hub / center of windmills
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = 0;
        dm_uvel (k,j,i) = 0;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = 0;
        dm_rho_v(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          real z         = (k+0.5)*dz + qpoints(kk)*dz;
          real theta     = compute_theta(z);
          real p         = pressGLL(k,kk);
          real rho_theta = std::pow( p/C0 , 1._fp/gamma );
          real rho       = rho_theta / theta;
          real ustar     = uref / std::log((href+roughness)/roughness);
          real u         = ustar * std::log((z+roughness)/roughness);
          real v         = 0;
          real w         = 0;
          real T         = p/(rho*R_d);
          real rho_v     = 0;
          real wt = qweights(kk);
          dm_rho_d(k,j,i) += rho   * wt;
          dm_uvel (k,j,i) += u     * wt;
          dm_vvel (k,j,i) += v     * wt;
          dm_wvel (k,j,i) += w     * wt;
          dm_temp (k,j,i) += T     * wt;
          dm_rho_v(k,j,i) += rho_v * wt;
        }
        if (k == 0) dm_surface_temp(j,i) = theta0;
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_neutral2") {

      real uref       = coupler.get_option<real>("hub_height_wind_mag",12); // Velocity at hub height
      real theta0     = 300;
      real href       = coupler.get_option<real>("turbine_hub_height",90);  // Height of hub / center of windmills
      real slope = -grav*std::pow( p0 , R_d/cp_d ) / (cp_d*theta0);
      realHost1d press_host("press",nz);
      press_host(0) = std::pow( p0 , R_d/cp_d ) + slope*dz/2;
      for (int k=1; k < nz; k++) { press_host(k) = press_host(k-1) + slope*dz; }
      for (int k=0; k < nz; k++) { press_host(k) = std::pow( press_host(k) , cp_d/R_d ); }
      auto press = press_host.createDeviceCopy();
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real zloc = (k+0.5_fp)*dz;
        real ustar = uref / std::log((href+roughness)/roughness);
        real u     = ustar * std::log((zloc+roughness)/roughness);
        real p     = press(k);
        real rt    = std::pow( p/C0 , 1._fp/gamma );
        real r     = rt / theta0;
        real T     = p/R_d/r;
        dm_rho_d(k,j,i) = rt / theta0;
        dm_uvel (k,j,i) = u;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = T;
        dm_rho_v(k,j,i) = 0;
        if (k == 0) dm_surface_temp(j,i) = 300;
      });

    } else if (coupler.get_option<std::string>("init_data") == "AWAKEN_neutral") {

      real uref   = 6.27;
      real angle  = 4.33;
      real theta0 = 305;
      real href   = 89;
      real pwr    = 0.116;
      real slope  = -grav*std::pow( p0 , R_d/cp_d ) / (cp_d*theta0);
      realHost1d press_host("press",nz);
      press_host(0) = std::pow( p0 , R_d/cp_d ) + slope*dz/2;
      for (int k=1; k < nz; k++) { press_host(k) = press_host(k-1) + slope*dz; }
      for (int k=0; k < nz; k++) { press_host(k) = std::pow( press_host(k) , cp_d/R_d ); }
      auto press = press_host.createDeviceCopy();
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        real zloc = (k+0.5_fp)*dz;
        real ustar = uref;
        real u     = ustar * std::pow( zloc/href , pwr );
        real p     = press(k);
        real rt    = std::pow( p/C0 , 1._fp/gamma );
        real r     = rt / theta0;
        real T     = p/R_d/r;
        dm_rho_d(k,j,i) = rt / theta0;
        dm_uvel (k,j,i) = u*std::cos(angle/180*M_PI);
        dm_vvel (k,j,i) = u*std::sin(angle/180*M_PI);
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = T;
        dm_rho_v(k,j,i) = 0;
        if (k == 0) dm_surface_temp(j,i) = theta0;
      });

    } else if (coupler.get_option<std::string>("init_data") == "supercell") {

      auto compute_temp = KOKKOS_LAMBDA (real z) -> real {
        if (z < 12000) { return 300.+(213.-300.)/12000.*z; }
        else           { return 213.; }
      };
      auto compute_qv = KOKKOS_LAMBDA (real z) -> real {
        if (z < 12000) { return std::min(0.014_fp,std::pow(-3.17e-5_fp*z+0.53_fp,6._fp)); }
        else           { return std::pow(4.62e-6_fp*z+0.1_fp,6._fp); }
      };
      auto pressGLL = modules::integrate_hydrostatic_pressure_gll_temp_qv(compute_temp,compute_qv,nz,zlen,p0,grav,R_d,R_v).createDeviceCopy();
      parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , KOKKOS_LAMBDA (int k, int j, int i) {
        dm_rho_d(k,j,i) = 0;
        dm_uvel (k,j,i) = 0;
        dm_vvel (k,j,i) = 0;
        dm_wvel (k,j,i) = 0;
        dm_temp (k,j,i) = 0;
        dm_rho_v(k,j,i) = 0;
        for (int kk=0; kk<nqpoints; kk++) {
          real z         = (k+0.5)*dz + qpoints(kk)*dz;
          real T         = compute_temp(z);
          real qv        = compute_qv(z);
          real p         = pressGLL(k,kk);
          real rho_d     = p/((R_d+qv*R_v)*T);
          real u         = z < 5000 ? -15+30*z/5000 : 15;
          real v         = 0;
          real w         = 0;
          real rho_v     = qv*rho_d;
          real wt = qweights(kk);
          dm_rho_d(k,j,i) += rho_d * wt;
          dm_uvel (k,j,i) += u     * wt;
          dm_vvel (k,j,i) += v     * wt;
          dm_wvel (k,j,i) += w     * wt;
          dm_temp (k,j,i) += T     * wt;
          dm_rho_v(k,j,i) += rho_v * wt;
        }
        if (k == 0) dm_surface_temp(j,i) = 300;
        if (k == 0) dm_surface_khf (j,i) = 0;
        real xloc = (i+i_beg+0.5_fp)*dx;
        real yloc = (j+j_beg+0.5_fp)*dy;
        real zloc = (k      +0.5_fp)*dz;
        real x0 = xlen / 2;
        real y0 = ylen / 2;
        real z0 = 1500;
        real radx = 10000;
        real rady = 10000;
        real radz = 1500;
        real amp  = 5;
        real xn = (xloc - x0) / radx;
        real yn = (yloc - y0) / rady;
        real zn = (zloc - z0) / radz;
        real rad = sqrt( xn*xn + yn*yn + zn*zn );
        if (rad < 1) dm_temp(k,j,i) += amp * pow( cos(M_PI*rad/2) , 2._fp );
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
    parallel_for( YIKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) , KOKKOS_LAMBDA (int kk, int j, int i) {
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


