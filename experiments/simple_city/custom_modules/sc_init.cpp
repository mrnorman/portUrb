
#include "sc_init.h"

namespace custom_modules {

  void sc_init( core::Coupler & coupler ) {
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

    const int nqpoints = 9;
    SArray<real,1,nqpoints> qpoints;
    SArray<real,1,nqpoints> qweights;
    TransformMatrices::get_gll_points (qpoints );
    TransformMatrices::get_gll_weights(qweights);

    coupler.add_option<std::string>("bc_x","periodic");
    coupler.add_option<std::string>("bc_y","periodic");
    coupler.add_option<std::string>("bc_z","solid_wall");
    coupler.add_option<bool       >("enable_gravity",true);

    YAML::Node config = YAML::LoadFile(coupler.get_option<std::string>("standalone_input_file"));

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
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
      });

    } else if (coupler.get_option<std::string>("init_data") == "building") {

      auto u_g = config["geostrophic_u"].as<real>(10.);
      auto v_g = config["geostrophic_v"].as<real>(0. );
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
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_neutral") {

      auto compute_theta = YAKL_LAMBDA (real z) -> real {
        if      (z <  500)            { return 300;                        }
        else if (z >= 500 && z < 650) { return 300+0.08*(z-500);           }
        else                          { return 300+0.08*150+0.003*(z-650); }
      };
      // Integrate RHS over GLL interval using GLL quadrature
      real cst = -grav*std::pow( p0 , R_d/cp_d ) / cp_d;
      real2d rhs("rhs",nz,nqpoints-1);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,nqpoints-1) , YAKL_LAMBDA (int k1, int k2) {
        real z1 = (k1+0.5_fp)*dz + qpoints(k2  )*dz;
        real z2 = (k1+0.5_fp)*dz + qpoints(k2+1)*dz;
        rhs(k1,k2) = 0;
        for (int k3 = 0; k3 < nqpoints; k3++) {
          real z = 0.5_fp*(z1+z2) + qpoints(k3)*(z2-z1);
          rhs(k1,k2) += cst/compute_theta(z) * qweights(k3);
        }
        rhs(k1,k2) *= (z2-z1);
      });
      auto rhs_host = rhs.createHostCopy();
      realHost2d pressGLL_host("pressGLL",nz,nqpoints);
      // Sum the pressure using RHS, and apply the correct power. Prefix sum over low dimensions, so do on host
      pressGLL_host(0,0) = std::pow( p0 , R_d/cp_d );
      for (int k = 0; k < nz; k++) {
        for (int kk = 0; kk < nqpoints-1; kk++) {
          pressGLL_host(k,kk+1) = pressGLL_host(k,kk) + rhs_host(k,kk);
        }
        if (k < nz-1) {
          pressGLL_host(k+1,0) = pressGLL_host(k,nqpoints-1);
        }
      }
      for (int k = 0; k < nz; k++) {
        for (int kk = 0; kk < nqpoints; kk++) { 
          pressGLL_host(k,kk) = std::pow( pressGLL_host(k,kk) , cp_d/R_d );
        }
      }
      auto pressGLL = pressGLL_host.createDeviceCopy();
      auto u_g = config["geostrophic_u"].as<real>(10.);
      auto v_g = config["geostrophic_v"].as<real>(0. );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_convective") {

      auto compute_theta = YAKL_LAMBDA (real z) -> real {
        if   (z <  600) { return 309;               }
        else            { return 309+0.004*(z-600); }
      };
      // Integrate RHS over GLL interval using GLL quadrature
      real cst = -grav*std::pow( p0 , R_d/cp_d ) / cp_d;
      real2d rhs("rhs",nz,nqpoints-1);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,nqpoints-1) , YAKL_LAMBDA (int k1, int k2) {
        real z1 = (k1+0.5_fp)*dz + qpoints(k2  )*dz;
        real z2 = (k1+0.5_fp)*dz + qpoints(k2+1)*dz;
        rhs(k1,k2) = 0;
        for (int k3 = 0; k3 < nqpoints; k3++) {
          real z = 0.5_fp*(z1+z2) + qpoints(k3)*(z2-z1);
          rhs(k1,k2) += cst/compute_theta(z) * qweights(k3);
        }
        rhs(k1,k2) *= (z2-z1);
      });
      auto rhs_host = rhs.createHostCopy();
      realHost2d pressGLL_host("pressGLL",nz,nqpoints);
      // Sum the pressure using RHS, and apply the correct power. Prefix sum over low dimensions, so do on host
      pressGLL_host(0,0) = std::pow( p0 , R_d/cp_d );
      for (int k = 0; k < nz; k++) {
        for (int kk = 0; kk < nqpoints-1; kk++) {
          pressGLL_host(k,kk+1) = pressGLL_host(k,kk) + rhs_host(k,kk);
        }
        if (k < nz-1) {
          pressGLL_host(k+1,0) = pressGLL_host(k,nqpoints-1);
        }
      }
      for (int k = 0; k < nz; k++) {
        for (int kk = 0; kk < nqpoints; kk++) { 
          pressGLL_host(k,kk) = std::pow( pressGLL_host(k,kk) , cp_d/R_d );
        }
      }
      auto pressGLL = pressGLL_host.createDeviceCopy();
      auto u_g = config["geostrophic_u"].as<real>(10.);
      auto v_g = config["geostrophic_v"].as<real>(0. );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_stable") {

      auto compute_theta = YAKL_LAMBDA (real z) -> real {
        if   (z <  100) { return 265;              }
        else            { return 265+0.01*(z-100); }
      };
      // Integrate RHS over GLL interval using GLL quadrature
      real cst = -grav*std::pow( p0 , R_d/cp_d ) / cp_d;
      real2d rhs("rhs",nz,nqpoints-1);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,nqpoints-1) , YAKL_LAMBDA (int k1, int k2) {
        real z1 = (k1+0.5_fp)*dz + qpoints(k2  )*dz;
        real z2 = (k1+0.5_fp)*dz + qpoints(k2+1)*dz;
        rhs(k1,k2) = 0;
        for (int k3 = 0; k3 < nqpoints; k3++) {
          real z = 0.5_fp*(z1+z2) + qpoints(k3)*(z2-z1);
          rhs(k1,k2) += cst/compute_theta(z) * qweights(k3);
        }
        rhs(k1,k2) *= (z2-z1);
      });
      auto rhs_host = rhs.createHostCopy();
      realHost2d pressGLL_host("pressGLL",nz,nqpoints);
      // Sum the pressure using RHS, and apply the correct power. Prefix sum over low dimensions, so do on host
      pressGLL_host(0,0) = std::pow( p0 , R_d/cp_d );
      for (int k = 0; k < nz; k++) {
        for (int kk = 0; kk < nqpoints-1; kk++) {
          pressGLL_host(k,kk+1) = pressGLL_host(k,kk) + rhs_host(k,kk);
        }
        if (k < nz-1) {
          pressGLL_host(k+1,0) = pressGLL_host(k,nqpoints-1);
        }
      }
      for (int k = 0; k < nz; k++) {
        for (int kk = 0; kk < nqpoints; kk++) { 
          pressGLL_host(k,kk) = std::pow( pressGLL_host(k,kk) , cp_d/R_d );
        }
      }
      auto pressGLL = pressGLL_host.createDeviceCopy();
      auto u_g = config["geostrophic_u"].as<real>(10.);
      auto v_g = config["geostrophic_v"].as<real>(0. );
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
      });

    } else if (coupler.get_option<std::string>("init_data") == "ABL_neutral2") {

      real constexpr uref       = 12;   // Velocity at hub height
      real constexpr theta0     = 300;
      real constexpr href       = 150;   // Height of hub / center of windmills
      real constexpr von_karman = 0.40;
      real slope = -grav*std::pow( p0 , R_d/cp_d ) / (cp_d*theta0);
      realHost1d press_host("press",nz);
      press_host(0) = std::pow( p0 , R_d/cp_d ) + slope*dz/2;
      for (int k=1; k < nz; k++) { press_host(k) = press_host(k-1) + slope*dz; }
      for (int k=0; k < nz; k++) { press_host(k) = std::pow( press_host(k) , cp_d/R_d ); }
      auto press = press_host.createDeviceCopy();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
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
      });

    }

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
    auto immersed_khf_halos        = dm.get<real,3>("immersed_khf_halos"       );
    auto immersed_roughness_halos  = dm.get<real,3>("immersed_roughness_halos" );
    if (coupler.get_option<std::string>("bc_z") == "solid_wall") {
      auto khf = config["convective_khf"].as<real>(0);
      bool abl_convective = coupler.get_option<std::string>("init_data") == "ABL_convective";
      bool abl_stable     = coupler.get_option<std::string>("init_data") == "ABL_stable"    ;
      if (coupler.is_mainproc()) std::cout << "Heat flux (K/s): " << khf*1003*3600 << std::endl;
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(hs,ny+2*hs,nx+2*hs) , YAKL_LAMBDA (int kk, int j, int i) {
        immersed_proportion_halos(      kk,j,i) = 1;
        immersed_proportion_halos(hs+nz+kk,j,i) = 1;
        immersed_roughness_halos (      kk,j,i) = roughness;
        immersed_roughness_halos (hs+nz+kk,j,i) = 0;
        if (abl_convective) immersed_khf_halos(kk,j,i) = khf;
      });
    }
  }

}


