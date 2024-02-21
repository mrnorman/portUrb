
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  struct WindmillActuators {

    int static constexpr MAX_TRACES = 1e7;
    realHost2d power_trace;
    int num_traces;

    void init( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx      = coupler.get_nx  ();
      auto ny      = coupler.get_ny  ();
      auto nz      = coupler.get_nz  ();
      auto nens    = coupler.get_nens();
      auto dx      = coupler.get_dx  ();
      auto dy      = coupler.get_dy  ();
      auto dz      = coupler.get_dz  ();
      auto xlen    = coupler.get_xlen();
      auto ylen    = coupler.get_ylen();
      auto i_beg   = coupler.get_i_beg();
      auto j_beg   = coupler.get_j_beg();
      auto nx_glob = coupler.get_nx_glob();
      auto ny_glob = coupler.get_ny_glob();
      auto &dm     = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<real>( "windmill_prop" , "" , {nz,ny,nx,nens} , {"z","y","x","nens"} );
      auto windmill_prop = dm.get<real,4>("windmill_prop");
      windmill_prop = 0;
      real z0  = 150;
      real rad = 121.118878225;
      int ny_t = std::ceil(rad*2/dy);
      int nz_t = std::ceil((z0+rad)/dz);
      real y0 = ny_t*dy/2;
      int constexpr N = 1000;
      real ddy = dy / N;
      real ddz = dz / N;
      realHost2d templ_host("templ",nz_t,ny_t);
      size_t count_total = 0;
      for (int k=0; k < nz_t; k++) {
        for (int j=0; j < ny_t; j++) {
          int count = 0;
          for (int kk=0; kk < N; kk++) {
            for (int jj=0; jj < N; jj++) {
              real z = k*dz + kk*ddz;
              real y = j*dy + jj*ddy;
              if (std::sqrt((z-z0)*(z-z0)+(y-y0)*(y-y0))/rad < 1) { count++; count_total++; }
            }
          }
          templ_host(k,j) = static_cast<double>(count);
        }
      }
      for (int k=0; k < nz_t; k++) {
        for (int j=0; j < ny_t; j++) {
          templ_host(k,j) = M_PI*rad*rad*templ_host(k,j)/static_cast<double>(count_total)/(dx*dy*dz);
        }
      }
      auto templ = templ_host.createDeviceCopy();
      std::vector<int> xind_vec ;
      std::vector<int> yind0_vec;
      for (int i=nx_glob/4; i <= 3*nx_glob/4; i += (int) std::ceil(rad*2*10/dx)) { xind_vec .push_back(i); }
      for (int j=ny_glob/4; j <= 3*ny_glob/4; j += (int) std::ceil(rad*2*5 /dy)) { yind0_vec.push_back(j); }
      intHost1d xind_host ("wf_xind" ,xind_vec .size());
      intHost1d yind0_host("wf_yind0",yind0_vec.size());
      for (int i=0; i < xind_vec .size(); i++) { xind_host (i) = xind_vec [i]; }
      for (int j=0; j < yind0_vec.size(); j++) { yind0_host(j) = yind0_vec[j]; }
      auto xind  = xind_host .createDeviceCopy();
      auto yind0 = yind0_host.createDeviceCopy();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz_t,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        // Loop over windmills
        for (int jj=0; jj < yind0.size(); jj++) {
          for (int ii=0; ii < xind .size(); ii++) {
            // If we're at the right index for a given windmill, then copy in the template
            if (i_beg+i == xind(ii) && j_beg+j >= yind0(jj) && j_beg+j <= yind0(jj)+ny_t-1) {
              windmill_prop(k,j,i,iens) = templ(k,j_beg+j-yind0(jj));
            }
          }
        }
      });
      if (coupler.get_myrank() == 0) {
        power_trace = realHost2d("power_trace",MAX_TRACES,nens);
        num_traces = 0;
      }
      coupler.register_write_output_module( [=] (core::Coupler &coupler, yakl::SimplePNetCDF &nc) {
        nc.redef();
        nc.create_var<real>( "windmill_prop" , {"z","y","x","ens"});
        nc.enddef();
        if (num_traces > 0) {
          nc.redef();
          auto nens = coupler.get_nens();
          nc.create_dim( "num_time_steps" , num_traces );
          nc.create_var<real>( "power_trace" , {"num_time_steps","ens"} );
          nc.enddef();
          nc.begin_indep_data();
          if (coupler.get_myrank() == 0) {
            nc.write( power_trace.subset_slowest_dimension(0,num_traces-1) , "power_trace" );
          }
          nc.end_indep_data();
        }
        auto i_beg = coupler.get_i_beg();
        auto j_beg = coupler.get_j_beg();
        std::vector<MPI_Offset> start_3d = {0,(MPI_Offset)j_beg,(MPI_Offset)i_beg,0};
        nc.write_all(coupler.get_data_manager_readonly().get<real const,4>("windmill_prop"),"windmill_prop",start_3d);
      });
    }


    void apply( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx             = coupler.get_nx  ();
      auto ny             = coupler.get_ny  ();
      auto nz             = coupler.get_nz  ();
      auto nens           = coupler.get_nens();
      auto dx             = coupler.get_dx  ();
      auto dy             = coupler.get_dy  ();
      auto dz             = coupler.get_dz  ();
      auto &dm            = coupler.get_data_manager_readwrite();
      auto windmill_prop  = dm.get<real const,4>("windmill_prop");
      auto rho_d          = dm.get<real const,4>("density_dry");
      auto uvel           = dm.get<real      ,4>("uvel"       );
      auto vvel           = dm.get<real      ,4>("vvel"       );
      auto wvel           = dm.get<real      ,4>("wvel"       );
      auto tke            = dm.get<real      ,4>("TKE"        );
      auto tend_u         = uvel.createDeviceObject();
      auto tend_v         = vvel.createDeviceObject();
      auto tend_w         = wvel.createDeviceObject();
      auto tend_tke       = tke .createDeviceObject();
      real4d windmill_power("windmill_power",nens,nz,ny,nx);
      real rad = 121.118878225;

      real2d ref_umag, ref_thr_coef, ref_pow_coef, ref_pow;
      get_turbine_table(ref_umag, ref_thr_coef, ref_pow_coef, ref_pow, nens);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        real r     = rho_d(k,j,i,iens);
        real u     = uvel (k,j,i,iens);
        real v     = vvel (k,j,i,iens);
        real w     = wvel (k,j,i,iens);
        real mag   = std::sqrt(u*u + v*v + w*w);
        real C_T;
        real a = 0.3;
        // Iterate to ensure we get a converged estimate of C_T and a since they depend on each other
        for (int iter = 0; iter < 20; iter++) {
          C_T = interp( ref_umag , ref_thr_coef , mag/(1-a) , iens );
          a   = 0.5_fp * ( 1 - std::sqrt(1-C_T) );
        }
        real C_P   = interp( ref_umag , ref_pow_coef , mag/(1-a) , iens );
        real pwr   = interp( ref_umag , ref_pow      , mag/(1-a) , iens );
        real f_TKE = 0.25_fp; // Recommended by Archer et al., 2020, MWR "Two corrections for TKE ..."
        real C_TKE = f_TKE * (C_T - C_P);
        real mag0  = mag/(1-a);
        real u0    = u  /(1-a);
        real v0    = v  /(1-a);
        real w0    = w  /(1-a);
        tend_u        (k,j,i,iens) = -0.5_fp  *C_T  *mag0*u0       *windmill_prop(k,j,i,iens);
        tend_v        (k,j,i,iens) = -0.5_fp  *C_T  *mag0*v0       *windmill_prop(k,j,i,iens);
        tend_w        (k,j,i,iens) = -0.5_fp  *C_T  *mag0*w0       *windmill_prop(k,j,i,iens);
        tend_tke      (k,j,i,iens) =  0.5_fp*r*C_TKE*mag0*mag0*mag0*windmill_prop(k,j,i,iens);
        windmill_power(iens,k,j,i) = pwr/(M_PI*rad*rad)*dx*dy*dz   *windmill_prop(k,j,i,iens);
      });

      // Save the total power produced on the grid
      realHost1d power_loc("power_loc",nens);
      for (int iens = 0; iens < nens; iens++) {
        power_loc(iens) = yakl::intrinsics::sum(windmill_power.slice<3>(iens,0,0,0));
      }
      auto power_glob = power_loc.createHostObject();
      MPI_Reduce( power_loc.data() , power_glob.data() , nens , coupler.get_mpi_data_type() , MPI_SUM , 0 , MPI_COMM_WORLD );
      if (coupler.get_myrank() == 0) {
        for (int iens=0; iens < nens; iens++) {
          power_trace(num_traces,iens) = power_glob(iens);
          num_traces++;
        }
      }

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        uvel(k,j,i,iens) += dt * tend_u  (k,j,i,iens);
        vvel(k,j,i,iens) += dt * tend_v  (k,j,i,iens);
        wvel(k,j,i,iens) += dt * tend_w  (k,j,i,iens);
        tke (k,j,i,iens) += dt * tend_tke(k,j,i,iens);
      });
    }


    YAKL_INLINE static real interp( real2d const &ref_umag , real2d const &ref_var , real umag , int iens ) {
      int imax = ref_umag.extent(0)-1; // Max index for the table
      // If umag exceeds the bounds of the reference data, the turbine is idle and producing no power
      if ( umag < ref_umag(0,iens) || umag > ref_umag(imax,iens) ) return 0;
      // Find the index such that umag lies between ref_umag(i,iens) and ref_umag(i+1,iens)
      int i = 0;
      while (umag > ref_umag(i,iens)) { i++; }
      if (i > 0) i--;
      // Linear interpolation: higher weight for left if it's closer to left
      real fac = (ref_umag(i+1,iens) - umag) / (ref_umag(i+1,iens)-ref_umag(i,iens));
      return fac*ref_var(i,iens) + (1-fac)*ref_var(i+1,iens);
    }


    // TODO: Hard coded for the moment until I implement Fitch file reading during init
    // Currently using a recommended floating offshore turbine
    void get_turbine_table(real2d &ref_umag, real2d &ref_thr_coef, real2d &ref_pow_coef, real2d &ref_pow, int nens) {
      realHost2d umag    ("ref_umag"    ,50,nens);
      realHost2d thr_coef("ref_thr_coef",50,nens);
      realHost2d pow_coef("ref_pow_coef",50,nens);
      realHost2d pow     ("ref_pow"     ,50,nens);
      for (int iens = 0; iens < nens; iens++) {
        umag( 0,iens) = 3;
        umag( 1,iens) = 3.5495323704249 ;
        umag( 2,iens) = 4.06790077095851;
        umag( 3,iens) = 4.55390684810376;
        umag( 4,iens) = 5.0064270629228 ;
        umag( 5,iens) = 5.42441528841141;
        umag( 6,iens) = 5.8069052279141 ;
        umag( 7,iens) = 6.15301264898898;
        umag( 8,iens) = 6.4619374275583 ;
        umag( 9,iens) = 6.73296539761899;
        umag(10,iens) = 6.96547000223702;
        umag(11,iens) = 7.158913742009  ;
        umag(12,iens) = 7.31284941764227;
        umag(13,iens) = 7.42692116378119;
        umag(14,iens) = 7.50086527168933;
        umag(15,iens) = 7.53451079888601;
        umag(16,iens) = 7.54124163344443;
        umag(17,iens) = 7.58833326955188;
        umag(18,iens) = 7.67567684172437;
        umag(19,iens) = 7.80307043087297;
        umag(20,iens) = 7.97021953109627;
        umag(21,iens) = 8.17673773051311;
        umag(22,iens) = 8.42214760456168;
        umag(23,iens) = 8.70588181969988;
        umag(24,iens) = 9.02728444495546;
        umag(25,iens) = 9.38561246829382;
        umag(26,iens) = 9.78003751429813;
        umag(27,iens) = 10.2096477591907;
        umag(28,iens) = 10.658458087787 ;
        umag(29,iens) = 10.6734500387685;
        umag(30,iens) = 11.1703721443803;
        umag(31,iens) = 11.699265301636 ;
        umag(32,iens) = 12.2589068261207;
        umag(33,iens) = 12.8480029499711;
        umag(34,iens) = 13.4651918127816;
        umag(35,iens) = 14.1090466099259;
        umag(36,iens) = 14.7780788910142;
        umag(37,iens) = 15.4707420008629;
        umag(38,iens) = 16.1854346550203;
        umag(39,iens) = 16.9205046415837;
        umag(40,iens) = 17.6742526407487;
        umag(41,iens) = 18.4449361532621;
        umag(42,iens) = 19.2307735286966;
        umag(43,iens) = 20.0299480842335;
        umag(44,iens) = 20.8406123044333;
        umag(45,iens) = 21.6608921122838;
        umag(46,iens) = 22.4888912016534;
        umag(47,iens) = 23.3226954211331;
        umag(48,iens) = 24.1603771991324;
        umag(49,iens) = 25;
   
        thr_coef( 0,iens) = 0.807421729819682 ;
        thr_coef( 1,iens) = 0.784655296619303 ;
        thr_coef( 2,iens) = 0.781771245165634 ;
        thr_coef( 3,iens) = 0.785377072260277 ;
        thr_coef( 4,iens) = 0.788045583661826 ;
        thr_coef( 5,iens) = 0.78992211898824  ;
        thr_coef( 6,iens) = 0.790464624846794 ;
        thr_coef( 7,iens) = 0.789868339117058 ;
        thr_coef( 8,iens) = 0.788727582310645 ;
        thr_coef( 9,iens) = 0.78735934847018  ;
        thr_coef(10,iens) = 0.785895401744818 ;
        thr_coef(11,iens) = 0.778275898677896 ;
        thr_coef(12,iens) = 0.778275898677896 ;
        thr_coef(13,iens) = 0.778275898677896 ;
        thr_coef(14,iens) = 0.778275898677896 ;
        thr_coef(15,iens) = 0.778275898677896 ;
        thr_coef(16,iens) = 0.778275898677897 ;
        thr_coef(17,iens) = 0.778275898677896 ;
        thr_coef(18,iens) = 0.778275898677897 ;
        thr_coef(19,iens) = 0.778275898677896 ;
        thr_coef(20,iens) = 0.778275898677896 ;
        thr_coef(21,iens) = 0.778275898677896 ;
        thr_coef(22,iens) = 0.778275898677896 ;
        thr_coef(23,iens) = 0.778275898677896 ;
        thr_coef(24,iens) = 0.778275898677896 ;
        thr_coef(25,iens) = 0.778275898677897 ;
        thr_coef(26,iens) = 0.778275898677896 ;
        thr_coef(27,iens) = 0.778275898677896 ;
        thr_coef(28,iens) = 0.771761720156744 ;
        thr_coef(29,iens) = 0.747149662867425 ;
        thr_coef(30,iens) = 0.562338457087221 ;
        thr_coef(31,iens) = 0.463477776718559 ;
        thr_coef(32,iens) = 0.389083718285403 ;
        thr_coef(33,iens) = 0.329822385282868 ;
        thr_coef(34,iens) = 0.28146507057366  ;
        thr_coef(35,iens) = 0.241494344724715 ;
        thr_coef(36,iens) = 0.208180574447689 ;
        thr_coef(37,iens) = 0.180257567913967 ;
        thr_coef(38,iens) = 0.156747534794659 ;
        thr_coef(39,iens) = 0.136877529405041 ;
        thr_coef(40,iens) = 0.120026379071282 ;
        thr_coef(41,iens) = 0.105689427235168 ;
        thr_coef(42,iens) = 0.0934537422432234;
        thr_coef(43,iens) = 0.0829796374703292;
        thr_coef(44,iens) = 0.0739864566418158;
        thr_coef(45,iens) = 0.066241165664466 ;
        thr_coef(46,iens) = 0.0595521065777677;
        thr_coef(47,iens) = 0.0537568660141535;
        thr_coef(48,iens) = 0.0487216617389531;
        thr_coef(49,iens) = 0.0443341965779814;
   
        pow_coef( 0,iens) = 0.0567441552796702;
        pow_coef( 1,iens) = 0.234562420438232 ;
        pow_coef( 2,iens) = 0.323807551903174 ;
        pow_coef( 3,iens) = 0.372458740900724 ;
        pow_coef( 4,iens) = 0.400568452698569 ;
        pow_coef( 5,iens) = 0.417505345979046 ;
        pow_coef( 6,iens) = 0.428006011865078 ;
        pow_coef( 7,iens) = 0.434644888701106 ;
        pow_coef( 8,iens) = 0.438881386118999 ;
        pow_coef( 9,iens) = 0.441592276334657 ;
        pow_coef(10,iens) = 0.443304138402109 ;
        pow_coef(11,iens) = 0.44434487193455  ;
        pow_coef(12,iens) = 0.445121261808611 ;
        pow_coef(13,iens) = 0.445595523035674 ;
        pow_coef(14,iens) = 0.445877516848509 ;
        pow_coef(15,iens) = 0.445994544074082 ;
        pow_coef(16,iens) = 0.446017935537279 ;
        pow_coef(17,iens) = 0.446180742972356 ;
        pow_coef(18,iens) = 0.446450527409401 ;
        pow_coef(19,iens) = 0.446802304016223 ;
        pow_coef(20,iens) = 0.447186617490563 ;
        pow_coef(21,iens) = 0.447555680846568 ;
        pow_coef(22,iens) = 0.447872401633662 ;
        pow_coef(23,iens) = 0.447631643002248 ;
        pow_coef(24,iens) = 0.446807038858815 ;
        pow_coef(25,iens) = 0.446009074959174 ;
        pow_coef(26,iens) = 0.445245595255813 ;
        pow_coef(27,iens) = 0.44450455075814  ;
        pow_coef(28,iens) = 0.444146721244444 ;
        pow_coef(29,iens) = 0.442277836513612 ;
        pow_coef(30,iens) = 0.385838745499723 ;
        pow_coef(31,iens) = 0.335841302457673 ;
        pow_coef(32,iens) = 0.291913640615431 ;
        pow_coef(33,iens) = 0.253572833608594 ;
        pow_coef(34,iens) = 0.220277652463292 ;
        pow_coef(35,iens) = 0.191478024936967 ;
        pow_coef(36,iens) = 0.16663152532331  ;
        pow_coef(37,iens) = 0.145236980958304 ;
        pow_coef(38,iens) = 0.126834454424766 ;
        pow_coef(39,iens) = 0.111012069957434 ;
        pow_coef(40,iens) = 0.0974062443168656;
        pow_coef(41,iens) = 0.0856995174942566;
        pow_coef(42,iens) = 0.0756170072234015;
        pow_coef(43,iens) = 0.0669221985452482;
        pow_coef(44,iens) = 0.0594125508680607;
        pow_coef(45,iens) = 0.0529147641549829;
        pow_coef(46,iens) = 0.0472830504453918;
        pow_coef(47,iens) = 0.0423909781343514;
        pow_coef(48,iens) = 0.0381327910105708;
        pow_coef(49,iens) = 0.0344183286144622;

        pow( 0,iens) = 0.0427333121148901;
        pow( 1,iens) = 0.292585980850948 ;
        pow( 2,iens) = 0.607966543011364 ;
        pow( 3,iens) = 0.981097692589018 ;
        pow( 4,iens) = 1.40198084046754  ;
        pow( 5,iens) = 1.85867086002364  ;
        pow( 6,iens) = 2.33757599673149  ;
        pow( 7,iens) = 2.82409730163349  ;
        pow( 8,iens) = 3.30306455989741  ;
        pow( 9,iens) = 3.75943232841862  ;
        pow(10,iens) = 4.17863771395059  ;
        pow(11,iens) = 4.54719120966456  ;
        pow(12,iens) = 4.85534268155905  ;
        pow(13,iens) = 5.09153713879626  ;
        pow(14,iens) = 5.24845313718114  ;
        pow(15,iens) = 5.32079320666984  ;
        pow(16,iens) = 5.33534549819263  ;
        pow(17,iens) = 5.43790563013573  ;
        pow(18,iens) = 5.63125302473156  ;
        pow(19,iens) = 5.92098062599689  ;
        pow(20,iens) = 6.31511560153229  ;
        pow(21,iens) = 6.82447006710142  ;
        pow(22,iens) = 7.46284638911     ;
        pow(23,iens) = 8.23835944848833  ;
        pow(24,iens) = 9.16796703004427  ;
        pow(25,iens) = 10.2852109969638  ;
        pow(26,iens) = 11.6172369930119  ;
        pow(27,iens) = 13.1944151149981  ;
        pow(28,iens) = 15                ;
        pow(29,iens) = 15.0000012933983  ;
        pow(30,iens) = 14.9999709606919  ;
        pow(31,iens) = 15.0000093357198  ;
        pow(32,iens) = 15.0000006288972  ;
        pow(33,iens) = 15.0000001060089  ;
        pow(34,iens) = 14.9999471165792  ;
        pow(35,iens) = 15.0000808219012  ;
        pow(36,iens) = 15.0000520884067  ;
        pow(37,iens) = 15.0000359166391  ;
        pow(38,iens) = 15.0000256172453  ;
        pow(39,iens) = 15.0000183483305  ;
        pow(40,iens) = 15.0000128080501  ;
        pow(41,iens) = 15.0000083543035  ;
        pow(42,iens) = 15.0000048846532  ;
        pow(43,iens) = 15.000002325659   ;
        pow(44,iens) = 15.0000006588681  ;
        pow(45,iens) = 14.9998714763666  ;
        pow(46,iens) = 15.000000469539   ;
        pow(47,iens) = 15.0000019387927  ;
        pow(48,iens) = 15.0000041656168  ;
        pow(49,iens) = 15.0000068768872  ;
      }

      ref_umag     = umag    .createDeviceCopy();
      ref_thr_coef = thr_coef.createDeviceCopy();
      ref_pow_coef = pow_coef.createDeviceCopy();
      ref_pow      = pow     .createDeviceCopy();
    }

  };

}

