
#include "microphysics_p3.h"

namespace modules {

  // Set constants and likely num_tracers as well, and anything else you can do immediately
  Microphysics_P3::Microphysics_P3() {
    R_d        = 287.042;
    cp_d       = 1004.64;
    cv_d       = cp_d - R_d;
    gamma_d    = cp_d / cv_d;
    kappa_d    = R_d  / cp_d;
    R_v        = 461.505;
    cp_v       = 1859;
    cv_v       = R_v - cp_v;
    p0         = 1.e5;
    grav       = 9.80616;
    first_step = true;
    cp_l       = 4188.0;
  }


  // Can do whatever you want, but mainly for registering tracers and allocating data
  void Microphysics_P3::init(core::Coupler &coupler) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;

    int nx   = coupler.get_nx();
    int ny   = coupler.get_ny();
    int nz   = coupler.get_nz();

    // Register tracers in the coupler
    //                 name                description            positive   adds mass    diffuse
    coupler.add_tracer("cloud_water"     , "Cloud Water Mass"   , true     , true       , true);
    coupler.add_tracer("cloud_water_num" , "Cloud Water Number" , true     , false      , true);
    coupler.add_tracer("rain"            , "Rain Water Mass"    , true     , true       , true);
    coupler.add_tracer("rain_num"        , "Rain Water Number"  , true     , false      , true);
    coupler.add_tracer("ice"             , "Ice Mass"           , true     , true       , true);
    coupler.add_tracer("ice_num"         , "Ice Number"         , true     , false      , true);
    coupler.add_tracer("ice_rime"        , "Ice-Rime Mass"      , true     , false      , true);
    coupler.add_tracer("ice_rime_vol"    , "Ice-Rime Volume"    , true     , false      , true);
    coupler.add_tracer("water_vapor"     , "Water Vapor"        , true     , true       , true);

    auto &dm = coupler.get_data_manager_readwrite();

    dm.register_and_allocate<real>("qv_prev","qv from prev step"         ,{nz,ny,nx},{"z","y","x"});
    dm.register_and_allocate<real>("t_prev" ,"Temperature from prev step",{nz,ny,nx},{"z","y","x"});

    dm.get<real,3>( "cloud_water"     ) = 0;
    dm.get<real,3>( "cloud_water_num" ) = 0;
    dm.get<real,3>( "rain"            ) = 0;
    dm.get<real,3>( "rain_num"        ) = 0;
    dm.get<real,3>( "ice"             ) = 0;
    dm.get<real,3>( "ice_num"         ) = 0;
    dm.get<real,3>( "ice_rime"        ) = 0;
    dm.get<real,3>( "ice_rime_vol"    ) = 0;
    dm.get<real,3>( "water_vapor"     ) = 0;
    dm.get<real,3>( "qv_prev"         ) = 0;
    dm.get<real,3>( "t_prev"          ) = 0;

    real rhoh2o = 1000.;
    real mwdry  = 28.966;
    real mwh2o  = 18.016;
    real latvap = 2501000.0;
    real latice = 333700.0;
    real tmelt  = 273.15;
    real pi     = 3.14159265;
    int  iulog  = 1;
    bool mainproc = true;
    micro_p3_utils_init_fortran( cp_d , R_d , R_v , rhoh2o , mwh2o , mwdry ,
                                 grav , latvap , latice, cp_l , tmelt , pi , iulog , mainproc );

    std::string dir = "../model/modules/helpers/microphysics_p3";
    std::string ver = "4.1.1";
    int dir_len = dir.length();
    int ver_len = ver.length();
    p3_init_fortran( dir.c_str() , dir_len , ver.c_str() , ver_len );

    coupler.set_option<std::string>("micro","p3");
    coupler.set_option<real>("R_d"    ,R_d    );
    coupler.set_option<real>("cp_d"   ,cp_d   );
    coupler.set_option<real>("cv_d"   ,cv_d   );
    coupler.set_option<real>("gamma_d",gamma_d);
    coupler.set_option<real>("kappa_d",kappa_d);
    coupler.set_option<real>("R_v"    ,R_v    );
    coupler.set_option<real>("cp_v"   ,cp_v   );
    coupler.set_option<real>("cv_v"   ,cv_v   );
    coupler.set_option<real>("p0"     ,p0     );
    coupler.set_option<real>("grav"   ,grav   );
  }


  void Microphysics_P3::time_step( core::Coupler &coupler , real dt ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;

    // Get the dimensions sizes
    int nz   = coupler.get_nz();
    int ny   = coupler.get_ny();
    int nx   = coupler.get_nx();
    int ncol = ny*nx;

    real crm_dx = coupler.get_dx();
    real crm_dy = coupler.get_ny_glob() == 1 ? crm_dx : coupler.get_dy();

    auto &dm = coupler.get_data_manager_readwrite();

    // Get tracers dimensioned as (nz,ny*nx)
    auto rho_c  = dm.get_lev_col<real>("cloud_water"    );
    auto rho_nc = dm.get_lev_col<real>("cloud_water_num");
    auto rho_r  = dm.get_lev_col<real>("rain"           );
    auto rho_nr = dm.get_lev_col<real>("rain_num"       );
    auto rho_i  = dm.get_lev_col<real>("ice"            );
    auto rho_ni = dm.get_lev_col<real>("ice_num"        );
    auto rho_m  = dm.get_lev_col<real>("ice_rime"       );
    auto rho_bm = dm.get_lev_col<real>("ice_rime_vol"   );
    auto rho_v  = dm.get_lev_col<real>("water_vapor"    );

    // Get coupler state
    auto rho_dry = dm.get_lev_col<real>("density_dry");
    auto temp    = dm.get_lev_col<real>("temp"       );

    real dz = coupler.get_dz();

    // Calculate the grid spacing
    real2d dz_arr("dz_arr",nz,ncol);
    yakl::memset( dz_arr , dz );

    // Get everything from the DataManager that's not a tracer but is persistent across multiple micro calls
    auto qv_prev = dm.get_lev_col<real>("qv_prev");
    auto t_prev  = dm.get_lev_col<real>("t_prev" );

    // Allocates inputs and outputs
    int p3_nout = 49;
    real2d qc                ( "qc"                 ,           nz   , ncol );
    real2d nc                ( "nc"                 ,           nz   , ncol );
    real2d qr                ( "qr"                 ,           nz   , ncol );
    real2d nr                ( "nr"                 ,           nz   , ncol );
    real2d qi                ( "qi"                 ,           nz   , ncol );
    real2d ni                ( "ni"                 ,           nz   , ncol );
    real2d qm                ( "qm"                 ,           nz   , ncol );
    real2d bm                ( "bm"                 ,           nz   , ncol );
    real2d qv                ( "qv"                 ,           nz   , ncol );
    real2d pressure          ( "pressure"           ,           nz   , ncol );
    real2d theta             ( "theta"              ,           nz   , ncol );
    real2d exner             ( "exner"              ,           nz   , ncol );
    real2d inv_exner         ( "inv_exner"          ,           nz   , ncol );
    real2d dpres             ( "dpres"              ,           nz   , ncol );
    real2d nc_nuceat_tend    ( "nc_nuceat_tend"     ,           nz   , ncol );
    real2d nccn_prescribed   ( "nccn_prescribed"    ,           nz   , ncol );
    real2d ni_activated      ( "ni_activated"       ,           nz   , ncol );
    real2d cld_frac_i        ( "cld_frac_i"         ,           nz   , ncol );
    real2d cld_frac_l        ( "cld_frac_l"         ,           nz   , ncol );
    real2d cld_frac_r        ( "cld_frac_r"         ,           nz   , ncol );
    real2d inv_qc_relvar     ( "inv_qc_relvar"      ,           nz   , ncol );
    real2d col_location      ( "col_location"       ,           3    , ncol );
    real1d precip_liq_surf   ( "precip_liq_surf"    ,                  ncol );
    real1d precip_ice_surf   ( "precip_ice_surf"    ,                  ncol );
    real2d diag_eff_radius_qc( "diag_eff_radius_qc" ,           nz   , ncol );
    real2d diag_eff_radius_qi( "diag_eff_radius_qi" ,           nz   , ncol );
    real2d bulk_qi           ( "bulk_qi"            ,           nz   , ncol );
    real2d mu_c              ( "mu_c"               ,           nz   , ncol );
    real2d lamc              ( "lamc"               ,           nz   , ncol );
    real2d qv2qi_depos_tend  ( "qv2qi_depos_tend"   ,           nz   , ncol );
    real2d precip_total_tend ( "precip_total_tend"  ,           nz   , ncol );
    real2d nevapr            ( "nevapr"             ,           nz   , ncol );
    real2d qr_evap_tend      ( "qr_evap_tend"       ,           nz   , ncol );
    real2d precip_liq_flux   ( "precip_liq_flux"    ,           nz+1 , ncol );
    real2d precip_ice_flux   ( "precip_ice_flux"    ,           nz+1 , ncol );
    real2d liq_ice_exchange  ( "liq_ice_exchange"   ,           nz   , ncol );
    real2d vap_liq_exchange  ( "vap_liq_exchange"   ,           nz   , ncol );
    real2d vap_ice_exchange  ( "vap_ice_exchange"   ,           nz   , ncol );
    real3d p3_tend_out       ( "p3_tend_out"        , p3_nout , nz   , ncol );

    //////////////////////////////////////////////////////////////////////////////
    // Compute quantities needed for inputs to P3
    //////////////////////////////////////////////////////////////////////////////
    // Force constants into local scope
    real R_d     = this->R_d;
    real R_v     = this->R_v;
    real cp_d    = this->cp_d;
    real cp_v    = this->cp_v;
    real cp_l    = this->cp_l;
    real p0      = this->p0;

    YAKL_SCOPE( first_step , this->first_step );
    YAKL_SCOPE( grav       , this->grav       );

    // Save initial state, and compute inputs for p3(...)
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
      // Compute total density
      real rho = rho_dry(k,i) + rho_c(k,i) + rho_r(k,i) + rho_i(k,i) + rho_v(k,i);

      compute_adjusted_state( rho, rho_dry(k,i) , rho_v(k,i) , rho_c(k,i) , temp(k,i),
                              R_v , cp_d , cp_v , cp_l );

      // Compute quantities for P3
      qc       (k,i) = rho_c (k,i) / rho_dry(k,i);
      nc       (k,i) = rho_nc(k,i) / rho_dry(k,i);
      qr       (k,i) = rho_r (k,i) / rho_dry(k,i);
      nr       (k,i) = rho_nr(k,i) / rho_dry(k,i);
      qi       (k,i) = rho_i (k,i) / rho_dry(k,i);
      ni       (k,i) = rho_ni(k,i) / rho_dry(k,i);
      qm       (k,i) = rho_m (k,i) / rho_dry(k,i);
      bm       (k,i) = rho_bm(k,i) / rho_dry(k,i);
      qv       (k,i) = rho_v (k,i) / rho_dry(k,i);
      pressure (k,i) = R_d * rho_dry(k,i) * temp(k,i) + R_v * rho_v(k,i) * temp(k,i);
      exner    (k,i) = pow( pressure(k,i) / p0 , R_d / cp_d );
      inv_exner(k,i) = 1. / exner(k,i);
      theta    (k,i) = temp(k,i) / exner(k,i);
      // P3 uses dpres to calculate density via the hydrostatic assumption.
      // So we just reverse this to compute dpres to give true density
      dpres(k,i) = rho_dry(k,i) * grav * dz;
      // nc_nuceat_tend, nccn_prescribed, and ni_activated are not used
      nc_nuceat_tend (k,i) = 0;
      nccn_prescribed(k,i) = 0;
      ni_activated   (k,i) = 0;
      // col_location is for debugging only, and it will be ignored for now
      if (k < 3) { col_location(k,i) = 1; }

      if (first_step) {
        qv_prev(k,i) = qv  (k,i);
        t_prev (k,i) = temp(k,i);
      }
    });

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
      // Assume cloud fracton is always 1
      cld_frac_l(k,i) = 1;
      cld_frac_i(k,i) = 1;
      cld_frac_r(k,i) = 1;
      // inv_qc_relvar is always set to one
      inv_qc_relvar(k,i) = 1;
    });
    double elapsed_s;
    int its, ite, kts, kte;
    int it = 1;
    bool do_predict_nc = false;
    bool do_prescribed_CCN = false;

    its = 1;
    ite = ncol;
    kts = 1;
    kte = nz;
    auto qc_host                 = qc                .createHostCopy();
    auto nc_host                 = nc                .createHostCopy();
    auto qr_host                 = qr                .createHostCopy();
    auto nr_host                 = nr                .createHostCopy();
    auto theta_host              = theta             .createHostCopy();
    auto qv_host                 = qv                .createHostCopy();
    auto qi_host                 = qi                .createHostCopy();
    auto qm_host                 = qm                .createHostCopy();
    auto ni_host                 = ni                .createHostCopy();
    auto bm_host                 = bm                .createHostCopy();
    auto pressure_host           = pressure          .createHostCopy();
    auto dz_host                 = dz_arr            .createHostCopy();
    auto nc_nuceat_tend_host     = nc_nuceat_tend    .createHostCopy();
    auto nccn_prescribed_host    = nccn_prescribed   .createHostCopy();
    auto ni_activated_host       = ni_activated      .createHostCopy();
    auto inv_qc_relvar_host      = inv_qc_relvar     .createHostCopy();
    auto precip_liq_surf_host    = precip_liq_surf   .createHostCopy();
    auto precip_ice_surf_host    = precip_ice_surf   .createHostCopy();
    auto diag_eff_radius_qc_host = diag_eff_radius_qc.createHostCopy();
    auto diag_eff_radius_qi_host = diag_eff_radius_qi.createHostCopy();
    auto bulk_qi_host            = bulk_qi           .createHostCopy();
    auto dpres_host              = dpres             .createHostCopy();
    auto inv_exner_host          = inv_exner         .createHostCopy();
    auto qv2qi_depos_tend_host   = qv2qi_depos_tend  .createHostCopy();
    auto precip_total_tend_host  = precip_total_tend .createHostCopy();
    auto nevapr_host             = nevapr            .createHostCopy();
    auto qr_evap_tend_host       = qr_evap_tend      .createHostCopy();
    auto precip_liq_flux_host    = precip_liq_flux   .createHostCopy();
    auto precip_ice_flux_host    = precip_ice_flux   .createHostCopy();
    auto cld_frac_r_host         = cld_frac_r        .createHostCopy();
    auto cld_frac_l_host         = cld_frac_l        .createHostCopy();
    auto cld_frac_i_host         = cld_frac_i        .createHostCopy();
    auto p3_tend_out_host        = p3_tend_out       .createHostCopy();
    auto mu_c_host               = mu_c              .createHostCopy();
    auto lamc_host               = lamc              .createHostCopy();
    auto liq_ice_exchange_host   = liq_ice_exchange  .createHostCopy();
    auto vap_liq_exchange_host   = vap_liq_exchange  .createHostCopy();
    auto vap_ice_exchange_host   = vap_ice_exchange  .createHostCopy();
    auto qv_prev_host            = qv_prev           .createHostCopy();
    auto t_prev_host             = t_prev            .createHostCopy();
    auto col_location_host       = col_location      .createHostCopy();

    p3_main_fortran(qc_host.data() , nc_host.data() , qr_host.data() , nr_host.data() , theta_host.data() ,
                    qv_host.data() , dt , qi_host.data() , qm_host.data() , ni_host.data() , bm_host.data() ,
                    pressure_host.data() , dz_host.data() , nc_nuceat_tend_host.data() ,
                    nccn_prescribed_host.data() , ni_activated_host.data() , inv_qc_relvar_host.data() , it ,
                    precip_liq_surf_host.data() , precip_ice_surf_host.data() , its , ite , kts , kte ,
                    diag_eff_radius_qc_host.data() , diag_eff_radius_qi_host.data() , bulk_qi_host.data() ,
                    do_predict_nc , do_prescribed_CCN , dpres_host.data() , inv_exner_host.data() ,
                    qv2qi_depos_tend_host.data() , precip_total_tend_host.data() , nevapr_host.data() ,
                    qr_evap_tend_host.data() , precip_liq_flux_host.data() , precip_ice_flux_host.data() ,
                    cld_frac_r_host.data() , cld_frac_l_host.data() , cld_frac_i_host.data() ,
                    p3_tend_out_host.data() , mu_c_host.data() , lamc_host.data() , liq_ice_exchange_host.data() ,
                    vap_liq_exchange_host.data() , vap_ice_exchange_host.data() , qv_prev_host.data() ,
                    t_prev_host.data() , col_location_host.data() , &elapsed_s );

    qc_host                .deep_copy_to( qc                 );
    nc_host                .deep_copy_to( nc                 );
    qr_host                .deep_copy_to( qr                 );
    nr_host                .deep_copy_to( nr                 );
    theta_host             .deep_copy_to( theta              );
    qv_host                .deep_copy_to( qv                 );
    qi_host                .deep_copy_to( qi                 );
    qm_host                .deep_copy_to( qm                 );
    ni_host                .deep_copy_to( ni                 );
    bm_host                .deep_copy_to( bm                 );
    pressure_host          .deep_copy_to( pressure           );
    dz_host                .deep_copy_to( dz_arr             );
    nc_nuceat_tend_host    .deep_copy_to( nc_nuceat_tend     );
    nccn_prescribed_host   .deep_copy_to( nccn_prescribed    );
    ni_activated_host      .deep_copy_to( ni_activated       );
    inv_qc_relvar_host     .deep_copy_to( inv_qc_relvar      );
    precip_liq_surf_host   .deep_copy_to( precip_liq_surf    );
    precip_ice_surf_host   .deep_copy_to( precip_ice_surf    );
    diag_eff_radius_qc_host.deep_copy_to( diag_eff_radius_qc );
    diag_eff_radius_qi_host.deep_copy_to( diag_eff_radius_qi );
    bulk_qi_host           .deep_copy_to( bulk_qi            );
    dpres_host             .deep_copy_to( dpres              );
    inv_exner_host         .deep_copy_to( inv_exner          );
    qv2qi_depos_tend_host  .deep_copy_to( qv2qi_depos_tend   );
    precip_total_tend_host .deep_copy_to( precip_total_tend  );
    nevapr_host            .deep_copy_to( nevapr             );
    qr_evap_tend_host      .deep_copy_to( qr_evap_tend       );
    precip_liq_flux_host   .deep_copy_to( precip_liq_flux    );
    precip_ice_flux_host   .deep_copy_to( precip_ice_flux    );
    cld_frac_r_host        .deep_copy_to( cld_frac_r         );
    cld_frac_l_host        .deep_copy_to( cld_frac_l         );
    cld_frac_i_host        .deep_copy_to( cld_frac_i         );
    p3_tend_out_host       .deep_copy_to( p3_tend_out        );
    mu_c_host              .deep_copy_to( mu_c               );
    lamc_host              .deep_copy_to( lamc               );
    liq_ice_exchange_host  .deep_copy_to( liq_ice_exchange   );
    vap_liq_exchange_host  .deep_copy_to( vap_liq_exchange   );
    vap_ice_exchange_host  .deep_copy_to( vap_ice_exchange   );
    qv_prev_host           .deep_copy_to( qv_prev            );
    t_prev_host            .deep_copy_to( t_prev             );
    col_location_host      .deep_copy_to( col_location       );
    
    ///////////////////////////////////////////////////////////////////////////////
    // Convert P3 outputs into dynamics coupler state and tracer masses
    ///////////////////////////////////////////////////////////////////////////////
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
      rho_c  (k,i) = std::max( qc(k,i)*rho_dry(k,i) , 0._fp );
      rho_nc (k,i) = std::max( nc(k,i)*rho_dry(k,i) , 0._fp );
      rho_r  (k,i) = std::max( qr(k,i)*rho_dry(k,i) , 0._fp );
      rho_nr (k,i) = std::max( nr(k,i)*rho_dry(k,i) , 0._fp );
      rho_i  (k,i) = std::max( qi(k,i)*rho_dry(k,i) , 0._fp );
      rho_ni (k,i) = std::max( ni(k,i)*rho_dry(k,i) , 0._fp );
      rho_m  (k,i) = std::max( qm(k,i)*rho_dry(k,i) , 0._fp );
      rho_bm (k,i) = std::max( bm(k,i)*rho_dry(k,i) , 0._fp );
      rho_v  (k,i) = std::max( qv(k,i)*rho_dry(k,i) , 0._fp );
      // While micro changes total pressure, thus changing exner, the definition
      // of theta depends on the old exner pressure, so we'll use old exner here
      temp   (k,i) = theta(k,i) * exner(k,i);
      // Save qv and temperature for the next call to p3_main
      qv_prev(k,i) = std::max( qv(k,i) , 0._fp );
      t_prev (k,i) = temp(k,i);
    });

    first_step = false;
  }


  std::string Microphysics_P3::micro_name() const {
    return "p3";
  }

}


