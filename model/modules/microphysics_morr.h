
#pragma once

#include "coupler.h"


extern "C"
void mp_morr_two_moment(float *t, float *qv, float *qc, float *qr, float *qi, float *qs,
                        float *qg, float *ni, float *ns, float *nr, float *ng, float *rho,
                        float *p, float *dt_in, float *dz, float *rainnc, float *rainncv,
                        float *sr, float *snownc, float *snowncv, float *graupelnc, float *graupelncv,
                        float *qrcuten, float *qscuten,
                        float *qicuten, int *ncol, int *nz,
                        float *qlsink, float *precr, float *preci, float *precs, float *precg);


extern "C"
void morr_two_moment_init(int *morr_rimed_ice);


namespace modules {

  class Microphysics_Morrison {
  public:
    // Doesn't actually have to be static or constexpr. Could be assigned in the constructor
    int static constexpr num_tracers = 10;

    // This must return the correct # of tracers **BEFORE** init(...) is called
    KOKKOS_INLINE_FUNCTION static int get_num_tracers() { return num_tracers; }


    // Can do whatever you want, but mainly for registering tracers and allocating data
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();

      // Register tracers in the coupler
      //                 name              description   positive   adds mass    diffuse
      coupler.add_tracer("water_vapor"   , ""          , true     , true       , true);
      coupler.add_tracer("cloud_water"   , ""          , true     , true       , true);
      coupler.add_tracer("rain_water"    , ""          , true     , true       , true);
      coupler.add_tracer("cloud_ice"     , ""          , true     , true       , true);
      coupler.add_tracer("snow"          , ""          , true     , true       , true);
      coupler.add_tracer("graupel"       , ""          , true     , true       , true);
      coupler.add_tracer("cloud_ice_num" , ""          , true     , false      , true);
      coupler.add_tracer("snow_num"      , ""          , true     , false      , true);
      coupler.add_tracer("rain_num"      , ""          , true     , false      , true);
      coupler.add_tracer("graupel_num"   , ""          , true     , false      , true);

      auto &dm = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<real>("micro_rainnc"   ,"accumulated precipitation (mm)"      ,{ny,nx},{"y","x"});
      dm.register_and_allocate<real>("micro_snownc"   ,"accumulated snow plus cloud ice (mm)",{ny,nx},{"y","x"});
      dm.register_and_allocate<real>("micro_graupelnc","accumulated graupel (mm)"            ,{ny,nx},{"y","x"});

      dm.get_collapsed<real>( "water_vapor"     ) = 0;
      dm.get_collapsed<real>( "cloud_water"     ) = 0;
      dm.get_collapsed<real>( "rain_water"      ) = 0;
      dm.get_collapsed<real>( "cloud_ice"       ) = 0;
      dm.get_collapsed<real>( "snow"            ) = 0;
      dm.get_collapsed<real>( "graupel"         ) = 0;
      dm.get_collapsed<real>( "cloud_ice_num"   ) = 0;
      dm.get_collapsed<real>( "snow_num"        ) = 0;
      dm.get_collapsed<real>( "rain_num"        ) = 0;
      dm.get_collapsed<real>( "graupel_num"     ) = 0;
      dm.get_collapsed<real>( "micro_rainnc"    ) = 0;
      dm.get_collapsed<real>( "micro_snownc"    ) = 0;
      dm.get_collapsed<real>( "micro_graupelnc" ) = 0;

      int morr_rimed_ice = 1;
      morr_two_moment_init( &morr_rimed_ice );

      coupler.set_option<std::string>("micro","p3");
      real R_d        = 287.;
      real cp_d       = 7.*R_d/2.;
      real cv_d       = cp_d - R_d;
      real gamma_d    = cp_d / cv_d;
      real kappa_d    = R_d  / cp_d;
      real R_v        = 461.6;
      real cp_v       = 4.*R_v;
      real cv_v       = cp_v - R_v;
      real p0         = 1.e5;
      real grav       = 9.81;
      real cp_l       = 4190;
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



    void time_step( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      // Get the dimensions sizes
      int nz   = coupler.get_nz();
      int ny   = coupler.get_ny();
      int nx   = coupler.get_nx();
      int ncol = ny*nx;

      // Get tracers and persistent variables dimensioned as (nz,ny*nx)
      auto &dm = coupler.get_data_manager_readwrite();
      auto dm_rho_v     = dm.get_lev_col  <real>("water_vapor"    );
      auto dm_rho_c     = dm.get_lev_col  <real>("cloud_water"    );
      auto dm_rho_r     = dm.get_lev_col  <real>("rain_water"     );
      auto dm_rho_i     = dm.get_lev_col  <real>("cloud_ice"      );
      auto dm_rho_s     = dm.get_lev_col  <real>("snow"           );
      auto dm_rho_g     = dm.get_lev_col  <real>("graupel"        );
      auto dm_rho_in    = dm.get_lev_col  <real>("cloud_ice_num"  );
      auto dm_rho_sn    = dm.get_lev_col  <real>("snow_num"       );
      auto dm_rho_rn    = dm.get_lev_col  <real>("rain_num"       );
      auto dm_rho_gn    = dm.get_lev_col  <real>("graupel_num"    );
      auto dm_rho_dry   = dm.get_lev_col  <real>("density_dry"    );
      auto dm_temp      = dm.get_lev_col  <real>("temp"           );
      auto dm_rainnc    = dm.get_collapsed<real>("micro_rainnc"   );
      auto dm_snownc    = dm.get_collapsed<real>("micro_snownc"   );
      auto dm_graupelnc = dm.get_collapsed<real>("micro_graupelnc");
      real dz = coupler.get_dz();
      float2d dz_arr("dz_arr",nz,ncol);
      dz_arr = dz;

      // Allocates inputs and outputs
      float dt_in=dt;
      float2d qv        ("qv        ",nz,ncol);
      float2d qc        ("qc        ",nz,ncol);
      float2d qr        ("qr        ",nz,ncol);
      float2d qi        ("qi        ",nz,ncol);
      float2d qs        ("qs        ",nz,ncol);
      float2d qg        ("qg        ",nz,ncol);
      float2d ni        ("ni        ",nz,ncol);
      float2d ns        ("ns        ",nz,ncol);
      float2d nr        ("nr        ",nz,ncol);
      float2d t         ("t         ",nz,ncol);
      float2d ng        ("ng        ",nz,ncol);
      float2d qlsink    ("qlsink    ",nz,ncol);
      float2d preci     ("preci     ",nz,ncol);
      float2d precs     ("precs     ",nz,ncol);
      float2d precg     ("precg     ",nz,ncol);
      float2d precr     ("precr     ",nz,ncol);
      float2d p         ("p         ",nz,ncol);
      float2d rho       ("rho       ",nz,ncol);
      float2d qrcuten   ("qrcuten   ",nz,ncol);
      float2d qscuten   ("qscuten   ",nz,ncol);
      float2d qicuten   ("qicuten   ",nz,ncol);
      float1d rainncv   ("rainncv   ",   ncol);
      float1d sr        ("sr        ",   ncol);
      float1d snowncv   ("snowncv   ",   ncol);
      float1d graupelncv("graupelncv",   ncol);
      float1d rainnc    ("rainnc    ",   ncol);
      float1d snownc    ("snownc    ",   ncol);
      float1d graupelnc ("graupelnc ",   ncol);

      //////////////////////////////////////////////////////////////////////////////
      // Compute quantities needed for inputs to Morrison 2-mom
      //////////////////////////////////////////////////////////////////////////////
      real R_d  = coupler.get_option<real>("R_d" );
      real R_v  = coupler.get_option<real>("R_v" );
      real cp_d = coupler.get_option<real>("cp_d");
      real p0   = coupler.get_option<real>("p0"  );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
        rho    (k,i) = dm_rho_dry(k,i) + dm_rho_c(k,i) + dm_rho_r(k,i) + dm_rho_i(k,i) + dm_rho_v(k,i);
        qv     (k,i) = dm_rho_v (k,i)/dm_rho_dry(k,i);
        qc     (k,i) = dm_rho_c (k,i)/dm_rho_dry(k,i);
        qr     (k,i) = dm_rho_r (k,i)/dm_rho_dry(k,i);
        qi     (k,i) = dm_rho_i (k,i)/dm_rho_dry(k,i);
        qs     (k,i) = dm_rho_s (k,i)/dm_rho_dry(k,i);
        qg     (k,i) = dm_rho_g (k,i)/dm_rho_dry(k,i);
        ni     (k,i) = dm_rho_in(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        ns     (k,i) = dm_rho_sn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        nr     (k,i) = dm_rho_rn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        ng     (k,i) = dm_rho_gn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        p      (k,i) = R_d*dm_rho_dry(k,i)*dm_temp(k,i) + R_v*dm_rho_v(k,i)*dm_temp(k,i);
        t      (k,i) = dm_temp(k,i);
        qrcuten(k,i) = 0;
        qscuten(k,i) = 0;
        qicuten(k,i) = 0;
        if (k == 0) {
          rainnc   (i) = dm_rainnc   (i);
          snownc   (i) = dm_snownc   (i);
          graupelnc(i) = dm_graupelnc(i);
        }
      });
      auto host_rainnc     = rainnc    .createHostObject();
      auto host_snownc     = snownc    .createHostObject();
      auto host_graupelnc  = graupelnc .createHostObject();
      auto host_qv         = qv        .createHostObject();
      auto host_qc         = qc        .createHostObject();
      auto host_qr         = qr        .createHostObject();
      auto host_qi         = qi        .createHostObject();
      auto host_qs         = qs        .createHostObject();
      auto host_qg         = qg        .createHostObject();
      auto host_ni         = ni        .createHostObject();
      auto host_ns         = ns        .createHostObject();
      auto host_nr         = nr        .createHostObject();
      auto host_t          = t         .createHostObject();
      auto host_ng         = ng        .createHostObject();
      auto host_qlsink     = qlsink    .createHostObject();
      auto host_preci      = preci     .createHostObject();
      auto host_precs      = precs     .createHostObject();
      auto host_precg      = precg     .createHostObject();
      auto host_precr      = precr     .createHostObject();
      auto host_p          = p         .createHostObject();
      auto host_rho        = rho       .createHostObject();
      auto host_qrcuten    = qrcuten   .createHostObject();
      auto host_qscuten    = qscuten   .createHostObject();
      auto host_qicuten    = qicuten   .createHostObject();
      auto host_rainncv    = rainncv   .createHostObject();
      auto host_sr         = sr        .createHostObject();
      auto host_snowncv    = snowncv   .createHostObject();
      auto host_graupelncv = graupelncv.createHostObject();
      auto host_dz         = dz_arr    .createHostObject();
      rainnc    .deep_copy_to(host_rainnc    );
      snownc    .deep_copy_to(host_snownc    );
      graupelnc .deep_copy_to(host_graupelnc );
      qv        .deep_copy_to(host_qv        );
      qc        .deep_copy_to(host_qc        );
      qr        .deep_copy_to(host_qr        );
      qi        .deep_copy_to(host_qi        );
      qs        .deep_copy_to(host_qs        );
      qg        .deep_copy_to(host_qg        );
      ni        .deep_copy_to(host_ni        );
      ns        .deep_copy_to(host_ns        );
      nr        .deep_copy_to(host_nr        );
      t         .deep_copy_to(host_t         );
      ng        .deep_copy_to(host_ng        );
      p         .deep_copy_to(host_p         );
      rho       .deep_copy_to(host_rho       );
      qrcuten   .deep_copy_to(host_qrcuten   );
      qscuten   .deep_copy_to(host_qscuten   );
      qicuten   .deep_copy_to(host_qicuten   );
      dz_arr    .deep_copy_to(host_dz        );
      Kokkos::fence();

      mp_morr_two_moment(host_t.data(), host_qv.data(), host_qc.data(), host_qr.data(), host_qi.data(), host_qs.data(),
                         host_qg.data(), host_ni.data(), host_ns.data(), host_nr.data(), host_ng.data(), host_rho.data(),
                         host_p.data(), &dt_in, host_dz.data(), host_rainnc.data(), host_rainncv.data(),
                         host_sr.data(), host_snownc.data(), host_snowncv.data(), host_graupelnc.data(), host_graupelncv.data(),
                         host_qrcuten.data(), host_qscuten.data(),
                         host_qicuten.data(), &ncol, &nz,
                         host_qlsink.data(), host_precr.data(), host_preci.data(), host_precs.data(), host_precg.data());

      host_rainnc    .deep_copy_to(rainnc    );
      host_snownc    .deep_copy_to(snownc    );
      host_graupelnc .deep_copy_to(graupelnc );
      host_qv        .deep_copy_to(qv        );
      host_qc        .deep_copy_to(qc        );
      host_qr        .deep_copy_to(qr        );
      host_qi        .deep_copy_to(qi        );
      host_qs        .deep_copy_to(qs        );
      host_qg        .deep_copy_to(qg        );
      host_ni        .deep_copy_to(ni        );
      host_ns        .deep_copy_to(ns        );
      host_nr        .deep_copy_to(nr        );
      host_t         .deep_copy_to(t         );
      host_ng        .deep_copy_to(ng        );
      
      ///////////////////////////////////////////////////////////////////////////////
      // Convert Morrison 2mom outputs into dynamics coupler state and tracer masses
      ///////////////////////////////////////////////////////////////////////////////
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
        dm_rho_v (k,i) = qv(k,i)*dm_rho_dry(k,i);
        dm_rho_c (k,i) = qc(k,i)*dm_rho_dry(k,i);
        dm_rho_r (k,i) = qr(k,i)*dm_rho_dry(k,i);
        dm_rho_i (k,i) = qi(k,i)*dm_rho_dry(k,i);
        dm_rho_s (k,i) = qs(k,i)*dm_rho_dry(k,i);
        dm_rho_g (k,i) = qg(k,i)*dm_rho_dry(k,i);
        dm_rho_in(k,i) = ni(k,i)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_rho_sn(k,i) = ns(k,i)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_rho_rn(k,i) = nr(k,i)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_rho_gn(k,i) = ng(k,i)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_temp  (k,i) = t(k,i);
        if (k == 0) {
          dm_rainnc   (i) = rainnc   (i);
          dm_snownc   (i) = snownc   (i);
          dm_graupelnc(i) = graupelnc(i);
        }
      });
    }


    std::string micro_name() const { return "p3"; }
  };

}


