
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

  struct Microphysics_Morrison {
    typedef yakl::Array<float      ,1,yakl::memDevice,yakl::styleFortran> float1d_F;
    typedef yakl::Array<float      ,2,yakl::memDevice,yakl::styleFortran> float2d_F;
    typedef yakl::Array<float const,1,yakl::memDevice,yakl::styleFortran> floatConst1d_F;
    typedef yakl::Array<float const,2,yakl::memDevice,yakl::styleFortran> floatConst2d_F;
    typedef yakl::Array<float      ,1,yakl::memHost  ,yakl::styleFortran> floatHost1d_F;
    typedef yakl::Array<float      ,2,yakl::memHost  ,yakl::styleFortran> floatHost2d_F;
    typedef yakl::Array<float const,1,yakl::memHost  ,yakl::styleFortran> floatHostConst1d_F;
    typedef yakl::Array<float const,2,yakl::memHost  ,yakl::styleFortran> floatHostConst2d_F;
    typedef yakl::Array<int        ,1,yakl::memDevice,yakl::styleFortran> int1d_F;
    typedef yakl::Array<int        ,2,yakl::memDevice,yakl::styleFortran> int2d_F;
    typedef yakl::Array<int   const,1,yakl::memDevice,yakl::styleFortran> intConst1d_F;
    typedef yakl::Array<int   const,2,yakl::memDevice,yakl::styleFortran> intConst2d_F;
    typedef yakl::Array<int        ,1,yakl::memHost  ,yakl::styleFortran> intHost1d_F;
    typedef yakl::Array<int        ,2,yakl::memHost  ,yakl::styleFortran> intHost2d_F;
    typedef yakl::Array<int   const,1,yakl::memHost  ,yakl::styleFortran> intHostConst1d_F;
    typedef yakl::Array<int   const,2,yakl::memHost  ,yakl::styleFortran> intHostConst2d_F;
    typedef yakl::Array<bool       ,1,yakl::memDevice,yakl::styleFortran> bool1d_F;
    typedef yakl::Array<bool       ,2,yakl::memDevice,yakl::styleFortran> bool2d_F;
    typedef yakl::Array<bool  const,1,yakl::memDevice,yakl::styleFortran> boolConst1d_F;
    typedef yakl::Array<bool  const,2,yakl::memDevice,yakl::styleFortran> boolConst2d_F;
    typedef yakl::Array<bool       ,1,yakl::memHost  ,yakl::styleFortran> boolHost1d_F;
    typedef yakl::Array<bool       ,2,yakl::memHost  ,yakl::styleFortran> boolHost2d_F;
    typedef yakl::Array<bool  const,1,yakl::memHost  ,yakl::styleFortran> boolHostConst1d_F;
    typedef yakl::Array<bool  const,2,yakl::memHost  ,yakl::styleFortran> boolHostConst2d_F;
    // Doesn't actually have to be static or constexpr. Could be assigned in the constructor
    int static constexpr num_tracers = 10;


    KOKKOS_INLINE_FUNCTION static int get_num_tracers() { return num_tracers; }


    // Can do whatever you want, but mainly for registering tracers and allocating data
    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      init_two_moment( 1 );

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

      // Allocates inputs and outputs
      float dt_in=dt;
      float2d_F qv        ("qv        ",ncol,nz);
      float2d_F qc        ("qc        ",ncol,nz);
      float2d_F qr        ("qr        ",ncol,nz);
      float2d_F qi        ("qi        ",ncol,nz);
      float2d_F qs        ("qs        ",ncol,nz);
      float2d_F qg        ("qg        ",ncol,nz);
      float2d_F ni        ("ni        ",ncol,nz);
      float2d_F ns        ("ns        ",ncol,nz);
      float2d_F nr        ("nr        ",ncol,nz);
      float2d_F t         ("t         ",ncol,nz);
      float2d_F ng        ("ng        ",ncol,nz);
      float2d_F qlsink    ("qlsink    ",ncol,nz);
      float2d_F preci     ("preci     ",ncol,nz);
      float2d_F precs     ("precs     ",ncol,nz);
      float2d_F precg     ("precg     ",ncol,nz);
      float2d_F precr     ("precr     ",ncol,nz);
      float2d_F p         ("p         ",ncol,nz);
      float2d_F rho       ("rho       ",ncol,nz);
      float2d_F qrcuten   ("qrcuten   ",ncol,nz);
      float2d_F qscuten   ("qscuten   ",ncol,nz);
      float2d_F qicuten   ("qicuten   ",ncol,nz);
      float2d_F dz_arr    ("dz_arr"    ,ncol,nz);
      float1d_F rainncv   ("rainncv   ",ncol   );
      float1d_F sr        ("sr        ",ncol   );
      float1d_F snowncv   ("snowncv   ",ncol   );
      float1d_F graupelncv("graupelncv",ncol   );
      float1d_F rainnc    ("rainnc    ",ncol   );
      float1d_F snownc    ("snownc    ",ncol   );
      float1d_F graupelnc ("graupelnc ",ncol   );

      //////////////////////////////////////////////////////////////////////////////
      // Compute quantities needed for inputs to Morrison 2-mom
      //////////////////////////////////////////////////////////////////////////////
      real R_d  = coupler.get_option<real>("R_d" );
      real R_v  = coupler.get_option<real>("R_v" );
      real cp_d = coupler.get_option<real>("cp_d");
      real p0   = coupler.get_option<real>("p0"  );

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
        rho    (i+1,k+1) = dm_rho_dry(k,i) + dm_rho_c(k,i) + dm_rho_r(k,i) + dm_rho_i(k,i) + dm_rho_v(k,i);
        qv     (i+1,k+1) = dm_rho_v (k,i)/dm_rho_dry(k,i);
        qc     (i+1,k+1) = dm_rho_c (k,i)/dm_rho_dry(k,i);
        qr     (i+1,k+1) = dm_rho_r (k,i)/dm_rho_dry(k,i);
        qi     (i+1,k+1) = dm_rho_i (k,i)/dm_rho_dry(k,i);
        qs     (i+1,k+1) = dm_rho_s (k,i)/dm_rho_dry(k,i);
        qg     (i+1,k+1) = dm_rho_g (k,i)/dm_rho_dry(k,i);
        ni     (i+1,k+1) = dm_rho_in(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        ns     (i+1,k+1) = dm_rho_sn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        nr     (i+1,k+1) = dm_rho_rn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        ng     (i+1,k+1) = dm_rho_gn(k,i)/dm_rho_dry(k,i)/dm_rho_dry(k,i);
        p      (i+1,k+1) = R_d*dm_rho_dry(k,i)*dm_temp(k,i) + R_v*dm_rho_v(k,i)*dm_temp(k,i);
        t      (i+1,k+1) = dm_temp(k,i);
        qrcuten(i+1,k+1) = 0;
        qscuten(i+1,k+1) = 0;
        qicuten(i+1,k+1) = 0;
        dz_arr (i+1,k+1) = dz;
        if (k == 0) {
          rainnc   (i+1) = dm_rainnc   (i);
          snownc   (i+1) = dm_snownc   (i);
          graupelnc(i+1) = dm_graupelnc(i);
        }
      });

      // auto host_rainnc     = rainnc    .createHostObject();
      // auto host_snownc     = snownc    .createHostObject();
      // auto host_graupelnc  = graupelnc .createHostObject();
      // auto host_qv         = qv        .createHostObject();
      // auto host_qc         = qc        .createHostObject();
      // auto host_qr         = qr        .createHostObject();
      // auto host_qi         = qi        .createHostObject();
      // auto host_qs         = qs        .createHostObject();
      // auto host_qg         = qg        .createHostObject();
      // auto host_ni         = ni        .createHostObject();
      // auto host_ns         = ns        .createHostObject();
      // auto host_nr         = nr        .createHostObject();
      // auto host_t          = t         .createHostObject();
      // auto host_ng         = ng        .createHostObject();
      // auto host_qlsink     = qlsink    .createHostObject();
      // auto host_preci      = preci     .createHostObject();
      // auto host_precs      = precs     .createHostObject();
      // auto host_precg      = precg     .createHostObject();
      // auto host_precr      = precr     .createHostObject();
      // auto host_p          = p         .createHostObject();
      // auto host_rho        = rho       .createHostObject();
      // auto host_qrcuten    = qrcuten   .createHostObject();
      // auto host_qscuten    = qscuten   .createHostObject();
      // auto host_qicuten    = qicuten   .createHostObject();
      // auto host_rainncv    = rainncv   .createHostObject();
      // auto host_sr         = sr        .createHostObject();
      // auto host_snowncv    = snowncv   .createHostObject();
      // auto host_graupelncv = graupelncv.createHostObject();
      // auto host_dz         = dz_arr    .createHostObject();

      // rainnc    .deep_copy_to(host_rainnc    );
      // snownc    .deep_copy_to(host_snownc    );
      // graupelnc .deep_copy_to(host_graupelnc );
      // qv        .deep_copy_to(host_qv        );
      // qc        .deep_copy_to(host_qc        );
      // qr        .deep_copy_to(host_qr        );
      // qi        .deep_copy_to(host_qi        );
      // qs        .deep_copy_to(host_qs        );
      // qg        .deep_copy_to(host_qg        );
      // ni        .deep_copy_to(host_ni        );
      // ns        .deep_copy_to(host_ns        );
      // nr        .deep_copy_to(host_nr        );
      // t         .deep_copy_to(host_t         );
      // ng        .deep_copy_to(host_ng        );
      // p         .deep_copy_to(host_p         );
      // rho       .deep_copy_to(host_rho       );
      // qrcuten   .deep_copy_to(host_qrcuten   );
      // qscuten   .deep_copy_to(host_qscuten   );
      // qicuten   .deep_copy_to(host_qicuten   );
      // dz_arr    .deep_copy_to(host_dz        );

      // Kokkos::fence();

      // mp_morr_two_moment(host_t.data(), host_qv.data(), host_qc.data(), host_qr.data(), host_qi.data(), host_qs.data(),
      //                    host_qg.data(), host_ni.data(), host_ns.data(), host_nr.data(), host_ng.data(), host_rho.data(),
      //                    host_p.data(), &dt_in, host_dz.data(), host_rainnc.data(), host_rainncv.data(),
      //                    host_sr.data(), host_snownc.data(), host_snowncv.data(), host_graupelnc.data(), host_graupelncv.data(),
      //                    host_qrcuten.data(), host_qscuten.data(),
      //                    host_qicuten.data(), &ncol, &nz,
      //                    host_qlsink.data(), host_precr.data(), host_preci.data(), host_precs.data(), host_precg.data());

      // host_rainnc    .deep_copy_to(rainnc    );
      // host_snownc    .deep_copy_to(snownc    );
      // host_graupelnc .deep_copy_to(graupelnc );
      // host_qv        .deep_copy_to(qv        );
      // host_qc        .deep_copy_to(qc        );
      // host_qr        .deep_copy_to(qr        );
      // host_qi        .deep_copy_to(qi        );
      // host_qs        .deep_copy_to(qs        );
      // host_qg        .deep_copy_to(qg        );
      // host_ni        .deep_copy_to(ni        );
      // host_ns        .deep_copy_to(ns        );
      // host_nr        .deep_copy_to(nr        );
      // host_t         .deep_copy_to(t         );
      // host_ng        .deep_copy_to(ng        );

      two_mom_wrapper(t, qv, qc, qr, qi, qs, qg, ni, ns, nr,
                      ng, rho, p, dt_in, dz_arr, rainnc, rainncv, sr, snownc,
                      snowncv, graupelnc, graupelncv, qrcuten, qscuten, qicuten, ncol,
                      nz, qlsink, precr, preci, precs, precg);
      
      ///////////////////////////////////////////////////////////////////////////////
      // Convert Morrison 2mom outputs into dynamics coupler state and tracer masses
      ///////////////////////////////////////////////////////////////////////////////
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
        dm_rho_v (k,i) = qv(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_c (k,i) = qc(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_r (k,i) = qr(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_i (k,i) = qi(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_s (k,i) = qs(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_g (k,i) = qg(i+1,k+1)*dm_rho_dry(k,i);
        dm_rho_in(k,i) = ni(i+1,k+1)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_rho_sn(k,i) = ns(i+1,k+1)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_rho_rn(k,i) = nr(i+1,k+1)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_rho_gn(k,i) = ng(i+1,k+1)*dm_rho_dry(k,i)*dm_rho_dry(k,i);
        dm_temp  (k,i) = t (i+1,k+1);
        if (k == 0) {
          dm_rainnc   (i) = rainnc   (i+1);
          dm_snownc   (i) = snownc   (i+1);
          dm_graupelnc(i) = graupelnc(i+1);
        }
      });
    }


    std::string micro_name() const { return "p3"; }


    int   static constexpr inum    = 1;
    int   static constexpr iact    = 2;
    int   static constexpr ibase   = 2;
    int   static constexpr isub    = 0;
    int   static constexpr iliq    = 0;
    int   static constexpr inuc    = 0;
    int   static constexpr igraup  = 0;
    float static constexpr pi      = 3.1415926535897932384626434;
    float static constexpr xxx     = 0.9189385332046727417803297;
    float static constexpr r       = 287.;
    float static constexpr rv      = 461.6;
    float static constexpr g       = 9.81;
    float static constexpr cp      = 1004.5;
    float static constexpr ep_2    = 0.621750433;
    float static constexpr ndcnst  = 250.;
    float static constexpr ai      = 700.;
    float static constexpr ac      = 3.e7;
    float static constexpr as      = 11.72;
    float static constexpr ar      = 841.99667;
    float static constexpr bi      = 1.;
    float static constexpr bc      = 2.;
    float static constexpr bs      = 0.41;
    float static constexpr br      = 0.8;
    float static constexpr rhosu   = 85000./(287.15*273.15);
    float static constexpr rhow    = 997.;
    float static constexpr rhoi    = 500.;
    float static constexpr rhosn   = 100.;
    float static constexpr aimm    = 0.66;
    float static constexpr bimm    = 100.;
    float static constexpr ecr     = 1.;
    float static constexpr dcs     = 125.e-6;
    float static constexpr mg0     = 1.6e-10;
    float static constexpr f1s     = 0.86;
    float static constexpr f2s     = 0.28;
    float static constexpr f1r     = 0.78;
    float static constexpr f2r     = 0.308;
    float static constexpr qsmall  = 1.e-14;
    float static constexpr eii     = 0.1;
    float static constexpr eci     = 0.7;
    float static constexpr cpw     = 4187.;
    float static constexpr di      = 3.;
    float static constexpr ds      = 3.;
    float static constexpr dg      = 3.;
    float static constexpr rin     = 0.1e-6;
    float static constexpr lammaxi = 1./1.e-6;
    float static constexpr lammaxr = 1./20.e-6;
    float static constexpr lamminr = 1./2800.e-6;
    float static constexpr lammaxs = 1./10.e-6;
    float static constexpr lammins = 1./2000.e-6;
    float static constexpr lammaxg = 1./20.e-6;
    float static constexpr lamming = 1./2000.e-6;
    float static constexpr k1      = 0.4;
    float static constexpr c1      = 120. ;
    float static constexpr mw      = 0.018;
    float static constexpr osm     = 1.;
    float static constexpr vi      = 3.;
    float static constexpr epsm    = 0.7;
    float static constexpr rhoa    = 1777.;
    float static constexpr map     = 0.132;
    float static constexpr ma      = 0.0284;
    float static constexpr rr      = 8.3145;
    float static constexpr rm1     = 0.052e-6;
    float static constexpr sig1    = 2.04;
    float static constexpr nanew1  = 72.2e6;
    float static constexpr rm2     = 1.3e-6;
    float static constexpr sig2    = 2.5;
    float static constexpr nanew2  = 1.8e6;
    float ag,bg,rhog,mi0,ci,cs,cg,mmult,lammini,bact,f11,f21,f12,f22,cons1,cons2,cons3,cons4,cons5,
          cons6,cons7,cons8,cons9,cons10,cons11,cons12,cons13,cons14,cons15,cons16,cons17,cons18,cons19,
          cons20,cons21,cons22,cons23,cons24,cons25,cons26,cons27,cons28,cons29,cons30,cons31,cons32,
          cons33,cons34,cons35,cons36,cons37,cons38,cons39,cons40,cons41;


    void init_two_moment(int ihail) {
      ag      = ihail == 0 ? 19.3 : 114.5;
      bg      = ihail == 0 ? 0.37 : 0.5  ;
      rhog    = ihail == 0 ? 400. : 900. ;
      mi0     = 4./3.*pi*rhoi*pow(10.e-6,3);
      ci      = rhoi*pi/6.;
      cs      = rhosn*pi/6.;
      cg      = rhog*pi/6.;
      mmult   = 4./3.*pi*rhoi*pow(5.e-6,3);
      lammini = 1./(2.*dcs+100.e-6);
      bact    = vi*osm*epsm*mw*rhoa/(map*rhow);
      f11     = 0.5*std::exp(2.5*pow(std::log(sig1),2));
      f21     = 1.+0.25*std::log(sig1);
      f12     = 0.5*std::exp(2.5*pow(std::log(sig2),2));
      f22     = 1.+0.25*std::log(sig2);
      cons1   = gamma(1.+ds)*cs;
      cons2   = gamma(1.+dg)*cg;
      cons3   = gamma(4.+bs)/6.;
      cons4   = gamma(4.+br)/6.;
      cons5   = gamma(1.+bs);
      cons6   = gamma(1.+br);
      cons7   = gamma(4.+bg)/6.;
      cons8   = gamma(1.+bg);
      cons9   = gamma(5./2.+br/2.);
      cons10  = gamma(5./2.+bs/2.);
      cons11  = gamma(5./2.+bg/2.);
      cons12  = gamma(1.+di)*ci;
      cons13  = gamma(bs+3.)*pi/4.*eci;
      cons14  = gamma(bg+3.)*pi/4.*eci;
      cons15  = -1108.*eii*pow(pi,(1.-bs)/3.)*pow(rhosn,(-2.-bs)/3.)/(4.*720.);
      cons16  = gamma(bi+3.)*pi/4.*eci;
      cons17  = 4.*2.*3.*rhosu*pi*eci*eci*gamma(2.*bs+2.)/(8.*(rhog-rhosn));
      cons18  = rhosn*rhosn;
      cons19  = rhow*rhow;
      cons20  = 20.*pi*pi*rhow*bimm;
      cons21  = 4./(dcs*rhoi);
      cons22  = pi*rhoi*pow(dcs,3)/6.;
      cons23  = pi/4.*eii*gamma(bs+3.);
      cons24  = pi/4.*ecr*gamma(br+3.);
      cons25  = pi*pi/24.*rhow*ecr*gamma(br+6.);
      cons26  = pi/6.*rhow;
      cons27  = gamma(1.+bi);
      cons28  = gamma(4.+bi)/6.;
      cons29  = 4./3.*pi*rhow*pow(25.e-6,3);
      cons30  = 4./3.*pi*rhow;
      cons31  = pi*pi*ecr*rhosn;
      cons32  = pi/2.*ecr;
      cons33  = pi*pi*ecr*rhog;
      cons34  = 5./2.+br/2.;
      cons35  = 5./2.+bs/2.;
      cons36  = 5./2.+bg/2.;
      cons37  = 4.*pi*1.38e-23/(6.*pi*rin);
      cons38  = pi*pi/3.*rhow;
      cons39  = pi*pi/36.*rhow*bimm;
      cons40  = pi/6.*bimm;
      cons41  = pi*pi*ecr*rhow;
    }



    void two_mom_wrapper(float2d_F const &t, float2d_F const &qv, float2d_F const &qc, float2d_F const &qr,
                         float2d_F const &qi, float2d_F const &qs, float2d_F const &qg, float2d_F const &ni,
                         float2d_F const &ns, float2d_F const &nr, float2d_F const &ng, floatConst2d_F rho,
                         floatConst2d_F p, float dt_in, floatConst2d_F dz, float1d_F const &rainnc,
                         float1d_F const &rainncv, float1d_F const &sr, float1d_F const &snownc,
                         float1d_F const &snowncv, float1d_F const &graupelnc, float1d_F const &graupelncv,
                         floatConst2d_F qrcuten, floatConst2d_F qscuten, floatConst2d_F qicuten, int ncol,
                         int nz, float2d_F const &qlsink, float2d_F const &precr, float2d_F const &preci,
                         float2d_F const &precs, float2d_F const &precg) {
      using yakl::fortran::parallel_for;
      using yakl::fortran::SimpleBounds;
      float2d_F c2prec("c2prec",ncol,nz);
      float dt    = dt_in;
      int  iinum = 1;
      run_two_mom(qc, qi, qs, qr ,ni, ns, nr, t, qv, p, dz, rainncv, sr, snowncv, graupelncv, dt, ncol, nz, qg,
                  ng, qrcuten, qscuten, qicuten, iinum, c2prec, preci, precs, precg, precr);
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (qc(i,k) > 1.e-10) { qlsink(i,k) = c2prec(i,k)/qc(i,k); }
          else                  { qlsink(i,k) = 0.0;                 }
          if (k == 1) {
            rainnc    (i) = rainnc(i)+rainncv(i);
            snownc    (i) = snownc(i)+snowncv(i);
            graupelnc (i) = graupelnc(i)+graupelncv(i);
            sr        (i) = sr(i)/(rainncv(i)+1.e-12);
          }
      });
    }



    void run_two_mom(float2d_F const &qc3d, float2d_F const &qi3d, float2d_F const &qni3d, float2d_F const &qr3d,
                     float2d_F const &ni3d, float2d_F const &ns3d, float2d_F const &nr3d, float2d_F const &t3d,
                     float2d_F const &qv3d, floatConst2d_F pres, floatConst2d_F dzq, float1d_F const &precrt,
                     float1d_F const &snowrt, float1d_F const &snowprt, float1d_F const &grplprt, float dt, int ncol,
                     int nz, float2d_F const &qg3d, float2d_F const &ng3d, floatConst2d_F qrcu1d,
                     floatConst2d_F qscu1d, floatConst2d_F qicu1d, int iinum, float2d_F const &c2prec,
                     float2d_F const &ised, float2d_F const &ssed, float2d_F const &gsed, float2d_F const &rsed) {
      using yakl::fortran::parallel_for;
      using yakl::fortran::SimpleBounds;
      YAKL_SCOPE( ag      , this->ag      );     
      YAKL_SCOPE( bg      , this->bg      );     
      YAKL_SCOPE( rhog    , this->rhog    );       
      YAKL_SCOPE( mi0     , this->mi0     );      
      YAKL_SCOPE( ci      , this->ci      );     
      YAKL_SCOPE( cs      , this->cs      );     
      YAKL_SCOPE( cg      , this->cg      );     
      YAKL_SCOPE( mmult   , this->mmult   );        
      YAKL_SCOPE( lammini , this->lammini );          
      YAKL_SCOPE( bact    , this->bact    );       
      YAKL_SCOPE( f11     , this->f11     );      
      YAKL_SCOPE( f21     , this->f21     );      
      YAKL_SCOPE( f12     , this->f12     );      
      YAKL_SCOPE( f22     , this->f22     );      
      YAKL_SCOPE( cons1   , this->cons1   );        
      YAKL_SCOPE( cons2   , this->cons2   );        
      YAKL_SCOPE( cons3   , this->cons3   );        
      YAKL_SCOPE( cons4   , this->cons4   );        
      YAKL_SCOPE( cons5   , this->cons5   );        
      YAKL_SCOPE( cons6   , this->cons6   );        
      YAKL_SCOPE( cons7   , this->cons7   );        
      YAKL_SCOPE( cons8   , this->cons8   );        
      YAKL_SCOPE( cons9   , this->cons9   );        
      YAKL_SCOPE( cons10  , this->cons10  );         
      YAKL_SCOPE( cons11  , this->cons11  );         
      YAKL_SCOPE( cons12  , this->cons12  );         
      YAKL_SCOPE( cons13  , this->cons13  );         
      YAKL_SCOPE( cons14  , this->cons14  );         
      YAKL_SCOPE( cons15  , this->cons15  );         
      YAKL_SCOPE( cons16  , this->cons16  );         
      YAKL_SCOPE( cons17  , this->cons17  );         
      YAKL_SCOPE( cons18  , this->cons18  );         
      YAKL_SCOPE( cons19  , this->cons19  );         
      YAKL_SCOPE( cons20  , this->cons20  );         
      YAKL_SCOPE( cons21  , this->cons21  );         
      YAKL_SCOPE( cons22  , this->cons22  );         
      YAKL_SCOPE( cons23  , this->cons23  );         
      YAKL_SCOPE( cons24  , this->cons24  );         
      YAKL_SCOPE( cons25  , this->cons25  );         
      YAKL_SCOPE( cons26  , this->cons26  );         
      YAKL_SCOPE( cons27  , this->cons27  );         
      YAKL_SCOPE( cons28  , this->cons28  );         
      YAKL_SCOPE( cons29  , this->cons29  );         
      YAKL_SCOPE( cons30  , this->cons30  );         
      YAKL_SCOPE( cons31  , this->cons31  );         
      YAKL_SCOPE( cons32  , this->cons32  );         
      YAKL_SCOPE( cons33  , this->cons33  );         
      YAKL_SCOPE( cons34  , this->cons34  );         
      YAKL_SCOPE( cons35  , this->cons35  );         
      YAKL_SCOPE( cons36  , this->cons36  );         
      YAKL_SCOPE( cons37  , this->cons37  );         
      YAKL_SCOPE( cons38  , this->cons38  );         
      YAKL_SCOPE( cons39  , this->cons39  );         
      YAKL_SCOPE( cons40  , this->cons40  );         
      YAKL_SCOPE( cons41  , this->cons41  );         
      float2d_F ng3dten   ("ng3dten   ",ncol,nz);  // graupel numb conc tendency (1/kg/s)
      float2d_F qg3dten   ("qg3dten   ",ncol,nz);  // graupel mix ratio tendency (kg/kg/s)
      float2d_F effc      ("effc      ",ncol,nz);  // droplet effective radius (micron)
      float2d_F effi      ("effi      ",ncol,nz);  // cloud ice effective radius (micron)
      float2d_F effs      ("effs      ",ncol,nz);  // snow effective radius (micron)
      float2d_F effr      ("effr      ",ncol,nz);  // rain effective radius (micron)
      float2d_F effg      ("effg      ",ncol,nz);  // graupel effective radius (micron)
      float2d_F t3dten    ("t3dten    ",ncol,nz);  // temperature tendency (k/s)
      float2d_F qv3dten   ("qv3dten   ",ncol,nz);  // water vapor mixing ratio tendency (kg/kg/s)
      float2d_F qc3dten   ("qc3dten   ",ncol,nz);  // cloud water mixing ratio tendency (kg/kg/s)
      float2d_F qi3dten   ("qi3dten   ",ncol,nz);  // cloud ice mixing ratio tendency (kg/kg/s)
      float2d_F qni3dten  ("qni3dten  ",ncol,nz);  // snow mixing ratio tendency (kg/kg/s)
      float2d_F qr3dten   ("qr3dten   ",ncol,nz);  // rain mixing ratio tendency (kg/kg/s)
      float2d_F ni3dten   ("ni3dten   ",ncol,nz);  // cloud ice number concentration (1/kg/s)
      float2d_F ns3dten   ("ns3dten   ",ncol,nz);  // snow number concentration (1/kg/s)
      float2d_F nr3dten   ("nr3dten   ",ncol,nz);  // rain number concentration (1/kg/s)
      float2d_F csed      ("csed      ",ncol,nz);  //
      float2d_F qgsten    ("qgsten    ",ncol,nz);  // graupel sed tend (kg/kg/s)
      float2d_F qrsten    ("qrsten    ",ncol,nz);  // rain sed tend (kg/kg/s)
      float2d_F qisten    ("qisten    ",ncol,nz);  // cloud ice sed tend (kg/kg/s)
      float2d_F qnisten   ("qnisten   ",ncol,nz);  // snow sed tend (kg/kg/s)
      float2d_F qcsten    ("qcsten    ",ncol,nz);  // cloud wat sed tend (kg/kg/s)      
      float2d_F nc3d      ("nc3d      ",ncol,nz);  //
      float2d_F nc3dten   ("nc3dten   ",ncol,nz);  //
      float2d_F lamc      ("lamc      ",ncol,nz);  // slope parameter for droplets (m-1)
      float2d_F lami      ("lami      ",ncol,nz);  // slope parameter for cloud ice (m-1)
      float2d_F lams      ("lams      ",ncol,nz);  // slope parameter for snow (m-1)
      float2d_F lamr      ("lamr      ",ncol,nz);  // slope parameter for rain (m-1)
      float2d_F lamg      ("lamg      ",ncol,nz);  // slope parameter for graupel (m-1)
      float2d_F cdist1    ("cdist1    ",ncol,nz);  // psd parameter for droplets
      float2d_F n0i       ("n0i       ",ncol,nz);  // intercept parameter for cloud ice (kg-1 m-1)
      float2d_F n0s       ("n0s       ",ncol,nz);  // intercept parameter for snow (kg-1 m-1)
      float2d_F n0rr      ("n0rr      ",ncol,nz);  // intercept parameter for rain (kg-1 m-1)
      float2d_F n0g       ("n0g       ",ncol,nz);  // intercept parameter for graupel (kg-1 m-1)
      float2d_F pgam      ("pgam      ",ncol,nz);  // spectral shape parameter for droplets
      float2d_F nsubc     ("nsubc     ",ncol,nz);  // loss of nc during evap
      float2d_F nsubi     ("nsubi     ",ncol,nz);  // loss of ni during sub.
      float2d_F nsubs     ("nsubs     ",ncol,nz);  // loss of ns during sub.
      float2d_F nsubr     ("nsubr     ",ncol,nz);  // loss of nr during evap
      float2d_F prd       ("prd       ",ncol,nz);  // dep cloud ice
      float2d_F pre       ("pre       ",ncol,nz);  // evap of rain
      float2d_F prds      ("prds      ",ncol,nz);  // dep snow
      float2d_F nnuccc    ("nnuccc    ",ncol,nz);  // change n due to contact freez droplets
      float2d_F mnuccc    ("mnuccc    ",ncol,nz);  // change q due to contact freez droplets
      float2d_F pra       ("pra       ",ncol,nz);  // accretion droplets by rain
      float2d_F prc       ("prc       ",ncol,nz);  // autoconversion droplets
      float2d_F pcc       ("pcc       ",ncol,nz);  // cond/evap droplets
      float2d_F nnuccd    ("nnuccd    ",ncol,nz);  // change n freezing aerosol (prim ice nucleation)
      float2d_F mnuccd    ("mnuccd    ",ncol,nz);  // change q freezing aerosol (prim ice nucleation)
      float2d_F mnuccr    ("mnuccr    ",ncol,nz);  // change q due to contact freez rain
      float2d_F nnuccr    ("nnuccr    ",ncol,nz);  // change n due to contact freez rain
      float2d_F npra      ("npra      ",ncol,nz);  // change in n due to droplet acc by rain
      float2d_F nragg     ("nragg     ",ncol,nz);  // self-collection/breakup of rain
      float2d_F nsagg     ("nsagg     ",ncol,nz);  // self-collection of snow
      float2d_F nprc      ("nprc      ",ncol,nz);  // change nc autoconversion droplets
      float2d_F nprc1     ("nprc1     ",ncol,nz);  // change nr autoconversion droplets
      float2d_F prai      ("prai      ",ncol,nz);  // change q accretion cloud ice by snow
      float2d_F prci      ("prci      ",ncol,nz);  // change q autoconversin cloud ice to snow
      float2d_F psacws    ("psacws    ",ncol,nz);  // change q droplet accretion by snow
      float2d_F npsacws   ("npsacws   ",ncol,nz);  // change n droplet accretion by snow
      float2d_F psacwi    ("psacwi    ",ncol,nz);  // change q droplet accretion by cloud ice
      float2d_F npsacwi   ("npsacwi   ",ncol,nz);  // change n droplet accretion by cloud ice
      float2d_F nprci     ("nprci     ",ncol,nz);  // change n autoconversion cloud ice by snow
      float2d_F nprai     ("nprai     ",ncol,nz);  // change n accretion cloud ice
      float2d_F nmults    ("nmults    ",ncol,nz);  // ice mult due to riming droplets by snow
      float2d_F nmultr    ("nmultr    ",ncol,nz);  // ice mult due to riming rain by snow
      float2d_F qmults    ("qmults    ",ncol,nz);  // change q due to ice mult droplets/snow
      float2d_F qmultr    ("qmultr    ",ncol,nz);  // change q due to ice rain/snow
      float2d_F pracs     ("pracs     ",ncol,nz);  // change q rain-snow collection
      float2d_F npracs    ("npracs    ",ncol,nz);  // change n rain-snow collection
      float2d_F pccn      ("pccn      ",ncol,nz);  // change q droplet activation
      float2d_F psmlt     ("psmlt     ",ncol,nz);  // change q melting snow to rain
      float2d_F evpms     ("evpms     ",ncol,nz);  // chnage q melting snow evaporating
      float2d_F nsmlts    ("nsmlts    ",ncol,nz);  // change n melting snow
      float2d_F nsmltr    ("nsmltr    ",ncol,nz);  // change n melting snow to rain
      float2d_F piacr     ("piacr     ",ncol,nz);  // change qr, ice-rain collection
      float2d_F niacr     ("niacr     ",ncol,nz);  // change n, ice-rain collection
      float2d_F praci     ("praci     ",ncol,nz);  // change qi, ice-rain collection
      float2d_F piacrs    ("piacrs    ",ncol,nz);  // change qr, ice rain collision, added to snow
      float2d_F niacrs    ("niacrs    ",ncol,nz);  // change n, ice rain collision, added to snow
      float2d_F pracis    ("pracis    ",ncol,nz);  // change qi, ice rain collision, added to snow
      float2d_F eprd      ("eprd      ",ncol,nz);  // sublimation cloud ice
      float2d_F eprds     ("eprds     ",ncol,nz);  // sublimation snow
      float2d_F pracg     ("pracg     ",ncol,nz);  // change in q collection rain by graupel
      float2d_F psacwg    ("psacwg    ",ncol,nz);  // change in q collection droplets by graupel
      float2d_F pgsacw    ("pgsacw    ",ncol,nz);  // conversion q to graupel due to collection droplets by snow
      float2d_F pgracs    ("pgracs    ",ncol,nz);  // conversion q to graupel due to collection rain by snow
      float2d_F prdg      ("prdg      ",ncol,nz);  // dep of graupel
      float2d_F eprdg     ("eprdg     ",ncol,nz);  // sub of graupel
      float2d_F evpmg     ("evpmg     ",ncol,nz);  // change q melting of graupel and evaporation
      float2d_F pgmlt     ("pgmlt     ",ncol,nz);  // change q melting of graupel
      float2d_F npracg    ("npracg    ",ncol,nz);  // change n collection rain by graupel
      float2d_F npsacwg   ("npsacwg   ",ncol,nz);  // change n collection droplets by graupel
      float2d_F nscng     ("nscng     ",ncol,nz);  // change n conversion to graupel due to collection droplets by snow
      float2d_F ngracs    ("ngracs    ",ncol,nz);  // change n conversion to graupel due to collection rain by snow
      float2d_F ngmltg    ("ngmltg    ",ncol,nz);  // change n melting graupel
      float2d_F ngmltr    ("ngmltr    ",ncol,nz);  // change n melting graupel to rain
      float2d_F nsubg     ("nsubg     ",ncol,nz);  // change n sub/dep of graupel
      float2d_F psacr     ("psacr     ",ncol,nz);  // conversion due to coll of snow by rain
      float2d_F nmultg    ("nmultg    ",ncol,nz);  // ice mult due to acc droplets by graupel
      float2d_F nmultrg   ("nmultrg   ",ncol,nz);  // ice mult due to acc rain by graupel
      float2d_F qmultg    ("qmultg    ",ncol,nz);  // change q due to ice mult droplets/graupel
      float2d_F qmultrg   ("qmultrg   ",ncol,nz);  // change q due to ice mult rain/graupel
      float2d_F kap       ("kap       ",ncol,nz);  // thermal conductivity of air
      float2d_F evs       ("evs       ",ncol,nz);  // saturation vapor pressure
      float2d_F eis       ("eis       ",ncol,nz);  // ice saturation vapor pressure
      float2d_F qvs       ("qvs       ",ncol,nz);  // saturation mixing ratio
      float2d_F qvi       ("qvi       ",ncol,nz);  // ice saturation mixing ratio
      float2d_F qvqvs     ("qvqvs     ",ncol,nz);  // sautration ratio
      float2d_F qvqvsi    ("qvqvsi    ",ncol,nz);  // ice saturaion ratio
      float2d_F dv        ("dv        ",ncol,nz);  // diffusivity of water vapor in air
      float2d_F xxls      ("xxls      ",ncol,nz);  // latent heat of sublimation
      float2d_F xxlv      ("xxlv      ",ncol,nz);  // latent heat of vaporization
      float2d_F cpm       ("cpm       ",ncol,nz);  // specific heat at const pressure for moist air
      float2d_F mu        ("mu        ",ncol,nz);  // viscocity of air
      float2d_F sc        ("sc        ",ncol,nz);  // schmidt number
      float2d_F xlf       ("xlf       ",ncol,nz);  // latent heat of freezing
      float2d_F rho       ("rho       ",ncol,nz);  // air density
      float2d_F ab        ("ab        ",ncol,nz);  // correction to condensation rate due to latent heating
      float2d_F abi       ("abi       ",ncol,nz);  // correction to deposition rate due to latent heating
      float2d_F dap       ("dap       ",ncol,nz);  // diffusivity of aerosol
      float2d_F dumi      ("dumi      ",ncol,nz);  //
      float2d_F dumr      ("dumr      ",ncol,nz);  //
      float2d_F dumfni    ("dumfni    ",ncol,nz);  //
      float2d_F dumg      ("dumg      ",ncol,nz);  //
      float2d_F dumfng    ("dumfng    ",ncol,nz);  //
      float2d_F fr        ("fr        ",ncol,nz);  //
      float2d_F fi        ("fi        ",ncol,nz);  //
      float2d_F fni       ("fni       ",ncol,nz);  //
      float2d_F fg        ("fg        ",ncol,nz);  //
      float2d_F fng       ("fng       ",ncol,nz);  //
      float2d_F faloutr   ("faloutr   ",ncol,nz);  //
      float2d_F falouti   ("falouti   ",ncol,nz);  //
      float2d_F faloutni  ("faloutni  ",ncol,nz);  //
      float2d_F dumqs     ("dumqs     ",ncol,nz);  //
      float2d_F dumfns    ("dumfns    ",ncol,nz);  //
      float2d_F fs        ("fs        ",ncol,nz);  //
      float2d_F fns       ("fns       ",ncol,nz);  //
      float2d_F falouts   ("falouts   ",ncol,nz);  //
      float2d_F faloutns  ("faloutns  ",ncol,nz);  //
      float2d_F faloutg   ("faloutg   ",ncol,nz);  //
      float2d_F faloutng  ("faloutng  ",ncol,nz);  //
      float2d_F dumc      ("dumc      ",ncol,nz);  //
      float2d_F dumfnc    ("dumfnc    ",ncol,nz);  //
      float2d_F fc        ("fc        ",ncol,nz);  //
      float2d_F faloutc   ("faloutc   ",ncol,nz);  //
      float2d_F faloutnc  ("faloutnc  ",ncol,nz);  //
      float2d_F fnc       ("fnc       ",ncol,nz);  //
      float2d_F dumfnr    ("dumfnr    ",ncol,nz);  //
      float2d_F faloutnr  ("faloutnr  ",ncol,nz);  //
      float2d_F fnr       ("fnr       ",ncol,nz);  //
      float2d_F ain       ("ain       ",ncol,nz);  //
      float2d_F arn       ("arn       ",ncol,nz);  //
      float2d_F asn       ("asn       ",ncol,nz);  //
      float2d_F acn       ("acn       ",ncol,nz);  //
      float2d_F agn       ("agn       ",ncol,nz);  //
      float2d_F tqimelt   ("tqimelt   ",ncol,nz);  // melting of cloud ice (tendency)
      int1d_F   nstep     ("nstep     ",ncol);     //
      float dum;
      bool2d_F skip_micro("skip_micro",ncol,nz);
      bool2d_F t_ge_273  ("t_ge_273"  ,ncol,nz);
      bool2d_F no_cirg   ("no_cirg"   ,ncol,nz);      //
      bool1d_F hydro_pres("hydro_pres",ncol);      //
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<1>(ncol) , KOKKOS_LAMBDA (int i) {
        hydro_pres(i) = false;
        precrt    (i) = 0.;
        snowrt    (i) = 0.;
        snowprt   (i) = 0.;
        grplprt   (i) = 0.;
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          float dum;
          skip_micro(i,k) = false;
          nc3d    (i,k) = 0.;
          ng3dten (i,k) = 0.;
          qg3dten (i,k) = 0.;
          t3dten  (i,k) = 0.;
          qv3dten (i,k) = 0.;
          qc3dten (i,k) = 0.;
          qi3dten (i,k) = 0.;
          qni3dten(i,k) = 0.;
          qr3dten (i,k) = 0.;
          ni3dten (i,k) = 0.;
          ns3dten (i,k) = 0.;
          nr3dten (i,k) = 0.;
          nc3dten (i,k) = 0.;
          c2prec  (i,k) = 0.;
          csed    (i,k) = 0.;
          ised    (i,k) = 0.;
          ssed    (i,k) = 0.;
          gsed    (i,k) = 0.;
          rsed    (i,k) = 0.;
          xxlv    (i,k) = 3.1484e6-2370.*t3d(i,k);
          xxls    (i,k) = 3.15e6-2370.*t3d(i,k)+0.3337e6;
          cpm     (i,k) = cp*(1.+0.887*qv3d(i,k));
          evs     (i,k) = min(0.99*pres(i,k),polysvp(t3d(i,k),0));
          eis     (i,k) = min(0.99*pres(i,k),polysvp(t3d(i,k),1));
          if (eis(i,k) > evs(i,k)) eis(i,k) = evs(i,k);
          qvs     (i,k) = ep_2*evs(i,k)/(pres(i,k)-evs(i,k));
          qvi     (i,k) = ep_2*eis(i,k)/(pres(i,k)-eis(i,k));
          qvqvs   (i,k) = qv3d(i,k)/qvs(i,k);
          qvqvsi  (i,k) = qv3d(i,k)/qvi(i,k);
          rho     (i,k) = pres(i,k)/(r*t3d(i,k));
          if (qrcu1d(i,k) >= 1.e-10) nr3d(i,k) = nr3d(i,k)+1.8e5*pow(qrcu1d(i,k)*dt/(pi*rhow*pow(rho(i,k),3)),0.25);
          if (qscu1d(i,k) >= 1.e-10) ns3d(i,k) = ns3d(i,k)+3.e5*pow(qscu1d(i,k)*dt/(cons1*pow(rho(i,k),3)),1./(ds+1.));
          if (qicu1d(i,k) >= 1.e-10) ni3d(i,k) = ni3d(i,k)+qicu1d(i,k)*dt/(ci*pow(80.e-6,di));
          if (qvqvs(i,k) < 0.9) {
            if (qr3d(i,k) < 1.e-8) {
               qv3d(i,k)=qv3d(i,k)+qr3d(i,k);
               t3d (i,k)=t3d(i,k)-qr3d(i,k)*xxlv(i,k)/cpm(i,k);
               qr3d(i,k)=0.;
            }
            if (qc3d(i,k) < 1.e-8) {
               qv3d(i,k)=qv3d(i,k)+qc3d(i,k);
               t3d (i,k)=t3d(i,k)-qc3d(i,k)*xxlv(i,k)/cpm(i,k);
               qc3d(i,k)=0.;
            }
          }
          if (qvqvsi(i,k) < 0.9) {
            if (qi3d(i,k) < 1.e-8) {
               qv3d(i,k)=qv3d(i,k)+qi3d(i,k);
               t3d (i,k)=t3d(i,k)-qi3d(i,k)*xxls(i,k)/cpm(i,k);
               qi3d(i,k)=0.;
            }
            if (qni3d(i,k) < 1.e-8) {
               qv3d (i,k)=qv3d(i,k)+qni3d(i,k);
               t3d  (i,k)=t3d(i,k)-qni3d(i,k)*xxls(i,k)/cpm(i,k);
               qni3d(i,k)=0.;
            }
            if (qg3d(i,k) < 1.e-8) {
               qv3d(i,k)=qv3d(i,k)+qg3d(i,k);
               t3d (i,k)=t3d(i,k)-qg3d(i,k)*xxls(i,k)/cpm(i,k);
               qg3d(i,k)=0.;
            }
          }
          xlf(i,k) = xxls(i,k)-xxlv(i,k);
          if (qc3d(i,k) < qsmall) {
            qc3d(i,k) = 0.;
            nc3d(i,k) = 0.;
            effc(i,k) = 0.;
          }
          if (qr3d(i,k) < qsmall) {
            qr3d(i,k) = 0.;
            nr3d(i,k) = 0.;
            effr(i,k) = 0.;
          }
          if (qi3d(i,k) < qsmall) {
            qi3d(i,k) = 0.;
            ni3d(i,k) = 0.;
            effi(i,k) = 0.;
          }
          if (qni3d(i,k) < qsmall) {
            qni3d(i,k) = 0.;
            ns3d (i,k) = 0.;
            effs (i,k) = 0.;
          }
          if (qg3d(i,k) < qsmall) {
            qg3d(i,k) = 0.;
            ng3d(i,k) = 0.;
            effg(i,k) = 0.;
          }
          qrsten (i,k) = 0.;
          qisten (i,k) = 0.;
          qnisten(i,k) = 0.;
          qcsten (i,k) = 0.;
          qgsten (i,k) = 0.;
          mu     (i,k) = 1.496e-6*pow(t3d(i,k),1.5)/(t3d(i,k)+120.);
          dum          = pow(rhosu/rho(i,k),0.54);
          ain    (i,k) = pow(rhosu/rho(i,k),0.35)*ai;
          arn    (i,k) = dum*ar;
          asn    (i,k) = dum*as;
          acn    (i,k) = g*rhow/(18.*mu(i,k));
          agn    (i,k) = dum*ag;
          lami   (i,k) = 0.;
          // if there is no cloud/precip water, and if subsaturated, then skip microphysics for this cell
          if (qc3d(i,k) < qsmall && qi3d(i,k) < qsmall && qni3d(i,k) < qsmall && qr3d(i,k) < qsmall && qg3d(i,k) < qsmall) {
            if (t3d(i,k) <  273.15 && qvqvsi(i,k) < 0.999) skip_micro(i,k) = true;
            if (t3d(i,k) >= 273.15 && qvqvs (i,k) < 0.999) skip_micro(i,k) = true;
          }
          if (! skip_micro(i,k)) {
            kap   (i,k) = 1.414e3*mu(i,k);
            dv    (i,k) = 8.794e-5*pow(t3d(i,k),1.81)/pres(i,k);
            sc    (i,k) = mu(i,k)/(rho(i,k)*dv(i,k));
            dum         = (rv*pow(t3d(i,k),2));
            float dqsdt  = xxlv(i,k)*qvs(i,k)/dum;
            float dqsidt = xxls(i,k)*qvi(i,k)/dum;
            abi   (i,k) = 1.+dqsidt*xxls(i,k)/cpm(i,k);
            ab    (i,k) = 1.+dqsdt*xxlv(i,k)/cpm(i,k);
            t_ge_273(i,k) = t3d(i,k) >= 273.15;
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          float dum;
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if (iinum==1) {
                nc3d(i,k) = ndcnst*1.e6/rho(i,k);
              }
              if (qni3d(i,k) < 1.e-6) {
                qr3d (i,k) = qr3d(i,k)+qni3d(i,k);
                nr3d (i,k) = nr3d(i,k)+ns3d (i,k);
                t3d  (i,k) = t3d (i,k)-qni3d(i,k)*xlf(i,k)/cpm(i,k);
                qni3d(i,k) = 0.;
                ns3d (i,k) = 0.;
              }
              if (qg3d(i,k) < 1.e-6) {
                qr3d(i,k) = qr3d(i,k)+qg3d(i,k);
                nr3d(i,k) = nr3d(i,k)+ng3d(i,k);
                t3d (i,k) = t3d (i,k)-qg3d(i,k)*xlf(i,k)/cpm(i,k);
                qg3d(i,k) = 0.;
                ng3d(i,k) = 0.;
              }
              // True if there's no cloud, ice, rain, or graupel
              no_cirg(i,k) = qc3d(i,k) < qsmall && qni3d(i,k) < 1.e-8 && qr3d(i,k) < qsmall && qg3d(i,k) < 1.e-8;
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if ( ! no_cirg(i,k)) {  // If there's cloud or ice or rain or graupel
                float dum;
                ns3d(i,k) = max(0.,ns3d(i,k));
                nc3d(i,k) = max(0.,nc3d(i,k));
                nr3d(i,k) = max(0.,nr3d(i,k));
                ng3d(i,k) = max(0.,ng3d(i,k));
                if (qr3d(i,k) >= qsmall) {
                  lamr(i,k) = pow(pi*rhow*nr3d(i,k)/qr3d(i,k),1./3.);
                  n0rr(i,k) = nr3d(i,k)*lamr(i,k);
                  if (lamr(i,k) < lamminr) {
                    lamr(i,k) = lamminr;
                    n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                    nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                  } else if (lamr(i,k) > lammaxr) {
                    lamr(i,k) = lammaxr;
                    n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                    nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                  }
                  if (qc3d(i,k) >= qsmall) {
                    dum     =  pres(i,k)/(287.15*t3d(i,k));
                    pgam(i,k) = 0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714;
                    pgam(i,k) = 1./(pow(pgam(i,k),2))-1.;
                    pgam(i,k) = max(pgam(i,k),2.);
                    pgam(i,k) = min(pgam(i,k),10.);
                    lamc(i,k) = pow(cons26*nc3d(i,k)*gamma(pgam(i,k)+4.)/(qc3d(i,k)*gamma(pgam(i,k)+1.)),1./3.);
                    float lammin  = (pgam(i,k)+1.)/60.e-6;
                    float lammax  = (pgam(i,k)+1.)/1.e-6;
                    if (lamc(i,k) < lammin) {
                      lamc(i,k) = lammin;
                      nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(gamma(pgam(i,k)+1.))-std::log(gamma(pgam(i,k)+4.)))/cons26;
                    } else if (lamc(i,k) > lammax) {
                      lamc(i,k) = lammax;
                      nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(gamma(pgam(i,k)+1.))-std::log(gamma(pgam(i,k)+4.)))/cons26;
                    }
                  }
                }
                if (qni3d(i,k) >= qsmall) {
                  lams(i,k) = pow(cons1*ns3d(i,k)/qni3d(i,k),1./ds);
                  n0s(i,k) = ns3d(i,k)*lams(i,k);
                  if (lams(i,k) < lammins) {
                    lams(i,k) = lammins;
                    n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1;
                    ns3d(i,k) = n0s(i,k)/lams(i,k);
                  } else if (lams(i,k) > lammaxs) {
                    lams(i,k) = lammaxs;
                    n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1;
                    ns3d(i,k) = n0s(i,k)/lams(i,k);
                  }
                }
                if (qg3d(i,k) >= qsmall) {
                  lamg(i,k) = pow(cons2*ng3d(i,k)/qg3d(i,k),1./dg);
                  n0g(i,k) = ng3d(i,k)*lamg(i,k);
                  if (lamg(i,k) < lamming) {
                    lamg(i,k) = lamming;
                    n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                    ng3d(i,k) = n0g(i,k)/lamg(i,k);
                  } else if (lamg(i,k) > lammaxg) {
                    lamg(i,k) = lammaxg;
                    n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                    ng3d(i,k) = n0g(i,k)/lamg(i,k);
                  }
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          float dum;
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if ( ! no_cirg(i,k)) {  // If there's cloud or ice or rain or graupel
                prc(i,k) = 0.;
                nprc(i,k) = 0.;
                nprc1(i,k) = 0.;
                pra(i,k) = 0.;
                npra(i,k) = 0.;
                nragg(i,k) = 0.;
                nsmlts(i,k) = 0.;
                nsmltr(i,k) = 0.;
                evpms(i,k) = 0.;
                pcc(i,k) = 0.;
                pre(i,k) = 0.;
                nsubc(i,k) = 0.;
                nsubr(i,k) = 0.;
                pracg(i,k) = 0.;
                npracg(i,k) = 0.;
                psmlt(i,k) = 0.;
                pgmlt(i,k) = 0.;
                evpmg(i,k) = 0.;
                pracs(i,k) = 0.;
                npracs(i,k) = 0.;
                ngmltg(i,k) = 0.;
                ngmltr(i,k) = 0.;
                if (qc3d(i,k) >= 1.e-6) {
                  prc(i,k)=1350.*pow(qc3d(i,k),2.47)*pow(nc3d(i,k)/1.e6*rho(i,k),-1.79);
                  nprc1(i,k) = prc(i,k)/cons29;
                  nprc(i,k) = prc(i,k)/(qc3d(i,k)/nc3d(i,k));
                  nprc(i,k) = min( nprc(i,k) , nc3d(i,k)/dt );
                  nprc1(i,k) = min( nprc1(i,k) , nprc(i,k)    );
                }
                if (qr3d(i,k) >= 1.e-8 && qni3d(i,k) >= 1.e-8) {
                  float ums = asn(i,k)*cons3/pow(lams(i,k),bs);
                  float umr = arn(i,k)*cons4/pow(lamr(i,k),br);
                  float uns = asn(i,k)*cons5/pow(lams(i,k),bs);
                  float unr = arn(i,k)*cons6/pow(lamr(i,k),br);
                  float dum = pow(rhosu/rho(i,k),0.54);
                  ums = min( ums , 1.2*dum );
                  uns = min( uns , 1.2*dum );
                  umr = min( umr , 9.1*dum );
                  unr = min( unr , 9.1*dum );
                  pracs(i,k) = cons41*(pow(pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/pow(lamr(i,k),3)*
                              (5./(pow(lamr(i,k),3)*lams(i,k))+2./(pow(lamr(i,k),2)*pow(lams(i,k),2))+0.5/(lamr(i,k)*pow(lams(i,k),3))));
                }
                if (qr3d(i,k) >= 1.e-8 && qg3d(i,k) >= 1.e-8) {
                  float umg = agn(i,k)*cons7/pow(lamg(i,k),bg);
                  float umr = arn(i,k)*cons4/pow(lamr(i,k),br);
                  float ung = agn(i,k)*cons8/pow(lamg(i,k),bg);
                  float unr = arn(i,k)*cons6/pow(lamr(i,k),br);
                  float dum = pow(rhosu/rho(i,k),0.54);
                  umg = min( umg , 20.*dum );
                  ung = min( ung , 20.*dum );
                  umr = min( umr , 9.1*dum );
                  unr = min( unr , 9.1*dum );
                  pracg(i,k)  = cons41*(pow(pow(1.2*umr-0.95*umg,2)+0.08*umg*umr,0.5)*rho(i,k)*n0rr(i,k)*n0g(i,k)/pow(lamr(i,k),3)*
                              (5./(pow(lamr(i,k),3)*lamg(i,k))+2./(pow(lamr(i,k),2)*pow(lamg(i,k),2))+0.5/(lamr(i,k)*pow(lamg(i,k),3))));
                  dum       = pracg(i,k)/5.2e-7;
                  npracg(i,k) = cons32*rho(i,k)*pow(1.7*pow(unr-ung,2)+0.3*unr*ung,0.5)*n0rr(i,k)*n0g(i,k)*
                              (1./(pow(lamr(i,k),3)*lamg(i,k))+1./(pow(lamr(i,k),2)*pow(lamg(i,k),2))+1./(lamr(i,k)*pow(lamg(i,k),3)));
                  npracg(i,k) = npracg(i,k)-dum;
                }
                if (qr3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= 1.e-8) {
                  dum     = (qc3d(i,k)*qr3d(i,k));
                  pra(i,k) = 67.*pow(dum,1.15);
                  npra(i,k) = pra(i,k)/(qc3d(i,k)/nc3d(i,k));
                }
                if (qr3d(i,k) >= 1.e-8) {
                  float dum1=300.e-6;
                  if (1./lamr(i,k) < dum1) {
                    dum=1.;
                  } else if (1./lamr(i,k) >= dum1) {
                    dum=2.-std::exp(2300.*(1./lamr(i,k)-dum1));
                  }
                  nragg(i,k) = -5.78*dum*nr3d(i,k)*qr3d(i,k)*rho(i,k);
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          float dum;
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if ( ! no_cirg(i,k)) {  // If there's cloud or ice or rain or graupel
                float epsr;
                if (qr3d(i,k) >= qsmall) {
                  epsr = 2.*pi*n0rr(i,k)*rho(i,k)*dv(i,k)*(f1r/(lamr(i,k)*lamr(i,k))+f2r*pow(arn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons9/(pow(lamr(i,k),cons34)));
                } else {
                  epsr = 0.;
                }
                if (qv3d(i,k) < qvs(i,k)) {
                  pre(i,k) = epsr*(qv3d(i,k)-qvs(i,k))/ab(i,k);
                  pre(i,k) = min(pre(i,k),0.);
                } else {
                  pre(i,k) = 0.;
                }
                if (qni3d(i,k) >= 1.e-8) {
                  dum      = -cpw/xlf(i,k)*(t3d(i,k)-273.15)*pracs(i,k);
                  psmlt(i,k) = 2.*pi*n0s(i,k)*kap(i,k)*(273.15-t3d(i,k))/xlf(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons10/(pow(lams(i,k),cons35)))+dum;
                  if (qvqvs(i,k) < 1.) {
                    float epss     = 2.*pi*n0s(i,k)*rho(i,k)*dv(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons10/(pow(lams(i,k),cons35)));
                    evpms(i,k) = (qv3d(i,k)-qvs(i,k))*epss/ab(i,k)    ;
                    evpms(i,k) = max(evpms(i,k),psmlt(i,k));
                    psmlt(i,k) = psmlt(i,k)-evpms(i,k);
                  }
                }
                if (qg3d(i,k) >= 1.e-8) {
                  dum      = -cpw/xlf(i,k)*(t3d(i,k)-273.15)*pracg(i,k);
                  pgmlt(i,k) = 2.*pi*n0g(i,k)*kap(i,k)*(273.15-t3d(i,k))/xlf(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*pow(agn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons11/(pow(lamg(i,k),cons36)))+dum;
                  if (qvqvs(i,k) < 1.) {
                    float epsg     = 2.*pi*n0g(i,k)*rho(i,k)*dv(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*pow(agn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons11/(pow(lamg(i,k),cons36)));
                    evpmg(i,k) = (qv3d(i,k)-qvs(i,k))*epsg/ab(i,k);
                    evpmg(i,k) = max(evpmg(i,k),pgmlt(i,k));
                    pgmlt(i,k) = pgmlt(i,k)-evpmg(i,k);
                  }
                }
                pracg(i,k) = 0.;
                pracs(i,k) = 0.;
                dum = (prc(i,k)+pra(i,k))*dt;
                if (dum > qc3d(i,k) && qc3d(i,k) >= qsmall) {
                  float ratio = qc3d(i,k)/dum;
                  prc(i,k) = prc(i,k)*ratio;
                  pra(i,k) = pra(i,k)*ratio;
                }
                dum = (-psmlt(i,k)-evpms(i,k)+pracs(i,k))*dt;
                if (dum > qni3d(i,k) && qni3d(i,k) >= qsmall) {
                  float ratio    = qni3d(i,k)/dum;
                  psmlt(i,k) = psmlt(i,k)*ratio;
                  evpms(i,k) = evpms(i,k)*ratio;
                  pracs(i,k) = pracs(i,k)*ratio;
                }
                dum = (-pgmlt(i,k)-evpmg(i,k)+pracg(i,k))*dt;
                if (dum > qg3d(i,k) && qg3d(i,k) >= qsmall) {
                  float ratio    = qg3d (i,k)/dum;
                  pgmlt(i,k) = pgmlt(i,k)*ratio;
                  evpmg(i,k) = evpmg(i,k)*ratio;
                  pracg(i,k) = pracg(i,k)*ratio;
                }
                dum = (-pracs(i,k)-pracg(i,k)-pre(i,k)-pra(i,k)-prc(i,k)+psmlt(i,k)+pgmlt(i,k))*dt;
                if (dum > qr3d(i,k) && qr3d(i,k) >= qsmall) {
                  float ratio  = (qr3d(i,k)/dt+pracs(i,k)+pracg(i,k)+pra(i,k)+prc(i,k)-psmlt(i,k)-pgmlt(i,k))/(-pre(i,k));
                  pre(i,k) = pre(i,k)*ratio;
                }
                qv3dten (i,k) = qv3dten (i,k) + (-pre(i,k)-evpms(i,k)-evpmg(i,k));
                t3dten  (i,k) = t3dten  (i,k) + (pre(i,k)*xxlv(i,k)+(evpms(i,k)+evpmg(i,k))*xxls(i,k)+(psmlt(i,k)+pgmlt(i,k)-pracs(i,k)-pracg(i,k))*xlf(i,k))/cpm(i,k);
                qc3dten (i,k) = qc3dten (i,k) + (-pra(i,k)-prc(i,k));;
                qr3dten (i,k) = qr3dten (i,k) + (pre(i,k)+pra(i,k)+prc(i,k)-psmlt(i,k)-pgmlt(i,k)+pracs(i,k)+pracg(i,k));
                qni3dten(i,k) = qni3dten(i,k) + (psmlt(i,k)+evpms(i,k)-pracs(i,k));
                qg3dten (i,k) = qg3dten (i,k) + (pgmlt(i,k)+evpmg(i,k)-pracg(i,k));
                nc3dten (i,k) = nc3dten (i,k) + (-npra(i,k)-nprc(i,k));
                nr3dten (i,k) = nr3dten (i,k) + (nprc1(i,k)+nragg(i,k)-npracg(i,k));
                c2prec  (i,k) = pra(i,k)+prc(i,k);
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          float dum;
          if (! skip_micro(i,k)) {
            if (t_ge_273(i,k)) {
              if ( ! no_cirg(i,k)) {  // If there's cloud or ice or rain or graupel
                if (pre(i,k) < 0.) {
                  dum      = pre(i,k)*dt/qr3d(i,k);
                  dum      = max(-1.,dum);
                  nsubr(i,k) = dum*nr3d(i,k)/dt;
                }
                if (evpms(i,k)+psmlt(i,k) < 0.) {
                  dum       = (evpms(i,k)+psmlt(i,k))*dt/qni3d(i,k);
                  dum       = max(-1.,dum);
                  nsmlts(i,k) = dum*ns3d(i,k)/dt;
                }
                if (psmlt(i,k) < 0.) {
                  dum       = psmlt(i,k)*dt/qni3d(i,k);
                  dum       = max(-1.0,dum);
                  nsmltr(i,k) = dum*ns3d(i,k)/dt;
                }
                if (evpmg(i,k)+pgmlt(i,k) < 0.) {
                  dum       = (evpmg(i,k)+pgmlt(i,k))*dt/qg3d(i,k);
                  dum       = max(-1.,dum);
                  ngmltg(i,k) = dum*ng3d(i,k)/dt;
                }
                if (pgmlt(i,k) < 0.) {
                  dum       = pgmlt(i,k)*dt/qg3d(i,k);
                  dum       = max(-1.0,dum);
                  ngmltr(i,k) = dum*ng3d(i,k)/dt;
                }
                ns3dten(i,k) = ns3dten(i,k)+(nsmlts(i,k));
                ng3dten(i,k) = ng3dten(i,k)+(ngmltg(i,k));
                nr3dten(i,k) = nr3dten(i,k)+(nsubr(i,k)-nsmltr(i,k)-ngmltr(i,k));
              } // if ( ! no_cirg(i,k)) 
              float dumt   = t3d(i,k)+dt*t3dten(i,k);
              float dumqv  = qv3d(i,k)+dt*qv3dten(i,k);
              float dum=min(0.99*pres(i,k),polysvp(dumt,0));
              float dumqss = ep_2*dum/(pres(i,k)-dum);
              float dumqc  = qc3d(i,k)+dt*qc3dten(i,k);
              dumqc  = max(dumqc,0.);
              float dums   = dumqv-dumqss;
              pcc    (i,k) = dums/(1.+pow(xxlv(i,k),2)*dumqss/(cpm(i,k)*rv*pow(dumt,2)))/dt;
              if (pcc(i,k)*dt+dumqc < 0.)  pcc(i,k) = -dumqc/dt;
              qv3dten(i,k) = qv3dten(i,k)-pcc(i,k);
              t3dten (i,k) = t3dten (i,k)+pcc(i,k)*xxlv(i,k)/cpm(i,k);
              qc3dten(i,k) = qc3dten(i,k)+pcc(i,k);
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              if (iinum==1) {
                nc3d(i,k)=ndcnst*1.e6/rho(i,k);
              }
              ni3d(i,k) = max(0.,ni3d(i,k));
              ns3d(i,k) = max(0.,ns3d(i,k));
              nc3d(i,k) = max(0.,nc3d(i,k));
              nr3d(i,k) = max(0.,nr3d(i,k));
              ng3d(i,k) = max(0.,ng3d(i,k));
              if (qi3d(i,k) >= qsmall) {
                lami(i,k) = pow(cons12*ni3d(i,k)/qi3d(i,k),1./di);
                n0i(i,k) = ni3d(i,k)*lami(i,k);
                if (lami(i,k) < lammini) {
                  lami(i,k) = lammini;
                  n0i(i,k) = pow(lami(i,k),4)*qi3d(i,k)/cons12;
                  ni3d(i,k) = n0i(i,k)/lami(i,k);
                } else if (lami(i,k) > lammaxi) {
                  lami(i,k) = lammaxi;
                  n0i(i,k) = pow(lami(i,k),4)*qi3d(i,k)/cons12;
                  ni3d(i,k) = n0i(i,k)/lami(i,k);
                }
              }
              if (qr3d(i,k) >= qsmall) {
                lamr(i,k) = pow(pi*rhow*nr3d(i,k)/qr3d(i,k),1./3.);
                n0rr(i,k) = nr3d(i,k)*lamr(i,k);
                if (lamr(i,k) < lamminr) {
                  lamr(i,k) = lamminr;
                  n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                  nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                } else if (lamr(i,k) > lammaxr) {
                  lamr(i,k) = lammaxr;
                  n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                  nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                }
              }
              if (qc3d(i,k) >= qsmall) {
                float dum     = pres(i,k)/(287.15*t3d(i,k));
                pgam(i,k) = 0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714;
                pgam(i,k) = 1./(pow(pgam(i,k),2))-1.;
                pgam(i,k) = max(pgam(i,k),2.);
                pgam(i,k) = min(pgam(i,k),10.);
                lamc(i,k) = pow(cons26*nc3d(i,k)*gamma(pgam(i,k)+4.)/(qc3d(i,k)*gamma(pgam(i,k)+1.)),1./3.);
                float lammin  = (pgam(i,k)+1.)/60.e-6;
                float lammax  = (pgam(i,k)+1.)/1.e-6;
                if (lamc(i,k) < lammin) {
                  lamc(i,k) = lammin;
                  nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(gamma(pgam(i,k)+1.))-std::log(gamma(pgam(i,k)+4.)))/cons26;
                } else if (lamc(i,k) > lammax) {
                  lamc(i,k) = lammax;
                  nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(gamma(pgam(i,k)+1.))-std::log(gamma(pgam(i,k)+4.)))/cons26;
                }
                cdist1(i,k) = nc3d(i,k)/gamma(pgam(i,k)+1.);
              }
              if (qni3d(i,k) >= qsmall) {
                lams(i,k) = pow(cons1*ns3d(i,k)/qni3d(i,k),1./ds);
                n0s(i,k) = ns3d(i,k)*lams(i,k);
                if (lams(i,k) < lammins) {
                  lams(i,k) = lammins;
                  n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1;
                  ns3d(i,k) = n0s(i,k)/lams(i,k);
                } else if (lams(i,k) > lammaxs) {
                  lams(i,k) = lammaxs;
                  n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1;
                  ns3d(i,k) = n0s(i,k)/lams(i,k);
                }
              }
              if (qg3d(i,k) >= qsmall) {
                lamg(i,k) = pow(cons2*ng3d(i,k)/qg3d(i,k),1./dg);
                n0g(i,k) = ng3d(i,k)*lamg(i,k);
                if (lamg(i,k) < lamming) {
                  lamg(i,k) = lamming;
                  n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                  ng3d(i,k) = n0g(i,k)/lamg(i,k);
                } else if (lamg(i,k) > lammaxg) {
                  lamg(i,k) = lammaxg;
                  n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                  ng3d(i,k) = n0g(i,k)/lamg(i,k);
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              mnuccc(i,k) = 0.;
              nnuccc(i,k) = 0.;
              prc(i,k) = 0.;
              nprc(i,k) = 0.;
              nprc1(i,k) = 0.;
              nsagg(i,k) = 0.;
              psacws(i,k) = 0.;
              npsacws(i,k) = 0.;
              psacwi(i,k) = 0.;
              npsacwi(i,k) = 0.;
              pracs(i,k) = 0.;
              npracs(i,k) = 0.;
              nmults(i,k) = 0.;
              qmults(i,k) = 0.;
              nmultr(i,k) = 0.;
              qmultr(i,k) = 0.;
              nmultg(i,k) = 0.;
              qmultg(i,k) = 0.;
              nmultrg(i,k) = 0.;
              qmultrg(i,k) = 0.;
              mnuccr(i,k) = 0.;
              nnuccr(i,k) = 0.;
              pra(i,k) = 0.;
              npra(i,k) = 0.;
              nragg(i,k) = 0.;
              prci(i,k) = 0.;
              nprci(i,k) = 0.;
              prai(i,k) = 0.;
              nprai(i,k) = 0.;
              nnuccd(i,k) = 0.;
              mnuccd(i,k) = 0.;
              pcc(i,k) = 0.;
              pre(i,k) = 0.;
              prd(i,k) = 0.;
              prds(i,k) = 0.;
              eprd(i,k) = 0.;
              eprds(i,k) = 0.;
              nsubc(i,k) = 0.;
              nsubi(i,k) = 0.;
              nsubs(i,k) = 0.;
              nsubr(i,k) = 0.;
              piacr(i,k) = 0.;
              niacr(i,k) = 0.;
              praci(i,k) = 0.;
              piacrs(i,k) = 0.;
              niacrs(i,k) = 0.;
              pracis(i,k) = 0.;
              pracg(i,k) = 0.;
              psacr(i,k) = 0.;
              psacwg(i,k) = 0.;
              pgsacw(i,k) = 0.;
              pgracs(i,k) = 0.;
              prdg(i,k) = 0.;
              eprdg(i,k) = 0.;
              npracg(i,k) = 0.;
              npsacwg(i,k) = 0.;
              nscng(i,k) = 0.;
              ngracs(i,k) = 0.;
              nsubg(i,k) = 0.;
              if (qc3d(i,k) >= qsmall  &&  t3d(i,k) < 269.15) {
                float nacnt     = std::exp(-2.80+0.262*(273.15-t3d(i,k)))*1000.;
                float dum       = 7.37*t3d(i,k)/(288.*10.*pres(i,k))/100.;
                dap(i,k) = cons37*t3d(i,k)*(1.+dum/rin)/mu(i,k);
                mnuccc(i,k) = cons38*dap(i,k)*nacnt*std::exp(std::log(cdist1(i,k))+std::log(gamma(pgam(i,k)+5.))-4.*std::log(lamc(i,k)));
                nnuccc(i,k) = 2.*pi*dap(i,k)*nacnt*cdist1(i,k)*gamma(pgam(i,k)+2.)/lamc(i,k);
                mnuccc(i,k) = mnuccc(i,k)+cons39*std::exp(std::log(cdist1(i,k))+std::log(gamma(7.+pgam(i,k)))-6.*std::log(lamc(i,k)))*(std::exp(aimm*(273.15-t3d(i,k)))-1.);
                nnuccc(i,k) = nnuccc(i,k)+cons40*std::exp(std::log(cdist1(i,k))+std::log(gamma(pgam(i,k)+4.))-3.*std::log(lamc(i,k)))*(std::exp(aimm*(273.15-t3d(i,k)))-1.);
                nnuccc(i,k) = min(nnuccc(i,k),nc3d(i,k)/dt);
              }
              if (qc3d(i,k) >= 1.e-6) {
                prc(i,k) = 1350.*pow(qc3d(i,k),2.47)*pow(nc3d(i,k)/1.e6*rho(i,k),-1.79);
                nprc1(i,k) = prc(i,k)/cons29;
                nprc(i,k) = prc(i,k)/(qc3d(i,k)/nc3d(i,k));
                nprc(i,k) = min( nprc(i,k) , nc3d(i,k)/dt );
                nprc1(i,k) = min( nprc1(i,k) , nprc(i,k)    );
              }
              if (qni3d(i,k) >= 1.e-8) {
                nsagg(i,k) = cons15*asn(i,k)*pow(rho(i,k),(2.+bs)/3.)*pow(qni3d(i,k),(2.+bs)/3.)*pow(ns3d(i,k)*rho(i,k),(4.-bs)/3.)/(rho(i,k));
              }
              if (qni3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= qsmall) {
                psacws(i,k) = cons13*asn(i,k)*qc3d(i,k)*rho(i,k)*n0s(i,k)/pow(lams(i,k),bs+3.);
                npsacws(i,k) = cons13*asn(i,k)*nc3d(i,k)*rho(i,k)*n0s(i,k)/pow(lams(i,k),bs+3.);
              }
              if (qg3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= qsmall) {
                psacwg(i,k) = cons14*agn(i,k)*qc3d(i,k)*rho(i,k)*n0g(i,k)/pow(lamg(i,k),bg+3.);
                npsacwg(i,k) = cons14*agn(i,k)*nc3d(i,k)*rho(i,k)*n0g(i,k)/pow(lamg(i,k),bg+3.);
              }
              if (qi3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= qsmall) {
                if (1./lami(i,k) >= 100.e-6) {
                  psacwi(i,k) = cons16*ain(i,k)*qc3d(i,k)*rho(i,k)*n0i(i,k)/pow(lami(i,k),bi+3.);
                  npsacwi(i,k) = cons16*ain(i,k)*nc3d(i,k)*rho(i,k)*n0i(i,k)/pow(lami(i,k),bi+3.);
                }
              }
              if (qr3d(i,k) >= 1.e-8 && qni3d(i,k) >= 1.e-8) {
                float ums = asn(i,k)*cons3/pow(lams(i,k),bs);
                float umr = arn(i,k)*cons4/pow(lamr(i,k),br);
                float uns = asn(i,k)*cons5/pow(lams(i,k),bs);
                float unr = arn(i,k)*cons6/pow(lamr(i,k),br);
                float dum = pow(rhosu/rho(i,k),0.54);
                ums = min( ums , 1.2*dum );
                uns = min( uns , 1.2*dum );
                umr = min( umr , 9.1*dum );
                unr = min( unr , 9.1*dum );
                pracs(i,k) = cons41*(pow(pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/pow(lamr(i,k),3)*
                           (5./(pow(lamr(i,k),3)*lams(i,k))+2./(pow(lamr(i,k),2)*pow(lams(i,k),2))+0.5/(lamr(i,k)*pow(lams(i,k),3))));
                npracs(i,k) = cons32*rho(i,k)*pow(1.7*pow(unr-uns,2)+0.3*unr*uns,0.5)*n0rr(i,k)*n0s(i,k)*(1./(pow(lamr(i,k),3)*lams(i,k))+
                            1./(pow(lamr(i,k),2)*pow(lams(i,k),2))+1./(lamr(i,k)*pow(lams(i,k),3)));
                pracs(i,k) = min(pracs(i,k),qr3d(i,k)/dt);
                if (qni3d(i,k) >= 0.1e-3 && qr3d(i,k) >= 0.1e-3) {
                  psacr(i,k) = cons31*(pow(pow(1.2*umr-0.95*ums,2)+0.08*ums*umr,0.5)*rho(i,k)*n0rr(i,k)*n0s(i,k)/pow(lams(i,k),3)* 
                             (5./(pow(lams(i,k),3)*lamr(i,k))+2./(pow(lams(i,k),2)*pow(lamr(i,k),2))+0.5/(lams(i,k)*pow(lamr(i,k),3))))            ;
                }
              }
              if (qr3d(i,k) >= 1.e-8 && qg3d(i,k) >= 1.e-8) {
                float umg = agn(i,k)*cons7/pow(lamg(i,k),bg);
                float umr = arn(i,k)*cons4/pow(lamr(i,k),br);
                float ung = agn(i,k)*cons8/pow(lamg(i,k),bg);
                float unr = arn(i,k)*cons6/pow(lamr(i,k),br);
                float dum = pow(rhosu/rho(i,k),0.54);
                umg = min( umg , 20.*dum );
                ung = min( ung , 20.*dum );
                umr = min( umr , 9.1*dum );
                unr = min( unr , 9.1*dum );
                pracg(i,k) = cons41*(pow(pow(1.2*umr-0.95*umg,2)+0.08*umg*umr,0.5)*rho(i,k)*n0rr(i,k)*n0g(i,k)/pow(lamr(i,k),3)* 
                            (5./(pow(lamr(i,k),3)*lamg(i,k))+2./(pow(lamr(i,k),2)*pow(lamg(i,k),2))+0.5/(lamr(i,k)*pow(lamg(i,k),3))));
                npracg(i,k) = cons32*rho(i,k)*pow(1.7*pow(unr-ung,2)+0.3*unr*ung,0.5)*n0rr(i,k)*n0g(i,k)*(1./(pow(lamr(i,k),3)*lamg(i,k))+
                            1./(pow(lamr(i,k),2)*pow(lamg(i,k),2))+1./(lamr(i,k)*pow(lamg(i,k),3)));
                pracg(i,k) = min(pracg(i,k),qr3d(i,k)/dt);
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              if (qni3d(i,k) >= 0.1e-3) {
                if (qc3d(i,k) >= 0.5e-3 || qr3d(i,k) >= 0.1e-3) {
                  if (psacws(i,k) > 0. || pracs(i,k) > 0.) {
                    if (t3d(i,k) < 270.16  &&  t3d(i,k) > 265.16) {
                      float fmult;
                      if (t3d(i,k) > 270.16) {
                        fmult = 0.;
                      } else if (t3d(i,k) <= 270.16 && t3d(i,k) > 268.16)  {
                        fmult = (270.16-t3d(i,k))/2.;
                      } else if (t3d(i,k) >= 265.16 && t3d(i,k) <= 268.16)   {
                        fmult = (t3d(i,k)-265.16)/3.;
                      } else if (t3d(i,k) < 265.16) {
                        fmult = 0.;
                      }
                      if (psacws(i,k) > 0.) {
                        nmults(i,k) = 35.e4*psacws(i,k)*fmult*1000.;
                        qmults(i,k) = nmults(i,k)*mmult;
                        qmults(i,k) = min(qmults(i,k),psacws(i,k));
                        psacws(i,k) = psacws(i,k)-qmults(i,k);
                      }
                      if (pracs(i,k) > 0.) {
                        nmultr(i,k) = 35.e4*pracs(i,k)*fmult*1000.;
                        qmultr(i,k) = nmultr(i,k)*mmult;
                        qmultr(i,k) = min(qmultr(i,k),pracs(i,k));
                        pracs(i,k) = pracs(i,k)-qmultr(i,k);
                      }
                    }
                  }
                }
              }
              if (qg3d(i,k) >= 0.1e-3) {
                if (qc3d(i,k) >= 0.5e-3 || qr3d(i,k) >= 0.1e-3) {
                  if (psacwg(i,k) > 0. || pracg(i,k) > 0.) {
                    if (t3d(i,k) < 270.16  &&  t3d(i,k) > 265.16) {
                      float fmult;
                      if (t3d(i,k) > 270.16) {
                        fmult = 0.;
                      } else if (t3d(i,k) <= 270.16 && t3d(i,k) > 268.16)  {
                        fmult = (270.16-t3d(i,k))/2.;
                      } else if (t3d(i,k) >= 265.16 && t3d(i,k) <= 268.16)   {
                        fmult = (t3d(i,k)-265.16)/3.;
                      } else if (t3d(i,k) < 265.16) {
                        fmult = 0.;
                      }
                      if (psacwg(i,k) > 0.) {
                        nmultg(i,k) = 35.e4*psacwg(i,k)*fmult*1000.;
                        qmultg(i,k) = nmultg(i,k)*mmult;
                        qmultg(i,k) = min(qmultg(i,k),psacwg(i,k));
                        psacwg(i,k) = psacwg(i,k)-qmultg(i,k);
                      }
                      if (pracg(i,k) > 0.) {
                        nmultrg(i,k) = 35.e4*pracg(i,k)*fmult*1000.;
                        qmultrg(i,k) = nmultrg(i,k)*mmult;
                        qmultrg(i,k) = min(qmultrg(i,k),pracg(i,k));
                        pracg(i,k) = pracg(i,k)-qmultrg(i,k);
                      }
                    }
                  }
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              if (psacws(i,k) > 0.) {
                if (qni3d(i,k) >= 0.1e-3 && qc3d(i,k) >= 0.5e-3) {
                  pgsacw(i,k) = min(psacws(i,k),cons17*dt*n0s(i,k)*qc3d(i,k)*qc3d(i,k)*asn(i,k)*asn(i,k)/(rho(i,k)*pow(lams(i,k),2.*bs+2.)));
                  float dum       = max(rhosn/(rhog-rhosn)*pgsacw(i,k),0.) ;
                  nscng(i,k) = dum/mg0*rho(i,k);
                  nscng(i,k) = min(nscng(i,k),ns3d(i,k)/dt);
                  psacws(i,k) = psacws(i,k) - pgsacw(i,k);
                }
              }
              if (pracs(i,k) > 0.) {
                if (qni3d(i,k) >= 0.1e-3 && qr3d(i,k) >= 0.1e-3) {
                  float dum = cons18*pow(4./lams(i,k),3)*pow(4./lams(i,k),3)/(cons18*pow(4./lams(i,k),3)*pow(4./lams(i,k),3)+
                              cons19*pow(4./lamr(i,k),3)*pow(4./lamr(i,k),3));
                  dum       = min( dum , 1. );
                  dum       = max( dum , 0. );
                  pgracs(i,k) = (1.-dum)*pracs(i,k);
                  ngracs(i,k) = (1.-dum)*npracs(i,k);
                  ngracs(i,k) = min(ngracs(i,k),nr3d(i,k)/dt);
                  ngracs(i,k) = min(ngracs(i,k),ns3d(i,k)/dt);
                  pracs(i,k) = pracs(i,k) - pgracs(i,k);
                  npracs(i,k) = npracs(i,k) - ngracs(i,k);
                  psacr(i,k) = psacr(i,k)*(1.-dum);
                }
              }
              if (t3d(i,k) < 269.15 && qr3d(i,k) >= qsmall) {
                mnuccr(i,k) = cons20*nr3d(i,k)*(std::exp(aimm*(273.15-t3d(i,k)))-1.)/pow(lamr(i,k),3)/pow(lamr(i,k),3);
                nnuccr(i,k) = pi*nr3d(i,k)*bimm*(std::exp(aimm*(273.15-t3d(i,k)))-1.)/pow(lamr(i,k),3);
                nnuccr(i,k) = min(nnuccr(i,k),nr3d(i,k)/dt);
              }
              if (qr3d(i,k) >= 1.e-8  &&  qc3d(i,k) >= 1.e-8) {
                float dum     = (qc3d(i,k)*qr3d(i,k));
                pra(i,k) = 67.*pow(dum,1.15);
                npra(i,k) = pra(i,k)/(qc3d(i,k)/nc3d(i,k));
              }
              if (qr3d(i,k) >= 1.e-8) {
                float dum1=300.e-6;
                float dum;
                if (1./lamr(i,k) < dum1) {
                  dum=1.;
                } else if (1./lamr(i,k) >= dum1) {
                  dum=2.-std::exp(2300.*(1./lamr(i,k)-dum1));
                }
                nragg(i,k) = -5.78*dum*nr3d(i,k)*qr3d(i,k)*rho(i,k);
              }
              if (qi3d(i,k) >= 1.e-8  && qvqvsi(i,k) >= 1.) {
                nprci(i,k) = cons21*(qv3d(i,k)-qvi(i,k))*rho(i,k)*n0i(i,k)*std::exp(-lami(i,k)*dcs)*dv(i,k)/abi(i,k);
                prci(i,k) = cons22*nprci(i,k);
                nprci(i,k) = min(nprci(i,k),ni3d(i,k)/dt);
              }
              if (qni3d(i,k) >= 1.e-8  &&  qi3d(i,k) >= qsmall) {
                prai(i,k) = cons23*asn(i,k)*qi3d(i,k)*rho(i,k)*n0s(i,k)/pow(lams(i,k),bs+3.);
                nprai(i,k) = cons23*asn(i,k)*ni3d(i,k)*rho(i,k)*n0s(i,k)/pow(lams(i,k),bs+3.);
                nprai(i,k) = min( nprai(i,k) , ni3d(i,k)/dt );
              }
              if (qr3d(i,k) >= 1.e-8  &&  qi3d(i,k) >= 1.e-8  &&  t3d(i,k) <= 273.15) {
                if (qr3d(i,k) >= 0.1e-3) {
                  niacr(i,k)=cons24*ni3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)*rho(i,k);
                  piacr(i,k)=cons25*ni3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)/pow(lamr(i,k),3)*rho(i,k);
                  praci(i,k)=cons24*qi3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)*rho(i,k);
                  niacr(i,k)=min(niacr(i,k),nr3d(i,k)/dt);
                  niacr(i,k)=min(niacr(i,k),ni3d(i,k)/dt);
                } else {
                  niacrs(i,k)=cons24*ni3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)*rho(i,k);
                  piacrs(i,k)=cons25*ni3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)/pow(lamr(i,k),3)*rho(i,k);
                  pracis(i,k)=cons24*qi3d(i,k)*n0rr(i,k)*arn(i,k)/pow(lamr(i,k),br+3.)*rho(i,k);
                  niacrs(i,k)=min(niacrs(i,k),nr3d(i,k)/dt);
                  niacrs(i,k)=min(niacrs(i,k),ni3d(i,k)/dt);
                }
              }
              if (inuc==0) {
                if ((qvqvs(i,k) >= 0.999  &&  t3d(i,k) <= 265.15)  ||  qvqvsi(i,k) >= 1.08) {
                  float kc2 = 0.005*std::exp(0.304*(273.15-t3d(i,k)))*1000.;
                  kc2 = min( kc2 ,500.e3 );
                  kc2 = max( kc2/rho(i,k) , 0. );
                  if (kc2 > ni3d(i,k)+ns3d(i,k)+ng3d(i,k)) {
                    nnuccd(i,k) = (kc2-ni3d(i,k)-ns3d(i,k)-ng3d(i,k))/dt;
                    mnuccd(i,k) = nnuccd(i,k)*mi0;
                  }
                }
              } else if (inuc==1) {
                if (t3d(i,k) < 273.15 && qvqvsi(i,k) > 1.) {
                  float kc2 = 0.16*1000./rho(i,k);
                  if (kc2 > ni3d(i,k)+ns3d(i,k)+ng3d(i,k)) {
                    nnuccd(i,k) = (kc2-ni3d(i,k)-ns3d(i,k)-ng3d(i,k))/dt;
                    mnuccd(i,k) = nnuccd(i,k)*mi0;
                  }
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              float epsi;
              if (qi3d(i,k) >= qsmall) {
                 epsi = 2.*pi*n0i(i,k)*rho(i,k)*dv(i,k)/(lami(i,k)*lami(i,k));
              } else {
                 epsi = 0.;
              }
              float epss;
              if (qni3d(i,k) >= qsmall) {
                epss = 2.*pi*n0s(i,k)*rho(i,k)*dv(i,k)*(f1s/(lams(i,k)*lams(i,k))+f2s*pow(asn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons10/(pow(lams(i,k),cons35)));
              } else {
                epss = 0.;
              }
              float epsg;
              if (qg3d(i,k) >= qsmall) {
                epsg = 2.*pi*n0g(i,k)*rho(i,k)*dv(i,k)*(f1s/(lamg(i,k)*lamg(i,k))+f2s*pow(agn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons11/(pow(lamg(i,k),cons36)));
              } else {
                epsg = 0.;
              }
              float epsr;
              if (qr3d(i,k) >= qsmall) {
                epsr = 2.*pi*n0rr(i,k)*rho(i,k)*dv(i,k)*(f1r/(lamr(i,k)*lamr(i,k))+f2r*pow(arn(i,k)*rho(i,k)/mu(i,k),0.5)*pow(sc(i,k),1./3.)*cons9/(pow(lamr(i,k),cons34)));
              } else {
                epsr = 0.;
              }
              float dum;
              if (qi3d(i,k) >= qsmall) {              
                dum    = (1.-std::exp(-lami(i,k)*dcs)*(1.+lami(i,k)*dcs));
                prd(i,k) = epsi*(qv3d(i,k)-qvi(i,k))/abi(i,k)*dum;
              } else {
                dum=0.;
              }
              if (qni3d(i,k) >= qsmall) {
                prds(i,k) = epss*(qv3d(i,k)-qvi(i,k))/abi(i,k)+epsi*(qv3d(i,k)-qvi(i,k))/abi(i,k)*(1.-dum);
              } else {
                prd(i,k) = prd(i,k)+epsi*(qv3d(i,k)-qvi(i,k))/abi(i,k)*(1.-dum);
              }
              prdg(i,k) = epsg*(qv3d(i,k)-qvi(i,k))/abi(i,k);
              if (qv3d(i,k) < qvs(i,k)) {
                pre(i,k) = epsr*(qv3d(i,k)-qvs(i,k))/ab(i,k);
                pre(i,k) = min( pre(i,k) , 0. );
              } else {
                pre(i,k) = 0.;
              }
              dum = (qv3d(i,k)-qvi(i,k))/dt;
              float fudgef = 0.9999;
              float sum_dep = prd(i,k)+prds(i,k)+mnuccd(i,k)+prdg(i,k);
              if( (dum > 0.  &&  sum_dep > dum*fudgef)  ||  (dum < 0.  &&  sum_dep < dum*fudgef) ) {
                mnuccd(i,k) = fudgef*mnuccd(i,k)*dum/sum_dep;
                prd(i,k) = fudgef*prd(i,k)*dum/sum_dep;
                prds(i,k) = fudgef*prds(i,k)*dum/sum_dep;
                prdg(i,k) = fudgef*prdg(i,k)*dum/sum_dep;
              }
              if (prd(i,k) < 0.) {
                eprd(i,k)=prd(i,k);
                prd(i,k)=0.;
              }
              if (prds(i,k) < 0.) {
                eprds(i,k)=prds(i,k);
                prds(i,k)=0.;
              }
              if (prdg(i,k) < 0.) {
                eprdg(i,k)=prdg(i,k);
                prdg(i,k)=0.;
              }
              if (iliq==1) {
                mnuccc(i,k)=0.;
                nnuccc(i,k)=0.;
                mnuccr(i,k)=0.;
                nnuccr(i,k)=0.;
                mnuccd(i,k)=0.;
                nnuccd(i,k)=0.;
              }
              if (igraup==1) {
                pracg(i,k) = 0.;
                psacr(i,k) = 0.;
                psacwg(i,k) = 0.;
                prdg(i,k) = 0.;
                eprdg(i,k) = 0.;
                evpmg(i,k) = 0.;
                pgmlt(i,k) = 0.;
                npracg(i,k) = 0.;
                npsacwg(i,k) = 0.;
                nscng(i,k) = 0.;
                ngracs(i,k) = 0.;
                nsubg(i,k) = 0.;
                ngmltg(i,k) = 0.;
                ngmltr(i,k) = 0.;
                piacrs(i,k) = piacrs(i,k)+piacr(i,k);
                piacr(i,k) = 0.;
                pracis(i,k) = pracis(i,k)+praci(i,k);
                praci(i,k) = 0.;
                psacws(i,k) = psacws(i,k)+pgsacw(i,k);
                pgsacw(i,k) = 0.;
                pracs(i,k) = pracs(i,k)+pgracs(i,k);
                pgracs(i,k) = 0.;
              }
              dum = (prc(i,k)+pra(i,k)+mnuccc(i,k)+psacws(i,k)+psacwi(i,k)+qmults(i,k)+psacwg(i,k)+pgsacw(i,k)+qmultg(i,k))*dt;
              if (dum > qc3d(i,k)  &&  qc3d(i,k) >= qsmall) {
                float ratio = qc3d(i,k)/dum;
                prc(i,k) = prc(i,k)*ratio;
                pra(i,k) = pra(i,k)*ratio;
                mnuccc(i,k) = mnuccc(i,k)*ratio;
                psacws(i,k) = psacws(i,k)*ratio;
                psacwi(i,k) = psacwi(i,k)*ratio;
                qmults(i,k) = qmults(i,k)*ratio;
                qmultg(i,k) = qmultg(i,k)*ratio;
                psacwg(i,k) = psacwg(i,k)*ratio;
                pgsacw(i,k) = pgsacw(i,k)*ratio;
              }
              dum = (-prd(i,k)-mnuccc(i,k)+prci(i,k)+prai(i,k)-qmults(i,k)-qmultg(i,k)-qmultr(i,k)-qmultrg(i,k)-mnuccd(i,k)+praci(i,k)+pracis(i,k)-eprd(i,k)-psacwi(i,k))*dt;
              if (dum > qi3d(i,k)  &&  qi3d(i,k) >= qsmall) {
                float ratio = (qi3d(i,k)/dt+prd(i,k)+mnuccc(i,k)+qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+mnuccd(i,k)+psacwi(i,k))/(prci(i,k)+prai(i,k)+praci(i,k)+pracis(i,k)-eprd(i,k));
                prci(i,k) = prci(i,k)*ratio;
                prai(i,k) = prai(i,k)*ratio;
                praci(i,k) = praci(i,k)*ratio;
                pracis(i,k) = pracis(i,k)*ratio;
                eprd(i,k) = eprd(i,k)*ratio;
              }
              dum = ((pracs(i,k)-pre(i,k))+(qmultr(i,k)+qmultrg(i,k)-prc(i,k))+(mnuccr(i,k)-pra(i,k))+piacr(i,k)+piacrs(i,k)+pgracs(i,k)+pracg(i,k))*dt;
              if (dum > qr3d(i,k) && qr3d(i,k) >= qsmall) {
                float ratio = (qr3d(i,k)/dt+prc(i,k)+pra(i,k))/(-pre(i,k)+qmultr(i,k)+qmultrg(i,k)+pracs(i,k)+mnuccr(i,k)+piacr(i,k)+piacrs(i,k)+pgracs(i,k)+pracg(i,k));
                pre(i,k) = pre(i,k)*ratio;
                pracs(i,k) = pracs(i,k)*ratio;
                qmultr(i,k) = qmultr(i,k)*ratio;
                qmultrg(i,k) = qmultrg(i,k)*ratio;
                mnuccr(i,k) = mnuccr(i,k)*ratio;
                piacr(i,k) = piacr(i,k)*ratio;
                piacrs(i,k) = piacrs(i,k)*ratio;
                pgracs(i,k) = pgracs(i,k)*ratio;
                pracg(i,k) = pracg(i,k)*ratio;
              }
              if (igraup==0) {
                dum = (-prds(i,k)-psacws(i,k)-prai(i,k)-prci(i,k)-pracs(i,k)-eprds(i,k)+psacr(i,k)-piacrs(i,k)-pracis(i,k))*dt;
                if (dum > qni3d(i,k) && qni3d(i,k) >= qsmall) {
                  float ratio = (qni3d(i,k)/dt+prds(i,k)+psacws(i,k)+prai(i,k)+prci(i,k)+pracs(i,k)+piacrs(i,k)+pracis(i,k))/(-eprds(i,k)+psacr(i,k));
                  eprds(i,k) = eprds(i,k)*ratio;
                  psacr(i,k) = psacr(i,k)*ratio;
                }
              } else if (igraup==1) {
                dum = (-prds(i,k)-psacws(i,k)-prai(i,k)-prci(i,k)-pracs(i,k)-eprds(i,k)+psacr(i,k)-piacrs(i,k)-pracis(i,k)-mnuccr(i,k))*dt;
                if (dum > qni3d(i,k) && qni3d(i,k) >= qsmall) {
                  float ratio = (qni3d(i,k)/dt+prds(i,k)+psacws(i,k)+prai(i,k)+prci(i,k)+pracs(i,k)+piacrs(i,k)+pracis(i,k)+mnuccr(i,k))/(-eprds(i,k)+psacr(i,k));
                  eprds(i,k) = eprds(i,k)*ratio;
                  psacr(i,k) = psacr(i,k)*ratio;
                }
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          float dum;
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              dum = (-psacwg(i,k)-pracg(i,k)-pgsacw(i,k)-pgracs(i,k)-prdg(i,k)-mnuccr(i,k)-eprdg(i,k)-piacr(i,k)-praci(i,k)-psacr(i,k))*dt;
              if (dum > qg3d(i,k) && qg3d(i,k) >= qsmall) {
                float ratio = (qg3d(i,k)/dt+psacwg(i,k)+pracg(i,k)+pgsacw(i,k)+pgracs(i,k)+prdg(i,k)+mnuccr(i,k)+psacr(i,k)+piacr(i,k)+praci(i,k))/(-eprdg(i,k));
                eprdg(i,k) = eprdg(i,k)*ratio;
              }
              qv3dten(i,k) = qv3dten(i,k)+(-pre(i,k)-prd(i,k)-prds(i,k)-mnuccd(i,k)-eprd(i,k)-eprds(i,k)-prdg(i,k)-eprdg(i,k));
              t3dten(i,k) = t3dten(i,k)+(pre(i,k)*xxlv(i,k)+(prd(i,k)+prds(i,k)+mnuccd(i,k)+eprd(i,k)+eprds(i,k)+prdg(i,k)+eprdg(i,k))*xxls(i,k)+(psacws(i,k)+psacwi(i,k)+mnuccc(i,k)+mnuccr(i,k)+qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+pracs(i,k)+psacwg(i,k)+pracg(i,k)+pgsacw(i,k)+pgracs(i,k)+piacr(i,k)+piacrs(i,k))*xlf(i,k))/cpm(i,k);
              qc3dten(i,k) = qc3dten(i,k)+(-pra(i,k)-prc(i,k)-mnuccc(i,k)+pcc(i,k)-psacws(i,k)-psacwi(i,k)-qmults(i,k)-qmultg(i,k)-psacwg(i,k)-pgsacw(i,k));
              qi3dten(i,k) = qi3dten(i,k)+(prd(i,k)+eprd(i,k)+psacwi(i,k)+mnuccc(i,k)-prci(i,k)-prai(i,k)+qmults(i,k)+qmultg(i,k)+qmultr(i,k)+qmultrg(i,k)+mnuccd(i,k)-praci(i,k)-pracis(i,k));
              qr3dten(i,k) = qr3dten(i,k)+(pre(i,k)+pra(i,k)+prc(i,k)-pracs(i,k)-mnuccr(i,k)-qmultr(i,k)-qmultrg(i,k)-piacr(i,k)-piacrs(i,k)-pracg(i,k)-pgracs(i,k));
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          float dum;
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              if (igraup==0) {
                qni3dten(i,k) = qni3dten(i,k)+(prai(i,k)+psacws(i,k)+prds(i,k)+pracs(i,k)+prci(i,k)+eprds(i,k)-psacr(i,k)+piacrs(i,k)+pracis(i,k));
                ns3dten(i,k) = ns3dten(i,k)+(nsagg(i,k)+nprci(i,k)-nscng(i,k)-ngracs(i,k)+niacrs(i,k));
                qg3dten(i,k) = qg3dten(i,k)+(pracg(i,k)+psacwg(i,k)+pgsacw(i,k)+pgracs(i,k)+prdg(i,k)+eprdg(i,k)+mnuccr(i,k)+piacr(i,k)+praci(i,k)+psacr(i,k));
                ng3dten(i,k) = ng3dten(i,k)+(nscng(i,k)+ngracs(i,k)+nnuccr(i,k)+niacr(i,k));
              } else if (igraup==1) {
                qni3dten(i,k) = qni3dten(i,k)+(prai(i,k)+psacws(i,k)+prds(i,k)+pracs(i,k)+prci(i,k)+eprds(i,k)-psacr(i,k)+piacrs(i,k)+pracis(i,k)+mnuccr(i,k));
                ns3dten(i,k) = ns3dten(i,k)+(nsagg(i,k)+nprci(i,k)-nscng(i,k)-ngracs(i,k)+niacrs(i,k)+nnuccr(i,k));
              }
            }
          }
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (! skip_micro(i,k)) {
            if (! t_ge_273(i,k)) {
              nc3dten(i,k) = nc3dten(i,k)+(-nnuccc(i,k)-npsacws(i,k)-npra(i,k)-nprc(i,k)-npsacwi(i,k)-npsacwg(i,k));
              ni3dten(i,k) = ni3dten(i,k)+(nnuccc(i,k)-nprci(i,k)-nprai(i,k)+nmults(i,k)+nmultg(i,k)+nmultr(i,k)+nmultrg(i,k)+nnuccd(i,k)-niacr(i,k)-niacrs(i,k));
              nr3dten(i,k) = nr3dten(i,k)+(nprc1(i,k)-npracs(i,k)-nnuccr(i,k)+nragg(i,k)-niacr(i,k)-niacrs(i,k)-npracg(i,k)-ngracs(i,k));
              c2prec (i,k) = pra(i,k)+prc(i,k)+psacws(i,k)+qmults(i,k)+qmultg(i,k)+psacwg(i,k)+pgsacw(i,k)+mnuccc(i,k)+psacwi(i,k);
              float dumt       = t3d(i,k)+dt*t3dten(i,k);
              float dumqv      = qv3d(i,k)+dt*qv3dten(i,k);
              float dum        = min( 0.99*pres(i,k) , polysvp(dumt,0) );
              float dumqss     = ep_2*dum/(pres(i,k)-dum);
              float dumqc      = qc3d(i,k)+dt*qc3dten(i,k);
              dumqc      = max( dumqc , 0. );
              float dums       = dumqv-dumqss;
              pcc(i,k)     = dums/(1.+pow(xxlv(i,k),2)*dumqss/(cpm(i,k)*rv*pow(dumt,2)))/dt;
              if (pcc(i,k)*dt+dumqc < 0.) pcc(i,k) = -dumqc/dt;
              qv3dten(i,k) = qv3dten(i,k)-pcc(i,k);
              t3dten (i,k) = t3dten (i,k)+pcc(i,k)*xxlv(i,k)/cpm(i,k);
              qc3dten(i,k) = qc3dten(i,k)+pcc(i,k);
              if (eprd(i,k) < 0.) {
                dum      = eprd(i,k)*dt/qi3d(i,k);
                dum      = max(-1.,dum);
                nsubi(i,k) = dum*ni3d(i,k)/dt;
              }
              if (eprds(i,k) < 0.) {
                dum      = eprds(i,k)*dt/qni3d(i,k);
                dum      = max(-1.,dum);
                nsubs(i,k) = dum*ns3d(i,k)/dt;
              }
              if (pre(i,k) < 0.) {
                dum      = pre(i,k)*dt/qr3d(i,k);
                dum      = max(-1.,dum);
                nsubr(i,k) = dum*nr3d(i,k)/dt;
              }
              if (eprdg(i,k) < 0.) {
                dum      = eprdg(i,k)*dt/qg3d(i,k);
                dum      = max(-1.,dum);
                nsubg(i,k) = dum*ng3d(i,k)/dt;
              }
              ni3dten(i,k) = ni3dten(i,k)+nsubi(i,k);
              ns3dten(i,k) = ns3dten(i,k)+nsubs(i,k);
              ng3dten(i,k) = ng3dten(i,k)+nsubg(i,k);
              nr3dten(i,k) = nr3dten(i,k)+nsubr(i,k);
            } // temperature
            hydro_pres(i) = true; // No hydrometeors are present. Skip the rest of the routine
          } // if (! skip_micro(i,k))
      });

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<1>(ncol) , KOKKOS_LAMBDA (int i) {
        if (hydro_pres(i)) {
          nstep(i) = 1;
          for (int k = nz; k >= 1; k--) {
            dumi(i,k) = qi3d (i,k)+qi3dten (i,k)*dt;
            dumqs(i,k) = qni3d(i,k)+qni3dten(i,k)*dt;
            dumr(i,k) = qr3d (i,k)+qr3dten (i,k)*dt;
            dumfni(i,k) = ni3d (i,k)+ni3dten (i,k)*dt;
            dumfns(i,k) = ns3d (i,k)+ns3dten (i,k)*dt;
            dumfnr(i,k) = nr3d (i,k)+nr3dten (i,k)*dt;
            dumc(i,k) = qc3d (i,k)+qc3dten (i,k)*dt;
            dumfnc(i,k) = nc3d (i,k)+nc3dten (i,k)*dt;
            dumg(i,k) = qg3d (i,k)+qg3dten (i,k)*dt;
            dumfng(i,k) = ng3d (i,k)+ng3dten (i,k)*dt;
            if (iinum==1) dumfnc(i,k) = nc3d(i,k);
            dumfni(i,k) = max( 0. , dumfni(i,k) );
            dumfns(i,k) = max( 0. , dumfns(i,k) );
            dumfnc(i,k) = max( 0. , dumfnc(i,k) );
            dumfnr(i,k) = max( 0. , dumfnr(i,k) );
            dumfng(i,k) = max( 0. , dumfng(i,k) );
            float dlami;
            if (dumi(i,k) >= qsmall) {
              dlami = pow(cons12*dumfni(i,k)/dumi(i,k),1./di);
              dlami = max( dlami , lammini );
              dlami = min( dlami , lammaxi );
            }
            float dlamr;
            if (dumr(i,k) >= qsmall) {
              dlamr = pow(pi*rhow*dumfnr(i,k)/dumr(i,k),1./3.);
              dlamr = max( dlamr , lamminr );
              dlamr = min( dlamr , lammaxr );
            }
            float dlamc;
            if (dumc(i,k) >= qsmall) {
              float dum     = pres(i,k)/(287.15*t3d(i,k));
              pgam(i,k) = 0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714;
              pgam(i,k) = 1./(pow(pgam(i,k),2))-1.;
              pgam(i,k) = max(pgam(i,k),2.);
              pgam(i,k) = min(pgam(i,k),10.);
              dlamc   = pow(cons26*dumfnc(i,k)*gamma(pgam(i,k)+4.)/(dumc(i,k)*gamma(pgam(i,k)+1.)),1./3.);
              float lammin  = (pgam(i,k)+1.)/60.e-6;
              float lammax  = (pgam(i,k)+1.)/1.e-6;
              dlamc   = max(dlamc,lammin);
              dlamc   = min(dlamc,lammax);
            }
            float dlams;
            if (dumqs(i,k) >= qsmall) {
              dlams = pow(cons1*dumfns(i,k)/ dumqs(i,k),1./ds);
              dlams=max(dlams,lammins);
              dlams=min(dlams,lammaxs);
            }
            float dlamg;
            if (dumg(i,k) >= qsmall) {
              dlamg = pow(cons2*dumfng(i,k)/ dumg(i,k),1./dg);
              dlamg=max(dlamg,lamming);
              dlamg=min(dlamg,lammaxg);
            }
            float unc, umc;
            if (dumc(i,k) >= qsmall) {
              unc =  acn(i,k)*gamma(1.+bc+pgam(i,k))/ (pow(dlamc,bc)*gamma(pgam(i,k)+1.));
              umc = acn(i,k)*gamma(4.+bc+pgam(i,k))/  (pow(dlamc,bc)*gamma(pgam(i,k)+4.));
            } else {
              umc = 0.;
              unc = 0.;
            }
            float uni, umi;
            if (dumi(i,k) >= qsmall) {
              uni = ain(i,k)*cons27/pow(dlami,bi);
              umi = ain(i,k)*cons28/pow(dlami,bi);
            } else {
              umi = 0.;
              uni = 0.;
            }
            float umr, unr;
            if (dumr(i,k) >= qsmall) {
              unr = arn(i,k)*cons6/pow(dlamr,br);
              umr = arn(i,k)*cons4/pow(dlamr,br);
            } else {
              umr = 0.;
              unr = 0.;
            }
            float ums, uns;
            if (dumqs(i,k) >= qsmall) {
              ums = asn(i,k)*cons3/pow(dlams,bs);
              uns = asn(i,k)*cons5/pow(dlams,bs);
            } else {
              ums = 0.;
              uns = 0.;
            }
            float umg, ung;
            if (dumg(i,k) >= qsmall) {
              umg = agn(i,k)*cons7/pow(dlamg,bg);
              ung = agn(i,k)*cons8/pow(dlamg,bg);
            } else {
              umg = 0.;
              ung = 0.;
            }
            float dum    = pow(rhosu/rho(i,k),0.54);
            ums    = min(ums,1.2*dum);
            uns    = min(uns,1.2*dum);
            umi    = min(umi,1.2*pow(rhosu/rho(i,k),0.35));
            uni    = min(uni,1.2*pow(rhosu/rho(i,k),0.35));
            umr    = min(umr,9.1*dum);
            unr    = min(unr,9.1*dum);
            umg    = min(umg,20.*dum);
            ung    = min(ung,20.*dum);
            fr(i,k) = umr;
            fi(i,k) = umi;
            fni(i,k) = uni;
            fs(i,k) = ums;
            fns(i,k) = uns;
            fnr(i,k) = unr;
            fc(i,k) = umc;
            fnc(i,k) = unc;
            fg(i,k) = umg;
            fng(i,k) = ung;
            if (k <= nz-1) {
              if (fr(i,k) < 1.e-10) fr(i,k) = fr (i,k+1);
              if (fi(i,k) < 1.e-10) fi(i,k) = fi (i,k+1);
              if (fni(i,k) < 1.e-10) fni(i,k) = fni(i,k+1);
              if (fs(i,k) < 1.e-10) fs(i,k) = fs (i,k+1);
              if (fns(i,k) < 1.e-10) fns(i,k) = fns(i,k+1);
              if (fnr(i,k) < 1.e-10) fnr(i,k) = fnr(i,k+1);
              if (fc(i,k) < 1.e-10) fc(i,k) = fc (i,k+1);
              if (fnc(i,k) < 1.e-10) fnc(i,k) = fnc(i,k+1);
              if (fg(i,k) < 1.e-10) fg(i,k) = fg (i,k+1);
              if (fng(i,k) < 1.e-10) fng(i,k) = fng(i,k+1);
            } // k le nz-1
            float rgvm = max(fr(i,k),max(fi(i,k),max(fs(i,k),max(fc(i,k),max(fni(i,k),max(fnr(i,k),max(fns(i,k),max(fnc(i,k),max(fg(i,k),fng(i,k))))))))));
            nstep(i) = max(int(rgvm*dt/dzq(i,k)+1.),nstep(i));
            dumr(i,k) = dumr(i,k)*rho(i,k);
            dumi(i,k) = dumi(i,k)*rho(i,k);
            dumfni(i,k) = dumfni(i,k)*rho(i,k);
            dumqs(i,k) = dumqs(i,k)*rho(i,k);
            dumfns(i,k) = dumfns(i,k)*rho(i,k);
            dumfnr(i,k) = dumfnr(i,k)*rho(i,k);
            dumc(i,k) = dumc(i,k)*rho(i,k);
            dumfnc(i,k) = dumfnc(i,k)*rho(i,k);
            dumg(i,k) = dumg(i,k)*rho(i,k);
            dumfng(i,k) = dumfng(i,k)*rho(i,k);
          }

          for (int n = 1; n <= nstep(i); n++) {
            for (int k = 1; k <= nz; k++) {
              faloutr(i,k) = fr(i,k)*dumr(i,k);
              falouti(i,k) = fi(i,k)*dumi(i,k);
              faloutni(i,k) = fni(i,k)*dumfni(i,k);
              falouts(i,k) = fs(i,k)*dumqs(i,k);
              faloutns(i,k) = fns(i,k)*dumfns(i,k);
              faloutnr(i,k) = fnr(i,k)*dumfnr(i,k);
              faloutc(i,k) = fc(i,k)*dumc(i,k);
              faloutnc(i,k) = fnc(i,k)*dumfnc(i,k);
              faloutg(i,k) = fg(i,k)*dumg(i,k);
              faloutng(i,k) = fng(i,k)*dumfng(i,k);
            }
            {
              int k = nz;
              float faltndr  = faloutr(i,k)/dzq(i,k);
              float faltndi  = falouti(i,k)/dzq(i,k);
              float faltndni = faloutni(i,k)/dzq(i,k);
              float faltnds  = falouts(i,k)/dzq(i,k);
              float faltndns = faloutns(i,k)/dzq(i,k);
              float faltndnr = faloutnr(i,k)/dzq(i,k);
              float faltndc  = faloutc(i,k)/dzq(i,k);
              float faltndnc = faloutnc(i,k)/dzq(i,k);
              float faltndg  = faloutg(i,k)/dzq(i,k);
              float faltndng = faloutng(i,k)/dzq(i,k);
              qrsten (i,k) = qrsten (i,k)-faltndr /nstep(i)/rho(i,k);
              qisten (i,k) = qisten (i,k)-faltndi /nstep(i)/rho(i,k);
              ni3dten(i,k) = ni3dten(i,k)-faltndni/nstep(i)/rho(i,k);
              qnisten(i,k) = qnisten(i,k)-faltnds /nstep(i)/rho(i,k);
              ns3dten(i,k) = ns3dten(i,k)-faltndns/nstep(i)/rho(i,k);
              nr3dten(i,k) = nr3dten(i,k)-faltndnr/nstep(i)/rho(i,k);
              qcsten (i,k) = qcsten (i,k)-faltndc /nstep(i)/rho(i,k);
              nc3dten(i,k) = nc3dten(i,k)-faltndnc/nstep(i)/rho(i,k);
              qgsten (i,k) = qgsten (i,k)-faltndg /nstep(i)/rho(i,k);
              ng3dten(i,k) = ng3dten(i,k)-faltndng/nstep(i)/rho(i,k);
              dumr(i,k) = dumr(i,k)-faltndr *dt/nstep(i);
              dumi(i,k) = dumi(i,k)-faltndi *dt/nstep(i);
              dumfni(i,k) = dumfni(i,k)-faltndni*dt/nstep(i);
              dumqs(i,k) = dumqs(i,k)-faltnds *dt/nstep(i);
              dumfns(i,k) = dumfns(i,k)-faltndns*dt/nstep(i);
              dumfnr(i,k) = dumfnr(i,k)-faltndnr*dt/nstep(i);
              dumc(i,k) = dumc(i,k)-faltndc *dt/nstep(i);
              dumfnc(i,k) = dumfnc(i,k)-faltndnc*dt/nstep(i);
              dumg(i,k) = dumg(i,k)-faltndg *dt/nstep(i);
              dumfng(i,k) = dumfng(i,k)-faltndng*dt/nstep(i);
            }
            for (int k = nz-1; k >= 1; k--) {
              float faltndr  = (faloutr (i,k+1)-faloutr(i,k))/dzq(i,k);
              float faltndi  = (falouti (i,k+1)-falouti(i,k))/dzq(i,k);
              float faltndni = (faloutni(i,k+1)-faloutni(i,k))/dzq(i,k);
              float faltnds  = (falouts (i,k+1)-falouts(i,k))/dzq(i,k);
              float faltndns = (faloutns(i,k+1)-faloutns(i,k))/dzq(i,k);
              float faltndnr = (faloutnr(i,k+1)-faloutnr(i,k))/dzq(i,k);
              float faltndc  = (faloutc (i,k+1)-faloutc(i,k))/dzq(i,k);
              float faltndnc = (faloutnc(i,k+1)-faloutnc(i,k))/dzq(i,k);
              float faltndg  = (faloutg (i,k+1)-faloutg(i,k))/dzq(i,k);
              float faltndng = (faloutng(i,k+1)-faloutng(i,k))/dzq(i,k);
              qrsten (i,k) = qrsten (i,k)+faltndr /nstep(i)/rho(i,k);
              qisten (i,k) = qisten (i,k)+faltndi /nstep(i)/rho(i,k);
              ni3dten(i,k) = ni3dten(i,k)+faltndni/nstep(i)/rho(i,k);
              qnisten(i,k) = qnisten(i,k)+faltnds /nstep(i)/rho(i,k);
              ns3dten(i,k) = ns3dten(i,k)+faltndns/nstep(i)/rho(i,k);
              nr3dten(i,k) = nr3dten(i,k)+faltndnr/nstep(i)/rho(i,k);
              qcsten (i,k) = qcsten (i,k)+faltndc /nstep(i)/rho(i,k);
              nc3dten(i,k) = nc3dten(i,k)+faltndnc/nstep(i)/rho(i,k);
              qgsten (i,k) = qgsten (i,k)+faltndg /nstep(i)/rho(i,k);
              ng3dten(i,k) = ng3dten(i,k)+faltndng/nstep(i)/rho(i,k);
              dumr(i,k) = dumr(i,k)+faltndr *dt/nstep(i);
              dumi(i,k) = dumi(i,k)+faltndi *dt/nstep(i);
              dumfni(i,k) = dumfni(i,k)+faltndni*dt/nstep(i);
              dumqs(i,k) = dumqs(i,k)+faltnds *dt/nstep(i);
              dumfns(i,k) = dumfns(i,k)+faltndns*dt/nstep(i);
              dumfnr(i,k) = dumfnr(i,k)+faltndnr*dt/nstep(i);
              dumc(i,k) = dumc(i,k)+faltndc *dt/nstep(i);
              dumfnc(i,k) = dumfnc(i,k)+faltndnc*dt/nstep(i);
              dumg(i,k) = dumg(i,k)+faltndg *dt/nstep(i);
              dumfng(i,k) = dumfng(i,k)+faltndng*dt/nstep(i);
              csed(i,k)=csed(i,k)+faloutc(i,k)/nstep(i);
              ised(i,k)=ised(i,k)+falouti(i,k)/nstep(i);
              ssed(i,k)=ssed(i,k)+falouts(i,k)/nstep(i);
              gsed(i,k)=gsed(i,k)+faloutg(i,k)/nstep(i);
              rsed(i,k)=rsed(i,k)+faloutr(i,k)/nstep(i);
            }
            precrt (i) = precrt (i)+(faloutr(i,1)+faloutc(i,1)+falouts(i,1)+falouti(i,1)+faloutg(i,1))*dt/nstep(i);
            snowrt (i) = snowrt (i)+(falouts(i,1)+falouti(i,1)+faloutg(i,1))*dt/nstep(i);
            snowprt(i) = snowprt(i)+(falouti(i,1)+falouts(i,1))*dt/nstep(i);
            grplprt(i) = grplprt(i)+(faloutg(i,1))*dt/nstep(i);
          } // nstep(i)
        }
      });


      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , KOKKOS_LAMBDA (int k, int i) {
          if (hydro_pres(i)) {
            qr3dten (i,k) = qr3dten (i,k) + qrsten (i,k);
            qi3dten (i,k) = qi3dten (i,k) + qisten (i,k);
            qc3dten (i,k) = qc3dten (i,k) + qcsten (i,k);
            qg3dten (i,k) = qg3dten (i,k) + qgsten (i,k);
            qni3dten(i,k) = qni3dten(i,k) + qnisten(i,k);;
            if (qi3d(i,k) >= qsmall && t3d(i,k) < 273.15 && lami(i,k) >= 1.e-10) {
              if (1./lami(i,k) >= 2.*dcs) {
                qni3dten(i,k) = qni3dten(i,k)+qi3d(i,k)/dt+ qi3dten(i,k);
                ns3dten(i,k) = ns3dten(i,k)+ni3d(i,k)/dt+   ni3dten(i,k);
                qi3dten(i,k) = -qi3d(i,k)/dt;
                ni3dten(i,k) = -ni3d(i,k)/dt;
              }
            }
            qc3d (i,k) = qc3d (i,k)+qc3dten (i,k)*dt;
            qi3d (i,k) = qi3d (i,k)+qi3dten (i,k)*dt;
            qni3d(i,k) = qni3d(i,k)+qni3dten(i,k)*dt;
            qr3d (i,k) = qr3d (i,k)+qr3dten (i,k)*dt;
            nc3d (i,k) = nc3d (i,k)+nc3dten (i,k)*dt;
            ni3d (i,k) = ni3d (i,k)+ni3dten (i,k)*dt;
            ns3d (i,k) = ns3d (i,k)+ns3dten (i,k)*dt;
            nr3d (i,k) = nr3d (i,k)+nr3dten (i,k)*dt;
            if (igraup==0) {
              qg3d(i,k) = qg3d(i,k)+qg3dten(i,k)*dt;
              ng3d(i,k) = ng3d(i,k)+ng3dten(i,k)*dt;
            }
            t3d (i,k) = t3d (i,k)+t3dten (i,k)*dt;
            qv3d(i,k) = qv3d(i,k)+qv3dten(i,k)*dt;
            evs(i,k) = min( 0.99*pres(i,k) , polysvp(t3d(i,k),0) )  ; // pa
            eis(i,k) = min( 0.99*pres(i,k) , polysvp(t3d(i,k),1) ) ;  // pa
            if (eis(i,k) > evs(i,k)) eis(i,k) = evs(i,k);
            qvs(i,k) = ep_2*evs(i,k)/(pres(i,k)-evs(i,k));
            qvi(i,k) = ep_2*eis(i,k)/(pres(i,k)-eis(i,k));
            qvqvs(i,k) = qv3d(i,k)/qvs(i,k);
            qvqvsi(i,k) = qv3d(i,k)/qvi(i,k);
            if (qvqvs(i,k) < 0.9) {
              if (qr3d(i,k) < 1.e-8) {
                qv3d(i,k)=qv3d(i,k)+qr3d(i,k);
                t3d (i,k)=t3d (i,k)-qr3d(i,k)*xxlv(i,k)/cpm(i,k);
                qr3d(i,k)=0.;
              }
              if (qc3d(i,k) < 1.e-8) {
                qv3d(i,k)=qv3d(i,k)+qc3d(i,k);
                t3d (i,k)=t3d (i,k)-qc3d(i,k)*xxlv(i,k)/cpm(i,k);
                qc3d(i,k)=0.;
              }
            }
            if (qvqvsi(i,k) < 0.9) {
              if (qi3d(i,k) < 1.e-8) {
                qv3d(i,k)=qv3d(i,k)+qi3d(i,k);
                t3d (i,k)=t3d (i,k)-qi3d(i,k)*xxls(i,k)/cpm(i,k);
                qi3d(i,k)=0.;
              }
              if (qni3d(i,k) < 1.e-8) {
                qv3d (i,k)=qv3d(i,k)+qni3d(i,k);
                t3d  (i,k)=t3d (i,k)-qni3d(i,k)*xxls(i,k)/cpm(i,k);
                qni3d(i,k)=0.;
              }
              if (qg3d(i,k) < 1.e-8) {
                qv3d(i,k)=qv3d(i,k)+qg3d(i,k);
                t3d (i,k)=t3d (i,k)-qg3d(i,k)*xxls(i,k)/cpm(i,k);
                qg3d(i,k)=0.;
              }
            }
            if (qc3d(i,k) < qsmall) {
              qc3d(i,k) = 0.;
              nc3d(i,k) = 0.;
              effc(i,k) = 0.;
            }
            if (qr3d(i,k) < qsmall) {
              qr3d(i,k) = 0.;
              nr3d(i,k) = 0.;
              effr(i,k) = 0.;
            }
            if (qi3d(i,k) < qsmall) {
              qi3d(i,k) = 0.;
              ni3d(i,k) = 0.;
              effi(i,k) = 0.;
            }
            if (qni3d(i,k) < qsmall) {
              qni3d(i,k) = 0.;
              ns3d (i,k) = 0.;
              effs (i,k) = 0.;
            }
            if (qg3d(i,k) < qsmall) {
              qg3d(i,k) = 0.;
              ng3d(i,k) = 0.;
              effg(i,k) = 0.;
            }
            if ( !  (qc3d(i,k) < qsmall && qi3d(i,k) < qsmall && qni3d(i,k) < qsmall  && qr3d(i,k) < qsmall && qg3d(i,k) < qsmall)) {
              if (qi3d(i,k) >= qsmall && t3d(i,k) >= 273.15) {
                qr3d(i,k) = qr3d(i,k)+qi3d(i,k);
                t3d(i,k) = t3d(i,k)-qi3d(i,k)*xlf(i,k)/cpm(i,k);
                qi3d(i,k) = 0.;
                nr3d(i,k) = nr3d(i,k)+ni3d(i,k);
                ni3d(i,k) = 0.;
              }
              if (iliq != 1) {
                if (t3d(i,k) <= 233.15 && qc3d(i,k) >= qsmall) {
                  qi3d(i,k)=qi3d(i,k)+qc3d(i,k);
                  t3d (i,k)=t3d (i,k)+qc3d(i,k)*xlf(i,k)/cpm(i,k);
                  qc3d(i,k)=0.;
                  ni3d(i,k)=ni3d(i,k)+nc3d(i,k);
                  nc3d(i,k)=0.;
                }
                if (igraup==0) {
                  if (t3d(i,k) <= 233.15 && qr3d(i,k) >= qsmall) {
                     qg3d(i,k) = qg3d(i,k)+qr3d(i,k);
                     t3d (i,k) = t3d (i,k)+qr3d(i,k)*xlf(i,k)/cpm(i,k);
                     qr3d(i,k) = 0.;
                     ng3d(i,k) = ng3d(i,k)+ nr3d(i,k);
                     nr3d(i,k) = 0.;
                  }
                } else if (igraup==1) {
                  if (t3d(i,k) <= 233.15 && qr3d(i,k) >= qsmall) {
                    qni3d(i,k) = qni3d(i,k)+qr3d(i,k);
                    t3d  (i,k) = t3d  (i,k)+qr3d(i,k)*xlf(i,k)/cpm(i,k);
                    qr3d (i,k) = 0.;
                    ns3d (i,k) = ns3d (i,k)+nr3d(i,k);
                    nr3d (i,k) = 0.;
                  }
                }
              }
              ni3d(i,k) = max( 0. , ni3d(i,k) );
              ns3d(i,k) = max( 0. , ns3d(i,k) );
              nc3d(i,k) = max( 0. , nc3d(i,k) );
              nr3d(i,k) = max( 0. , nr3d(i,k) );
              ng3d(i,k) = max( 0. , ng3d(i,k) );
              if (qi3d(i,k) >= qsmall) {
                lami(i,k) = pow(cons12*ni3d(i,k)/qi3d(i,k),1./di);
                if (lami(i,k) < lammini) {
                  lami(i,k) = lammini;
                  n0i(i,k) = pow(lami(i,k),4)*qi3d(i,k)/cons12;
                  ni3d(i,k) = n0i(i,k)/lami(i,k);
                } else if (lami(i,k) > lammaxi) {
                  lami(i,k) = lammaxi;
                  n0i(i,k) = pow(lami(i,k),4)*qi3d(i,k)/cons12;
                  ni3d(i,k) = n0i(i,k)/lami(i,k);
                }
              }
              if (qr3d(i,k) >= qsmall) {
                lamr(i,k) = pow(pi*rhow*nr3d(i,k)/qr3d(i,k),1./3.);
                if (lamr(i,k) < lamminr) {
                  lamr(i,k) = lamminr;
                  n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                  nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                } else if (lamr(i,k) > lammaxr) {
                  lamr(i,k) = lammaxr;
                  n0rr(i,k) = pow(lamr(i,k),4)*qr3d(i,k)/(pi*rhow);
                  nr3d(i,k) = n0rr(i,k)/lamr(i,k);
                }
              }
              if (qc3d(i,k) >= qsmall) {
                float dum = pres(i,k)/(287.15*t3d(i,k));
                pgam(i,k)=0.0005714*(nc3d(i,k)/1.e6*dum)+0.2714;
                pgam(i,k)=1./(pow(pgam(i,k),2))-1.;
                pgam(i,k)=max(pgam(i,k),2.);
                pgam(i,k)=min(pgam(i,k),10.);
                lamc(i,k) = pow(cons26*nc3d(i,k)*gamma(pgam(i,k)+4.)/(qc3d(i,k)*gamma(pgam(i,k)+1.)),1./3.);
                float lammin = (pgam(i,k)+1.)/60.e-6;
                float lammax = (pgam(i,k)+1.)/1.e-6;
                if (lamc(i,k) < lammin) {
                  lamc(i,k) = lammin;
                  nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(gamma(pgam(i,k)+1.))-std::log(gamma(pgam(i,k)+4.)))/cons26;
                } else if (lamc(i,k) > lammax) {
                  lamc(i,k) = lammax;
                  nc3d(i,k) = std::exp(3.*std::log(lamc(i,k))+std::log(qc3d(i,k))+std::log(gamma(pgam(i,k)+1.))-std::log(gamma(pgam(i,k)+4.)))/cons26;
                }
              }
              if (qni3d(i,k) >= qsmall) {
                lams(i,k) = pow(cons1*ns3d(i,k)/qni3d(i,k),1./ds);
                if (lams(i,k) < lammins) {
                  lams(i,k) = lammins;
                  n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1;
                  ns3d(i,k) = n0s(i,k)/lams(i,k);
                } else if (lams(i,k) > lammaxs) {
                  lams(i,k) = lammaxs;
                  n0s(i,k) = pow(lams(i,k),4)*qni3d(i,k)/cons1;
                  ns3d(i,k) = n0s(i,k)/lams(i,k);
                }
              }
              if (qg3d(i,k) >= qsmall) {
                lamg(i,k) = pow(cons2*ng3d(i,k)/qg3d(i,k),1./dg);
                if (lamg(i,k) < lamming) {
                  lamg(i,k) = lamming;
                  n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                  ng3d(i,k) = n0g(i,k)/lamg(i,k);
                } else if (lamg(i,k) > lammaxg) {
                  lamg(i,k) = lammaxg;
                  n0g(i,k) = pow(lamg(i,k),4)*qg3d(i,k)/cons2;
                  ng3d(i,k) = n0g(i,k)/lamg(i,k);
                }
              }
            }
            if (qi3d(i,k) >= qsmall) {
              effi(i,k) = 3./lami(i,k)/2.*1.e6;
            } else {
              effi(i,k) = 25.;
            }
            if (qni3d(i,k) >= qsmall) {
              effs(i,k) = 3./lams(i,k)/2.*1.e6;
            } else {
              effs(i,k) = 25.;
            }
            if (qr3d(i,k) >= qsmall) {
              effr(i,k) = 3./lamr(i,k)/2.*1.e6;
            } else {
              effr(i,k) = 25.;
            }
            if (qc3d(i,k) >= qsmall) {
              effc(i,k) = gamma(pgam(i,k)+4.)/gamma(pgam(i,k)+3.)/lamc(i,k)/2.*1.e6;
            } else {
              effc(i,k) = 25.;
            }
            if (qg3d(i,k) >= qsmall) {
              effg(i,k) = 3./lamg(i,k)/2.*1.e6;
            } else {
              effg(i,k) = 25.;
            }
            ni3d(i,k) = min( ni3d(i,k) , 0.3e6/rho(i,k) );
            if (iinum==0 && iact==2) {
              nc3d(i,k) = min( nc3d(i,k) , (nanew1+nanew2)/rho(i,k) );
            }
            if (iinum==1) { 
              nc3d(i,k) = ndcnst*1.e6/rho(i,k);
            }
          } // If (hydro_pres(i))
      });
    }


    template <class T1, class T2> KOKKOS_INLINE_FUNCTION static float max(T1 a, T2 b) { return a > b ? a : b; }
    template <class T1, class T2> KOKKOS_INLINE_FUNCTION static float min(T1 a, T2 b) { return a < b ? a : b; }
    template <class T1, class T2> KOKKOS_INLINE_FUNCTION static float pow(T1 a, T2 b) { return std::pow(static_cast<float>(a),static_cast<float>(b)); }
    KOKKOS_INLINE_FUNCTION static float gamma(float  z) { return std::tgamma(z); }
    KOKKOS_INLINE_FUNCTION static float gamma(double z) { return std::tgamma(static_cast<float>(z)); }



    KOKKOS_INLINE_FUNCTION static float polysvp( float t , int type) {
      float a0 = 6.11239921;
      float a1 = 0.443987641;
      float a2 = 0.142986287e-1;
      float a3 = 0.264847430e-3;
      float a4 = 0.302950461e-5;
      float a5 = 0.206739458e-7;
      float a6 = 0.640689451e-10;
      float a7 = -0.952447341e-13;
      float a8 = -0.976195544e-15;
      float a0i = 6.11147274;
      float a1i = 0.503160820;
      float a2i = 0.188439774e-1;
      float a3i = 0.420895665e-3;
      float a4i = 0.615021634e-5;
      float a5i = 0.602588177e-7;
      float a6i = 0.385852041e-9;
      float a7i = 0.146898966e-11;
      float a8i = 0.252751365e-14;
      float ret;
      // ice
      if (type==1) {
        if (t >= 195.8) {
          float dt=t-273.15;
          ret = a0i + dt*(a1i+dt*(a2i+dt*(a3i+dt*(a4i+dt*(a5i+dt*(a6i+dt*(a7i+a8i*dt))))))) ;
          ret = ret*100.;
        } else {
          ret = pow(10.,-9.09718*(273.16/t-1.)-3.56654*std::log10(273.16/t)+0.876793*(1.-t/273.16)+std::log10(6.1071))*100.;
        }
      }
      // liquid
      if (type==0) {
        if (t >= 202.0) {
          float dt = t-273.15;
          ret = a0 + dt*(a1+dt*(a2+dt*(a3+dt*(a4+dt*(a5+dt*(a6+dt*(a7+a8*dt)))))));
          ret = ret*100.;
        } else {
          ret = pow(10.,-7.90298*(373.16/t-1.)+5.02808*std::log10(373.16/t)-1.3816e-7*(pow(10.,11.344*(1.-t/373.16))-1.)+
                8.1328e-3*(pow(10.,-3.49149*(373.16/t-1.))-1.)+std::log10(1013.246))*100.;
        }
      }
      return ret;
    }


  };

}


