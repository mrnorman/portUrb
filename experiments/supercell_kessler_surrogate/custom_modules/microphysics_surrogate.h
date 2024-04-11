
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "ponni.h"
#include "ponni_load_h5_weights.h"

namespace custom_modules {

  class Microphysics_Surrogate {
  public:
    int static constexpr num_tracers = 3;

    real R_d    ;
    real cp_d   ;
    real cv_d   ;
    real gamma_d;
    real kappa_d;
    real R_v    ;
    real cp_v   ;
    real cv_v   ;
    real p0     ;
    real grav   ;

    int static constexpr ID_V = 0;  // Local index for water vapor
    int static constexpr ID_C = 1;  // Local index for cloud liquid
    int static constexpr ID_R = 2;  // Local index for precipitated liquid (rain)

    real2d scl_in ;
    real2d scl_out;

    typedef decltype(ponni::create_inference_model(ponni::Matvec<float>(),
                                                   ponni::Bias  <float>(),
                                                   ponni::Relu  <float>(),
                                                   ponni::Matvec<float>(),
                                                   ponni::Bias  <float>())) MODEL;
    MODEL model;



    Microphysics_Surrogate() {
      R_d     = 287.;
      cp_d    = 1003.;
      cv_d    = cp_d - R_d;
      gamma_d = cp_d / cv_d;
      kappa_d = R_d  / cp_d;
      R_v     = 461.;
      cp_v    = 1859;
      cv_v    = R_v - cp_v;
      p0      = 1.e5;
      grav    = 9.81;
    }



    YAKL_INLINE static int get_num_tracers() {
      return num_tracers;
    }



    void init(core::Coupler &coupler) {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nens = coupler.get_nens();
      int nx   = coupler.get_nx();
      int ny   = coupler.get_ny();
      int nz   = coupler.get_nz();

      // Register tracers in the coupler
      //                 name              description       positive   adds mass    diffuse
      coupler.add_tracer("water_vapor"   , "Water Vapor"   , true     , true       , true);
      coupler.add_tracer("cloud_liquid"  , "Cloud liquid"  , true     , true       , true);
      coupler.add_tracer("precip_liquid" , "precip_liquid" , true     , true       , true);

      auto &dm = coupler.get_data_manager_readwrite();

      // Register and allocation non-tracer quantities used by the microphysics
      dm.register_and_allocate<real>( "precl" , "precipitation rate" , {ny,nx,nens} , {"y","x","nens"} );

      // Initialize all micro data to zero
      auto rho_v = dm.get<real,4>("water_vapor"  );
      auto rho_c = dm.get<real,4>("cloud_liquid" );
      auto rho_p = dm.get<real,4>("precip_liquid");
      auto precl = dm.get<real,3>("precl"        );

      parallel_for( YAKL_AUTO_LABEL() , Bounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        rho_v(k,j,i,iens) = 0;
        rho_c(k,j,i,iens) = 0;
        rho_p(k,j,i,iens) = 0;
        if (k == 0) precl(j,i,iens) = 0;
      });

      coupler.set_option<std::string>("micro","kessler");
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

      //////////////////////////////////////////////////
      // Read in the Neural Network surrogate model
      //////////////////////////////////////////////////
      auto inFile = coupler.get_option<std::string>("standalone_input_file");
      YAML::Node config = YAML::LoadFile(inFile);
      auto keras_weights_h5  = config["keras_file"          ].as<std::string>();
      auto nn_input_scaling  = config["inputs_scaling_file" ].as<std::string>();
      auto nn_output_scaling = config["outputs_scaling_file"].as<std::string>();

      // Create the model, and load the weights
      ponni::Matvec<float> matvec_1( ponni::load_h5_weights<2>( keras_weights_h5 , "/dense/dense" , "kernel:0" ) );
      ponni::Bias  <float> bias_1  ( ponni::load_h5_weights<1>( keras_weights_h5 , "/dense/dense" , "bias:0"   ) );
      ponni::Relu  <float> relu_1  ( bias_1.get_num_outputs() , 0.1 );  // LeakyReLU with negative slope of 0.1
      ponni::Matvec<float> matvec_2( ponni::load_h5_weights<2>( keras_weights_h5 , "/dense_1/dense_1" , "kernel:0" ) );
      ponni::Bias  <float> bias_2  ( ponni::load_h5_weights<1>( keras_weights_h5 , "/dense_1/dense_1" , "bias:0"   ) );

      this->model = ponni::create_inference_model(matvec_1, bias_1, relu_1, matvec_2, bias_2);
      model.validate();
      model.print();

      auto num_inputs  = model.get_num_inputs ();
      auto num_outputs = model.get_num_outputs();

      // Load the data scaling arrays
      scl_in  = real2d("scl_in" ,num_inputs ,2);
      scl_out = real2d("scl_out",num_outputs,2);
      auto scl_out_host = scl_out.createHostObject();
      auto scl_in_host  = scl_in .createHostObject();
      std::ifstream file1;
      file1.open( nn_input_scaling );
      for (int j = 0; j < num_inputs; j++) {
        for (int i = 0; i < 2; i++) {
          file1 >> scl_in_host(j,i);
        }
      }
      file1.close();
      file1.open( nn_output_scaling );
      for (int j = 0; j < num_outputs; j++) {
        for (int i = 0; i < 2; i++) {
          file1 >> scl_out_host(j,i);
        }
      }
      file1.close();
      scl_in_host .deep_copy_to(scl_in );
      scl_out_host.deep_copy_to(scl_out);
      yakl::fence();
    }



    void time_step( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;

      // Grab the dimension sizes
      auto dz   = coupler.get_dz();
      auto nz   = coupler.get_nz();
      auto ny   = coupler.get_ny();
      auto nx   = coupler.get_nx();
      auto nx_glob = coupler.get_nx_glob();
      auto ny_glob = coupler.get_ny_glob();
      auto nens = coupler.get_nens();
      auto ncol = ny*nx*nens;
      auto &dm = coupler.get_data_manager_readwrite();

      // Grab the data
      auto rho_v = dm.get_lev_col  <real      >("water_vapor"  );
      auto rho_c = dm.get_lev_col  <real      >("cloud_liquid" );
      auto rho_p = dm.get_lev_col  <real      >("precip_liquid");
      auto rho_d = dm.get_lev_col  <real const>("density_dry"  );
      auto wvel  = dm.get_lev_col  <real const>("wvel"         );
      auto temp  = dm.get_lev_col  <real      >("temp"         );
      auto precl = dm.get_collapsed<real      >("precl"        );

      // Save copies to re-run with NN surrogate
      auto init_rho_v  = rho_v  .createDeviceCopy();
      auto init_rho_c  = rho_c  .createDeviceCopy();
      auto init_rho_p  = rho_p  .createDeviceCopy();
      auto init_rho_d  = rho_d  .createDeviceCopy();
      auto init_temp   = temp   .createDeviceCopy();
      auto init_precl  = precl  .createDeviceCopy();

      ////////////////////////////////////////////////////
      // ORIGINAL CODE
      ////////////////////////////////////////////////////
      // These are inputs to kessler(...)
      real2d qv      ("qv"      ,nz,ncol);
      real2d qc      ("qc"      ,nz,ncol);
      real2d qr      ("qr"      ,nz,ncol);
      real2d pressure("pressure",nz,ncol);
      real2d theta   ("theta"   ,nz,ncol);
      real2d exner   ("exner"   ,nz,ncol);
      real2d zmid    ("zmid"    ,nz,ncol);

      // Force constants into local scope
      real R_d  = this->R_d;
      real R_v  = this->R_v;
      real cp_d = this->cp_d;
      real p0   = this->p0;

      // Save initial state, and compute inputs for kessler(...)
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        zmid    (k,i) = (k+0.5_fp) * dz;
        qv      (k,i) = rho_v(k,i) / rho_d(k,i);
        qc      (k,i) = rho_c(k,i) / rho_d(k,i);
        qr      (k,i) = rho_p(k,i) / rho_d(k,i);
        pressure(k,i) = R_d * rho_d(k,i) * temp(k,i) + R_v * rho_v(k,i) * temp(k,i);
        exner   (k,i) = pow( pressure(k,i) / p0 , R_d / cp_d );
        theta   (k,i) = temp(k,i) / exner(k,i);
      });

      // Call Kessler code
      kessler(theta, qv, qc, qr, rho_d, precl, zmid, exner, dt, R_d, cp_d, p0);

      // Post-process microphysics changes back to the coupler state
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        rho_v   (k,i) = qv(k,i)*rho_d(k,i);
        rho_c   (k,i) = qc(k,i)*rho_d(k,i);
        rho_p   (k,i) = qr(k,i)*rho_d(k,i);
        // While micro changes total pressure, thus changing exner, the definition
        // of theta depends on the old exner pressure, so we'll use old exner here
        temp    (k,i) = theta(k,i) * exner(k,i);
      });

      auto orig_rho_v = rho_v.createDeviceCopy();
      auto orig_rho_c = rho_c.createDeviceCopy();
      auto orig_rho_p = rho_p.createDeviceCopy();
      auto orig_temp  = temp .createDeviceCopy();

      ////////////////////////////////////////////////////
      // NEURAL NETWORK SURROGATE
      ////////////////////////////////////////////////////
      // Restore original initial data
      init_rho_v.deep_copy_to(rho_v);
      init_rho_c.deep_copy_to(rho_c);
      init_rho_p.deep_copy_to(rho_p);
      init_temp .deep_copy_to(temp );

      // inputs_vec .push_back( {rho_d_1_h(i),rho_v_1_h(i),rho_c_1_h(i),rho_p_1_h(i),wvel_1_h(i),temp_1_h(i)} );
      // outputs_vec.push_back( {c_to_v_h(i),c_to_p_h(i),p_adv_net_h(i)} );

      int num_inputs  = model.get_num_inputs ();
      int num_outputs = model.get_num_outputs();
      nens = 1;
      int iens = 0;

      float3d inputs ("inputs" ,num_inputs ,nz*ncol,nens);
      float3d outputs("outputs",num_outputs,nz*ncol,nens);
      // Initialize model for nz*ncol batch size and one model ensemble
      model.init( nz*ncol , 1 );

      YAKL_SCOPE( scl_in  , this->scl_in  );
      YAKL_SCOPE( scl_out , this->scl_out );
      YAKL_SCOPE( model   , this->model   );

      real Lv = 2.5e6_fp;  //  latent heat of vaporization (J/kg)
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        // Collect the inputs for this batch index
        int ibatch = k*ncol+i;
        inputs(0,ibatch,iens) = (rho_d(k,i)-scl_in(0,0))/(scl_in(0,1)-scl_in(0,0));
        inputs(1,ibatch,iens) = (rho_v(k,i)-scl_in(1,0))/(scl_in(1,1)-scl_in(1,0));
        inputs(2,ibatch,iens) = (rho_c(k,i)-scl_in(2,0))/(scl_in(2,1)-scl_in(2,0));
        inputs(3,ibatch,iens) = (rho_p(k,i)-scl_in(3,0))/(scl_in(3,1)-scl_in(3,0));
        inputs(4,ibatch,iens) = (wvel (k,i)-scl_in(4,0))/(scl_in(4,1)-scl_in(4,0));
        inputs(5,ibatch,iens) = (temp (k,i)-scl_in(5,0))/(scl_in(5,1)-scl_in(5,0));
        // Run the Neural Network for this batch index
        model.forward_batch_parallel_in_kernel( inputs , outputs , model.params , ibatch , iens );
        // Process the outputs for this batch index
        real c_to_v    = outputs(0,ibatch,iens)*(scl_out(0,1)-scl_out(0,0)) + scl_out(0,0);
        real c_to_p    = outputs(1,ibatch,iens)*(scl_out(1,1)-scl_out(1,0)) + scl_out(1,0);
        real p_adv_net = outputs(2,ibatch,iens)*(scl_out(2,1)-scl_out(2,0)) + scl_out(2,0);
        // Evaporation and condensation
        c_to_v = std::min( rho_c(k,i),c_to_v);  // Don't evaporate more cloud than we have
        c_to_v = std::max(-rho_v(k,i),c_to_v);  // Don't condense more water vapor than we have
        rho_v(k,i) += c_to_v;
        rho_c(k,i) -= c_to_v;
        // Auto converstion to precipitation
        c_to_p = std::min( rho_c(k,i),c_to_p);  // Don't autoconvert more cloud than we have
        rho_c(k,i) -= c_to_p;
        rho_p(k,i) += c_to_p;
        // Advection of precipitation downward (fallout at terminal velocity)
        rho_p(k,i) += std::max(0._fp,p_adv_net);
        // Ensure non-negativity
        rho_v(k,i) = std::max(0._fp,rho_v(k,i));
        rho_c(k,i) = std::max(0._fp,rho_c(k,i));
        rho_p(k,i) = std::max(0._fp,rho_p(k,i));
        // Change temperature according to evaporation / condensation
        temp(k,i) -= c_to_v*Lv/(rho_d(k,i)*cp_d);
      });

      auto surrogate_rho_v = rho_v.createDeviceCopy();
      auto surrogate_rho_c = rho_c.createDeviceCopy();
      auto surrogate_rho_p = rho_p.createDeviceCopy();
      auto surrogate_temp  = temp .createDeviceCopy();

      using yakl::componentwise::operator-;
      using yakl::componentwise::operator/;
      using yakl::intrinsics::abs;
      using yakl::intrinsics::sum;
      yakl::SArray<real,1,8> sums_loc;
      yakl::SArray<real,1,8> sums_glob;
      sums_loc(0) = sum(abs(surrogate_rho_v - orig_rho_v));
      sums_loc(1) = sum(abs(surrogate_rho_c - orig_rho_c));
      sums_loc(2) = sum(abs(surrogate_rho_p - orig_rho_p));
      sums_loc(3) = sum(abs(surrogate_temp  - orig_temp ));
      sums_loc(4) = sum(abs(orig_rho_v));
      sums_loc(5) = sum(abs(orig_rho_c));
      sums_loc(6) = sum(abs(orig_rho_p));
      sums_loc(7) = sum(abs(orig_temp ));
      MPI_Allreduce(sums_loc.data(),sums_glob.data(),sums_loc.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

      if (coupler.is_mainproc()) {
        std::cout << "rho_v,rho_c,rho_p,temp: " << std::scientific 
                                                << sums_glob(0)/sums_glob(4) << " , "
                                                << sums_glob(1)/sums_glob(5) << " , "
                                                << sums_glob(2)/sums_glob(6) << " , "
                                                << sums_glob(3)/sums_glob(7) << std::endl;
      }

      orig_rho_v.deep_copy_to(rho_v);
      orig_rho_c.deep_copy_to(rho_c);
      orig_rho_p.deep_copy_to(rho_p);
      orig_temp .deep_copy_to(temp );
    }



    ///////////////////////////////////////////////////////////////////////////////
    //
    //  Version:  2.0
    //
    //  Date:  January 22nd, 2015
    //
    //  Change log:
    //  v2 - Added sub-cycling of rain sedimentation so as not to violate
    //       CFL condition.
    //
    //  The KESSLER subroutine implements the Kessler (1969) microphysics
    //  parameterization as described by Soong and Ogura (1973) and Klemp
    //  and Wilhelmson (1978, KW). KESSLER is called at the end of each
    //  time step and makes the final adjustments to the potential
    //  temperature and moisture variables due to microphysical processes
    //  occurring during that time step. KESSLER is called once for each
    //  vertical column of grid cells. Increments are computed and added
    //  into the respective variables. The Kessler scheme contains three
    //  moisture categories: water vapor, cloud water (liquid water that
    //  moves with the flow), and rain water (liquid water that falls
    //  relative to the surrounding air). There  are no ice categories.
    //  
    //  Variables in the column are ordered from the surface to the top.
    //
    //  Parameters:
    //     theta (inout) - dry potential temperature (K)
    //     qv    (inout) - water vapor mixing ratio (gm/gm) (dry mixing ratio)
    //     qc    (inout) - cloud water mixing ratio (gm/gm) (dry mixing ratio)
    //     qr    (inout) - rain  water mixing ratio (gm/gm) (dry mixing ratio)
    //     rho   (in   ) - dry air density (not mean state as in KW) (kg/m^3)
    //     pk    (in   ) - Exner function  (not mean state as in KW) (p/p0)**(R/cp)
    //     dt    (in   ) - time step (s)
    //     z     (in   ) - heights of thermodynamic levels in the grid column (m)
    //     precl (  out) - Precipitation rate (m_water/s)
    //     Rd    (in   ) - Dry air ideal gas constant
    //     cp    (in   ) - Specific heat of dry air at constant pressure
    //     p0    (in   ) - Reference pressure (Pa)
    //
    // Output variables:
    //     Increments are added into t, qv, qc, qr, and precl which are
    //     returned to the routine from which KESSLER was called. To obtain
    //     the total precip qt, after calling the KESSLER routine, compute:
    //
    //       qt = sum over surface grid cells of (precl * cell area)  (kg)
    //       [here, the conversion to kg uses (10^3 kg/m^3)*(10^-3 m/mm) = 1]
    //
    //
    //  Written in Fortran by: Paul Ullrich
    //                         University of California, Davis
    //                         Email: paullrich@ucdavis.edu
    //
    //  Ported to C++ / YAKL by: Matt Norman
    //                           Oak Ridge National Laboratory
    //                           normanmr@ornl.gov
    //                           https://mrnorman.github.io
    //
    //  Based on a code by Joseph Klemp
    //  (National Center for Atmospheric Research)
    //
    //  Reference:
    //
    //    Klemp, J. B., W. C. Skamarock, W. C., and S.-H. Park, 2015:
    //    Idealized Global Nonhydrostatic Atmospheric Test Cases on a Reduced
    //    Radius Sphere. Journal of Advances in Modeling Earth Systems. 
    //    doi:10.1002/2015MS000435
    //
    ///////////////////////////////////////////////////////////////////////////////

    void kessler(real2d const &theta, real2d const &qv, real2d const &qc, real2d const &qr, realConst2d rho,
                 real1d const &precl, realConst2d z, realConst2d pk, real dt, real Rd, real cp, real p0) const {
      using yakl::c::parallel_for;
      using yakl::c::Bounds;

      int nz   = theta.dimension[0];
      int ncol = theta.dimension[1];

      // Maximum time step size in accordance with CFL condition
      if (dt <= 0) { endrun("kessler.f90 called with nonpositive dt"); }

      real psl    = p0 / 100;  //  pressure at sea level (mb)
      real rhoqr  = 1000._fp;  //  density of liquid water (kg/m^3)
      real lv     = 2.5e6_fp;  //  latent heat of vaporization (J/kg)

      real2d r    ("r"    ,nz  ,ncol);
      real2d rhalf("rhalf",nz  ,ncol);
      real2d pc   ("pc"   ,nz  ,ncol);
      real2d velqr("velqr",nz  ,ncol);
      real2d dt2d ("dt2d" ,nz-1,ncol);

      parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
        r    (k,i) = 0.001_fp * rho(k,i);
        rhalf(k,i) = sqrt( rho(0,i) / rho(k,i) );
        pc   (k,i) = 3.8_fp / ( pow( pk(k,i) , cp/Rd ) * psl );
        // Liquid water terminal velocity (m/s) following KW eq. 2.15
        velqr(k,i) = 36.34_fp * pow( qr(k,i)*r(k,i) , 0.1364_fp ) * rhalf(k,i);
        // Compute maximum stable time step for each cell
        if (k < nz-1) {
          if (velqr(k,i) > 1.e-10_fp) {
            dt2d(k,i) = 0.8_fp * (z(k+1,i)-z(k,i))/velqr(k,i);
          } else {
            dt2d(k,i) = dt;
          }
        }
        // Initialize precip rate to zero
        if (k == 0) {
          precl(i) = 0;
        }
      });

      // Reduce down the minimum time step among the cells
      real dt_max = yakl::intrinsics::minval(dt2d);

      // Number of subcycles
      int rainsplit = ceil(dt / dt_max);
      real dt0 = dt / static_cast<real>(rainsplit);

      real2d sed("sed",nz,ncol);

      // Subcycle through rain process
      for (int nt=0; nt < rainsplit; nt++) {

        // Sedimentation term using upstream differencing
        parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
          if (k == 0) {
            // Precipitation rate (m/s)
            precl(i) = precl(i) + rho(0,i) * qr(0,i) * velqr(0,i) / rhoqr;
          }
          if (k == nz-1) {
            sed(nz-1,i) = -dt0*qr(nz-1,i)*velqr(nz-1,i)/(0.5_fp * (z(nz-1,i)-z(nz-2,i)));
          } else {
            sed(k,i) = dt0 * ( r(k+1,i)*qr(k+1,i)*velqr(k+1,i) - 
                               r(k  ,i)*qr(k  ,i)*velqr(k  ,i) ) / ( r(k,i)*(z(k+1,i)-z(k,i)) );
          }
        });

        // Adjustment terms
        parallel_for( YAKL_AUTO_LABEL() , Bounds<2>(nz,ncol) , YAKL_LAMBDA (int k, int i) {
          // Autoconversion and accretion rates following KW eq. 2.13a,b
          real qrprod = qc(k,i) - ( qc(k,i)-dt0*std::max( 0.001_fp * (qc(k,i)-0.001_fp) , 0._fp ) ) /
                                  ( 1 + dt0 * 2.2_fp * pow( qr(k,i) , 0.875_fp ) );
          qc(k,i) = std::max( qc(k,i)-qrprod , 0._fp );
          qr(k,i) = std::max( qr(k,i)+qrprod+sed(k,i) , 0._fp );

          // Saturation vapor mixing ratio (gm/gm) following KW eq. 2.11
          real tmp = pk(k,i)*theta(k,i)-36._fp;
          real qvs = pc(k,i)*exp( 17.27_fp * (pk(k,i)*theta(k,i)-273._fp) / tmp );
          real prod = (qv(k,i)-qvs) / (1._fp + qvs*(4093._fp * lv/cp)/(tmp*tmp));

          // Evaporation rate following KW eq. 2.14a,b
          real tmp1 = dt0*( ( ( 1.6_fp + 124.9_fp * pow( r(k,i)*qr(k,i) , 0.2046_fp ) ) *
                              pow( r(k,i)*qr(k,i) , 0.525_fp ) ) /
                            ( 2550000._fp * pc(k,i) / (3.8_fp * qvs)+540000._fp) ) * 
                          ( std::max(qvs-qv(k,i),0._fp) / (r(k,i)*qvs) );
          real tmp2 = std::max( -prod-qc(k,i) , 0._fp );
          real tmp3 = qr(k,i);
          real ern = std::min( tmp1 , std::min( tmp2 , tmp3 ) );

          // Saturation adjustment following KW eq. 3.10
          theta(k,i)= theta(k,i) + lv / (cp*pk(k,i)) * 
                                   ( std::max( prod , -qc(k,i) ) - ern );
          qv(k,i) = std::max( qv(k,i) - std::max( prod , -qc(k,i) ) + ern , 0._fp );
          qc(k,i) = qc(k,i) + std::max( prod , -qc(k,i) );
          qr(k,i) = qr(k,i) - ern;

          // Recalculate liquid water terminal velocity
          velqr(k,i)  = 36.34_fp * pow( qr(k,i)*r(k,i) , 0.1364_fp ) * rhalf(k,i);
          if (k == 0 && nt == rainsplit-1) {
            precl(i) = precl(i) / static_cast<real>(rainsplit);
          }
        });

      }

    }


    std::string micro_name() const { return "surrogate"; }

  };

}


