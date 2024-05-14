
#pragma once

#include "coupler.h"


extern "C"
void p3_main_fortran(double *qc , double *nc , double *qr , double *nr , double *th_atm , double *qv ,
                     double &dt , double *qi , double *qm , double *ni , double *bm , double *pres ,
                     double *dz , double *nc_nuceat_tend , double *nccn_prescribed , double *ni_activated ,
                     double *inv_qc_relvar , int &it , double *precip_liq_surf , double *precip_ice_surf ,
                     int &its , int &ite , int &kts , int &kte , double *diag_eff_radius_qc ,
                     double *diag_eff_radius_qi , double *rho_qi , bool &do_predict_nc , 
                     bool &do_prescribed_CCN ,double *dpres , double *exner , double *qv2qi_depos_tend ,
                     double *precip_total_tend , double *nevapr , double *qr_evap_tend ,
                     double *precip_liq_flux , double *precip_ice_flux , double *cld_frac_r ,
                     double *cld_frac_l , double *cld_frac_i , double *p3_tend_out , double *mu_c ,
                     double *lamc , double *liq_ice_exchange , double *vap_liq_exchange , 
                     double *vap_ice_exchange , double *qv_prev , double *t_prev , double *col_location ,
                     double *elapsed_s );


extern "C"
void micro_p3_utils_init_fortran(real &cpair , real &rair , real &rh2o , real &rhoh2o , real &mwh2o ,
                                 real &mwdry , real &gravit , real &latvap , real &latice , real &cpliq ,
                                 real &tmelt , real &pi , int &iulog , bool &mainproc );


extern "C"
void p3_init_fortran(char const *lookup_file_dir , int &dir_len , char const *version_p3 , int &ver_len );


namespace modules {

  class Microphysics_P3 {
  public:
    // Doesn't actually have to be static or constexpr. Could be assigned in the constructor
    int static constexpr num_tracers = 9;

    // You should set these in the constructor
    real R_d    ;
    real cp_d   ;
    real cv_d   ;
    real gamma_d;
    real kappa_d;
    real R_v    ;
    real cp_v   ;
    real cv_v   ;
    real p0     ;

    real grav;
    real cp_l;

    bool first_step;

    // Indices for all of your tracer quantities
    int static constexpr ID_C  = 0;  // Local index for Cloud Water Mass  
    int static constexpr ID_NC = 1;  // Local index for Cloud Water Number
    int static constexpr ID_R  = 2;  // Local index for Rain Water Mass   
    int static constexpr ID_NR = 3;  // Local index for Rain Water Number 
    int static constexpr ID_I  = 4;  // Local index for Ice Mass          
    int static constexpr ID_M  = 5;  // Local index for Ice Number        
    int static constexpr ID_NI = 6;  // Local index for Ice-Rime Mass     
    int static constexpr ID_BM = 7;  // Local index for Ice-Rime Volume   
    int static constexpr ID_V  = 8;  // Local index for Water Vapor       



    // Set constants and likely num_tracers as well, and anything else you can do immediately
    Microphysics_P3();


    // This must return the correct # of tracers **BEFORE** init(...) is called
    YAKL_INLINE static int get_num_tracers() {
      return num_tracers;
    }



    // Can do whatever you want, but mainly for registering tracers and allocating data
    void init(core::Coupler &coupler);


    void time_step( core::Coupler &coupler , real dt );


    // Returns saturation vapor pressure
    YAKL_INLINE static real saturation_vapor_pressure(real temp) {
      real tc = temp - 273.15;
      return 610.94 * exp( 17.625*tc / (243.04+tc) );
    }


    YAKL_INLINE static real latent_heat_condensation(real temp) {
      real tc = temp - 273.15;
      return (2500.8 - 2.36*tc + 0.0016*tc*tc - 0.00006*tc*tc*tc)*1000;
    }


    YAKL_INLINE static real cp_moist(real rho_d, real rho_v, real rho_c, real cp_d, real cp_v, real cp_l) {
      // For the moist specific heat, ignore other species than water vapor and cloud droplets
      real rho = rho_d + rho_v + rho_c;
      return rho_d / rho * cp_d  +  rho_v / rho * cp_v  +  rho_c / rho * cp_l;
    }


    // Compute an instantaneous adjustment of sub or super saturation
    YAKL_INLINE static void compute_adjusted_state(real rho, real rho_d , real &rho_v , real &rho_c , real &temp,
                                                   real R_v , real cp_d , real cp_v , real cp_l) {
      // Define a tolerance for convergence
      real tol = 1.e-6;

      // Saturation vapor pressure at this temperature
      real svp = saturation_vapor_pressure( temp );

      // Vapor pressure at this temperature
      real pv = rho_v * R_v * temp;

      // If we're super-saturated, we need to condense until saturation is reached
      if        (pv > svp) {
        ////////////////////////////////////////////////////////
        // Bisection method
        ////////////////////////////////////////////////////////
        // Set bounds on how much mass to condense
        real cond1  = 0;     // Minimum amount we can condense out
        real cond2 = rho_v;  // Maximum amount we can condense out

        bool keep_iterating = true;
        while (keep_iterating) {
          real rho_cond = (cond1 + cond2) / 2;                    // How much water vapor to condense for this iteration
          real rv_loc = std::max( 0._fp , rho_v - rho_cond );          // New vapor density
          real rc_loc = std::max( 0._fp , rho_c + rho_cond );          // New cloud liquid density
          real Lv = latent_heat_condensation(temp);               // Compute latent heat of condensation
          real cp = cp_moist(rho_d,rv_loc,rc_loc,cp_d,cp_v,cp_l); // New moist specific heat at constant pressure
          real temp_loc = temp + rho_cond*Lv/(rho*cp);            // New temperature after condensation
          real svp_loc = saturation_vapor_pressure(temp_loc);     // New saturation vapor pressure after condensation
          real pv_loc = rv_loc * R_v * temp_loc;                  // New vapor pressure after condensation
          // If we're supersaturated still, we need to condense out more water vapor
          // otherwise, we need to condense out less water vapor
          if (pv_loc > svp_loc) {
            cond1 = rho_cond;
          } else {
            cond2 = rho_cond;
          }
          // If we've converged, then we can stop iterating
          if (abs(cond2-cond1) <= tol) {
            rho_v = rv_loc;
            rho_c = rc_loc;
            temp  = temp_loc;
            keep_iterating = false;
          }
        }

      // If we are unsaturated and have cloud liquid
      } else if (pv < svp && rho_c > 0) {
        // If there's cloud, evaporate enough to achieve saturation
        // or all of it if there isn't enough to reach saturation
        ////////////////////////////////////////////////////////
        // Bisection method
        ////////////////////////////////////////////////////////
        // Set bounds on how much mass to evaporate
        real evap1 = 0;     // minimum amount we can evaporate
        real evap2 = rho_c; // maximum amount we can evaporate

        bool keep_iterating = true;
        while (keep_iterating) {
          real rho_evap = (evap1 + evap2) / 2;                    // How much water vapor to evapense
          real rv_loc = std::max( 0._fp , rho_v + rho_evap );          // New vapor density
          real rc_loc = std::max( 0._fp , rho_c - rho_evap );          // New cloud liquid density
          real Lv = latent_heat_condensation(temp);               // Compute latent heat of condensation for water
          real cp = cp_moist(rho_d,rv_loc,rc_loc,cp_d,cp_v,cp_l); // New moist specific heat
          real temp_loc = temp - rho_evap*Lv/(rho*cp);            // New temperature after evaporation
          real svp_loc = saturation_vapor_pressure(temp_loc);     // New saturation vapor pressure after evaporation
          real pv_loc = rv_loc * R_v * temp_loc;                  // New vapor pressure after evaporation
          // If we're unsaturated still, we need to evaporate out more water vapor
          // otherwise, we need to evaporate out less water vapor
          if (pv_loc < svp_loc) {
            evap1 = rho_evap;
          } else {
            evap2 = rho_evap;
          }
          // If we've converged, then we can stop iterating
          if (abs(evap2-evap1) <= tol) {
            rho_v = rv_loc;
            rho_c = rc_loc;
            temp  = temp_loc;
            keep_iterating = false;
          }
        }
      }
    }


    std::string micro_name() const;
  };

}


