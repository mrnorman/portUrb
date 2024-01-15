
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
#include "PPM_Limiter.h"
#include "MinmodLimiter.h"
#include <random>
#include <sstream>

namespace custom_modules {

  struct CompareLES {
    int  static constexpr ord = 5;
    int  static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")
    int  static constexpr num_state = 5;   // Number of state variables
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature


    void operator() ( core::Coupler & coupler ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      using yakl::intrinsics::matmul_cr;
      using yakl::intrinsics::sum;
      using yakl::componentwise::operator-;
      using yakl::componentwise::operator*;
      auto nens              = coupler.get_nens();
      auto nx                = coupler.get_nx();    // Proces-local number of cells
      auto ny                = coupler.get_ny();    // Proces-local number of cells
      auto nz                = coupler.get_nz();    // Total vertical cells
      auto nx_glob           = coupler.get_nx_glob();
      auto ny_glob           = coupler.get_ny_glob();
      auto nproc_x           = coupler.get_nproc_x();
      auto nproc_y           = coupler.get_nproc_y();
      auto i_beg             = coupler.get_i_beg  ();
      auto j_beg             = coupler.get_j_beg  ();
      auto px                = coupler.get_px();
      auto py                = coupler.get_py();
      auto dx                = coupler.get_dx();    // grid spacing
      auto dy                = coupler.get_dy();    // grid spacing
      auto dz                = coupler.get_dz();    // grid spacing
      auto sim2d             = coupler.is_sim2d();  // Is this a 2-D simulation?
      auto C0                = coupler.get_option<real>("C0"     );  // pressure = C0*pow(rho*theta,gamma)
      auto grav              = coupler.get_option<real>("grav"   );  // Gravity
      auto gamma             = coupler.get_option<real>("gamma_d");  // cp_dry / cv_dry (about 1.4)
      auto num_tracers       = coupler.get_num_tracers();            // Number of tracers
      auto &dm               = coupler.get_data_manager_readonly();  // Grab read-only data manager
      auto tracer_positive   = dm.get<bool const,1>("tracer_positive"  ); // Is tracer positive-definite?
      auto hy_dens_cells     = dm.get<real const,2>("hy_dens_cells"    ); // Hydrostatic density
      auto hy_theta_cells    = dm.get<real const,2>("hy_theta_cells"   ); // Hydrostatic potential temp
      auto hy_pressure_cells = dm.get<real const,2>("hy_pressure_cells"); // Hydrostatic pressure
      auto tracer_names      = coupler.get_tracer_names();
      int idTKE = -1;
      for (int i=0; i < num_tracers; i++) { if (tracer_names[i] == "TKE") { idTKE = i; break; } }
      real5d state  ("state"  ,num_state  ,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real5d tracers("tracers",num_tracers,nz+2*hs,ny+2*hs,nx+2*hs,nens);
      convert_coupler_to_dynamics( coupler , state , tracers );
      auto tke = tracers.slice<4>(idTKE,0,0,0,0);
      real delta = std::pow( dx*dy*dz , 1._fp/3._fp );
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      SArray<real,2,ord,ord> s2c, c2g, c2d, s2g, s2d2g, g2c, g2d2g;
      TransformMatrices::sten_to_coefs (s2c);
      TransformMatrices::coefs_to_gll  (c2g);
      TransformMatrices::gll_to_coefs  (g2c);
      TransformMatrices::coefs_to_deriv(c2d);
      s2g   = matmul_cr(c2g,s2c);
      s2d2g = matmul_cr(c2g,matmul_cr(c2d,s2c));
      g2d2g = matmul_cr(c2g,matmul_cr(c2d,g2c));
      real r_dx = 1./dx; // reciprocal of grid spacing
      real r_dy = 1./dy; // reciprocal of grid spacing
      real r_dz = 1./dz; // reciprocal of grid spacing
      real4d pressure("pressure",nz+2*hs,ny+2*hs,nx+2*hs,nens); // Holds pressure perturbation

      // Compute pressure perturbation. Divide rho*u, rho*v, rho*w, rho*theta, rho*tracer by density to get
      //   velocities, theta, and tracer concentrations to make boundary conditions easier to implement.
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        pressure(hs+k,hs+j,hs+i,iens) = C0*std::pow(state(idT,hs+k,hs+j,hs+i,iens),gamma) - hy_pressure_cells(hs+k,iens);
        real rdens = 1._fp / state(idR,hs+k,hs+j,hs+i,iens);
        state(idU,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idV,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idW,hs+k,hs+j,hs+i,iens) *= rdens;
        state(idT,hs+k,hs+j,hs+i,iens) *= rdens;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) *= rdens; }
      });

      // Perform periodic halo exchange in the horizontal, and implement vertical no-slip solid wall boundary conditions
      {
        core::MultiField<real,4> fields;
        for (int l=0; l < num_state  ; l++) { fields.add_field( state  .slice<4>(l,0,0,0,0) ); }
        for (int l=0; l < num_tracers; l++) { fields.add_field( tracers.slice<4>(l,0,0,0,0) ); }
        fields.add_field( pressure );
        if (ord > 1) halo_exchange( coupler , fields );
      }

      halo_boundary_conditions( coupler , state , tracers , pressure );

      real4d uu("uu",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d uv("uv",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d uw("uw",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d ut("ut",nz+2*hs,ny+2*hs,nx+2*hs,nens);

      real4d vu("vu",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d vv("vv",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d vw("vw",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d vt("vt",nz+2*hs,ny+2*hs,nx+2*hs,nens);

      real4d wu("wu",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d wv("wv",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d ww("ww",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d wt("wt",nz+2*hs,ny+2*hs,nx+2*hs,nens);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil, gll;
        SArray<real,1,ord> gll_u, gll_v, gll_w, gll_t;
        //////////////////
        // x-direction
        //////////////////
        {
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idU,hs+k,hs+j,i+ii,iens); } // u-velocity
          reconstruct_gll_values( stencil , gll , s2g );
          for (int ii=0; ii < ord; ii++) { gll_u(ii) = gll(ii); }
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idV,hs+k,hs+j,i+ii,iens); } // v-velocity
          reconstruct_gll_values( stencil , gll , s2g );
          for (int ii=0; ii < ord; ii++) { gll_v(ii) = gll(ii); }
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idW,hs+k,hs+j,i+ii,iens); } // w-velocity
          reconstruct_gll_values( stencil , gll , s2g );
          for (int ii=0; ii < ord; ii++) { gll_w(ii) = gll(ii); }
          for (int ii=0; ii < ord; ii++) { stencil(ii) = state(idT,hs+k,hs+j,i+ii,iens); } // theta
          reconstruct_gll_values( stencil , gll , s2g );
          for (int ii=0; ii < ord; ii++) { gll_t(ii) = gll(ii); }
          gll_u = gll_u - sum(gll_u)/gll_u.size();
          gll_v = gll_v - sum(gll_v)/gll_v.size();
          gll_w = gll_w - sum(gll_w)/gll_w.size();
          gll_t = gll_t - sum(gll_t)/gll_t.size();
          auto gll_uu = gll_u*gll_u;
          auto gll_uv = gll_u*gll_v;
          auto gll_uw = gll_u*gll_w;
          auto gll_ut = gll_u*gll_t;
          uu(hs+k,hs+j,hs+i,iens) = sum(gll_uu)/gll_uu.size();
          uv(hs+k,hs+j,hs+i,iens) = sum(gll_uv)/gll_uv.size();
          uw(hs+k,hs+j,hs+i,iens) = sum(gll_uw)/gll_uw.size();
          ut(hs+k,hs+j,hs+i,iens) = sum(gll_ut)/gll_ut.size();
        }

        //////////////////
        // y-direction
        //////////////////
        {
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idU,hs+k,j+jj,hs+i,iens); } // u-velocity
          reconstruct_gll_values( stencil , gll , s2g );
          for (int jj=0; jj < ord; jj++) { gll_u(jj) = gll(jj); }
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idV,hs+k,j+jj,hs+i,iens); } // v-velocity
          reconstruct_gll_values( stencil , gll , s2g );
          for (int jj=0; jj < ord; jj++) { gll_v(jj) = gll(jj); }
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idW,hs+k,j+jj,hs+i,iens); } // w-velocity
          reconstruct_gll_values( stencil , gll , s2g );
          for (int jj=0; jj < ord; jj++) { gll_w(jj) = gll(jj); }
          for (int jj=0; jj < ord; jj++) { stencil(jj) = state(idT,hs+k,j+jj,hs+i,iens); } // theta
          reconstruct_gll_values( stencil , gll , s2g );
          for (int jj=0; jj < ord; jj++) { gll_t(jj) = gll(jj); }
          gll_u = gll_u - sum(gll_u)/gll_u.size();
          gll_v = gll_v - sum(gll_v)/gll_v.size();
          gll_w = gll_w - sum(gll_w)/gll_w.size();
          gll_t = gll_t - sum(gll_t)/gll_t.size();
          auto gll_vu = gll_v*gll_u;
          auto gll_vv = gll_v*gll_v;
          auto gll_vw = gll_v*gll_w;
          auto gll_vt = gll_v*gll_t;
          vu(hs+k,hs+j,hs+i,iens) = sum(gll_vu)/gll_vu.size();
          vv(hs+k,hs+j,hs+i,iens) = sum(gll_vv)/gll_vv.size();
          vw(hs+k,hs+j,hs+i,iens) = sum(gll_vw)/gll_vw.size();
          vt(hs+k,hs+j,hs+i,iens) = sum(gll_vt)/gll_vt.size();
        }

        //////////////////
        // z-direction
        //////////////////
        {
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idU,k+kk,hs+j,hs+i,iens); } // u-velocity
          reconstruct_gll_values( stencil , gll , s2g );
          for (int kk=0; kk < ord; kk++) { gll_u(kk) = gll(kk); }
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idV,k+kk,hs+j,hs+i,iens); } // v-velocity
          reconstruct_gll_values( stencil , gll , s2g );
          for (int kk=0; kk < ord; kk++) { gll_v(kk) = gll(kk); }
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idW,k+kk,hs+j,hs+i,iens); } // w-velocity
          reconstruct_gll_values( stencil , gll , s2g );
          for (int kk=0; kk < ord; kk++) { gll_w(kk) = gll(kk); }
          for (int kk=0; kk < ord; kk++) { stencil(kk) = state(idT,k+kk,hs+j,hs+i,iens); } // theta
          reconstruct_gll_values( stencil , gll , s2g );
          for (int kk=0; kk < ord; kk++) { gll_t(kk) = gll(kk); }
          gll_u = gll_u - sum(gll_u)/gll_u.size();
          gll_v = gll_v - sum(gll_v)/gll_v.size();
          gll_w = gll_w - sum(gll_w)/gll_w.size();
          gll_t = gll_t - sum(gll_t)/gll_t.size();
          auto gll_wu = gll_w*gll_u;
          auto gll_wv = gll_w*gll_v;
          auto gll_ww = gll_w*gll_w;
          auto gll_wt = gll_w*gll_t;
          wu(hs+k,hs+j,hs+i,iens) = sum(gll_wu)/gll_wu.size();
          wv(hs+k,hs+j,hs+i,iens) = sum(gll_wv)/gll_wv.size();
          ww(hs+k,hs+j,hs+i,iens) = sum(gll_ww)/gll_ww.size();
          wt(hs+k,hs+j,hs+i,iens) = sum(gll_wt)/gll_wt.size();
        }
      });

      {
        core::MultiField<real,4> fields;
        fields.add_field(uu);
        fields.add_field(uv);
        fields.add_field(uw);
        fields.add_field(ut);
        fields.add_field(vu);
        fields.add_field(vv);
        fields.add_field(vw);
        fields.add_field(vt);
        fields.add_field(wu);
        fields.add_field(wv);
        fields.add_field(ww);
        fields.add_field(wt);
        if (ord > 1) halo_exchange( coupler , fields );
      }

      halo_boundary_conditions( coupler , uu , uv , uw , ut , vu , vv , vw , vt , wu , wv , ww , wt );

      real4d explicit_u_x("explicit_u_x",nz,ny,nx+1,nens);
      real4d explicit_v_x("explicit_v_x",nz,ny,nx+1,nens);
      real4d explicit_w_x("explicit_w_x",nz,ny,nx+1,nens);
      real4d explicit_t_x("explicit_t_x",nz,ny,nx+1,nens);
      real4d explicit_u_y("explicit_u_y",nz,ny+1,nx,nens);
      real4d explicit_v_y("explicit_v_y",nz,ny+1,nx,nens);
      real4d explicit_w_y("explicit_w_y",nz,ny+1,nx,nens);
      real4d explicit_t_y("explicit_t_y",nz,ny+1,nx,nens);
      real4d explicit_u_z("explicit_u_z",nz+1,ny,nx,nens);
      real4d explicit_v_z("explicit_v_z",nz+1,ny,nx,nens);
      real4d explicit_w_z("explicit_w_z",nz+1,ny,nx,nens);
      real4d explicit_t_z("explicit_t_z",nz+1,ny,nx,nens);
      real4d closure_u_x ("closure_u_x" ,nz,ny,nx+1,nens);
      real4d closure_v_x ("closure_v_x" ,nz,ny,nx+1,nens);
      real4d closure_w_x ("closure_w_x" ,nz,ny,nx+1,nens);
      real4d closure_t_x ("closure_t_x" ,nz,ny,nx+1,nens);
      real4d closure_u_y ("closure_u_y" ,nz,ny+1,nx,nens);
      real4d closure_v_y ("closure_v_y" ,nz,ny+1,nx,nens);
      real4d closure_w_y ("closure_w_y" ,nz,ny+1,nx,nens);
      real4d closure_t_y ("closure_t_y" ,nz,ny+1,nx,nens);
      real4d closure_u_z ("closure_u_z" ,nz+1,ny,nx,nens);
      real4d closure_v_z ("closure_v_z" ,nz+1,ny,nx,nens);
      real4d closure_w_z ("closure_w_z" ,nz+1,ny,nx,nens);
      real4d closure_t_z ("closure_t_z" ,nz+1,ny,nx,nens);
      real4d tke_x       ("tke_x"       ,nz,ny,nx+1,nens);
      real4d tke_y       ("tke_y"       ,nz,ny+1,nx,nens);
      real4d tke_z       ("tke_z"       ,nz+1,ny,nx,nens);
      real4d density_x   ("density_x"   ,nz,ny,nx+1,nens);
      real4d density_y   ("density_y"   ,nz,ny+1,nx,nens);
      real4d density_z   ("density_z"   ,nz+1,ny,nx,nens);
      real4d uvel_x      ("uvel_x"      ,nz,ny,nx+1,nens);
      real4d uvel_y      ("uvel_y"      ,nz,ny+1,nx,nens);
      real4d uvel_z      ("uvel_z"      ,nz+1,ny,nx,nens);
      real4d vvel_x      ("vvel_x"      ,nz,ny,nx+1,nens);
      real4d vvel_y      ("vvel_y"      ,nz,ny+1,nx,nens);
      real4d vvel_z      ("vvel_z"      ,nz+1,ny,nx,nens);
      real4d wvel_x      ("wvel_x"      ,nz,ny,nx+1,nens);
      real4d wvel_y      ("wvel_y"      ,nz,ny+1,nx,nens);
      real4d wvel_z      ("wvel_z"      ,nz+1,ny,nx,nens);
      real4d theta_x     ("theta_x"     ,nz,ny,nx+1,nens);
      real4d theta_y     ("theta_y"     ,nz,ny+1,nx,nens);
      real4d theta_z     ("theta_z"     ,nz+1,ny,nx,nens);

      // TODO: Reconstruct fluxes of sum of products and density to compute fluxes
      //       Assume w' is zero at the boundary, so zero flux for terms involving w'
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz+1,ny+1,nx+1,nens) ,
                                        YAKL_LAMBDA (int k, int j, int i, int iens) {
        if (j < ny && k < nz) {
          real rho   = 0.5_fp * ( state(idR,hs+k,hs+j,hs+i-1,iens) + state(idR,hs+k,hs+j,hs+i,iens) );
          explicit_u_x(k,j,i,iens) = rho*(uu(hs+k,hs+j,hs+i-1,iens)+uu(hs+k,hs+j,hs+i,iens))/2;
          explicit_v_x(k,j,i,iens) = rho*(uv(hs+k,hs+j,hs+i-1,iens)+uv(hs+k,hs+j,hs+i,iens))/2;
          explicit_w_x(k,j,i,iens) = rho*(uw(hs+k,hs+j,hs+i-1,iens)+uw(hs+k,hs+j,hs+i,iens))/2;
          explicit_t_x(k,j,i,iens) = rho*(ut(hs+k,hs+j,hs+i-1,iens)+ut(hs+k,hs+j,hs+i,iens))/2;

          tke_x    (k,j,i,iens) = (tke  (    hs+k,hs+j,hs+i-1,iens)+tke  (    hs+k,hs+j,hs+i,iens))/2;
          density_x(k,j,i,iens) = (state(idR,hs+k,hs+j,hs+i-1,iens)+state(idR,hs+k,hs+j,hs+i,iens))/2;
          uvel_x   (k,j,i,iens) = (state(idU,hs+k,hs+j,hs+i-1,iens)+state(idU,hs+k,hs+j,hs+i,iens))/2;
          uvel_x   (k,j,i,iens) = (state(idV,hs+k,hs+j,hs+i-1,iens)+state(idV,hs+k,hs+j,hs+i,iens))/2;
          wvel_x   (k,j,i,iens) = (state(idW,hs+k,hs+j,hs+i-1,iens)+state(idW,hs+k,hs+j,hs+i,iens))/2;
          theta_x  (k,j,i,iens) = (state(idT,hs+k,hs+j,hs+i-1,iens)+state(idT,hs+k,hs+j,hs+i,iens))/2;

          real K     = 0.5_fp * ( tke      (hs+k,hs+j,hs+i-1,iens) + tke      (hs+k,hs+j,hs+i,iens) );
          real t     = 0.5_fp * ( state(idT,hs+k,hs+j,hs+i-1,iens) + state(idT,hs+k,hs+j,hs+i,iens) );
          real dt_dz = 0.5_fp * ( (state(idT,hs+k+1,hs+j,hs+i-1,iens)-state(idT,hs+k-1,hs+j,hs+i-1,iens))/(2*dz) +
                                  (state(idT,hs+k+1,hs+j,hs+i  ,iens)-state(idT,hs+k-1,hs+j,hs+i  ,iens))/(2*dz) );
          real du_dz = 0.5_fp * ( (state(idU,hs+k+1,hs+j,hs+i-1,iens)-state(idU,hs+k-1,hs+j,hs+i-1,iens))/(2*dz) +
                                  (state(idU,hs+k+1,hs+j,hs+i  ,iens)-state(idU,hs+k-1,hs+j,hs+i  ,iens))/(2*dz) );
          real du_dy = 0.5_fp * ( (state(idU,hs+k,hs+j+1,hs+i-1,iens)-state(idU,hs+k,hs+j-1,hs+i-1,iens))/(2*dy) +
                                  (state(idU,hs+k,hs+j+1,hs+i  ,iens)-state(idU,hs+k,hs+j-1,hs+i  ,iens))/(2*dy) );
          real N     = dt_dz >= 0 ? std::sqrt(grav/t*dt_dz) : 0;
          real ell   = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20_fp) , delta );
          real km    = 0.1_fp * ell * std::sqrt(K);
          real Pr    = delta / (1+2*ell);
          real du_dx = (state(idU,hs+k,hs+j,hs+i,iens) - state(idU,hs+k,hs+j,hs+i-1,iens))/dx;
          real dv_dx = (state(idV,hs+k,hs+j,hs+i,iens) - state(idV,hs+k,hs+j,hs+i-1,iens))/dx;
          real dw_dx = (state(idW,hs+k,hs+j,hs+i,iens) - state(idW,hs+k,hs+j,hs+i-1,iens))/dx;
          real dt_dx = (state(idT,hs+k,hs+j,hs+i,iens) - state(idT,hs+k,hs+j,hs+i-1,iens))/dx;
          closure_u_x(k,j,i,iens) = -rho*km   *(du_dx + du_dx);
          closure_v_x(k,j,i,iens) = -rho*km   *(dv_dx + du_dy);
          closure_w_x(k,j,i,iens) = -rho*km   *(dw_dx + du_dz);
          closure_t_x(k,j,i,iens) = -rho*km/Pr*dt_dx;
        }
        if (i < nx && k < nz) {
          real rho   = 0.5_fp * ( state(idR,hs+k,hs+j-1,hs+i,iens) + state(idR,hs+k,hs+j,hs+i,iens) );
          explicit_u_y(k,j,i,iens) = rho*(vu(hs+k,hs+j-1,hs+i,iens)+vu(hs+k,hs+j,hs+i,iens))/2;
          explicit_v_y(k,j,i,iens) = rho*(vv(hs+k,hs+j-1,hs+i,iens)+vv(hs+k,hs+j,hs+i,iens))/2;
          explicit_w_y(k,j,i,iens) = rho*(vw(hs+k,hs+j-1,hs+i,iens)+vw(hs+k,hs+j,hs+i,iens))/2;
          explicit_t_y(k,j,i,iens) = rho*(vt(hs+k,hs+j-1,hs+i,iens)+vt(hs+k,hs+j,hs+i,iens))/2;

          tke_y    (k,j,i,iens) = (tke  (    hs+k,hs+j-1,hs+i,iens)+tke  (    hs+k,hs+j,hs+i,iens))/2;
          density_y(k,j,i,iens) = (state(idR,hs+k,hs+j-1,hs+i,iens)+state(idR,hs+k,hs+j,hs+i,iens))/2;
          uvel_y   (k,j,i,iens) = (state(idU,hs+k,hs+j-1,hs+i,iens)+state(idU,hs+k,hs+j,hs+i,iens))/2;
          uvel_y   (k,j,i,iens) = (state(idV,hs+k,hs+j-1,hs+i,iens)+state(idV,hs+k,hs+j,hs+i,iens))/2;
          wvel_y   (k,j,i,iens) = (state(idW,hs+k,hs+j-1,hs+i,iens)+state(idW,hs+k,hs+j,hs+i,iens))/2;
          theta_y  (k,j,i,iens) = (state(idT,hs+k,hs+j-1,hs+i,iens)+state(idT,hs+k,hs+j,hs+i,iens))/2;

          real K     = 0.5_fp * ( tke      (hs+k,hs+j-1,hs+i,iens) + tke      (hs+k,hs+j,hs+i,iens) );
          real t     = 0.5_fp * ( state(idT,hs+k,hs+j-1,hs+i,iens) + state(idT,hs+k,hs+j,hs+i,iens) );
          real dt_dz = 0.5_fp * ( (state(idT,hs+k+1,hs+j-1,hs+i,iens)-state(idT,hs+k-1,hs+j-1,hs+i,iens))/(2*dz) +
                                  (state(idT,hs+k+1,hs+j  ,hs+i,iens)-state(idT,hs+k-1,hs+j  ,hs+i,iens))/(2*dz) );
          real dv_dz = 0.5_fp * ( (state(idV,hs+k+1,hs+j-1,hs+i,iens)-state(idV,hs+k-1,hs+j-1,hs+i,iens))/(2*dz) +
                                  (state(idV,hs+k+1,hs+j  ,hs+i,iens)-state(idV,hs+k-1,hs+j  ,hs+i,iens))/(2*dz) );
          real dv_dx = 0.5_fp * ( (state(idV,hs+k,hs+j-1,hs+i+1,iens) - state(idV,hs+k,hs+j-1,hs+i-1,iens))/(2*dx) +
                                  (state(idV,hs+k,hs+j  ,hs+i+1,iens) - state(idV,hs+k,hs+j  ,hs+i-1,iens))/(2*dx) );
          real N     = dt_dz >= 0 ? std::sqrt(grav/t*dt_dz) : 0;
          real ell   = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20_fp) , delta );
          real km    = 0.1_fp * ell * std::sqrt(K);
          real Pr    = delta / (1+2*ell);
          real du_dy = (state(idU,hs+k,hs+j,hs+i,iens) - state(idU,hs+k,hs+j-1,hs+i,iens))/dy;
          real dv_dy = (state(idV,hs+k,hs+j,hs+i,iens) - state(idV,hs+k,hs+j-1,hs+i,iens))/dy;
          real dw_dy = (state(idW,hs+k,hs+j,hs+i,iens) - state(idW,hs+k,hs+j-1,hs+i,iens))/dy;
          real dt_dy = (state(idT,hs+k,hs+j,hs+i,iens) - state(idT,hs+k,hs+j-1,hs+i,iens))/dy;
          closure_u_y(k,j,i,iens) = -rho*km   *(du_dy + dv_dx);
          closure_v_y(k,j,i,iens) = -rho*km   *(dv_dy + dv_dy);
          closure_w_y(k,j,i,iens) = -rho*km   *(dw_dy + dv_dz);
          closure_t_y(k,j,i,iens) = -rho*km/Pr*dt_dy;
        }
        if (i < nx && j < ny) {
          real rho   = 0.5_fp * ( state(idR,hs+k-1,hs+j,hs+i,iens) + state(idR,hs+k,hs+j,hs+i,iens) );
          if (k > 0 && k < nz) {
            explicit_u_z(k,j,i,iens) = rho*(wu(hs+k-1,hs+j,hs+i,iens)+wu(hs+k,hs+j,hs+i,iens))/2;
            explicit_v_z(k,j,i,iens) = rho*(wv(hs+k-1,hs+j,hs+i,iens)+wv(hs+k,hs+j,hs+i,iens))/2;
            explicit_w_z(k,j,i,iens) = rho*(ww(hs+k-1,hs+j,hs+i,iens)+ww(hs+k,hs+j,hs+i,iens))/2;
            explicit_t_z(k,j,i,iens) = rho*(wt(hs+k-1,hs+j,hs+i,iens)+wt(hs+k,hs+j,hs+i,iens))/2;
          } else  {
            explicit_u_z(k,j,i,iens) = 0;
            explicit_v_z(k,j,i,iens) = 0;
            explicit_w_z(k,j,i,iens) = 0;
            explicit_t_z(k,j,i,iens) = 0;
          }

          tke_z    (k,j,i,iens) = (tke  (    hs+k-1,hs+j,hs+i,iens)+tke  (    hs+k,hs+j,hs+i,iens))/2;
          density_z(k,j,i,iens) = (state(idR,hs+k-1,hs+j,hs+i,iens)+state(idR,hs+k,hs+j,hs+i,iens))/2;
          uvel_z   (k,j,i,iens) = (state(idU,hs+k-1,hs+j,hs+i,iens)+state(idU,hs+k,hs+j,hs+i,iens))/2;
          uvel_z   (k,j,i,iens) = (state(idV,hs+k-1,hs+j,hs+i,iens)+state(idV,hs+k,hs+j,hs+i,iens))/2;
          wvel_z   (k,j,i,iens) = (state(idW,hs+k-1,hs+j,hs+i,iens)+state(idW,hs+k,hs+j,hs+i,iens))/2;
          theta_z  (k,j,i,iens) = (state(idT,hs+k-1,hs+j,hs+i,iens)+state(idT,hs+k,hs+j,hs+i,iens))/2;

          real K     = 0.5_fp * ( tke      (hs+k-1,hs+j,hs+i,iens) + tke      (hs+k,hs+j,hs+i,iens) );
          real t     = 0.5_fp * ( state(idT,hs+k-1,hs+j,hs+i,iens) + state(idT,hs+k,hs+j,hs+i,iens) );
          real dt_dz = (state(idT,hs+k,hs+j,hs+i,iens) - state(idT,hs+k-1,hs+j,hs+i,iens))/dz;
          real N     = dt_dz >= 0 ? std::sqrt(grav/t*dt_dz) : 0;
          real ell   = std::min( 0.76_fp*std::sqrt(K)/(N+1.e-20_fp) , delta );
          real km    = 0.1_fp * ell * std::sqrt(K);
          real Pr    = delta / (1+2*ell);
          real dw_dx = 0.5_fp * ( (state(idW,hs+k-1,hs+j,hs+i+1,iens) - state(idW,hs+k-1,hs+j,hs+i-1,iens))/(2*dx) +
                                  (state(idW,hs+k  ,hs+j,hs+i+1,iens) - state(idW,hs+k  ,hs+j,hs+i-1,iens))/(2*dx) );
          real dw_dy = 0.5_fp * ( (state(idW,hs+k-1,hs+j+1,hs+i,iens) - state(idW,hs+k-1,hs+j-1,hs+i,iens))/(2*dy) +
                                  (state(idW,hs+k  ,hs+j+1,hs+i,iens) - state(idW,hs+k  ,hs+j-1,hs+i,iens))/(2*dy) );
          real du_dz = (state(idU,hs+k,hs+j,hs+i,iens) - state(idU,hs+k-1,hs+j,hs+i,iens))/dz;
          real dv_dz = (state(idV,hs+k,hs+j,hs+i,iens) - state(idV,hs+k-1,hs+j,hs+i,iens))/dz;
          real dw_dz = (state(idW,hs+k,hs+j,hs+i,iens) - state(idW,hs+k-1,hs+j,hs+i,iens))/dz;
          closure_u_z(k,j,i,iens) = -rho*km   *(du_dz + dw_dx);
          closure_v_z(k,j,i,iens) = -rho*km   *(dv_dz + dw_dy);
          closure_w_z(k,j,i,iens) = -rho*km   *(dw_dz + dw_dz);
          closure_t_z(k,j,i,iens) = -rho*km/Pr*dt_dz;
        }
      });

      DEBUG_PRINT_MAIN_AVG(explicit_u_x)
      DEBUG_PRINT_MAIN_AVG(explicit_v_x)
      DEBUG_PRINT_MAIN_AVG(explicit_w_x)
      DEBUG_PRINT_MAIN_AVG(explicit_t_x)
      DEBUG_PRINT_MAIN_AVG(explicit_u_y)
      DEBUG_PRINT_MAIN_AVG(explicit_v_y)
      DEBUG_PRINT_MAIN_AVG(explicit_w_y)
      DEBUG_PRINT_MAIN_AVG(explicit_t_y)
      DEBUG_PRINT_MAIN_AVG(explicit_u_z)
      DEBUG_PRINT_MAIN_AVG(explicit_v_z)
      DEBUG_PRINT_MAIN_AVG(explicit_w_z)
      DEBUG_PRINT_MAIN_AVG(explicit_t_z)
      DEBUG_PRINT_MAIN_AVG(closure_u_x)
      DEBUG_PRINT_MAIN_AVG(closure_v_x)
      DEBUG_PRINT_MAIN_AVG(closure_w_x)
      DEBUG_PRINT_MAIN_AVG(closure_t_x)
      DEBUG_PRINT_MAIN_AVG(closure_u_y)
      DEBUG_PRINT_MAIN_AVG(closure_v_y)
      DEBUG_PRINT_MAIN_AVG(closure_w_y)
      DEBUG_PRINT_MAIN_AVG(closure_t_y)
      DEBUG_PRINT_MAIN_AVG(closure_u_z)
      DEBUG_PRINT_MAIN_AVG(closure_v_z)
      DEBUG_PRINT_MAIN_AVG(closure_w_z)
      DEBUG_PRINT_MAIN_AVG(closure_t_z)

      // TODO: change to pnetcdf writes
      yakl::SimplePNetCDF nc;
      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Info_set(info, "romio_no_indep_rw",    "true");
      MPI_Info_set(info, "nc_header_align_size", "1048576");
      MPI_Info_set(info, "nc_var_align_size",    "1048576");
      nc.create( "compare_les.nc" , NC_CLOBBER | NC_64BIT_DATA , info );
      nc.create_dim( "nx"   , nx_glob   );
      nc.create_dim( "ny"   , ny_glob   );
      nc.create_dim( "nz"   , nz        );
      nc.create_dim( "nxp1" , nx_glob+1 );
      nc.create_dim( "nyp1" , ny_glob+1 );
      nc.create_dim( "nzp1" , nz     +1 );
      nc.create_dim( "nens" , nens      );
      nc.create_var<real>( "explicit_u_x" , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "explicit_v_x" , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "explicit_w_x" , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "explicit_t_x" , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "explicit_u_y" , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "explicit_v_y" , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "explicit_w_y" , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "explicit_t_y" , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "explicit_u_z" , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "explicit_v_z" , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "explicit_w_z" , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "explicit_t_z" , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "closure_u_x"  , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "closure_v_x"  , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "closure_w_x"  , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "closure_t_x"  , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "closure_u_y"  , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "closure_v_y"  , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "closure_w_y"  , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "closure_t_y"  , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "closure_u_z"  , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "closure_v_z"  , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "closure_w_z"  , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "closure_t_z"  , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "tke_x"        , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "tke_y"        , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "tke_z"        , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "density_x"    , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "density_y"    , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "density_z"    , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "uvel_x"       , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "uvel_y"       , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "uvel_z"       , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "vvel_x"       , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "vvel_y"       , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "vvel_z"       , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "wvel_x"       , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "wvel_y"       , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "wvel_z"       , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.create_var<real>( "theta_x"      , {"nz"  ,"ny"  ,"nxp1","nens"} );
      nc.create_var<real>( "theta_y"      , {"nz"  ,"nyp1","nx"  ,"nens"} );
      nc.create_var<real>( "theta_z"      , {"nzp1","ny"  ,"nx"  ,"nens"} );
      nc.enddef();
      
      int constexpr DIR_X = 0;
      int constexpr DIR_Y = 1;
      int constexpr DIR_Z = 2;
      auto write_array = [&] ( real4d const &array , std::string label , int dir ) {
        real4d data("data",nz,ny,nx,nens);
        parallel_for( SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
          data(k,j,i,iens) = array(k,j,i,iens);
        });
        std::vector<MPI_Offset> start = {(MPI_Offset)0,(MPI_Offset)j_beg,(MPI_Offset)i_beg,(MPI_Offset)0};
        if      (dir == DIR_X) { nc.write_all( px < nproc_x-1 ? data.createHostCopy() : array.createHostCopy() , label , start ); }
        else if (dir == DIR_Y) { nc.write_all( py < nproc_y-1 ? data.createHostCopy() : array.createHostCopy() , label , start ); }
        else                   { nc.write_all(                                          array.createHostCopy() , label , start ); }
      };

      write_array( explicit_u_x , "explicit_u_x" , DIR_X );
      write_array( explicit_v_x , "explicit_v_x" , DIR_X );
      write_array( explicit_w_x , "explicit_w_x" , DIR_X );
      write_array( explicit_t_x , "explicit_t_x" , DIR_X );
      write_array( explicit_u_y , "explicit_u_y" , DIR_Y );
      write_array( explicit_v_y , "explicit_v_y" , DIR_Y );
      write_array( explicit_w_y , "explicit_w_y" , DIR_Y );
      write_array( explicit_t_y , "explicit_t_y" , DIR_Y );
      write_array( explicit_u_z , "explicit_u_z" , DIR_Z );
      write_array( explicit_v_z , "explicit_v_z" , DIR_Z );
      write_array( explicit_w_z , "explicit_w_z" , DIR_Z );
      write_array( explicit_t_z , "explicit_t_z" , DIR_Z );
      write_array( closure_u_x  , "closure_u_x"  , DIR_X );
      write_array( closure_v_x  , "closure_v_x"  , DIR_X );
      write_array( closure_w_x  , "closure_w_x"  , DIR_X );
      write_array( closure_t_x  , "closure_t_x"  , DIR_X );
      write_array( closure_u_y  , "closure_u_y"  , DIR_Y );
      write_array( closure_v_y  , "closure_v_y"  , DIR_Y );
      write_array( closure_w_y  , "closure_w_y"  , DIR_Y );
      write_array( closure_t_y  , "closure_t_y"  , DIR_Y );
      write_array( closure_u_z  , "closure_u_z"  , DIR_Z );
      write_array( closure_v_z  , "closure_v_z"  , DIR_Z );
      write_array( closure_w_z  , "closure_w_z"  , DIR_Z );
      write_array( closure_t_z  , "closure_t_z"  , DIR_Z );
      write_array( tke_x        , "tke_x"        , DIR_X );
      write_array( tke_y        , "tke_y"        , DIR_Y );
      write_array( tke_z        , "tke_z"        , DIR_Z );
      write_array( density_x    , "density_x"    , DIR_X );
      write_array( density_y    , "density_y"    , DIR_Y );
      write_array( density_z    , "density_z"    , DIR_Z );
      write_array( uvel_x       , "uvel_x"       , DIR_X );
      write_array( uvel_y       , "uvel_y"       , DIR_Y );
      write_array( uvel_z       , "uvel_z"       , DIR_Z );
      write_array( vvel_x       , "vvel_x"       , DIR_X );
      write_array( vvel_y       , "vvel_y"       , DIR_Y );
      write_array( vvel_z       , "vvel_z"       , DIR_Z );
      write_array( wvel_x       , "wvel_x"       , DIR_X );
      write_array( wvel_y       , "wvel_y"       , DIR_Y );
      write_array( wvel_z       , "wvel_z"       , DIR_Z );
      write_array( theta_x      , "theta_x"      , DIR_X );
      write_array( theta_y      , "theta_y"      , DIR_Y );
      write_array( theta_z      , "theta_z"      , DIR_Z );
      nc.close();
    }



    // Project stencil averages to cell-edge interpolations (No limiter)
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>     const & stencil     ,
                                                    SArray<real,1,ord>           & gll         ,
                                                    SArray<real,2,ord,ord> const & sten_to_gll ) {
      // Compute left and right cell edge estimates from stencil cell averages
      for (int ii=0; ii<ord; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) { tmp += sten_to_gll(s,ii) * stencil(s); }
        gll(ii) = tmp;
      }
    }



    // Exchange halo values periodically in the horizontal, and apply vertical no-slip solid-wall BC's
    void halo_exchange( core::Coupler const & coupler , core::MultiField<real,4> & fields ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens   = coupler.get_nens();
      auto nx     = coupler.get_nx();
      auto ny     = coupler.get_ny();
      auto nz     = coupler.get_nz();
      auto &neigh = coupler.get_neighbor_rankid_matrix();
      auto dtype  = coupler.get_mpi_data_type();
      MPI_Request sReq [2], rReq [2];
      MPI_Status  sStat[2], rStat[2];
      auto comm = MPI_COMM_WORLD;
      int npack = fields.size();

      // x-direction exchanges
      {
        real5d halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs,nens);
        real5d halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs,nens);
        real5d halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs,nens);
        real5d halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,ny,hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
          halo_send_buf_W(v,k,j,ii,iens) = fields(v,hs+k,hs+j,hs+ii,iens);
          halo_send_buf_E(v,k,j,ii,iens) = fields(v,hs+k,hs+j,nx+ii,iens);
        });
        yakl::timer_start("halo_exchange_mpi");
        #ifdef MW_GPU_AWARE_MPI
          yakl::fence();
          MPI_Irecv( halo_recv_buf_W.data() , halo_recv_buf_W.size() , dtype , neigh(1,0) , 0 , comm , &rReq[0] );
          MPI_Irecv( halo_recv_buf_E.data() , halo_recv_buf_E.size() , dtype , neigh(1,2) , 1 , comm , &rReq[1] );
          MPI_Isend( halo_send_buf_W.data() , halo_send_buf_W.size() , dtype , neigh(1,0) , 1 , comm , &sReq[0] );
          MPI_Isend( halo_send_buf_E.data() , halo_send_buf_E.size() , dtype , neigh(1,2) , 0 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("halo_exchange_mpi");
        #else
          auto halo_send_buf_W_host = halo_send_buf_W.createHostCopy();
          auto halo_send_buf_E_host = halo_send_buf_E.createHostCopy();
          auto halo_recv_buf_W_host = halo_recv_buf_W.createHostObject();
          auto halo_recv_buf_E_host = halo_recv_buf_E.createHostObject();
          MPI_Irecv( halo_recv_buf_W_host.data() , halo_recv_buf_W_host.size() , dtype , neigh(1,0) , 0 , comm , &rReq[0] );
          MPI_Irecv( halo_recv_buf_E_host.data() , halo_recv_buf_E_host.size() , dtype , neigh(1,2) , 1 , comm , &rReq[1] );
          halo_send_buf_W.deep_copy_to(halo_send_buf_W_host);
          halo_send_buf_E.deep_copy_to(halo_send_buf_E_host);
          yakl::fence();
          MPI_Isend( halo_send_buf_W_host.data() , halo_send_buf_W_host.size() , dtype , neigh(1,0) , 1 , comm , &sReq[0] );
          MPI_Isend( halo_send_buf_E_host.data() , halo_send_buf_E_host.size() , dtype , neigh(1,2) , 0 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("halo_exchange_mpi");
          halo_recv_buf_W_host.deep_copy_to(halo_recv_buf_W);
          halo_recv_buf_E_host.deep_copy_to(halo_recv_buf_E);
        #endif
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,ny,hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
          fields(v,hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
          fields(v,hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
        });
      }

      // y-direction exchanges
      {
        real5d halo_send_buf_S("halo_send_buf_S",npack,nz,hs,nx+2*hs,nens);
        real5d halo_send_buf_N("halo_send_buf_N",npack,nz,hs,nx+2*hs,nens);
        real5d halo_recv_buf_S("halo_recv_buf_S",npack,nz,hs,nx+2*hs,nens);
        real5d halo_recv_buf_N("halo_recv_buf_N",npack,nz,hs,nx+2*hs,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          halo_send_buf_S(v,k,jj,i,iens) = fields(v,hs+k,hs+jj,i,iens);
          halo_send_buf_N(v,k,jj,i,iens) = fields(v,hs+k,ny+jj,i,iens);
        });
        yakl::timer_start("halo_exchange_mpi");
        #ifdef MW_GPU_AWARE_MPI
          yakl::fence();
          MPI_Irecv( halo_recv_buf_S.data() , halo_recv_buf_S.size() , dtype , neigh(0,1) , 2 , comm , &rReq[0] );
          MPI_Irecv( halo_recv_buf_N.data() , halo_recv_buf_N.size() , dtype , neigh(2,1) , 3 , comm , &rReq[1] );
          MPI_Isend( halo_send_buf_S.data() , halo_send_buf_S.size() , dtype , neigh(0,1) , 3 , comm , &sReq[0] );
          MPI_Isend( halo_send_buf_N.data() , halo_send_buf_N.size() , dtype , neigh(2,1) , 2 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("halo_exchange_mpi");
        #else
          auto halo_send_buf_S_host = halo_send_buf_S.createHostCopy();
          auto halo_send_buf_N_host = halo_send_buf_N.createHostCopy();
          auto halo_recv_buf_S_host = halo_recv_buf_S.createHostObject();
          auto halo_recv_buf_N_host = halo_recv_buf_N.createHostObject();
          MPI_Irecv( halo_recv_buf_S_host.data() , halo_recv_buf_S_host.size() , dtype , neigh(0,1) , 2 , comm , &rReq[0] );
          MPI_Irecv( halo_recv_buf_N_host.data() , halo_recv_buf_N_host.size() , dtype , neigh(2,1) , 3 , comm , &rReq[1] );
          halo_send_buf_S.deep_copy_to(halo_send_buf_S_host);
          halo_send_buf_N.deep_copy_to(halo_send_buf_N_host);
          yakl::fence();
          MPI_Isend( halo_send_buf_S_host.data() , halo_send_buf_S_host.size() , dtype , neigh(0,1) , 3 , comm , &sReq[0] );
          MPI_Isend( halo_send_buf_N_host.data() , halo_send_buf_N_host.size() , dtype , neigh(2,1) , 2 , comm , &sReq[1] );
          MPI_Waitall(2, sReq, sStat);
          MPI_Waitall(2, rReq, rStat);
          yakl::timer_stop("halo_exchange_mpi");
          halo_recv_buf_S_host.deep_copy_to(halo_recv_buf_S);
          halo_recv_buf_N_host.deep_copy_to(halo_recv_buf_N);
        #endif
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,hs,nx+2*hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int jj, int i, int iens) {
          fields(v,hs+k,      jj,i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
          fields(v,hs+k,ny+hs+jj,i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
        });
      }
    }



    void halo_boundary_conditions( core::Coupler const & coupler  ,
                                   real5d        const & state    ,
                                   real5d        const & tracers  ,
                                   real4d        const & pressure ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens          = coupler.get_nens();
      auto nx            = coupler.get_nx();
      auto ny            = coupler.get_ny();
      auto nz            = coupler.get_nz();
      auto dz            = coupler.get_dz();
      auto num_tracers   = coupler.get_num_tracers();
      auto grav          = coupler.get_option<real>("grav");
      auto gamma         = coupler.get_option<real>("gamma_d");
      auto C0            = coupler.get_option<real>("C0");
      auto hy_dens_cells = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_cells");
      // z-direction BC's
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(hs,ny+2*hs,nx+2*hs,nens) ,
                                        YAKL_LAMBDA (int kk, int j, int i, int iens) {
        state(idR,kk,j,i,iens) = hy_dens_cells (kk,iens);
        state(idU,kk,j,i,iens) = state(idU,hs+0,j,i,iens);
        state(idV,kk,j,i,iens) = state(idV,hs+0,j,i,iens);
        state(idW,kk,j,i,iens) = 0;
        state(idT,kk,j,i,iens) = state(idT,hs+0,j,i,iens);
        pressure( kk,j,i,iens) = pressure( hs+0,j,i,iens);
        state(idR,hs+nz+kk,j,i,iens) = hy_dens_cells(hs+nz+kk,iens);
        state(idU,hs+nz+kk,j,i,iens) = state(idU,hs+nz-1,j,i,iens);
        state(idV,hs+nz+kk,j,i,iens) = state(idV,hs+nz-1,j,i,iens);
        state(idW,hs+nz+kk,j,i,iens) = 0;
        state(idT,hs+nz+kk,j,i,iens) = state(idT,hs+nz-1,j,i,iens);
        pressure( hs+nz+kk,j,i,iens) = pressure( hs+nz-1,j,i,iens);
        for (int l=0; l < num_tracers; l++) {
          tracers(l,      kk,j,i,iens) = tracers(l,hs+0   ,j,i,iens);
          tracers(l,hs+nz+kk,j,i,iens) = tracers(l,hs+nz-1,j,i,iens);
        }
      });
    }



    void halo_boundary_conditions( core::Coupler const & coupler  ,
                                   real4d const &uu ,
                                   real4d const &uv ,
                                   real4d const &uw ,
                                   real4d const &ut ,
                                   real4d const &vu ,
                                   real4d const &vv ,
                                   real4d const &vw ,
                                   real4d const &vt ,
                                   real4d const &wu ,
                                   real4d const &wv ,
                                   real4d const &ww ,
                                   real4d const &wt ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens          = coupler.get_nens();
      auto nx            = coupler.get_nx();
      auto ny            = coupler.get_ny();
      auto nz            = coupler.get_nz();
      // z-direction BC's
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(hs,ny+2*hs,nx+2*hs,nens) ,
                                        YAKL_LAMBDA (int kk, int j, int i, int iens) {
        uu(kk,j,i,iens) = uu(hs,j,i,iens);
        uv(kk,j,i,iens) = uv(hs,j,i,iens);
        uw(kk,j,i,iens) = 0;
        ut(kk,j,i,iens) = ut(hs,j,i,iens);
        vu(kk,j,i,iens) = vu(hs,j,i,iens);
        vv(kk,j,i,iens) = vv(hs,j,i,iens);
        vw(kk,j,i,iens) = 0;
        vt(kk,j,i,iens) = vt(hs,j,i,iens);
        wu(kk,j,i,iens) = 0;
        wv(kk,j,i,iens) = 0;
        ww(kk,j,i,iens) = 0;
        wt(kk,j,i,iens) = 0;

        uu(hs+nz+kk,j,i,iens) = uu(hs+nz-1,j,i,iens);
        uv(hs+nz+kk,j,i,iens) = uv(hs+nz-1,j,i,iens);
        uw(hs+nz+kk,j,i,iens) = 0;
        ut(hs+nz+kk,j,i,iens) = ut(hs+nz-1,j,i,iens);
        vu(hs+nz+kk,j,i,iens) = vu(hs+nz-1,j,i,iens);
        vv(hs+nz+kk,j,i,iens) = vv(hs+nz-1,j,i,iens);
        vw(hs+nz+kk,j,i,iens) = 0;
        vt(hs+nz+kk,j,i,iens) = vt(hs+nz-1,j,i,iens);
        wu(hs+nz+kk,j,i,iens) = 0;
        wv(hs+nz+kk,j,i,iens) = 0;
        ww(hs+nz+kk,j,i,iens) = 0;
        wt(hs+nz+kk,j,i,iens) = 0;
      });
    }



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler , realConst5d state , realConst5d tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto R_d         = coupler.get_option<real>("R_d"    );
      auto R_v         = coupler.get_option<real>("R_v"    );
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0"     );
      auto idWV        = coupler.get_option<int >("idWV"   );
      auto num_tracers = coupler.get_num_tracers();
      auto &dm = coupler.get_data_manager_readwrite();
      auto dm_rho_d = dm.get<real,4>("density_dry");
      auto dm_uvel  = dm.get<real,4>("uvel"       );
      auto dm_vvel  = dm.get<real,4>("vvel"       );
      auto dm_wvel  = dm.get<real,4>("wvel"       );
      auto dm_temp  = dm.get<real,4>("temp"       );
      auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      core::MultiField<real,4> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real,4>(tracer_names[tr]) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho   = state(idR,hs+k,hs+j,hs+i,iens);
        real u     = state(idU,hs+k,hs+j,hs+i,iens) / rho;
        real v     = state(idV,hs+k,hs+j,hs+i,iens) / rho;
        real w     = state(idW,hs+k,hs+j,hs+i,iens) / rho;
        real theta = state(idT,hs+k,hs+j,hs+i,iens) / rho;
        real press = C0 * pow( rho*theta , gamma );
        real rho_v = tracers(idWV,hs+k,hs+j,hs+i,iens);
        real rho_d = rho;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho_d -= tracers(tr,hs+k,hs+j,hs+i,iens); }
        real temp = press / ( rho_d * R_d + rho_v * R_v );
        dm_rho_d(k,j,i,iens) = rho_d;
        dm_uvel (k,j,i,iens) = u;
        dm_vvel (k,j,i,iens) = v;
        dm_wvel (k,j,i,iens) = w;
        dm_temp (k,j,i,iens) = temp;
        for (int tr=0; tr < num_tracers; tr++) { dm_tracers(tr,k,j,i,iens) = tracers(tr,hs+k,hs+j,hs+i,iens); }
      });
    }



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler , real5d &state , real5d &tracers ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nens        = coupler.get_nens();
      auto nx          = coupler.get_nx();
      auto ny          = coupler.get_ny();
      auto nz          = coupler.get_nz();
      auto R_d         = coupler.get_option<real>("R_d"    );
      auto R_v         = coupler.get_option<real>("R_v"    );
      auto gamma       = coupler.get_option<real>("gamma_d");
      auto C0          = coupler.get_option<real>("C0"     );
      auto idWV        = coupler.get_option<int >("idWV"   );
      auto num_tracers = coupler.get_num_tracers();
      auto &dm = coupler.get_data_manager_readonly();
      auto dm_rho_d = dm.get<real const,4>("density_dry");
      auto dm_uvel  = dm.get<real const,4>("uvel"       );
      auto dm_vvel  = dm.get<real const,4>("vvel"       );
      auto dm_wvel  = dm.get<real const,4>("wvel"       );
      auto dm_temp  = dm.get<real const,4>("temp"       );
      auto tracer_adds_mass = dm.get<bool const,1>("tracer_adds_mass");
      core::MultiField<real const,4> dm_tracers;
      auto tracer_names = coupler.get_tracer_names();
      for (int tr=0; tr < num_tracers; tr++) { dm_tracers.add_field( dm.get<real const,4>(tracer_names[tr]) ); }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real rho_d = dm_rho_d(k,j,i,iens);
        real u     = dm_uvel (k,j,i,iens);
        real v     = dm_vvel (k,j,i,iens);
        real w     = dm_wvel (k,j,i,iens);
        real temp  = dm_temp (k,j,i,iens);
        real rho_v = dm_tracers(idWV,k,j,i,iens);
        real press = rho_d * R_d * temp + rho_v * R_v * temp;
        real rho = rho_d;
        for (int tr=0; tr < num_tracers; tr++) { if (tracer_adds_mass(tr)) rho += dm_tracers(tr,k,j,i,iens); }
        real theta = pow( press/C0 , 1._fp / gamma ) / rho;
        state(idR,hs+k,hs+j,hs+i,iens) = rho;
        state(idU,hs+k,hs+j,hs+i,iens) = rho * u;
        state(idV,hs+k,hs+j,hs+i,iens) = rho * v;
        state(idW,hs+k,hs+j,hs+i,iens) = rho * w;
        state(idT,hs+k,hs+j,hs+i,iens) = rho * theta;
        for (int tr=0; tr < num_tracers; tr++) { tracers(tr,hs+k,hs+j,hs+i,iens) = dm_tracers(tr,k,j,i,iens); }
      });
    }


  };

}


