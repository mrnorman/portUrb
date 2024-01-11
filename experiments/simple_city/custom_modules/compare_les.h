
#pragma once

#include "main_header.h"
#include "MultipleFields.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
#include "PPM_Limiter.h"
#include "MinmodLimiter.h"
#include <random>
#include <sstream>

namespace modules {

  struct CompareLES {
    int  static constexpr hs  = (ord-1)/2; // Number of halo cells ("hs" == "halo size")
    int  static constexpr num_state = 5;   // Number of state variables
    int  static constexpr idR = 0;  // Density
    int  static constexpr idU = 1;  // u-momentum
    int  static constexpr idV = 2;  // v-momentum
    int  static constexpr idW = 3;  // w-momentum
    int  static constexpr idT = 4;  // Density * potential temperature


    void compare_les( core::Coupler & coupler ) const {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      using yakl::intrinsics::matmul_cr;
      auto nens                      = coupler.get_nens();
      auto nx                        = coupler.get_nx();    // Proces-local number of cells
      auto ny                        = coupler.get_ny();    // Proces-local number of cells
      auto nz                        = coupler.get_nz();    // Total vertical cells
      auto dx                        = coupler.get_dx();    // grid spacing
      auto dy                        = coupler.get_dy();    // grid spacing
      auto dz                        = coupler.get_dz();    // grid spacing
      auto sim2d                     = coupler.is_sim2d();  // Is this a 2-D simulation?
      auto C0                        = coupler.get_option<real>("C0"     );  // pressure = C0*pow(rho*theta,gamma)
      auto grav                      = coupler.get_option<real>("grav"   );  // Gravity
      auto gamma                     = coupler.get_option<real>("gamma_d");  // cp_dry / cv_dry (about 1.4)
      auto latitude                  = coupler.get_option<real>("latitude"); // For coriolis
      auto num_tracers               = coupler.get_num_tracers();            // Number of tracers
      auto &dm                       = coupler.get_data_manager_readonly();  // Grab read-only data manager
      auto tracer_positive           = dm.get<bool const,1>("tracer_positive"          ); // Is a tracer
                                                                                          //   positive-definite?
      auto immersed_proportion_halos = dm.get<real const,4>("immersed_proportion_halos"); // Proportion of immersed
                                                                                          //   material in each cell
      auto fully_immersed_halos      = dm.get<bool const,4>("fully_immersed_halos"     ); // Is a cell fullly immersed?
      auto hy_dens_cells             = dm.get<real const,2>("hy_dens_cells"            ); // Hydrostatic density
      auto hy_theta_cells            = dm.get<real const,2>("hy_theta_cells"           ); // Hydrostatic potential
                                                                                          //  temperature
      auto hy_dens_edges             = dm.get<real const,2>("hy_dens_edges"            ); // Hydrostatic density
      auto hy_theta_edges            = dm.get<real const,2>("hy_theta_edges"           ); // Hydrostatic potential
                                                                                          //  temperature
      auto hy_pressure_cells         = dm.get<real const,2>("hy_pressure_cells"        ); // Hydrostatic pressure
      // Compute matrices to convert polynomial coefficients to 2 GLL points and stencil values to 2 GLL points
      // These matrices will be in column-row format. That performed better than row-column format in performance tests
      SArray<real,2,ord,ord> s2c, c2g, c2d, s2g, s2d2g, g2c, g2d2g;
      TransformMatrices::sten_to_coefs (s2c);
      TransformMatrices::coefs_to_gll  (c2g);
      TransformMatrices::gll_to_coefs  (g2d);
      TransformMatrices::coefs_to_deriv(c2d);
      s2g   = matmul_cr(c2g,s2c);
      s2d2g = matmul_cr(c2g,matmul_cr(c2d,s2c);
      g2d2g = matmul_cr(c2g,matmul_cr(c2d,g2c);
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
      if (ord > 1) halo_exchange( coupler , state , tracers , pressure );

      real4d uu("uu",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d uv("uv",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d uw("uw",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d ut("ut",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d vv("vv",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d vw("vw",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d vt("vt",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d ww("ww",nz+2*hs,ny+2*hs,nx+2*hs,nens);
      real4d wt("wt",nz+2*hs,ny+2*hs,nx+2*hs,nens);

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        SArray<real,1,ord> stencil, gll;
        SArray<real,3,ord,ord,ord> gll_tmp1, gll_tmp2, gll_u, gll_v, gll_w, gll_t;
        //////////////////
        // u-velocity
        //////////////////
        // x direction
        for (int kk=0; kk < ord; kk++) {
        for (int jj=0; jj < ord; jj++) {
        for (int ii=0; ii < ord; ii++) {
          stencil(ii) = state(idU,k+kk,j+jj,i+ii,iens);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_tmp1(kk,jj,ii) = gll(ii);
        } } }
        // y direction
        for (int kk=0; kk < ord; kk++) {
        for (int ii=0; ii < ord; ii++) {
        for (int jj=0; jj < ord; jj++) {
          stencil(jj) = gll_tmp1(kk,jj,ii);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_tmp2(kk,jj,ii) = gll(jj);
        } } }
        // z direction
        for (int jj=0; jj < ord; jj++) {
        for (int ii=0; ii < ord; ii++) {
        for (int kk=0; kk < ord; kk++) {
          stencil(kk) = gll_tmp2(kk,jj,ii);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_u(kk,jj,ii) = gll(kk);
        } } }
        //////////////////
        // v-velocity
        //////////////////
        // x direction
        for (int kk=0; kk < ord; kk++) {
        for (int jj=0; jj < ord; jj++) {
        for (int ii=0; ii < ord; ii++) {
          stencil(ii) = state(idV,k+kk,j+jj,i+ii,iens);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_tmp1(kk,jj,ii) = gll(ii);
        } } }
        // y direction
        for (int kk=0; kk < ord; kk++) {
        for (int ii=0; ii < ord; ii++) {
        for (int jj=0; jj < ord; jj++) {
          stencil(jj) = gll_tmp1(kk,jj,ii);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_tmp2(kk,jj,ii) = gll(jj);
        } } }
        // z direction
        for (int jj=0; jj < ord; jj++) {
        for (int ii=0; ii < ord; ii++) {
        for (int kk=0; kk < ord; kk++) {
          stencil(kk) = gll_tmp2(kk,jj,ii);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_v(kk,jj,ii) = gll(kk);
        } } }
        //////////////////
        // w-velocity
        //////////////////
        // x direction
        for (int kk=0; kk < ord; kk++) {
        for (int jj=0; jj < ord; jj++) {
        for (int ii=0; ii < ord; ii++) {
          stencil(ii) = state(idW,k+kk,j+jj,i+ii,iens);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_tmp1(kk,jj,ii) = gll(ii);
        } } }
        // y direction
        for (int kk=0; kk < ord; kk++) {
        for (int ii=0; ii < ord; ii++) {
        for (int jj=0; jj < ord; jj++) {
          stencil(jj) = gll_tmp1(kk,jj,ii);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_tmp2(kk,jj,ii) = gll(jj);
        } } }
        // z direction
        for (int jj=0; jj < ord; jj++) {
        for (int ii=0; ii < ord; ii++) {
        for (int kk=0; kk < ord; kk++) {
          stencil(kk) = gll_tmp2(kk,jj,ii);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_w(kk,jj,ii) = gll(kk);
        } } }
        //////////////////
        // theta
        //////////////////
        // x direction
        for (int kk=0; kk < ord; kk++) {
        for (int jj=0; jj < ord; jj++) {
        for (int ii=0; ii < ord; ii++) {
          stencil(ii) = state(idT,k+kk,j+jj,i+ii,iens);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_tmp1(kk,jj,ii) = gll(ii);
        } } }
        // y direction
        for (int kk=0; kk < ord; kk++) {
        for (int ii=0; ii < ord; ii++) {
        for (int jj=0; jj < ord; jj++) {
          stencil(jj) = gll_tmp1(kk,jj,ii);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_tmp2(kk,jj,ii) = gll(jj);
        } } }
        // z direction
        for (int jj=0; jj < ord; jj++) {
        for (int ii=0; ii < ord; ii++) {
        for (int kk=0; kk < ord; kk++) {
          stencil(kk) = gll_tmp2(kk,jj,ii);
          reconstruct_gll_values( stencil , gll , s2g );
          gll_t(kk,jj,ii) = gll(kk);
        } } }
        ///////////////////////////////////////////////////
        // Remove the means, then compute sum of products
        ///////////////////////////////////////////////////
        using yakl::componentwise::operator*;
        using yakl::componentwise::operator-;
        using yakl::intrinsics::sum;
        gll_u = gll_u - sum(gll_u)/gll_u.size();
        gll_v = gll_v - sum(gll_v)/gll_v.size();
        gll_w = gll_w - sum(gll_w)/gll_w.size();
        gll_t = gll_t - sum(gll_t)/gll_t.size();
        uu(hs+k,hs+j,hs+i,iens) = sum(gll_u*gll_u);
        uv(hs+k,hs+j,hs+i,iens) = sum(gll_u*gll_v);
        uw(hs+k,hs+j,hs+i,iens) = sum(gll_u*gll_w);
        ut(hs+k,hs+j,hs+i,iens) = sum(gll_u*gll_t);
        vv(hs+k,hs+j,hs+i,iens) = sum(gll_v*gll_v);
        vw(hs+k,hs+j,hs+i,iens) = sum(gll_v*gll_w);
        vt(hs+k,hs+j,hs+i,iens) = sum(gll_v*gll_t);
        ww(hs+k,hs+j,hs+i,iens) = sum(gll_w*gll_w);
        wt(hs+k,hs+j,hs+i,iens) = sum(gll_w*gll_t);
      });

      // TODO: Halo exchange for sum of product quantities
      //       For vertical boundaries, w' is zero, but u' and v' are the same as the last adjacent cell


      // TODO: Reconstruct fluxes of sum of products and density to compute fluxes
      //       Assume w' is zero at the boundary, so zero flux for terms involving w'


      // TODO: Create halo routine that takes in a MultiField to make it more general

    }



    // Project stencil averages to cell-edge interpolations (No limiter)
    YAKL_INLINE static void reconstruct_gll_values( SArray<real,1,ord>   const & stencil     ,
                                                    SArray<real,1,2  >         & gll         ,
                                                    SArray<real,2,ord,2> const & sten_to_gll ) {
      // Compute left and right cell edge estimates from stencil cell averages
      for (int ii=0; ii<2; ii++) {
        real tmp = 0;
        for (int s=0; s < ord; s++) { tmp += sten_to_gll(s,ii) * stencil(s); }
        gll(ii) = tmp;
      }
    }



    // Exchange halo values periodically in the horizontal, and apply vertical no-slip solid-wall BC's
    void halo_exchange( core::Coupler const & coupler  ,
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
      auto &neigh        = coupler.get_neighbor_rankid_matrix();
      auto dtype         = coupler.get_mpi_data_type();
      auto grav          = coupler.get_option<real>("grav");
      auto gamma         = coupler.get_option<real>("gamma_d");
      auto C0            = coupler.get_option<real>("C0");
      auto hy_dens_cells = coupler.get_data_manager_readonly().get<real const,2>("hy_dens_cells");
      MPI_Request sReq [2];
      MPI_Request rReq [2];
      MPI_Status  sStat[2];
      MPI_Status  rStat[2];
      auto comm = MPI_COMM_WORLD;
      int npack = num_state + num_tracers + 1;

      // x-direction exchanges
      {
        real5d halo_send_buf_W("halo_send_buf_W",npack,nz,ny,hs,nens);
        real5d halo_send_buf_E("halo_send_buf_E",npack,nz,ny,hs,nens);
        real5d halo_recv_buf_W("halo_recv_buf_W",npack,nz,ny,hs,nens);
        real5d halo_recv_buf_E("halo_recv_buf_E",npack,nz,ny,hs,nens);
        parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<5>(npack,nz,ny,hs,nens) ,
                                          YAKL_LAMBDA (int v, int k, int j, int ii, int iens) {
          if        (v < num_state) {
            halo_send_buf_W(v,k,j,ii,iens) = state  (v          ,hs+k,hs+j,hs+ii,iens);
            halo_send_buf_E(v,k,j,ii,iens) = state  (v          ,hs+k,hs+j,nx+ii,iens);
          } else if (v < num_state + num_tracers) {
            halo_send_buf_W(v,k,j,ii,iens) = tracers(v-num_state,hs+k,hs+j,hs+ii,iens);
            halo_send_buf_E(v,k,j,ii,iens) = tracers(v-num_state,hs+k,hs+j,nx+ii,iens);
          } else {
            halo_send_buf_W(v,k,j,ii,iens) = pressure(hs+k,hs+j,hs+ii,iens);
            halo_send_buf_E(v,k,j,ii,iens) = pressure(hs+k,hs+j,nx+ii,iens);
          }
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
          realHost5d halo_send_buf_W_host("halo_send_buf_W_host",npack,nz,ny,hs,nens);
          realHost5d halo_send_buf_E_host("halo_send_buf_E_host",npack,nz,ny,hs,nens);
          realHost5d halo_recv_buf_W_host("halo_recv_buf_W_host",npack,nz,ny,hs,nens);
          realHost5d halo_recv_buf_E_host("halo_recv_buf_E_host",npack,nz,ny,hs,nens);
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
          if        (v < num_state) {
            state  (v          ,hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
            state  (v          ,hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
          } else if (v < num_state + num_tracers) {
            tracers(v-num_state,hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
            tracers(v-num_state,hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
          } else {
            pressure(hs+k,hs+j,      ii,iens) = halo_recv_buf_W(v,k,j,ii,iens);
            pressure(hs+k,hs+j,nx+hs+ii,iens) = halo_recv_buf_E(v,k,j,ii,iens);
          }
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
          if        (v < num_state) {
            halo_send_buf_S(v,k,jj,i,iens) = state  (v          ,hs+k,hs+jj,i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = state  (v          ,hs+k,ny+jj,i,iens);
          } else if (v < num_state + num_tracers) {
            halo_send_buf_S(v,k,jj,i,iens) = tracers(v-num_state,hs+k,hs+jj,i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = tracers(v-num_state,hs+k,ny+jj,i,iens);
          } else {
            halo_send_buf_S(v,k,jj,i,iens) = pressure(hs+k,hs+jj,i,iens);
            halo_send_buf_N(v,k,jj,i,iens) = pressure(hs+k,ny+jj,i,iens);
          }
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
          realHost5d halo_send_buf_S_host("halo_send_buf_S_host",npack,nz,hs,nx+2*hs,nens);
          realHost5d halo_send_buf_N_host("halo_send_buf_N_host",npack,nz,hs,nx+2*hs,nens);
          realHost5d halo_recv_buf_S_host("halo_recv_buf_S_host",npack,nz,hs,nx+2*hs,nens);
          realHost5d halo_recv_buf_N_host("halo_recv_buf_N_host",npack,nz,hs,nx+2*hs,nens);
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
          if        (v < num_state) {
            state  (v          ,hs+k,      jj,i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            state  (v          ,hs+k,ny+hs+jj,i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          } else if (v < num_state + num_tracers) {
            tracers(v-num_state,hs+k,      jj,i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            tracers(v-num_state,hs+k,ny+hs+jj,i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          } else {
            pressure(hs+k,      jj,i,iens) = halo_recv_buf_S(v,k,jj,i,iens);
            pressure(hs+k,ny+hs+jj,i,iens) = halo_recv_buf_N(v,k,jj,i,iens);
          }
        });
      }

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


