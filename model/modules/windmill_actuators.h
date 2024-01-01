
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  struct WindmillActuators {

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
      real z0  = 90;
      real rad = 60;
      int ny_t = std::ceil(rad*2/dy);
      int nz_t = std::ceil((z0+rad)/dz);
      real y0 = ny_t*dy/2;
      int constexpr N = 1000;
      real ddy = dy / N;
      real ddz = dz / N;
      realHost2d templ_host("templ",nz_t,ny_t);
      for (int k=0; k < nz_t; k++) {
        for (int j=0; j < ny_t; j++) {
          int count = 0;
          for (int kk=0; kk < N; kk++) {
            for (int jj=0; jj < N; jj++) {
              real z = k*dz + kk*ddz;
              real y = j*dy + jj*ddy;
              if (std::sqrt((z-z0)*(z-z0)+(y-y0)*(y-y0))/rad < 1) count++;
            }
          }
          templ_host(k,j) = static_cast<double>(count)/static_cast<double>(N*N);
        }
      }
      auto sm = yakl::intrinsics::sum(templ_host);
      for (int k=0; k < nz_t; k++) {
        for (int j=0; j < ny_t; j++) {
          templ_host(k,j) = M_PI*rad*rad*templ_host(k,j)/sm;
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
              windmill_prop(k,j,i,iens) = templ(k,j_beg+j-yind0(jj))/(dx*dy*dz);
            }
          }
        }
      });
    }


    void apply( core::Coupler &coupler , real dt ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx  ();
      auto ny    = coupler.get_ny  ();
      auto nz    = coupler.get_nz  ();
      auto nens  = coupler.get_nens();
      auto dx    = coupler.get_dx  ();
      auto dy    = coupler.get_dy  ();
      auto dz    = coupler.get_dz  ();
      auto &dm   = coupler.get_data_manager_readwrite();
      auto windmill_prop = dm.get<real const,4>("windmill_prop");
      auto rho_d = dm.get<real const,4>("density_dry");
      auto uvel  = dm.get<real      ,4>("uvel"       );
      auto vvel  = dm.get<real      ,4>("vvel"       );
      auto wvel  = dm.get<real      ,4>("wvel"       );
      auto tke   = dm.get<real      ,4>("TKE"        );
      auto tend_u   = uvel.createDeviceObject();
      auto tend_v   = vvel.createDeviceObject();
      auto tend_w   = wvel.createDeviceObject();
      auto tend_tke = tke .createDeviceObject();
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        real r     = rho_d(k,j,i,iens);
        real u     = uvel (k,j,i,iens);
        real v     = vvel (k,j,i,iens);
        real w     = vvel (k,j,i,iens);
        real mag   = std::sqrt(u*u + v*v + w*w);
        real C_T   = 0.80;
        real C_P   = 0.60;
        real f_TKE = 0.5;
        real C_TKE = f_TKE * (C_T - C_P);
        real a     = 0.25_fp;
        real mag0  = mag/(1-a);
        real u0    = u  /(1-a);
        real v0    = v  /(1-a);
        real w0    = w  /(1-a);
        tend_u  (k,j,i,iens) = -0.5_fp  *C_T  *mag0*u0       *windmill_prop(k,j,i,iens);
        tend_v  (k,j,i,iens) = -0.5_fp  *C_T  *mag0*v0       *windmill_prop(k,j,i,iens);
        tend_w  (k,j,i,iens) = -0.5_fp  *C_T  *mag0*w0       *windmill_prop(k,j,i,iens);
        tend_tke(k,j,i,iens) =  0.5_fp*r*C_TKE*mag0*mag0*mag0*windmill_prop(k,j,i,iens);
      });
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        uvel(k,j,i,iens) += dt * tend_u  (k,j,i,iens);
        vvel(k,j,i,iens) += dt * tend_v  (k,j,i,iens);
        wvel(k,j,i,iens) += dt * tend_w  (k,j,i,iens);
        tke (k,j,i,iens) += dt * tend_tke(k,j,i,iens);
      });
    }

  };

}

