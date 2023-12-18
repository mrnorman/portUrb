
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  struct WindmillActuators {

    void init( core::Coupler &coupler ) {
      using yakl::c::parallel_for;
      using yakl::c::SimpleBounds;
      auto nx    = coupler.get_nx  ();
      auto ny    = coupler.get_ny  ();
      auto nz    = coupler.get_nz  ();
      auto nens  = coupler.get_nens();
      auto dx    = coupler.get_dx  ();
      auto dy    = coupler.get_dy  ();
      auto dz    = coupler.get_dz  ();
      auto xlen  = coupler.get_xlen();
      auto ylen  = coupler.get_ylen();
      auto i_beg = coupler.get_i_beg();
      auto j_beg = coupler.get_j_beg();
      auto &dm   = coupler.get_data_manager_readwrite();
      dm.register_and_allocate<real>( "windmill_prop" , "" , {nz,ny,nx,nens} , {"z","y","x","nens"} );
      auto windmill_prop = dm.get<real,4>("windmill_prop");
      real x0  = xlen/2;
      real y0  = ylen/2;
      real z0  = 90;
      real rad = 58;
      int constexpr N = 100;
      real ddy = dy / N;
      real ddz = dz / N;
      int ny_t = std::ceil(rad*2/dy);
      int nz_t = std::ceil(rad*2/dz);
      realHost2d templ("templ",nz_t,ny_t);
      for (int k=0; k < nz; k++) {
      for (int j=0; j < ny; j++) {
        for (int kk=0; kk < N; kk++) {
        for (int jj=0; jj < N; jj++) {
        }
        }
      }
      }
      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<4>(nz,ny,nx,nens) , YAKL_LAMBDA (int k, int j, int i, int iens) {
        if ( (i_beg+i)*dx >= x0     && (i_beg+i+1)*dx <  x0     &&
             (j_beg+j)*dy >= y0-rad && (j_beg+j+1)*dy <= y0+rad &&
             (      k)*dz >= z0-rad && (      k+1)*dz <= z0+rad ) {
          int count = 0;
          for (int kk = 0; kk < N; kk++) {
            for (int jj = 0; jj < N; jj++) {
              real y = (j_beg+j-0.5_fp)*dy + jj*ddy;
              real z = (      k-0.5_fp)*dz + kk*ddz;
              real norm = std::sqrt(((y-y0)*(y-y0) + (z-z0)*(z-z0)))/rad;
              if (norm <= 1) count++;
            }
          }
          windmill_prop(k,j,i,iens) = static_cast<double>(count)/static_cast<double>(N*N);
        }
      });
      if (coupler.is_mainproc()) std::cout << windmill_prop;
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
        real vel   = std::sqrt(u*u + v*v);
        real C_T   = 0.8;
        real C_P   = 0.7;
        real f_TKE = 0.5;
        real C_TKE = f_TKE * (C_T - C_P);
        real a     = 0.5_fp*(1-std::sqrt(1-C_T*vel));
        real vel0  = vel/(1-a);
        real u0    = u  /(1-a);
        real v0    = v  /(1-a);
        tend_u  (k,j,i,iens) = -0.5_fp*C_T*vel0*u0*windmill_prop(k,j,i,iens);
        tend_v  (k,j,i,iens) = -0.5_fp*C_T*vel0*v0*windmill_prop(k,j,i,iens);
        tend_w  (k,j,i,iens) = 0;
        tend_tke(k,j,i,iens) = 0.5_fp*r*C_TKE*vel0*vel0*vel0*windmill_prop(k,j,i,iens);
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

