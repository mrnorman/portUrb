
#include "coupler.h"

namespace modules {

  void perturb_temperature( core::Coupler &coupler , int num_levels , bool thermal = false ) {
    using yakl::c::parallel_for;
    using yakl::c::SimpleBounds;

    auto nz    = coupler.get_nz();
    auto ny    = coupler.get_ny();
    auto nx    = coupler.get_nx();
    auto i_beg = coupler.get_i_beg();
    auto j_beg = coupler.get_j_beg();
    auto dx    = coupler.get_dx();
    auto dy    = coupler.get_dy();
    auto dz    = coupler.get_dz();
    auto xlen  = coupler.get_xlen();
    auto ylen  = coupler.get_ylen();

    auto &dm = coupler.get_data_manager_readwrite();

    real magnitude  = 3;
    size_t seed = static_cast<size_t>(coupler.get_myrank()*nz*nx*ny);

    // ny*nx can all be globbed together for this routine
    auto temp          = dm.get<real      ,3>("temp");
    auto immersed_prop = dm.get<real const,3>("immersed_proportion");

    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(num_levels,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
      yakl::Random prng(seed+k*ny*nx+j*nx+i);  // seed is a globally unique identifier
      real rand = prng.genFP<real>()*2._fp - 1._fp;  // Random number in [-1,1]
      real scaling = ( num_levels - static_cast<real>(k) ) / num_levels;  // Less effect at higher levels
      if (immersed_prop(k,j,i) == 0) temp(k,j,i) += rand * magnitude * scaling;
    });

    if (thermal) {
      auto temp = dm.get<real,3>("temp");

      parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(nz,ny,nx) , YAKL_LAMBDA (int k, int j, int i) {
        real xloc = (i+i_beg+0.5_fp)*dx;
        real yloc = (j+j_beg+0.5_fp)*dy;
        real zloc = (k      +0.5_fp)*dz;
        real x0 = xlen / 2;
        real y0 = ylen / 2;
        real z0 = 1500;
        real radx = 10000;
        real rady = 10000;
        real radz = 1500;
        real amp  = 5;
        real xn = (xloc - x0) / radx;
        real yn = (yloc - y0) / rady;
        real zn = (zloc - z0) / radz;
        real rad = sqrt( xn*xn + yn*yn + zn*zn );
        if (rad < 1) {
          temp(k,j,i) += amp * pow( cos(M_PI*rad/2) , 2._fp );
        }
      });
    }
  }

}


