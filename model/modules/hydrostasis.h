
#pragma once

#include "main_header.h"
#include "coupler.h"
#include "TransformMatrices.h"

namespace modules {

  typedef std::function<real(real)> FUNC_Z;

  template <yakl::index_t nq = 9>
  realHost2d integrate_hydrostatic_pressure_gll_temp_qv( FUNC_Z  func_T      ,
                                                         FUNC_Z  func_qv     ,
                                                         int     nz          ,
                                                         real    zlen        ,
                                                         real    p0   = 1.e5 ,
                                                         real    grav = 9.81 ,
                                                         real    R_d  = 287  ,
                                                         real    R_v  = 461  );

  template <yakl::index_t nq = 9>
  realHost2d integrate_hydrostatic_pressure_gll_theta( FUNC_Z  func_th     ,
                                                       int     nz          ,
                                                       real    zlen        ,
                                                       real    p0   = 1.e5 ,
                                                       real    grav = 9.81 ,
                                                       real    R_d  = 287  ,
                                                       real    c_p  = 1003 );

}


