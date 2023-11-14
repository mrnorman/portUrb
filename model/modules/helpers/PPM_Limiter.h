
#pragma once

#include "main_header.h"


namespace limiter {

  template <yakl::index_t ord>
  YAKL_INLINE void compute_high_order_edges( yakl::SArray<real,1,ord> const &s , real &L , real &R ) {
    int constexpr hs = (ord-1)/2;
    if        constexpr (ord == 1) {
      L=s(hs);
      R=s(hs);
    } else if constexpr (ord == 3) {
      L= 0.3333333333333333333_fp*s(hs-1)+0.8333333333333333333_fp*s(hs)-0.1666666666666666667_fp*s(hs+1);
      R=-0.1666666666666666667_fp*s(hs-1)+0.8333333333333333333_fp*s(hs)+0.3333333333333333333_fp*s(hs+1);
    } else if constexpr (ord == 5) {
      L=-0.05000000000000000000_fp*s(hs-2)+0.4500000000000000000_fp*s(hs-1)+0.7833333333333333333_fp*s(hs)-0.2166666666666666667_fp*s(hs+1)+0.03333333333333333333_fp*s(hs+2);
      R= 0.03333333333333333333_fp*s(hs-2)-0.2166666666666666667_fp*s(hs-1)+0.7833333333333333333_fp*s(hs)+0.4500000000000000000_fp*s(hs+1)-0.05000000000000000000_fp*s(hs+2);
    } else if constexpr (ord == 7) {
      L= 0.009523809523809523810_fp*s(hs-3)-0.09047619047619047619_fp*s(hs-2)+0.5095238095238095238_fp*s(hs-1)+0.7595238095238095238_fp*s(hs)-0.2404761904761904762_fp*s(hs+1)+0.05952380952380952381_fp*s(hs+2)-0.007142857142857142857_fp*s(hs+3);
      R=-0.007142857142857142857_fp*s(hs-3)+0.05952380952380952381_fp*s(hs-2)-0.2404761904761904762_fp*s(hs-1)+0.7595238095238095238_fp*s(hs)+0.5095238095238095238_fp*s(hs+1)-0.09047619047619047619_fp*s(hs+2)+0.009523809523809523810_fp*s(hs+3);

    } else if constexpr (ord == 9) {
      L=-0.001984126984126984127_fp*s(hs-4)+0.02182539682539682540_fp*s(hs-3)-0.1210317460317460317_fp *s(hs-2)+0.5456349206349206349_fp*s(hs-1)+0.7456349206349206349_fp*s(hs)-0.2543650793650793651_fp*s(hs+1)+0.07896825396825396825_fp*s(hs+2)-0.01626984126984126984_fp*s(hs+3)+0.001587301587301587302_fp*s(hs+4);
      R= 0.001587301587301587302_fp*s(hs-4)-0.01626984126984126984_fp*s(hs-3)+0.07896825396825396825_fp*s(hs-2)-0.2543650793650793651_fp*s(hs-1)+0.7456349206349206349_fp*s(hs)+0.5456349206349206349_fp*s(hs+1)-0.1210317460317460317_fp *s(hs+2)+0.02182539682539682540_fp*s(hs+3)-0.001984126984126984127_fp*s(hs+4);
    } else if constexpr (ord == 11) {
      L=0.0004329004329004329004_fp*s(hs-5)-0.005519480519480519481_fp*s(hs-4)+0.03416305916305916306_fp*s(hs-3)-0.1444083694083694084_fp*s(hs-2)+0.5698773448773448773_fp*s(hs-1)+0.7365440115440115440_fp*s(hs)-0.2634559884559884560_fp*s(hs+1)+0.09368686868686868687_fp*s(hs+2)-0.02536075036075036075_fp*s(hs+3)+0.004401154401154401154_fp*s(hs+4)-0.0003607503607503607504_fp*s(hs+5);
      R=-0.0003607503607503607504_fp*s(hs-5)+0.004401154401154401154_fp*s(hs-4)-0.02536075036075036075_fp*s(hs-3)+0.09368686868686868687_fp*s(hs-2)-0.2634559884559884560_fp*s(hs-1)+0.7365440115440115440_fp*s(hs)+0.5698773448773448773_fp*s(hs+1)-0.1444083694083694084_fp*s(hs+2)+0.03416305916305916306_fp*s(hs+3)-0.005519480519480519481_fp*s(hs+4)+0.0004329004329004329004_fp*s(hs+5);
    }
  }


  template <yakl::index_t ord>
  YAKL_INLINE void limit_high_order_edges( yakl::SArray<real,1,ord> const &s , real &L , real &R ) {
    int constexpr hs = (ord-1)/2;
    L = std::max( std::min(s(hs-1),s(hs)) , L );
    L = std::min( std::max(s(hs-1),s(hs)) , L );
    R = std::max( std::min(s(hs),s(hs+1)) , R );
    R = std::min( std::max(s(hs),s(hs+1)) , R );
  }



  template <int ord>
  struct PPM_Limiter {
    PPM_Limiter() = default;

    YAKL_INLINE void compute_limited_coefs( SArray<real,1,ord> const &s , SArray<real,1,ord> &coefs_H ) const {
      if constexpr (ord == 1) { coefs_H(0) = s(0); return; }
      int constexpr hs = (ord-1)/2;
      real C, L, R;
      C = s(hs);
      compute_high_order_edges( s , L , R );
      limit_high_order_edges  ( s , L , R );
      real x_extr = -0.16666666666666667_fp*(L - R)/(2*C - L - R);
      // If left and right derivatives are of different sign, set to first-order
      // If left deriv magnitude is larger, move extremum to right interface by changing left interface value
      // If right deriv magnitude is larger, move extremum to left interface by changing right interface value
      if (x_extr > -0.5_fp && x_extr < 0.5_fp) {
        if      ( (C-L)*(R-C) < 0 )             { R = C;  L = C; }
        else if (std::abs(C-L) > std::abs(R-C)) { L = 3*C - 2*R; }
        else                                    { R = 3*C - 2*L; }
      }
      coefs_H(0) = 3/2*C - 1/4*L - 1/4*R;
      coefs_H(1) = -L + R;
      coefs_H(2) = -6*C + 3*L + 3*R;
      for (int ii=3; ii < ord; ii++) { coefs_H(ii) = 0; }
    }
  };


}


