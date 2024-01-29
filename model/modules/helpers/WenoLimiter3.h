
#pragma once

#include "main_header.h"
#include "TransformMatrices.h"


namespace limiter {

  template <class T>
  YAKL_INLINE void convexify(T & w1, T & w2, T & w3) {
    T tot = w1 + w2 + w3;
    if (tot > static_cast<T>(1.e-20)) { w1 /= tot;   w2 /= tot;   w3 /= tot; }
  }


  template <class T>
  YAKL_INLINE void convexify(T & w1, T & w2, T & w3, T & w4) {
    T tot = w1 + w2 + w3 + w4;
    if (tot > static_cast<T>(1.e-20)) { w1 /= tot;   w2 /= tot;   w3 /= tot;   w4 /= tot; }
  }


  template <class T>
  YAKL_INLINE void convexify(T & w1, T & w2, T & w3, T & w4, T & w5) {
    T tot = w1 + w2 + w3 + w4 + w5;
    if (tot > static_cast<T>(1.e-20)) { w1 /= tot;   w2 /= tot;   w3 /= tot;   w4 /= tot;   w5 /= tot; }
  }


  template <class T>
  YAKL_INLINE void convexify(T & w1, T & w2, T & w3, T & w4, T & w5, T & w6) {
    T tot = w1 + w2 + w3 + w4 + w5 + w6;
    if (tot > static_cast<T>(1.e-20)) { w1 /= tot;   w2 /= tot;   w3 /= tot;   w4 /= tot;   w5 /= tot;   w6 /= tot; }
  }



  template <int ord> struct WenoLimiter;



  template <> struct WenoLimiter<1> {
    YAKL_INLINE WenoLimiter() { }
    YAKL_INLINE void compute_limited_coefs( SArray<real,1,1> const &s , SArray<real,1,1> &coefs_H ) const {
      coefs_H(0) = s(0);
    }
  };



  template <> struct WenoLimiter<5> {
    struct Params {
      real alpha;
    };
    Params params;
    typedef SArray<real,1,3> Weights;

    void set_params(real alpha_in = 2) { params.alpha = alpha_in; }

    YAKL_INLINE static void compute_limited_coefs( SArray<real,1,5> const &s         ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      SArray<real,1,3> coefs_L, coefs_C, coefs_R;
      TransformMatrices::coefs3_shift1( coefs_L , s(0) , s(1) , s(2) );
      TransformMatrices::coefs3_shift2( coefs_C , s(1) , s(2) , s(3) );
      TransformMatrices::coefs3_shift3( coefs_R , s(2) , s(3) , s(4) );
      // Compute TV
      real TV_L = TransformMatrices::coefs_to_tv( coefs_L );
      real TV_C = TransformMatrices::coefs_to_tv( coefs_C );
      real TV_R = TransformMatrices::coefs_to_tv( coefs_R );
      convexify( TV_L , TV_C , TV_R );
      real avg = 0.33333333333333333_fp*(TV_L+TV_C+TV_R);
      real adiff = std::abs(TV_L-avg) + std::abs(TV_C-avg) + std::abs(TV_R-avg);
      // Left evaluation
      real w_L = 0.3_fp / (TV_L*TV_L + 1.e-20_fp);
      real w_C = 0.6_fp / (TV_C*TV_C + 1.e-20_fp);
      real w_R = 0.1_fp / (TV_R*TV_R + 1.e-20_fp);
      convexify( w_L , w_C , w_R );
      qL = ((2*s(1) + 5*s(2) - s(3))*w_C - (s(0) - 5*s(1) - 2*s(2))*w_L + (11*s(2) - 7*s(3) + 2*s(4))*w_R)/6._fp;
      // Right evaluation
      w_L = 0.1_fp / (TV_L*TV_L + 1.e-20_fp);
      w_C = 0.6_fp / (TV_C*TV_C + 1.e-20_fp);
      w_R = 0.3_fp / (TV_R*TV_R + 1.e-20_fp);
      convexify( w_L , w_C , w_R );
      qR = (-(s(1) - 5*s(2) - 2*s(3))*w_C + (2*s(0) - 7*s(1) + 11*s(2))*w_L + (2*s(2) + 5*s(3) - s(4))*w_R)/6._fp;
    }

    YAKL_INLINE static void compute_limited_weights( SArray<real,1,5> const &s         ,
                                                     Weights                &weights   ,
                                                     Params           const &params_in ) {
      SArray<real,1,3> coefs_L, coefs_C, coefs_R;
      TransformMatrices::coefs3_shift1( coefs_L , s(0) , s(1) , s(2) );
      TransformMatrices::coefs3_shift2( coefs_C , s(1) , s(2) , s(3) );
      TransformMatrices::coefs3_shift3( coefs_R , s(2) , s(3) , s(4) );
      // Compute TV
      real TV_L = TransformMatrices::coefs_to_tv( coefs_L );
      real TV_C = TransformMatrices::coefs_to_tv( coefs_C );
      real TV_R = TransformMatrices::coefs_to_tv( coefs_R );
      convexify( TV_L , TV_C , TV_R );
      weights(0) = TV_L;
      weights(1) = TV_C;
      weights(2) = TV_R;
    }

    YAKL_INLINE static void apply_limited_weights( SArray<real ,1,5> const &s         ,
                                                   Weights           const &weights   ,
                                                   real                    &qL        ,
                                                   real                    &qR        ,
                                                   Params            const &params_in ) {
      real TV_L = weights(0);
      real TV_C = weights(1);
      real TV_R = weights(2);
      convexify( TV_L , TV_C , TV_R );
      real avg = 0.33333333333333333_fp*(TV_L+TV_C+TV_R);
      real adiff = std::abs(TV_L-avg) + std::abs(TV_C-avg) + std::abs(TV_R-avg);
      // Left evaluation
      real w_L = 0.3_fp / (TV_L*TV_L + 1.e-20_fp);
      real w_C = 0.6_fp / (TV_C*TV_C + 1.e-20_fp);
      real w_R = 0.1_fp / (TV_R*TV_R + 1.e-20_fp);
      convexify( w_L , w_C , w_R );
      qL = ((2*s(1) + 5*s(2) - s(3))*w_C - (s(0) - 5*s(1) - 2*s(2))*w_L + (11*s(2) - 7*s(3) + 2*s(4))*w_R)/6._fp;
      // Right evaluation
      w_L = 0.1_fp / (TV_L*TV_L + 1.e-20_fp);
      w_C = 0.6_fp / (TV_C*TV_C + 1.e-20_fp);
      w_R = 0.3_fp / (TV_R*TV_R + 1.e-20_fp);
      convexify( w_L , w_C , w_R );
      qR = (-(s(1) - 5*s(2) - 2*s(3))*w_C + (2*s(0) - 7*s(1) + 11*s(2))*w_L + (2*s(2) + 5*s(3) - s(4))*w_R)/6._fp;
    }
  };

}


