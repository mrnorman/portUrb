
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



  template <> struct WenoLimiter<3> {
    struct Params {
      float cutoff, idl_L, idl_R, idl_H;
      SArray<real,3,2,2,2> recon_lo;
      SArray<real,2,3,3>   recon_hi;
    };
    Params params;

    void set_params( real cutoff_in = 0.100 ,
                     real idl_L_in  = 1     ,
                     real idl_R_in  = 1     ,
                     real idl_H_in  = 250   ) {
      params.cutoff = cutoff_in;
      params.idl_L  = idl_L_in;
      params.idl_R  = idl_R_in;
      params.idl_H  = idl_H_in;
      convexify( params.idl_L , params.idl_R , params.idl_H );
      TransformMatrices::weno_lower_sten_to_coefs(params.recon_lo);
      TransformMatrices::sten_to_coefs           (params.recon_hi);
    }

    YAKL_INLINE static void compute_limited_coefs( SArray<real,1,3> const &stencil   ,
                                                   SArray<real,1,3>       &coefs_H   ,
                                                   Params           const &params_in ) {
      // Reconstruct high-order polynomial
      for (int ii=0; ii < 3; ii++) {
        real tmp_H = 0;
        for (int s=0; s < 3; s++) {
          tmp_H += params_in.recon_hi(s,ii) * stencil(0+s);
        }
        coefs_H(ii) = tmp_H;
      }
      // Reconstruct low-order polynomials
      SArray<real,1,2> coefs_L, coefs_R;
      for (int ii=0; ii < 2; ii++) {
        real tmp_L = 0, tmp_R = 0;
        for (int s=0; s < 2; s++) {
          tmp_L += params_in.recon_lo(0,s,ii) * stencil(0+s);
          tmp_R += params_in.recon_lo(1,s,ii) * stencil(1+s);
        }
        coefs_L(ii) = tmp_L;
        coefs_R(ii) = tmp_R;
      }
      // Compute total variation
      float w_L = TransformMatrices::coefs_to_tv( coefs_L );
      float w_R = TransformMatrices::coefs_to_tv( coefs_R );
      float w_H = TransformMatrices::coefs_to_tv( coefs_H );
      convexify( w_L , w_R , w_H );
      w_L = params_in.idl_L / (w_L*w_L + 1.e-20);
      w_R = params_in.idl_R / (w_R*w_R + 1.e-20);
      w_H = params_in.idl_H / (w_H*w_H + 1.e-20);
      convexify( w_L , w_R , w_H );
      if (w_L <= params_in.cutoff) w_L = 0;
      if (w_R <= params_in.cutoff) w_R = 0;
      convexify( w_L , w_R , w_H );
      coefs_H(0) = coefs_H(0)*w_H + coefs_L(0)*w_L + coefs_R(0)*w_R;
      coefs_H(1) = coefs_H(1)*w_H + coefs_L(1)*w_L + coefs_R(1)*w_R;
      coefs_H(2) = coefs_H(2)*w_H;
    }
  };



  template <> struct WenoLimiter<5> {
    struct Params {
      float cutoff, idl_L, idl_C, idl_R, idl_H;
      SArray<real,3,3,3,3> recon_lo;
      SArray<real,2,5,5>   recon_hi;
    };
    Params params;

    void set_params( real cutoff_in = 0.100 ,
                     real idl_L_in  = 1     ,
                     real idl_C_in  = 2     ,
                     real idl_R_in  = 1     ,
                     real idl_H_in  = 1000  ) {
      params.cutoff = cutoff_in;
      params.idl_L  = idl_L_in;
      params.idl_C  = idl_C_in;
      params.idl_R  = idl_R_in;
      params.idl_H  = idl_H_in;
      convexify( params.idl_L , params.idl_C , params.idl_R , params.idl_H );
      TransformMatrices::weno_lower_sten_to_coefs(params.recon_lo);
      TransformMatrices::sten_to_coefs           (params.recon_hi);
    }

    YAKL_INLINE static void compute_limited_coefs( SArray<real,1,5> const &stencil   ,
                                                   SArray<real,1,5>       &coefs_H   ,
                                                   Params           const &params_in ) {
      // Reconstruct high-order polynomial
      for (int ii=0; ii < 5; ii++) {
        real tmp_H = 0;
        for (int s=0; s < 5; s++) {
          tmp_H += params_in.recon_hi(s,ii) * stencil(0+s);
        }
        coefs_H(ii) = tmp_H;
      }
      // Reconstruct low-order polynomials
      SArray<real,1,3> coefs_L, coefs_C, coefs_R;
      for (int ii=0; ii < 3; ii++) {
        real tmp_L = 0, tmp_C = 0, tmp_R = 0;
        for (int s=0; s < 3; s++) {
          tmp_L += params_in.recon_lo(0,s,ii) * stencil(0+s);
          tmp_C += params_in.recon_lo(1,s,ii) * stencil(1+s);
          tmp_R += params_in.recon_lo(2,s,ii) * stencil(2+s);
        }
        coefs_L(ii) = tmp_L;
        coefs_C(ii) = tmp_C;
        coefs_R(ii) = tmp_R;
      }
      // Compute total variation
      float w_L = TransformMatrices::coefs_to_tv( coefs_L );
      float w_C = TransformMatrices::coefs_to_tv( coefs_C );
      float w_R = TransformMatrices::coefs_to_tv( coefs_R );
      float w_H = TransformMatrices::coefs_to_tv( coefs_H );
      convexify( w_L , w_C , w_R , w_H );
      w_L = params_in.idl_L / (w_L*w_L + 1.e-20);
      w_C = params_in.idl_C / (w_C*w_C + 1.e-20);
      w_R = params_in.idl_R / (w_R*w_R + 1.e-20);
      w_H = params_in.idl_H / (w_H*w_H + 1.e-20);
      convexify( w_L , w_C , w_R , w_H );
      if (w_L <= params_in.cutoff) w_L = 0;
      if (w_C <= params_in.cutoff) w_C = 0;
      if (w_R <= params_in.cutoff) w_R = 0;
      convexify( w_L , w_C , w_R , w_H );
      coefs_H(0) = coefs_H(0)*w_H + coefs_L(0)*w_L + coefs_C(0)*w_C + coefs_R(0)*w_R;
      coefs_H(1) = coefs_H(1)*w_H + coefs_L(1)*w_L + coefs_C(1)*w_C + coefs_R(1)*w_R;
      coefs_H(2) = coefs_H(2)*w_H + coefs_L(2)*w_L + coefs_C(2)*w_C + coefs_R(2)*w_R;
      coefs_H(3) = coefs_H(3)*w_H;
      coefs_H(4) = coefs_H(4)*w_H;
    }
  };


}


