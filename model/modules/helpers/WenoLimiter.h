
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
      real cutoff, idl_L, idl_R, idl_H;
    };
    Params params;
    typedef SArray<real,1,3> Weights;

    void set_params( real cutoff_in = 0.100 ,
                     real idl_L_in  = 1     ,
                     real idl_R_in  = 1     ,
                     real idl_H_in  = 250   ) {
      params.cutoff = cutoff_in;
      params.idl_L  = idl_L_in;
      params.idl_R  = idl_R_in;
      params.idl_H  = idl_H_in;
      convexify( params.idl_L , params.idl_R , params.idl_H );
    }

    YAKL_INLINE static void compute_limited_coefs( SArray<real,1,3> const &s         ,
                                                   SArray<real,1,3>       &coefs_H   ,
                                                   Params           const &params_in ) {
      TransformMatrices::coefs3_shift2( coefs_H , s(0) , s(1) , s(2) );
      SArray<real,1,2> coefs_L, coefs_R;
      TransformMatrices::coefs2_shift1( coefs_L , s(0) , s(1) );
      TransformMatrices::coefs2_shift2( coefs_R , s(1) , s(2) );
      real w_L = TransformMatrices::coefs_to_tv( coefs_L );
      real w_R = TransformMatrices::coefs_to_tv( coefs_R );
      real w_H = TransformMatrices::coefs_to_tv( coefs_H );
      convexify( w_L , w_R , w_H );
      w_L = params_in.idl_L / (w_L*w_L + 1.e-20_fp);
      w_R = params_in.idl_R / (w_R*w_R + 1.e-20_fp);
      w_H = params_in.idl_H / (w_H*w_H + 1.e-20_fp);
      convexify( w_L , w_R , w_H );
      if (w_L <= params_in.cutoff) w_L = 0;
      if (w_R <= params_in.cutoff) w_R = 0;
      convexify( w_L , w_R , w_H );
      coefs_H(0) = coefs_H(0)*w_H + coefs_L(0)*w_L + coefs_R(0)*w_R;
      coefs_H(1) = coefs_H(1)*w_H + coefs_L(1)*w_L + coefs_R(1)*w_R;
      coefs_H(2) = coefs_H(2)*w_H;
    }

    YAKL_INLINE static void compute_limited_weights( SArray<real,1,3> const &s         ,
                                                     Weights                 &weights   ,
                                                     Params            const &params_in ) {
      SArray<real,1,3> coefs_H;
      TransformMatrices::coefs3_shift2( coefs_H , s(0) , s(1) , s(2) );
      SArray<real,1,2> coefs_L, coefs_R;
      TransformMatrices::coefs2_shift1( coefs_L , s(0) , s(1) );
      TransformMatrices::coefs2_shift2( coefs_R , s(1) , s(2) );
      real w_L = TransformMatrices::coefs_to_tv( coefs_L );
      real w_R = TransformMatrices::coefs_to_tv( coefs_R );
      real w_H = TransformMatrices::coefs_to_tv( coefs_H );
      convexify( w_L , w_R , w_H );
      w_L = params_in.idl_L / (w_L*w_L + 1.e-20_fp);
      w_R = params_in.idl_R / (w_R*w_R + 1.e-20_fp);
      w_H = params_in.idl_H / (w_H*w_H + 1.e-20_fp);
      convexify( w_L , w_R , w_H );
      if (w_L <= params_in.cutoff) w_L = 0;
      if (w_R <= params_in.cutoff) w_R = 0;
      convexify( w_L , w_R , w_H );
      weights(0) = w_H;
      weights(1) = w_L;
      weights(2) = w_R;
    }

    YAKL_INLINE static void apply_limited_weights( SArray<real ,1,3> const &s       ,
                                                   Weights           const &weights ,
                                                   SArray<real ,1,3>       &coefs_H ) {
      TransformMatrices::coefs3_shift2( coefs_H , s(0) , s(1) , s(2) );
      SArray<real,1,2> coefs_L, coefs_R;
      TransformMatrices::coefs2_shift1( coefs_L , s(0) , s(1) );
      TransformMatrices::coefs2_shift2( coefs_R , s(1) , s(2) );
      real w_H = weights(0);
      real w_L = weights(1);
      real w_R = weights(2);
      convexify( w_L , w_R , w_H );
      coefs_H(0) = coefs_H(0)*w_H + coefs_L(0)*w_L + coefs_R(0)*w_R;
      coefs_H(1) = coefs_H(1)*w_H + coefs_L(1)*w_L + coefs_R(1)*w_R;
      coefs_H(2) = coefs_H(2)*w_H;
    }
  };



  template <> struct WenoLimiter<5> {
    struct Params {
      real cutoff, idl_L, idl_C, idl_R, idl_H;
    };
    Params params;
    typedef SArray<real,1,4> Weights;

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
    }

    YAKL_INLINE static void compute_limited_coefs( SArray<real,1,5> const &s         ,
                                                   SArray<real,1,5>       &coefs_H   ,
                                                   Params           const &params_in ) {
      TransformMatrices::coefs5( coefs_H , s(0) , s(1) , s(2) , s(3) , s(4) );
      SArray<real,1,3> coefs_L, coefs_C, coefs_R;
      TransformMatrices::coefs3_shift1( coefs_L , s(0) , s(1) , s(2) );
      TransformMatrices::coefs3_shift2( coefs_C , s(1) , s(2) , s(3) );
      TransformMatrices::coefs3_shift3( coefs_R , s(2) , s(3) , s(4) );
      real w_L = TransformMatrices::coefs_to_tv( coefs_L );
      real w_C = TransformMatrices::coefs_to_tv( coefs_C );
      real w_R = TransformMatrices::coefs_to_tv( coefs_R );
      real w_H = TransformMatrices::coefs_to_tv( coefs_H );
      convexify( w_L , w_C , w_R , w_H );
      w_L = params_in.idl_L / (w_L*w_L + 1.e-20_fp);
      w_C = params_in.idl_C / (w_C*w_C + 1.e-20_fp);
      w_R = params_in.idl_R / (w_R*w_R + 1.e-20_fp);
      w_H = params_in.idl_H / (w_H*w_H + 1.e-20_fp);
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

    YAKL_INLINE static void compute_limited_weights( SArray<real,1,5> const &s         ,
                                                     Weights                 &weights   ,
                                                     Params            const &params_in ) {
      real mn=s(0);  mn=std::min(mn,s(1));  mn=std::min(mn,s(2));  mn=std::min(mn,s(3));  mn=std::min(mn,s(4));
      real mx=s(0);  mx=std::max(mx,s(1));  mx=std::max(mx,s(2));  mx=std::max(mx,s(3));  mx=std::max(mx,s(4));
      real mult = 1;
      if (mx-mn > 1.e-10_fp) mult = 1._fp / (mx-mn);
      for (int ii=0; ii < s.size(); ii++) { s(ii) = (s(ii)-mn)*mult; }
      SArray<real,1,5> coefs_H;
      TransformMatrices::coefs5( coefs_H , s(0) , s(1) , s(2) , s(3) , s(4) );
      SArray<real,1,3> coefs_L, coefs_C, coefs_R;
      TransformMatrices::coefs3_shift1( coefs_L , s(0) , s(1) , s(2) );
      TransformMatrices::coefs3_shift2( coefs_C , s(1) , s(2) , s(3) );
      TransformMatrices::coefs3_shift3( coefs_R , s(2) , s(3) , s(4) );
      real w_L = TransformMatrices::coefs_to_tv( coefs_L );
      real w_C = TransformMatrices::coefs_to_tv( coefs_C );
      real w_R = TransformMatrices::coefs_to_tv( coefs_R );
      real w_H = TransformMatrices::coefs_to_tv( coefs_H );
      convexify( w_L , w_C , w_R , w_H );
      w_L = params_in.idl_L / (w_L*w_L + 1.e-20_fp);
      w_C = params_in.idl_C / (w_C*w_C + 1.e-20_fp);
      w_R = params_in.idl_R / (w_R*w_R + 1.e-20_fp);
      w_H = params_in.idl_H / (w_H*w_H + 1.e-20_fp);
      convexify( w_L , w_C , w_R , w_H );
      if (w_L <= params_in.cutoff) w_L = 0;
      if (w_C <= params_in.cutoff) w_C = 0;
      if (w_R <= params_in.cutoff) w_R = 0;
      convexify( w_L , w_C , w_R , w_H );
      weights(0) = w_H;
      weights(1) = w_L;
      weights(2) = w_C;
      weights(3) = w_R;
    }

    YAKL_INLINE static void apply_limited_weights( SArray<real ,1,5> const &s       ,
                                                   Weights           const &weights ,
                                                   SArray<real ,1,5>       &coefs_H ) {
      TransformMatrices::coefs5( coefs_H , s(0) , s(1) , s(2) , s(3) , s(4) );
      SArray<real,1,3> coefs_L, coefs_C, coefs_R;
      TransformMatrices::coefs3_shift1( coefs_L , s(0) , s(1) , s(2) );
      TransformMatrices::coefs3_shift2( coefs_C , s(1) , s(2) , s(3) );
      TransformMatrices::coefs3_shift3( coefs_R , s(2) , s(3) , s(4) );
      real w_H = weights(0);
      real w_L = weights(1);
      real w_C = weights(2);
      real w_R = weights(3);
      convexify( w_L , w_C , w_R , w_H );
      coefs_H(0) = coefs_H(0)*w_H + coefs_L(0)*w_L + coefs_C(0)*w_C + coefs_R(0)*w_R;
      coefs_H(1) = coefs_H(1)*w_H + coefs_L(1)*w_L + coefs_C(1)*w_C + coefs_R(1)*w_R;
      coefs_H(2) = coefs_H(2)*w_H + coefs_L(2)*w_L + coefs_C(2)*w_C + coefs_R(2)*w_R;
      coefs_H(3) = coefs_H(3)*w_H;
      coefs_H(4) = coefs_H(4)*w_H;
    }
  };


}


