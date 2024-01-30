
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



  // YAKL_INLINE void map( real &w , real d ) { w = w*(d+d*d-3*d*w+w*w)/(d*d+(1-2*d)*w); }
  YAKL_INLINE void map_im( real &w , real d , int k=2, real A=0.1_fp ) {
    real term = w-d;
    for (int i=1; i<k; i++) { term *= w-d; }
    w = d + (A*term*(w-d))/(A*term+w*(1-w));
  }
  YAKL_INLINE void map_rs( real &w , real d , int k=6 , int m=3 , real s=2000._fp ) {
    real term1 = w-d;
    for (int i=1; i<k; i++) { term1 *= w-d; }
    real term2 = w*(1-w);
    for (int i=1; i<m; i++) { term2 *= w*(1-w); }
    w = d + (term1*(w-d))/(term1+s*term2);
  }



  template <> struct WenoLimiter<5> {
    struct Params {};
    Params params;
    typedef SArray<real,1,3> Weights;

    void set_params() { }

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
      // Left evaluation
      {
        real i_L = 0.3;
        real i_C = 0.6;
        real i_R = 0.1;
        real w_L = i_L / (TV_L*TV_L + 1.e-20_fp);
        real w_C = i_C / (TV_C*TV_C + 1.e-20_fp);
        real w_R = i_R / (TV_R*TV_R + 1.e-20_fp);
        convexify( w_L , w_C , w_R );
        map_rs(w_L,i_L);
        map_rs(w_C,i_C);
        map_rs(w_R,i_R);
        convexify( w_L , w_C , w_R );
        qL = ((2*s(1) + 5*s(2) - s(3))*w_C - (s(0) - 5*s(1) - 2*s(2))*w_L + (11*s(2) - 7*s(3) + 2*s(4))*w_R)/6._fp;
      }
      // Right evaluation
      {
        real i_L = 0.1;
        real i_C = 0.6;
        real i_R = 0.3;
        real w_L = i_L / (TV_L*TV_L + 1.e-20_fp);
        real w_C = i_C / (TV_C*TV_C + 1.e-20_fp);
        real w_R = i_R / (TV_R*TV_R + 1.e-20_fp);
        convexify( w_L , w_C , w_R );
        map_rs(w_L,i_L);
        map_rs(w_C,i_C);
        map_rs(w_R,i_R);
        convexify( w_L , w_C , w_R );
        qR = (-(s(1) - 5*s(2) - 2*s(3))*w_C + (2*s(0) - 7*s(1) + 11*s(2))*w_L + (2*s(2) + 5*s(3) - s(4))*w_R)/6._fp;
      }
    }

    YAKL_INLINE static void compute_limited_weights( SArray<real,1,5> const &s         ,
                                                     Weights                 &weights   ,
                                                     Params            const &params_in ) {
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
      // Left evaluation
      {
        real i_L = 0.3;
        real i_C = 0.6;
        real i_R = 0.1;
        real w_L = i_L / (TV_L*TV_L + 1.e-20_fp);
        real w_C = i_C / (TV_C*TV_C + 1.e-20_fp);
        real w_R = i_R / (TV_R*TV_R + 1.e-20_fp);
        convexify( w_L , w_C , w_R );
        map_rs(w_L,i_L);
        map_rs(w_C,i_C);
        map_rs(w_R,i_R);
        convexify( w_L , w_C , w_R );
        qL = ((2*s(1) + 5*s(2) - s(3))*w_C - (s(0) - 5*s(1) - 2*s(2))*w_L + (11*s(2) - 7*s(3) + 2*s(4))*w_R)/6._fp;
      }
      // Right evaluation
      {
        real i_L = 0.1;
        real i_C = 0.6;
        real i_R = 0.3;
        real w_L = i_L / (TV_L*TV_L + 1.e-20_fp);
        real w_C = i_C / (TV_C*TV_C + 1.e-20_fp);
        real w_R = i_R / (TV_R*TV_R + 1.e-20_fp);
        convexify( w_L , w_C , w_R );
        map_rs(w_L,i_L);
        map_rs(w_C,i_C);
        map_rs(w_R,i_R);
        convexify( w_L , w_C , w_R );
        qR = (-(s(1) - 5*s(2) - 2*s(3))*w_C + (2*s(0) - 7*s(1) + 11*s(2))*w_L + (2*s(2) + 5*s(3) - s(4))*w_R)/6._fp;
      }
    }
  };



  template <> struct WenoLimiter<7> {
    struct Params {};
    Params params;
    typedef SArray<real,1,4> Weights;

    void set_params() { }

    YAKL_INLINE static void compute_limited_coefs( SArray<real,1,7> const &s         ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      SArray<real,1,4> coefs_1, coefs_2, coefs_3, coefs_4;
      TransformMatrices::coefs4_shift1( coefs_1 , s(0) , s(1) , s(2) , s(3) );
      TransformMatrices::coefs4_shift2( coefs_2 , s(1) , s(2) , s(3) , s(4) );
      TransformMatrices::coefs4_shift3( coefs_3 , s(2) , s(3) , s(4) , s(5) );
      TransformMatrices::coefs4_shift4( coefs_4 , s(3) , s(4) , s(5) , s(6) );
      // Compute TV
      real TV_1 = TransformMatrices::coefs_to_tv( coefs_1 );
      real TV_2 = TransformMatrices::coefs_to_tv( coefs_2 );
      real TV_3 = TransformMatrices::coefs_to_tv( coefs_3 );
      real TV_4 = TransformMatrices::coefs_to_tv( coefs_4 );
      convexify( TV_1 , TV_2 , TV_3 , TV_4 );
      // Left evaluation
      {
        real i_1 = 4._fp /35._fp;
        real i_2 = 18._fp/35._fp;
        real i_3 = 12._fp/35._fp;
        real i_4 = 1._fp /35._fp;
        real w_1 = i_1 / (TV_1*TV_1 + 1.e-20_fp);
        real w_2 = i_2 / (TV_2*TV_2 + 1.e-20_fp);
        real w_3 = i_3 / (TV_3*TV_3 + 1.e-20_fp);
        real w_4 = i_4 / (TV_4*TV_4 + 1.e-20_fp);
        convexify( w_1 , w_2 , w_3 , w_4 );
        map_rs(w_1,i_1);
        map_rs(w_2,i_2);
        map_rs(w_3,i_3);
        map_rs(w_4,i_4);
        convexify( w_1 , w_2 , w_3 , w_4 );
        qL = 0.08333333333333333333_fp*s(0)*w_1-0.4166666666666666667_fp*s(1)*w_1+1.083333333333333333_fp*s(2)*w_1+0.2500000000000000000_fp*s(3)*w_1-0.08333333333333333333_fp*s(1)*w_2+0.5833333333333333333_fp*s(2)*w_2+0.5833333333333333333_fp*s(3)*w_2-0.08333333333333333333_fp*s(4)*w_2+0.2500000000000000000_fp*s(2)*w_3+1.083333333333333333_fp*s(3)*w_3-0.4166666666666666667_fp*s(4)*w_3+0.08333333333333333333_fp*s(5)*w_3+2.083333333333333333_fp*s(3)*w_4-1.916666666666666667_fp*s(4)*w_4+1.083333333333333333_fp*s(5)*w_4-0.2500000000000000000_fp*s(6)*w_4;
      }
      // Right evaluation
      {
        real i_1 = 1._fp /35._fp;
        real i_2 = 12._fp/35._fp;
        real i_3 = 18._fp/35._fp;
        real i_4 = 4._fp /35._fp;
        real w_1 = i_1 / (TV_1*TV_1 + 1.e-20_fp);
        real w_2 = i_2 / (TV_2*TV_2 + 1.e-20_fp);
        real w_3 = i_3 / (TV_3*TV_3 + 1.e-20_fp);
        real w_4 = i_4 / (TV_4*TV_4 + 1.e-20_fp);
        convexify( w_1 , w_2 , w_3 , w_4 );
        map_rs(w_1,i_1);
        map_rs(w_2,i_2);
        map_rs(w_3,i_3);
        map_rs(w_4,i_4);
        convexify( w_1 , w_2 , w_3 , w_4 );
        qR = -0.2500000000000000000_fp*s(0)*w_1+1.083333333333333333_fp*s(1)*w_1-1.916666666666666667_fp*s(2)*w_1+2.083333333333333333_fp*s(3)*w_1+0.08333333333333333333_fp*s(1)*w_2-0.4166666666666666667_fp*s(2)*w_2+1.083333333333333333_fp*s(3)*w_2+0.2500000000000000000_fp*s(4)*w_2-0.08333333333333333333_fp*s(2)*w_3+0.5833333333333333333_fp*s(3)*w_3+0.5833333333333333333_fp*s(4)*w_3-0.08333333333333333333_fp*s(5)*w_3+0.2500000000000000000_fp*s(3)*w_4+1.083333333333333333_fp*s(4)*w_4-0.4166666666666666667_fp*s(5)*w_4+0.08333333333333333333_fp*s(6)*w_4;
      }
    }

    YAKL_INLINE static void compute_limited_weights( SArray<real,1,7> const &s         ,
                                                     Weights                 &weights   ,
                                                     Params            const &params_in ) {
      SArray<real,1,4> coefs_1, coefs_2, coefs_3, coefs_4;
      TransformMatrices::coefs4_shift1( coefs_1 , s(0) , s(1) , s(2) , s(3) );
      TransformMatrices::coefs4_shift2( coefs_2 , s(1) , s(2) , s(3) , s(4) );
      TransformMatrices::coefs4_shift3( coefs_3 , s(2) , s(3) , s(4) , s(5) );
      TransformMatrices::coefs4_shift4( coefs_4 , s(3) , s(4) , s(5) , s(6) );
      // Compute TV
      real TV_1 = TransformMatrices::coefs_to_tv( coefs_1 );
      real TV_2 = TransformMatrices::coefs_to_tv( coefs_2 );
      real TV_3 = TransformMatrices::coefs_to_tv( coefs_3 );
      real TV_4 = TransformMatrices::coefs_to_tv( coefs_4 );
      convexify( TV_1 , TV_2 , TV_3 , TV_4 );
      weights(0) = TV_1;
      weights(1) = TV_2;
      weights(2) = TV_3;
      weights(3) = TV_4;
    }

    YAKL_INLINE static void apply_limited_weights( SArray<real ,1,7> const &s         ,
                                                   Weights           const &weights   ,
                                                   real                    &qL        ,
                                                   real                    &qR        ,
                                                   Params            const &params_in ) {
      real TV_1 = weights(0);
      real TV_2 = weights(1);
      real TV_3 = weights(2);
      real TV_4 = weights(3);
      convexify( TV_1 , TV_2 , TV_3 , TV_4 );
      // Left evaluation
      {
        real i_1 = 4._fp /35._fp;
        real i_2 = 18._fp/35._fp;
        real i_3 = 12._fp/35._fp;
        real i_4 = 1._fp /35._fp;
        real w_1 = i_1 / (TV_1*TV_1 + 1.e-20_fp);
        real w_2 = i_2 / (TV_2*TV_2 + 1.e-20_fp);
        real w_3 = i_3 / (TV_3*TV_3 + 1.e-20_fp);
        real w_4 = i_4 / (TV_4*TV_4 + 1.e-20_fp);
        convexify( w_1 , w_2 , w_3 , w_4 );
        map_rs(w_1,i_1);
        map_rs(w_2,i_2);
        map_rs(w_3,i_3);
        map_rs(w_4,i_4);
        convexify( w_1 , w_2 , w_3 , w_4 );
        qL = 0.08333333333333333333_fp*s(0)*w_1-0.4166666666666666667_fp*s(1)*w_1+1.083333333333333333_fp*s(2)*w_1+0.2500000000000000000_fp*s(3)*w_1-0.08333333333333333333_fp*s(1)*w_2+0.5833333333333333333_fp*s(2)*w_2+0.5833333333333333333_fp*s(3)*w_2-0.08333333333333333333_fp*s(4)*w_2+0.2500000000000000000_fp*s(2)*w_3+1.083333333333333333_fp*s(3)*w_3-0.4166666666666666667_fp*s(4)*w_3+0.08333333333333333333_fp*s(5)*w_3+2.083333333333333333_fp*s(3)*w_4-1.916666666666666667_fp*s(4)*w_4+1.083333333333333333_fp*s(5)*w_4-0.2500000000000000000_fp*s(6)*w_4;
      }
      // Right evaluation
      {
        real i_1 = 1._fp /35._fp;
        real i_2 = 12._fp/35._fp;
        real i_3 = 18._fp/35._fp;
        real i_4 = 4._fp /35._fp;
        real w_1 = i_1 / (TV_1*TV_1 + 1.e-20_fp);
        real w_2 = i_2 / (TV_2*TV_2 + 1.e-20_fp);
        real w_3 = i_3 / (TV_3*TV_3 + 1.e-20_fp);
        real w_4 = i_4 / (TV_4*TV_4 + 1.e-20_fp);
        convexify( w_1 , w_2 , w_3 , w_4 );
        map_rs(w_1,i_1);
        map_rs(w_2,i_2);
        map_rs(w_3,i_3);
        map_rs(w_4,i_4);
        convexify( w_1 , w_2 , w_3 , w_4 );
        qR = -0.2500000000000000000_fp*s(0)*w_1+1.083333333333333333_fp*s(1)*w_1-1.916666666666666667_fp*s(2)*w_1+2.083333333333333333_fp*s(3)*w_1+0.08333333333333333333_fp*s(1)*w_2-0.4166666666666666667_fp*s(2)*w_2+1.083333333333333333_fp*s(3)*w_2+0.2500000000000000000_fp*s(4)*w_2-0.08333333333333333333_fp*s(2)*w_3+0.5833333333333333333_fp*s(3)*w_3+0.5833333333333333333_fp*s(4)*w_3-0.08333333333333333333_fp*s(5)*w_3+0.2500000000000000000_fp*s(3)*w_4+1.083333333333333333_fp*s(4)*w_4-0.4166666666666666667_fp*s(5)*w_4+0.08333333333333333333_fp*s(6)*w_4;
      }
    }
  };

}


