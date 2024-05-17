
#pragma once

#include "main_header.h"
#include "TransformMatrices.h"


namespace limiter {

  template <class T>
  YAKL_INLINE void convexify(T & w1, T & w2) {
    T tot = w1 + w2;
    if (tot > static_cast<T>(1.e-20)) { w1 /= tot;   w2 /= tot; }
  }

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



  template <class real>
  YAKL_INLINE void static map( real &w , real d ) { w = w*(d+d*d-3*d*w+w*w)/(d*d+(1-2*d)*w); }


  template <class real>
  YAKL_INLINE void static map_im( real &w , real d , int k=2, real A=0.1_fp ) {
    real term = w-d;
    for (int i=1; i<k; i++) { term *= w-d; }
    w = d + (A*term*(w-d))/(A*term+w*(1-w));
  }


  template <class real>
  YAKL_INLINE void static map_rs( real &w , real d , int k=6 , int m=3 , real s=2000._fp ) {
    real term1 = w-d;
    for (int i=1; i<k; i++) { term1 *= w-d; }
    real term2 = w*(1-w);
    for (int i=1; i<m; i++) { term2 *= w*(1-w); }
    w = d + (term1*(w-d))/(term1+s*term2);
  }


  template <yakl::index_t N, class real>
  YAKL_INLINE void static normalize( SArray<real,1,N> &s , real &mn , real &scale ) {
    mn = s(0);
    real mx = s(0);
    for (int i=1; i < N; i++) {
      mn = std::min( mn , s(i) );
      mx = std::max( mx , s(i) );
    }
    scale = 1;
    if (mx-mn > 1.e-10) scale = mx-mn;
    for (int i=0; i < N; i++) { s(i) = (s(i) - mn) / scale; }
  }



  template <> struct WenoLimiter<1> {
    struct Params { bool do_map; };
    Params params;
    typedef SArray<real,1,1> Weights;
    void set_params( bool do_map = true ) { params.do_map = do_map; }
    YAKL_INLINE WenoLimiter() { }
    YAKL_INLINE static void compute_limited_edges( SArray<real,1,1> const &s         ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      qL = s(1);
      qR = s(1);
    }
    YAKL_INLINE static void compute_limited_weights( SArray<real,1,5> const &s         ,
                                                     Weights                &weights   ,
                                                     Params           const &params_in ) { }
    YAKL_INLINE static void apply_limited_weights( SArray<real ,1,5> const &s         ,
                                                   Weights           const &weights   ,
                                                   real                    &qL        ,
                                                   real                    &qR        ,
                                                   Params            const &params_in ) {
      qL = s(1);
      qR = s(1);
    }
    YAKL_INLINE static void compute_non_limited_edges( SArray<real,1,5> const &s , real &qL , real &qR ) {
      qL = s(1);
      qR = s(1);
    }
  };



  template <> struct WenoLimiter<3> {
    struct Params { bool do_map; };
    Params params;
    typedef SArray<real,1,2> Weights;

    void set_params( bool do_map = true ) { params.do_map = do_map; }

    YAKL_INLINE static void compute_limited_edges( SArray<real,1,3> const &s         ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      SArray<real,1,2> coefs_L, coefs_R;
      TransformMatrices::coefs2_shift1( coefs_L , s(0) , s(1) );
      TransformMatrices::coefs2_shift2( coefs_R , s(1) , s(2) );
      // Compute TV
      real TV_L = TransformMatrices::coefs_to_tv( coefs_L );
      real TV_R = TransformMatrices::coefs_to_tv( coefs_R );
      convexify( TV_L , TV_R );
      // Left evaluation
      {
        real i_L = 2._fp/3._fp;
        real i_R = 1._fp/3._fp;
        real w_L = i_L / (TV_L*TV_L + 1.e-20_fp);
        real w_R = i_R / (TV_R*TV_R + 1.e-20_fp);
        convexify( w_L , w_R );
        if (params_in.do_map) {
          map( w_L , i_L );
          map( w_R , i_R );
          convexify( w_L , w_R );
        }
        qL = 0.5000000000000000000_fp*(s(0)+s(1))*w_L+0.5000000000000000000_fp*(3*s(1)-s(2))*w_R;
      }
      // Right evaluation
      {
        real i_L = 1._fp/3._fp;
        real i_R = 2._fp/3._fp;
        real w_L = i_L / (TV_L*TV_L + 1.e-20_fp);
        real w_R = i_R / (TV_R*TV_R + 1.e-20_fp);
        convexify( w_L , w_R );
        if (params_in.do_map) {
          map( w_L , i_L );
          map( w_R , i_R );
          convexify( w_L , w_R );
        }
        qR = -0.5000000000000000000_fp*(s(0)-3*s(1))*w_L+0.5000000000000000000_fp*(s(1)+s(2))*w_R;
      }
      qL = 0.3333333333333333333_fp*s(0)+0.8333333333333333333_fp*s(1)-0.1666666666666666667_fp*s(2);
      qR = -0.1666666666666666667_fp*s(0)+0.8333333333333333333_fp*s(1)+0.3333333333333333333_fp*s(2);
    }

    YAKL_INLINE static void compute_limited_weights( SArray<real,1,3> const &s         ,
                                                     Weights                &weights   ,
                                                     Params           const &params_in ) {
      SArray<real,1,2> coefs_L, coefs_R;
      TransformMatrices::coefs2_shift1( coefs_L , s(0) , s(1) );
      TransformMatrices::coefs2_shift2( coefs_R , s(1) , s(2) );
      // Compute TV
      real TV_L = TransformMatrices::coefs_to_tv( coefs_L );
      real TV_R = TransformMatrices::coefs_to_tv( coefs_R );
      convexify( TV_L , TV_R );
      weights(0) = TV_L;
      weights(1) = TV_R;
    }

    YAKL_INLINE static void apply_limited_weights( SArray<real,1,3> const &s         ,
                                                   Weights          const &weights   ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      real TV_L = weights(0);
      real TV_R = weights(1);
      convexify( TV_L , TV_R );
      // Left evaluation
      {
        real i_L = 2._fp/3._fp;
        real i_R = 1._fp/3._fp;
        real w_L = i_L / (TV_L*TV_L + 1.e-20_fp);
        real w_R = i_R / (TV_R*TV_R + 1.e-20_fp);
        convexify( w_L , w_R );
        if (params_in.do_map) {
          map( w_L , i_L );
          map( w_R , i_R );
          convexify( w_L , w_R );
        }
        qL = 0.5000000000000000000_fp*(s(0)+s(1))*w_L+0.5000000000000000000_fp*(3*s(1)-s(2))*w_R;
      }
      // Right evaluation
      {
        real i_L = 1._fp/3._fp;
        real i_R = 2._fp/3._fp;
        real w_L = i_L / (TV_L*TV_L + 1.e-20_fp);
        real w_R = i_R / (TV_R*TV_R + 1.e-20_fp);
        convexify( w_L , w_R );
        if (params_in.do_map) {
          map( w_L , i_L );
          map( w_R , i_R );
          convexify( w_L , w_R );
        }
        qR = -0.5000000000000000000_fp*(s(0)-3*s(1))*w_L+0.5000000000000000000_fp*(s(1)+s(2))*w_R;
      }
      qL = 0.3333333333333333333_fp*s(0)+0.8333333333333333333_fp*s(1)-0.1666666666666666667_fp*s(2);
      qR = -0.1666666666666666667_fp*s(0)+0.8333333333333333333_fp*s(1)+0.3333333333333333333_fp*s(2);
    }

    YAKL_INLINE static void compute_non_limited_edges( SArray<real,1,3> const &s , real &qL , real &qR ) {
      qL = 0.3333333333333333333_fp*s(0)+0.8333333333333333333_fp*s(1)-0.1666666666666666667_fp*s(2);
      qR = -0.1666666666666666667_fp*s(0)+0.8333333333333333333_fp*s(1)+0.3333333333333333333_fp*s(2);
    }
  };



  template <> struct WenoLimiter<5> {
    struct Params { bool do_map; bool imm_L; bool imm_R; };
    Params params;
    typedef SArray<float,1,3> Weights;

    void set_params( bool do_map = false , bool imm_L = false , bool imm_R = false ) {
      params.do_map = do_map;
      params.imm_L  = imm_L; 
      params.imm_R  = imm_R;
    }

    // Don't allow the stencil that doesn't contain the left interface to dominate
    YAKL_INLINE static void imm_L_alter( float &TV_1 , float &TV_2 , float &TV_3 ) {
      TV_3 = std::max(std::max(TV_1,TV_2),TV_3);
    }

    // Don't allow the stencil that doesn't contain the right interface to dominate
    YAKL_INLINE static void imm_R_alter( float &TV_1 , float &TV_2 , float &TV_3 ) {
      TV_1 = std::max(std::max(TV_1,TV_2),TV_3);
    }

    YAKL_INLINE static void compute_limited_edges( SArray<real,1,5> const &s         ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      SArray<float,1,3> coefs_L, coefs_C, coefs_R;
      TransformMatrices::coefs3_shift1( coefs_L , static_cast<float>(s(0)) , static_cast<float>(s(1)) , static_cast<float>(s(2)) );
      TransformMatrices::coefs3_shift2( coefs_C , static_cast<float>(s(1)) , static_cast<float>(s(2)) , static_cast<float>(s(3)) );
      TransformMatrices::coefs3_shift3( coefs_R , static_cast<float>(s(2)) , static_cast<float>(s(3)) , static_cast<float>(s(4)) );
      // Compute TV
      float TV_L = TransformMatrices::coefs_to_tv( coefs_L );
      float TV_C = TransformMatrices::coefs_to_tv( coefs_C );
      float TV_R = TransformMatrices::coefs_to_tv( coefs_R );
      convexify( TV_L , TV_C , TV_R );
      if (params_in.imm_L) { imm_L_alter(TV_L,TV_C,TV_R); }
      if (params_in.imm_R) { imm_R_alter(TV_L,TV_C,TV_R); }
      // Left evaluation
      {
        float i_L = 0.3;
        float i_C = 0.6;
        float i_R = 0.1;
        float w_L = i_L / (TV_L*TV_L + 1.e-20f);
        float w_C = i_C / (TV_C*TV_C + 1.e-20f);
        float w_R = i_R / (TV_R*TV_R + 1.e-20f);
        convexify( w_L , w_C , w_R );
        if (params_in.do_map) {
          map_rs( w_L , i_L );
          map_rs( w_C , i_C );
          map_rs( w_R , i_R );
          convexify( w_L , w_C , w_R );
        }
        qL = 0.1666666666666666667_fp*(2*s(1)+5*s(2)-s(3))*w_C-0.1666666666666666667_fp*(s(0)-5*s(1)-2*s(2))*w_L+0.1666666666666666667_fp*(11*s(2)-7*s(3)+2*s(4))*w_R;
      }
      // Right evaluation
      {
        float i_L = 0.1;
        float i_C = 0.6;
        float i_R = 0.3;
        float w_L = i_L / (TV_L*TV_L + 1.e-20f);
        float w_C = i_C / (TV_C*TV_C + 1.e-20f);
        float w_R = i_R / (TV_R*TV_R + 1.e-20f);
        convexify( w_L , w_C , w_R );
        if (params_in.do_map) {
          map_rs( w_L , i_L );
          map_rs( w_C , i_C );
          map_rs( w_R , i_R );
          convexify( w_L , w_C , w_R );
        }
        qR = -0.1666666666666666667_fp*(s(1)-5*s(2)-2*s(3))*w_C+0.1666666666666666667_fp*(2*s(0)-7*s(1)+11*s(2))*w_L+0.1666666666666666667_fp*(2*s(2)+5*s(3)-s(4))*w_R;
      }
    }

    YAKL_INLINE static void compute_limited_weights( SArray<real,1,5> const &s         ,
                                                     Weights                &weights   ,
                                                     Params           const &params_in ) {
      SArray<float,1,3> coefs_L, coefs_C, coefs_R;
      TransformMatrices::coefs3_shift1( coefs_L , static_cast<float>(s(0)) , static_cast<float>(s(1)) , static_cast<float>(s(2)) );
      TransformMatrices::coefs3_shift2( coefs_C , static_cast<float>(s(1)) , static_cast<float>(s(2)) , static_cast<float>(s(3)) );
      TransformMatrices::coefs3_shift3( coefs_R , static_cast<float>(s(2)) , static_cast<float>(s(3)) , static_cast<float>(s(4)) );
      // Compute TV
      float TV_L = TransformMatrices::coefs_to_tv( coefs_L );
      float TV_C = TransformMatrices::coefs_to_tv( coefs_C );
      float TV_R = TransformMatrices::coefs_to_tv( coefs_R );
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
      float TV_L = weights(0);
      float TV_C = weights(1);
      float TV_R = weights(2);
      convexify( TV_L , TV_C , TV_R );
      // Left evaluation
      {
        float i_L = 0.3f;
        float i_C = 0.6f;
        float i_R = 0.1f;
        float w_L = i_L / (TV_L*TV_L + 1.e-20f);
        float w_C = i_C / (TV_C*TV_C + 1.e-20f);
        float w_R = i_R / (TV_R*TV_R + 1.e-20f);
        convexify( w_L , w_C , w_R );
        if (params_in.do_map) {
          map_im( w_L , i_L );
          map_im( w_C , i_C );
          map_im( w_R , i_R );
          convexify( w_L , w_C , w_R );
        }
        qL = 0.1666666666666666667_fp*(2*s(1)+5*s(2)-s(3))*w_C-0.1666666666666666667_fp*(s(0)-5*s(1)-2*s(2))*w_L+0.1666666666666666667_fp*(11*s(2)-7*s(3)+2*s(4))*w_R;
      }
      // Right evaluation
      {
        float i_L = 0.1f;
        float i_C = 0.6f;
        float i_R = 0.3f;
        float w_L = i_L / (TV_L*TV_L + 1.e-20f);
        float w_C = i_C / (TV_C*TV_C + 1.e-20f);
        float w_R = i_R / (TV_R*TV_R + 1.e-20f);
        convexify( w_L , w_C , w_R );
        if (params_in.do_map) {
          map_im( w_L , i_L );
          map_im( w_C , i_C );
          map_im( w_R , i_R );
          convexify( w_L , w_C , w_R );
        }
        qR = -0.1666666666666666667_fp*(s(1)-5*s(2)-2*s(3))*w_C+0.1666666666666666667_fp*(2*s(0)-7*s(1)+11*s(2))*w_L+0.1666666666666666667_fp*(2*s(2)+5*s(3)-s(4))*w_R;
      }
    }

    YAKL_INLINE static void compute_non_limited_edges( SArray<real,1,5> const &s , real &qL , real &qR ) {
      qL = -0.05000000000000000000_fp*s(0)+0.4500000000000000000_fp*s(1)+0.7833333333333333333_fp*s(2)-0.2166666666666666667_fp*s(3)+0.03333333333333333333_fp*s(4);
      qR = 0.03333333333333333333_fp*s(0)-0.2166666666666666667_fp*s(1)+0.7833333333333333333_fp*s(2)+0.4500000000000000000_fp*s(3)-0.05000000000000000000_fp*s(4);
    }
  };



  template <> struct WenoLimiter<7> {
    struct Params { bool do_map; };
    Params params;
    typedef SArray<real,1,4> Weights;

    void set_params( bool do_map = true ) { params.do_map = do_map; }

    YAKL_INLINE static void compute_limited_edges( SArray<real,1,7> const &s         ,
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
        if (params_in.do_map) {
          map( w_1 , i_1 );
          map( w_2 , i_2 );
          map( w_3 , i_3 );
          map( w_4 , i_4 );
          convexify( w_1 , w_2 , w_3 , w_4 );
        }
        qL = 0.08333333333333333333_fp*(s(0)-5*s(1)+13*s(2)+3*s(3))*w_1-0.08333333333333333333_fp*(s(1)-7*s(2)-7*s(3)+s(4))*w_2+0.08333333333333333333_fp*(3*s(2)+13*s(3)-5*s(4)+s(5))*w_3+0.08333333333333333333_fp*(25*s(3)-23*s(4)+13*s(5)-3*s(6))*w_4;
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
        if (params_in.do_map) {
          map( w_1 , i_1 );
          map( w_2 , i_2 );
          map( w_3 , i_3 );
          map( w_4 , i_4 );
          convexify( w_1 , w_2 , w_3 , w_4 );
        }
        qR = -0.08333333333333333333_fp*(3*s(0)-13*s(1)+23*s(2)-25*s(3))*w_1+0.08333333333333333333_fp*(s(1)-5*s(2)+13*s(3)+3*s(4))*w_2-0.08333333333333333333_fp*(s(2)-7*s(3)-7*s(4)+s(5))*w_3+0.08333333333333333333_fp*(3*s(3)+13*s(4)-5*s(5)+s(6))*w_4;
      }
    }

    YAKL_INLINE static void compute_limited_weights( SArray<real,1,7>  const &s         ,
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

    YAKL_INLINE static void apply_limited_weights( SArray<real ,1,7>       &s         ,
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
        if (params_in.do_map) {
          map( w_1 , i_1 );
          map( w_2 , i_2 );
          map( w_3 , i_3 );
          map( w_4 , i_4 );
          convexify( w_1 , w_2 , w_3 , w_4 );
        }
        qL = 0.08333333333333333333_fp*(s(0)-5*s(1)+13*s(2)+3*s(3))*w_1-0.08333333333333333333_fp*(s(1)-7*s(2)-7*s(3)+s(4))*w_2+0.08333333333333333333_fp*(3*s(2)+13*s(3)-5*s(4)+s(5))*w_3+0.08333333333333333333_fp*(25*s(3)-23*s(4)+13*s(5)-3*s(6))*w_4;
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
        if (params_in.do_map) {
          map( w_1 , i_1 );
          map( w_2 , i_2 );
          map( w_3 , i_3 );
          map( w_4 , i_4 );
          convexify( w_1 , w_2 , w_3 , w_4 );
        }
        qR = -0.08333333333333333333_fp*(3*s(0)-13*s(1)+23*s(2)-25*s(3))*w_1+0.08333333333333333333_fp*(s(1)-5*s(2)+13*s(3)+3*s(4))*w_2-0.08333333333333333333_fp*(s(2)-7*s(3)-7*s(4)+s(5))*w_3+0.08333333333333333333_fp*(3*s(3)+13*s(4)-5*s(5)+s(6))*w_4;
      }
    }

    YAKL_INLINE static void compute_non_limited_edges( SArray<real,1,7> const &s , real &qL , real &qR ) {
      qL = 0.009523809523809523810_fp*s(0)-0.09047619047619047619_fp*s(1)+0.5095238095238095238_fp*s(2)+0.7595238095238095238_fp*s(3)-0.2404761904761904762_fp*s(4)+0.05952380952380952381_fp*s(5)-0.007142857142857142857_fp*s(6);
      qR = -0.007142857142857142857_fp*s(0)+0.05952380952380952381_fp*s(1)-0.2404761904761904762_fp*s(2)+0.7595238095238095238_fp*s(3)+0.5095238095238095238_fp*s(4)-0.09047619047619047619_fp*s(5)+0.009523809523809523810_fp*s(6);
    }
  };



  template <> struct WenoLimiter<9> {
    struct Params { bool do_map; bool imm_L; bool imm_R; };
    Params params;
    typedef SArray<float,1,5> Weights;

    void set_params( bool do_map = false , bool imm_L = false , bool imm_R = false ) {
      params.do_map = do_map;
      params.imm_L  = imm_L; 
      params.imm_R  = imm_R;
    }

    // Don't allow the stencil that doesn't contain the left interface to dominate
    YAKL_INLINE static void imm_L_alter( float &TV_1 , float &TV_2 , float &TV_3 , float &TV_4 , float &TV_5 ) {
      TV_5 = std::max(std::max(std::max(std::max(TV_1,TV_2),TV_3),TV_4),TV_5);
    }

    // Don't allow the stencil that doesn't contain the right interface to dominate
    YAKL_INLINE static void imm_R_alter( float &TV_1 , float &TV_2 , float &TV_3 , float &TV_4 , float &TV_5 ) {
      TV_1 = std::max(std::max(std::max(std::max(TV_1,TV_2),TV_3),TV_4),TV_5);
    }

    YAKL_INLINE static void compute_limited_edges( SArray<real,1,9>       &s         ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      SArray<float,1,5> coefs_1, coefs_2, coefs_3, coefs_4, coefs_5;
      TransformMatrices::coefs5_shift1( coefs_1 , static_cast<float>(s(0)) , static_cast<float>(s(1)) , static_cast<float>(s(2)) , static_cast<float>(s(3)) , static_cast<float>(s(4)) );
      TransformMatrices::coefs5_shift2( coefs_2 , static_cast<float>(s(1)) , static_cast<float>(s(2)) , static_cast<float>(s(3)) , static_cast<float>(s(4)) , static_cast<float>(s(5)) );
      TransformMatrices::coefs5_shift3( coefs_3 , static_cast<float>(s(2)) , static_cast<float>(s(3)) , static_cast<float>(s(4)) , static_cast<float>(s(5)) , static_cast<float>(s(6)) );
      TransformMatrices::coefs5_shift4( coefs_4 , static_cast<float>(s(3)) , static_cast<float>(s(4)) , static_cast<float>(s(5)) , static_cast<float>(s(6)) , static_cast<float>(s(7)) );
      TransformMatrices::coefs5_shift5( coefs_5 , static_cast<float>(s(4)) , static_cast<float>(s(5)) , static_cast<float>(s(6)) , static_cast<float>(s(7)) , static_cast<float>(s(8)) );
      // Compute TV
      float TV_1 = TransformMatrices::coefs_to_tv( coefs_1 );
      float TV_2 = TransformMatrices::coefs_to_tv( coefs_2 );
      float TV_3 = TransformMatrices::coefs_to_tv( coefs_3 );
      float TV_4 = TransformMatrices::coefs_to_tv( coefs_4 );
      float TV_5 = TransformMatrices::coefs_to_tv( coefs_5 );
      convexify(TV_1,TV_2,TV_3,TV_4,TV_5);
      if (params_in.imm_L) { imm_L_alter(TV_1,TV_2,TV_3,TV_4,TV_5); }
      if (params_in.imm_R) { imm_R_alter(TV_1,TV_2,TV_3,TV_4,TV_5); }
      // Left evaluation
      {
        float i_1 =  0.03968253968253968254f;
        float i_2 =   0.3174603174603174603f;
        float i_3 =   0.4761904761904761905f;
        float i_4 =   0.1587301587301587302f;
        float i_5 = 0.007936507936507936508f;
        float w_1 = i_1 / (TV_1*TV_1 + 1.e-20f);
        float w_2 = i_2 / (TV_2*TV_2 + 1.e-20f);
        float w_3 = i_3 / (TV_3*TV_3 + 1.e-20f);
        float w_4 = i_4 / (TV_4*TV_4 + 1.e-20f);
        float w_5 = i_5 / (TV_5*TV_5 + 1.e-20f);
        convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
        if (params_in.do_map) {
          map_im( w_1 , i_1 );
          map_im( w_2 , i_2 );
          map_im( w_3 , i_3 );
          map_im( w_4 , i_4 );
          map_im( w_5 , i_5 );
          convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
        }
        qL = -0.01666666666666666667_fp*(3*s(0)-17*s(1)+43*s(2)-77*s(3)-12*s(4))*w_1+0.01666666666666666667_fp*(2*s(1)-13*s(2)+47*s(3)+27*s(4)-3*s(5))*w_2-0.01666666666666666667_fp*(3*s(2)-27*s(3)-47*s(4)+13*s(5)-2*s(6))*w_3+0.01666666666666666667_fp*(12*s(3)+77*s(4)-43*s(5)+17*s(6)-3*s(7))*w_4+0.01666666666666666667_fp*(137*s(4)-163*s(5)+137*s(6)-63*s(7)+12*s(8))*w_5;
      }
      // Right evaluation
      {
        float i_1 = 0.007936507936507936508f;
        float i_2 =   0.1587301587301587302f;
        float i_3 =   0.4761904761904761905f;
        float i_4 =   0.3174603174603174603f;
        float i_5 =  0.03968253968253968254f;
        float w_1 = i_1 / (TV_1*TV_1 + 1.e-20f);
        float w_2 = i_2 / (TV_2*TV_2 + 1.e-20f);
        float w_3 = i_3 / (TV_3*TV_3 + 1.e-20f);
        float w_4 = i_4 / (TV_4*TV_4 + 1.e-20f);
        float w_5 = i_5 / (TV_5*TV_5 + 1.e-20f);
        convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
        if (params_in.do_map) {
          map_im( w_1 , i_1 );
          map_im( w_2 , i_2 );
          map_im( w_3 , i_3 );
          map_im( w_4 , i_4 );
          map_im( w_5 , i_5 );
          convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
        }
        qR = 0.01666666666666666667_fp*(12*s(0)-63*s(1)+137*s(2)-163*s(3)+137*s(4))*w_1-0.01666666666666666667_fp*(3*s(1)-17*s(2)+43*s(3)-77*s(4)-12*s(5))*w_2+0.01666666666666666667_fp*(2*s(2)-13*s(3)+47*s(4)+27*s(5)-3*s(6))*w_3-0.01666666666666666667_fp*(3*s(3)-27*s(4)-47*s(5)+13*s(6)-2*s(7))*w_4+0.01666666666666666667_fp*(12*s(4)+77*s(5)-43*s(6)+17*s(7)-3*s(8))*w_5;
      }
    }

    YAKL_INLINE static void compute_limited_weights( SArray<real,1,9>       &s         ,
                                                     Weights                 &weights   ,
                                                     Params            const &params_in ) {
      SArray<float,1,5> coefs_1, coefs_2, coefs_3, coefs_4, coefs_5;
      TransformMatrices::coefs5_shift1( coefs_1 , static_cast<float>(s(0)) , static_cast<float>(s(1)) , static_cast<float>(s(2)) , static_cast<float>(s(3)) , static_cast<float>(s(4)) );
      TransformMatrices::coefs5_shift2( coefs_2 , static_cast<float>(s(1)) , static_cast<float>(s(2)) , static_cast<float>(s(3)) , static_cast<float>(s(4)) , static_cast<float>(s(5)) );
      TransformMatrices::coefs5_shift3( coefs_3 , static_cast<float>(s(2)) , static_cast<float>(s(3)) , static_cast<float>(s(4)) , static_cast<float>(s(5)) , static_cast<float>(s(6)) );
      TransformMatrices::coefs5_shift4( coefs_4 , static_cast<float>(s(3)) , static_cast<float>(s(4)) , static_cast<float>(s(5)) , static_cast<float>(s(6)) , static_cast<float>(s(7)) );
      TransformMatrices::coefs5_shift5( coefs_5 , static_cast<float>(s(4)) , static_cast<float>(s(5)) , static_cast<float>(s(6)) , static_cast<float>(s(7)) , static_cast<float>(s(8)) );
      // Compute TV
      float TV_1 = TransformMatrices::coefs_to_tv( coefs_1 );
      float TV_2 = TransformMatrices::coefs_to_tv( coefs_2 );
      float TV_3 = TransformMatrices::coefs_to_tv( coefs_3 );
      float TV_4 = TransformMatrices::coefs_to_tv( coefs_4 );
      float TV_5 = TransformMatrices::coefs_to_tv( coefs_5 );
      convexify( TV_1 , TV_2 , TV_3 , TV_4 , TV_5 );
      weights(0) = TV_1;
      weights(1) = TV_2;
      weights(2) = TV_3;
      weights(3) = TV_4;
      weights(4) = TV_5;
    }

    YAKL_INLINE static void apply_limited_weights( SArray<real ,1,9>       &s         ,
                                                   Weights           const &weights   ,
                                                   real                    &qL        ,
                                                   real                    &qR        ,
                                                   Params            const &params_in ) {
      float TV_1 = weights(0);
      float TV_2 = weights(1);
      float TV_3 = weights(2);
      float TV_4 = weights(3);
      float TV_5 = weights(4);
      if (params_in.imm_L) { imm_L_alter(TV_1,TV_2,TV_3,TV_4,TV_5); }
      if (params_in.imm_R) { imm_R_alter(TV_1,TV_2,TV_3,TV_4,TV_5); }
      convexify( TV_1 , TV_2 , TV_3 , TV_4 , TV_5 );
      // Left evaluation
      {
        float i_1 =  0.03968253968253968254f;
        float i_2 =   0.3174603174603174603f;
        float i_3 =   0.4761904761904761905f;
        float i_4 =   0.1587301587301587302f;
        float i_5 = 0.007936507936507936508f;
        float w_1 = i_1 / (TV_1*TV_1 + 1.e-20f);
        float w_2 = i_2 / (TV_2*TV_2 + 1.e-20f);
        float w_3 = i_3 / (TV_3*TV_3 + 1.e-20f);
        float w_4 = i_4 / (TV_4*TV_4 + 1.e-20f);
        float w_5 = i_5 / (TV_5*TV_5 + 1.e-20f);
        convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
        if (params_in.do_map) {
          map_im( w_1 , i_1 );
          map_im( w_2 , i_2 );
          map_im( w_3 , i_3 );
          map_im( w_4 , i_4 );
          map_im( w_5 , i_5 );
          convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
        }
        qL = -0.01666666666666666667_fp*(3*s(0)-17*s(1)+43*s(2)-77*s(3)-12*s(4))*w_1+0.01666666666666666667_fp*(2*s(1)-13*s(2)+47*s(3)+27*s(4)-3*s(5))*w_2-0.01666666666666666667_fp*(3*s(2)-27*s(3)-47*s(4)+13*s(5)-2*s(6))*w_3+0.01666666666666666667_fp*(12*s(3)+77*s(4)-43*s(5)+17*s(6)-3*s(7))*w_4+0.01666666666666666667_fp*(137*s(4)-163*s(5)+137*s(6)-63*s(7)+12*s(8))*w_5;
      }
      // Right evaluation
      {
        float i_1 = 0.007936507936507936508f;
        float i_2 =   0.1587301587301587302f;
        float i_3 =   0.4761904761904761905f;
        float i_4 =   0.3174603174603174603f;
        float i_5 =  0.03968253968253968254f;
        float w_1 = i_1 / (TV_1*TV_1 + 1.e-20f);
        float w_2 = i_2 / (TV_2*TV_2 + 1.e-20f);
        float w_3 = i_3 / (TV_3*TV_3 + 1.e-20f);
        float w_4 = i_4 / (TV_4*TV_4 + 1.e-20f);
        float w_5 = i_5 / (TV_5*TV_5 + 1.e-20f);
        convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
        if (params_in.do_map) {
          map_im( w_1 , i_1 );
          map_im( w_2 , i_2 );
          map_im( w_3 , i_3 );
          map_im( w_4 , i_4 );
          map_im( w_5 , i_5 );
          convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
        }
        qR = 0.01666666666666666667_fp*(12*s(0)-63*s(1)+137*s(2)-163*s(3)+137*s(4))*w_1-0.01666666666666666667_fp*(3*s(1)-17*s(2)+43*s(3)-77*s(4)-12*s(5))*w_2+0.01666666666666666667_fp*(2*s(2)-13*s(3)+47*s(4)+27*s(5)-3*s(6))*w_3-0.01666666666666666667_fp*(3*s(3)-27*s(4)-47*s(5)+13*s(6)-2*s(7))*w_4+0.01666666666666666667_fp*(12*s(4)+77*s(5)-43*s(6)+17*s(7)-3*s(8))*w_5;
      }
    }

    YAKL_INLINE static void compute_non_limited_edges( SArray<real,1,9> const &s , real &qL , real &qR ) {
      qL = -0.001984126984126984127_fp*s(0)+0.02182539682539682540_fp*s(1)-0.1210317460317460317_fp*s(2)+0.5456349206349206349_fp*s(3)+0.7456349206349206349_fp*s(4)-0.2543650793650793651_fp*s(5)+0.07896825396825396825_fp*s(6)-0.01626984126984126984_fp*s(7)+0.001587301587301587302_fp*s(8);
      qR = 0.001587301587301587302_fp*s(0)-0.01626984126984126984_fp*s(1)+0.07896825396825396825_fp*s(2)-0.2543650793650793651_fp*s(3)+0.7456349206349206349_fp*s(4)+0.5456349206349206349_fp*s(5)-0.1210317460317460317_fp*s(6)+0.02182539682539682540_fp*s(7)-0.001984126984126984127_fp*s(8);
    }
  };

}


