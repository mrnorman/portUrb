
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



  template <class real, int ord> struct WenoLimiter;



  template <class real>
  YAKL_INLINE void static map( real &w , real d ) { w = w*(d+d*d-3*d*w+w*w)/(d*d+(1-2*d)*w); }


  template <class real>
  YAKL_INLINE void static map_im( real &w , real d , int k=2, real A=static_cast<real>(0.1) ) {
    real term = w-d;
    for (int i=1; i<k; i++) { term *= w-d; }
    w = d + (A*term*(w-d))/(A*term+w*(1-w));
  }


  template <class real>
  YAKL_INLINE void static map_rs( real &w , real d , int k=6 , int m=3 , real s=static_cast<real>(2000.) ) {
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
    if (mx-mn > static_cast<real>(1.e-20)) scale = mx-mn;
    for (int i=0; i < N; i++) { s(i) = (s(i) - mn) / scale; }
  }



  template <class real> struct WenoLimiter<real,1> {
    struct Params { bool do_map; };
    Params params;
    void set_params( bool do_map = true ) { params.do_map = do_map; }
    YAKL_INLINE WenoLimiter() { }
    YAKL_INLINE static void compute_limited_edges( SArray<real,1,1> const &s         ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      qL = s(1);
      qR = s(1);
    }
  };



  template <class real> struct WenoLimiter<real,3> {
    real static constexpr eps = static_cast<real>(1.e-20);
    struct Params { bool do_map; bool imm_L; bool imm_R; };
    Params params;

    void set_params( bool do_map = false , bool imm_L = false , bool imm_R = false ) {
      params.do_map = do_map;
      params.imm_L  = imm_L; 
      params.imm_R  = imm_R;
    }

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
      if (params_in.imm_L) { imm_L_alter(TV_L,TV_R); }
      if (params_in.imm_R) { imm_R_alter(TV_L,TV_R); }
      qL = gll_1(TV_L,TV_R,params_in.do_map,s);
      qR = gll_2(TV_L,TV_R,params_in.do_map,s);
    }

    YAKL_INLINE static real gll_1(real TV_L, real TV_R, bool do_map, SArray<real,1,3> const &s) {
      real i_L = static_cast<real>(0.66666666666666666666666666666666666667);
      real i_R = static_cast<real>(0.33333333333333333333333333333333333333);
      real w_L = i_L / (TV_L*TV_L + eps);
      real w_R = i_R / (TV_R*TV_R + eps);
      convexify( w_L , w_R );
      if (do_map) {
        map( w_L , i_L );
        map( w_R , i_R );
        convexify( w_L , w_R );
      }
      return (static_cast<real>(0.50000000000000000000000000000000000000)*s(0)+static_cast<real>(0.50000000000000000000000000000000000000)*s(1))*w_L+(static_cast<real>(1.5000000000000000000000000000000000000)*s(1)-static_cast<real>(0.50000000000000000000000000000000000000)*s(2))*w_R;
    }

    YAKL_INLINE static real gll_2(real TV_L, real TV_R, bool do_map, SArray<real,1,3> const &s) {
      real i_L = static_cast<real>(0.33333333333333333333333333333333333333);
      real i_R = static_cast<real>(0.66666666666666666666666666666666666666);
      real w_L = i_L / (TV_L*TV_L + eps);
      real w_R = i_R / (TV_R*TV_R + eps);
      convexify( w_L , w_R );
      if (do_map) {
        map( w_L , i_L );
        map( w_R , i_R );
        convexify( w_L , w_R );
      }
      return (-static_cast<real>(0.50000000000000000000000000000000000000)*s(0)+static_cast<real>(1.5000000000000000000000000000000000000)*s(1))*w_L+(static_cast<real>(0.50000000000000000000000000000000000000)*s(1)+static_cast<real>(0.50000000000000000000000000000000000000)*s(2))*w_R;
    }

    // Don't allow the stencil that doesn't contain the left interface to dominate
    YAKL_INLINE static void imm_L_alter( real &TV_1 , real &TV_2 ) {
      TV_2 = std::max(TV_1,TV_2);
    }

    // Don't allow the stencil that doesn't contain the right interface to dominate
    YAKL_INLINE static void imm_R_alter( real &TV_1 , real &TV_2 ) {
      TV_1 = std::max(TV_1,TV_2);
    }

  };



  template <class real> struct WenoLimiter<real,5> {
    real static constexpr eps = static_cast<real>(1.e-20);
    struct Params { bool do_map; bool imm_L; bool imm_R; };
    Params params;

    void set_params( bool do_map = false , bool imm_L = false , bool imm_R = false ) {
      params.do_map = do_map;
      params.imm_L  = imm_L; 
      params.imm_R  = imm_R;
    }

    YAKL_INLINE static void compute_limited_edges( SArray<real,1,5> const &s         ,
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
      if (params_in.imm_L) { imm_L_alter(TV_L,TV_C,TV_R); }
      if (params_in.imm_R) { imm_R_alter(TV_L,TV_C,TV_R); }
      qL = gll_1(TV_L,TV_C,TV_R,params_in.do_map,s);
      qR = gll_2(TV_L,TV_C,TV_R,params_in.do_map,s);
    }

    YAKL_INLINE static real gll_1(real TV_L, real TV_C, real TV_R, bool do_map, SArray<real,1,5> const &s) {
      real i_L = static_cast<real>(0.30000000000000000000000000000000000001);
      real i_C = static_cast<real>(0.60000000000000000000000000000000000001);
      real i_R = static_cast<real>(0.10000000000000000000000000000000000000);
      real w_L = i_L / (TV_L*TV_L + static_cast<real>(1.e-20));
      real w_C = i_C / (TV_C*TV_C + static_cast<real>(1.e-20));
      real w_R = i_R / (TV_R*TV_R + static_cast<real>(1.e-20));
      convexify( w_L , w_C , w_R );
      if (do_map) {
        map_rs( w_L , i_L );
        map_rs( w_C , i_C );
        map_rs( w_R , i_R );
        convexify( w_L , w_C , w_R );
      }
      return (static_cast<real>(0.33333333333333333333333333333333333333)*s(1)+static_cast<real>(0.83333333333333333333333333333333333333)*s(2)-static_cast<real>(0.16666666666666666666666666666666666667)*s(3))*w_C+(-static_cast<real>(0.16666666666666666666666666666666666667)*s(0)+static_cast<real>(0.83333333333333333333333333333333333333)*s(1)+static_cast<real>(0.33333333333333333333333333333333333333)*s(2))*w_L+(static_cast<real>(1.8333333333333333333333333333333333333)*s(2)-static_cast<real>(1.1666666666666666666666666666666666667)*s(3)+static_cast<real>(0.33333333333333333333333333333333333333)*s(4))*w_R;
    }

    YAKL_INLINE static real gll_2(real TV_L, real TV_C, real TV_R, bool do_map, SArray<real,1,5> const &s) {
      real i_L = static_cast<real>(0.10000000000000000000000000000000000001);
      real i_C = static_cast<real>(0.60000000000000000000000000000000000000);
      real i_R = static_cast<real>(0.30000000000000000000000000000000000000);
      real w_L = i_L / (TV_L*TV_L + static_cast<real>(1.e-20));
      real w_C = i_C / (TV_C*TV_C + static_cast<real>(1.e-20));
      real w_R = i_R / (TV_R*TV_R + static_cast<real>(1.e-20));
      convexify( w_L , w_C , w_R );
      if (do_map) {
        map_rs( w_L , i_L );
        map_rs( w_C , i_C );
        map_rs( w_R , i_R );
        convexify( w_L , w_C , w_R );
      }
      return (-static_cast<real>(0.16666666666666666666666666666666666667)*s(1)+static_cast<real>(0.83333333333333333333333333333333333333)*s(2)+static_cast<real>(0.33333333333333333333333333333333333333)*s(3))*w_C+(static_cast<real>(0.33333333333333333333333333333333333333)*s(0)-static_cast<real>(1.1666666666666666666666666666666666667)*s(1)+static_cast<real>(1.8333333333333333333333333333333333333)*s(2))*w_L+(static_cast<real>(0.33333333333333333333333333333333333333)*s(2)+static_cast<real>(0.83333333333333333333333333333333333333)*s(3)-static_cast<real>(0.16666666666666666666666666666666666667)*s(4))*w_R;
    }

    // Don't allow the stencil that doesn't contain the left interface to dominate
    YAKL_INLINE static void imm_L_alter( real &TV_1 , real &TV_2 , real &TV_3 ) {
      TV_3 = std::max(std::max(TV_1,TV_2),TV_3);
    }

    // Don't allow the stencil that doesn't contain the right interface to dominate
    YAKL_INLINE static void imm_R_alter( real &TV_1 , real &TV_2 , real &TV_3 ) {
      TV_1 = std::max(std::max(TV_1,TV_2),TV_3);
    }

  };



  template <class real> struct WenoLimiter<real,7> {
    real static constexpr eps = static_cast<real>(1.e-20);
    struct Params { bool do_map; bool imm_L; bool imm_R; };
    Params params;

    void set_params( bool do_map = false , bool imm_L = false , bool imm_R = false ) {
      params.do_map = do_map;
      params.imm_L  = imm_L; 
      params.imm_R  = imm_R;
    }

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
      if (params_in.imm_L) { imm_L_alter(TV_1,TV_2,TV_3,TV_4); }
      if (params_in.imm_R) { imm_R_alter(TV_1,TV_2,TV_3,TV_4); }
      qL = gll_1(TV_1,TV_2,TV_3,TV_4,params_in.do_map,s);
      qR = gll_2(TV_1,TV_2,TV_3,TV_4,params_in.do_map,s);
    }

    YAKL_INLINE static real gll_1(real TV_1, real TV_2, real TV_3, real TV_4, bool do_map, SArray<real,1,7> const &s) {
      real i_1 = static_cast<real>( 0.11428571428571428571428571428571428574);
      real i_2 = static_cast<real>( 0.51428571428571428571428571428571428564);
      real i_3 = static_cast<real>( 0.34285714285714285714285714285714285724);
      real i_4 = static_cast<real>(0.028571428571428571428571428571428571427);
      real w_1 = i_1 / (TV_1*TV_1 + static_cast<real>(1.e-20));
      real w_2 = i_2 / (TV_2*TV_2 + static_cast<real>(1.e-20));
      real w_3 = i_3 / (TV_3*TV_3 + static_cast<real>(1.e-20));
      real w_4 = i_4 / (TV_4*TV_4 + static_cast<real>(1.e-20));
      convexify( w_1 , w_2 , w_3 , w_4 );
      if (do_map) {
        map_rs( w_1 , i_1 );
        map_rs( w_2 , i_2 );
        map_rs( w_3 , i_3 );
        map_rs( w_4 , i_4 );
        convexify( w_1 , w_2 , w_3 , w_4 );
      }
      return (static_cast<real>(0.083333333333333333333333333333333333333)*s(0)-static_cast<real>(0.41666666666666666666666666666666666667)*s(1)+static_cast<real>(1.0833333333333333333333333333333333333)*s(2)+static_cast<real>(0.25000000000000000000000000000000000000)*s(3))*w_1+(-static_cast<real>(0.083333333333333333333333333333333333333)*s(1)+static_cast<real>(0.58333333333333333333333333333333333333)*s(2)+static_cast<real>(0.58333333333333333333333333333333333333)*s(3)-static_cast<real>(0.083333333333333333333333333333333333333)*s(4))*w_2+(static_cast<real>(0.25000000000000000000000000000000000000)*s(2)+static_cast<real>(1.0833333333333333333333333333333333333)*s(3)-static_cast<real>(0.41666666666666666666666666666666666667)*s(4)+static_cast<real>(0.083333333333333333333333333333333333333)*s(5))*w_3+(static_cast<real>(2.0833333333333333333333333333333333333)*s(3)-static_cast<real>(1.9166666666666666666666666666666666667)*s(4)+static_cast<real>(1.0833333333333333333333333333333333333)*s(5)-static_cast<real>(0.25000000000000000000000000000000000000)*s(6))*w_4;
    }

    YAKL_INLINE static real gll_2(real TV_1, real TV_2, real TV_3, real TV_4, bool do_map, SArray<real,1,7> const &s) {
      real i_1 = static_cast<real>(0.028571428571428571428571428571428571410);
      real i_2 = static_cast<real>( 0.34285714285714285714285714285714285721);
      real i_3 = static_cast<real>( 0.51428571428571428571428571428571428551);
      real i_4 = static_cast<real>( 0.11428571428571428571428571428571428575);
      real w_1 = i_1 / (TV_1*TV_1 + static_cast<real>(1.e-20));
      real w_2 = i_2 / (TV_2*TV_2 + static_cast<real>(1.e-20));
      real w_3 = i_3 / (TV_3*TV_3 + static_cast<real>(1.e-20));
      real w_4 = i_4 / (TV_4*TV_4 + static_cast<real>(1.e-20));
      convexify( w_1 , w_2 , w_3 , w_4 );
      if (do_map) {
        map_rs( w_1 , i_1 );
        map_rs( w_2 , i_2 );
        map_rs( w_3 , i_3 );
        map_rs( w_4 , i_4 );
        convexify( w_1 , w_2 , w_3 , w_4 );
      }
      return (-static_cast<real>(0.25000000000000000000000000000000000000)*s(0)+static_cast<real>(1.0833333333333333333333333333333333333)*s(1)-static_cast<real>(1.9166666666666666666666666666666666667)*s(2)+static_cast<real>(2.0833333333333333333333333333333333333)*s(3))*w_1+(static_cast<real>(0.083333333333333333333333333333333333333)*s(1)-static_cast<real>(0.41666666666666666666666666666666666667)*s(2)+static_cast<real>(1.0833333333333333333333333333333333333)*s(3)+static_cast<real>(0.25000000000000000000000000000000000000)*s(4))*w_2+(-static_cast<real>(0.083333333333333333333333333333333333333)*s(2)+static_cast<real>(0.58333333333333333333333333333333333333)*s(3)+static_cast<real>(0.58333333333333333333333333333333333333)*s(4)-static_cast<real>(0.083333333333333333333333333333333333333)*s(5))*w_3+(static_cast<real>(0.25000000000000000000000000000000000000)*s(3)+static_cast<real>(1.0833333333333333333333333333333333333)*s(4)-static_cast<real>(0.41666666666666666666666666666666666667)*s(5)+static_cast<real>(0.083333333333333333333333333333333333333)*s(6))*w_4;
    }

    // Don't allow the stencil that doesn't contain the left interface to dominate
    YAKL_INLINE static void imm_L_alter( real &TV_1 , real &TV_2 , real &TV_3 , real &TV_4 ) {
      TV_4 = std::max(std::max(std::max(TV_1,TV_2),TV_3),TV_4);
    }

    // Don't allow the stencil that doesn't contain the right interface to dominate
    YAKL_INLINE static void imm_R_alter( real &TV_1 , real &TV_2 , real &TV_3 , real &TV_4 ) {
      TV_1 = std::max(std::max(std::max(TV_1,TV_2),TV_3),TV_4);
    }
  };



  template <class real> struct WenoLimiter<real,9> {
    struct Params { bool do_map; bool imm_L; bool imm_R; };
    Params params;

    void set_params( bool do_map = false , bool imm_L = false , bool imm_R = false ) {
      params.do_map = do_map;
      params.imm_L  = imm_L; 
      params.imm_R  = imm_R;
    }

    YAKL_INLINE static void compute_limited_edges( SArray<real,1,9>       &s         ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      SArray<real,1,5> coefs_1, coefs_2, coefs_3, coefs_4, coefs_5;
      TransformMatrices::coefs5_shift1( coefs_1 , s(0) , s(1) , s(2) , s(3) , s(4) );
      TransformMatrices::coefs5_shift2( coefs_2 , s(1) , s(2) , s(3) , s(4) , s(5) );
      TransformMatrices::coefs5_shift3( coefs_3 , s(2) , s(3) , s(4) , s(5) , s(6) );
      TransformMatrices::coefs5_shift4( coefs_4 , s(3) , s(4) , s(5) , s(6) , s(7) );
      TransformMatrices::coefs5_shift5( coefs_5 , s(4) , s(5) , s(6) , s(7) , s(8) );
      // Compute TV
      real TV_1 = TransformMatrices::coefs_to_tv( coefs_1 );
      real TV_2 = TransformMatrices::coefs_to_tv( coefs_2 );
      real TV_3 = TransformMatrices::coefs_to_tv( coefs_3 );
      real TV_4 = TransformMatrices::coefs_to_tv( coefs_4 );
      real TV_5 = TransformMatrices::coefs_to_tv( coefs_5 );
      convexify(TV_1,TV_2,TV_3,TV_4,TV_5);
      if (params_in.imm_L) { imm_L_alter(TV_1,TV_2,TV_3,TV_4,TV_5); }
      if (params_in.imm_R) { imm_R_alter(TV_1,TV_2,TV_3,TV_4,TV_5); }
      qL = gll_1(TV_1,TV_2,TV_3,TV_4,TV_5,params_in.do_map,s);
      qR = gll_2(TV_1,TV_2,TV_3,TV_4,TV_5,params_in.do_map,s);
    }

    YAKL_INLINE static real gll_1(real TV_1, real TV_2, real TV_3, real TV_4, real TV_5, bool do_map, SArray<real,1,9> const &s) {
      real i_1 = static_cast<real>( 0.039682539682539682539682539682539682577);
      real i_2 = static_cast<real>(  0.31746031746031746031746031746031746117);
      real i_3 = static_cast<real>(  0.47619047619047619047619047619047618949);
      real i_4 = static_cast<real>(  0.15873015873015873015873015873015872971);
      real i_5 = static_cast<real>(0.0079365079365079365079365079365079365201);
      real w_1 = i_1 / (TV_1*TV_1 + static_cast<real>(1.e-20));
      real w_2 = i_2 / (TV_2*TV_2 + static_cast<real>(1.e-20));
      real w_3 = i_3 / (TV_3*TV_3 + static_cast<real>(1.e-20));
      real w_4 = i_4 / (TV_4*TV_4 + static_cast<real>(1.e-20));
      real w_5 = i_5 / (TV_5*TV_5 + static_cast<real>(1.e-20));
      convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
      if (do_map) {
        map_rs( w_1 , i_1 );
        map_rs( w_2 , i_2 );
        map_rs( w_3 , i_3 );
        map_rs( w_4 , i_4 );
        map_rs( w_5 , i_5 );
        convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
      }
      return (-static_cast<real>(0.050000000000000000000000000000000000000)*s(0)+static_cast<real>(0.28333333333333333333333333333333333333)*s(1)-static_cast<real>(0.71666666666666666666666666666666666667)*s(2)+static_cast<real>(1.2833333333333333333333333333333333333)*s(3)+static_cast<real>(0.20000000000000000000000000000000000000)*s(4))*w_1+(static_cast<real>(0.033333333333333333333333333333333333333)*s(1)-static_cast<real>(0.21666666666666666666666666666666666667)*s(2)+static_cast<real>(0.78333333333333333333333333333333333333)*s(3)+static_cast<real>(0.45000000000000000000000000000000000000)*s(4)-static_cast<real>(0.050000000000000000000000000000000000000)*s(5))*w_2+(-static_cast<real>(0.050000000000000000000000000000000000000)*s(2)+static_cast<real>(0.45000000000000000000000000000000000000)*s(3)+static_cast<real>(0.78333333333333333333333333333333333333)*s(4)-static_cast<real>(0.21666666666666666666666666666666666667)*s(5)+static_cast<real>(0.033333333333333333333333333333333333333)*s(6))*w_3+(static_cast<real>(0.20000000000000000000000000000000000000)*s(3)+static_cast<real>(1.2833333333333333333333333333333333333)*s(4)-static_cast<real>(0.71666666666666666666666666666666666667)*s(5)+static_cast<real>(0.28333333333333333333333333333333333333)*s(6)-static_cast<real>(0.050000000000000000000000000000000000000)*s(7))*w_4+(static_cast<real>(2.2833333333333333333333333333333333333)*s(4)-static_cast<real>(2.7166666666666666666666666666666666667)*s(5)+static_cast<real>(2.2833333333333333333333333333333333333)*s(6)-static_cast<real>(1.0500000000000000000000000000000000000)*s(7)+static_cast<real>(0.20000000000000000000000000000000000000)*s(8))*w_5;
    }

    YAKL_INLINE static real gll_2(real TV_1, real TV_2, real TV_3, real TV_4, real TV_5, bool do_map, SArray<real,1,9> const &s) {
      real i_1 = static_cast<real>(0.0079365079365079365079365079365079363972);
      real i_2 = static_cast<real>(  0.15873015873015873015873015873015873070);
      real i_3 = static_cast<real>(  0.47619047619047619047619047619047618996);
      real i_4 = static_cast<real>(  0.31746031746031746031746031746031746110);
      real i_5 = static_cast<real>( 0.039682539682539682539682539682539682256);
      real w_1 = i_1 / (TV_1*TV_1 + static_cast<real>(1.e-20));
      real w_2 = i_2 / (TV_2*TV_2 + static_cast<real>(1.e-20));
      real w_3 = i_3 / (TV_3*TV_3 + static_cast<real>(1.e-20));
      real w_4 = i_4 / (TV_4*TV_4 + static_cast<real>(1.e-20));
      real w_5 = i_5 / (TV_5*TV_5 + static_cast<real>(1.e-20));
      convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
      if (do_map) {
        map_rs( w_1 , i_1 );
        map_rs( w_2 , i_2 );
        map_rs( w_3 , i_3 );
        map_rs( w_4 , i_4 );
        map_rs( w_5 , i_5 );
        convexify( w_1 , w_2 , w_3 , w_4 , w_5 );
      }
      return (static_cast<real>(0.20000000000000000000000000000000000000)*s(0)-static_cast<real>(1.0500000000000000000000000000000000000)*s(1)+static_cast<real>(2.2833333333333333333333333333333333333)*s(2)-static_cast<real>(2.7166666666666666666666666666666666667)*s(3)+static_cast<real>(2.2833333333333333333333333333333333333)*s(4))*w_1+(-static_cast<real>(0.050000000000000000000000000000000000000)*s(1)+static_cast<real>(0.28333333333333333333333333333333333333)*s(2)-static_cast<real>(0.71666666666666666666666666666666666667)*s(3)+static_cast<real>(1.2833333333333333333333333333333333333)*s(4)+static_cast<real>(0.20000000000000000000000000000000000000)*s(5))*w_2+(static_cast<real>(0.033333333333333333333333333333333333333)*s(2)-static_cast<real>(0.21666666666666666666666666666666666667)*s(3)+static_cast<real>(0.78333333333333333333333333333333333333)*s(4)+static_cast<real>(0.45000000000000000000000000000000000000)*s(5)-static_cast<real>(0.050000000000000000000000000000000000000)*s(6))*w_3+(-static_cast<real>(0.050000000000000000000000000000000000000)*s(3)+static_cast<real>(0.45000000000000000000000000000000000000)*s(4)+static_cast<real>(0.78333333333333333333333333333333333333)*s(5)-static_cast<real>(0.21666666666666666666666666666666666667)*s(6)+static_cast<real>(0.033333333333333333333333333333333333333)*s(7))*w_4+(static_cast<real>(0.20000000000000000000000000000000000000)*s(4)+static_cast<real>(1.2833333333333333333333333333333333333)*s(5)-static_cast<real>(0.71666666666666666666666666666666666667)*s(6)+static_cast<real>(0.28333333333333333333333333333333333333)*s(7)-static_cast<real>(0.050000000000000000000000000000000000000)*s(8))*w_5;
    }

    // Don't allow the stencil that doesn't contain the left interface to dominate
    YAKL_INLINE static void imm_L_alter( real &TV_1 , real &TV_2 , real &TV_3 , real &TV_4 , real &TV_5 ) {
      TV_5 = std::max(std::max(std::max(std::max(TV_1,TV_2),TV_3),TV_4),TV_5);
    }

    // Don't allow the stencil that doesn't contain the right interface to dominate
    YAKL_INLINE static void imm_R_alter( real &TV_1 , real &TV_2 , real &TV_3 , real &TV_4 , real &TV_5 ) {
      TV_1 = std::max(std::max(std::max(std::max(TV_1,TV_2),TV_3),TV_4),TV_5);
    }
  };



  template <class real> struct WenoLimiter<real,11> {
    struct Params { bool do_map; bool imm_L; bool imm_R; };
    Params params;

    void set_params( bool do_map = false , bool imm_L = false , bool imm_R = false ) {
      params.do_map = do_map;
      params.imm_L  = imm_L; 
      params.imm_R  = imm_R;
    }

    YAKL_INLINE static void compute_limited_edges( SArray<real,1,11>       &s         ,
                                                   real                   &qL        ,
                                                   real                   &qR        ,
                                                   Params           const &params_in ) {
      SArray<real,1,6> coefs_1, coefs_2, coefs_3, coefs_4, coefs_5, coefs_6;
      TransformMatrices::coefs6_shift1( coefs_1 , s(0) , s(1) , s(2) , s(3) , s(4) , s( 5) );
      TransformMatrices::coefs6_shift2( coefs_2 , s(1) , s(2) , s(3) , s(4) , s(5) , s( 6) );
      TransformMatrices::coefs6_shift3( coefs_3 , s(2) , s(3) , s(4) , s(5) , s(6) , s( 7) );
      TransformMatrices::coefs6_shift4( coefs_4 , s(3) , s(4) , s(5) , s(6) , s(7) , s( 8) );
      TransformMatrices::coefs6_shift5( coefs_5 , s(4) , s(5) , s(6) , s(7) , s(8) , s( 9) );
      TransformMatrices::coefs6_shift6( coefs_6 , s(5) , s(6) , s(7) , s(8) , s(9) , s(10) );
      // Compute TV
      real TV_1 = TransformMatrices::coefs_to_tv( coefs_1 );
      real TV_2 = TransformMatrices::coefs_to_tv( coefs_2 );
      real TV_3 = TransformMatrices::coefs_to_tv( coefs_3 );
      real TV_4 = TransformMatrices::coefs_to_tv( coefs_4 );
      real TV_5 = TransformMatrices::coefs_to_tv( coefs_5 );
      real TV_6 = TransformMatrices::coefs_to_tv( coefs_6 );
      convexify(TV_1,TV_2,TV_3,TV_4,TV_5,TV_6);
      if (params_in.imm_L) { imm_L_alter(TV_1,TV_2,TV_3,TV_4,TV_5,TV_6); }
      if (params_in.imm_R) { imm_R_alter(TV_1,TV_2,TV_3,TV_4,TV_5,TV_6); }
      qL = gll_1(TV_1,TV_2,TV_3,TV_4,TV_5,TV_6,params_in.do_map,s);
      qR = gll_2(TV_1,TV_2,TV_3,TV_4,TV_5,TV_6,params_in.do_map,s);
    }

    YAKL_INLINE static real gll_1(real TV_1, real TV_2, real TV_3, real TV_4, real TV_5, real TV_6, bool do_map, SArray<real,1,11> const &s) {
      real i_1 = static_cast<real>( 0.012987012987012987012987012987012987729);
      real i_2 = static_cast<real>(  0.16233766233766233766233766233766233499);
      real i_3 = static_cast<real>(  0.43290043290043290043290043290043289902);
      real i_4 = static_cast<real>(  0.32467532467532467532467532467532467340);
      real i_5 = static_cast<real>( 0.064935064935064935064935064935064937333);
      real i_6 = static_cast<real>(0.0021645021645021645021645021645021643707);
      real w_1 = i_1 / (TV_1*TV_1 + static_cast<real>(1.e-20));
      real w_2 = i_2 / (TV_2*TV_2 + static_cast<real>(1.e-20));
      real w_3 = i_3 / (TV_3*TV_3 + static_cast<real>(1.e-20));
      real w_4 = i_4 / (TV_4*TV_4 + static_cast<real>(1.e-20));
      real w_5 = i_5 / (TV_5*TV_5 + static_cast<real>(1.e-20));
      real w_6 = i_6 / (TV_6*TV_6 + static_cast<real>(1.e-20));
      convexify( w_1 , w_2 , w_3 , w_4 , w_5 , w_6 );
      if (do_map) {
        map_rs( w_1 , i_1 );
        map_rs( w_2 , i_2 );
        map_rs( w_3 , i_3 );
        map_rs( w_4 , i_4 );
        map_rs( w_5 , i_5 );
        map_rs( w_6 , i_6 );
        convexify( w_1 , w_2 , w_3 , w_4 , w_5 , w_6 );
      }
      return (static_cast<real>(0.033333333333333333333333333333333333333)*s(0)-static_cast<real>(0.21666666666666666666666666666666666667)*s(1)+static_cast<real>(0.61666666666666666666666666666666666667)*s(2)-static_cast<real>(1.0500000000000000000000000000000000000)*s(3)+static_cast<real>(1.4500000000000000000000000000000000000)*s(4)+static_cast<real>(0.16666666666666666666666666666666666667)*s(5))*w_1+(-static_cast<real>(0.016666666666666666666666666666666666667)*s(1)+static_cast<real>(0.11666666666666666666666666666666666667)*s(2)-static_cast<real>(0.38333333333333333333333333333333333333)*s(3)+static_cast<real>(0.95000000000000000000000000000000000000)*s(4)+static_cast<real>(0.36666666666666666666666666666666666667)*s(5)-static_cast<real>(0.033333333333333333333333333333333333333)*s(6))*w_2+(static_cast<real>(0.016666666666666666666666666666666666667)*s(2)-static_cast<real>(0.13333333333333333333333333333333333333)*s(3)+static_cast<real>(0.61666666666666666666666666666666666667)*s(4)+static_cast<real>(0.61666666666666666666666666666666666667)*s(5)-static_cast<real>(0.13333333333333333333333333333333333333)*s(6)+static_cast<real>(0.016666666666666666666666666666666666667)*s(7))*w_3+(-static_cast<real>(0.033333333333333333333333333333333333333)*s(3)+static_cast<real>(0.36666666666666666666666666666666666667)*s(4)+static_cast<real>(0.95000000000000000000000000000000000000)*s(5)-static_cast<real>(0.38333333333333333333333333333333333333)*s(6)+static_cast<real>(0.11666666666666666666666666666666666667)*s(7)-static_cast<real>(0.016666666666666666666666666666666666667)*s(8))*w_4+(static_cast<real>(0.16666666666666666666666666666666666667)*s(4)+static_cast<real>(1.4500000000000000000000000000000000000)*s(5)-static_cast<real>(1.0500000000000000000000000000000000000)*s(6)+static_cast<real>(0.61666666666666666666666666666666666667)*s(7)-static_cast<real>(0.21666666666666666666666666666666666667)*s(8)+static_cast<real>(0.033333333333333333333333333333333333333)*s(9))*w_5+(-static_cast<real>(0.16666666666666666666666666666666666667)*s(10)+static_cast<real>(2.4500000000000000000000000000000000000)*s(5)-static_cast<real>(3.5500000000000000000000000000000000000)*s(6)+static_cast<real>(3.9500000000000000000000000000000000000)*s(7)-static_cast<real>(2.7166666666666666666666666666666666667)*s(8)+static_cast<real>(1.0333333333333333333333333333333333333)*s(9))*w_6;
    }

    YAKL_INLINE static real gll_2(real TV_1, real TV_2, real TV_3, real TV_4, real TV_5, real TV_6, bool do_map, SArray<real,1,11> const &s) {
      real i_1 = static_cast<real>(0.0021645021645021645021645021645021643722);
      real i_2 = static_cast<real>( 0.064935064935064935064935064935064936585);
      real i_3 = static_cast<real>(  0.32467532467532467532467532467532466353);
      real i_4 = static_cast<real>(  0.43290043290043290043290043290043291990);
      real i_5 = static_cast<real>(  0.16233766233766233766233766233766232663);
      real i_6 = static_cast<real>( 0.012987012987012987012987012987012988474);
      real w_1 = i_1 / (TV_1*TV_1 + static_cast<real>(1.e-20));
      real w_2 = i_2 / (TV_2*TV_2 + static_cast<real>(1.e-20));
      real w_3 = i_3 / (TV_3*TV_3 + static_cast<real>(1.e-20));
      real w_4 = i_4 / (TV_4*TV_4 + static_cast<real>(1.e-20));
      real w_5 = i_5 / (TV_5*TV_5 + static_cast<real>(1.e-20));
      real w_6 = i_6 / (TV_6*TV_6 + static_cast<real>(1.e-20));
      convexify( w_1 , w_2 , w_3 , w_4 , w_5 , w_6 );
      if (do_map) {
        map_rs( w_1 , i_1 );
        map_rs( w_2 , i_2 );
        map_rs( w_3 , i_3 );
        map_rs( w_4 , i_4 );
        map_rs( w_5 , i_5 );
        map_rs( w_6 , i_6 );
        convexify( w_1 , w_2 , w_3 , w_4 , w_5 , w_6 );
      }
      return (-static_cast<real>(0.16666666666666666666666666666666666667)*s(0)+static_cast<real>(1.0333333333333333333333333333333333333)*s(1)-static_cast<real>(2.7166666666666666666666666666666666667)*s(2)+static_cast<real>(3.9500000000000000000000000000000000000)*s(3)-static_cast<real>(3.5500000000000000000000000000000000000)*s(4)+static_cast<real>(2.4500000000000000000000000000000000000)*s(5))*w_1+(static_cast<real>(0.033333333333333333333333333333333333333)*s(1)-static_cast<real>(0.21666666666666666666666666666666666667)*s(2)+static_cast<real>(0.61666666666666666666666666666666666667)*s(3)-static_cast<real>(1.0500000000000000000000000000000000000)*s(4)+static_cast<real>(1.4500000000000000000000000000000000000)*s(5)+static_cast<real>(0.16666666666666666666666666666666666667)*s(6))*w_2+(-static_cast<real>(0.016666666666666666666666666666666666667)*s(2)+static_cast<real>(0.11666666666666666666666666666666666667)*s(3)-static_cast<real>(0.38333333333333333333333333333333333333)*s(4)+static_cast<real>(0.95000000000000000000000000000000000000)*s(5)+static_cast<real>(0.36666666666666666666666666666666666667)*s(6)-static_cast<real>(0.033333333333333333333333333333333333333)*s(7))*w_3+(static_cast<real>(0.016666666666666666666666666666666666667)*s(3)-static_cast<real>(0.13333333333333333333333333333333333333)*s(4)+static_cast<real>(0.61666666666666666666666666666666666667)*s(5)+static_cast<real>(0.61666666666666666666666666666666666667)*s(6)-static_cast<real>(0.13333333333333333333333333333333333333)*s(7)+static_cast<real>(0.016666666666666666666666666666666666667)*s(8))*w_4+(-static_cast<real>(0.033333333333333333333333333333333333333)*s(4)+static_cast<real>(0.36666666666666666666666666666666666667)*s(5)+static_cast<real>(0.95000000000000000000000000000000000000)*s(6)-static_cast<real>(0.38333333333333333333333333333333333333)*s(7)+static_cast<real>(0.11666666666666666666666666666666666667)*s(8)-static_cast<real>(0.016666666666666666666666666666666666667)*s(9))*w_5+(static_cast<real>(0.033333333333333333333333333333333333333)*s(10)+static_cast<real>(0.16666666666666666666666666666666666667)*s(5)+static_cast<real>(1.4500000000000000000000000000000000000)*s(6)-static_cast<real>(1.0500000000000000000000000000000000000)*s(7)+static_cast<real>(0.61666666666666666666666666666666666666)*s(8)-static_cast<real>(0.21666666666666666666666666666666666667)*s(9))*w_6;
    }

    // Don't allow the stencil that doesn't contain the left interface to dominate
    YAKL_INLINE static void imm_L_alter( real &TV_1 , real &TV_2 , real &TV_3 , real &TV_4 , real &TV_5 , real &TV_6 ) {
      TV_6 = std::max(std::max(std::max(std::max(std::max(TV_1,TV_2),TV_3),TV_4),TV_5),TV_6);
    }

    // Don't allow the stencil that doesn't contain the right interface to dominate
    YAKL_INLINE static void imm_R_alter( real &TV_1 , real &TV_2 , real &TV_3 , real &TV_4 , real &TV_5 , real &TV_6 ) {
      TV_1 = std::max(std::max(std::max(std::max(std::max(TV_1,TV_2),TV_3),TV_4),TV_5),TV_6);
    }
  };

}


