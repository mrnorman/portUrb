
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



  template <> struct WenoLimiter<1> {
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



  template <> struct WenoLimiter<3> {
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
      qR = gll_4(TV_L,TV_R,params_in.do_map,s);
    }

    YAKL_INLINE static void compute_limited_gll( SArray<real,1,3> const &s         ,
                                                 SArray<real,1,4>       &gll       ,
                                                 Params           const &params_in ) {
      SArray<real,1,2> coefs_L, coefs_R;
      TransformMatrices::coefs2_shift1( coefs_L , s(0) , s(1) );
      TransformMatrices::coefs2_shift2( coefs_R , s(1) , s(2) );
      real TV_L = TransformMatrices::coefs_to_tv( coefs_L );
      real TV_R = TransformMatrices::coefs_to_tv( coefs_R );
      convexify( TV_L , TV_R );
      if (params_in.imm_L) { imm_L_alter(TV_L,TV_R); }
      if (params_in.imm_R) { imm_R_alter(TV_L,TV_R); }
      gll(0) = gll_1(TV_L,TV_R,params_in.do_map,s);
      gll(1) = gll_2(TV_L,TV_R,params_in.do_map,s);
      gll(2) = gll_3(TV_L,TV_R,params_in.do_map,s);
      gll(3) = gll_4(TV_L,TV_R,params_in.do_map,s);
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
      real i_L = static_cast<real>(0.42546440075000701011969421104229079215);
      real i_R = static_cast<real>(0.57453559924999298988030578895770920783);
      real w_L = i_L / (TV_L*TV_L + eps);
      real w_R = i_R / (TV_R*TV_R + eps);
      convexify( w_L , w_R );
      if (do_map) {
        map( w_L , i_L );
        map( w_R , i_R );
        convexify( w_L , w_R );
      }
      return (static_cast<real>(0.22360679774997896964091736687312762354)*s(0)+static_cast<real>(0.77639320225002103035908263312687237646)*s(1))*w_L+(static_cast<real>(1.2236067977499789696409173668731276235)*s(1)-static_cast<real>(0.22360679774997896964091736687312762354)*s(2))*w_R;
    }

    YAKL_INLINE static real gll_3(real TV_L, real TV_R, bool do_map, SArray<real,1,3> const &s) {
      real i_L = static_cast<real>(0.57453559924999298988030578895770920785);
      real i_R = static_cast<real>(0.42546440075000701011969421104229079215);
      real w_L = i_L / (TV_L*TV_L + eps);
      real w_R = i_R / (TV_R*TV_R + eps);
      convexify( w_L , w_R );
      if (do_map) {
        map( w_L , i_L );
        map( w_R , i_R );
        convexify( w_L , w_R );
      }
      return (-static_cast<real>(0.22360679774997896964091736687312762354)*s(0)+static_cast<real>(1.2236067977499789696409173668731276235)*s(1))*w_L+(static_cast<real>(0.77639320225002103035908263312687237646)*s(1)+static_cast<real>(0.22360679774997896964091736687312762354)*s(2))*w_R;
    }

    YAKL_INLINE static real gll_4(real TV_L, real TV_R, bool do_map, SArray<real,1,3> const &s) {
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



  template <> struct WenoLimiter<5> {
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
      qR = gll_4(TV_L,TV_C,TV_R,params_in.do_map,s);
    }

    YAKL_INLINE static void compute_limited_gll( SArray<real,1,5> const &s         ,
                                                 SArray<real,1,4>       &gll       ,
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
      gll(0) = gll_1(TV_L,TV_C,TV_R,params_in.do_map,s);
      gll(1) = gll_2(TV_L,TV_C,TV_R,params_in.do_map,s);
      gll(2) = gll_3(TV_L,TV_C,TV_R,params_in.do_map,s);
      gll(3) = gll_4(TV_L,TV_C,TV_R,params_in.do_map,s);
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
      real i_L = static_cast<real>(0.16108042773295884711890326586686025883);
      real i_C = static_cast<real>(0.58636363636363636363636363636363636378);
      real i_R = static_cast<real>(0.25255593590340478924473309776950337754);
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
      return (static_cast<real>(0.095136732208322818153792016769897145105)*s(1)+static_cast<real>(1.0333333333333333333333333333333333333)*s(2)-static_cast<real>(0.12847006554165615148712535010323047844)*s(3))*w_C+(-static_cast<real>(0.12847006554165615148712535010323047844)*s(0)+static_cast<real>(0.48054692883329127261516806707958858042)*s(1)+static_cast<real>(0.64792313670836487887195728302364189802)*s(2))*w_L+(static_cast<real>(1.3187435299583017877947093836430247687)*s(2)-static_cast<real>(0.41388026216662460594850140041292191376)*s(3)+static_cast<real>(0.095136732208322818153792016769897145105)*s(4))*w_R;
    }

    YAKL_INLINE static real gll_3(real TV_L, real TV_C, real TV_R, bool do_map, SArray<real,1,5> const &s) {
      real i_L = static_cast<real>(0.25255593590340478924473309776950337749);
      real i_C = static_cast<real>(0.58636363636363636363636363636363636368);
      real i_R = static_cast<real>(0.16108042773295884711890326586686025878);
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
      return (-static_cast<real>(0.12847006554165615148712535010323047844)*s(1)+static_cast<real>(1.0333333333333333333333333333333333333)*s(2)+static_cast<real>(0.095136732208322818153792016769897145105)*s(3))*w_C+(static_cast<real>(0.095136732208322818153792016769897145105)*s(0)-static_cast<real>(0.41388026216662460594850140041292191376)*s(1)+static_cast<real>(1.3187435299583017877947093836430247687)*s(2))*w_L+(static_cast<real>(0.64792313670836487887195728302364189802)*s(2)+static_cast<real>(0.48054692883329127261516806707958858042)*s(3)-static_cast<real>(0.12847006554165615148712535010323047844)*s(4))*w_R;
    }

    YAKL_INLINE static real gll_4(real TV_L, real TV_C, real TV_R, bool do_map, SArray<real,1,5> const &s) {
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



  template <> struct WenoLimiter<7> {
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
      qR = gll_4(TV_1,TV_2,TV_3,TV_4,params_in.do_map,s);
    }

    YAKL_INLINE static void compute_limited_gll( SArray<real,1,7> const &s         ,
                                                 SArray<real,1,4>       &gll       ,
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
      gll(0) = gll_1(TV_1,TV_2,TV_3,TV_4,params_in.do_map,s);
      gll(1) = gll_2(TV_1,TV_2,TV_3,TV_4,params_in.do_map,s);
      gll(2) = gll_3(TV_1,TV_2,TV_3,TV_4,params_in.do_map,s);
      gll(3) = gll_4(TV_1,TV_2,TV_3,TV_4,params_in.do_map,s);
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
      real i_1 = static_cast<real>(0.053977600146718912936464875080264021071);
      real i_2 = static_cast<real>( 0.39814215062078559023090059667975992069);
      real i_3 = static_cast<real>( 0.44880020526893872054604175920996438985);
      real i_4 = static_cast<real>(0.099080043963556776286592769030011667848);
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
      return (static_cast<real>(0.083748705991660357558941876728604953730)*s(0)-static_cast<real>(0.37971618351663722416395098028904533963)*s(1)+static_cast<real>(0.73179304680827234529199369726540344161)*s(2)+static_cast<real>(0.56417443071670452131301540629503694429)*s(3))*w_1+(-static_cast<real>(0.044721359549995793928183473374625524709)*s(1)+static_cast<real>(0.22930081085831019993834243689377371923)*s(2)+static_cast<real>(0.89916925468334595154878291320945675920)*s(3)-static_cast<real>(0.083748705991660357558941876728604953730)*s(4))*w_2+(static_cast<real>(0.050415372658327024225608543395271620396)*s(2)+static_cast<real>(1.1674974119833207151178837534572099075)*s(3)-static_cast<real>(0.26263414419164353327167577022710705256)*s(4)+static_cast<real>(0.044721359549995793928183473374625524709)*s(5))*w_3+(static_cast<real>(1.3691589026166288120203179270382963890)*s(3)-static_cast<real>(0.56512638014160567862532703059873677495)*s(4)+static_cast<real>(0.24638285018330389083061764695571200630)*s(5)-static_cast<real>(0.050415372658327024225608543395271620396)*s(6))*w_4;
    }

    YAKL_INLINE static real gll_3(real TV_1, real TV_2, real TV_3, real TV_4, bool do_map, SArray<real,1,7> const &s) {
      real i_1 = static_cast<real>(0.099080043963556776286592769030011667841);
      real i_2 = static_cast<real>( 0.44880020526893872054604175920996439143);
      real i_3 = static_cast<real>( 0.39814215062078559023090059667975991989);
      real i_4 = static_cast<real>(0.053977600146718912936464875080264021407);
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
      return (-static_cast<real>(0.050415372658327024225608543395271620397)*s(0)+static_cast<real>(0.24638285018330389083061764695571200630)*s(1)-static_cast<real>(0.56512638014160567862532703059873677494)*s(2)+static_cast<real>(1.3691589026166288120203179270382963890)*s(3))*w_1+(static_cast<real>(0.044721359549995793928183473374625524709)*s(1)-static_cast<real>(0.26263414419164353327167577022710705256)*s(2)+static_cast<real>(1.1674974119833207151178837534572099075)*s(3)+static_cast<real>(0.050415372658327024225608543395271620396)*s(4))*w_2+(-static_cast<real>(0.083748705991660357558941876728604953730)*s(2)+static_cast<real>(0.89916925468334595154878291320945675921)*s(3)+static_cast<real>(0.22930081085831019993834243689377371923)*s(4)-static_cast<real>(0.044721359549995793928183473374625524709)*s(5))*w_3+(static_cast<real>(0.56417443071670452131301540629503694429)*s(3)+static_cast<real>(0.73179304680827234529199369726540344161)*s(4)-static_cast<real>(0.37971618351663722416395098028904533963)*s(5)+static_cast<real>(0.083748705991660357558941876728604953730)*s(6))*w_4;
    }

    YAKL_INLINE static real gll_4(real TV_1, real TV_2, real TV_3, real TV_4, bool do_map, SArray<real,1,7> const &s) {
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



  template <> struct WenoLimiter<9> {
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
      qR = gll_4(TV_1,TV_2,TV_3,TV_4,TV_5,params_in.do_map,s);
    }

    YAKL_INLINE static void compute_limited_gll( SArray<real,1,9>       &s         ,
                                                 SArray<real,1,4>       &gll       ,
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
      gll(0) = gll_1(TV_1,TV_2,TV_3,TV_4,TV_5,params_in.do_map,s);
      gll(1) = gll_2(TV_1,TV_2,TV_3,TV_4,TV_5,params_in.do_map,s);
      gll(2) = gll_3(TV_1,TV_2,TV_3,TV_4,TV_5,params_in.do_map,s);
      gll(3) = gll_4(TV_1,TV_2,TV_3,TV_4,TV_5,params_in.do_map,s);
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
      real i_1 = static_cast<real>(0.016948779263302963782914971416963657677);
      real i_2 = static_cast<real>( 0.21003604298771460454685438745558161027);
      real i_3 = static_cast<real>( 0.47509364192604416068103218941207768183);
      real i_4 = static_cast<real>( 0.26120990820048223633705865004939861202);
      real i_5 = static_cast<real>(0.036711627622456034652139801665978439661);
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
      return (-static_cast<real>(0.059721359549995793928183473374625524709)*s(0)+static_cast<real>(0.32263414419164353327167577022710705256)*s(1)-static_cast<real>(0.73804434081661198773305182053679848788)*s(2)+static_cast<real>(0.97067848500825552100472759076390554045)*s(3)+static_cast<real>(0.50445307116670872738483193292041141958)*s(4))*w_1+(static_cast<real>(0.024027346441664563630758403353979429021)*s(1)-static_cast<real>(0.14083074531665404845121708679054324079)*s(2)+static_cast<real>(0.37346488950829758172289285701765029336)*s(3)+static_cast<real>(0.80305986891668769702574929979353904312)*s(4)-static_cast<real>(0.059721359549995793928183473374625524709)*s(5))*w_2+(-static_cast<real>(0.020694013108331230297425070020646095688)*s(2)+static_cast<real>(0.13319142509165194541530882347785600315)*s(3)+static_cast<real>(1.0433333333333333333333333333333333333)*s(4)-static_cast<real>(0.17985809175831861208197549014452266981)*s(5)+static_cast<real>(0.024027346441664563630758403353979429021)*s(6))*w_3+(static_cast<real>(0.029721359549995793928183473374625524709)*s(3)+static_cast<real>(1.2502734644166456363075840335397942902)*s(4)-static_cast<real>(0.38679822284163091505622619035098362669)*s(5)+static_cast<real>(0.12749741198332071511788375345720990746)*s(6)-static_cast<real>(0.020694013108331230297425070020646095688)*s(7))*w_4+(static_cast<real>(1.3988802621666246059485014004129219138)*s(4)-static_cast<real>(0.68401181834158885433806092409723887378)*s(5)+static_cast<real>(0.42471100748327865439971848720346515455)*s(6)-static_cast<real>(0.16930081085831019993834243689377371923)*s(7)+static_cast<real>(0.029721359549995793928183473374625524709)*s(8))*w_5;
    }

    YAKL_INLINE static real gll_3(real TV_1, real TV_2, real TV_3, real TV_4, real TV_5, bool do_map, SArray<real,1,9> const &s) {
      real i_1 = static_cast<real>(0.036711627622456034652139801665978441440);
      real i_2 = static_cast<real>( 0.26120990820048223633705865004939861328);
      real i_3 = static_cast<real>( 0.47509364192604416068103218941207768994);
      real i_4 = static_cast<real>( 0.21003604298771460454685438745558160544);
      real i_5 = static_cast<real>(0.016948779263302963782914971416963658113);
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
      return (static_cast<real>(0.029721359549995793928183473374625524709)*s(0)-static_cast<real>(0.16930081085831019993834243689377371923)*s(1)+static_cast<real>(0.42471100748327865439971848720346515455)*s(2)-static_cast<real>(0.68401181834158885433806092409723887378)*s(3)+static_cast<real>(1.3988802621666246059485014004129219138)*s(4))*w_1+(-static_cast<real>(0.020694013108331230297425070020646095688)*s(1)+static_cast<real>(0.12749741198332071511788375345720990746)*s(2)-static_cast<real>(0.38679822284163091505622619035098362669)*s(3)+static_cast<real>(1.2502734644166456363075840335397942902)*s(4)+static_cast<real>(0.029721359549995793928183473374625524709)*s(5))*w_2+(static_cast<real>(0.024027346441664563630758403353979429021)*s(2)-static_cast<real>(0.17985809175831861208197549014452266981)*s(3)+static_cast<real>(1.0433333333333333333333333333333333333)*s(4)+static_cast<real>(0.13319142509165194541530882347785600315)*s(5)-static_cast<real>(0.020694013108331230297425070020646095688)*s(6))*w_3+(-static_cast<real>(0.059721359549995793928183473374625524709)*s(3)+static_cast<real>(0.80305986891668769702574929979353904313)*s(4)+static_cast<real>(0.37346488950829758172289285701765029336)*s(5)-static_cast<real>(0.14083074531665404845121708679054324079)*s(6)+static_cast<real>(0.024027346441664563630758403353979429021)*s(7))*w_4+(static_cast<real>(0.50445307116670872738483193292041141958)*s(4)+static_cast<real>(0.97067848500825552100472759076390554045)*s(5)-static_cast<real>(0.73804434081661198773305182053679848788)*s(6)+static_cast<real>(0.32263414419164353327167577022710705256)*s(7)-static_cast<real>(0.059721359549995793928183473374625524709)*s(8))*w_5;
    }

    YAKL_INLINE static real gll_4(real TV_1, real TV_2, real TV_3, real TV_4, real TV_5, bool do_map, SArray<real,1,9> const &s) {
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

}


