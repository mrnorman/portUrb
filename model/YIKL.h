
#pragma once

#define YIKL_AUTO_LABEL() (std::string(basename(__FILE__)) + std::string(":") + std::to_string(__LINE__)).c_str()

namespace yikl {

  class LBnd {
  public:
    int static constexpr default_lbound = 0;
    int l, u, s;
    KOKKOS_INLINE_FUNCTION LBnd() {
      this->l = -1;
      this->u = -1;
      this->s = -1;
    }
    KOKKOS_INLINE_FUNCTION LBnd(size_t u) {
      this->l = 0;
      this->u = u-1;
      this->s = 1;
    }
    KOKKOS_INLINE_FUNCTION LBnd(int u) {
      this->l = 0;
      this->u = u-1;
      this->s = 1;
    }
    KOKKOS_INLINE_FUNCTION LBnd(int l, int u) {
      this->l = l;
      this->u = u;
      this->s = 1;
      #ifdef KOKKOS_DEBUG
        if (u < l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound");
      #endif
    }
    KOKKOS_INLINE_FUNCTION LBnd(int l, int u, int s) {
      this->l = l;
      this->u = u;
      this->s = s;
      #ifdef KOKKOS_DEBUG
        if (u < l) Kokkos::abort("ERROR: cannot specify an upper bound < lower bound");
        if (s < 1) Kokkos::abort("ERROR: negative strides not yet supported.");
      #endif
    }
    KOKKOS_INLINE_FUNCTION bool valid() const { return this->s > 0; }
  };


  template <int N, bool simple = false> class Bounds;


  template<int N> class Bounds<N,true> {
  public:
    size_t nIter;
    size_t dims[N];
    KOKKOS_INLINE_FUNCTION Bounds( size_t b0 , size_t b1=0 , size_t b2=0 , size_t b3=0 , size_t b4=0 ,
                                               size_t b5=0 , size_t b6=0 , size_t b7=0 ) {
      if constexpr (N >= 1) dims[0] = b0;
      if constexpr (N >= 2) dims[1] = b1;
      if constexpr (N >= 3) dims[2] = b2;
      if constexpr (N >= 4) dims[3] = b3;
      if constexpr (N >= 5) dims[4] = b4;
      if constexpr (N >= 6) dims[5] = b5;
      if constexpr (N >= 7) dims[6] = b6;
      if constexpr (N >= 8) dims[7] = b7;
      #ifdef KOKKOS_DEBUG
        if (N >= 2) { if (b1 == 0) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 3) { if (b2 == 0) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 4) { if (b3 == 0) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 5) { if (b4 == 0) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 6) { if (b5 == 0) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 7) { if (b6 == 0) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 8) { if (b7 == 0) Kokkos::abort("ERROR: Too few bounds specified"); }
        int num_bounds = 1;
        if (b1 > 0) num_bounds++;
        if (b2 > 0) num_bounds++;
        if (b3 > 0) num_bounds++;
        if (b4 > 0) num_bounds++;
        if (b5 > 0) num_bounds++;
        if (b6 > 0) num_bounds++;
        if (b7 > 0) num_bounds++;
        if (num_bounds != N) Kokkos::abort("ERROR: Number of bounds passed does not match templated number of bounds.");
      #endif
      nIter = 1;
      for (int i=0; i<N; i++) { nIter *= dims[i]; }
    }
    KOKKOS_INLINE_FUNCTION int lbound(int i) const {
      #ifdef KOKKOS_DEBUG
        if (i < 0 || i > N-1) Kokkos::abort("ERROR: Calling lbound() on an out of bounds integer");
      #endif
      return 0;
    }
    KOKKOS_INLINE_FUNCTION int dim   (int i) const {
      #ifdef KOKKOS_DEBUG
        if (i < 0 || i > N-1) Kokkos::abort("ERROR: Calling dim() on an out of bounds integer");
      #endif
      return dims[i];
    }
    KOKKOS_INLINE_FUNCTION int stride(int i) const {
      #ifdef KOKKOS_DEBUG
        if (i < 0 || i > N-1) Kokkos::abort("ERROR: Calling stride() on an out of bounds integer");
      #endif
      return 1;
    }
    KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , int indices[N] ) const {
      if constexpr        (N == 1) {
        indices[0] = iGlob;
      } else if constexpr (N == 2) {
        indices[0] = iGlob/dims[1];
        indices[1] = iGlob - dims[1]*indices[0];
      } else if constexpr (N == 3) {
        size_t fac, term;
                                fac = dims[1]*dims[2]; indices[0] =  iGlob         / fac;
        term  = indices[0]*fac; fac =         dims[2]; indices[1] = (iGlob - term) / fac;
        term += indices[1]*fac;                        indices[2] =  iGlob - term;
      } else if constexpr (N == 4) {
        size_t fac, term;
                                fac = dims[1]*dims[2]*dims[3]; indices[0] =  iGlob         / fac;
        term  = indices[0]*fac; fac =         dims[2]*dims[3]; indices[1] = (iGlob - term) / fac;
        term += indices[1]*fac; fac =                 dims[3]; indices[2] = (iGlob - term) / fac;
        term += indices[2]*fac;                                indices[3] =  iGlob - term;
      } else if constexpr (N == 5) {
        size_t fac, term;
                                fac = dims[1]*dims[2]*dims[3]*dims[4]; indices[0] =  iGlob         / fac;
        term  = indices[0]*fac; fac =         dims[2]*dims[3]*dims[4]; indices[1] = (iGlob - term) / fac;
        term += indices[1]*fac; fac =                 dims[3]*dims[4]; indices[2] = (iGlob - term) / fac;
        term += indices[2]*fac; fac =                         dims[4]; indices[3] = (iGlob - term) / fac;
        term += indices[3]*fac;                                        indices[4] =  iGlob - term;
      } else if constexpr (N == 6) {
        size_t term, fac4=dims[5], fac3=fac4*dims[4], fac2=fac3*dims[3], fac1=fac2*dims[2], fac0=fac1*dims[1];
                                 indices[0] =  iGlob         / fac0;
        term  = indices[0]*fac0; indices[1] = (iGlob - term) / fac1;
        term += indices[1]*fac1; indices[2] = (iGlob - term) / fac2;
        term += indices[2]*fac2; indices[3] = (iGlob - term) / fac3;
        term += indices[3]*fac3; indices[4] = (iGlob - term) / fac4;
        term += indices[4]*fac4; indices[5] =  iGlob - term;
      } else if constexpr (N == 7) {
        size_t term, fac5=dims[6], fac4=fac5*dims[5], fac3=fac4*dims[4], fac2=fac3*dims[3], fac1=fac2*dims[2], fac0=fac1*dims[1];
                                 indices[0] =  iGlob         / fac0;
        term  = indices[0]*fac0; indices[1] = (iGlob - term) / fac1;
        term += indices[1]*fac1; indices[2] = (iGlob - term) / fac2;
        term += indices[2]*fac2; indices[3] = (iGlob - term) / fac3;
        term += indices[3]*fac3; indices[4] = (iGlob - term) / fac4;
        term += indices[4]*fac4; indices[5] = (iGlob - term) / fac5;
        term += indices[5]*fac5; indices[6] =  iGlob - term;
      } else if constexpr (N == 8) {
        size_t term, fac6=dims[7], fac5=fac6*dims[6], fac4=fac5*dims[5], fac3=fac4*dims[4], fac2=fac3*dims[3], fac1=fac2*dims[2], fac0=fac1*dims[1];
                                 indices[0] =  iGlob         / fac0;
        term  = indices[0]*fac0; indices[1] = (iGlob - term) / fac1;
        term += indices[1]*fac1; indices[2] = (iGlob - term) / fac2;
        term += indices[2]*fac2; indices[3] = (iGlob - term) / fac3;
        term += indices[3]*fac3; indices[4] = (iGlob - term) / fac4;
        term += indices[4]*fac4; indices[5] = (iGlob - term) / fac5;
        term += indices[5]*fac5; indices[6] = (iGlob - term) / fac6;
        term += indices[6]*fac6; indices[7] =  iGlob - term;
      }
    }
  };


  template<int N> class Bounds<N,false> {
  public:
    size_t nIter;
    int     lbounds[N];
    size_t dims[N];
    size_t strides[N];
    KOKKOS_INLINE_FUNCTION Bounds( LBnd const &b0 , LBnd const &b1 = LBnd() , LBnd const &b2 = LBnd() , LBnd const &b3 = LBnd() ,
                                         LBnd const &b4 = LBnd() , LBnd const &b5 = LBnd() , LBnd const &b6 = LBnd() ,
                                         LBnd const &b7 = LBnd() ) {
      if constexpr (N >= 1) { lbounds[0] = b0.l;   strides[0] =  b0.s;   dims[0] = ( b0.u - b0.l + 1 ) / b0.s; }
      if constexpr (N >= 2) { lbounds[1] = b1.l;   strides[1] =  b1.s;   dims[1] = ( b1.u - b1.l + 1 ) / b1.s; }
      if constexpr (N >= 3) { lbounds[2] = b2.l;   strides[2] =  b2.s;   dims[2] = ( b2.u - b2.l + 1 ) / b2.s; }
      if constexpr (N >= 4) { lbounds[3] = b3.l;   strides[3] =  b3.s;   dims[3] = ( b3.u - b3.l + 1 ) / b3.s; }
      if constexpr (N >= 5) { lbounds[4] = b4.l;   strides[4] =  b4.s;   dims[4] = ( b4.u - b4.l + 1 ) / b4.s; }
      if constexpr (N >= 6) { lbounds[5] = b5.l;   strides[5] =  b5.s;   dims[5] = ( b5.u - b5.l + 1 ) / b5.s; }
      if constexpr (N >= 7) { lbounds[6] = b6.l;   strides[6] =  b6.s;   dims[6] = ( b6.u - b6.l + 1 ) / b6.s; }
      if constexpr (N >= 8) { lbounds[7] = b7.l;   strides[7] =  b7.s;   dims[7] = ( b7.u - b7.l + 1 ) / b7.s; }
      #ifdef KOKKOS_DEBUG
        if (N >= 2) { if (! b1.valid()) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 3) { if (! b2.valid()) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 4) { if (! b3.valid()) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 5) { if (! b4.valid()) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 6) { if (! b5.valid()) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 7) { if (! b6.valid()) Kokkos::abort("ERROR: Too few bounds specified"); }
        if (N >= 8) { if (! b7.valid()) Kokkos::abort("ERROR: Too few bounds specified"); }
        int num_bounds = 1;
        if (b1.valid()) num_bounds++;
        if (b2.valid()) num_bounds++;
        if (b3.valid()) num_bounds++;
        if (b4.valid()) num_bounds++;
        if (b5.valid()) num_bounds++;
        if (b6.valid()) num_bounds++;
        if (b7.valid()) num_bounds++;
        if (num_bounds != N) Kokkos::abort("ERROR: Number of bounds passed does not match templated number of bounds.");
      #endif
      nIter = 1;
      for (int i=0; i<N; i++) { nIter *= dims[i]; }
    }
    KOKKOS_INLINE_FUNCTION int lbound(int i) const {
      #ifdef KOKKOS_DEBUG
        if (i < 0 || i > N-1) Kokkos::abort("ERROR: Calling lbound() on an out of bounds integer");
      #endif
      return lbounds[i];
    }
    KOKKOS_INLINE_FUNCTION int dim   (int i) const {
      #ifdef KOKKOS_DEBUG
        if (i < 0 || i > N-1) Kokkos::abort("ERROR: Calling dim() on an out of bounds integer");
      #endif
      return dims   [i];
    }
    KOKKOS_INLINE_FUNCTION int stride(int i) const {
      #ifdef KOKKOS_DEBUG
        if (i < 0 || i > N-1) Kokkos::abort("ERROR: Calling stride() on an out of bounds integer");
      #endif
      return strides[i];
    }
    KOKKOS_INLINE_FUNCTION void unpackIndices( size_t iGlob , int indices[N] ) const {
      // Compute base indices
      if constexpr        (N == 1) {
        indices[0] = iGlob;
      } else if constexpr (N == 2) {
        indices[0] = iGlob/dims[1];
        indices[1] = iGlob - dims[1]*indices[0];
      } else if constexpr (N == 3) {
        size_t fac, term;
                                fac = dims[1]*dims[2]; indices[0] =  iGlob         / fac;
        term  = indices[0]*fac; fac =         dims[2]; indices[1] = (iGlob - term) / fac;
        term += indices[1]*fac;                        indices[2] =  iGlob - term;
      } else if constexpr (N == 4) {
        size_t fac, term;
                                fac = dims[1]*dims[2]*dims[3]; indices[0] =  iGlob         / fac;
        term  = indices[0]*fac; fac =         dims[2]*dims[3]; indices[1] = (iGlob - term) / fac;
        term += indices[1]*fac; fac =                 dims[3]; indices[2] = (iGlob - term) / fac;
        term += indices[2]*fac;                                indices[3] =  iGlob - term;
      } else if constexpr (N == 5) {
        size_t fac, term;
                                fac = dims[1]*dims[2]*dims[3]*dims[4]; indices[0] =  iGlob         / fac;
        term  = indices[0]*fac; fac =         dims[2]*dims[3]*dims[4]; indices[1] = (iGlob - term) / fac;
        term += indices[1]*fac; fac =                 dims[3]*dims[4]; indices[2] = (iGlob - term) / fac;
        term += indices[2]*fac; fac =                         dims[4]; indices[3] = (iGlob - term) / fac;
        term += indices[3]*fac;                                        indices[4] =  iGlob - term;
      } else if constexpr (N == 6) {
        size_t term, fac4=dims[5], fac3=fac4*dims[4], fac2=fac3*dims[3], fac1=fac2*dims[2], fac0=fac1*dims[1];
                                 indices[0] =  iGlob         / fac0;
        term  = indices[0]*fac0; indices[1] = (iGlob - term) / fac1;
        term += indices[1]*fac1; indices[2] = (iGlob - term) / fac2;
        term += indices[2]*fac2; indices[3] = (iGlob - term) / fac3;
        term += indices[3]*fac3; indices[4] = (iGlob - term) / fac4;
        term += indices[4]*fac4; indices[5] =  iGlob - term;
      } else if constexpr (N == 7) {
        size_t term, fac5=dims[6], fac4=fac5*dims[5], fac3=fac4*dims[4], fac2=fac3*dims[3], fac1=fac2*dims[2], fac0=fac1*dims[1];
                                 indices[0] =  iGlob         / fac0;
        term  = indices[0]*fac0; indices[1] = (iGlob - term) / fac1;
        term += indices[1]*fac1; indices[2] = (iGlob - term) / fac2;
        term += indices[2]*fac2; indices[3] = (iGlob - term) / fac3;
        term += indices[3]*fac3; indices[4] = (iGlob - term) / fac4;
        term += indices[4]*fac4; indices[5] = (iGlob - term) / fac5;
        term += indices[5]*fac5; indices[6] =  iGlob - term;
      } else if constexpr (N == 8) {
        size_t term, fac6=dims[7], fac5=fac6*dims[6], fac4=fac5*dims[5], fac3=fac4*dims[4], fac2=fac3*dims[3], fac1=fac2*dims[2], fac0=fac1*dims[1];
                                 indices[0] =  iGlob         / fac0;
        term  = indices[0]*fac0; indices[1] = (iGlob - term) / fac1;
        term += indices[1]*fac1; indices[2] = (iGlob - term) / fac2;
        term += indices[2]*fac2; indices[3] = (iGlob - term) / fac3;
        term += indices[3]*fac3; indices[4] = (iGlob - term) / fac4;
        term += indices[4]*fac4; indices[5] = (iGlob - term) / fac5;
        term += indices[5]*fac5; indices[6] = (iGlob - term) / fac6;
        term += indices[6]*fac6; indices[7] =  iGlob - term;
      }

      // Apply strides and lower bounds
      if constexpr (N >= 1) indices[0] = indices[0]*strides[0] + lbounds[0];
      if constexpr (N >= 2) indices[1] = indices[1]*strides[1] + lbounds[1];
      if constexpr (N >= 3) indices[2] = indices[2]*strides[2] + lbounds[2];
      if constexpr (N >= 4) indices[3] = indices[3]*strides[3] + lbounds[3];
      if constexpr (N >= 5) indices[4] = indices[4]*strides[4] + lbounds[4];
      if constexpr (N >= 6) indices[5] = indices[5]*strides[5] + lbounds[5];
      if constexpr (N >= 7) indices[6] = indices[6]*strides[6] + lbounds[6];
      if constexpr (N >= 8) indices[7] = indices[7]*strides[7] + lbounds[7];
    }
  };

  template <int N> using SimpleBounds = Bounds<N,true>;


  template <class F, bool simple, int N>
  KOKKOS_INLINE_FUNCTION void callFunctor(F const &f , Bounds<N,simple> const &bnd , int const i ) {
    int ind[N];
    bnd.unpackIndices( i , ind );
    if constexpr (N == 1) f(ind[0]);
    if constexpr (N == 2) f(ind[0],ind[1]);
    if constexpr (N == 3) f(ind[0],ind[1],ind[2]);
    if constexpr (N == 4) f(ind[0],ind[1],ind[2],ind[3]);
    if constexpr (N == 5) f(ind[0],ind[1],ind[2],ind[3],ind[4]);
    if constexpr (N == 6) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5]);
    if constexpr (N == 7) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6]);
    if constexpr (N == 8) f(ind[0],ind[1],ind[2],ind[3],ind[4],ind[5],ind[6],ind[7]);
  }


  template <class F, int N, bool simple>
  inline void parallel_for( char const * str , Bounds<N,simple> const &bounds , F const &f ) {
    if (bounds.nIter == 0) return;
    Kokkos::parallel_for( str , bounds.nIter , KOKKOS_LAMBDA (int i) { callFunctor(f,bounds,i); });
  }

}


