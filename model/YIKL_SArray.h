
#pragma once

namespace yikl {

  template <class T, int rank, size_t D0, size_t D1=1, size_t D2=1, size_t D3=1>
  class SArray {
  protected:
    static size_t constexpr OFF0 = D3*D2*D1;
    static size_t constexpr OFF1 = D3*D2;
    static size_t constexpr OFF2 = D3;
    static size_t constexpr OFF3 = 1;
    T mutable myData[D0*D1*D2*D3];

  public :

    typedef typename std::remove_cv<T>::type       type;
    typedef          T                             value_type;
    typedef typename std::add_const<type>::type    const_value_type;
    typedef typename std::remove_const<type>::type non_const_value_type;

    KOKKOS_INLINE_FUNCTION SArray(T init_fill) { for (int i=0; i < size(); i++) { myData[i] = init_fill; } }
    SArray()  = default;
    ~SArray() = default;

    KOKKOS_INLINE_FUNCTION T &operator()(size_t const i0) const {
      static_assert(rank==1,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_DEBUG
          if constexpr (rank >= 1) { if (i0>D0-1) { KOKKOS_IF_ON_HOST( printf("SArray i0 out of bounds (i0: %zu; lb0: %d; ub0: %zu)\n",i0,0,D0-1); ) } }
          if constexpr (rank >= 1) { if (i0>D0-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
      #endif
      return myData[i0];
    }
    KOKKOS_INLINE_FUNCTION T &operator()(size_t const i0, size_t const i1) const {
      static_assert(rank==2,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_DEBUG
        if constexpr (rank >= 1) { if (i0>D0-1) { KOKKOS_IF_ON_HOST( printf("SArray i0 out of bounds (i0: %zu; lb0: %d; ub0: %zu)\n",i0,0,D0-1); ) } }
        if constexpr (rank >= 2) { if (i1>D1-1) { KOKKOS_IF_ON_HOST( printf("SArray i1 out of bounds (i1: %zu; lb1: %d; ub1: %zu)\n",i1,0,D1-1); ) } }
        if constexpr (rank >= 1) { if (i0>D0-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
        if constexpr (rank >= 2) { if (i1>D1-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
      #endif
      return myData[i0*OFF0 + i1];
    }
    KOKKOS_INLINE_FUNCTION T &operator()(size_t const i0, size_t const i1, size_t const i2) const {
      static_assert(rank==3,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_DEBUG
        if constexpr (rank >= 1) { if (i0>D0-1) { KOKKOS_IF_ON_HOST( printf("SArray i0 out of bounds (i0: %zu; lb0: %d; ub0: %zu)\n",i0,0,D0-1); ) } }
        if constexpr (rank >= 2) { if (i1>D1-1) { KOKKOS_IF_ON_HOST( printf("SArray i1 out of bounds (i1: %zu; lb1: %d; ub1: %zu)\n",i1,0,D1-1); ) } }
        if constexpr (rank >= 3) { if (i2>D2-1) { KOKKOS_IF_ON_HOST( printf("SArray i2 out of bounds (i2: %zu; lb2: %d; ub2: %zu)\n",i2,0,D2-1); ) } }
        if constexpr (rank >= 1) { if (i0>D0-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
        if constexpr (rank >= 2) { if (i1>D1-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
        if constexpr (rank >= 3) { if (i2>D2-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
      #endif
      return myData[i0*OFF0 + i1*OFF1 + i2];
    }
    KOKKOS_INLINE_FUNCTION T &operator()(size_t const i0, size_t const i1, size_t const i2, size_t const i3) const {
      static_assert(rank==4,"ERROR: Improper number of dimensions specified in operator()");
      #ifdef KOKKOS_DEBUG
        if constexpr (rank >= 1) { if (i0>D0-1) { KOKKOS_IF_ON_HOST( printf("SArray i0 out of bounds (i0: %zu; lb0: %d; ub0: %zu)\n",i0,0,D0-1); ) } }
        if constexpr (rank >= 2) { if (i1>D1-1) { KOKKOS_IF_ON_HOST( printf("SArray i1 out of bounds (i1: %zu; lb1: %d; ub1: %zu)\n",i1,0,D1-1); ) } }
        if constexpr (rank >= 3) { if (i2>D2-1) { KOKKOS_IF_ON_HOST( printf("SArray i2 out of bounds (i2: %zu; lb2: %d; ub2: %zu)\n",i2,0,D2-1); ) } }
        if constexpr (rank >= 4) { if (i3>D3-1) { KOKKOS_IF_ON_HOST( printf("SArray i3 out of bounds (i3: %zu; lb3: %d; ub3: %zu)\n",i3,0,D3-1); ) } }
        if constexpr (rank >= 1) { if (i0>D0-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
        if constexpr (rank >= 2) { if (i1>D1-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
        if constexpr (rank >= 3) { if (i2>D2-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
        if constexpr (rank >= 4) { if (i3>D3-1) { Kokkos::abort("ERROR: SArray index out of bounds"); } }
      #endif
      return myData[i0*OFF0 + i1*OFF1 + i2*OFF2 + i3];
    }

    template <class TLOC , typename std::enable_if<std::is_arithmetic<TLOC>::value,int>::type = 0 >
    KOKKOS_INLINE_FUNCTION void operator= (TLOC val) { for (int i=0 ; i < totElems() ; i++) { myData[i] = val; } }

    KOKKOS_INLINE_FUNCTION T *data    () const { return myData; }
    KOKKOS_INLINE_FUNCTION T *get_data() const { return myData; }
    KOKKOS_INLINE_FUNCTION T *begin() const { return myData; }
    KOKKOS_INLINE_FUNCTION T *end() const { return begin() + size(); }
    static size_t constexpr totElems      () { return D3*D2*D1*D0; }
    static size_t constexpr get_totElems  () { return D3*D2*D1*D0; }
    static size_t constexpr size          () { return D3*D2*D1*D0; }
    static size_t constexpr get_elem_count() { return D3*D2*D1*D0; }
    static size_t constexpr get_rank      () { return rank; }
    static bool   constexpr span_is_contiguous() { return true; }
    static bool   constexpr initialized() { return true; }

    inline friend std::ostream &operator<<(std::ostream& os, SArray<T,rank,D0,D1,D2,D3> const &v) {
      for (size_t i=0; i<totElems(); i++) { os << std::setw(12) << v.myData[i] << "\n"; }
      os << "\n";
      return os;
    }
    
    KOKKOS_INLINE_FUNCTION SArray<size_t,1,rank> get_dimensions() const {
      SArray<size_t,1,rank> ret;
      if constexpr (rank >= 1) ret(0) = D0;
      if constexpr (rank >= 2) ret(1) = D1;
      if constexpr (rank >= 3) ret(2) = D2;
      if constexpr (rank >= 4) ret(3) = D3;
      return ret;
    }

    KOKKOS_INLINE_FUNCTION SArray<size_t,1,rank> get_lbounds() const {
      SArray<size_t,1,rank> ret;
      if constexpr (rank >= 1) ret(0) = 0;
      if constexpr (rank >= 2) ret(1) = 0;
      if constexpr (rank >= 3) ret(2) = 0;
      if constexpr (rank >= 4) ret(3) = 0;
      return ret;
    }

    KOKKOS_INLINE_FUNCTION SArray<size_t,1,rank> get_ubounds() const {
      SArray<size_t,1,rank> ret;
      if constexpr (rank >= 1) ret(0) = D0-1;
      if constexpr (rank >= 2) ret(1) = D1-1;
      if constexpr (rank >= 3) ret(2) = D2-1;
      if constexpr (rank >= 4) ret(3) = D3-1;
      return ret;
    }

  };

}

