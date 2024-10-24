
#pragma once

namespace yikl {

  namespace intrinsics {
    
    template <class T, int N>
    inline T minval(yakl::Array<T,N,yakl::memDevice> const &arr) {
      T result;
      Kokkos::parallel_reduce( YIKL_AUTO_LABEL() , arr.size() , KOKKOS_LAMBDA (int i, T & lmin) {
        T val = arr.data()[i];
        lmin = val < lmin ? val : lmin;
      }, Kokkos::Min<T>(result) );
      return result;
    }

  }

}

