
#pragma once

namespace yikl {

  namespace intrinsics {
    
    template <class T, int N>
    inline T maxval(yakl::Array<T,N,yakl::memDevice> const &arr) {
      T result;
      Kokkos::parallel_reduce( YIKL_AUTO_LABEL() , arr.size() , KOKKOS_LAMBDA (int i, T & lmax) {
        T val = arr.data()[i];
        lmax = val > lmax ? val : lmax;
      }, Kokkos::Max<T>(result) );
      return result;
    }

  }

}

