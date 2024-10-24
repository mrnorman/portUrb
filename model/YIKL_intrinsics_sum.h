
#pragma once

namespace yikl {

  namespace intrinsics {
    
    template <class T, int N>
    inline T sum(yakl::Array<T,N,yakl::memDevice> const &arr) {
      T result;
      Kokkos::parallel_reduce( YIKL_AUTO_LABEL() , arr.size() , KOKKOS_LAMBDA (int i, T & lsum) {
        lsum += arr.data()[i];
      }, Kokkos::Sum<T>(result) );
      return result;
    }

  }

}

