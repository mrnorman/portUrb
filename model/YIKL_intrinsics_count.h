
#pragma once

namespace yikl {

  namespace intrinsics {
    
    template <int N>
    inline int count(yakl::Array<bool,N,yakl::memDevice> const &arr) {
      int result;
      Kokkos::parallel_reduce( YIKL_AUTO_LABEL() , arr.size() , KOKKOS_LAMBDA (int i, int & lsum) {
        if (arr.data()[i]) lsum++;
      }, Kokkos::Sum<int>(result) );
      return result;
    }

  }

}

