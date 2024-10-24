
#pragma once

namespace yikl {

  namespace intrinsics {

    ///////////////////////////////////////////////////////////
    // Matrix multiplication routines for column-row format
    ///////////////////////////////////////////////////////////
    template <class T, size_t COL_L, size_t ROW_L, size_t COL_R>
    YAKL_INLINE SArray<T,2,COL_R,ROW_L>
    matmul_cr ( SArray<T,2,COL_L,ROW_L> const &left ,
                SArray<T,2,COL_R,COL_L> const &right ) {
      SArray<T,2,COL_R,ROW_L> ret;
      for (size_t i=0; i < COL_R; i++) {
        for (size_t j=0; j < ROW_L; j++) {
          T tmp = 0;
          for (size_t k=0; k < COL_L; k++) {
            tmp += left(k,j) * right(i,k);
          }
          ret(i,j) = tmp;
        }
      }
      return ret;
    }

    template<class T, size_t COL_L, size_t ROW_L>
    YAKL_INLINE SArray<T,1,ROW_L>
    matmul_cr ( SArray<T,2,COL_L,ROW_L> const &left ,
                SArray<T,1,COL_L>       const &right ) {
      SArray<T,1,ROW_L> ret;
      for (size_t j=0; j < ROW_L; j++) {
        T tmp = 0;
        for (size_t k=0; k < COL_L; k++) {
          tmp += left(k,j) * right(k);
        }
        ret(j) = tmp;
      }
      return ret;
    }


    ///////////////////////////////////////////////////////////
    // Matrix multiplication routines for row-column format
    ///////////////////////////////////////////////////////////
    template <class T, size_t COL_L, size_t ROW_L, size_t COL_R>
    YAKL_INLINE SArray<T,2,ROW_L,COL_R>
    matmul_rc ( SArray<T,2,ROW_L,COL_L> const &left ,
                SArray<T,2,COL_L,COL_R> const &right ) {
      SArray<T,2,ROW_L,COL_R> ret;
      for (size_t i=0; i < COL_R; i++) {
        for (size_t j=0; j < ROW_L; j++) {
          T tmp = 0;
          for (size_t k=0; k < COL_L; k++) {
            tmp += left(j,k) * right(k,i);
          }
          ret(j,i) = tmp;
        }
      }
      return ret;
    }

    template<class T, size_t COL_L, size_t ROW_L>
    YAKL_INLINE SArray<T,1,ROW_L>
    matmul_rc ( SArray<T,2,ROW_L,COL_L> const &left ,
                SArray<T,1,COL_L>       const &right ) {
      SArray<T,1,ROW_L> ret;
      for (size_t j=0; j < ROW_L; j++) {
        T tmp = 0;
        for (size_t k=0; k < COL_L; k++) {
          tmp += left(j,k) * right(k);
        }
        ret(j) = tmp;
      }
      return ret;
    }

  }

}
