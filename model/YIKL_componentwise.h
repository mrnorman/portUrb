
#pragma once

namespace yikl {

  namespace componentwise {

    ///////////////////////////////////////////////////////////////////////
    // Binary operators with Array LHS and scalar RHS
    ///////////////////////////////////////////////////////////////////////

    // Addition
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()+T2()),N,D0,D1,D2,D3>
    operator+( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()+T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] + right; }
      return ret;
    }

    // Subtraction
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()-T2()),N,D0,D1,D2,D3>
    operator-( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()-T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] - right; }
      return ret;
    }

    // Multiplication
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()*T2()),N,D0,D1,D2,D3>
    operator*( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()*T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] * right; }
      return ret;
    }

    // Division
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()/T2()),N,D0,D1,D2,D3>
    operator/( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()/T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] / right; }
      return ret;
    }

    // Greater than >
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()>T2()),N,D0,D1,D2,D3>
    operator>( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()>T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] > right; }
      return ret;
    }

    // Less than <
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()<T2()),N,D0,D1,D2,D3>
    operator<( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()<T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] < right; }
      return ret;
    }

    // Greater than or equal to >=
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3>
    operator>=( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] >= right; }
      return ret;
    }

    // Less than or equal to <=
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3>
    operator<=( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] <= right; }
      return ret;
    }

    // Equal to ==
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()==T2()),N,D0,D1,D2,D3>
    operator==( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()==T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] == right; }
      return ret;
    }

    // Not equal to !=
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3>
    operator!=( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] != right; }
      return ret;
    }

    // logical and &&
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3>
    operator&&( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] && right; }
      return ret;
    }

    // logical or ||
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T2>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()||T2()),N,D0,D1,D2,D3>
    operator||( SArray<T1,N,D0,D1,D2,D3> const &left , T2 const &right ) {
      SArray<decltype(T1()||T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] || right; }
      return ret;
    }


    ///////////////////////////////////////////////////////////////////////
    // Binary operators with scalar LHS and Array RHS
    ///////////////////////////////////////////////////////////////////////

    // Addition
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()+T2()),N,D0,D1,D2,D3>
    operator+( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()+T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left + right.data()[i]; }
      return ret;
    }

    // Subtraction
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()-T2()),N,D0,D1,D2,D3>
    operator-( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()-T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left - right.data()[i]; }
      return ret;
    }

    // Multiplication
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()*T2()),N,D0,D1,D2,D3>
    operator*( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()*T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left * right.data()[i]; }
      return ret;
    }

    // Division
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()/T2()),N,D0,D1,D2,D3>
    operator/( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()/T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left / right.data()[i]; }
      return ret;
    }

    // Greater than >
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()>T2()),N,D0,D1,D2,D3>
    operator>( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()>T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left > right.data()[i]; }
      return ret;
    }

    // Less than <
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()<T2()),N,D0,D1,D2,D3>
    operator<( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()<T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left < right.data()[i]; }
      return ret;
    }

    // Greater than or equal to >=
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3>
    operator>=( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left >= right.data()[i]; }
      return ret;
    }

    // Less than or equal to <=
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3>
    operator<=( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left <= right.data()[i]; }
      return ret;
    }

    // Equal to ==
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()==T2()),N,D0,D1,D2,D3>
    operator==( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()==T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left == right.data()[i]; }
      return ret;
    }

    // Not equal to !=
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3>
    operator!=( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left != right.data()[i]; }
      return ret;
    }

    // logical and &&
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3>
    operator&&( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left && right.data()[i]; }
      return ret;
    }

    // logical or ||
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3,
              typename std::enable_if<std::is_arithmetic<T1>::value,bool>::type = false>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()||T2()),N,D0,D1,D2,D3>
    operator||( T1 const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()||T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left || right.data()[i]; }
      return ret;
    }


    ///////////////////////////////////////////////////////////////////////
    // Binary operators with Array LHS and Array RHS
    ///////////////////////////////////////////////////////////////////////

    // Addition
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()+T2()),N,D0,D1,D2,D3>
    operator+( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()+T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] + right.data()[i]; }
      return ret;
    }

    // Subtraction
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()-T2()),N,D0,D1,D2,D3>
    operator-( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()-T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] - right.data()[i]; }
      return ret;
    }

    // Multiplication
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()*T2()),N,D0,D1,D2,D3>
    operator*( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()*T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] * right.data()[i]; }
      return ret;
    }

    // Division
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()/T2()),N,D0,D1,D2,D3>
    operator/( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()/T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] / right.data()[i]; }
      return ret;
    }

    // Greater than >
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()>T2()),N,D0,D1,D2,D3>
    operator>( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()>T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] > right.data()[i]; }
      return ret;
    }

    // Less than <
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()<T2()),N,D0,D1,D2,D3>
    operator<( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()<T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] < right.data()[i]; }
      return ret;
    }

    // Greater than or equal to >=
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3>
    operator>=( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()>=T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] >= right.data()[i]; }
      return ret;
    }

    // Less than or equal to <=
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3>
    operator<=( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()<=T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] <= right.data()[i]; }
      return ret;
    }

    // Equal to ==
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()==T2()),N,D0,D1,D2,D3>
    operator==( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()==T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] == right.data()[i]; }
      return ret;
    }

    // Not equal to !=
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3>
    operator!=( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()!=T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] != right.data()[i]; }
      return ret;
    }

    // logical and &&
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3>
    operator&&( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()&&T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] && right.data()[i]; }
      return ret;
    }

    // logical or ||
    template <class T1, class T2, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()||T2()),N,D0,D1,D2,D3>
    operator||( SArray<T1,N,D0,D1,D2,D3> const &left , SArray<T2,N,D0,D1,D2,D3> const &right ) {
      SArray<decltype(T1()||T2()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i] || right.data()[i]; }
      return ret;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Unary operators
    ///////////////////////////////////////////////////////////////////////////////////////////

    // logical not !
    template <class T1, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(!T1()),N,D0,D1,D2,D3>
    operator!( SArray<T1,N,D0,D1,D2,D3> const &left ) {
      SArray<decltype(!T1()),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = ! left.data()[i]; }
      return ret;
    }

    // increment ++
    template <class T1, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()+1),N,D0,D1,D2,D3>
    operator++( SArray<T1,N,D0,D1,D2,D3> const &left ) {
      SArray<decltype(T1()+1),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]+1; }
      return ret;
    }

    // increment ++
    template <class T1, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()+1),N,D0,D1,D2,D3>
    operator++( SArray<T1,N,D0,D1,D2,D3> const &left , int dummy) {
      SArray<decltype(T1()+1),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]+1; }
      return ret;
    }

    // decrement --
    template <class T1, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()-1),N,D0,D1,D2,D3>
    operator--( SArray<T1,N,D0,D1,D2,D3> const &left ) {
      SArray<decltype(T1()-1),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]-1; }
      return ret;
    }

    // decrement --
    template <class T1, int N, size_t D0, size_t D1, size_t D2, size_t D3>
    KOKKOS_INLINE_FUNCTION SArray<decltype(T1()-1),N,D0,D1,D2,D3>
    operator--( SArray<T1,N,D0,D1,D2,D3> const &left , int dummy ) {
      SArray<decltype(T1()-1),N,D0,D1,D2,D3> ret;
      for (size_t i=0; i < ret.totElems(); i++) { ret.data()[i] = left.data()[i]-1; }
      return ret;
    }

  }

}

