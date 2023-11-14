
#pragma once

namespace limiter {

  template <int ord>
  struct MinmodLimiter {
    MinmodLimiter() = default;
    YAKL_INLINE void compute_limited_coefs( SArray<real,1,ord> const &s , SArray<real,1,ord> &coefs_H ) const {
      int constexpr hs = (ord-1)/2;
      real d1 = s(hs  ) - s(hs-1);
      real d2 = s(hs+1) - s(hs  );
      real der;
      if      (d1*d2 < 0)                   { der = 0; }
      else if (std::abs(d1) < std::abs(d2)) { der = d1; }
      else                                  { der = d2; }
      coefs_H(0) = s(hs);
      coefs_H(1) = der  ;
      for (int i=2; i < ord; i++) { coefs_H(i) = 0; }
    }
  };

}


