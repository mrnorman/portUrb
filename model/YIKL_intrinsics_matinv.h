
#pragma once

namespace yikl {

  namespace intrinsics {

    template <size_t n, class real>
    YAKL_INLINE SArray<real,2,n,n> matinv_ge(SArray<real,2,n,n> const &a) {
      SArray<real,2,n,n> scratch;
      SArray<real,2,n,n> inv;

      // Initialize inverse as identity
      for (int icol = 0; icol < n; icol++) {
        for (int irow = 0; irow < n; irow++) {
          scratch(icol,irow) = a(icol,irow);
          if (icol == irow) {
            inv(irow,icol) = 1;
          } else {
            inv(irow,icol) = 0;
          }
        }
      }

      // Gaussian elimination to zero out lower
      for (int idiag = 0; idiag < n; idiag++) {
        // Divide out the diagonal component from the first row
        real factor = static_cast<real>(1)/scratch(idiag,idiag);
        for (int icol = idiag; icol < n; icol++) {
          scratch(idiag,icol) *= factor;
        }
        for (int icol = 0; icol < n; icol++) {
          inv(idiag,icol) *= factor;
        }
        for (int irow = idiag+1; irow < n; irow++) {
          real factor = scratch(irow,idiag);
          for (int icol = idiag; icol < n; icol++) {
            scratch(irow,icol) -= factor * scratch(idiag,icol);
          }
          for (int icol = 0; icol < n; icol++) {
            inv    (irow,icol) -= factor * inv    (idiag,icol);
          }
        }
      }

      // Gaussian elimination to zero out upper
      for (int idiag = n-1; idiag >= 1; idiag--) {
        for (int irow = 0; irow < idiag; irow++) {
          real factor = scratch(irow,idiag);
          for (int icol = irow+1; icol < n; icol++) {
            scratch(irow,icol) -= factor * scratch(idiag,icol);
          }
          for (int icol = 0; icol < n; icol++) {
            inv    (irow,icol) -= factor * inv    (idiag,icol);
          }
        }
      }

      return inv;
    }



  }

}

