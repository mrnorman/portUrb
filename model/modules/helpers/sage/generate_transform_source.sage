#!/usr/bin/env /home/imn/sage-6.4.1-i686-Linux/sage -python

load("transformation_matrices.sage")
load("c_utils.sage")

N1 = 2
N2 = 9

print('#pragma once')

print('#include "main_header.h"')
print('#include <math.h>')
print('using yakl::SArray;\n')

print('namespace TransformMatrices {\n')

for N in range(N1,N2+1) :
    hs = (N-1)/2

    print('  template <class FP> YAKL_INLINE void get_gll_points(SArray<FP,1,%s> &rslt) {'%(N))
    pts,wts = lobatto_weights_nodes(N,129,False,1.e-35,100)
    pts = pts / 2
    wts = wts / 2
    print(add_spaces(4,c_vector('rslt',N,force_fp(pts,129),'none')))
    print('  }\n');

    print('  template <class FP> YAKL_INLINE void get_gll_weights(SArray<FP,1,%s> &rslt) {'%(N))
    pts,wts = lobatto_weights_nodes(N,129,False,1.e-35,100)
    pts = pts / 2
    wts = wts / 2
    print(add_spaces(4,c_vector('rslt',N,force_fp(wts,129),'none')))
    print('  }\n');

    print('  template <class FP> YAKL_INLINE void gll_to_coefs(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
    p2c,c2p = points_gll_to_coefs(N)
    print(add_spaces(4,c_matrix('rslt',N,N,force_fp(p2c,129),'none')))
    print('  }\n');

    print('  template <class FP> YAKL_INLINE void coefs_to_gll(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
    p2c,c2p = points_gll_to_coefs(N)
    print(add_spaces(4,c_matrix('rslt',N,N,force_fp(c2p,129),'none')))
    print('  }\n');

    print('  template <class FP> YAKL_INLINE void coefs_to_deriv(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
    c2d = coefs_to_deriv(N)
    print(add_spaces(4,c_matrix('rslt',N,N,force_fp(c2d,129),'none')))
    print('  }\n');

    print('  template <class FP> YAKL_INLINE float coefs_to_tv(SArray<FP,1,%s> &a) {'%(N))
    print('    FP rslt;')
    rslt = coefs_to_TV(N)
    print(add_spaces(4,c_scalar_float('rslt',force_fp(rslt,129),'a')))
    print('    return rslt;')
    print('  }\n');

    c2g = coefs_to_gll_lower(N)
    for M in range(1,N+1) :
        print('  template <class FP> YAKL_INLINE void coefs_to_gll_lower(SArray<FP,2,%s,%s> &rslt) {'%(N,M))
        print(add_spaces(4,c_matrix_aoa('rslt',M,N,c2g[M-1],'none')))
        print('  }\n');

    if (N%2 == 1) :
        print('  template <class FP> YAKL_INLINE void sten_to_coefs(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
        p2c,c2p = stencil_to_coefs(N)
        print(add_spaces(4,c_matrix('rslt',N,N,force_fp(p2c,129),'none')))
        print('  }\n');

        print('  template <class FP> YAKL_INLINE void coefs_to_sten(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
        p2c,c2p = stencil_to_coefs(N)
        print(add_spaces(4,c_matrix('rslt',N,N,force_fp(c2p,129),'none')))
        print('  }\n');

        s2g = sten_to_gll_lower(N)
        for M in range(1,N+1) :
            print('  template <class FP> YAKL_INLINE void sten_to_gll_lower(SArray<FP,2,%s,%s> &rslt) {'%(N,M))
            print(add_spaces(4,c_matrix_aoa('rslt',M,N,s2g[M-1],'none')))
            print('  }\n');

        print('  template <class FP> YAKL_INLINE void weno_lower_sten_to_coefs(SArray<FP,3,%s,%s,%s> &rslt) {'%(hs+1,hs+1,hs+1))
        weno = weno_lower_sten_to_coefs(N)
        print(add_spaces(4,c_3d('rslt',hs+1,hs+1,hs+1,weno,'none')))
        print('  }\n');


N = 1

print('  template <class FP> YAKL_INLINE void get_gll_points(SArray<FP,1,%s> &rslt) {'%(N))
print('    rslt(0) = 0;')
print('  }\n');

print('  template <class FP> YAKL_INLINE void get_gll_weights(SArray<FP,1,%s> &rslt) {'%(N))
print('    rslt(0) = 1;')
print('  }\n');

print('  template <class FP> YAKL_INLINE void gll_to_coefs(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
print('    rslt(0,0) = 1;')
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs_to_gll(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
print('    rslt(0,0) = 1;')
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs_to_deriv(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
print('    rslt(0,0) = 0;')
print('  }\n');

print('  template <class FP> YAKL_INLINE FP coefs_to_tv(SArray<FP,1,%s> &a) {'%(N))
print('    return 0;')
print('  }\n');

print('  template <class FP> YAKL_INLINE void sten_to_coefs(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
print('    rslt(0,0) = 1;')
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs_to_sten(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
print('    rslt(0,0) = 1;')
print('  }\n');

s2g = sten_to_gll_lower(N)
for M in range(1,N+1) :
    print('  template <class FP> YAKL_INLINE void sten_to_gll_lower(SArray<FP,2,%s,%s> &rslt) {'%(N,M))
    print('    rslt(0,0) = 1;')
    print('  }\n');

c2g = coefs_to_gll_lower(N)
for M in range(1,N+1) :
    print('  template <class FP> YAKL_INLINE void coefs_to_gll_lower(SArray<FP,2,%s,%s> &rslt) {'%(N,M))
    print('    rslt(0,0) = 1;')
    print('  }\n');

print('  template <class FP> YAKL_INLINE void weno_sten_to_coefs(SArray<FP,3,%s,%s,%s> &rslt) {'%(N,N,N))
print('    rslt(0,0,0) = 1;')
print('  }\n');

print('}\n')

