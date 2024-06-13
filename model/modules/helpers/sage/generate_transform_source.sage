#!/usr/bin/env /home/imn/sage-6.4.1-i686-Linux/sage -python

load("transformation_matrices.sage")
load("c_utils.sage")

N1 = 2
N2 = 15

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
    print(add_spaces(4,c_vector('rslt',N,force_fp(pts,129),'none')),end="")
    print('  }\n');

    print('  template <class FP> YAKL_INLINE void get_gll_weights(SArray<FP,1,%s> &rslt) {'%(N))
    pts,wts = lobatto_weights_nodes(N,129,False,1.e-35,100)
    pts = pts / 2
    wts = wts / 2
    print(add_spaces(4,c_vector('rslt',N,force_fp(wts,129),'none')),end="")
    print('  }\n');

    print('  template <class FP> YAKL_INLINE void gll_to_coefs(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
    p2c,c2p = points_gll_to_coefs(N)
    print(add_spaces(4,c_matrix('rslt',N,N,force_fp(p2c,129),'none')),end="")
    print('  }\n');

    print('  template <class FP> YAKL_INLINE void coefs_to_gll(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
    p2c,c2p = points_gll_to_coefs(N)
    print(add_spaces(4,c_matrix('rslt',N,N,force_fp(c2p,129),'none')),end="")
    print('  }\n');

    print('  template <class FP> YAKL_INLINE void coefs_to_deriv(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
    c2d = coefs_to_deriv(N)
    print(add_spaces(4,c_matrix('rslt',N,N,force_fp(c2d,129),'none')),end="")
    print('  }\n');

    print('  template <class FP> YAKL_INLINE float coefs_to_tv(SArray<FP,1,%s> &a) {'%(N))
    print('    FP rslt;')
    rslt = coefs_to_TV(N)
    print(add_spaces(4,c_scalar('rslt',force_fp(rslt,129),'a')),end="")
    print('    return rslt;')
    print('  }\n');

    c2g = coefs_to_gll_lower(N)
    for M in range(1,N+1) :
        print('  template <class FP> YAKL_INLINE void coefs_to_gll_lower(SArray<FP,2,%s,%s> &rslt) {'%(N,M))
        print(add_spaces(4,c_matrix_aoa('rslt',M,N,c2g[M-1],'none')),end="")
        print('  }\n');

    if (N%2 == 1) :
        print('  template <class FP> YAKL_INLINE void sten_to_coefs(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
        p2c,c2p = stencil_to_coefs(N)
        print(add_spaces(4,c_matrix('rslt',N,N,force_fp(p2c,129),'none')),end="")
        print('  }\n');

        print('  template <class FP> YAKL_INLINE void coefs_to_sten(SArray<FP,2,%s,%s> &rslt) {'%(N,N))
        p2c,c2p = stencil_to_coefs(N)
        print(add_spaces(4,c_matrix('rslt',N,N,force_fp(c2p,129),'none')),end="")
        print('  }\n');

        s2g = sten_to_gll_lower(N)
        for M in range(1,N+1) :
            print('  template <class FP> YAKL_INLINE void sten_to_gll_lower(SArray<FP,2,%s,%s> &rslt) {'%(N,M))
            print(add_spaces(4,c_matrix_aoa('rslt',M,N,s2g[M-1],'none')),end="")
            print('  }\n');

        print('  template <class FP> YAKL_INLINE void weno_lower_sten_to_coefs(SArray<FP,3,%s,%s,%s> &rslt) {'%(hs+1,hs+1,hs+1))
        weno = weno_lower_sten_to_coefs(N)
        print(add_spaces(4,c_3d('rslt',hs+1,hs+1,hs+1,weno,'none')),end="")
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



print('  template <class FP> YAKL_INLINE void coefs2_shift1(SArray<FP,1,2> &coefs , FP s0 , FP s1) {')
N=2
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs2_shift2(SArray<FP,1,2> &coefs , FP s0 , FP s1) {')
N=2
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs3_shift1(SArray<FP,1,3> &coefs , FP s0 , FP s1, FP s2) {')
N=3
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs3_shift2(SArray<FP,1,3> &coefs , FP s0 , FP s1, FP s2) {')
N=3
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs3_shift3(SArray<FP,1,3> &coefs , FP s0 , FP s1, FP s2) {')
N=3
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs4_shift1(SArray<FP,1,4> &coefs , FP s0 , FP s1, FP s2, FP s3) {')
N=4
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs4_shift2(SArray<FP,1,4> &coefs , FP s0 , FP s1, FP s2, FP s3) {')
N=4
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs4_shift3(SArray<FP,1,4> &coefs , FP s0 , FP s1, FP s2, FP s3) {')
N=4
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs4_shift4(SArray<FP,1,4> &coefs , FP s0 , FP s1, FP s2, FP s3) {')
N=4
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs5_shift1(SArray<FP,1,5> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4) {')
N=5
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs5_shift2(SArray<FP,1,5> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4) {')
N=5
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs5_shift3(SArray<FP,1,5> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4) {')
N=5
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs5_shift4(SArray<FP,1,5> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4) {')
N=5
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs5_shift5(SArray<FP,1,5> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4) {')
N=5
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs6_shift1(SArray<FP,1,6> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5) {')
N=6
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-11/2,-9/2) ,
                  integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs6_shift2(SArray<FP,1,6> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5) {')
N=6
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs6_shift3(SArray<FP,1,6> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5) {')
N=6
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs6_shift4(SArray<FP,1,6> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5) {')
N=6
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs6_shift5(SArray<FP,1,6> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5) {')
N=6
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs6_shift6(SArray<FP,1,6> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5) {')
N=6
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ,
                  integrate(p,x, 9/2, 11/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs7_shift1(SArray<FP,1,7> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6) {')
N=7
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-13/2,-11/2) ,
                  integrate(p,x,-11/2,-9/2) ,
                  integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs7_shift2(SArray<FP,1,7> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6) {')
N=7
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-11/2,-9/2) ,
                  integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs7_shift3(SArray<FP,1,7> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6) {')
N=7
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs7_shift4(SArray<FP,1,7> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6) {')
N=7
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs7_shift5(SArray<FP,1,7> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6) {')
N=7
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs7_shift6(SArray<FP,1,7> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6) {')
N=7
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ,
                  integrate(p,x, 9/2, 11/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs7_shift7(SArray<FP,1,7> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6) {')
N=7
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ,
                  integrate(p,x, 9/2, 11/2) ,
                  integrate(p,x, 11/2, 13/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs8_shift1(SArray<FP,1,8> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7) {')
N=8
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-15/2,-13/2) ,
                  integrate(p,x,-13/2,-11/2) ,
                  integrate(p,x,-11/2,-9/2) ,
                  integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs8_shift2(SArray<FP,1,8> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7) {')
N=8
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-13/2,-11/2) ,
                  integrate(p,x,-11/2,-9/2) ,
                  integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs8_shift3(SArray<FP,1,8> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7) {')
N=8
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-11/2,-9/2) ,
                  integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs8_shift4(SArray<FP,1,8> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7) {')
N=8
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs8_shift5(SArray<FP,1,8> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7) {')
N=8
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs8_shift6(SArray<FP,1,8> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7) {')
N=8
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ,
                  integrate(p,x, 9/2, 11/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs8_shift7(SArray<FP,1,8> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7) {')
N=8
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ,
                  integrate(p,x, 9/2, 11/2) ,
                  integrate(p,x, 11/2, 13/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs8_shift8(SArray<FP,1,8> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7) {')
N=8
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ,
                  integrate(p,x, 9/2, 11/2) ,
                  integrate(p,x, 11/2, 13/2) ,
                  integrate(p,x, 13/2, 15/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');


print('  template <class FP> YAKL_INLINE void coefs9(SArray<FP,1,9> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7, FP s8) {')
N=9
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');

print('  template <class FP> YAKL_INLINE void coefs11(SArray<FP,1,11> &coefs , FP s0 , FP s1, FP s2, FP s3, FP s4, FP s5, FP s6, FP s7, FP s8, FP s9, FP s10) {')
N=11
coefs = coefs_1d(N,0,'a')
p = poly_1d(N,coefs,x)
constr = vector([ integrate(p,x,-11/2,-9/2) ,
                  integrate(p,x,-9/2,-7/2) ,
                  integrate(p,x,-7/2,-5/2) ,
                  integrate(p,x,-5/2,-3/2) ,
                  integrate(p,x,-3/2,-1/2) ,
                  integrate(p,x,-1/2, 1/2) ,
                  integrate(p,x, 1/2, 3/2) ,
                  integrate(p,x, 3/2, 5/2) ,
                  integrate(p,x, 5/2, 7/2) ,
                  integrate(p,x, 7/2, 9/2) ,
                  integrate(p,x, 9/2, 11/2) ])
constr_to_coefs = jacobian(constr,coefs)^-1
coefs = constr_to_coefs*coefs_1d(N,0,'s')
print(add_spaces(4,c_vector('coefs',N,force_fp(coefs,129),'none')),end="")
print('  }\n');



print('}\n')

