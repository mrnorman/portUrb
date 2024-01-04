
#pragma once

namespace limiter {

  template <class T>
  YAKL_INLINE void convexify(T & w1, T & w2, T & w3) {
    T tot = w1 + w2 + w3;
    if (tot > static_cast<T>(1.e-20)) { w1 /= tot;   w2 /= tot;   w3 /= tot; }
  }


  template <class T>
  YAKL_INLINE void convexify(T & w1, T & w2, T & w3, T & w4) {
    T tot = w1 + w2 + w3 + w4;
    if (tot > static_cast<T>(1.e-20)) { w1 /= tot;   w2 /= tot;   w3 /= tot;   w4 /= tot; }
  }


  template <class T>
  YAKL_INLINE void convexify(T & w1, T & w2, T & w3, T & w4, T & w5) {
    T tot = w1 + w2 + w3 + w4 + w5;
    if (tot > static_cast<T>(1.e-20)) { w1 /= tot;   w2 /= tot;   w3 /= tot;   w4 /= tot;   w5 /= tot; }
  }


  template <class T>
  YAKL_INLINE void convexify(T & w1, T & w2, T & w3, T & w4, T & w5, T & w6) {
    T tot = w1 + w2 + w3 + w4 + w5 + w6;
    if (tot > static_cast<T>(1.e-20)) { w1 /= tot;   w2 /= tot;   w3 /= tot;   w4 /= tot;   w5 /= tot;   w6 /= tot; }
  }


  YAKL_INLINE real TV(SArray<real,1,2> &a) {
    real TV;
    TV=1.0000000000000000000000000000000000000_fp*(a(1)*a(1));

    return TV;
  }

  YAKL_INLINE float TV(SArray<real,1,3> &a) {
    float TV;
    TV=1.0000000000000000000000000000000000000f*(static_cast<float>(a(1))*static_cast<float>(a(1)))+4.3333333333333333333333333333333333333f*(static_cast<float>(a(2))*static_cast<float>(a(2)));

    return TV;
  }

  YAKL_INLINE real TV(SArray<real,1,4> &a) {
    real TV;
    TV=1.0000000000000000000000000000000000000_fp*(a(1)*a(1))+4.3333333333333333333333333333333333333_fp*(a(2)*a(2))+0.50000000000000000000000000000000000000_fp*a(1)*a(3)+39.112500000000000000000000000000000000_fp*(a(3)*a(3));

    return TV;
  }

  YAKL_INLINE float TV(SArray<real,1,5> &a) {
    float TV;
    TV=1.0000000000000000000000000000000000000f*(static_cast<float>(a(1))*static_cast<float>(a(1)))+4.3333333333333333333333333333333333333f*(static_cast<float>(a(2))*static_cast<float>(a(2)))+0.50000000000000000000000000000000000000f*static_cast<float>(a(1))*static_cast<float>(a(3))+39.112500000000000000000000000000000000f*(static_cast<float>(a(3))*static_cast<float>(a(3)))+4.2000000000000000000000000000000000000f*static_cast<float>(a(2))*static_cast<float>(a(4))+625.83571428571428571428571428571428571f*(static_cast<float>(a(4))*static_cast<float>(a(4)));

    return TV;
  }

  YAKL_INLINE real TV(SArray<real,1,7> &a) {
    real TV;
    TV=1.0000000000000000000000000000000000000_fp*(a(1)*a(1))+4.3333333333333333333333333333333333333_fp*(a(2)*a(2))+0.50000000000000000000000000000000000000_fp*a(1)*a(3)+39.112500000000000000000000000000000000_fp*(a(3)*a(3))+4.2000000000000000000000000000000000000_fp*a(2)*a(4)+625.83571428571428571428571428571428571_fp*(a(4)*a(4))+0.12500000000000000000000000000000000000_fp*a(1)*a(5)+63.066964285714285714285714285714285714_fp*a(3)*a(5)+15645.903707837301587301587301587301587_fp*(a(5)*a(5))+1.5535714285714285714285714285714285714_fp*a(2)*a(6)+1513.6279761904761904761904761904761905_fp*a(4)*a(6)+563252.53667816558441558441558441558442_fp*(a(6)*a(6));

    return TV;
  }

  YAKL_INLINE real TV(SArray<real,1,9> &a) {
    real TV;
    TV=1.0000000000000000000000000000000000000_fp*(a(1)*a(1))+4.3333333333333333333333333333333333333_fp*(a(2)*a(2))+0.50000000000000000000000000000000000000_fp*a(1)*a(3)+39.112500000000000000000000000000000000_fp*(a(3)*a(3))+4.2000000000000000000000000000000000000_fp*a(2)*a(4)+625.83571428571428571428571428571428571_fp*(a(4)*a(4))+0.12500000000000000000000000000000000000_fp*a(1)*a(5)+63.066964285714285714285714285714285714_fp*a(3)*a(5)+15645.903707837301587301587301587301587_fp*(a(5)*a(5))+1.5535714285714285714285714285714285714_fp*a(2)*a(6)+1513.6279761904761904761904761904761905_fp*a(4)*a(6)+563252.53667816558441558441558441558442_fp*(a(6)*a(6))+0.031250000000000000000000000000000000000_fp*a(1)*a(7)+32.643229166666666666666666666666666667_fp*a(3)*a(7)+52976.985381155303030303030303030303030_fp*a(5)*a(7)+2.7599374298150335992132867132867132867e7_fp*(a(7)*a(7))+0.51388888888888888888888888888888888889_fp*a(2)*a(8)+1044.5890151515151515151515151515151515_fp*a(4)*a(8)+2.5428953000983391608391608391608391608e6_fp*a(6)*a(8)+1.7663599550818819201631701631701631702e9_fp*(a(8)*a(8));

    return TV;
  }

  YAKL_INLINE void coefs2_shift1(SArray<real,1,2> &coefs2_1, real v0, real v1) {
    coefs2_1(0)=1.0000000000000000000000000000000000000_fp*v1;
    coefs2_1(1)=-1.0000000000000000000000000000000000000_fp*v0+1.0000000000000000000000000000000000000_fp*v1;

  }

  YAKL_INLINE void coefs2_shift2(SArray<real,1,2> &coefs2_2, real v0, real v1) {
    coefs2_2(0)=1.0000000000000000000000000000000000000_fp*v0;
    coefs2_2(1)=-1.0000000000000000000000000000000000000_fp*v0+1.0000000000000000000000000000000000000_fp*v1;

  }

  YAKL_INLINE void coefs3_shift1(SArray<real,1,3> &coefs3_1, real v0, real v1, real v2) {
    coefs3_1(0)=-0.041666666666666666666666666666666666667_fp*v0+0.083333333333333333333333333333333333333_fp*v1+0.95833333333333333333333333333333333333_fp*v2;
    coefs3_1(1)=0.50000000000000000000000000000000000000_fp*v0-2.0000000000000000000000000000000000000_fp*v1+1.5000000000000000000000000000000000000_fp*v2;
    coefs3_1(2)=0.50000000000000000000000000000000000000_fp*v0-1.0000000000000000000000000000000000000_fp*v1+0.50000000000000000000000000000000000000_fp*v2;

  }

  YAKL_INLINE void coefs3_shift2(SArray<real,1,3> &coefs3_2, real v0, real v1, real v2) {
    coefs3_2(0)=-0.041666666666666666666666666666666666667_fp*v0+1.0833333333333333333333333333333333333_fp*v1-0.041666666666666666666666666666666666667_fp*v2;
    coefs3_2(1)=-0.50000000000000000000000000000000000000_fp*v0+0.50000000000000000000000000000000000000_fp*v2;
    coefs3_2(2)=0.50000000000000000000000000000000000000_fp*v0-1.0000000000000000000000000000000000000_fp*v1+0.50000000000000000000000000000000000000_fp*v2;

  }

  YAKL_INLINE void coefs3_shift3(SArray<real,1,3> &coefs3_3, real v0, real v1, real v2) {
    coefs3_3(0)=0.95833333333333333333333333333333333333_fp*v0+0.083333333333333333333333333333333333333_fp*v1-0.041666666666666666666666666666666666667_fp*v2;
    coefs3_3(1)=-1.5000000000000000000000000000000000000_fp*v0+2.0000000000000000000000000000000000000_fp*v1-0.50000000000000000000000000000000000000_fp*v2;
    coefs3_3(2)=0.50000000000000000000000000000000000000_fp*v0-1.0000000000000000000000000000000000000_fp*v1+0.50000000000000000000000000000000000000_fp*v2;

  }

  YAKL_INLINE void coefs4_shift1(SArray<real,1,4> &coefs4_1, real v0, real v1, real v2, real v3) {
    coefs4_1(0)=0.041666666666666666666666666666666666667_fp*v0-0.16666666666666666666666666666666666667_fp*v1+0.20833333333333333333333333333333333333_fp*v2+0.91666666666666666666666666666666666667_fp*v3;
    coefs4_1(1)=-0.29166666666666666666666666666666666667_fp*v0+1.3750000000000000000000000000000000000_fp*v1-2.8750000000000000000000000000000000000_fp*v2+1.7916666666666666666666666666666666667_fp*v3;
    coefs4_1(2)=-0.50000000000000000000000000000000000000_fp*v0+2.0000000000000000000000000000000000000_fp*v1-2.5000000000000000000000000000000000000_fp*v2+1.0000000000000000000000000000000000000_fp*v3;
    coefs4_1(3)=-0.16666666666666666666666666666666666667_fp*v0+0.50000000000000000000000000000000000000_fp*v1-0.50000000000000000000000000000000000000_fp*v2+0.16666666666666666666666666666666666667_fp*v3;

  }

  YAKL_INLINE void coefs4_shift2(SArray<real,1,4> &coefs4_2, real v0, real v1, real v2, real v3) {
    coefs4_2(0)=-0.041666666666666666666666666666666666667_fp*v1+1.0833333333333333333333333333333333333_fp*v2-0.041666666666666666666666666666666666667_fp*v3;
    coefs4_2(1)=0.20833333333333333333333333333333333333_fp*v0-1.1250000000000000000000000000000000000_fp*v1+0.62500000000000000000000000000000000000_fp*v2+0.29166666666666666666666666666666666667_fp*v3;
    coefs4_2(2)=0.50000000000000000000000000000000000000_fp*v1-1.0000000000000000000000000000000000000_fp*v2+0.50000000000000000000000000000000000000_fp*v3;
    coefs4_2(3)=-0.16666666666666666666666666666666666667_fp*v0+0.50000000000000000000000000000000000000_fp*v1-0.50000000000000000000000000000000000000_fp*v2+0.16666666666666666666666666666666666667_fp*v3;

  }

  YAKL_INLINE void coefs4_shift3(SArray<real,1,4> &coefs4_3, real v0, real v1, real v2, real v3) {
    coefs4_3(0)=-0.041666666666666666666666666666666666667_fp*v0+1.0833333333333333333333333333333333333_fp*v1-0.041666666666666666666666666666666666667_fp*v2;
    coefs4_3(1)=-0.29166666666666666666666666666666666667_fp*v0-0.62500000000000000000000000000000000000_fp*v1+1.1250000000000000000000000000000000000_fp*v2-0.20833333333333333333333333333333333333_fp*v3;
    coefs4_3(2)=0.50000000000000000000000000000000000000_fp*v0-1.0000000000000000000000000000000000000_fp*v1+0.50000000000000000000000000000000000000_fp*v2;
    coefs4_3(3)=-0.16666666666666666666666666666666666667_fp*v0+0.50000000000000000000000000000000000000_fp*v1-0.50000000000000000000000000000000000000_fp*v2+0.16666666666666666666666666666666666667_fp*v3;

  }

  YAKL_INLINE void coefs4_shift4(SArray<real,1,4> &coefs4_4, real v0, real v1, real v2, real v3) {
    coefs4_4(0)=0.91666666666666666666666666666666666667_fp*v0+0.20833333333333333333333333333333333333_fp*v1-0.16666666666666666666666666666666666667_fp*v2+0.041666666666666666666666666666666666667_fp*v3;
    coefs4_4(1)=-1.7916666666666666666666666666666666667_fp*v0+2.8750000000000000000000000000000000000_fp*v1-1.3750000000000000000000000000000000000_fp*v2+0.29166666666666666666666666666666666667_fp*v3;
    coefs4_4(2)=1.0000000000000000000000000000000000000_fp*v0-2.5000000000000000000000000000000000000_fp*v1+2.0000000000000000000000000000000000000_fp*v2-0.50000000000000000000000000000000000000_fp*v3;
    coefs4_4(3)=-0.16666666666666666666666666666666666667_fp*v0+0.50000000000000000000000000000000000000_fp*v1-0.50000000000000000000000000000000000000_fp*v2+0.16666666666666666666666666666666666667_fp*v3;

  }

  YAKL_INLINE void coefs5_shift1(SArray<real,1,5> &coefs5_1, real v0, real v1, real v2, real v3, real v4) {
    coefs5_1(0)=-0.036979166666666666666666666666666666667_fp*v0+0.18958333333333333333333333333333333333_fp*v1-0.38854166666666666666666666666666666667_fp*v2+0.35625000000000000000000000000000000000_fp*v3+0.87968750000000000000000000000000000000_fp*v4;
    coefs5_1(1)=0.18750000000000000000000000000000000000_fp*v0-1.0416666666666666666666666666666666667_fp*v1+2.5000000000000000000000000000000000000_fp*v2-3.6250000000000000000000000000000000000_fp*v3+1.9791666666666666666666666666666666667_fp*v4;
    coefs5_1(2)=0.43750000000000000000000000000000000000_fp*v0-2.2500000000000000000000000000000000000_fp*v1+4.6250000000000000000000000000000000000_fp*v2-4.2500000000000000000000000000000000000_fp*v3+1.4375000000000000000000000000000000000_fp*v4;
    coefs5_1(3)=0.25000000000000000000000000000000000000_fp*v0-1.1666666666666666666666666666666666667_fp*v1+2.0000000000000000000000000000000000000_fp*v2-1.5000000000000000000000000000000000000_fp*v3+0.41666666666666666666666666666666666667_fp*v4;
    coefs5_1(4)=0.041666666666666666666666666666666666667_fp*v0-0.16666666666666666666666666666666666667_fp*v1+0.25000000000000000000000000000000000000_fp*v2-0.16666666666666666666666666666666666667_fp*v3+0.041666666666666666666666666666666666667_fp*v4;

  }

  YAKL_INLINE void coefs5_shift2(SArray<real,1,5> &coefs5_2, real v0, real v1, real v2, real v3, real v4) {
    coefs5_2(0)=0.0046875000000000000000000000000000000000_fp*v0-0.018750000000000000000000000000000000000_fp*v1-0.013541666666666666666666666666666666667_fp*v2+1.0645833333333333333333333333333333333_fp*v3-0.036979166666666666666666666666666666667_fp*v4;
    coefs5_2(1)=-0.10416666666666666666666666666666666667_fp*v0+0.62500000000000000000000000000000000000_fp*v1-1.7500000000000000000000000000000000000_fp*v2+1.0416666666666666666666666666666666667_fp*v3+0.18750000000000000000000000000000000000_fp*v4;
    coefs5_2(2)=-0.062500000000000000000000000000000000000_fp*v0+0.25000000000000000000000000000000000000_fp*v1+0.12500000000000000000000000000000000000_fp*v2-0.75000000000000000000000000000000000000_fp*v3+0.43750000000000000000000000000000000000_fp*v4;
    coefs5_2(3)=0.083333333333333333333333333333333333333_fp*v0-0.50000000000000000000000000000000000000_fp*v1+1.0000000000000000000000000000000000000_fp*v2-0.83333333333333333333333333333333333333_fp*v3+0.25000000000000000000000000000000000000_fp*v4;
    coefs5_2(4)=0.041666666666666666666666666666666666667_fp*v0-0.16666666666666666666666666666666666667_fp*v1+0.25000000000000000000000000000000000000_fp*v2-0.16666666666666666666666666666666666667_fp*v3+0.041666666666666666666666666666666666667_fp*v4;

  }

  YAKL_INLINE void coefs5_shift3(SArray<real,1,5> &coefs5_3, real v0, real v1, real v2, real v3, real v4) {
    coefs5_3(0)=0.0046875000000000000000000000000000000000_fp*v0-0.060416666666666666666666666666666666667_fp*v1+1.1114583333333333333333333333333333333_fp*v2-0.060416666666666666666666666666666666667_fp*v3+0.0046875000000000000000000000000000000000_fp*v4;
    coefs5_3(1)=0.10416666666666666666666666666666666667_fp*v0-0.70833333333333333333333333333333333333_fp*v1+0.70833333333333333333333333333333333333_fp*v3-0.10416666666666666666666666666666666667_fp*v4;
    coefs5_3(2)=-0.062500000000000000000000000000000000000_fp*v0+0.75000000000000000000000000000000000000_fp*v1-1.3750000000000000000000000000000000000_fp*v2+0.75000000000000000000000000000000000000_fp*v3-0.062500000000000000000000000000000000000_fp*v4;
    coefs5_3(3)=-0.083333333333333333333333333333333333333_fp*v0+0.16666666666666666666666666666666666667_fp*v1-0.16666666666666666666666666666666666667_fp*v3+0.083333333333333333333333333333333333333_fp*v4;
    coefs5_3(4)=0.041666666666666666666666666666666666667_fp*v0-0.16666666666666666666666666666666666667_fp*v1+0.25000000000000000000000000000000000000_fp*v2-0.16666666666666666666666666666666666667_fp*v3+0.041666666666666666666666666666666666667_fp*v4;

  }

  YAKL_INLINE void coefs5_shift4(SArray<real,1,5> &coefs5_4, real v0, real v1, real v2, real v3, real v4) {
    coefs5_4(0)=-0.036979166666666666666666666666666666667_fp*v0+1.0645833333333333333333333333333333333_fp*v1-0.013541666666666666666666666666666666667_fp*v2-0.018750000000000000000000000000000000000_fp*v3+0.0046875000000000000000000000000000000000_fp*v4;
    coefs5_4(1)=-0.18750000000000000000000000000000000000_fp*v0-1.0416666666666666666666666666666666667_fp*v1+1.7500000000000000000000000000000000000_fp*v2-0.62500000000000000000000000000000000000_fp*v3+0.10416666666666666666666666666666666667_fp*v4;
    coefs5_4(2)=0.43750000000000000000000000000000000000_fp*v0-0.75000000000000000000000000000000000000_fp*v1+0.12500000000000000000000000000000000000_fp*v2+0.25000000000000000000000000000000000000_fp*v3-0.062500000000000000000000000000000000000_fp*v4;
    coefs5_4(3)=-0.25000000000000000000000000000000000000_fp*v0+0.83333333333333333333333333333333333333_fp*v1-1.0000000000000000000000000000000000000_fp*v2+0.50000000000000000000000000000000000000_fp*v3-0.083333333333333333333333333333333333333_fp*v4;
    coefs5_4(4)=0.041666666666666666666666666666666666667_fp*v0-0.16666666666666666666666666666666666667_fp*v1+0.25000000000000000000000000000000000000_fp*v2-0.16666666666666666666666666666666666667_fp*v3+0.041666666666666666666666666666666666667_fp*v4;

  }

  YAKL_INLINE void coefs5_shift5(SArray<real,1,5> &coefs5_5, real v0, real v1, real v2, real v3, real v4) {
    coefs5_5(0)=0.87968750000000000000000000000000000000_fp*v0+0.35625000000000000000000000000000000000_fp*v1-0.38854166666666666666666666666666666667_fp*v2+0.18958333333333333333333333333333333333_fp*v3-0.036979166666666666666666666666666666667_fp*v4;
    coefs5_5(1)=-1.9791666666666666666666666666666666667_fp*v0+3.6250000000000000000000000000000000000_fp*v1-2.5000000000000000000000000000000000000_fp*v2+1.0416666666666666666666666666666666667_fp*v3-0.18750000000000000000000000000000000000_fp*v4;
    coefs5_5(2)=1.4375000000000000000000000000000000000_fp*v0-4.2500000000000000000000000000000000000_fp*v1+4.6250000000000000000000000000000000000_fp*v2-2.2500000000000000000000000000000000000_fp*v3+0.43750000000000000000000000000000000000_fp*v4;
    coefs5_5(3)=-0.41666666666666666666666666666666666667_fp*v0+1.5000000000000000000000000000000000000_fp*v1-2.0000000000000000000000000000000000000_fp*v2+1.1666666666666666666666666666666666667_fp*v3-0.25000000000000000000000000000000000000_fp*v4;
    coefs5_5(4)=0.041666666666666666666666666666666666667_fp*v0-0.16666666666666666666666666666666666667_fp*v1+0.25000000000000000000000000000000000000_fp*v2-0.16666666666666666666666666666666666667_fp*v3+0.041666666666666666666666666666666666667_fp*v4;

  }

  YAKL_INLINE void coefs7(SArray<real,1,7> &coefs7, real v0, real v1, real v2, real v3, real v4, real v5, real v6) {
    coefs7(0)=-0.00069754464285714285714285714285714285714_fp*v0+0.0088727678571428571428571428571428571429_fp*v1-0.070879836309523809523809523809523809524_fp*v2+1.1254092261904761904761904761904761905_fp*v3-0.070879836309523809523809523809523809524_fp*v4+0.0088727678571428571428571428571428571429_fp*v5-0.00069754464285714285714285714285714285714_fp*v6;
    coefs7(1)=-0.022482638888888888888888888888888888889_fp*v0+0.19409722222222222222222222222222222222_fp*v1-0.82074652777777777777777777777777777778_fp*v2+0.82074652777777777777777777777777777778_fp*v4-0.19409722222222222222222222222222222222_fp*v5+0.022482638888888888888888888888888888889_fp*v6;
    coefs7(2)=0.0096354166666666666666666666666666666667_fp*v0-0.12031250000000000000000000000000000000_fp*v1+0.89453125000000000000000000000000000000_fp*v2-1.5677083333333333333333333333333333333_fp*v3+0.89453125000000000000000000000000000000_fp*v4-0.12031250000000000000000000000000000000_fp*v5+0.0096354166666666666666666666666666666667_fp*v6;
    coefs7(3)=0.024305555555555555555555555555555555556_fp*v0-0.18055555555555555555555555555555555556_fp*v1+0.28819444444444444444444444444444444444_fp*v2-0.28819444444444444444444444444444444444_fp*v4+0.18055555555555555555555555555555555556_fp*v5-0.024305555555555555555555555555555555556_fp*v6;
    coefs7(4)=-0.0086805555555555555555555555555555555555_fp*v0+0.093750000000000000000000000000000000000_fp*v1-0.29687500000000000000000000000000000000_fp*v2+0.42361111111111111111111111111111111111_fp*v3-0.29687500000000000000000000000000000000_fp*v4+0.093750000000000000000000000000000000000_fp*v5-0.0086805555555555555555555555555555555555_fp*v6;
    coefs7(5)=-0.0041666666666666666666666666666666666667_fp*v0+0.016666666666666666666666666666666666667_fp*v1-0.020833333333333333333333333333333333333_fp*v2+0.020833333333333333333333333333333333333_fp*v4-0.016666666666666666666666666666666666667_fp*v5+0.0041666666666666666666666666666666666667_fp*v6;
    coefs7(6)=0.0013888888888888888888888888888888888889_fp*v0-0.0083333333333333333333333333333333333333_fp*v1+0.020833333333333333333333333333333333333_fp*v2-0.027777777777777777777777777777777777778_fp*v3+0.020833333333333333333333333333333333333_fp*v4-0.0083333333333333333333333333333333333333_fp*v5+0.0013888888888888888888888888888888888889_fp*v6;

  }

  YAKL_INLINE void coefs9(SArray<real,1,9> &coefs9, real v0, real v1, real v2, real v3, real v4, real v5, real v6, real v7, real v8) {
    coefs9(0)=0.00011867947048611111111111111111111111111_fp*v0-0.0016469804067460317460317460317460317460_fp*v1+0.012195793030753968253968253968253968254_fp*v2-0.077525886656746031746031746031746031746_fp*v3+1.1337167891245039682539682539682539683_fp*v4-0.077525886656746031746031746031746031746_fp*v5+0.012195793030753968253968253968253968254_fp*v6-0.0016469804067460317460317460317460317460_fp*v7+0.00011867947048611111111111111111111111111_fp*v8;
    coefs9(1)=0.0050052703373015873015873015873015873016_fp*v0-0.052514260912698412698412698412698412698_fp*v1+0.26417100694444444444444444444444444444_fp*v2-0.89082031250000000000000000000000000000_fp*v3+0.89082031250000000000000000000000000000_fp*v5-0.26417100694444444444444444444444444444_fp*v6+0.052514260912698412698412698412698412698_fp*v7-0.0050052703373015873015873015873015873016_fp*v8;
    coefs9(2)=-0.0016684234457671957671957671957671957672_fp*v0+0.022982804232804232804232804232804232804_fp*v1-0.16702835648148148148148148148148148148_fp*v2+0.98796296296296296296296296296296296296_fp*v3-1.6844979745370370370370370370370370370_fp*v4+0.98796296296296296296296296296296296296_fp*v5-0.16702835648148148148148148148148148148_fp*v6+0.022982804232804232804232804232804232804_fp*v7-0.0016684234457671957671957671957671957672_fp*v8;
    coefs9(3)=-0.0061197916666666666666666666666666666667_fp*v0+0.061024305555555555555555555555555555556_fp*v1-0.26623263888888888888888888888888888889_fp*v2+0.37387152777777777777777777777777777778_fp*v3-0.37387152777777777777777777777777777778_fp*v5+0.26623263888888888888888888888888888889_fp*v6-0.061024305555555555555555555555555555556_fp*v7+0.0061197916666666666666666666666666666667_fp*v8;
    coefs9(4)=0.0016999421296296296296296296296296296296_fp*v0-0.022280092592592592592592592592592592593_fp*v1+0.14134837962962962962962962962962962963_fp*v2-0.39207175925925925925925925925925925926_fp*v3+0.54260706018518518518518518518518518518_fp*v4-0.39207175925925925925925925925925925926_fp*v5+0.14134837962962962962962962962962962963_fp*v6-0.022280092592592592592592592592592592593_fp*v7+0.0016999421296296296296296296296296296296_fp*v8;
    coefs9(5)=0.0015625000000000000000000000000000000000_fp*v0-0.013541666666666666666666666666666666667_fp*v1+0.038541666666666666666666666666666666667_fp*v2-0.042708333333333333333333333333333333333_fp*v3+0.042708333333333333333333333333333333333_fp*v5-0.038541666666666666666666666666666666667_fp*v6+0.013541666666666666666666666666666666667_fp*v7-0.0015625000000000000000000000000000000000_fp*v8;
    coefs9(6)=-0.00040509259259259259259259259259259259259_fp*v0+0.0046296296296296296296296296296296296296_fp*v1-0.019675925925925925925925925925925925926_fp*v2+0.043518518518518518518518518518518518519_fp*v3-0.056134259259259259259259259259259259259_fp*v4+0.043518518518518518518518518518518518519_fp*v5-0.019675925925925925925925925925925925926_fp*v6+0.0046296296296296296296296296296296296296_fp*v7-0.00040509259259259259259259259259259259259_fp*v8;
    coefs9(7)=-0.000099206349206349206349206349206349206349_fp*v0+0.00059523809523809523809523809523809523810_fp*v1-0.0013888888888888888888888888888888888889_fp*v2+0.0013888888888888888888888888888888888889_fp*v3-0.0013888888888888888888888888888888888889_fp*v5+0.0013888888888888888888888888888888888889_fp*v6-0.00059523809523809523809523809523809523810_fp*v7+0.000099206349206349206349206349206349206349_fp*v8;
    coefs9(8)=0.000024801587301587301587301587301587301587_fp*v0-0.00019841269841269841269841269841269841270_fp*v1+0.00069444444444444444444444444444444444444_fp*v2-0.0013888888888888888888888888888888888889_fp*v3+0.0017361111111111111111111111111111111111_fp*v4-0.0013888888888888888888888888888888888889_fp*v5+0.00069444444444444444444444444444444444444_fp*v6-0.00019841269841269841269841269841269841270_fp*v7+0.000024801587301587301587301587301587301587_fp*v8;

  }

}
