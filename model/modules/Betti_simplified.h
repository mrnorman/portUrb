
#include "main_header.h"
#include <random>

namespace modules {

  // TODO: For some regimes, we can try blending in the Joint North Sea Wave Observation Project JONSWAP spectrum
  struct Floating_motions_betti {
    int     static constexpr nfreq = 400; // Number of frequency intervals to sum over in PM spectrum
    real    static constexpr dt_max         = 0.05;
    size_t  static constexpr rand_pool_size = 1024*10;
    SArray<real,1,6>               state;      // Current state vector
    real                           etime;      // Current elapsed time
    std::vector<real>              rand_pool;
    int                            rand_pool_counter;


    // Assume arr is ordered lowest to highest. Return index of arr(index) nearest to "val"
    int nearest_index(realHost1d const & arr, real val) {
      int  ind      = 0;
      real min_diff = std::abs(arr(0)-val);
      for (int i=1; i < arr.size(); i++) {
        real diff = std::abs(arr(i)-val);
        if (diff < min_diff) { ind = i; min_diff = diff; }
      }
      return ind;
    }


    // TODO: Add some time steps for a spin-up phase in here
    void init() {
      this->state(0)  = -2;
      this->state(1)  = 0;
      this->state(2)  = 37.550;
      this->state(3)  = 0;
      this->state(4)  = 0;
      this->state(5)  = 0;
      this->etime     = 0;
      rand_pool = std::vector<real>(rand_pool_size);
      std::random_device rd;
      std::mt19937 generator(137); // TODO: Undo this????
      std::uniform_real_distribution<real> distribution(0.,2.*M_PI);
      for (int i=0; i < rand_pool_size; i++) { rand_pool.at(i) = distribution(generator); }
      rand_pool_counter = 0;
    }


    // Computes Pierson Moskowitz spectrum outputs
    // wind_19_5m: average wind velocity at 19.5m (m/s)
    // zeta      : "the x component to evaluate"
    // eta       : "the y component to evaluate .Note: the coordinate system here is different
    //                from the Betti model. The downward is negative in this case"
    // t         : the time to evaluate (s)
    // N         : The number of frequency intervals to use
    // returns  : Tuple of [wave_eta,v_x,v_y,a_x,a_y]
    // wave_eta : Wave elevation (m)
    // v_x      : x-direction wave velocity
    // v_y      : y-direction wave velocity
    // a_x      : x-direction wave acceleration
    // a_y      : y-direction wave acceleration
    // N = 400 does the job well enough
    // Fully develped oceans have essentially random wave phases at different frequencies. This spectrum empirically
    //   provides significant wave heights and pitch, roll, surge, heave, sway patterns. This approximates local wind
    //   waves and not long term swells from larger systems.
    // TODO: Experiment with different values of f_pm
    std::tuple<real,real,real,real,real>
    pierson_moskowitz_spectrum( real wind_19_5m , real zeta , real eta , real t , int N) {
      real constexpr g       = 9.81;           // gravity acceleration
      real constexpr alpha   = 0.0081;         // Phillip's constant
      real constexpr start_f = 0.1;            // starting frequency
      real f_pm    = 0.14*(g/wind_19_5m); // peak frequency
      real cutof_f = 3*f_pm;         // cutoff frequency
      real sum_wave_eta = 0;
      real sum_v_x      = 0;
      real sum_v_y      = 0;
      real sum_a_x      = 0;
      real sum_a_y      = 0;
      for (int i=0; i < N; i++) {
        real random_phase  = rand_pool.at(rand_pool_counter++);
        if (rand_pool_counter >= rand_pool_size) rand_pool_counter = 0;
        real delta_f       = (cutof_f - start_f) / (N-1);
        real f             = start_f + i*delta_f;
        real omega         = f*2*M_PI;
        real pi_term       = std::pow(2.*M_PI,4.);
        real f_term1       = std::pow(f,5.);
        real f_term2       = std::pow(f_pm/f,4.);
        real S_pm          = (alpha*g*g /(pi_term*f_term1))*std::exp(-(5./4.)*f_term2);
        real a             = std::sqrt(2*S_pm*delta_f);
        real k             = omega*omega/g;
        real sin_component = std::sin(omega*t - k*zeta + random_phase);
        real cos_component = std::cos(omega*t - k*zeta + random_phase);
        real exp_component = std::exp(k*eta);
        sum_wave_eta += a*sin_component;
        sum_v_x      += omega*a*exp_component*sin_component;
        sum_v_y      += omega*a*exp_component*cos_component;
        sum_a_x      += omega*omega*a*exp_component*cos_component;
        sum_a_y      -= omega*omega*a*exp_component*sin_component;
      }
      return std::make_tuple( sum_wave_eta , sum_v_x , sum_v_y , sum_a_x , sum_a_y );
    }


    std::tuple<SArray<real,1,6>,real,real>
    structure( SArray<real,1,6> const & x_1 , real t , real turbine_wind , real wind_19_5m , real Ct ) {
      real constexpr g       = 9.80665 ;         // (m/s^2)  gravity acceleration
      real constexpr rho_w   = 1025    ;         // (kg/m^3) water density
      real constexpr M_N     = 240000  ;         // (kg)     Mass of nacelle
      real constexpr M_P     = 110000  ;         // (kg)     Mass of blades and hub
      real constexpr M_S     = 8947870 ;         // (kg)     Mass of "structure" (tower and floater)
      real constexpr m_x     = 11127000;         // (kg)     Added mass in horizontal direction
      real constexpr m_y     = 1504400 ;         // (kg)     Added mass in vertical direction
      real constexpr d_Nh    = -1.8    ;         // (m)      Horizontal distance between BS and BN
      real constexpr d_Nv    = 126.9003;         // (m)      Vertical distance between BS and BN
      real constexpr d_Ph    = 5.4305  ;         // (m)      Horizontal distance between BS and BP
      real constexpr d_Pv    = 127.5879;         // (m)      Vertical distance between BS and BP
      real constexpr J_S     = 3.4917e9;         // (kg*m^2) "Structure" moment of inertia
      real constexpr J_N     = 2607890 ;         // (kg*m^2) Nacelle moment of inertia
      real constexpr J_P     = 50365000;         // (kg*m^2) Blades, hub and low speed shaft moment of inertia
      real constexpr h       = 200     ;         // (m)      Depth of water
      real constexpr h_pt    = 47.89   ;         // (m)      Height of the floating structure
      real constexpr r_g     = 9       ;         // (m)      Radius of floater
      real constexpr d_Sbott = 10.3397 ;         // (m)      Vertical distance between BS and floater bottom
      real constexpr r_tb    = 3       ;         // (m)      Maximum radius of the tower
      real constexpr d_t     = 10.3397 ;         // (m)      Vertical distance between BS and hooks of tie rods
      real constexpr l_a     = 27      ;         // (m)      Distance between the hooks of tie rods
      real constexpr l_0     = 151.73  ;         // (m)      Rest length of tie rods
      real constexpr K_T1    = 2*(1.5/l_0)*1.e9; // (N/m)    Spring constant of lateral tie rods
      real constexpr K_T2    = 2*(1.5/l_0)*1.e9; // (N/m)    Spring constant of lateral tie rods
      real constexpr K_T3    = 4*(1.5/l_0)*1.e9; // (N/m)    Spring constant of central tie rod
      real constexpr d_T     = 75.7843 ;         // (m)      Vertical distance between BS and BT
      real constexpr rho     = 1.225   ;         // (kg/m^3) Density of air
      real constexpr C_dN    = 1       ;         // (-)      Nacelle drag coefficient
      real constexpr A_N     = 9.62    ;         // (m^2)    Nacelle area
      real constexpr C_dT    = 1       ;         // (-)      tower drag coefficient
      real constexpr A       = 12469   ;         // (m^2)    Rotor area
      real constexpr n_dg    = 2       ;         //（-）     Number of floater sub-cylinders
      real constexpr C_dgper = 1       ;         // (-)      Perpendicular cylinder drag coefficient
      real constexpr C_dgpar = 0.006   ;         // (-)      Parallel cylinder drag coefficient
      real constexpr C_dgb   = 1.9     ;         // (-)      Floater bottom drag coefficient
      real constexpr R       = 63      ;         // (m)      Radius of rotor
      real constexpr den_l   = 116.027 ;         // (kg/m)   the mass density of the mooring lines
      real constexpr dia_l   = 0.127   ;         // (m)      the diameter of the mooring lines
      real constexpr h_T     = 87.6    ;         // (m)      the height of the tower
      real constexpr D_T     = 4.935   ;         // (m)      the main diameter of the tower
      real zeta   = x_1(0); // surge (x) position
      real v_zeta = x_1(1); // surge velocity
      real eta    = x_1(2); // heave (y) position
      real v_eta  = x_1(3); // heave velocity
      real alpha  = x_1(4); // pitch position
      real omega  = x_1(5); // pitch velocity    
      real M_X   = M_S + m_x + M_N + M_P;
      real M_Y   = M_S + m_y + M_N + M_P;
      real d_N   = std::sqrt(d_Nh*d_Nh + d_Nv*d_Nv);
      real d_P   = std::sqrt(d_Ph*d_Ph + d_Pv*d_Pv);
      real M_d   = M_N*d_N + M_P*d_P;
      real J_TOT = J_S + J_N + J_P + M_N*d_N*d_N + M_P*d_P*d_P;
      real sin_alpha = std::sin(alpha);
      real cos_alpha = std::cos(alpha);
      SArray<real,2,6,6> E;
      E = 0;
      E(0,0) = 1            ;
      E(1,1) = M_X          ;
      E(1,5) = M_d*cos_alpha;
      E(2,2) = 1            ;
      E(3,3) = M_Y          ;
      E(3,5) = M_d*sin_alpha;
      E(4,4) = 1            ;
      E(5,1) = M_d*cos_alpha;
      E(5,3) = M_d*sin_alpha;
      E(5,5) = J_TOT        ;

      // Weight Forces
      real Qwe_zeta  = 0;
      real Qwe_eta   = (M_N + M_P + M_S)*g;
      real Qwe_alpha = ((M_N*d_Nv + M_P*d_Pv)*sin_alpha + (M_N*d_Nh + M_P*d_Ph )*cos_alpha)*g;

      // Buoyancy Forces d00 - d12 are unused dummy variables
      auto [h_wave,d01,d02,d03,d04] = pierson_moskowitz_spectrum( wind_19_5m , zeta       , 0 , t , nfreq );
      auto [h_p_rg,d05,d06,d07,d08] = pierson_moskowitz_spectrum( wind_19_5m , zeta + r_g , 0 , t , nfreq );
      auto [h_n_rg,d09,d10,d11,d12] = pierson_moskowitz_spectrum( wind_19_5m , zeta - r_g , 0 , t , nfreq );
      h_wave += h;
      h_p_rg += h;
      h_n_rg += h;
      real h_w      = (h_wave + h_p_rg + h_n_rg)/3;
      real h_sub    = std::min( h_w - h + eta + d_Sbott , h_pt );
      real d_G      = eta - h_sub/2;
      real V_g      = h_sub*M_PI*r_g*r_g + std::max( (h_w - h + eta + d_Sbott) - h_pt , 0._fp )*M_PI*r_tb*r_tb;
      real Qb_zeta  = 0;
      real Qb_eta   = -rho_w*V_g*g;
      real Qb_alpha = -rho_w*V_g*g*d_G*sin_alpha;

      // Tie Rod Force
      real D_x        = l_a;
      real l_1        = std::sqrt(std::pow(h   - eta  - l_a*sin_alpha - d_t*cos_alpha,2._fp) +
                                  std::pow(D_x - zeta - l_a*cos_alpha + d_t*sin_alpha,2._fp));
      real l_2        = std::sqrt(std::pow(h   - eta  + l_a*sin_alpha - d_t*cos_alpha,2._fp) +
                                  std::pow(D_x + zeta - l_a*cos_alpha - d_t*sin_alpha,2._fp));
      real l_3        = std::sqrt(std::pow(h   - eta  - d_t*cos_alpha,2._fp) +
                                  std::pow(      zeta - d_t*sin_alpha,2._fp));
      real f_1        = std::max( 0._fp , K_T1*(l_1 - l_0) );
      real f_2        = std::max( 0._fp , K_T2*(l_2 - l_0) );
      real f_3        = std::max( 0._fp , K_T3*(l_3 - l_0) );
      real theta_1    = std::atan((D_x - zeta - l_a*cos_alpha + d_t*sin_alpha) /
                                  (h   - eta  - l_a*sin_alpha - d_t*cos_alpha));
      real theta_2    = std::atan((D_x + zeta - l_a*cos_alpha - d_t*sin_alpha) /
                                  (h   - eta  + l_a*sin_alpha - d_t*cos_alpha));
      real theta_3    = std::atan((      zeta - d_t*sin_alpha) /
                                  (h   - eta  - d_t*cos_alpha));
      real v_tir      = (0.5*dia_l)*(0.5*dia_l)*M_PI;
      real w_tir      = den_l*g;
      real b_tir      = rho_w*g*v_tir;
      real lambda_tir = w_tir - b_tir;
      real Qt_zeta    = f_1*std::sin(theta_1) - f_2*std::sin(theta_2) - f_3*std::sin(theta_3);
      real Qt_eta     = f_1*std::cos(theta_1) + f_2*std::cos(theta_2) + f_3*std::cos(theta_3) + 4*lambda_tir*l_0;
      real Qt_alpha   = ( f_1*(l_a*std::cos(theta_1 + alpha) - d_t*std::sin(theta_1 + alpha)) - 
                          f_2*(l_a*std::cos(theta_2 - alpha) - d_t*std::sin(theta_2 - alpha)) +
                          f_3* d_t*std::sin(theta_3 - alpha) +
                           lambda_tir*l_0*(l_a*cos_alpha - d_t*sin_alpha) -
                           lambda_tir*l_0*(l_a*cos_alpha + d_t*sin_alpha) -
                         2*lambda_tir*l_0*                       d_t*sin_alpha  );
                     
      // Wind Force
      real v_in      = turbine_wind + v_zeta + d_P*omega*cos_alpha;
      real FA        = 0.5*rho*A*Ct*v_in*v_in;
      real FAN       = 0.5*rho*C_dN*A_N    *cos_alpha*std::pow(turbine_wind+v_zeta+d_N*omega*cos_alpha,2._fp);
      real FAT       = 0.5*rho*C_dT*h_T*D_T*cos_alpha*std::pow(turbine_wind+v_zeta+d_T*omega*cos_alpha,2._fp);
      real Qwi_zeta  = -(FA + FAN + FAT);
      real Qwi_eta   = 0;
      real Qwi_alpha = (-FA *(d_Pv*cos_alpha - d_Ph*sin_alpha)
                        -FAN*(d_Nv*cos_alpha - d_Nh*sin_alpha)
                        -FAT  *d_T*cos_alpha);

      // Wave and Drag Forces
      real Qh_zeta   = 0;
      real Qh_eta    = 0;
      real Qwa_zeta  = 0;
      real Qwa_eta   = 0;
      real Qh_alpha  = 0;
      real Qwa_alpha = 0;
      real v_par0;
      for (int i=0; i < n_dg; i++) {
        real h_pg         = (i + 1 - 0.5)*h_sub/n_dg;
        real height       = -(h_sub - h_pg);
        auto [d1,v_x,v_y,a_x,a_y] = pierson_moskowitz_spectrum(wind_19_5m, zeta, height, t, nfreq);
        real v_par        = ((v_zeta + (h_pg - d_Sbott)*omega*cos_alpha - v_x)*-sin_alpha +
                             (v_eta  + (h_pg - d_Sbott)*omega*sin_alpha - v_y)*cos_alpha);
        if (i == 0) v_par0 = v_par;
        real v_per        = ((v_zeta + (h_pg - d_Sbott)*omega*cos_alpha - v_x)*cos_alpha +
                             (v_eta  + (h_pg - d_Sbott)*omega*sin_alpha - v_y)*sin_alpha);
        real a_per        = a_x*cos_alpha + a_y*sin_alpha;
        real tempQh_zeta  = (-0.5*C_dgper*rho_w     *2*r_g*(h_sub/n_dg)* std::abs(v_per)*v_per*cos_alpha -
                              0.5*C_dgpar*rho_w*M_PI*2*r_g*(h_sub/n_dg)* std::abs(v_par)*v_par*sin_alpha);
        real tempQh_eta   = (-0.5*C_dgper*rho_w     *2*r_g*(h_sub/n_dg)* std::abs(v_per)*v_per*sin_alpha -
                              0.5*C_dgpar*rho_w*M_PI*2*r_g*(h_sub/n_dg)* std::abs(v_par)*v_par*cos_alpha);
        real tempQwa_zeta = (rho_w*V_g + m_x)*a_per*cos_alpha/n_dg;
        real tempQwa_eta  = (rho_w*V_g + m_x)*a_per*sin_alpha/n_dg;
        Qh_zeta   +=  tempQh_zeta ;
        Qh_eta    +=  tempQh_eta  ;
        Qwa_zeta  +=  tempQwa_zeta;
        Qwa_eta   +=  tempQwa_eta ;
        Qh_alpha  += (tempQh_zeta *(h_pg - d_Sbott)*cos_alpha + tempQh_eta *(h_pg - d_Sbott)*sin_alpha);
        Qwa_alpha += (tempQwa_zeta*(h_pg - d_Sbott)*cos_alpha + tempQwa_eta*(h_pg - d_Sbott)*sin_alpha);
      }

      Qh_zeta -= 0.5*C_dgb*rho_w*M_PI*r_g*r_g*std::abs(v_par0)*v_par0*sin_alpha;
      Qh_eta  -= 0.5*C_dgb*rho_w*M_PI*r_g*r_g*std::abs(v_par0)*v_par0*cos_alpha;
      real Q_zeta  = Qwe_zeta  + Qb_zeta  + Qt_zeta  + Qh_zeta  + Qwa_zeta  + Qwi_zeta  + Qh_zeta ; // net force in x
      real Q_eta   = Qwe_eta   + Qb_eta   + Qt_eta   + Qh_eta   + Qwa_eta   + Qwi_eta   + Qh_eta  ; // net force in y
      real Q_alpha = Qwe_alpha + Qb_alpha + Qt_alpha + Qh_alpha + Qwa_alpha + Qwi_alpha + Qh_alpha; // net torque in pitch
      SArray<real,1,6> F;
      F(0) = v_zeta;
      F(1) = Q_zeta + M_d*omega*omega*sin_alpha;
      F(2) = v_eta;
      F(3) = Q_eta  - M_d*omega*omega*cos_alpha;
      F(4) = omega;
      F(5) = Q_alpha;
      real avegQ_t = std::sqrt(Qt_zeta*Qt_zeta+Qt_eta*Qt_eta)/8;
      auto deriv = yikl::intrinsics::matmul_rc( yikl::intrinsics::matinv_ge(E) , F );
      return std::make_tuple( deriv , v_in , avegQ_t );
    }


    SArray<real,1,6>
    Betti_tend( SArray<real,1,6> const & x , real t , real turbine_wind , real wind_19_5m , real Ct ) {
      auto [dx1dt,v_in,Q_t] = structure( x , t , turbine_wind , wind_19_5m , Ct);
      return dx1dt;
    }


    real time_step( real dt , real turbine_wind , real wind_19_5m , real Ct ) {
      using yikl::componentwise::operator+;
      using yikl::componentwise::operator*;
      using yikl::componentwise::operator/;
      int niter = std::ceil( dt / dt_max );
      dt = dt / niter;
      real wind = 0;
      for (int iter=0; iter < niter; iter++) {
        real constexpr d_Ph = 5.4305    ; // (m) Horizontal distance between BS and BP
        real constexpr d_Pv = 127.5879  ; // (m) Vertical distance between BS and BP
        auto k1 = Betti_tend( state         , etime      ,  turbine_wind , wind_19_5m , Ct );
        auto k2 = Betti_tend( state+dt/2*k1 , etime+dt/2 ,  turbine_wind , wind_19_5m , Ct );
        auto k3 = Betti_tend( state+dt/2*k2 , etime+dt/2 ,  turbine_wind , wind_19_5m , Ct );
        auto k4 = Betti_tend( state+dt  *k3 , etime+dt   ,  turbine_wind , wind_19_5m , Ct );
        state  = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6;
        etime += dt;
        wind  += state(1) + std::sqrt(d_Ph*d_Ph + d_Pv*d_Pv)*state(5)*std::cos(state(4));
      }
      return wind / niter;
    }

  };

}


