

struct floating_motions_betti {
  realHost2d C_p_arr;    // 2-D Array of coefficient of thrust based on (TSR,pitch)
  realHost2d C_t_arr;    // 2-D Array of coefficient of power  based on (TSR,pitch)
  realHost1d pitch_list; // Reference pitch angles corresponding to extent(1) of C_p_arr and C_t_arr
  realHost1d TSR_list;   // Reference TSR values   corresponding to extent(0) of C_p_arr and C_t_arr
  real       rand_seed;  // For random phases in pm spectrum


  // Assume arr is ordered lowest to highest. Return index of arr(index) nearest to "val"
  int nearest_index(realHost1d const & arr, real val) {
    int  ind = 0;
    real diff = std::abs(arr(ind)-val);
    for (int i=1; i < arr.size(); i++) {
      real loc = std::abs(arr(ind)-val);
      if ( loc > diff ) { return i-1; }
      else              { diff = loc; }
    }
    return n-1;
  }



  void init( std::string fname ) {
    yakl::SimpleNetCDF nc;
    nc.open( fname , yakl::NETCDF_MODE_READ );
    nc.read( C_p_arr    , "C_p"          );
    nc.read( C_t_arr    , "C_t"          );
    nc.read( pitch_list , "pitch_angles" );
    nc.read( TSR_list   , "TSR_values"   );
    nc.close();
    rand_seed = 0;
  }



  // Return first-order-accurate interpolation of C_p and C_t based on provided TSR and beta
  // TSR : tip speed ratio (dimensionless)
  // beta: blade pitch angle (radians)
  // returns   : Tuple of [C_p,C_t], which are coefficients of power and thrust, respectively
  std::tuple<real,real>
  CpCtCq( real TSR , real beta ) {
    int pitch_index = nearest_index( pitch_list , beta / M_PI * 180 );
    int TSR_index   = nearest_index( TSR_list   , TSR  );
    return std::make_tuple( C_p_arr(TSR_index,pitch_index) , C_t_arr(TSR_index,pitch_index) );
  }



  // Computes Pierson Moskowitz spectrum outputs
  // U19_5: average wind velocity at 19.5m (m/s)
  // zeta : "the x component to evaluate"
  // eta  : "the y component to evaluate .Note: the coordinate system here is different
  //           from the Betti model. The downward is negative in this case"
  // t    : the time to evaluate (s)
  // N    : The number of frequency intervals to use
  // returns  : Tuple of [wave_eta,v_x,v_y,a_x,a_y]
  // wave_eta : Wave elevation (m)
  // v_x      : x-direction wave velocity
  // v_y      : y-direction wave velocity
  // a_x      : x-direction wave acceleration
  // a_y      : y-direction wave acceleration
  std::tuple<real,real,real,real,real>
  pierson_moskowitz_spectrum( real U19_5 , real zeta , real eta , real t , int N) {
    real constexpr g       = 9.81;           // gravity acceleration
    real constexpr alpha   = 0.0081;         // Phillip's constant
    real constexpr f_pm    = 0.14*(g/U19_5); // peak frequency
    real constexpr cutof_f = 3*f_pm;         // cutoff frequency
    real constexpr start_f = 0.1;            // starting frequency
    real sum_wave_eta = 0;
    real sum_v_x      = 0;
    real sum_v_y      = 0;
    real sum_a_x      = 0;
    real sum_a_y      = 0;
    for (int i=0; i < N; i++) {
      yakl::Random prng(rand_seed);
      real random_phase  = prng.genFP<real>(0.,2.*M_PI);
      real delta_f       = (cutof_f - start_f) / (N-1);
      real f             = start_f + i*delta_f;
      real omega         = f*2*M_PI;
      real pi_term       = std::pow(2.*M_PI,4.);
      real f_term1       = std::pow(f,5.);
      real f_term2       = std::pow(f_pm/f,4.);
      //   S_pm          = (alpha*g**2/((2*np.pi)**4*f**5   ))*  np.exp(-(5 /4 )*(f_pm/f)**4)
      real S_pm          = (alpha*g*g /(pi_term     *f_term1))*std::exp(-(5./4.)*fterm2     );
      real a             = std::sqrt(2*S_pm*delta_f);
      real k             = omega*omega/g;
      real sin_component = std::sin(omega*t - k*zeta + random_phase);
      real cos_component = std::cos(omega*t - k*zeta + random_phase);
      real exp_component = std::cos(k*eta);
      sum_wave_eta += a*sin_component;
      sum_v_x      += omega*a*exp_component*sin_component;
      sum_v_y      += omega*a*exp_component*cos_component;
      sum_a_x      += omega*omega*a*exp_component*cos_component;
      sum_a_y      -= omega*omega*a*exp_component*sin_component;
    }
    return std::make_tuple( sum_wave_eta/N , sum_v_x/N , sum_v_y/N , sum_a_x/N , sum_a_y/N );
  }

};


int main() {
  auto [C_p,C_t] = CpCtCq( TSR , beta , C_p_arr , C_t_arr , pitch_list , TSR_list );
}


