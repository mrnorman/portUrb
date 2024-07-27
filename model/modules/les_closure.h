
#pragma once

#include "main_header.h"
#include "coupler.h"

namespace modules {

  struct LES_Closure {
    int static constexpr hs        = 1;
    int static constexpr num_state = 5;
    int static constexpr idR = 0;
    int static constexpr idU = 1;
    int static constexpr idV = 2;
    int static constexpr idW = 3;
    int static constexpr idT = 4;


    void init( core::Coupler &coupler ) const;



    void apply( core::Coupler &coupler , real dtphys ) const;



    // Convert coupler's data to state and tracers arrays
    void convert_coupler_to_dynamics( core::Coupler const &coupler ,
                                      real4d              &state   ,
                                      real4d              &tracers ,
                                      real3d              &tke     ) const;



    // Convert dynamics state and tracers arrays to the coupler state and write to the coupler's data
    void convert_dynamics_to_coupler( core::Coupler &coupler ,
                                      realConst4d    state   ,
                                      realConst4d    tracers ,
                                      realConst3d    tke     ) const;



    void halo_bcs_z( core::Coupler const & coupler ,
                     real4d        const & state   ,
                     real4d        const & tracers ,
                     real3d        const & tke     ) const;

    void halo_bcs_zero_vel( core::Coupler const & coupler ,
                            real4d        const & state   ,
                            real4d        const & tracers ,
                            real3d        const & tke     ) const;
  };

}

