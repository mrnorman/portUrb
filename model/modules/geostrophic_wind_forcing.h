
#pragma once

#include "coupler.h"

namespace modules {

  void geostrophic_wind_forcing( core::Coupler &coupler , real dt , real lat_g , real u_g , real v_g );
}

