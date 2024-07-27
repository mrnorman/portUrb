
#pragma once

#include "coupler.h"

namespace custom_modules {
  
  void precursor_sponge( core::Coupler            & coupler_main      ,
                         core::Coupler      const & coupler_precursor ,
                         real                       dt                ,
                         real                       time_scale        ,
                         std::vector<std::string>   vnames            ,
                         int                        cells_x1 = 0      ,
                         int                        cells_x2 = 0      ,
                         int                        cells_y1 = 0      ,
                         int                        cells_y2 = 0      );
}


