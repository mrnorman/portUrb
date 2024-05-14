
#pragma once

#include "coupler.h"
#include "MultipleFields.h"

namespace modules {


  class ColumnNudger {
  public:
    std::vector<std::string> names;
    real2d column;

    void set_column( core::Coupler &coupler , std::vector<std::string> names_in = {"uvel"} );

    void nudge_to_column( core::Coupler &coupler , real dt , real time_scale = 900 );

    template <class T>
    real2d get_column_average( core::Coupler const &coupler , core::MultiField<T,3> &state ) const;
  };

}


