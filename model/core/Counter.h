
#pragma once

#include "main_header.h"

namespace core {

  struct Counter {
    double etime;
    double freq;
    Counter(double freq = 1, double etime = 0) {this->freq = freq; this->etime = etime; }
    bool update_and_check( double dt ) { etime += dt; return check(); }
    void update( double dt ) { etime += dt; }
    bool check() const { return etime >= freq - 1.e-10; }
    void reset() { etime -= freq; }
  };

}

