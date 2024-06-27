
#include "Betti_simplified.h"

int main(int argc, char **argv) {
  yakl::init();
  {
    Floating_motions_betti floating_motions;
    real dt = 0.005;
    for (real uvel = 3; uvel <= 25; uvel++) {
      floating_motions.init(std::string(argv[1]));
      std::cout << "\n" << uvel << std::endl;
      for (int i=0; i < 20000; i++) {
        real wind = floating_motions.time_step( dt , uvel , uvel );
        if (i%200 == 0) std::cout << wind << std::endl;
      }
    }
  }
  yakl::finalize();
}
