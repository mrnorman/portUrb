
#pragma once

#define YIKL_AUTO_LABEL() (std::string(basename(__FILE__)) + std::string(":") + std::to_string(__LINE__)).c_str()
#define YIKL_SCOPE(a,b) auto &a = std::ref(b).get()

#include "YIKL_parallel_for.h"

