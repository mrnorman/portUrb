
#pragma once

#define YIKL_AUTO_LABEL() (std::string(basename(__FILE__)) + std::string(":") + std::to_string(__LINE__)).c_str()
#define YIKL_SCOPE(a,b) auto &a = std::ref(b).get()

#include "YIKL_parallel_for.h"
#include "YIKL_SArray.h"
#include "YIKL_componentwise.h"
#include "YIKL_intrinsics_matmul.h"
#include "YIKL_intrinsics_matinv.h"
#include "YIKL_intrinsics_minval.h"
#include "YIKL_intrinsics_maxval.h"
#include "YIKL_intrinsics_count.h"
#include "YIKL_intrinsics_sum.h"

