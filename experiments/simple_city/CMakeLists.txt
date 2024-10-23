cmake_minimum_required(VERSION 3.0)
project(portUrb)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories($ENV{PORTURB_HOME}/model
                    $ENV{PORTURB_HOME}/model/core
                    $ENV{PORTURB_HOME}/model/modules
                    $ENV{PORTURB_HOME}/model/modules/helpers
                    $ENV{PORTURB_HOME}/model/modules/helpers/microphysics_p3
                    $ENV{PORTURB_HOME}/external/yaml-cpp/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules
                    )

add_subdirectory($ENV{PORTURB_HOME}/model model)

add_library(custom_modules INTERFACE)
target_include_directories(custom_modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

set(mylibs modules custom_modules model core yaml-cpp yakl kokkos "$ENV{PORTURB_LINK_FLAGS}")

include(${YAKL_HOME}/yakl_utils.cmake)

add_executable(wind_farm2 wind_farm2.cpp)
target_compile_options(wind_farm2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(wind_farm2 ${mylibs})

add_executable(wind_farm wind_farm.cpp)
target_compile_options(wind_farm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(wind_farm ${mylibs})

add_executable(city city.cpp)
target_compile_options(city PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(city ${mylibs})

add_executable(abl abl.cpp)
target_compile_options(abl PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(abl ${mylibs})

add_executable(abl_neutral abl_neutral.cpp)
target_compile_options(abl_neutral PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(abl_neutral ${mylibs})

add_executable(awaken_simplest awaken_simplest.cpp)
target_compile_options(awaken_simplest PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(awaken_simplest ${mylibs})

add_executable(abl_convective abl_convective.cpp)
target_compile_options(abl_convective PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(abl_convective ${mylibs})

add_executable(supercell supercell.cpp)
target_compile_options(supercell PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(supercell ${mylibs})

add_executable(windfarm_ensemble windfarm_ensemble.cpp)
target_compile_options(windfarm_ensemble PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(windfarm_ensemble ${mylibs})

add_executable(turbine_neutral_ensemble turbine_neutral_ensemble.cpp)
target_compile_options(turbine_neutral_ensemble PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(turbine_neutral_ensemble ${mylibs})

add_executable(turbine_stable_ensemble turbine_stable_ensemble.cpp)
target_compile_options(turbine_stable_ensemble PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(turbine_stable_ensemble ${mylibs})

add_executable(turbine_convective_ensemble turbine_convective_ensemble.cpp)
target_compile_options(turbine_convective_ensemble PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(turbine_convective_ensemble ${mylibs})

add_executable(windfarm_ensemble_3x3 windfarm_ensemble_3x3.cpp)
target_compile_options(windfarm_ensemble_3x3 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:$ENV{PORTURB_CXX_FLAGS}>)
target_link_libraries(windfarm_ensemble_3x3 ${mylibs})

