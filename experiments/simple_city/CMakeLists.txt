cmake_minimum_required(VERSION 3.0)
project(portUrb)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

include_directories(${PORTURB_HOME}/model
                    ${PORTURB_HOME}/model/core
                    ${PORTURB_HOME}/model/modules
                    ${PORTURB_HOME}/model/modules/helpers
                    ${PORTURB_HOME}/model/modules/helpers/microphysics_p3
                    ${PORTURB_HOME}/external/yaml-cpp/include
                    ${PORTURB_HOME}/external/ponni/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules
                    )

add_subdirectory(${PORTURB_HOME}/model model)

add_library(custom_modules INTERFACE)
target_include_directories(custom_modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

set(mylibs modules custom_modules model core ponni yaml-cpp "${PORTURB_LINK_FLAGS}")

include(${YAKL_HOME}/yakl_utils.cmake)

add_executable(wind_farm2 wind_farm2.cpp)
yakl_process_target(wind_farm2)
target_link_libraries(wind_farm2 ${mylibs})

add_executable(wind_farm wind_farm.cpp)
yakl_process_target(wind_farm)
target_link_libraries(wind_farm ${mylibs})

add_executable(city city.cpp)
yakl_process_target(city)
target_link_libraries(city ${mylibs})

add_executable(abl abl.cpp)
yakl_process_target(abl)
target_link_libraries(abl ${mylibs})

add_executable(abl_neutral abl_neutral.cpp)
yakl_process_target(abl_neutral)
target_link_libraries(abl_neutral ${mylibs})

add_executable(awaken_simplest awaken_simplest.cpp)
yakl_process_target(awaken_simplest)
target_link_libraries(awaken_simplest ${mylibs})

add_executable(abl_convective abl_convective.cpp)
yakl_process_target(abl_convective)
target_link_libraries(abl_convective ${mylibs})

add_executable(supercell supercell.cpp)
yakl_process_target(supercell)
target_link_libraries(supercell ${mylibs})

add_executable(windfarm_ensemble windfarm_ensemble.cpp)
yakl_process_target(windfarm_ensemble)
target_link_libraries(windfarm_ensemble ${mylibs})

add_executable(turbine_neutral_ensemble turbine_neutral_ensemble.cpp)
yakl_process_target(turbine_neutral_ensemble)
target_link_libraries(turbine_neutral_ensemble ${mylibs})

add_executable(turbine_stable_ensemble turbine_stable_ensemble.cpp)
yakl_process_target(turbine_stable_ensemble)
target_link_libraries(turbine_stable_ensemble ${mylibs})

add_executable(windfarm_ensemble_3x3 windfarm_ensemble_3x3.cpp)
yakl_process_target(windfarm_ensemble_3x3)
target_link_libraries(windfarm_ensemble_3x3 ${mylibs})

