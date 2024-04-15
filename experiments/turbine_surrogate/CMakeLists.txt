cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)

add_executable(turbine_hires turbine_hires.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(turbine_hires)
target_link_libraries(turbine_hires model "${MW_LINK_FLAGS}")
target_include_directories(turbine_hires PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

add_executable(turbine_surrogate turbine_surrogate.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(turbine_surrogate)
target_link_libraries(turbine_surrogate model "${MW_LINK_FLAGS}")
target_include_directories(turbine_surrogate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)


