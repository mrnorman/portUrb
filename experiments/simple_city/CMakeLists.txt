cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)


add_executable(cube cube.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(cube)
target_link_libraries(cube model "${MW_LINK_FLAGS}")
target_include_directories(cube PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)


add_executable(wind_farm wind_farm.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(wind_farm)
target_link_libraries(wind_farm model "${MW_LINK_FLAGS}")
target_include_directories(wind_farm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)


