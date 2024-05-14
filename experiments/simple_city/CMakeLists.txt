cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)

add_library(custom_modules custom_modules/precursor_sponge.cpp
                           custom_modules/sc_init.cpp
                           )
yakl_process_target(custom_modules)
target_include_directories(custom_modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)
target_link_libraries(custom_modules model)

add_executable(wind_farm wind_farm.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(wind_farm)
target_link_libraries(wind_farm model custom_modules "${MW_LINK_FLAGS}")

add_executable(city city.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(city)
target_link_libraries(city model custom_modules "${MW_LINK_FLAGS}")

add_executable(abl abl.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(abl)
target_link_libraries(abl model custom_modules "${MW_LINK_FLAGS}")


