cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)

add_executable(taylor_green taylor_green.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(taylor_green)
target_link_libraries(taylor_green model "${MW_LINK_FLAGS}")
target_include_directories(taylor_green PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

