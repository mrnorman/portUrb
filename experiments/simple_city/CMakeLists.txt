cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)

add_executable(wind_farm wind_farm.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(wind_farm)
target_link_libraries(wind_farm model "${MW_LINK_FLAGS}")
target_include_directories(wind_farm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

add_executable(city city.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(city)
target_link_libraries(city model "${MW_LINK_FLAGS}")
target_include_directories(city PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

add_executable(abl abl.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(abl)
target_link_libraries(abl model "${MW_LINK_FLAGS}")
target_include_directories(abl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

# add_executable(compare_closures compare_closures.cpp)
# include(${YAKL_HOME}/yakl_utils.cmake)
# yakl_process_target(compare_closures)
# target_link_libraries(compare_closures model "${MW_LINK_FLAGS}")
# target_include_directories(compare_closures PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)


