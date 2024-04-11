cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(${MINIWEATHER_ML_HOME}/model model)

add_executable(driver driver.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(driver)
target_link_libraries(driver model "${MW_LINK_FLAGS}")
target_include_directories(driver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

add_executable(generate_training_data generate_training_data.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(generate_training_data)
target_link_libraries(generate_training_data model "${MW_LINK_FLAGS}")
target_include_directories(generate_training_data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)

add_executable(deploy_surrogate deploy_surrogate.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(deploy_surrogate)
target_link_libraries(deploy_surrogate model "${MW_LINK_FLAGS}")
target_include_directories(deploy_surrogate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/custom_modules)
