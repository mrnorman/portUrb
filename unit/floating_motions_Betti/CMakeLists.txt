cmake_minimum_required(VERSION 3.0)
project(portUrb)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/../../external/YAKL)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl)

set(YAKL_CXX_FLAGS "-Ofast -ffast-math")
set(YAKL_DEBUG OFF CACHE BOOL "" FORCE)

add_subdirectory(${YAKL_HOME} ${YAKL_BIN})
include_directories(${YAKL_BIN})

include(${YAKL_HOME}/yakl_utils.cmake)

execute_process (
    COMMAND bash -c "nc-config --libs"
    OUTPUT_VARIABLE NC_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../external/yaml-cpp ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp)

add_executable(Betti Betti.cpp)
yakl_process_target(Betti)
target_link_libraries(Betti ${NC_LIBS} yaml-cpp)

add_executable(Betti_simplified Betti_simplified.cpp)
yakl_process_target(Betti_simplified)
target_link_libraries(Betti_simplified ${NC_LIBS} yaml-cpp)

