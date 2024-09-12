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

add_executable(driver driver.cpp)
yakl_process_target(driver)
target_link_libraries(driver ${mylibs})
