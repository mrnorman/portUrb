
add_subdirectory(helpers/microphysics_p3 microphysics_p3)

add_library(modules INTERFACE)
target_include_directories(modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(modules INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/helpers)

