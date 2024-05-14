
add_library(core INTERFACE)
target_link_libraries(core INTERFACE yaml-cpp)
target_include_directories(core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(core INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

