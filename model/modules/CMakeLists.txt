
add_subdirectory(helpers/microphysics_p3 microphysics_p3)

add_library(modules column_nudging.cpp
                    geostrophic_wind_forcing.cpp
                    les_closure.cpp
                    dynamics_rk_main.cpp
                    dynamics_rk_init.cpp
                    dynamics_rk_compute_tendencies.cpp
                    dynamics_rk_edge_exchange.cpp
                    hydrostasis.cpp
                    microphysics_kessler.cpp
                    microphysics_p3.cpp
                    perturb_temperature.cpp
                    sponge_layer.cpp
                    surface_flux.cpp
                    )
yakl_process_target(modules)
target_link_libraries(modules microphysics_p3 core)
target_include_directories(modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(modules PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/helpers)

