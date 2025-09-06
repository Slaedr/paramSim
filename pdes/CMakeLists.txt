add_library(paramsim_dpdes pdebase.cpp)

add_subdirectory(poisson)
add_subdirectory(nonlinear_elliptic)

target_link_libraries(paramsim_dpdes paramsim_utils)
deal_ii_setup_target(paramsim_dpdes)
