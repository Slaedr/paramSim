add_library(paramsim_solvers pdebase.cpp)

add_subdirectory(poisson)
add_subdirectory(convdiff)
add_subdirectory(nonlinear_elliptic)

deal_ii_setup_target(paramsim_solvers)
