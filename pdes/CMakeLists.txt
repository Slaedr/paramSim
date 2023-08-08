add_library(paramsim_solvers pdebase.cpp)

add_subdirectory(poisson)
add_subdirectory(convdiff)

deal_ii_setup_target(paramsim_solvers)
