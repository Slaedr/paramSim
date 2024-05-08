add_library(paramsim_cases case.cpp)

add_subdirectory(convdiff)
add_subdirectory(poisson)
add_subdirectory(minimal_surface)

deal_ii_setup_target(paramsim_cases)
