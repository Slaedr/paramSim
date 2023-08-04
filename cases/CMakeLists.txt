add_library(paramsim_cases case.cpp)

add_subdirectory(convdiff)
add_subdirectory(poisson)

deal_ii_setup_target(paramsim_cases)
