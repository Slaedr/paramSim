add_library(paramsim_cases case.cpp)

add_subdirectory(poisson)
add_subdirectory(cube)
add_subdirectory(minimal_surface)
add_subdirectory(gelfand)

deal_ii_setup_target(paramsim_cases)
