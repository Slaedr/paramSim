cmake_minimum_required(VERSION 3.13.4)

project(paramsim LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(deal.II 9.5.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
deal_ii_initialize_cached_variables()

find_package(Boost 1.60.0 COMPONENTS program_options)

add_subdirectory(pdes)
add_subdirectory(utils)
add_subdirectory(cases)

enable_testing()

add_subdirectory(test)

add_executable(run_case run_case.cpp)
deal_ii_setup_target(run_case)
target_link_libraries(run_case paramsim_solvers paramsim_cases paramsim_utils utils)
