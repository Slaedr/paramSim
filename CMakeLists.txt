cmake_minimum_required(VERSION 3.13.4)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(deal.II 9.5.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
deal_ii_initialize_cached_variables()

find_package(Boost 1.60.0 COMPONENTS program_options)

project(paramsim)

add_library(utils cmdparser.cpp)
deal_ii_setup_target(utils)
if(Boost_FOUND)
    target_link_libraries(utils Boost::program_options)
else()
    target_link_libraries(utils "/usr/lib/x86_64-linux-gnu/libboost_program_options.so")
endif()

add_subdirectory(pdes)
add_subdirectory(cases)

add_executable(run_case run_case.cpp)
deal_ii_setup_target(run_case)
target_link_libraries(run_case paramsim_solvers paramsim_cases utils)

