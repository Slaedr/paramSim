cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.5.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
deal_ii_initialize_cached_variables()

project(convdiff_hdg)

add_library(convdiff_hdg convdiff_hdg.cpp)
deal_ii_setup_target(convdiff_hdg)

add_executable(verification verification.cpp)
deal_ii_setup_target(verification)
target_link_libraries(verification convdiff_hdg)

