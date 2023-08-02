
add_library(cases_convdiff step51.cpp)
deal_ii_setup_target(cases_convdiff)
target_link_libraries(cases_convdiff convdiff_hdg)
