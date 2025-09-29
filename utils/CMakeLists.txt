
add_library(utils cmdparser.cpp)
if(Boost_FOUND)
    target_link_libraries(utils Boost::program_options)
else()
    target_link_libraries(utils "/usr/lib/x86_64-linux-gnu/libboost_program_options.so")
endif()

add_library(paramsim_utils run_handler.cpp)
deal_ii_setup_target(paramsim_utils)
target_link_libraries(paramsim_utils utils)
