add_executable(gelfand_test gelfand.cpp)
deal_ii_setup_target(gelfand_test)
target_link_libraries(gelfand_test paramsim_dpdes paramsim_cases
                      paramsim_utils utils testutils GTest::gtest_main)
gtest_discover_tests(gelfand_test)

add_executable(minimal_surface_test minimal_surface.cpp)
deal_ii_setup_target(minimal_surface_test)
target_link_libraries(minimal_surface_test paramsim_dpdes paramsim_cases paramsim_utils utils
                     testutils GTest::gtest_main)
gtest_discover_tests(minimal_surface_test)

add_executable(norms_test norms.cpp)
deal_ii_setup_target(norms_test)
target_link_libraries(norms_test paramsim_dpdes paramsim_cases paramsim_utils utils
                      testutils GTest::gtest_main)
gtest_discover_tests(norms_test)

add_executable(poisson_cg_test poisson_cg.cpp)
deal_ii_setup_target(poisson_cg_test)
target_link_libraries(poisson_cg_test paramsim_dpdes paramsim_cases paramsim_utils utils testutils
                      GTest::gtest_main)
gtest_discover_tests(poisson_cg_test)
