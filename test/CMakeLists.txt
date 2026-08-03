add_library(testutils grid_convergence.cpp)
target_link_libraries(testutils paramsim_solvers paramsim_dpdes paramsim_cases)
deal_ii_setup_target(testutils)

find_package(GTest 1.10.0)
if(NOT GTest_FOUND)	
	message(STATUS "Fetching external GTest")
	include(FetchContent)
	FetchContent_Declare(
	    googletest
	    GIT_REPOSITORY https://github.com/google/googletest.git
	    GIT_TAG        v1.14.0
	)
	# need to set the variables in CACHE due to CMP0077
	set(gtest_disable_pthreads ON CACHE INTERNAL "")
	if(MSVC)
	    set(gtest_force_shared_crt ON CACHE INTERNAL "")
	endif()
	#FetchContent_GetProperties(googletest)
	#if(NOT googletest_POPULATED)
	#    FetchContent_Populate(googletest)
	#    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
	#endif()
	FetchContent_MakeAvailable(googletest)
	# make sure the tests DLLs are placed in the working path for CTest
	#set_target_properties(gtest gtest_main PROPERTIES
	#    RUNTIME_OUTPUT_DIRECTORY "${ParamSim_LIBRARY_PATH}"
	#   ARCHIVE_OUTPUT_DIRECTORY "${ParamSim_LIBRARY_PATH}"
	#    LIBRARY_OUTPUT_DIRECTORY "${ParamSim_LIBRARY_PATH}")
	# If CXX compiler is dpcpp, use -ffp-model=precise
	# Otherwise, it will throw src/gtest.cc:1583:8: error: comparison with NaN always evaluates to false in fast floating point modes
	if(CMAKE_CXX_COMPILER MATCHES "dpcpp|icpx")
	    target_compile_options(gtest PRIVATE "-ffp-model=precise")
	    target_compile_options(gtest_main PRIVATE "-ffp-model=precise")
	endif()
	# by default, the outdated targets are not being exported
	add_library(GTest::Main ALIAS gtest_main)
	add_library(GTest::GTest ALIAS gtest)
endif()

include(GoogleTest)

add_executable(geometry_test geometry.cpp)
deal_ii_setup_target(geometry_test)
target_link_libraries(geometry_test GTest::gtest_main)

gtest_discover_tests(geometry_test)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy"
        RESULT_VARIABLE PARAMSIM_NUMPY_STATUS
        OUTPUT_QUIET ERROR_QUIET
    )
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import libensemble"
        RESULT_VARIABLE PARAMSIM_LIBENSEMBLE_STATUS
        OUTPUT_QUIET ERROR_QUIET
    )

    if(PARAMSIM_NUMPY_STATUS EQUAL 0)
        add_test(
            NAME ensemble_setup_case
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_setup_case.py
        )
        add_test(
            NAME ensemble_results_checker
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_check_ensemble_results.py
        )
    else()
        message(STATUS "Skipping paramSim Python tests that need NumPy: not importable by ${Python3_EXECUTABLE}")
    endif()

    if(PARAMSIM_LIBENSEMBLE_STATUS EQUAL 0)
        add_test(
            NAME ensemble_drivers
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_ensemble_drivers.py
        )
        # End-to-end: runs a real libEnsemble ensemble (with a stub
        # executable, plus the real run_case binary when available) and
        # checks that a failing simulation is recorded as a failure in the
        # saved history .npy file.
        add_test(
            NAME ensemble_failure_history
            COMMAND ${CMAKE_COMMAND} -E env PARAMSIM_RUN_CASE=$<TARGET_FILE:run_case>
                    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_ensemble_failure_history.py
        )
        set_tests_properties(ensemble_failure_history PROPERTIES
            TIMEOUT 900
            RUN_SERIAL TRUE
            SKIP_RETURN_CODE 77
        )
    else()
        message(STATUS "Skipping paramSim Python tests that need libEnsemble: not importable by ${Python3_EXECUTABLE}")
    endif()
endif()

add_subdirectory(pdes)
add_subdirectory(cases)
