#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <gtest/gtest.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>

#include "../../cases/poisson/verify.hpp"
#include "../../pdes/poisson/poisson_cg.hpp"

#include "../../utils/cmdparser.hpp"
#include "../../utils/run_handler.hpp"
#include "../grid_convergence.hpp"

using namespace paramsim;
// namespace bpo = boost::program_options;

constexpr int dim = 2;

class PoissonVerification : public testing::Test {
protected:
    void set_up(const int nargs, const char args[][100])
    {
        argv = allocate_setup_args(nargs, args);
    }

    ~PoissonVerification()
    {
        std::free(argv);
    }

    static constexpr double eps = 2e-1;
    RunData<dim> run_data;
    const char **argv = NULL;
};

TEST_F(PoissonVerification, ConvergesP1)
{
    const int nargs = 15;
    const char args[][100] = {"prog",
                              "--dimension",
                              "2",
                              "--pde",
                              "poisson_cg",
                              "--refine_levels",
                              "4",
                              "--initial_resolution",
                              "4",
                              "--case",
                              "poisson_verify",
                              "--fe_degree",
                              "1",
                              "--max_its",
                              "3"};
    this->set_up(nargs, args);
    this->run_data = get_run_data<dim>(nargs, this->argv);
    const double conv_slope = testutils::test_grid_convergence(
        this->run_data.test_case, this->run_data.pde_params,
        this->run_data.solver_params);
    EXPECT_NEAR(conv_slope, 2.0, this->eps);
}

TEST_F(PoissonVerification, ConvergesP2)
{
    const int nargs = 15;
    const char args[][100] = {"prog",
                              "--dimension",
                              "2",
                              "--pde",
                              "poisson_cg",
                              "--refine_levels",
                              "4",
                              "--initial_resolution",
                              "4",
                              "--case",
                              "poisson_verify",
                              "--fe_degree",
                              "2",
                              "--max_its",
                              "5"};
    this->set_up(nargs, args);
    this->run_data = get_run_data<dim>(nargs, this->argv);
    const double conv_slope = testutils::test_grid_convergence(
        this->run_data.test_case, this->run_data.pde_params,
        this->run_data.solver_params);
    EXPECT_NEAR(conv_slope, 3.0, this->eps);
}

TEST(PoissonVerification3d, ConvergesP1)
{
    constexpr int test_dim = 3;
    const int nargs = 15;
    // clang-format off
    const char args[][100] = {"prog",
                              "--dimension", "3",
                              "--pde", "poisson_cg",
                              "--refine_levels", "5",
                              "--initial_resolution", "2",
                              "--case", "poisson_verify",
                              "--fe_degree", "1",
                              "--max_its", "3"};
    // clang-format on
    const char **argv = allocate_setup_args(nargs, args);

    const auto run_data = get_run_data<test_dim>(nargs, argv);
    const double conv_slope = testutils::test_grid_convergence(
        run_data.test_case, run_data.pde_params, run_data.solver_params);

    EXPECT_NEAR(conv_slope, 2.0, 3e-1);
    std::free(argv);
}

TEST(PoissonVerification3d, ConvergesP2)
{
    constexpr int test_dim = 3;
    const int nargs = 15;
    // clang-format off
    const char args[][100] = {"prog",
                              "--dimension", "3",
                              "--pde", "poisson_cg",
                              "--refine_levels", "4",
                              "--initial_resolution", "2",
                              "--case", "poisson_verify",
                              "--fe_degree", "2",
                              "--max_its", "5"};
    // clang-format on
    const char **argv = allocate_setup_args(nargs, args);

    const auto run_data = get_run_data<test_dim>(nargs, argv);
    const double conv_slope = testutils::test_grid_convergence(
        run_data.test_case, run_data.pde_params, run_data.solver_params);

    EXPECT_NEAR(conv_slope, 3.0, 3e-1);
    std::free(argv);
}

TEST(RunData, RejectsDimensionMismatch)
{
    const int nargs = 9;
    const char args[][100] = {
        "prog",           "--dimension",          "3",
        "--pde",          "poisson_cg",           "--case",
        "poisson_verify", "--initial_resolution", "8"};
    const char **argv = allocate_setup_args(nargs, args);

    EXPECT_THROW((get_run_data<2>(nargs, argv)), std::invalid_argument);

    std::free(argv);
}
