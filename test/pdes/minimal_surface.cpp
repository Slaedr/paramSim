#include <set>
#include <limits>
#include <gtest/gtest.h>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>

#include "../../cases/minimal_surface/minimal_surface.hpp"
#include "../../cases/minimal_surface/gaussians.hpp"

#include "../../utils/cmdparser.hpp"

using namespace paramsim;
namespace bpo = boost::program_options;

constexpr int dim = 2;

class MinimalSurfaceDiskSinusoidalVerification : public testing::Test
{
protected:
    void set_up(const int nargs, const char args[][100])
    {
        argv = static_cast<const char**>(std::malloc(nargs*sizeof(char**)));
        for(int i = 0; i < nargs; i++) {
            argv[i] = args[i];
        }
    }

    ~MinimalSurfaceDiskSinusoidalVerification() {
        std::free(argv);
    }

    static constexpr double eps = 2e-1;
    RunData<dim> run_data;
    const char **argv;
};


TEST_F(MinimalSurfaceDiskSinusoidalVerification, ConvergesP1)
{
    const int nargs = 13;
    const char args[][100] = {"prog", "--solver", "minimal_surface", "--refine_levels", "4",
        "--initial_resolution", "4",
        "--case", "minimal_surface_disk_sinusoidal", "--fe_degree", "1", "--max_its", "100"};
    this->set_up(nargs, args);
    this->run_data = get_run_data<dim>(nargs, this->argv);
    const double conv_slope = testutils::test_grid_convergence(
        this->run_data.test_case, this->run_data.pde_params, this->run_data.solver_params);
    EXPECT_NEAR(conv_slope, 2.0, this->eps);
}

TEST_F(MinimalSurfaceDiskSinusoidalVerification, ConvergesP2)
{
    const int nargs = 13;
    const char args[][100] = {"prog", "--solver", "poisson_cg", "--refine_levels", "4",
        "--initial_resolution", "4",
        "--case", "poisson_verify", "--fe_degree", "2", "--max_its", "100"};
    this->set_up(nargs, args);
    this->run_data = get_run_data<dim>(nargs, this->argv);
    const double conv_slope = testutils::test_grid_convergence(
        this->run_data.test_case, this->run_data.pde_params, this->run_data.solver_params);
    EXPECT_NEAR(conv_slope, 3.0, this->eps);
}
