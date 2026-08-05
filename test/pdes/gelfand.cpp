#include <gtest/gtest.h>

#include <deal.II/grid/tria.h>

#include "../../utils/run_handler.hpp"
#include "../grid_convergence.hpp"

namespace {

template <int dim>
double run_gelfand_convergence(const char *dimension,
                               const char *initial_resolution,
                               const char *fe_degree)
{
    constexpr int argument_count = 17;
    const char *const arguments[] = {"prog",
                                     "--dimension",
                                     dimension,
                                     "--pde",
                                     "gelfand",
                                     "--refine_levels",
                                     "4",
                                     "--initial_resolution",
                                     initial_resolution,
                                     "--case",
                                     "gelfand_verify",
                                     "--fe_degree",
                                     fe_degree,
                                     "--max_its",
                                     "12",
                                     "--tolerance",
                                     "1e-8"};

    const auto run_data =
        paramsim::get_run_data<dim>(argument_count, arguments);
    return paramsim::testutils::test_grid_convergence(
        run_data.test_case, run_data.pde_params, run_data.solver_params);
}

template <int dim>
void expect_gelfand_boundary_markers(const char *dimension)
{
    constexpr int argument_count = 9;
    const char *const arguments[] = {
        "prog",    "--dimension", dimension,        "--pde",
        "gelfand", "--case",      "gelfand_verify", "--initial_resolution",
        "2"};

    const auto run_data =
        paramsim::get_run_data<dim>(argument_count, arguments);
    ASSERT_EQ(run_data.test_case->get_dirichlet_bcs().size(), 1u);

    const auto boundary_id =
        run_data.test_case->get_dirichlet_bcs().front().bc_id;
    dealii::Triangulation<dim> triangulation;
    run_data.test_case->get_geometry()->generate_grid(triangulation, 2);
    run_data.test_case->get_geometry()->set_boundary_ids(triangulation);

    for (const auto& cell : triangulation.cell_iterators()) {
        for (const auto& face : cell->face_iterators()) {
            if (face->at_boundary()) {
                EXPECT_EQ(face->boundary_id(), boundary_id);
            }
        }
    }
}

} // namespace

TEST(GelfandVerification2d, ConvergesP1)
{
    EXPECT_NEAR(run_gelfand_convergence<2>("2", "4", "1"), 2.0, 2e-1);
}

TEST(GelfandVerification2d, ConvergesP2)
{
    EXPECT_NEAR(run_gelfand_convergence<2>("2", "4", "2"), 3.0, 2e-1);
}

TEST(GelfandVerification3d, ConvergesP1)
{
    EXPECT_NEAR(run_gelfand_convergence<3>("3", "2", "1"), 2.0, 3e-1);
}

TEST(GelfandVerification3d, ConvergesP2)
{
    EXPECT_NEAR(run_gelfand_convergence<3>("3", "2", "2"), 3.0, 3e-1);
}

TEST(GelfandVerification, AppliesDirichletConditionToEntireBoundary)
{
    expect_gelfand_boundary_markers<2>("2");
    expect_gelfand_boundary_markers<3>("3");
}
