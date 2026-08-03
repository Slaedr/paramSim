#include <gtest/gtest.h>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>

#include "../../cases/cube/fourier.hpp"
#include "../../pdes/nonlinear_elliptic/minimal_surface.hpp"
#include "../../pdes/poisson/poisson_cg.hpp"
#include "../utils/temporary_parameter_file.hpp"
#include "solve_grid_sequence.hpp"

namespace {

using namespace paramsim::test;

constexpr char cross_term_parameters_2d[] =
    "1\n"
    "1 3 3\n"
    "0.04 -0.03 0.02 -0.01\n";

constexpr char cross_term_parameters_3d[] =
    "1\n"
    "1 3 3 3\n"
    "0.08 -0.07 0.06 -0.05 0.04 -0.03 0.02 -0.01\n";

TEST(CubeFourierPoisson, SolverConvergesWithP1In2D)
{
    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::PoissonCG>(
                  32, 3, "poisson_cg", poisson_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeFourierPoisson, SolverConvergesWithP1In3D)
{
    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::PoissonCG>(
                  8, 3, "poisson_cg", poisson_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeFourier, GeometryIsUnitCube)
{
    auto test_case = std::make_shared<paramsim::cases::cube::CubeFourier<3>>();
    test_case->initialize(boost::program_options::variables_map{});
    auto geom = test_case->get_geometry();
    ASSERT_TRUE(std::dynamic_pointer_cast<const paramsim::geom::Cube<3>>(geom));
}

TEST(CubeFourier, BoundaryTags)
{
    auto test_case = std::make_shared<paramsim::cases::cube::CubeFourier<3>>();
    test_case->initialize(boost::program_options::variables_map{});
    auto geom = test_case->get_geometry();
    dealii::Triangulation<3> tria;
    geom->generate_grid(tria, 3);
    geom->set_boundary_ids(tria);
    dealii::DoFHandler<3> dof_handler(tria);
    dealii::FE_Q<3> fe(1);
    dof_handler.distribute_dofs(fe);

    for(const auto &cell : dof_handler.active_cell_iterators()) {
        for(const auto &face : cell->face_iterators()) {
            if(!face->at_boundary()) {
                continue;
            }
            EXPECT_TRUE(face->boundary_id() == 1);
        }
    }
}

TEST(CubeFourierMinimalSurface, SolverConvergesWithP1In2D)
{
    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::MinimalSurface>(
                  16, 3, "minimal_surface",
                  minimal_surface_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeFourierMinimalSurface, SolverConvergesWithP1In3D)
{
    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::MinimalSurface>(
                  16, 2, "minimal_surface",
                  minimal_surface_maximum_newton_iterations)),
              nonlinear_tolerance);
}

TEST(CubeFourierCrossTerms, PoissonConvergesWithNonzeroCoefficientsIn2D)
{
    const TemporaryParameterFile parameter_file(cross_term_parameters_2d);

    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::PoissonCG>(
                  32, 3, "poisson_cg", poisson_maximum_newton_iterations,
                  "default", parameter_file.path())),
              nonlinear_tolerance);
}

TEST(CubeFourierCrossTerms, MinimalSurfaceConvergesWithNonzeroCoefficientsIn2D)
{
    const TemporaryParameterFile parameter_file(cross_term_parameters_2d);

    EXPECT_LT((solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::MinimalSurface>(
                  16, 3, "minimal_surface",
                  minimal_surface_maximum_newton_iterations, "default",
                  parameter_file.path())),
              nonlinear_tolerance);
}

TEST(CubeFourierCrossTerms, PoissonConvergesWithNonzeroCoefficientsIn3D)
{
    const TemporaryParameterFile parameter_file(cross_term_parameters_3d);

    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::PoissonCG>(
                  8, 3, "poisson_cg", poisson_maximum_newton_iterations,
                  "default", parameter_file.path())),
              nonlinear_tolerance);
}

TEST(CubeFourierCrossTerms, MinimalSurfaceConvergesWithNonzeroCoefficientsIn3D)
{
    const TemporaryParameterFile parameter_file(cross_term_parameters_3d);

    EXPECT_LT((solve_grid_sequence<3, paramsim::cases::cube::CubeFourier,
                                   paramsim::pde::MinimalSurface>(
                  16, 2, "minimal_surface",
                  minimal_surface_maximum_newton_iterations, "default",
                  parameter_file.path())),
              nonlinear_tolerance);
}

TEST(CubeFourierMinimalSurface, DefaultInitialGuessMatchesExplicitPoissonIn2D)
{
    const double default_residual =
        solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                            paramsim::pde::MinimalSurface>(
            4, 2, "minimal_surface", 15);
    const double explicit_residual =
        solve_grid_sequence<2, paramsim::cases::cube::CubeFourier,
                            paramsim::pde::MinimalSurface>(
            4, 2, "minimal_surface", 15, "none");

    EXPECT_NEAR(default_residual, explicit_residual, 1e-9);
    EXPECT_LT(default_residual, nonlinear_tolerance);
}

} // namespace
