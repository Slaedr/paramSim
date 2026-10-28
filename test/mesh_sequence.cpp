#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include "../cases/poisson/verify.hpp"
#include "../pdes/poisson/poisson_cg.hpp"
#include "../solvers/grid_refinement_solve.hpp"

namespace ps = paramsim;

namespace {

template <int dim>
class LinearFunction : public dealii::Function<dim> {
public:
    double value(const dealii::Point<dim>& point,
                 const unsigned int = 0) const override
    {
        double result = 1.0;
        for (int direction = 0; direction < dim; ++direction) {
            result += static_cast<double>(direction + 1) * point[direction];
        }
        return result;
    }
};

template <int dim>
class CurvedFunction : public dealii::Function<dim> {
public:
    double value(const dealii::Point<dim>& point,
                 const unsigned int = 0) const override
    {
        double result = 0.0;
        for (int direction = 0; direction < dim; ++direction) {
            result += std::pow(point[direction], direction + 2);
        }
        return result;
    }
};

template <int dim>
std::shared_ptr<ps::cases::PoissonVerify<dim>> make_case()
{
    auto test_case = std::make_shared<ps::cases::PoissonVerify<dim>>();
    boost::program_options::variables_map parameters;
    test_case->initialize(parameters);
    return test_case;
}

template <int dim>
void check_uniform_sequence()
{
    const std::string prefix =
        "/tmp/paramsim-mesh-sequence-" + std::to_string(dim);
    const ps::PDEParams parameters{"poisson_cg", 1, 1, 3, 0, ps::CoarseGridSolution::interpolation, false, prefix};
    const ps::SolverParams solver_parameters{1e-8, 4};
    auto test_case = make_case<dim>();
    auto pde = std::make_shared<ps::pde::PoissonCG<dim>>(test_case, parameters);
    ps::solver::MeshSequence<dim> sequence;

    sequence.run_grid_refinement(parameters, solver_parameters, test_case, pde,
                                 ps::solver::none_init_pde, solver_parameters);

    ASSERT_EQ(sequence.mesh_count(), 3);
    const auto coarse_cell_count = sequence.get_mesh(0)->n_active_cells();
    for (int level = 0; level < sequence.mesh_count(); ++level) {
        const auto mesh_index = static_cast<std::size_t>(level);
        EXPECT_EQ(sequence.get_mesh(mesh_index)->n_active_cells(),
                  coarse_cell_count *
                      (1U << (dim * static_cast<unsigned int>(level))));
        if (level > 0) {
            EXPECT_NE(sequence.get_mesh(mesh_index - 1).get(),
                      sequence.get_mesh(mesh_index).get());
        }
    }
    EXPECT_EQ(sequence.get_mesh(0)->n_active_cells(), coarse_cell_count);
    std::remove((prefix + "-2.vtk").c_str());
}

} // namespace

TEST(MeshSequence, RetainsUniformMeshesInTwoDimensions)
{
    check_uniform_sequence<2>();
}

TEST(MeshSequence, RetainsUniformMeshesInThreeDimensions)
{
    check_uniform_sequence<3>();
}

TEST(DiscretePDEMesh, RebindsAndRejectsNullMeshes)
{
    const ps::PDEParams parameters{"poisson_cg", 1, 1, 1, 0, ps::CoarseGridSolution::interpolation, false,
                                   "/tmp/unused"};
    auto pde =
        std::make_shared<ps::pde::PoissonCG<2>>(make_case<2>(), parameters);
    const auto old_dof_count = pde->get_dof_handler().n_dofs();
    auto mesh = std::make_shared<dealii::Triangulation<2>>();
    mesh->copy_triangulation(*pde->get_mesh());
    mesh->refine_global(1);

    pde->set_mesh(mesh);

    EXPECT_EQ(pde->get_mesh(), mesh);
    EXPECT_GT(pde->get_dof_handler().n_dofs(), old_dof_count);
    EXPECT_THROW(pde->set_mesh(nullptr), std::invalid_argument);
}

TEST(DiscretePDEMesh, TransfersRepresentableFieldExactly)
{
    const ps::PDEParams parameters{"poisson_cg", 1, 2, 1, 0, ps::CoarseGridSolution::interpolation, false,
                                   "/tmp/unused"};
    auto test_case = make_case<2>();
    ps::pde::PoissonCG<2> coarse(test_case, parameters);
    ps::pde::PoissonCG<2> fine(test_case, parameters);
    ps::DiscretePDEBase::vector_type fine_solution;
    fine.allocate_solution_vector(fine_solution);
    dealii::VectorTools::interpolate(fine.get_dof_handler(),
                                     LinearFunction<2>(), fine_solution);
    fine.refine_mesh_and_interpolate_solution(fine_solution);

    ps::DiscretePDEBase::vector_type transferred;
    coarse.transfer_solution_from(fine, fine_solution, transferred);
    ps::DiscretePDEBase::vector_type expected;
    coarse.allocate_solution_vector(expected);
    dealii::VectorTools::interpolate(coarse.get_dof_handler(),
                                     LinearFunction<2>(), expected);
    transferred -= expected;

    EXPECT_LT(transferred.l2_norm(), 1e-12);
}

TEST(DiscretePDEMesh, AdaptiveTransferSatisfiesDestinationConstraints)
{
    ps::PDEParams parameters{"poisson_cg", 1, 3, 1, 0, ps::CoarseGridSolution::interpolation, true, "/tmp/unused"};
    auto test_case = make_case<2>();
    ps::pde::PoissonCG<2> fine(test_case, parameters);
    ps::DiscretePDEBase::vector_type fine_solution;
    fine.allocate_solution_vector(fine_solution);
    dealii::VectorTools::interpolate(fine.get_dof_handler(),
                                     CurvedFunction<2>(), fine_solution);
    fine.refine_mesh_and_interpolate_solution(fine_solution);
    auto intermediate_mesh = std::make_shared<dealii::Triangulation<2>>();
    intermediate_mesh->copy_triangulation(*fine.get_mesh());
    fine.refine_mesh_and_interpolate_solution(fine_solution);

    ps::pde::PoissonCG<2> destination(test_case, parameters);
    destination.set_mesh(intermediate_mesh);
    ps::DiscretePDEBase::vector_type transferred;
    destination.transfer_solution_from(fine, fine_solution, transferred);
    auto constrained = transferred;
    destination.impose_constraints(constrained);
    constrained -= transferred;

    EXPECT_LT(constrained.l2_norm(), 1e-12);
    for (const double value : transferred) {
        EXPECT_TRUE(std::isfinite(value));
    }
}
