#include <cmath>
#include <memory>

#include <gtest/gtest.h>
#include <boost/program_options/variables_map.hpp>

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/vector_tools.h>

#include "../../cases/cube/exponential.hpp"
#include "../../pdes/poisson/poisson_cg.hpp"

namespace {

using namespace paramsim;

/// The cube cases live on [-1, 1]^dim, so the domain volume is 2^dim.
template <int dim>
constexpr double domain_volume()
{
    return dim == 2 ? 4.0 : 8.0;
}

/// f(x) = x_0, whose L2 norm over the cube is known analytically.
template <int dim>
class FirstCoordinate : public dealii::Function<dim> {
public:
    double value(const dealii::Point<dim>& p,
                 const unsigned int = 0) const override
    {
        return p[0];
    }
};

/// Builds a discretization on the cube purely to exercise its norm routine.
template <int dim>
std::shared_ptr<pde::PoissonCG<dim>> make_discretization()
{
    auto test_case = std::make_shared<cases::cube::CubeExponential<dim>>();
    test_case->initialize(boost::program_options::variables_map{});
    const PDEParams pde_parameters{"poisson_cg", 1, 4, 1, false, "_"};
    return std::make_shared<pde::PoissonCG<dim>>(test_case, pde_parameters);
}

/* A constant field pins down the overall scaling of the norm: an
 * implementation that pairs each quadrature weight with the wrong number of
 * field values gets this wrong by a factor of sqrt(n_quadrature_points).
 */
TEST(ComputeLpNorm, ConstantFieldIn2D)
{
    auto discrete_pde = make_discretization<2>();
    DiscretePDEBase::vector_type u;
    discrete_pde->allocate_solution_vector(u);
    u = 2.0;

    EXPECT_NEAR(discrete_pde->compute_lp_norm(u, 2),
                2.0 * std::sqrt(domain_volume<2>()), 1e-12);
}

TEST(ComputeLpNorm, ConstantFieldIn3D)
{
    auto discrete_pde = make_discretization<3>();
    DiscretePDEBase::vector_type u;
    discrete_pde->allocate_solution_vector(u);
    u = 2.0;

    EXPECT_NEAR(discrete_pde->compute_lp_norm(u, 2),
                2.0 * std::sqrt(domain_volume<3>()), 1e-12);
}

/* A field that actually varies across a cell additionally pins down the
 * pairing of quadrature values with quadrature weights, which a constant field
 * cannot distinguish. Q1 elements represent x_0 exactly and the Gauss rule
 * integrates its square exactly, so the expected value is analytic.
 */
TEST(ComputeLpNorm, LinearFieldIn2D)
{
    auto discrete_pde = make_discretization<2>();
    DiscretePDEBase::vector_type u;
    discrete_pde->allocate_solution_vector(u);
    dealii::VectorTools::interpolate(discrete_pde->get_dof_handler(),
                                     FirstCoordinate<2>(), u);

    // \int_{[-1,1]^2} x^2 dx dy = (2/3) * 2
    EXPECT_NEAR(discrete_pde->compute_lp_norm(u, 2), std::sqrt(4.0 / 3.0),
                1e-12);
}

TEST(ComputeLpNorm, LinearFieldIn3D)
{
    auto discrete_pde = make_discretization<3>();
    DiscretePDEBase::vector_type u;
    discrete_pde->allocate_solution_vector(u);
    dealii::VectorTools::interpolate(discrete_pde->get_dof_handler(),
                                     FirstCoordinate<3>(), u);

    // \int_{[-1,1]^3} x^2 dx dy dz = (2/3) * 2 * 2
    EXPECT_NEAR(discrete_pde->compute_lp_norm(u, 2), std::sqrt(8.0 / 3.0),
                1e-12);
}

/* The L1 norm of a sign-changing field is the integral of its absolute value,
 * which distinguishes a genuine norm from a signed sum.
 */
TEST(ComputeLpNorm, L1NormUsesAbsoluteValues)
{
    auto discrete_pde = make_discretization<2>();
    DiscretePDEBase::vector_type u;
    discrete_pde->allocate_solution_vector(u);
    dealii::VectorTools::interpolate(discrete_pde->get_dof_handler(),
                                     FirstCoordinate<2>(), u);

    // \int_{[-1,1]^2} |x| dx dy = 1 * 2, whereas the signed integral is zero.
    EXPECT_NEAR(discrete_pde->compute_lp_norm(u, 1), 2.0, 1e-12);
}

} // namespace
