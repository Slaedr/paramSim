#ifndef PARAMSIM_CASES_MINIMAL_SURFACE_VERIFY_HPP_
#define PARAMSIM_CASES_MINIMAL_SURFACE_VERIFY_HPP_


#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include "../case.hpp"


namespace paramsim {
namespace cases {


namespace minsurf_verify {

using namespace dealii;

/** Manufactured minimal-surface solution.
 *
 * @tparam dim Spatial dimension.
 */
template <int dim>
class Solution : public Function<dim>
{
public:
    /** Evaluate the manufactured solution.
     *
     * @param p Evaluation point.
     * @param component Scalar component, ignored.
     * @return Product of the coordinate-wise sine values.
     */
    double value(const Point<dim>& p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        double prod = 1.0;
        for (int i = 0; i < dim; ++i) {
            prod *= std::sin(p[i]);
        }
        return prod;
    }
};

/** Right-hand side corresponding to the manufactured solution.
 *
 * @tparam dim Spatial dimension.
 */
template <int dim>
class RightHandSide : public Function<dim>
{
public:
    /** Evaluate the forcing for the minimal-surface equation.
     *
     * @param p Evaluation point.
     * @param component Scalar component, ignored.
     * @return Manufactured forcing value.
     */
    double value(const Point<dim>& p,
                 const unsigned int component = 0) const override
    {
        (void)component;
        double solution = 1.0;
        for (int i = 0; i < dim; ++i) {
            solution *= std::sin(p[i]);
        }

        Tensor<1, dim> gradient;
        Tensor<2, dim> hessian;
        for (int i = 0; i < dim; ++i) {
            gradient[i] = std::cos(p[i]);
            for (int k = 0; k < dim; ++k) {
                if (k != i) {
                    gradient[i] *= std::sin(p[k]);
                }
            }
            hessian[i][i] = -solution;

            for (int j = i + 1; j < dim; ++j) {
                double mixed_derivative =
                    std::cos(p[i]) * std::cos(p[j]);
                for (int k = 0; k < dim; ++k) {
                    if (k != i && k != j) {
                        mixed_derivative *= std::sin(p[k]);
                    }
                }
                hessian[i][j] = mixed_derivative;
                hessian[j][i] = mixed_derivative;
            }
        }

        double gradient_hessian_gradient = 0.0;
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                gradient_hessian_gradient +=
                    gradient[i] * hessian[i][j] * gradient[j];
            }
        }

        const double normalization =
            std::sqrt(1.0 + gradient.norm_square());
        return dim * solution / normalization +
               gradient_hessian_gradient /
                   std::pow(normalization, 3);
    }
};

} // namespace minsurf_verify


namespace bpo = boost::program_options;


/// Verification test case for minimal surface problem on a ball domain.
template <int dim>
class MinSurfBallVerify final : public Case<dim>, public HasExactSolution<dim>
{
public:
    void initialize(const bpo::variables_map&) override;
    void add_case_cmd_args(bpo::options_description&) const override;
};

/// Verification test case for minimal surface problem on a square domain.
template <int dim>
class MinSurfCubeVerify final : public Case<dim>, public HasExactSolution<dim>
{
public:
    void initialize(const bpo::variables_map&) override;
    void add_case_cmd_args(bpo::options_description&) const override;
};


}
}

#endif // MINIMAL_SURFACE_H_
