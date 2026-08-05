#ifndef PARAMSIM_CASES_GELFAND_VERIFY_HPP_
#define PARAMSIM_CASES_GELFAND_VERIFY_HPP_

#include <cmath>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include "../case.hpp"

namespace paramsim {
namespace cases {

namespace gelfand_verify {

/** Manufactured solution for the Gelfand equation.
 *
 * @tparam dim Spatial dimension.
 */
template <int dim>
class Solution : public dealii::Function<dim> {
public:
    /** Evaluate the manufactured solution.
     *
     * @param point Evaluation point.
     * @param component Scalar component, ignored.
     * @return Product of the coordinate-wise sine values.
     */
    double value(const dealii::Point<dim>& point,
                 const unsigned int component = 0) const override
    {
        (void)component;
        double solution = 1.0;
        for (int direction = 0; direction < dim; ++direction) {
            solution *= std::sin(point[direction]);
        }
        return solution;
    }
};

/** Manufactured forcing for the Gelfand equation.
 *
 * @tparam dim Spatial dimension.
 */
template <int dim>
class RightHandSide : public dealii::Function<dim> {
public:
    /** Evaluate the forcing corresponding to the manufactured solution.
     *
     * @param point Evaluation point.
     * @param component Scalar component, ignored.
     * @return Manufactured forcing value.
     */
    double value(const dealii::Point<dim>& point,
                 const unsigned int component = 0) const override
    {
        (void)component;
        double solution = 1.0;
        for (int direction = 0; direction < dim; ++direction) {
            solution *= std::sin(point[direction]);
        }
        return static_cast<double>(dim) * solution - std::exp(solution);
    }
};

} // namespace gelfand_verify

namespace bpo = boost::program_options;

/** Manufactured-solution verification case for the Gelfand equation.
 *
 * Uses the cube [-1, 1]^dim and applies the exact solution as a Dirichlet
 * condition on the entire boundary.
 *
 * @tparam dim Spatial dimension.
 */
template <int dim>
class GelfandVerify final : public Case<dim>, public HasExactSolution<dim> {
public:
    /** Initialize the forcing, exact solution, boundary data, and geometry.
     *
     * @param params Case-specific command-line parameters, unused.
     */
    void initialize(const bpo::variables_map& params) override;

    /** Add case-specific command-line options.
     *
     * The verification case has no case-specific options.
     *
     * @param description Command-line option description, unchanged.
     */
    void
    add_case_cmd_args(bpo::options_description& description) const override;
};

} // namespace cases
} // namespace paramsim

#endif // PARAMSIM_CASES_GELFAND_VERIFY_HPP_
