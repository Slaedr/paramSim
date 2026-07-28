#ifndef PARAMSIM_CASES_CUBE_POLYNOMIAL_HPP_
#define PARAMSIM_CASES_CUBE_POLYNOMIAL_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/grid/grid_generator.h>

#include "../case.hpp"
#include "case_parameters.hpp"
#include "exponential.hpp"

namespace paramsim {
namespace cases {

namespace cube {
namespace polynomial {

using namespace dealii;

/**
 * @brief Runtime-sized multidimensional polynomial parameters.
 *
 * Coefficients are grouped by total degree and follow the ordering returned by
 * `degree_exponents`. The built-in default represents
 * `1 + Y + Y^2 + Y^3`.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
struct Params {
    static_assert(dim == 2 || dim == 3,
                  "Polynomial parameters require dimension 2 or 3.");

    std::array<double, dim> center{};
    std::vector<std::vector<double>> coefficients_by_degree = [] {
        constexpr unsigned default_degree_levels = 4;
        std::vector<std::vector<double>> coefficients;
        coefficients.reserve(default_degree_levels);
        for (unsigned degree = 0; degree < default_degree_levels; ++degree) {
            const auto exponents = degree_exponents<dim>(degree);
            std::vector<double> degree_coefficients(exponents.size(), 0.0);
            for (std::size_t index = 0; index < exponents.size(); ++index) {
                if (exponents[index][1] == degree) {
                    degree_coefficients[index] = 1.0;
                }
            }
            coefficients.push_back(std::move(degree_coefficients));
        }
        return coefficients;
    }();
};

/**
 * @brief Reads runtime-sized polynomial parameters from a file.
 *
 * The file always supplies three center coordinates. Only the first `dim`
 * coordinates are retained.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 * @param filename Parameter-file path.
 * @return Parsed center and coefficients grouped by total degree.
 * @throws std::runtime_error if the file is malformed.
 */
template <int dim>
Params<dim> read_parameters(const std::string& filename);

/**
 * @brief Evaluates a total-degree polynomial about a configurable center.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
class DirichletIn : public Function<dim> {
public:
    /** @brief Uses the built-in four-degree-level polynomial. */
    DirichletIn() = default;

    /**
     * @brief Uses supplied runtime-sized polynomial parameters.
     *
     * @param params Center and coefficients to evaluate.
     */
    explicit DirichletIn(const Params<dim>& params) : params_{params} {}

    /**
     * @brief Evaluates every configured monomial at a point.
     *
     * @param p Evaluation point.
     * @param component Ignored scalar component.
     * @return Polynomial value.
     */
    double value(const Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        std::array<double, dim> shifted_p{};
        for (int idim = 0; idim < dim; ++idim) {
            shifted_p[idim] = p[idim] - params_.center[idim];
        }

        double sum = 0.0;
        for (std::size_t degree = 0;
             degree < params_.coefficients_by_degree.size(); ++degree) {
            const auto exponents =
                degree_exponents<dim>(static_cast<unsigned>(degree));
            const auto& coefficients = params_.coefficients_by_degree[degree];
            for (std::size_t term = 0; term < exponents.size(); ++term) {
                double monomial = 1.0;
                for (int idim = 0; idim < dim; ++idim) {
                    monomial *=
                        std::pow(shifted_p[idim], exponents[term][idim]);
                }
                sum += coefficients[term] * monomial;
            }
        }
        return sum;
    }

    const Params<dim> params_;
};

template <int dim>
using RightHandSide = exponential::RightHandSide<dim>;

template <int dim>
using DirichletConstant = exponential::DirichletConstant<dim>;

} // namespace polynomial

namespace bpo = boost::program_options;

/**
 * @brief Cube case with a polynomial input-face boundary profile.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
class CubePolynomial final : public Case<dim> {
public:
    /**
     * @brief Initializes the polynomial cube case.
     *
     * @param params Parsed command-line parameters.
     */
    void initialize(const bpo::variables_map& params) override;
};

} // namespace cube
} // namespace cases
} // namespace paramsim

#endif
