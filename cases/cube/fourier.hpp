#ifndef PARAMSIM_CASES_CUBE_FOURIER_HPP_
#define PARAMSIM_CASES_CUBE_FOURIER_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/grid/grid_generator.h>

#include "../case.hpp"
#include "cube_case.hpp"
#include "exponential.hpp"

namespace paramsim {
namespace cases {

namespace cube {
namespace fourier {

using namespace dealii;

/**
 * @brief Tensor-product coefficients for one positive Fourier frequency.
 *
 * Coefficients are ordered lexicographically by directional factor, with
 * cosine before sine in each direction.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
struct Mode {
    static_assert(dim == 2 || dim == 3,
                  "Fourier modes require dimension 2 or 3.");

    static constexpr unsigned term_count = 1U << dim;
    std::array<double, term_count> coefficients = [] {
        std::array<double, term_count> values{};
        values.front() = 1.0;
        values.back() = 1.0;
        return values;
    }();
};

/**
 * @brief Runtime-sized Fourier boundary-profile parameters.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
struct Params {
    static_assert(dim == 2 || dim == 3,
                  "Fourier parameters require dimension 2 or 3.");

    std::vector<Mode<dim>> modes = std::vector<Mode<dim>>(2);
    double constant{1.0};
    std::array<double, dim> fundamental_wavelength = [] {
        std::array<double, dim> values{};
        values.fill(1.0);
        return values;
    }();
};

/**
 * @brief Reads runtime-sized Fourier parameters from a file.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 * @param filename Parameter-file path.
 * @return Parsed Fourier modes in increasing-frequency order.
 * @throws std::runtime_error if the file is malformed.
 */
template <int dim>
Params<dim> read_parameters(const std::string& filename);

/**
 * @brief Evaluates a multidimensional Fourier boundary profile.
 *
 * Each mode contains every tensor product of directional sine and cosine
 * terms. Products are ordered lexicographically with cosine before sine.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
class DirichletIn : public Function<dim> {
public:
    /** @brief Uses the built-in two-mode defaults. */
    DirichletIn() = default;

    /**
     * @brief Uses supplied runtime-sized parameters.
     *
     * @param params Fourier modes to evaluate.
     */
    explicit DirichletIn(const Params<dim>& params) : params_{params} {}

    /**
     * @brief Evaluates the constant and positive-frequency terms.
     *
     * @param p Evaluation point.
     * @param component Ignored scalar component.
     * @return Boundary-profile value.
     */
    double value(const Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        double sum = params_.constant;
        for (std::size_t index = 0; index < params_.modes.size(); ++index) {
            const double frequency = static_cast<double>(index + 1);
            std::array<double, dim> cosine_values{};
            std::array<double, dim> sine_values{};
            for (int idim = 0; idim < dim; ++idim) {
                const double angle = frequency * 2.0 * pi /
                                     params_.fundamental_wavelength[idim] *
                                     p[idim];
                cosine_values[idim] = std::cos(angle);
                sine_values[idim] = std::sin(angle);
            }

            for (unsigned term_index = 0;
                 term_index < Mode<dim>::term_count; ++term_index) {
                double product = 1.0;
                for (int idim = 0; idim < dim; ++idim) {
                    const unsigned direction_mask =
                        1U << static_cast<unsigned>(dim - idim - 1);
                    product *= (term_index & direction_mask) != 0U
                                   ? sine_values[idim]
                                   : cosine_values[idim];
                }
                sum += params_.modes[index].coefficients[term_index] * product;
            }
        }
        return sum;
    }

    const Params<dim> params_;
};

template <int dim>
using RightHandSide = exponential::RightHandSide<dim>;

} // namespace fourier

namespace bpo = boost::program_options;

/**
 * @brief Cube case with a tensor-product Fourier Dirichlet profile.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
class CubeFourier final : public CubeCase<dim> {
public:
    /**
     * @brief Initializes the case from optional runtime Fourier parameters.
     *
     * @param params Parsed command-line options.
     */
    void initialize(const bpo::variables_map& params) override;
};

} // namespace cube
} // namespace cases
} // namespace paramsim

#endif
