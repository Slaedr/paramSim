#ifndef PARAMSIM_CASES_CUBE_FOURIER_HPP_
#define PARAMSIM_CASES_CUBE_FOURIER_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/grid/grid_generator.h>

#include "../case.hpp"
#include "exponential.hpp"

namespace paramsim {
namespace cases {

namespace cube {
namespace fourier {

using namespace dealii;

/**
 * @brief Sine and cosine coefficients for one positive Fourier frequency.
 */
struct Mode {
    double cosine_coefficient{0.0};
    double sine_coefficient{0.0};
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

    std::vector<Mode> modes{{1.0, 1.0}, {1.0, 1.0}};
    double constant{1.0};
    double fundamental_wavelength{1.0};
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
 * @brief Evaluates a Fourier boundary profile along the y coordinate.
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
        const double fundamental_angular_frequency =
            2.0 * pi / params_.fundamental_wavelength;
        for (std::size_t index = 0; index < params_.modes.size(); ++index) {
            const double frequency = static_cast<double>(index + 1);
            const double angle =
                frequency * fundamental_angular_frequency * p[1];
            sum += params_.modes[index].cosine_coefficient * std::cos(angle) +
                   params_.modes[index].sine_coefficient * std::sin(angle);
        }
        return sum;
    }

    const Params<dim> params_;
};

template <int dim>
using RightHandSide = exponential::RightHandSide<dim>;

template <int dim>
using DirichletConstant = cases::DirichletConstant<dim>;

} // namespace fourier

namespace bpo = boost::program_options;

template <int dim>
class CubeFourier final : public Case<dim> {
public:
    void initialize(const bpo::variables_map &) override;
    void add_case_cmd_args(bpo::options_description &) const override;
};

} // namespace cube
} // namespace cases
} // namespace paramsim

#endif
