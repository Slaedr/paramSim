#ifndef PARAMSIM_CASES_CUBE_EXPONENTIAL_HPP_
#define PARAMSIM_CASES_CUBE_EXPONENTIAL_HPP_

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/grid/grid_generator.h>

#include "../case.hpp"

namespace paramsim {
namespace cases {

namespace cube {
namespace exponential {

using namespace dealii;

/**
 * @brief Parameters for one Gaussian center in the boundary profile.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
struct GaussianCenter {
    static_assert(dim == 2 || dim == 3,
                  "Gaussian centers require dimension 2 or 3.");

    std::array<double, dim> coordinates{};
    double coefficient{0.0};
    double width{1.0};
};

/**
 * @brief Runtime-sized Gaussian boundary-profile parameters.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
struct Params {
    static_assert(dim == 2 || dim == 3,
                  "Exponential parameters require dimension 2 or 3.");

    // Supplying two coordinates initializes z to zero when dim is three.
    std::vector<GaussianCenter<dim>> centers{
        {{{-1.0, -0.67}}, 0.27, 0.4},
        {{{-1.0, -0.01}}, 0.35, 0.4},
        {{{-1.0, 0.66}}, -0.34, 0.4}};
};

/**
 * @brief Reads runtime-sized exponential parameters from a file.
 *
 * All three schema coordinates are validated. Only the first `dim`
 * coordinates are retained.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 * @param filename Parameter-file path.
 * @return Parsed centers in file order.
 * @throws std::runtime_error if the file is malformed.
 */
template <int dim>
Params<dim> read_parameters(const std::string& filename);

// The last function we implement is the right hand side for the
// manufactured solution.
template <int dim>
class RightHandSide : public Function<dim> {
public:
    static constexpr int n_centers = 1;

    virtual double value(const Point<dim>& p,
                         const unsigned int /*component*/ = 0) const override
    {
        double sum = -0.0625;
        for (int i = 0; i < n_centers; ++i) {
            const Tensor<1, dim> x_minus_xi = p - centers[i];
            const double arg = -x_minus_xi.norm_square() / (width * width);
            sum += coeffs[i] * std::exp(arg) *
                   (static_cast<double>(dim) / 2.0 + arg);
        }
        return sum * 4.0 * gamma / (width * width);
    }

private:
    std::array<Point<dim>, n_centers> centers = [] {
        std::array<Point<dim>, n_centers> values{};
        values[0][0] = -0.125;
        values[0][1] = 0.125;
        return values;
    }();

    std::array<double, n_centers> coeffs{{1.0}};

    double width{0.5};
    double gamma{get_multiplier()};

    double get_multiplier() const
    {
        return 1.0 / std::pow(2. * numbers::PI * width * width, dim / 2.);
    }
};

/**
 * @brief Evaluates the Gaussian boundary profile on the cube input face.
 *
 * The current profile uses x and y in both supported dimensions.
 *
 * @tparam dim Active spatial dimension; must be 2 or 3.
 */
template <int dim>
class DirichletIn : public Function<dim> {
public:
    /** @brief Uses the built-in three-center defaults. */
    DirichletIn() = default;

    /**
     * @brief Uses supplied runtime-sized parameters.
     *
     * @param params Gaussian centers to evaluate.
     */
    explicit DirichletIn(const Params<dim>& params) : params_{params} {}

    /**
     * @brief Evaluates the normalized sum of Gaussian centers.
     *
     * @param p Evaluation point.
     * @param component Ignored scalar component.
     * @return Boundary-profile value.
     */
    double value(const Point<dim>& p,
                 const unsigned int /*component*/ = 0) const override
    {
        constexpr std::size_t profile_dim = 2;
        double sum = 0.0;
        for (const auto& center : params_.centers) {
            double distance_square = 0.0;
            for (std::size_t idim = 0; idim < profile_dim; ++idim) {
                distance_square +=
                    std::pow(p[idim] - center.coordinates[idim], 2);
            }
            const double width_square = center.width * center.width;
            const double multiplier =
                1.0 / std::pow(2.0 * numbers::PI * width_square,
                               profile_dim / 2.0);
            sum += center.coefficient *
                   std::exp(-distance_square / width_square) * multiplier;
        }
        return sum;
    }

    const Params<dim> params_;
};

template <int dim>
using DirichletConstant = cases::DirichletConstant<dim>;
} // namespace exponential

namespace bpo = boost::program_options;

template <int dim>
class CubeExponential final : public Case<dim> {
public:
    void initialize(const bpo::variables_map&) override;
};

} // namespace cube
} // namespace cases
} // namespace paramsim

#endif
