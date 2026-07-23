#ifndef PARAMSIM_CASES_POISSON_EXPS_HPP_
#define PARAMSIM_CASES_POISSON_EXPS_HPP_

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/grid/grid_generator.h>

#include "../case.hpp"
#include "verify.hpp"

namespace paramsim {
namespace cases {

namespace poisson_exp {

using namespace dealii;

template <int dim>
struct Params {
    static constexpr int n_centers = 3;

    std::array<Point<dim>, n_centers> centers = [] {
        std::array<Point<dim>, n_centers> values{};
        values[0][0] = -1.0;
        values[0][1] = -0.67;
        values[1][0] = -1.0;
        values[1][1] = -0.01;
        values[2][0] = -1.0;
        values[2][1] = 0.66;
        return values;
    }();

    std::array<double, n_centers> coeffs{{0.27, 0.35, -0.34}};

    /// Same width for all three hills
    double width{0.4};

    double gamma{get_multiplier()};

    double get_multiplier() const
    {
        constexpr int profile_dim = 2;
        return 1.0 /
               std::pow(2. * numbers::PI * width * width, profile_dim / 2.);
    }

    Params() {}

    Params(const std::array<Point<dim>, n_centers> &centerss,
           const std::array<double, n_centers> &coefficients,
           const double hill_width)
        : centers{centerss}, coeffs{coefficients}, width{hill_width},
          gamma{get_multiplier()}
    {
    }
};

// The last function we implement is the right hand side for the
// manufactured solution.
template <int dim>
class RightHandSide : public Function<dim> {
public:
    static constexpr int n_centers = 1;

    virtual double value(const Point<dim> &p,
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

template <int dim>
class DirichletIn : public Function<dim> {
public:
    DirichletIn() {}

    DirichletIn(const Params<dim> &params) : params_{params} {}

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
        double sum = 0;
        for (int i = 0; i < Params<dim>::n_centers; ++i) {
            // const Tensor<1, dim> x_minus_xi = p - params_.centers[i];
            double distance_square = 0.0;
            constexpr int profile_dim = 2;
            for (int idim = 0; idim < profile_dim; ++idim) {
                distance_square +=
                    std::pow(p[idim] - params_.centers[i][idim], 2);
            }
            // sum += params_.coeffs[i] *
            //   std::exp(-x_minus_xi.norm_square() /
            //            (params_.width * params_.width));
            sum += params_.coeffs[i] *
                   std::exp(-distance_square / (params_.width * params_.width));
        }

        return sum * params_.gamma;
    }

    const Params<dim> params_;
};

template <int dim>
using DirichletConstant = cases::DirichletConstant<dim>;
} // namespace poisson_exp

namespace bpo = boost::program_options;

template <int dim>
class PoissonBCExp final : public Case<dim> {
public:
    void initialize(const bpo::variables_map &) override;
    void add_case_cmd_args(bpo::options_description &) const override;

private:
    static const std::array<std::string, 3> dimnames;
};

} // namespace cases
} // namespace paramsim

#endif
