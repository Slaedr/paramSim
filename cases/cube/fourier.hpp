#ifndef PARAMSIM_CASES_CUBE_FOURIER_HPP_
#define PARAMSIM_CASES_CUBE_FOURIER_HPP_

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

template <int dim>
struct Params {
    static constexpr int n_modes = 2;

    std::array<double, n_modes> ac{{1.0, 1.0}};
    std::array<double, n_modes> bc{{1.0, 1.0}};
    double a0{1.0};
    double f_wavelength{1.0};

    Params() {}

    Params(const std::array<double, n_modes> &acoeffs,
           const std::array<double, n_modes> &bcoeffs, const double a0_coeff,
           const double fundamental_wavelength)
        : ac{acoeffs}, bc{bcoeffs}, a0{a0_coeff},
          f_wavelength{fundamental_wavelength}
    {
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
        double sum = params_.a0;
        for (int i = 0; i < Params<dim>::n_modes; ++i) {
            sum += params_.ac[i] *
                       std::cos(i * 2 * pi / params_.f_wavelength * p[1]) +
                   params_.bc[i] *
                       std::sin(i * 2 * pi / params_.f_wavelength * p[1]);
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
