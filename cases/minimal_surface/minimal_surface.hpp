#ifndef PARAMSIM_CASES_MINIMAL_SURFACE_HPP_
#define PARAMSIM_CASES_MINIMAL_SURFACE_HPP_

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include "../case.hpp"

namespace paramsim {
namespace cases {

namespace minsurf_sin {

using namespace dealii;

template <int dim>
struct Params {
    static constexpr int n_modes = 2;

    std::array<double, n_modes> ac{{1.0, 1.0}}; //< TODO: Remove
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

/// Right hand side for the manufactured solution.
template <int dim>
class RightHandSide : public Function<dim> {
public:
    RightHandSide() {}

    virtual double value(const Point<dim> & = 0,
                         const unsigned int /*component*/ = 0) const override
    {
        return 0.0;
    }
};

/**
 * Sinusoidal function on a boundary with zero values at the ends -1 and 1.
 */
template <int dim>
class Dirichlet : public Function<dim> {
public:
    Dirichlet() {}

    Dirichlet(const Params<dim> &params) : params_{params} {}

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
        double sum = 0;
        constexpr int profile_dim = 2;
        for (int i = 0; i < profile_dim; i++) {
            sum += p[i];
        }
        // return std::sin(2*pi/params_.f_wavelength*sum);
        double value = params_.a0;
        for (int imode = 0; imode < Params<dim>::n_modes; imode++) {
            // value += params_.ac[imode] * std::cos(imode *
            // 2*pi/params_.f_wavelength*sum)
            //   + params_.bc[imode] * std::sin(imode *
            //   2*pi/params_.f_wavelength*sum);
            value += params_.bc[imode] *
                     std::sin(imode * 2 * pi / params_.f_wavelength * sum);
        }
        return value;
    }

    const Params<dim> params_;
};

template <int dim>
using DirichletConstant = cases::DirichletConstant<dim>;

} // namespace minsurf_sin

namespace bpo = boost::program_options;

template <int dim>
class MinSurfDiskSinusoidal final : public Case<dim> {
public:
    void initialize(const bpo::variables_map &) override;
    void add_case_cmd_args(bpo::options_description &) const override;
};

} // namespace cases
} // namespace paramsim

#endif // MINIMAL_SURFACE_H_
