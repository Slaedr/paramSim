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
        for (int i = 0; i < dim; i++) {
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

template <int dim>
class MinSurfCubeLeft : public Case<dim> {
public:
    virtual void initialize(const bpo::variables_map &) override = 0;
    virtual void
    add_case_cmd_args(bpo::options_description &) const override = 0;

protected:
    void set_geometry_and_boundary(
        std::shared_ptr<dealii::Function<dim>> dirichlet1);
};

template <int dim>
class MinSurfCubeSinusoidal final : public MinSurfCubeLeft<dim> {
public:
    void initialize(const bpo::variables_map &) override;
    void add_case_cmd_args(bpo::options_description &) const override;
};

namespace minsurf_poly {

template <int dim>
struct Params {
    static constexpr int n_indep_coeffs = 4;

    std::array<double, n_indep_coeffs> ac{{1.0, 1.0, 1.0, 1.0}};
    double center{0.0};

    Params() {}

    Params(const std::array<double, n_indep_coeffs> &acoeffs,
           const double center_y)
        : ac{acoeffs}, center{center_y}
    {
    }
};

/**
 * Polynomial on left face that ensures the value is zero at both ends -1 and 1.
 *
 * Note that the four terms in Params are taken as the coefficients of
 * the four highest order terms.
 */
template <int dim>
class Dirichlet : public dealii::Function<dim> {
public:
    Dirichlet() : a(get_coeffs(params_.ac, params_.center)) {}

    Dirichlet(const Params<dim> &params)
        : params_{params}, a(get_coeffs(params_.ac, params_.center))
    {
    }

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
        constexpr int b_dim = 1;
        double value = 0;
        for (int i = 0; i < n_terms; i++) {
            value += a[i] * std::pow(p[b_dim] - params_.center, i);
        }
        return value;
    }

    const Params<dim> params_;
    static constexpr int n_indep_coeffs = Params<dim>::n_indep_coeffs;
    static constexpr int n_terms = n_indep_coeffs + 2;

protected:
    const std::array<double, n_terms> a;

    std::array<double, n_terms>
    get_coeffs(const std::array<double, n_indep_coeffs> &arr, const double y0)
    {
        double a1 = 0.0;
        for (int i = 0; i < n_indep_coeffs; i++) {
            a1 += arr[i] * (std::pow(-1.0, i) * std::pow(y0 + 1.0, i + 2) -
                            std::pow(1.0 - y0, i + 2));
        }
        a1 /= 2.0;
        double a0 = a1 * (y0 + 1.0);
        for (int i = 0; i < n_indep_coeffs; i++) {
            a0 += std::pow(-1.0, i + 1) * arr[i] * std::pow(y0 + 1.0, i + 2);
        }
        std::array<double, n_terms> ac;
        ac[0] = a0;
        ac[1] = a1;
        for (int i = 0; i < n_indep_coeffs; i++) {
            ac[i + 2] = arr[i];
        }
        return ac;
    }
};

} // namespace minsurf_poly

template <int dim>
class MinSurfCubePolynomial final : public MinSurfCubeLeft<dim> {
public:
    void initialize(const bpo::variables_map &) override;
    void add_case_cmd_args(bpo::options_description &) const override;
};

} // namespace cases
} // namespace paramsim

#endif // MINIMAL_SURFACE_H_
