#ifndef PARAMSIM_CASES_MINSURF_GAUSSIANS_HPP_
#define PARAMSIM_CASES_MINSURF_GAUSSIANS_HPP_

#include <cmath>

#include "minimal_surface.hpp"


namespace paramsim{
namespace cases {

namespace minsurf_gauss {

template <int dim>
struct Params {
    static constexpr int n_centers = 3;

    /// Y-coordinates of Gaussian hill centers
    const std::array<double, n_centers> yc{{0.0, 0.0, 0.0}};
    /// Coefficients of the hills
    const std::array<double, n_centers> a{{1.0, 1.0, 1.0}};
    /// Std deviation of the Gaussian hills
    const double w{0.5};

    Params() { }

    Params(const std::array<double, n_centers>& centers,
           const std::array<double, n_centers>& coeffs, const double width)
        : yc(centers), a(coeffs), w{width}
    { }
};

/**
 * Gaussian hills on left face that ensures the value is zero at both ends -1 and 1.
 *
 * Note that the four terms in Params are taken as the coefficients of the four highest order
 * terms.
 */
template <int dim>
class Dirichlet : public dealii::Function<dim>
{
public:
    Dirichlet()
      : cs(get_coeffs(params_))
    { }

    Dirichlet(const Params<dim>& params)
      : params_{params},  cs(get_coeffs(params))
    { }

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
        constexpr int b_dim = 1;
        double value = cs[0] + cs[1]*p[b_dim];
        for(int i = 0; i < n_centers; i++) {
            value += params_.a[i] * std::exp(-std::pow(p[b_dim] - params_.yc[i], 2) /
                                             (params_.w*params_.w));
        }
        return value;
    }

    const Params<dim> params_;
    static constexpr int n_centers = Params<dim>::n_centers;

protected:
    /// Coeffs of constant and linear terms to ensure zeros at boundaries
    const std::array<double, 2> cs;

    std::array<double, 2> get_coeffs(const Params<dim>& par)
    {
        std::array<double,2> c{{0.0, 0.0}};
        for(int i = 0; i < n_centers; i++) {
            c[0] += par.a[i] * (-std::exp(-std::pow(1.0+par.yc[i],2)/(par.w*par.w))
                                -std::exp(-std::pow(1.0-par.yc[i],2)/(par.w*par.w)));
            c[1] += par.a[i] * ( std::exp(-std::pow(1.0+par.yc[i],2)/(par.w*par.w))
                                -std::exp(-std::pow(1.0-par.yc[i],2)/(par.w*par.w)));
        }
        c[0] /= 2;
        c[1] /= 2;
        return c;
    }
};

} // namespace minsurf_gauss

template <int dim>
class MinSurfCubeGaussians final : public MinSurfCubeLeft<dim>
{
public:
    void initialize(const bpo::variables_map&) override;
    void add_case_cmd_args(bpo::options_description&) const override;
};

}
}


#endif // PARAMSIM_CASES_MINSURF_EXPS_HPP_
