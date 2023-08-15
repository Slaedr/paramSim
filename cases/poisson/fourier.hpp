#ifndef PARAMSIM_CASES_POISSON_FOURIER_HPP_
#define PARAMSIM_CASES_POISSON_FOURIER_HPP_

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/grid/grid_generator.h>

#include "../../pdes/pdebase.hpp"
#include "../case.hpp"
#include "verify.hpp"
#include "exps.hpp"

namespace paramsim {
namespace cases {

namespace poisson_fourier {

  using namespace dealii;

  template <int dim>
  using Cube = paramsim::cases::poisson_verify::Cube<dim>;

  template <int dim>
  struct Params {
    static constexpr int n_modes = 2;

    std::array<double, n_modes> ac{{1.0, 1.0}};
    std::array<double, n_modes> bc{{1.0, 1.0}};
    double a0{1.0};
    double f_wavelength{1.0};

    Params() { }

    Params(const std::array<double, n_modes>& acoeffs,
           const std::array<double, n_modes>& bcoeffs,
           const double a0_coeff, const double fundamental_wavelength)
        : ac{acoeffs}, bc{bcoeffs}, a0{a0_coeff}, f_wavelength{fundamental_wavelength}
    { }
  };

  template <int dim>
  class DirichletIn : public Function<dim>
  {
  public:
    DirichletIn()
    { }
    
    DirichletIn(const Params<dim>& params) : params_{params}
    { }

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      double sum = params_.a0;
      for (int i = 0; i < Params<dim>::n_modes; ++i)
      {
          sum += params_.ac[i] * std::cos(i*2*pi/params_.f_wavelength*p[1])
            + params_.bc[i] * std::sin(i*2*pi/params_.f_wavelength*p[1]);
      }

      return sum;
    }

    static constexpr double pi = 3.14159265358979323846;
    const Params<dim> params_;
  };

  template <int dim>
  using RightHandSide = paramsim::cases::poisson_exp::RightHandSide<dim>;

  template <int dim>
  using DirichletConstant = poisson_exp::DirichletConstant<dim>;

}


namespace bpo = boost::program_options;


template <int dim>
class PoissonBCFourier final : public Case<dim>
{
public:
    void initialize(const bpo::variables_map&) override;
    void add_case_cmd_args(bpo::options_description&) const override;
};

}
}

#endif
