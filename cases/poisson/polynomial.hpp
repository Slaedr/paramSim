#ifndef PARAMSIM_CASES_POISSON_POLYNOMIAL_HPP_
#define PARAMSIM_CASES_POISSON_POLYNOMIAL_HPP_

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/grid/grid_generator.h>

#include "../../pdes/pdebase.hpp"
#include "../case.hpp"
#include "verify.hpp"
#include "exps.hpp"

namespace paramsim {
namespace cases {

namespace poisson_poly {

  using namespace dealii;

  template <int dim>
  using Cube = paramsim::cases::poisson_verify::Cube<dim>;

  template <int dim>
  struct Params {
    static constexpr int n_terms = 4;

    std::array<double, n_terms> ac{{1.0, 1.0, 1.0, 1.0}};
    double center{0.0};

    Params() { }

    Params(const std::array<double, n_terms>& acoeffs, const double center_y)
        : ac{acoeffs}, center{center_y}
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
      double sum = 0;
      for (int i = 0; i < Params<dim>::n_terms; ++i)
      {
          sum += params_.ac[i] * std::pow(p[1]-params_.center, i);
      }

      return sum;
    }

    const Params<dim> params_;
  };

  template <int dim>
  using RightHandSide = paramsim::cases::poisson_exp::RightHandSide<dim>;

  template <int dim>
  using DirichletConstant = poisson_exp::DirichletConstant<dim>;

}


namespace bpo = boost::program_options;


template <int dim>
class PoissonBCPolynomial final : public Case<dim>
{
public:
    void initialize(const bpo::variables_map&) override;
    void add_case_cmd_args(bpo::options_description&) const override;
};

}
}

#endif
