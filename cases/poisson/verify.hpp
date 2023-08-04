#ifndef PARAMSIM_CASES_POISSON_VERIFY_HPP_
#define PARAMSIM_CASES_POISSON_VERIFY_HPP_

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/grid/grid_generator.h>

#include "../../pdes/pdebase.hpp"
#include "../case.hpp"

namespace paramsim {
namespace cases {

namespace poisson_verify {

  using namespace dealii;

  template <int dim>
  class Cube : public DomainGeometry<dim>
  {
  public:
      Cube(const std::vector<typename DomainGeometry<dim>::bc_mark_desc>& bcmarks)
          : DomainGeometry<dim>(bcmarks)
      { }
  
      virtual void generate_grid(dealii::Triangulation<dim>& tria,
              const unsigned int initial_resolution) const override
      {
          const double lower_left = -1;
          const double upper_right = 1;
          dealii::GridGenerator::subdivided_hyper_cube(tria, initial_resolution,
                  lower_left, upper_right);
      }
  };

  template <int dim>
  struct Params {
    static constexpr int n_centers = 2;
    std::array<Point<dim>, n_centers> centers{{
        Point<dim>{-0.5,-0.5},Point<dim>{0.55,0.55}
    }};
    std::array<double, n_centers> coeffs{{0.5, 0.5}};
    double width{0.25};

    double gamma;

    double get_multiplier() const {
        return 1.0 / std::pow(2. * numbers::PI * width * width, dim / 2.);
    }

    Params() {
        // don't use initializer list to make sure other vars are default-initialized
        gamma = get_multiplier();
        //std::cout << "Multiplier = " << gamma << std::endl;
    }

    Params(const std::array<Point<dim>, n_centers>& centerss,
        const std::array<double, n_centers>& coefficients,
        const double hill_width)
        : centers{centerss}, coeffs{coefficients}, width{hill_width},
        gamma{get_multiplier()}
    { }
  };

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution()
    { }
    
    Solution(const Params<dim>& params) : params_{params}
    { }

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      double sum = 0;
      for (int i = 0; i < Params<dim>::n_centers; ++i)
      {
          const Tensor<1, dim> x_minus_xi = p - params_.centers[i];
          sum += params_.coeffs[i] *
            std::exp(-x_minus_xi.norm_square() / (params_.width * params_.width));
      }

      return sum * params_.gamma;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> sum;
      for (unsigned int i = 0; i < Params<dim>::n_centers; ++i)
      {
          const Tensor<1, dim> x_minus_xi = p - params_.centers[i];

          sum += params_.coeffs[i] *
            (-2 / (params_.width * params_.width) *
             std::exp(-x_minus_xi.norm_square() / (params_.width * params_.width)) *
             x_minus_xi);
      }

      return sum * params_.gamma;
    }

    const Params<dim> params_;
  };


  // The last function we implement is the right hand side for the
  // manufactured solution.
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    static constexpr int n_centers = Params<dim>::n_centers;

    RightHandSide()
    { }

    RightHandSide(const Params<dim>& params) : params_{params}
    { }

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      double sum = 0;
      for (int i = 0; i < Params<dim>::n_centers; ++i)
      {
        const Tensor<1, dim> x_minus_xi = p - params_.centers[i];
        const double arg = -x_minus_xi.norm_square() / (params_.width * params_.width);
        sum += params_.coeffs[i] * std::exp(arg) * (1.0 + arg);
      }
      return sum * 4.0 * params_.gamma / (params_.width * params_.width);
    }

    const Params<dim> params_;
  };

  template <int dim>
  class Neumann : public FaceFunction<dim>
  {
  public:
    static constexpr int n_centers = 2;

    Neumann() { }

    Neumann(const Params<dim>& params)
        : exact_solution(params)
    { }

    virtual double value_normal(const Point<dim>& p, const Tensor<1,dim>& normal,
            const unsigned int = 0) const override
    {
        return -exact_solution.gradient(p) * normal;
    }
  private:
    Solution<dim> exact_solution;
  };

}


namespace bpo = boost::program_options;


template <int dim>
class PoissonVerify final : public CaseWithExactSolution<CaseWithNeumannBC<Case<dim>>>
{
public:
    void initialize(const bpo::variables_map&) override;
    void add_case_cmd_args(bpo::options_description&) const override;

private:
    static const std::array<std::string, 3> dimnames;
};

}
}

#endif
