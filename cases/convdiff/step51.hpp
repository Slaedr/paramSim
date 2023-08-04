#ifndef CONVDIFF_HDG_CASES_STEP51_HPP_
#define CONVDIFF_HDG_CASES_STEP51_HPP_

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/grid/grid_generator.h>

#include "../case.hpp"
#include "convdiffcase.hpp"
#include "../../pdes/pdebase.hpp"
#include "../../pdes/convdiff/convdiff_hdg.hpp"

namespace paramsim {
namespace cases {

namespace step51 {


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

  // @sect3{Equation data}
  //
  // The structure of the analytic solution is the same as in step-7. There are
  // two exceptions. Firstly, we also create a solution for the 3d case, and
  // secondly, we scale the solution so its norm is of order unity for all
  // values of the solution width.
  template <int dim>
  class SolutionBase
  {
  public:
    static constexpr unsigned int n_source_centers = 3;
  protected:
    const std::array<Point<dim>, n_source_centers> source_centers;
    const std::array<double, n_source_centers> coeffs{{1.0, 1.0, 1.0}};
    const double width = 1.0/5.0;

    std::array<Point<dim>, n_source_centers> get_source_centers() const
    {
        static_assert(dim > 0 && dim <= 3, "dim must be 1, 2 or 3!");
        std::array<Point<dim>, n_source_centers> points;
        if constexpr (dim == 1) {
            points =
            {{Point<1>(-1.0 / 3.0), Point<1>(0.0), Point<1>(+1.0 / 3.0)}};
        } else if constexpr (dim == 2) {
            points =
            {{Point<2>(-0.5, +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.5, -0.5)}};
        }
        else {
            points = {{
              Point<3>(-0.5, +0.5, 0.25),
              Point<3>(-0.6, -0.5, -0.125),
              Point<3>(+0.5, -0.5, 0.5)}};
        }
        return points;
    }

    SolutionBase() : source_centers(get_source_centers())
    { }

    SolutionBase(const std::array<Point<dim>, n_source_centers> centers,
            const std::array<double, n_source_centers> coefficients,
            const double width_sigma)
        : source_centers{centers}, coeffs{coefficients}, width{width_sigma}
    { }
  };


  template <int dim>
  class Solution : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    static constexpr unsigned int n_centers = SolutionBase<dim>::n_source_centers;

    Solution()
    { }
    
    Solution(const std::array<Point<dim>, n_centers> centers,
            const std::array<double, n_centers> coefficients,
            const double width_sigma)
        : SolutionBase<dim>(centers, coefficients, width_sigma)
    { }

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      double sum = 0;
      for (unsigned int i = 0; i < this->n_source_centers; ++i)
      {
          const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
          sum += this->coeffs[i] *
            std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
      }

      return sum /
             std::pow(2. * numbers::PI * this->width * this->width, dim / 2.);
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> sum;
      for (unsigned int i = 0; i < this->n_source_centers; ++i)
      {
          const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

          sum += this->coeffs[i] *
            (-2 / (this->width * this->width) *
             std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) *
             x_minus_xi);
      }

      return sum /
             std::pow(2. * numbers::PI * this->width * this->width, dim / 2.);
    }
  };


  // This class implements a function where the scalar solution and its negative
  // gradient are collected together. This function is used when computing the
  // error of the HDG approximation and its implementation is to simply call
  // value and gradient function of the Solution class.
  template <int dim>
  class SolutionAndGradient : public Function<dim>
  {
  public:
    SolutionAndGradient()
      : Function<dim>(dim + 1)
    {}

    static constexpr int n_centers = SolutionBase<dim>::n_source_centers;
    
    SolutionAndGradient(const std::array<Point<dim>, n_centers> centers,
            const std::array<double, n_centers> coefficients,
            const double width_sigma)
        : Function<dim>(dim+1), solution(centers, coefficients, width_sigma)
    { }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  v) const override
    {
      AssertDimension(v.size(), dim + 1);
      Tensor<1, dim> grad = solution.gradient(p);
      for (unsigned int d = 0; d < dim; ++d)
        v[d] = -grad[d];
      v[dim] = solution.value(p);
    }

  private:
    Solution<dim> solution;
  };



  // Next comes the implementation of the convection velocity. As described in
  // the introduction, we choose a velocity field that is $(y, -x)$ in 2d and
  // $(y, -x, 1)$ in 3d. This gives a divergence-free velocity field.
  template <int dim>
  class ConvectionVelocity : public TensorFunction<1, dim>
  {
  public:
    ConvectionVelocity()
      : TensorFunction<1, dim>()
    {}

    virtual Tensor<1, dim> value(const Point<dim> &p) const override
    {
      Tensor<1, dim> convection;
      switch (dim)
        {
          case 1:
            convection[0] = 1;
            break;
          case 2:
            convection[0] = p[1];
            convection[1] = -p[0];
            break;
          case 3:
            convection[0] = p[1];
            convection[1] = -p[0];
            convection[2] = 1;
            break;
          default:
            Assert(false, ExcNotImplemented());
        }
      return convection;
    }
  };


  // The last function we implement is the right hand side for the
  // manufactured solution. It is very similar to step-7, with the exception
  // that we now have a convection term instead of the reaction term. Since
  // the velocity field is incompressible, i.e., $\nabla \cdot \mathbf{c} =
  // 0$, the advection term simply reads $\mathbf{c} \nabla u$.
  template <int dim>
  class RightHandSide : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    static constexpr int n_centers = SolutionBase<dim>::n_source_centers;

    RightHandSide()
    { }

    RightHandSide(const std::array<Point<dim>, n_centers> centers,
            const std::array<double, n_centers> coefficients,
            const double width_sigma)
        : SolutionBase<dim>(centers, coefficients, width_sigma)
    { }

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      ConvectionVelocity<dim> convection_velocity;
      Tensor<1, dim>          convection = convection_velocity.value(p);
      double                  sum        = 0;
      for (unsigned int i = 0; i < this->n_source_centers; ++i)
        {
          const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

          sum += this->coeffs[i] *
            ((2 * dim - 2 * convection * x_minus_xi -
              4 * x_minus_xi.norm_square() / (this->width * this->width)) /
             (this->width * this->width) *
             std::exp(-x_minus_xi.norm_square() / (this->width * this->width)));
        }

      return sum /
             std::pow(2. * numbers::PI * this->width * this->width, dim / 2.);
    }
  };

  template <int dim>
  class Neumann : public FaceFunction<dim>
  {
  public:
    static constexpr int n_centers = SolutionBase<dim>::n_source_centers;

    Neumann() { }

    Neumann(const std::array<Point<dim>, n_centers> centers,
            const std::array<double, n_centers> coefficients,
            const double width_sigma)
        : exact_solution(centers, coefficients, width_sigma)
    { }

    virtual double value_normal(const Point<dim>& p, const Tensor<1,dim>& normal,
            const unsigned int = 0) const override
    {
        return -exact_solution.gradient(p) *
                            normal +
                          convection.value(p) * normal *
                            exact_solution.value(p);
    }
  private:
    Solution<dim> exact_solution;
    ConvectionVelocity<dim> convection;
  };

}



template <int dim>
class Step51 final : public convdiffcase_verification<dim> {
public:
    /** Sets up case parameters from command line parameters.
     *
     * \warning If the parameter width is not passed, it uses all default parameters,
     * ignoring any other parameters even if passed on the command line.
     */
    void initialize(const bpo::variables_map&) override;

    void add_case_cmd_args(bpo::options_description&) const override;

private:
    static const std::array<std::string, 3> dimnames;
};


namespace unused {
  template <int dim>
  class Rectangle : public DomainGeometry<dim>
  {
  public:
      virtual void generate_grid(dealii::Triangulation<dim>& tria,
              const unsigned int initial_resolution) const override
      {
          const dealii::Tensor<1,dim> dims = upper_right - lower_left;
          const auto num_cells_per_dim = get_cells_per_dim(dims, initial_resolution);
          // The 'true' below "colorizes" the mesh; boundary IDs are set
          dealii::GridGenerator::subdivided_hyper_rectangle(tria, num_cells_per_dim,
                  lower_left, upper_right);
      }
  
  protected:
      dealii::Point<dim> lower_left{-1.0, -1.0};
      dealii::Point<dim> upper_right{2.0, 1.0};
  
      std::vector<unsigned int> get_cells_per_dim(const dealii::Tensor<1,dim>& dims,
              const unsigned int init_res) const
      {
          static_assert(dim == 2, "Not implemented for anything but 2D!");
          const double min_dim = std::min(dims[0], dims[1]);
          const double max_dim = std::max(dims[0], dims[1]);
          const int min_idx = dims[0] < dims[1] ? 0 : 1;
          std::vector<unsigned int> num_cells_per_dim(dim);
          num_cells_per_dim[min_idx] = init_res;
          const auto other_res =
              static_cast<unsigned int>(std::round(max_dim / min_dim * init_res));
          const auto other_idx = (min_idx + 1) % dim;
          num_cells_per_dim[other_idx] = other_res;
          std::cout << "Num cells per dim = " << num_cells_per_dim[0] << ", " 
              << num_cells_per_dim[1] << std::endl;
          return num_cells_per_dim;
      }
  };
}

}
}

#endif
