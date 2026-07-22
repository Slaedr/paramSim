#ifndef PARAMSIM_CASES_MINIMAL_SURFACE_VERIFY_HPP_
#define PARAMSIM_CASES_MINIMAL_SURFACE_VERIFY_HPP_


#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include "../case.hpp"


namespace paramsim {
namespace cases {


namespace minsurf_verify {

using namespace dealii;

template <int dim>
class Solution : public Function<dim>
{
public:
    double value(const Point<dim>& p, unsigned) const override
    {
        double prod = 1.0;
        for(int i = 0; i < dim; i++) {
            prod *= std::sin(p[i]);
        }
        return prod;
    }
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    double value(const Point<dim>& p, unsigned) const override
    {
        static_assert(dim == 2, "Not yet defined for 3D!");
        const double b = std::sqrt(1.0 +
            std::cos(p[0])*std::cos(p[0])*std::sin(p[1])*std::sin(p[1]) +
            std::sin(p[0])*std::sin(p[0])*std::cos(p[1])*std::cos(p[1]));
        double f1 = 1.0/std::pow(b, 3);
        double f2 = 2.0*std::sin(p[0])*std::sin(p[1])*b*b;
        f2 += 0.5*(std::cos(p[0])*std::sin(p[1])*std::sin(2*p[0])*std::cos(2*p[1]) +
                   std::sin(p[0])*std::cos(p[1])*std::sin(2*p[1])*std::cos(2*p[0]));
        return f1*f2;
    }
};

} // namespace minsurf_verify


namespace bpo = boost::program_options;


/// Verification test case for minimal surface problem on a ball domain.
template <int dim>
class MinSurfBallVerify final : public Case<dim>, public HasExactSolution<dim>
{
public:
    void initialize(const bpo::variables_map&) override;
    void add_case_cmd_args(bpo::options_description&) const override;
};

/// Verification test case for minimal surface problem on a square domain.
template <int dim>
class MinSurfCubeVerify final : public Case<dim>, public HasExactSolution<dim>
{
public:
    void initialize(const bpo::variables_map&) override;
    void add_case_cmd_args(bpo::options_description&) const override;
};


}
}

#endif // MINIMAL_SURFACE_H_
