#ifndef PARAMSIM_CASES_CONVDIFFCASE_HPP_
#define PARAMSIM_CASES_CONVDIFFCASE_HPP_

#include <deal.II/base/tensor_function.h>

#include "../case.hpp"

namespace paramsim {

template <typename Base>
class ConvDiffCase : public Base
{
public:
    static constexpr int dim = Base::dim;

    std::shared_ptr<const dealii::TensorFunction<1, dim>> get_convection_velocity() const {
        return conv_vel_;
    }

protected:
    std::shared_ptr<dealii::TensorFunction<1, dim>> conv_vel_;
};

template <int dim>
using convdiffcase_verification =
    ConvDiffCase<CaseWithExactSolutionAndGradient<
        CaseWithExactSolution<CaseWithNeumannBC<Case<dim>>>
    >>;

}

#endif
