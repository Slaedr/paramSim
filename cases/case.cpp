#include "case.hpp"

#include "poisson/verify.hpp"
#include "convdiff/step51.hpp"

namespace paramsim {

template <int dim>
std::unique_ptr<Case<dim>> create_case(const std::string case_str)
{
    if(case_str == "convdiff_step51") {
        return std::make_unique<cases::Step51<dim>>();
    } else if(case_str == "poisson_verify") {
        return std::make_unique<cases::PoissonVerify<dim>>();
    } else {
        throw std::runtime_error("Non-existent case!");
    }
}

template std::unique_ptr<Case<2>> create_case(const std::string case_str);

}