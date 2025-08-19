#ifndef PARAMSIM_RUN_HANDLER_HPP_
#define PARAMSIM_RUN_HANDLER_HPP_

#include <memory>

#include "../pdes/pdebase.hpp"
#include "../cases/case.hpp"

namespace paramsim {

template <int dim>
struct RunData {
    PDEParams pde_params;
    SolverParams solver_params;
    std::shared_ptr<const Case<dim>> test_case;
};

template <int dim>
RunData<dim> get_run_data(int nargs, const char *const argv[]);

}


#endif // PARAMSIM_RUN_HANDLER_HPP_
