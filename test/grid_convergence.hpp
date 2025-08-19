#ifndef PARAMSIM_GRID_CONVERGENCE_HPP_
#define PARAMSIM_GRID_CONVERGENCE_HPP_

#include "../pdes/pdebase.hpp"
#include "../cases/case.hpp"

namespace paramsim {
namespace testutils {

/// Computes rate of grid convergence of a PDE case under uniform refinement.
/**
 * @return  Returns the slope of the convergence on log-log scale.
 */
template <int dim>
double test_grid_convergence(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params,
                             SolverParams sparams);

}
}


#endif // PARAMSIM_GRID_CONVERGENCE_HPP_
