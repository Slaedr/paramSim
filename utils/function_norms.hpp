#ifndef PARAMSIM_UTILS_FUNCTION_NORMS_HPP_
#define PARAMSIM_UTILS_FUNCTION_NORMS_HPP_


#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/dofs/dof_handler.h>


namespace paramsim {
namespace utils {

/**
 * Computes the _function_ L^p norm of a given finite element function.
 *
 * @param fe_values  deal.ii FEValues object that is initialized correctly with the FE,
 *   quadrature rule and update flags (for correct shape values and JxW values).
 */
template <typename scalar, int dim>
scalar compute_Lp_norm(dealii::FEValues<dim>& fe_values, const dealii::DoFHandler<dim>& dof_handler,
                       const dealii::Vector<scalar>& u, int p);

}
}


#endif // PARAMSIM_UTILS_FUNCTION_NORMS_HPP_
