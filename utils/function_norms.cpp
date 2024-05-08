#include "function_norms.hpp"

#include <cmath>
#include <cassert>

#include <deal.II/fe/fe_q.h>


namespace paramsim {
namespace utils {

template <typename scalar, int dim>
scalar compute_Lp_norm(dealii::FEValues<dim>& fe_values, const dealii::DoFHandler<dim>& dof_handler,
                       const dealii::Vector<scalar>& u, const int p)
{
    auto update_flags = fe_values.get_update_flags();
    assert(update_flags & dealii::update_values);
    assert(update_flags & dealii::update_JxW_values);
    auto& fe = fe_values.get_fe();
    const auto dofs_per_cell = fe.n_dofs_per_cell();
    const auto n_q_points = fe_values.get_quadrature().size();
    std::vector<scalar> u_quadrature_values(n_q_points);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    scalar normp = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(u, u_quadrature_values);
        for (unsigned int q = 0; q < n_q_points; ++q) {
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                normp += std::pow(u_quadrature_values[i], p) * fe_values.JxW(q);
            }
        }
    }

    return std::pow(normp, 1.0/p);
}

template double compute_Lp_norm(dealii::FEValues<2> &fe_values,
                                const dealii::DoFHandler<2> &dof_handler,
                                const dealii::Vector<double> &u, int p);

}
}
