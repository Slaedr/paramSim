#include "gelfand.hpp"

#include <cmath>
#include <vector>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

namespace paramsim {
namespace pde {

template <int dim>
Gelfand<dim>::Gelfand(std::shared_ptr<const Case<dim>> test_case,
                      const PDEParams& params)
    : DiscretePDE<dim, fe_type>(test_case, params)
{
}

template <int dim>
void Gelfand<dim>::assemble_system(AssemblyOptions, const vector_type& state,
                                   dealii::SparseMatrix<double>& mat,
                                   vector_type& rhs) const
{
    const dealii::QGauss<dim> quadrature_formula(fe_.degree + 1);
    dealii::FEValues<dim> fe_values(
        fe_, quadrature_formula,
        dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);
    std::vector<double> solution_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> solution_gradients(n_q_points);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);

    mat = 0;
    rhs = 0;

    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(state, solution_values);
        fe_values.get_function_gradients(state, solution_gradients);

        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q = 0; q < n_q_points; ++q) {
            const double exponential = std::exp(solution_values[q]);
            const double forcing = case_->get_right_hand_side()->value(
                fe_values.quadrature_point(q));
            const double weight = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double phi_i = fe_values.shape_value(i, q);
                const auto grad_phi_i = fe_values.shape_grad(i, q);

                cell_rhs(i) += ((exponential + forcing) * phi_i -
                                solution_gradients[q] * grad_phi_i) *
                               weight;

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) +=
                        (grad_phi_i * fe_values.shape_grad(j, q) -
                         exponential * phi_i * fe_values.shape_value(j, q)) *
                        weight;
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            rhs(local_dof_indices[i]) += cell_rhs(i);
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                mat.add(local_dof_indices[i], local_dof_indices[j],
                        cell_matrix(i, j));
            }
        }
    }

    affine_constraints_.condense(mat);
    affine_constraints_.condense(rhs);
}

template <int dim>
void Gelfand<dim>::evaluate_residual(const vector_type& state,
                                     vector_type& rhs) const
{
    const dealii::QGauss<dim> quadrature_formula(fe_.degree + 1);
    dealii::FEValues<dim> fe_values(
        fe_, quadrature_formula,
        dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::Vector<double> cell_rhs(dofs_per_cell);
    std::vector<double> solution_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> solution_gradients(n_q_points);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
        dofs_per_cell);

    rhs = 0;

    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(state, solution_values);
        fe_values.get_function_gradients(state, solution_gradients);

        cell_rhs = 0;

        for (unsigned int q = 0; q < n_q_points; ++q) {
            const double exponential = std::exp(solution_values[q]);
            const double forcing = case_->get_right_hand_side()->value(
                fe_values.quadrature_point(q));
            const double weight = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                cell_rhs(i) +=
                    ((exponential + forcing) * fe_values.shape_value(i, q) -
                     solution_gradients[q] * fe_values.shape_grad(i, q)) *
                    weight;
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

    affine_constraints_.condense(rhs);
}

template class Gelfand<2>;
template class Gelfand<3>;

} // namespace pde
} // namespace paramsim
