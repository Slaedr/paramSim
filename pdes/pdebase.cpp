
#include "pdebase.hpp"

#include <stdexcept>
#include <memory>
#include <iostream>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include "../utils/error_handling.hpp"
#include "../utils/function_norms.hpp"
#include "poisson/poisson_cg.hpp"
#include "nonlinear_elliptic/minimal_surface.hpp"

namespace paramsim {


DiscretePDEBase::DiscretePDEBase(const PDEParams& params)
    : params_{params}
{
}

void DiscretePDEBase::impose_constraints(vector_type& solution) const
{
    affine_constraints_.distribute(solution);
}

template <int dim, typename FE_t>
DiscretePDE<dim,FE_t>::DiscretePDE(std::shared_ptr<const Case<dim>> test_case,
                                   const PDEParams& params)
    : DiscretePDEBase(params), case_{test_case}, dof_handler_(tria_),
      fe_(params.fe_degree)
{
    this->make_grid(params.initial_resolution);
}

template <int dim, typename FE_t>
void DiscretePDE<dim,FE_t>::make_grid(const unsigned n_cell_dir)
{
    tria_.clear();
    auto geom = case_->get_geometry();
    geom->generate_grid(tria_, n_cell_dir);
    geom->set_boundary_ids(tria_);

    std::cout << "   Number of active cells: " << tria_.n_active_cells()
              << std::endl
              << "   Total number of cells: " << tria_.n_cells()
              << std::endl;

    dof_handler_.distribute_dofs(fe_);
    affine_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(dof_handler_,
                                            affine_constraints_);
    affine_constraints_.close();
}

template <int dim, typename FE_t>
void DiscretePDE<dim,FE_t>::refine_mesh_and_interpolate_solution(vector_type& solution)
{
    dealii::SolutionTransfer<dim> solution_transfer(dof_handler_);

    // We retrieve the old solution interpolated to the new
    // mesh. Since the SolutionTransfer function does not actually store the
    // values of the old solution, but rather indices, we need to preserve the
    // old solution vector until we have gotten the new interpolated
    // values. Thus, we have the old values written into a temporary vector,
    // and only delete it after interpolating them into the solution vector object.
    const vector_type old_solution = solution;

    if(params_.is_adaptive) {
        dealii::Vector<float> estimated_error_per_cell(tria_.n_active_cells());

        dealii::KellyErrorEstimator<dim>::estimate(
          dof_handler_,
          dealii::QGauss<dim-1>(fe_.degree + 1),
          std::map<dealii::types::boundary_id, const dealii::Function<dim> *>(),
          solution,
          estimated_error_per_cell);

        dealii::GridRefinement::refine_and_coarsen_fixed_number(tria_,
                                                        estimated_error_per_cell,
                                                        0.3,
                                                        0.03);

        // Then we need an additional step: if, for example, you flag a cell that
        // is once more refined than its neighbor, and that neighbor is not
        // flagged for refinement, we would end up with a jump of two refinement
        // levels across a cell interface.  To avoid these situations, the library
        // will silently also have to refine the neighbor cell once. It does so by
        // calling the Triangulation::prepare_coarsening_and_refinement function
        // before actually doing the refinement and coarsening.  This function
        // flags a set of additional cells for refinement or coarsening, to
        // enforce rules like the one-hanging-node rule.  The cells that are
        // flagged for refinement and coarsening after calling this function are
        // exactly the ones that will actually be refined or coarsened. Usually,
        // you don't have to do this by hand
        // (Triangulation::execute_coarsening_and_refinement does this for
        // you). However, we need to initialize the SolutionTransfer class and it
        // needs to know the final set of cells that will be coarsened or refined
        // in order to store the data from the old mesh and transfer to the new
        // one. Thus, we call the function by hand:
        tria_.prepare_coarsening_and_refinement();

        // With this out of the way, we initialize a SolutionTransfer object with
        // the present DoFHandler and attach the solution vector to it, followed
        // by doing the actual refinement and distribution of degrees of freedom
        // on the new mesh
        solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
        tria_.execute_coarsening_and_refinement();
    }
    else {
        tria_.prepare_coarsening_and_refinement();
        solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
        tria_.refine_global(1);
    }

    dof_handler_.distribute_dofs(fe_);
    solution.reinit(dof_handler_.n_dofs());
    solution_transfer.interpolate(solution);

    // On the new mesh, there are different hanging nodes, for which we have to
    // compute constraints again, after throwing away previous content of the
    // object. To be on the safe side, we should then also make sure that the
    // current solution's vector entries satisfy the hanging node constraints
    // (see the discussion in the documentation of the SolutionTransfer class
    // for why this is necessary). We could do this by calling
    // `affine_constraints_.distribute(solution_)` explicitly; we
    // omit this step because this will happen at the end of the call to
    // `set_boundary_values()` below, and it is not necessary to do it twice.
    affine_constraints_.clear();
    dealii::DoFTools::make_hanging_node_constraints(dof_handler_,
                                                    affine_constraints_);
    affine_constraints_.close();

    // Once we have the interpolated solution and all information about
    // hanging nodes, we have to make sure that the $u^n$ we now have
    // actually has the correct boundary values. As explained at the end of
    // the introduction, this is not automatically the case even if the
    // solution before refinement had the correct boundary values, and so we
    // have to explicitly make sure that it now has:
    //set_boundary_values(solution);

    // We end the function by updating all the remaining data structures,
    // indicating to <code>setup_dofs()</code> that this is not the first
    // go-around and that it needs to preserve the content of the solution
    // vector:
    //setup_system(false);
}

// As always in the setup-system function, we set up the variables of the
// finite element method. There are same differences to step-6, because
// there we start solving the PDE from scratch in every refinement cycle
// whereas here we need to take the solution from the previous mesh onto the
// current mesh. Consequently, we can't just reset solution vectors. The
// argument passed to this function thus indicates whether we can
// distributed degrees of freedom (plus compute constraints) and set the
// solution vector to zero or whether this has happened elsewhere already
// (specifically, in <code>refine_mesh()</code>).
template <int dim, typename FE_t>
void DiscretePDE<dim,FE_t>::setup_system(const bool,
                                         vector_type& rhs, vector_type& update,
                                         dealii::SparsityPattern& spp) const
{
    std::cout << "  Number of degrees of freedom: " << dof_handler_.n_dofs()
              << std::endl;
    update.reinit(dof_handler_.n_dofs());
    rhs.reinit(dof_handler_.n_dofs());

    dealii::DynamicSparsityPattern dsp(dof_handler_.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler_, dsp);

    affine_constraints_.condense(dsp);

    spp.copy_from(dsp);
}

template <int dim, typename FE_t>
void DiscretePDE<dim,FE_t>::allocate_solution_vector(vector_type& solution) const
{
    solution.reinit(dof_handler_.n_dofs());
}

template <int dim, typename FE_t>
void DiscretePDE<dim,FE_t>::apply_zero_boundary_values(vector_type& update,
                                                       dealii::SparseMatrix<double>& mat,
                                                       vector_type& rhs) const
{
    // apply zero boundary values to the linear system that defines the Newton updates
    // $\delta u^n$:
    for(auto bc : case_->get_dirichlet_bcs()) {
        std::map<dealii::types::global_dof_index, double> boundary_values;
        dealii::VectorTools::interpolate_boundary_values(
            dof_handler_, bc.bc_id, dealii::Functions::ZeroFunction<dim>(), boundary_values);
        dealii::MatrixTools::apply_boundary_values(boundary_values, mat, update, rhs);
    }
}

// If we have a hanging node right next to a new boundary node, then its value
// must also be adjusted to make sure that the finite element field
// remains continuous.
// This is what the call in the last line of this function does.
template <int dim, typename FE_t>
void DiscretePDE<dim,FE_t>::set_boundary_values(vector_type& solution) const
{
  for(auto bc : case_->get_dirichlet_bcs()) {
    std::map<dealii::types::global_dof_index, double> boundary_values;
    dealii::VectorTools::interpolate_boundary_values(dof_handler_,
                                             bc.bc_id, *bc.bc_fn,
                                             boundary_values);
    for (auto &boundary_value : boundary_values) {
      solution(boundary_value.first) = boundary_value.second;
    }
  }

  affine_constraints_.distribute(solution);
}

template <int dim, typename FE_t>
double DiscretePDE<dim, FE_t>::compute_lp_norm(const vector_type& u, const int p) const
{
    // set up FEValues for residual norm computation
    const dealii::QGauss<dim> quadrature_formula(fe_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_, quadrature_formula,
                                    dealii::update_JxW_values | dealii::update_values);
    return utils::compute_Lp_norm(fe_values, dof_handler_, u, p);
}

template class DiscretePDE<2, dealii::FE_Q<2>>;

template <int dim>
std::unique_ptr<DiscretePDEBase> create_discrete_pde(std::shared_ptr<const Case<dim>> test_case,
                                                     const PDEParams& params)
{
    if(params.pde_solver == "poisson_cg") {
        return std::make_unique<pde::PoissonCG<dim>>(test_case, params);
    } else if(params.pde_solver == "minimal_surface") {
        return std::make_unique<pde::MinimalSurface<dim>>(test_case, params);
    } else {
        throw std::runtime_error("Unsupported PDE solver!");
    }
}

template
std::unique_ptr<DiscretePDEBase> create_discrete_pde(std::shared_ptr<const Case<2>>,
                                                     const PDEParams&);

}
