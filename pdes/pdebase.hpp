#ifndef PARAMSIM_PDEBASE_HPP_
#define PARAMSIM_PDEBASE_HPP_

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>


#include "../geometrybase.hpp"
#include "../cases/case.hpp"

namespace paramsim {

using gl_int_t = dealii::types::global_dof_index;
 
//enum class MeshRefineMode
//{
//  global_refinement,
//  adaptive_refinement
//};

struct PDEParams {
    std::string pde_solver;
    int fe_degree;
    unsigned initial_resolution;
    int refine_levels;
    bool is_adaptive;
    std::string output_path;
};

struct SolverParams {
    double tolerance;
    int max_its;
};

/// Any needed options for the system assembly operation.
struct AssemblyOptions {
    /** For hybridized discretizations, whether to assemble the 'local' domain system (true) or
     * the 'global' skeleton solution (false).
     */
    bool reconstruct_from_trace;
};

/// Spatial-dimension-agnostic abstract PDE discretization interface.
class DiscretePDEBase
{
public:
    using vector_type = dealii::Vector<double>;

    /// Sets parameters.
    DiscretePDEBase(const PDEParams& params, const SolverParams& solver_params);
    virtual ~DiscretePDEBase() { }

    /// Returns true if the system matrix is always symmetric and positive definite.
    virtual bool is_symm_positive_definite() const = 0;

    /// Impose constraints on a given solution vector, especially those related to hanging nodes.
    virtual void impose_constraints(vector_type& solution) const;

    /** @brief Refine the internally-stored mesh and interpolate the provided solution.
     *
     * Whether the refinement is uniform or adaptive depends on the PDEParams::is_adaptive
     * parameter.
     * NOTE: System must be setup again after calling this function.
     */
    virtual void refine_mesh_and_interpolate_solution(vector_type& solution) = 0;

    /// Setup up vectors and sparsity pattern to solve this PDE.
    /** \param initial_step  Whether this is the first time it is called or not.
     *  The difference is that the first time around we need to distribute
     *  the degrees of freedom and set the solution vector for $u^n$ to
     *  the correct size. The following times, the function is called after
     *  we have already done these steps as part of refining the mesh.
     */
    virtual void setup_system(bool initial_step, vector_type& rhs,
                              vector_type& update, dealii::SparsityPattern& spp) const = 0;

    virtual void allocate_solution_vector(vector_type& solution) const = 0;

    virtual void assemble_system(AssemblyOptions opts, const vector_type& state,
                                 dealii::SparseMatrix<double>& mat, vector_type& rhs) const = 0;

    /// Apply boundary values on the nonlinear update vector.
    virtual void apply_zero_boundary_values(vector_type& update,
                                            dealii::SparseMatrix<double>& mat,
                                            vector_type& rhs) const = 0;

    /** @brief Sets the boundary values on the solution vector.
     *
     *  Ensures that the solution vector's entries respect the
     *  boundary values for our problem.  Having refined the mesh (or just
     *  started computations), there might be new nodal points on the
     *  boundary. These have values that are simply interpolated from the
     *  previous mesh in `refine_mesh()`, instead of the correct boundary
     *  values. This is fixed up by setting all boundary nodes of the current
     *  solution vector explicit to the right value.
     */
    virtual void set_boundary_values(vector_type& solution) const = 0;

    virtual double compute_lp_norm(const vector_type& u, int p) const = 0;

    /// Write VTK output files for relevant fields of the solution.
    virtual void output_results(int mesh_number, const vector_type& solution) const = 0;

protected:
    const PDEParams params_;
    const SolverParams solver_params_;

    dealii::AffineConstraints<double> affine_constraints_;

    // vector_type solution_;
    // vector_type update_;
    // vector_type rhs_;

    /// Create the grid internally with the given 1D resolution.
    virtual void make_grid(unsigned n_cell_dir) = 0;
};

/// Parts of the discretization that depend on the spatial dimension and type of finite element.
/**
 * This is still an abstract class since it does not assemble the discrete problem.
 */
template <int dim, typename FE_t>
class DiscretePDE : public DiscretePDEBase
{
public:
    using DiscretePDEBase::vector_type;

    /// Sets parameters, creates basic finite element objects and generates the initial mesh.
    DiscretePDE(std::shared_ptr<const Case<dim>> test_case, const PDEParams& params,
              const SolverParams& solver_params);

    virtual ~DiscretePDE() { }

    void refine_mesh_and_interpolate_solution(vector_type& solution) override;

    /// Initializes vectors and system matrix sparsity pattern
    void setup_system(bool initial_step, vector_type& rhs,
                      vector_type& update, dealii::SparsityPattern& spp) const override;

    void allocate_solution_vector(vector_type& solution) const override;

    void apply_zero_boundary_values(vector_type& update,
                                    dealii::SparseMatrix<double>& mat,
                                    vector_type& rhs) const override;

    void set_boundary_values(vector_type& solution) const override;

    double compute_lp_norm(const vector_type& u, int p) const override;

    const dealii::Triangulation<dim>& get_triangulation() const { return tria_; }
    dealii::Triangulation<dim>& get_triangulation() { return tria_; }
    const dealii::DoFHandler<dim>& get_dof_handler() const { return dof_handler_; }

protected:
    std::shared_ptr<const Case<dim>> case_;

    dealii::Triangulation<dim> tria_;
    dealii::DoFHandler<dim> dof_handler_;
    FE_t fe_;

    /// Creates the internal triangulation and updates the DoFHandler and the affine constraints.
    void make_grid(unsigned n_cell_dir) override;
};

template <int dim>
std::unique_ptr<DiscretePDEBase> create_discrete_pde(std::shared_ptr<const Case<dim>> test_case,
                                                      const PDEParams& params,
                                                      const SolverParams& solver_params);

}

#endif
