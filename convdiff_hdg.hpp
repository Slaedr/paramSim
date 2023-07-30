
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

// Defines finite element spaces on the faces
// of the triangulation, which we refer to as the 'skeleton'.
// These finite elements do not have any support on the element
// interior, and they represent polynomials that have a single
// value on each codimension-1 surface, but admit discontinuities
// on codimension-2 surfaces.
#include <deal.II/fe/fe_face.h>

// Defines a new type of sparse matrix.  The
// regular <code>SparseMatrix</code> type stores indices to all non-zero
// entries.  The <code>ChunkSparseMatrix</code> takes advantage of the coupled
// nature of DG solutions.  It stores an index to a matrix sub-block of a
// specified size.  In the HDG context, this sub-block-size is actually the
// number of degrees of freedom per face defined by the skeleton solution
// field. This reduces the memory consumption of the matrix by up to one third
// and results in similar speedups when using the matrix in solvers.
#include <deal.II/lac/chunk_sparse_matrix.h>

// Deals with data output.  Since
// we have a finite element field defined on the skeleton of the mesh,
// we would like to visualize what that solution actually is.
// DataOutFaces does exactly this; the interface is the almost the same
// as the familiar DataOut, but the output only has codimension-1 data for
// the simulation.
#include <deal.II/numerics/data_out_faces.h>

namespace convdiff_hdg {
  
  enum RefinementMode
  {
    global_refinement,
    adaptive_refinement
  };

  using namespace dealii;

  /// Abstract type for a function on a facet
  template <int dim>
  class FaceFunction
  {
  public:
    virtual double value_normal(const Point<dim>& p, const Tensor<1,dim>& normal,
            const unsigned int = 0) const = 0;
  };

  class DomainGeometry
  {
  public:
      static constexpr int dim = 2;
      virtual void generate_grid(dealii::Triangulation<dim>& tria,
              const unsigned int initial_resolution) const = 0;
  };

  template <int dim>
  struct CDParams {
    /// Geometry
    std::shared_ptr<const DomainGeometry> geom;
    /// Convection velocity function
    std::shared_ptr<const TensorFunction<1,dim>> conv_vel_function;
    /// Source term
    std::shared_ptr<const Function<dim>> rhs_function;
    /// Dirichlet boundary condition
    std::shared_ptr<const Function<dim>> dirichlet_bc_function;
    /// Neumann boundary condition
    std::shared_ptr<const FaceFunction<dim>> neumann_bc_function;
    /// Exact solution (if available)
    std::shared_ptr<const Function<dim>> solution_function;
    std::shared_ptr<const Function<dim>> solution_n_gradient;

    unsigned int dirichlet_marker{0};
    unsigned int neumann_marker{1};
  };
  
  // The HDG solution procedure follows closely that of step-7. The major
  // difference is the use of three different sets of DoFHandler and FE
  // objects, along with the ChunkSparseMatrix and the corresponding solutions
  // vectors. We also use WorkStream to enable a multithreaded local solution
  // process which exploits the embarrassingly parallel nature of the local
  // solver. For WorkStream, we define the local operations on a cell and a
  // copy function into the global matrix and vector. We do this both for the
  // assembly (which is run twice, once when we generate the system matrix and
  // once when we compute the element-interior solutions from the skeleton
  // values) and for the postprocessing where we extract a solution that
  // converges at higher order.
  template <int dim>
  class HDG
  {
  public:
    HDG(const unsigned int degree, const RefinementMode refinement_mode,
        const CDParams<dim>& convdiff_params);
    void run(int num_cycles, unsigned int initial_resolution);

  private:
    void setup_system();
    void assemble_system(const bool reconstruct_trace = false);
    void solve();
    void postprocess();
    void refine_grid(int cycle, unsigned int initial_resolution);
    void output_results(const int cycle);

    // Data for the assembly and solution of the primal variables.
    struct PerTaskData;
    struct ScratchData;

    // Post-processing the solution to obtain $u^*$ is an element-by-element
    // procedure; as such, we do not need to assemble any global data and do
    // not declare any 'task data' for WorkStream to use.
    struct PostProcessScratchData;

    // The following three functions are used by WorkStream to do the actual
    // work of the program.
    void assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData &                                         scratch,
      PerTaskData &                                         task_data);

    void copy_local_to_global(const PerTaskData &data);

    void postprocess_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      PostProcessScratchData &                              scratch,
      unsigned int &                                        empty_data);


    Triangulation<dim> triangulation;

    // The 'local' solutions are interior to each element.  These
    // represent the primal solution field $u$ as well as the auxiliary
    // field $\mathbf{q}$.
    FESystem<dim>   fe_local;
    DoFHandler<dim> dof_handler_local;
    Vector<double>  solution_local;

    // The new finite element type and corresponding <code>DoFHandler</code> are
    // used for the global skeleton solution that couples the element-level
    // local solutions.
    FE_FaceQ<dim>   fe;
    DoFHandler<dim> dof_handler;
    Vector<double>  solution;
    Vector<double>  system_rhs;

    // As stated in the introduction, HDG solutions can be post-processed to
    // attain superconvergence rates of $\mathcal{O}(h^{p+2})$.  The
    // post-processed solution is a discontinuous finite element solution
    // representing the primal variable on the interior of each cell.  We define
    // a FE type of degree $p+1$ to represent this post-processed solution,
    // which we only use for output after constructing it.
    FE_DGQ<dim>     fe_u_post;
    DoFHandler<dim> dof_handler_u_post;
    Vector<double>  solution_u_post;

    // The degrees of freedom corresponding to the skeleton strongly enforce
    // Dirichlet boundary conditions, just as in a continuous Galerkin finite
    // element method. We can enforce the boundary conditions in an analogous
    // manner via an AffineConstraints object. In addition, hanging nodes are
    // handled in the same way as for continuous finite elements: For the face
    // elements which only define degrees of freedom on the face, this process
    // sets the solution on the refined side to coincide with the
    // representation on the coarse side.
    //
    // Note that for HDG, the elimination of hanging nodes is not the only
    // possibility &mdash; in terms of the HDG theory, one could also use the
    // unknowns from the refined side and express the local solution on the
    // coarse side through the trace values on the refined side. However, such
    // a setup is not as easily implemented in terms of deal.II loops and not
    // further analyzed.
    AffineConstraints<double> constraints;

    // The usage of the ChunkSparseMatrix class is similar to the usual sparse
    // matrices: You need a sparsity pattern of type ChunkSparsityPattern and
    // the actual matrix object. When creating the sparsity pattern, we just
    // have to additionally pass the size of local blocks.
    ChunkSparsityPattern      sparsity_pattern;
    ChunkSparseMatrix<double> system_matrix;

    // Same as step-7:
    const RefinementMode refinement_mode;
    ConvergenceTable     convergence_table;

    CDParams<dim> cdparams;
  };


}
