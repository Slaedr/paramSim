#include "convdiff_hdg.hpp"

#include <algorithm>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <memory>

namespace paramsim {
namespace pde {
  
  using namespace dealii;

  // @sect3{The ConvdiffHDG class implementation}

  // @sect4{Constructor}
  // The constructor is similar to those in other examples, with the exception
  // of handling multiple DoFHandler and FiniteElement objects. Note that we
  // create a system of finite elements for the local DG part, including the
  // gradient/flux part and the scalar part.
  template <int dim>
  ConvdiffHDG<dim>
    ::ConvdiffHDG(std::shared_ptr<const convdiffcase_verification<dim>> tcase,
        const int degree, const unsigned initial_resolution,
        const MeshRefineMode refinement_mode, const int num_cycles)
    : PDESolver<dim>(tcase, degree, initial_resolution)
    , refinement_mode_(refinement_mode), num_cycles_{num_cycles}
    , fe_local(FE_DGQ<dim>(degree), dim, FE_DGQ<dim>(degree), 1)
    , dof_handler_local(triangulation)
    , fe(degree)
    , dof_handler(triangulation)
    , fe_u_post(degree + 1)
    , dof_handler_u_post(triangulation)
  {}

  // @sect4{ConvdiffHDG::setup_system}
  // The system for an HDG solution is setup in an analogous manner to most
  // of the other tutorial programs.  We are careful to distribute dofs with
  // all of our DoFHandler objects.  The @p solution and @p system_matrix
  // objects go with the global skeleton solution.
  template <int dim>
  void ConvdiffHDG<dim>::setup_system()
  {
    dof_handler_local.distribute_dofs(fe_local);
    dof_handler.distribute_dofs(fe);
    dof_handler_u_post.distribute_dofs(fe_u_post);

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    solution_local.reinit(dof_handler_local.n_dofs());
    solution_u_post.reinit(dof_handler_u_post.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[this->tcase_->get_dirichlet_marker()] =
        this->tcase_->get_dirichlet_bc().get();
    // Project boundary values to compute nodal values; these are stored in constrains.
    VectorTools::project_boundary_values(dof_handler,
                                         boundary_functions,
                                         QGauss<dim - 1>(fe.degree + 1),
                                         constraints);
    constraints.close();

    // When creating the chunk sparsity pattern, we first create the usual
    // dynamic sparsity pattern and then set the chunk size, which is equal
    // to the number of dofs on a face, when copying this into the final
    // sparsity pattern.
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      //sparsity_pattern.copy_from(dsp, fe.n_dofs_per_face());
      sparsity_pattern.copy_from(dsp);
    }
    system_matrix.reinit(sparsity_pattern);
  }



  // @sect4{ConvdiffHDG::PerTaskData}
  // Next comes the definition of the local data structures for the parallel
  // assembly. The first structure @p PerTaskData contains the local vector
  // and matrix that are written into the global matrix, whereas the
  // ScratchData contains all data that we need for the local assembly. There
  // is one variable worth noting here, namely the boolean variable @p
  // trace_reconstruct. As mentioned in the introduction, we solve the HDG
  // system in two steps. First, we create a linear system for the skeleton
  // system where we condense the local part into it via the Schur complement
  // $D-CA^{-1}B$. Then, we solve for the local part using the skeleton
  // solution. For these two steps, we need the same matrices on the elements
  // twice, which we want to compute by two assembly steps. Since most of the
  // code is similar, we do this with the same function but only switch
  // between the two based on a flag that we set when starting the
  // assembly. Since we need to pass this information on to the local worker
  // routines, we store it once in the task data.
  template <int dim>
  struct ConvdiffHDG<dim>::PerTaskData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_vector;
    std::vector<types::global_dof_index> dof_indices;

    bool trace_reconstruct;

    PerTaskData(const unsigned int n_dofs, const bool trace_reconstruct)
      : cell_matrix(n_dofs, n_dofs)
      , cell_vector(n_dofs)
      , dof_indices(n_dofs)
      , trace_reconstruct(trace_reconstruct)
    {}
  };



  // @sect4{ConvdiffHDG::ScratchData}
  // @p ScratchData contains persistent data for each
  // thread within WorkStream.  The FEValues, matrix,
  // and vector objects should be familiar by now.  There are two objects that
  // need to be discussed: `std::vector<std::vector<unsigned int> >
  // fe_local_support_on_face` and `std::vector<std::vector<unsigned int> >
  // fe_support_on_face`.  These are used to indicate whether or not the finite
  // elements chosen have support (non-zero values) on a given face of the
  // reference cell for the local part associated to @p fe_local and the
  // skeleton part @p fe. We extract this information in the
  // constructor and store it once for all cells that we work on.  Had we not
  // stored this information, we would be forced to assemble a large number of
  // zero terms on each cell, which would significantly slow the program.
  template <int dim>
  struct ConvdiffHDG<dim>::ScratchData
  {
    FEValues<dim>     fe_values_local;
    FEFaceValues<dim> fe_face_values_local;
    FEFaceValues<dim> fe_face_values;

    FullMatrix<double> ll_matrix;
    FullMatrix<double> lf_matrix;
    FullMatrix<double> fl_matrix;
    FullMatrix<double> tmp_matrix;
    Vector<double>     l_rhs;
    Vector<double>     tmp_rhs;

    std::vector<Tensor<1, dim>> q_phi;
    std::vector<double>         q_phi_div;
    std::vector<double>         u_phi;
    std::vector<Tensor<1, dim>> u_phi_grad;
    std::vector<double>         tr_phi;
    std::vector<double>         trace_values;

    std::vector<std::vector<unsigned int>> fe_local_support_on_face;
    std::vector<std::vector<unsigned int>> fe_support_on_face;

    const TensorFunction<1,dim> *const convection_velocity;
    const Function<dim> *const right_hand_side;
    const Function<dim> *const exact_solution;

    ScratchData(const FiniteElement<dim> &fe,
                const FiniteElement<dim> &fe_local,
                const QGauss<dim> &       quadrature_formula,
                const QGauss<dim - 1> &   face_quadrature_formula,
                const UpdateFlags         local_flags,
                const UpdateFlags         local_face_flags,
                const UpdateFlags         flags,
                const TensorFunction<1,dim>* conv_vel,
                const Function<dim>* rhs,
                const Function<dim>* solution_exact)
      : fe_values_local(fe_local, quadrature_formula, local_flags)
      , fe_face_values_local(fe_local,
                             face_quadrature_formula,
                             local_face_flags)
      , fe_face_values(fe, face_quadrature_formula, flags)
      , ll_matrix(fe_local.n_dofs_per_cell(), fe_local.n_dofs_per_cell())
      , lf_matrix(fe_local.n_dofs_per_cell(), fe.n_dofs_per_cell())
      , fl_matrix(fe.n_dofs_per_cell(), fe_local.n_dofs_per_cell())
      , tmp_matrix(fe.n_dofs_per_cell(), fe_local.n_dofs_per_cell())
      , l_rhs(fe_local.n_dofs_per_cell())
      , tmp_rhs(fe_local.n_dofs_per_cell())
      , q_phi(fe_local.n_dofs_per_cell())
      , q_phi_div(fe_local.n_dofs_per_cell())
      , u_phi(fe_local.n_dofs_per_cell())
      , u_phi_grad(fe_local.n_dofs_per_cell())
      , tr_phi(fe.n_dofs_per_cell())
      , trace_values(face_quadrature_formula.size())
      , fe_local_support_on_face(GeometryInfo<dim>::faces_per_cell)
      , fe_support_on_face(GeometryInfo<dim>::faces_per_cell)
      , convection_velocity{conv_vel}
      , right_hand_side{rhs}
      , exact_solution{solution_exact}
    {
      for (const unsigned int face_no : GeometryInfo<dim>::face_indices())
        for (unsigned int i = 0; i < fe_local.n_dofs_per_cell(); ++i)
          {
            if (fe_local.has_support_on_face(i, face_no))
              fe_local_support_on_face[face_no].push_back(i);
          }

      for (const unsigned int face_no : GeometryInfo<dim>::face_indices())
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          {
            if (fe.has_support_on_face(i, face_no))
              fe_support_on_face[face_no].push_back(i);
          }
    }

    ScratchData(const ScratchData &sd)
      : fe_values_local(sd.fe_values_local.get_fe(),
                        sd.fe_values_local.get_quadrature(),
                        sd.fe_values_local.get_update_flags())
      , fe_face_values_local(sd.fe_face_values_local.get_fe(),
                             sd.fe_face_values_local.get_quadrature(),
                             sd.fe_face_values_local.get_update_flags())
      , fe_face_values(sd.fe_face_values.get_fe(),
                       sd.fe_face_values.get_quadrature(),
                       sd.fe_face_values.get_update_flags())
      , ll_matrix(sd.ll_matrix)
      , lf_matrix(sd.lf_matrix)
      , fl_matrix(sd.fl_matrix)
      , tmp_matrix(sd.tmp_matrix)
      , l_rhs(sd.l_rhs)
      , tmp_rhs(sd.tmp_rhs)
      , q_phi(sd.q_phi)
      , q_phi_div(sd.q_phi_div)
      , u_phi(sd.u_phi)
      , u_phi_grad(sd.u_phi_grad)
      , tr_phi(sd.tr_phi)
      , trace_values(sd.trace_values)
      , fe_local_support_on_face(sd.fe_local_support_on_face)
      , fe_support_on_face(sd.fe_support_on_face)
      , convection_velocity{sd.convection_velocity}
      , right_hand_side{sd.right_hand_side}
      , exact_solution{sd.exact_solution}
    {}
  };



  // @sect4{ConvdiffHDG::PostProcessScratchData}
  // @p PostProcessScratchData contains the data used by WorkStream
  // when post-processing the local solution $u^*$.  It is similar, but much
  // simpler, than @p ScratchData.
  template <int dim>
  struct ConvdiffHDG<dim>::PostProcessScratchData
  {
    FEValues<dim> fe_values_local;
    FEValues<dim> fe_values;

    std::vector<double>         u_values;
    std::vector<Tensor<1, dim>> u_gradients;
    FullMatrix<double>          cell_matrix;

    Vector<double> cell_rhs;
    Vector<double> cell_sol;

    PostProcessScratchData(const FiniteElement<dim> &fe,
                           const FiniteElement<dim> &fe_local,
                           const QGauss<dim> &       quadrature_formula,
                           const UpdateFlags         local_flags,
                           const UpdateFlags         flags)
      : fe_values_local(fe_local, quadrature_formula, local_flags)
      , fe_values(fe, quadrature_formula, flags)
      , u_values(quadrature_formula.size())
      , u_gradients(quadrature_formula.size())
      , cell_matrix(fe.n_dofs_per_cell(), fe.n_dofs_per_cell())
      , cell_rhs(fe.n_dofs_per_cell())
      , cell_sol(fe.n_dofs_per_cell())
    {}

    PostProcessScratchData(const PostProcessScratchData &sd)
      : fe_values_local(sd.fe_values_local.get_fe(),
                        sd.fe_values_local.get_quadrature(),
                        sd.fe_values_local.get_update_flags())
      , fe_values(sd.fe_values.get_fe(),
                  sd.fe_values.get_quadrature(),
                  sd.fe_values.get_update_flags())
      , u_values(sd.u_values)
      , u_gradients(sd.u_gradients)
      , cell_matrix(sd.cell_matrix)
      , cell_rhs(sd.cell_rhs)
      , cell_sol(sd.cell_sol)
    {}
  };



  // @sect4{ConvdiffHDG::assemble_system}
  // The @p assemble_system function is similar to the one on Step-32, where
  // the quadrature formula and the update flags are set up, and then
  // <code>WorkStream</code> is used to do the work in a multi-threaded
  // manner.  The @p trace_reconstruct input parameter is used to decide
  // whether we are solving for the global skeleton solution (false) or the
  // local solution (true).
  //
  // One thing worth noting for the multi-threaded execution of assembly is
  // the fact that the local computations in `assemble_system_one_cell()` call
  // into BLAS and LAPACK functions if those are available in deal.II. Thus,
  // the underlying BLAS/LAPACK library must support calls from multiple
  // threads at the same time. Most implementations do support this, but some
  // libraries need to be built in a specific way to avoid problems. For
  // example, OpenBLAS compiled without multithreading inside the BLAS/LAPACK
  // calls needs to built with a flag called `USE_LOCKING` set to true.
  template <int dim>
  void ConvdiffHDG<dim>
  ::assemble_system(const bool trace_reconstruct)
  {
    const QGauss<dim>     quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    const UpdateFlags local_flags(update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points);

    const UpdateFlags local_face_flags(update_values);

    const UpdateFlags flags(update_values | update_normal_vectors |
                            update_quadrature_points | update_JxW_values);

    PerTaskData task_data(fe.n_dofs_per_cell(), trace_reconstruct);
    auto ccase = std::dynamic_pointer_cast<const convdiffcase_verification<dim>>(this->tcase_);
    ScratchData scratch(fe,
                        fe_local,
                        quadrature_formula,
                        face_quadrature_formula,
                        local_flags,
                        local_face_flags,
                        flags,
                        ccase->get_convection_velocity().get(),
                        ccase->get_right_hand_side().get(),
                        ccase->get_exact_solution().get());

    WorkStream::run(dof_handler.begin_active(),
        dof_handler.end(),
        *this,
        &ConvdiffHDG<dim>::assemble_system_one_cell,
        &ConvdiffHDG<dim>::copy_local_to_global,
        scratch, task_data);
  }



  // @sect4{ConvdiffHDG::assemble_system_one_cell}
  // The real work of the ConvdiffHDG program is done by @p assemble_system_one_cell.
  // Assembling the local matrices $A, B, C$ is done here, along with the
  // local contributions of the global matrix $D$.
  template <int dim>
  void ConvdiffHDG<dim>::assemble_system_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData &                                         scratch,
    PerTaskData &                                         task_data)
  {
    // Construct iterator for dof_handler_local for FEValues reinit function.
    const typename DoFHandler<dim>::active_cell_iterator loc_cell =
      cell->as_dof_handler_iterator(dof_handler_local);

    const unsigned int n_q_points =
      scratch.fe_values_local.get_quadrature().size();
    const unsigned int n_face_q_points =
      scratch.fe_face_values_local.get_quadrature().size();

    const unsigned int loc_dofs_per_cell =
      scratch.fe_values_local.get_fe().n_dofs_per_cell();

    const FEValuesExtractors::Vector fluxes(0);
    const FEValuesExtractors::Scalar scalar(dim);

    scratch.ll_matrix = 0;
    scratch.l_rhs     = 0;
    if (!task_data.trace_reconstruct)
      {
        scratch.lf_matrix     = 0;
        scratch.fl_matrix     = 0;
        task_data.cell_matrix = 0;
        task_data.cell_vector = 0;
      }
    scratch.fe_values_local.reinit(loc_cell);
    
    auto ccase = std::dynamic_pointer_cast<const CaseWithNeumannBC<Case<dim>>>(this->tcase_);
    if(!ccase) {
        throw std::runtime_error("Invalid case for Neumann BCs!");
    }

    // We first compute the cell-interior contribution to @p ll_matrix matrix
    // (referred to as matrix $A$ in the introduction) corresponding to
    // local-local coupling, as well as the local right-hand-side vector.  We
    // store the values at each quadrature point for the basis functions, the
    // right-hand-side value, and the convection velocity, in order to have
    // quick access to these fields.
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double rhs_value = scratch.right_hand_side->value(
          scratch.fe_values_local.quadrature_point(q));
        const Tensor<1, dim> convection = scratch.convection_velocity->value(
          scratch.fe_values_local.quadrature_point(q));
        const double JxW = scratch.fe_values_local.JxW(q);
        for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
          {
            scratch.q_phi[k] = scratch.fe_values_local[fluxes].value(k, q);
            scratch.q_phi_div[k] =
              scratch.fe_values_local[fluxes].divergence(k, q);
            scratch.u_phi[k] = scratch.fe_values_local[scalar].value(k, q);
            scratch.u_phi_grad[k] =
              scratch.fe_values_local[scalar].gradient(k, q);
          }
        for (unsigned int i = 0; i < loc_dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < loc_dofs_per_cell; ++j)
              scratch.ll_matrix(i, j) +=
                (scratch.q_phi[i] * scratch.q_phi[j] -
                 scratch.q_phi_div[i] * scratch.u_phi[j] +
                 scratch.u_phi[i] * scratch.q_phi_div[j] -
                 (scratch.u_phi_grad[i] * convection) * scratch.u_phi[j]) *
                JxW;
            scratch.l_rhs(i) += scratch.u_phi[i] * rhs_value * JxW;
          }
      }

    // Face terms are assembled on all faces of all elements. This is in
    // contrast to more traditional DG methods, where each face is only visited
    // once in the assembly procedure.
    for (const auto face_no : cell->face_indices())
      {
        scratch.fe_face_values_local.reinit(loc_cell, face_no);
        scratch.fe_face_values.reinit(cell, face_no);

        // The already obtained $\hat{u}$ values are needed when solving for the
        // local variables.
        if (task_data.trace_reconstruct)
          scratch.fe_face_values.get_function_values(solution,
                                                     scratch.trace_values);

        for (unsigned int q = 0; q < n_face_q_points; ++q)
          {
            const double     JxW = scratch.fe_face_values.JxW(q);
            const Point<dim> quadrature_point =
              scratch.fe_face_values.quadrature_point(q);
            const Tensor<1, dim> normal =
              scratch.fe_face_values.normal_vector(q);
            const Tensor<1, dim> convection =
              scratch.convection_velocity->value(quadrature_point);

            // Here we compute the stabilization parameter discussed in the
            // introduction: since the diffusion is one and the diffusion
            // length scale is set to 1/5, it simply results in a contribution
            // of 5 for the diffusion part and the magnitude of convection
            // through the element boundary in a centered scheme for the
            // convection part.
            const double tau_stab = (5. + std::abs(convection * normal));

            // We store the non-zero flux and scalar values, making use of the
            // support_on_face information we created in @p ScratchData.
            for (unsigned int k = 0;
                 k < scratch.fe_local_support_on_face[face_no].size();
                 ++k)
              {
                const unsigned int kk =
                  scratch.fe_local_support_on_face[face_no][k];
                scratch.q_phi[k] =
                  scratch.fe_face_values_local[fluxes].value(kk, q);
                scratch.u_phi[k] =
                  scratch.fe_face_values_local[scalar].value(kk, q);
              }

            // When @p trace_reconstruct=false, we are preparing to assemble the
            // system for the skeleton variable $\hat{u}$. If this is the case,
            // we must assemble all local matrices associated with the problem:
            // local-local, local-face, face-local, and face-face.  The
            // face-face matrix is stored as @p TaskData::cell_matrix, so that
            // it can be assembled into the global system by @p
            // copy_local_to_global.
            if (!task_data.trace_reconstruct)
              {
                for (unsigned int k = 0;
                     k < scratch.fe_support_on_face[face_no].size();
                     ++k)
                  scratch.tr_phi[k] = scratch.fe_face_values.shape_value(
                    scratch.fe_support_on_face[face_no][k], q);
                for (unsigned int i = 0;
                     i < scratch.fe_local_support_on_face[face_no].size();
                     ++i)
                  for (unsigned int j = 0;
                       j < scratch.fe_support_on_face[face_no].size();
                       ++j)
                    {
                      const unsigned int ii =
                        scratch.fe_local_support_on_face[face_no][i];
                      const unsigned int jj =
                        scratch.fe_support_on_face[face_no][j];
                      scratch.lf_matrix(ii, jj) +=
                        ((scratch.q_phi[i] * normal +
                          (convection * normal - tau_stab) * scratch.u_phi[i]) *
                         scratch.tr_phi[j]) *
                        JxW;

                      // Note the sign of the face_no-local matrix.  We negate
                      // the sign during assembly here so that we can use the
                      // FullMatrix::mmult with addition when computing the
                      // Schur complement.
                      scratch.fl_matrix(jj, ii) -=
                        ((scratch.q_phi[i] * normal +
                          tau_stab * scratch.u_phi[i]) *
                         scratch.tr_phi[j]) *
                        JxW;
                    }

                for (unsigned int i = 0;
                     i < scratch.fe_support_on_face[face_no].size();
                     ++i)
                  for (unsigned int j = 0;
                       j < scratch.fe_support_on_face[face_no].size();
                       ++j)
                    {
                      const unsigned int ii =
                        scratch.fe_support_on_face[face_no][i];
                      const unsigned int jj =
                        scratch.fe_support_on_face[face_no][j];
                      task_data.cell_matrix(ii, jj) +=
                        ((convection * normal - tau_stab) * scratch.tr_phi[i] *
                         scratch.tr_phi[j]) *
                        JxW;
                    }

                if (cell->face(face_no)->at_boundary() &&
                    (cell->face(face_no)->boundary_id() == ccase->get_neumann_marker()))
                  {
                    const double neumann_value =
                      ccase->get_neumann_bc()->value_normal(quadrature_point, normal);
                    for (unsigned int i = 0;
                         i < scratch.fe_support_on_face[face_no].size();
                         ++i)
                      {
                        const unsigned int ii =
                          scratch.fe_support_on_face[face_no][i];
                        task_data.cell_vector(ii) +=
                          scratch.tr_phi[i] * neumann_value * JxW;
                      }
                  }
              }

            // This last term adds the contribution of the term $\left<w,\tau
            // u_h\right>_{\partial \mathcal T}$ to the local matrix. As opposed
            // to the face matrices above, we need it in both assembly stages.
            for (unsigned int i = 0;
                 i < scratch.fe_local_support_on_face[face_no].size();
                 ++i)
              for (unsigned int j = 0;
                   j < scratch.fe_local_support_on_face[face_no].size();
                   ++j)
                {
                  const unsigned int ii =
                    scratch.fe_local_support_on_face[face_no][i];
                  const unsigned int jj =
                    scratch.fe_local_support_on_face[face_no][j];
                  scratch.ll_matrix(ii, jj) +=
                    tau_stab * scratch.u_phi[i] * scratch.u_phi[j] * JxW;
                }

            // When @p trace_reconstruct=true, we are solving for the local
            // solutions on an element by element basis.  The local
            // right-hand-side is calculated by replacing the basis functions @p
            // tr_phi in the @p lf_matrix computation by the computed values @p
            // trace_values.  Of course, the sign of the matrix is now minus
            // since we have moved everything to the other side of the equation.
            if (task_data.trace_reconstruct)
              for (unsigned int i = 0;
                   i < scratch.fe_local_support_on_face[face_no].size();
                   ++i)
                {
                  const unsigned int ii =
                    scratch.fe_local_support_on_face[face_no][i];
                  scratch.l_rhs(ii) -=
                    (scratch.q_phi[i] * normal +
                     scratch.u_phi[i] * (convection * normal - tau_stab)) *
                    scratch.trace_values[q] * JxW;
                }
          }
      }

    // Once assembly of all of the local contributions is complete, we must
    // either: (1) assemble the global system, or (2) compute the local solution
    // values and save them. In either case, the first step is to invert the
    // local-local matrix.
    scratch.ll_matrix.gauss_jordan();

    // For (1), we compute the Schur complement and add it to the @p
    // cell_matrix, matrix $D$ in the introduction.
    if (task_data.trace_reconstruct == false)
      {
        scratch.fl_matrix.mmult(scratch.tmp_matrix, scratch.ll_matrix);
        scratch.tmp_matrix.vmult_add(task_data.cell_vector, scratch.l_rhs);
        scratch.tmp_matrix.mmult(task_data.cell_matrix,
                                 scratch.lf_matrix,
                                 true);
        cell->get_dof_indices(task_data.dof_indices);
      }
    // For (2), we are simply solving (ll_matrix).(solution_local) = (l_rhs).
    // Hence, we multiply @p l_rhs by our already inverted local-local matrix
    // and store the result using the <code>set_dof_values</code> function.
    else
      {
        scratch.ll_matrix.vmult(scratch.tmp_rhs, scratch.l_rhs);
        loc_cell->set_dof_values(scratch.tmp_rhs, solution_local);
      }
  }



  // @sect4{ConvdiffHDG::copy_local_to_global}
  // If we are in the first step of the solution, i.e. @p trace_reconstruct=false,
  // then we assemble the local matrices into the global system.
  template <int dim>
  void ConvdiffHDG<dim>::copy_local_to_global(const PerTaskData &data)
  {
    if (data.trace_reconstruct == false)
      constraints.distribute_local_to_global(data.cell_matrix,
                                             data.cell_vector,
                                             data.dof_indices,
                                             system_matrix,
                                             system_rhs);
  }


  template <int dim>
  void ConvdiffHDG<dim>::solve()
  {
    SolverControl                  solver_control(system_matrix.m() * 10,
                                 1e-11 * system_rhs.l2_norm());
    PreconditionSOR<SparseMatrix<double> > precondition;
    precondition.initialize(system_matrix,
            PreconditionSOR<SparseMatrix<double>>::AdditionalData(.8));
    SolverBicgstab<Vector<double>> solver(solver_control);
    //solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    solver.solve(system_matrix, solution, system_rhs, precondition);

    std::cout << "   Number of BiCGStab iterations: "
              << solver_control.last_step() << std::endl;

    system_matrix.clear();
    //sparsity_pattern.reinit(0, 0, 0, 1);
    sparsity_pattern.reinit(0, 0, 0);

    constraints.distribute(solution);

    // Once we have solved for the skeleton solution,
    // we can solve for the local solutions in an element-by-element
    // fashion.  We do this by re-using the same @p assemble_system function
    // but switching @p trace_reconstruct to true.
    assemble_system(true);
  }



  // @sect4{ConvdiffHDG::postprocess}

  // The postprocess method serves two purposes. First, we want to construct a
  // post-processed scalar variables in the element space of degree $p+1$ that
  // we hope will converge at order $p+2$. This is again an element-by-element
  // process and only involves the scalar solution as well as the gradient on
  // the local cell. To do this, we introduce the already defined scratch data
  // together with some update flags and run the work stream to do this in
  // parallel.
  //
  // Secondly, we want to compute discretization errors just as we did in
  // step-7. The overall procedure is similar with calls to
  // VectorTools::integrate_difference. The difference is in how we compute
  // the errors for the scalar variable and the gradient variable. In step-7,
  // we did this by computing @p L2_norm or @p H1_seminorm
  // contributions. Here, we have a DoFHandler with these two contributions
  // computed and sorted by their vector component, <code>[0, dim)</code> for
  // the
  // gradient and @p dim for the scalar. To compute their value, we hence use
  // a ComponentSelectFunction with either of them, together with the @p
  // SolutionAndGradient class introduced above that contains the analytic
  // parts of either of them. Eventually, we also compute the L2-error of the
  // post-processed solution and add the results into the convergence table.
  template <int dim>
  void ConvdiffHDG<dim>::postprocess()
  {
    {
      const QGauss<dim> quadrature_formula(fe_u_post.degree + 1);
      const UpdateFlags local_flags(update_values);
      const UpdateFlags flags(update_values | update_gradients |
                              update_JxW_values);

      PostProcessScratchData scratch(
        fe_u_post, fe_local, quadrature_formula, local_flags, flags);

      WorkStream::run(
        dof_handler_u_post.begin_active(),
        dof_handler_u_post.end(),
        [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
               PostProcessScratchData &                              scratch,
               unsigned int &                                        data) {
          this->postprocess_one_cell(cell, scratch, data);
        },
        std::function<void(const unsigned int &)>(),
        scratch,
        0U);
    }

    Vector<float> difference_per_cell(triangulation.n_active_cells());

    auto ccase = std::dynamic_pointer_cast<const convdiffcase_verification<dim>>(this->tcase_);
    if(!ccase) {
        throw std::runtime_error("Invalid case for HDG convergence analysis!");
    }

    ComponentSelectFunction<dim> value_select(dim, dim + 1);
    VectorTools::integrate_difference(dof_handler_local,
                                      solution_local,
                                      *ccase->get_exact_solution_and_gradient(),
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm,
                                      &value_select);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    ComponentSelectFunction<dim> gradient_select(
      std::pair<unsigned int, unsigned int>(0, dim), dim + 1);
    VectorTools::integrate_difference(dof_handler_local,
                                      solution_local,
                                      *ccase->get_exact_solution_and_gradient(),
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm,
                                      &gradient_select);
    const double grad_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler_u_post,
                                      solution_u_post,
                                      *ccase->get_exact_solution(),
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 3),
                                      VectorTools::L2_norm);
    const double post_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    convergence_table.add_value("cells", triangulation.n_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());

    convergence_table.add_value("val L2", L2_error);
    convergence_table.set_scientific("val L2", true);
    convergence_table.set_precision("val L2", 3);

    convergence_table.add_value("grad L2", grad_error);
    convergence_table.set_scientific("grad L2", true);
    convergence_table.set_precision("grad L2", 3);

    convergence_table.add_value("val L2-post", post_error);
    convergence_table.set_scientific("val L2-post", true);
    convergence_table.set_precision("val L2-post", 3);
  }



  // @sect4{ConvdiffHDG::postprocess_one_cell}
  //
  // This is the actual work done for the postprocessing. According to the
  // discussion in the introduction, we need to set up a system that projects
  // the gradient part of the DG solution onto the gradient of the
  // post-processed variable. Moreover, we need to set the average of the new
  // post-processed variable to equal the average of the scalar DG solution
  // on the cell.
  //
  // More technically speaking, the projection of the gradient is a system
  // that would potentially fills our @p dofs_per_cell times @p dofs_per_cell
  // matrix but is singular (the sum of all rows would be zero because the
  // constant function has zero gradient). Therefore, we take one row away and
  // use it for imposing the average of the scalar value. We pick the first
  // row for the scalar part, even though we could pick any row for $\mathcal
  // Q_{-p}$ elements. However, had we used FE_DGP elements instead, the first
  // row would correspond to the constant part already and deleting e.g. the
  // last row would give us a singular system. This way, our program can also
  // be used for those elements.
  template <int dim>
  void ConvdiffHDG<dim>::postprocess_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    PostProcessScratchData &                              scratch,
    unsigned int &)
  {
    const typename DoFHandler<dim>::active_cell_iterator loc_cell =
      cell->as_dof_handler_iterator(dof_handler_local);

    scratch.fe_values_local.reinit(loc_cell);
    scratch.fe_values.reinit(cell);

    const FEValuesExtractors::Vector fluxes(0);
    const FEValuesExtractors::Scalar scalar(dim);

    const unsigned int n_q_points = scratch.fe_values.get_quadrature().size();
    const unsigned int dofs_per_cell = scratch.fe_values.dofs_per_cell;

    scratch.fe_values_local[scalar].get_function_values(solution_local,
                                                        scratch.u_values);
    scratch.fe_values_local[fluxes].get_function_values(solution_local,
                                                        scratch.u_gradients);

    double sum = 0;
    for (unsigned int i = 1; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            sum = 0;
            for (unsigned int q = 0; q < n_q_points; ++q)
              sum += (scratch.fe_values.shape_grad(i, q) *
                      scratch.fe_values.shape_grad(j, q)) *
                     scratch.fe_values.JxW(q);
            scratch.cell_matrix(i, j) = sum;
          }

        sum = 0;
        for (unsigned int q = 0; q < n_q_points; ++q)
          sum -= (scratch.fe_values.shape_grad(i, q) * scratch.u_gradients[q]) *
                 scratch.fe_values.JxW(q);
        scratch.cell_rhs(i) = sum;
      }
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        sum = 0;
        for (unsigned int q = 0; q < n_q_points; ++q)
          sum += scratch.fe_values.shape_value(j, q) * scratch.fe_values.JxW(q);
        scratch.cell_matrix(0, j) = sum;
      }
    {
      sum = 0;
      for (unsigned int q = 0; q < n_q_points; ++q)
        sum += scratch.u_values[q] * scratch.fe_values.JxW(q);
      scratch.cell_rhs(0) = sum;
    }

    // Having assembled all terms, we can again go on and solve the linear
    // system. We invert the matrix and then multiply the inverse by the
    // right hand side. An alternative (and more numerically stable) method
    // would have been to only factorize the matrix and apply the factorization.
    scratch.cell_matrix.gauss_jordan();
    scratch.cell_matrix.vmult(scratch.cell_sol, scratch.cell_rhs);
    cell->distribute_local_to_global(scratch.cell_sol, solution_u_post);
  }



  // @sect4{ConvdiffHDG::output_results}
  // We have 3 sets of results that we would like to output:  the local
  // solution, the post-processed local solution, and the skeleton solution. The
  // former 2 both 'live' on element volumes, whereas the latter lives on
  // codimension-1 surfaces
  // of the triangulation.  Our @p output_results function writes all local solutions
  // to the same vtk file, even though they correspond to different
  // DoFHandler objects.  The graphical output for the skeleton
  // variable is done through use of the DataOutFaces class.
  template <int dim>
  void ConvdiffHDG<dim>::output_results(const int cycle)
  {
    std::string filename;
    switch (refinement_mode_)
      {
          case MeshRefineMode::global_refinement:
          filename = "solution-global";
          break;
          case MeshRefineMode::adaptive_refinement:
          filename = "solution-adaptive";
          break;
        default:
          Assert(false, ExcNotImplemented());
      }

    std::string face_out(filename);
    face_out += "-face";
    std::string boundary_out(filename);
    boundary_out += "-boundary";

    filename += "-q" + Utilities::int_to_string(fe.degree, 1);
    filename += "-" + Utilities::int_to_string(cycle, 2);
    filename += ".vtk";
    std::ofstream output(filename);

    DataOut<dim> data_out;

    // We first define the names and types of the local solution,
    // and add the data to @p data_out.
    std::vector<std::string> names(dim, "gradient");
    names.emplace_back("solution");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        dim + 1, DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim] =
      DataComponentInterpretation::component_is_scalar;
    data_out.add_data_vector(dof_handler_local,
                             solution_local,
                             names,
                             component_interpretation);

    // The second data item we add is the post-processed solution.
    // In this case, it is a single scalar variable belonging to
    // a different DoFHandler.
    std::vector<std::string> post_name(1, "u_post");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      post_comp_type(1, DataComponentInterpretation::component_is_scalar);
    data_out.add_data_vector(dof_handler_u_post,
                             solution_u_post,
                             post_name,
                             post_comp_type);

    data_out.build_patches(fe.degree);
    data_out.write_vtk(output);
    output.close();

    face_out += "-q" + Utilities::int_to_string(fe.degree, 1);
    face_out += "-" + Utilities::int_to_string(cycle, 2);
    face_out += ".vtk";
    std::ofstream face_output(face_out);

    // The <code>DataOutFaces</code> class works analogously to the
    // <code>DataOut</code> class when we have a <code>DoFHandler</code> that
    // defines the solution on the skeleton of the triangulation.  We treat it
    // as such here, and the code is similar to that above.
    DataOutFaces<dim>        data_out_face(false);
    std::vector<std::string> face_name(1, "u_hat");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      face_component_type(1, DataComponentInterpretation::component_is_scalar);

    data_out_face.add_data_vector(dof_handler,
                                  solution,
                                  face_name,
                                  face_component_type);

    data_out_face.build_patches(fe.degree);
    data_out_face.write_vtk(face_output);
    face_output.close();
   
    // Write boundary data 
    boundary_out += "-q" + Utilities::int_to_string(fe.degree, 1);
    boundary_out += "-" + Utilities::int_to_string(cycle, 2);
    boundary_out += ".vtk";
    std::ofstream b_output(boundary_out);
    DataOutFaces<dim> data_out_boundary(true);
    face_name[0] = "u_post";
    data_out_boundary.add_data_vector(dof_handler_u_post,
                                  solution_u_post,
                                  face_name,
                                  face_component_type);

    data_out_boundary.build_patches(fe.degree);
    data_out_boundary.write_vtk(b_output);
    b_output.close();
  }

  // @sect4{ConvdiffHDG::refine_grid}

  // We implement two different refinement cases for HDG, just as in
  // <code>Step-7</code>: adaptive_refinement and global_refinement.  The
  // global_refinement option recreates the entire triangulation every
  // time. This is because we want to use a finer sequence of meshes than what
  // we would get with one refinement step, namely 2, 3, 4, 6, 8, 12, 16, ...
  // elements per direction.

  // The adaptive_refinement mode uses the <code>KellyErrorEstimator</code> to
  // give a decent indication of the non-regular regions in the scalar local
  // solutions.
  template <int dim>
  void ConvdiffHDG<dim>::refine_grid(const int cycle, const unsigned int initial_resolution)
  {
    if (cycle == 0)
      {
        this->tcase_->get_geometry()->generate_grid(triangulation, initial_resolution);
        triangulation.refine_global(3 - dim);
      }
    else
      switch (refinement_mode_)
        {
            case MeshRefineMode::global_refinement:
            {
              triangulation.clear();
              const auto resolution = initial_resolution + cycle % 2;
              this->tcase_->get_geometry()->generate_grid(triangulation, resolution);
              triangulation.refine_global(3 - dim + cycle / 2);
              break;
            }

            case MeshRefineMode::adaptive_refinement:
            {
              Vector<float> estimated_error_per_cell(
                triangulation.n_active_cells());

              const FEValuesExtractors::Scalar scalar(dim);
              std::map<types::boundary_id, const Function<dim> *>
                neumann_boundary;
              KellyErrorEstimator<dim>::estimate(dof_handler_local,
                                                 QGauss<dim - 1>(fe.degree + 1),
                                                 neumann_boundary,
                                                 solution_local,
                                                 estimated_error_per_cell,
                                                 fe_local.component_mask(
                                                   scalar));

              GridRefinement::refine_and_coarsen_fixed_number(
                triangulation, estimated_error_per_cell, 0.3, 0.);

              triangulation.execute_coarsening_and_refinement();

              break;
            }

          default:
            {
              Assert(false, ExcNotImplemented());
            }
        }

    // Just as in step-7, we set the boundary indicator of two of the faces to 1
    // where we want to specify Neumann boundary conditions instead of Dirichlet
    // conditions. Since we re-create the triangulation every time for global
    // refinement, the flags are set in every refinement step, not just at the
    // beginning.
    //for (const auto &cell : triangulation.cell_iterators())
    //  for (const auto &face : cell->face_iterators())
    //    if (face->at_boundary())
    //      if ((std::fabs(face->center()(0) - (-1)) < 1e-12) ||
    //          (std::fabs(face->center()(1) - (-1)) < 1e-12))
    //        face->set_boundary_id(1);
    this->tcase_->get_geometry()->set_boundary_ids(triangulation);
  }

  // @sect4{ConvdiffHDG::run}
  // The functionality here is basically the same as <code>Step-7</code>.
  // We loop over 10 cycles, refining the grid on each one.  At the end,
  // convergence tables are created.
  template <int dim>
  void ConvdiffHDG<dim>::run()
  {
    for (int cycle = 0; cycle < num_cycles_; ++cycle)
      {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        refine_grid(cycle, this->init_res_);
        setup_system();
        assemble_system(false);
        solve();
        postprocess();
        output_results(cycle);
      }

    // There is one minor change for the convergence table compared to step-7:
    // Since we did not refine our mesh by a factor two in each cycle (but
    // rather used the sequence 2, 3, 4, 6, 8, 12, ...), we need to tell the
    // convergence rate evaluation about this. We do this by setting the
    // number of cells as a reference column and additionally specifying the
    // dimension of the problem, which gives the necessary information for the
    // relation between number of cells and mesh size.
    if (refinement_mode_ == MeshRefineMode::global_refinement)
      {
        convergence_table.evaluate_convergence_rates(
          "val L2", "cells", ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates(
          "grad L2", "cells", ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates(
          "val L2-post", "cells", ConvergenceTable::reduction_rate_log2, dim);
      }
    convergence_table.write_text(std::cout);
  }

  template class ConvdiffHDG<2>;

}
}