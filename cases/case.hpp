#ifndef PARAMSIM_CASES_HPP_
#define PARAMSIM_CASES_HPP_

#include <vector>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <deal.II/base/function.h>

#include "../geometrybase.hpp"

namespace paramsim {

namespace bpo = boost::program_options;

template <int dim>
struct dirichlet_bc {
    bc_id_t bc_id;
    std::shared_ptr<const dealii::Function<dim>> bc_fn;
};

/// Abstract class gathering all testcase-specific data and descriptions
template <int dimension>
class Case
{
public:
    static constexpr int dim = dimension;

    virtual void add_case_cmd_args(bpo::options_description&) const = 0;

    virtual void initialize(const bpo::variables_map&) = 0;

    std::shared_ptr<const DomainGeometry<dim>> get_geometry() const {
        return geom_;
    }

    std::shared_ptr<const dealii::Function<dim>> get_right_hand_side() const {
        return rhs_;
    }

    /** Get dirichlet boundary conditions.
     */
    const std::vector<dirichlet_bc<dim>>& get_dirichlet_bcs() const {
        return bc_dirichlet_;
    }

protected:
    std::shared_ptr<const DomainGeometry<dim>> geom_;
    std::shared_ptr<dealii::Function<dim>> rhs_;
    std::vector<dirichlet_bc<dim>> bc_dirichlet_;
};

/** Generates a case given a string description.
 */
template <int dim>
std::unique_ptr<Case<dim>> create_case(const std::string case_str);

/* Are mixins really the right approach to use here (below)?
 * Dynamic-casting a Case<dim>* to a more specialized type is essentially impossible
 * without knowing the derivation order for the concrete type.
 */

template <typename Base>
class CaseWithNeumannBC : public Base
{
public:
    static constexpr int dim = Base::dim;

    std::shared_ptr<const FaceFunction<dim>> get_neumann_bc() const {
        return neumann_;
    }

    dealii::types::boundary_id get_neumann_marker() const { return bcid_neumann_; }

protected:
    std::shared_ptr<FaceFunction<dim>> neumann_;
    dealii::types::boundary_id bcid_neumann_;
};

template <typename Base>
class CaseWithExactSolution : public Base
{
public:
    static constexpr int dim = Base::dim;

    std::shared_ptr<const dealii::Function<dim>> get_exact_solution() const {
        return exact_soln_;
    }
protected:
    std::shared_ptr<dealii::Function<dim>> exact_soln_;
};

template <typename Base>
class CaseWithExactSolutionAndGradient : public Base
{
public:
    static constexpr int dim = Base::dim;

    std::shared_ptr<const dealii::Function<dim>> get_exact_solution_and_gradient() const {
        return exact_soln_and_grad_;
    }
protected:
    std::shared_ptr<dealii::Function<dim>> exact_soln_and_grad_;
};


}

#endif
