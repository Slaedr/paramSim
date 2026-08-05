#include "verify.hpp"

namespace paramsim {
namespace cases {

template <int dim>
void GelfandVerify<dim>::initialize(const bpo::variables_map&)
{
    this->exact_soln_ = std::make_shared<gelfand_verify::Solution<dim>>();
    this->rhs_ = std::make_shared<gelfand_verify::RightHandSide<dim>>();
    this->bc_dirichlet_.push_back(dirichlet_bc<dim>{1, this->exact_soln_});

    std::vector<typename DomainGeometry<dim>::bc_mark_desc> boundary_markers;
    boundary_markers.emplace_back(
        this->bc_dirichlet_[0].bc_id,
        [](const dealii::Point<dim>&) { return true; });
    this->geom_ = std::make_shared<geom::Cube<dim>>(boundary_markers);
}

template <int dim>
void GelfandVerify<dim>::add_case_cmd_args(bpo::options_description&) const
{
}

template class GelfandVerify<2>;
template class GelfandVerify<3>;

} // namespace cases
} // namespace paramsim
