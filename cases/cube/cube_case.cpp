#include "cube_case.hpp"

namespace paramsim {
namespace cases {
namespace cube {

template <int dim>
void CubeCase<dim>::initialize_dirichlet_everywhere(
    std::shared_ptr<dealii::Function<dim>> profile)
{
    this->bc_dirichlet_.push_back(dirichlet_bc<dim>{1, profile});

    std::vector<typename DomainGeometry<dim>::bc_mark_desc> bcmarks;
    bcmarks.push_back(std::make_pair(
        this->bc_dirichlet_[0].bc_id,
        [](const dealii::Point<dim>&) { return true; }));
    this->geom_ = std::make_shared<geom::Cube<dim>>(bcmarks);
}

template class CubeCase<2>;
template class CubeCase<3>;

} // namespace cube
} // namespace cases
} // namespace paramsim
