#include "verify.hpp"

namespace paramsim {
namespace cases {

using namespace dealii;

template <int dim>
void MinSurfBallVerify<dim>::initialize(const bpo::variables_map& params)
{
    std::shared_ptr<Function<dim>> dirichlet1;
    dirichlet1 = std::make_shared<minsurf_verify::Solution<dim>>();
    std::cout << "Case 'verify' for Poisson: default parameters.\n";

    this->rhs_ = std::make_shared<minsurf_verify::RightHandSide<dim>>();
    this->exact_soln_ = std::make_shared<minsurf_verify::Solution<dim>>();
    this->bc_dirichlet_.push_back(dirichlet_bc<dim>{1, dirichlet1});

    std::vector<typename DomainGeometry<dim>::bc_mark_desc> bcmarks;
    bcmarks.push_back(std::make_pair(this->bc_dirichlet_[0].bc_id,
        [](const dealii::Point<dim>&) { return true; }));
    this->geom_ = std::make_shared<geom::Ball<dim>>(bcmarks);
}

template <int dim>
void MinSurfBallVerify<dim>::add_case_cmd_args(bpo::options_description&) const
{
}

template class MinSurfBallVerify<2>;
template class MinSurfBallVerify<3>;

template <int dim>
void MinSurfCubeVerify<dim>::initialize(const bpo::variables_map& params)
{
    std::shared_ptr<Function<dim>> dirichlet1;
    dirichlet1 = std::make_shared<minsurf_verify::Solution<dim>>();
    std::cout << "Case 'verify_cube' for Poisson: default parameters.\n";

    this->rhs_ = std::make_shared<minsurf_verify::RightHandSide<dim>>();
    this->exact_soln_ = std::make_shared<minsurf_verify::Solution<dim>>();
    this->bc_dirichlet_.push_back(dirichlet_bc<dim>{1, dirichlet1});

    std::vector<typename DomainGeometry<dim>::bc_mark_desc> bcmarks;
    bcmarks.push_back(std::make_pair(this->bc_dirichlet_[0].bc_id,
        [](const dealii::Point<dim>&) { return true; }));
    this->geom_ = std::make_shared<geom::Cube<dim>>(bcmarks);
}

template <int dim>
void MinSurfCubeVerify<dim>::add_case_cmd_args(bpo::options_description&) const
{
}

template class MinSurfCubeVerify<2>;
template class MinSurfCubeVerify<3>;


}
}
