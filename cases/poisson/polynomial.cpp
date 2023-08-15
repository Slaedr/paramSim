#include "polynomial.hpp"

#include <limits>

namespace paramsim {
namespace cases {

using namespace dealii;

template <int dim>
void PoissonBCPolynomial<dim>::initialize(const bpo::variables_map& params)
{
    std::shared_ptr<poisson_poly::DirichletIn<dim>> dirichlet1;
    if(params.count("center_y")) {
        constexpr int n_terms = poisson_poly::Params<dim>::n_terms;
        std::array<double, n_terms> as;
        for(int ic = 0; ic < n_terms; ic++) {
            const std::string coflag =
                std::string("a") + std::to_string(ic);
            as[ic] = params[coflag.c_str()].as<double>();
            //eg. --a1=0.6 --a2=0.4
        }
        const double center = params["center_y"].as<double>();
        poisson_poly::Params<dim> params(as, center);
        dirichlet1 = std::make_shared<poisson_poly::DirichletIn<dim>>(params);

        // Write out params to confirm
        std::cout << "Case 'bc_polynomial' for Poisson: read parameters:\n";
        std::cout << "  Center = " << params.center << std::endl;
        for(int ic = 0; ic < n_terms; ic++) {
            std::cout << "  term " << ic << ": ";
            std::cout << as[ic];
        }
        std::cout << std::endl;
    } else {
        dirichlet1 = std::make_shared<poisson_poly::DirichletIn<dim>>();
        std::cout << "Case 'bc_polynomial' for Poisson: default parameters.\n";
    }

    this->rhs_ = std::make_shared<poisson_poly::RightHandSide<dim>>();

    auto dirichlet2 = std::make_shared<poisson_poly::DirichletConstant<dim>>(1.0);

    this->bc_dirichlet_.push_back(dirichlet_bc<dim>{1, dirichlet1});
    this->bc_dirichlet_.push_back(dirichlet_bc<dim>{2, dirichlet2});

    constexpr double tol = 1000*std::numeric_limits<double>::epsilon();
    std::vector<typename DomainGeometry<dim>::bc_mark_desc> bcmarks;
    bcmarks.push_back(std::make_pair(this->bc_dirichlet_[1].bc_id, 
        [](const dealii::Point<dim>& p) {
        if(std::abs(p[0] - (-1.0)) > tol) {
            return true;
        } else {
            return false;
        }
        }));
    bcmarks.push_back(std::make_pair(this->bc_dirichlet_[0].bc_id, 
        [](const dealii::Point<dim>& p) {
        if(std::abs(p[0] - (-1.0)) <= tol) {
            return true;
        } else {
            return false;
        }
        }));
    this->geom_ = std::make_shared<poisson_poly::Cube<dim>>(bcmarks);
}
    
template <int dim>
void PoissonBCPolynomial<dim>::add_case_cmd_args(bpo::options_description& desc) const
{
    desc.add_options()
        ("center_y", bpo::value<double>(), "The polynomial terms' center or offset");
    constexpr int n_terms = poisson_poly::Params<dim>::n_terms;
    for(int ic = 0; ic < n_terms; ic++) {
        const std::string coflag =
            std::string("a") + std::to_string(ic);
        const std::string descstr = "Coefficient of " + std::to_string(ic) + "th term";
        desc.add_options()
            (coflag.c_str(), bpo::value<double>(), descstr.c_str());
    }
}

template class PoissonBCPolynomial<2>;

}
}
