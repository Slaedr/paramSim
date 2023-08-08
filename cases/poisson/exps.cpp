#include "exps.hpp"

#include <limits>

namespace paramsim {
namespace cases {

using namespace dealii;
   
template <int dim>
const std::array<std::string, 3> PoissonBCExp<dim>::dimnames{{"x","y","z"}};

template <int dim>
void PoissonBCExp<dim>::initialize(const bpo::variables_map& params)
{
    std::shared_ptr<poisson_exp::DirichletIn<dim>> dirichlet1;
    if(params.count("width")) {
        constexpr int n_centers = poisson_exp::Params<dim>::n_centers;
        std::array<Point<dim>, n_centers> centers;
        std::array<double, n_centers> coeffs;
        for(int ic = 0; ic < n_centers; ic++) {
            const std::string flag =
                std::string("center") + std::to_string(ic) + "_y";
            centers[ic][0] = -1.0;
            centers[ic][1] = params[flag.c_str()].as<double>();
            //eg. centers[0][0] = params["center0_x"].as<double>();
            const std::string coflag = std::string("center") + std::to_string(ic) + "_coeff";
            // eg. "center1_coeff"
            coeffs[ic] = params[coflag.c_str()].as<double>();
        }
        const double width_sigma = params["width"].as<double>();
        poisson_exp::Params<dim> params(centers, coeffs, width_sigma);
        this->rhs_ = std::make_shared<poisson_exp::RightHandSide<dim>>();
        dirichlet1 = std::make_shared<poisson_exp::DirichletIn<dim>>(params);

        // Write out params to confirm
        std::cout << "Case 'bc_exp' for Poisson: read parameters:\n";
        std::cout << "  Width = " << width_sigma << std::endl;
        for(int ic = 0; ic < n_centers; ic++) {
            std::cout << "  Center " << ic << ": (";
            for(int idim = 0; idim < dim; idim++) {
                std::cout << centers[ic][idim] << ", ";
            }
            std::cout << "), coeff = " << coeffs[ic] << std::endl;
        }
    } else {
        this->rhs_ = std::make_shared<poisson_exp::RightHandSide<dim>>();
        dirichlet1 = std::make_shared<poisson_exp::DirichletIn<dim>>();
        std::cout << "Case 'bc_exp' for Poisson: default parameters.\n";
    }
        
    auto dirichlet2 = std::make_shared<poisson_exp::DirichletConstant<dim>>(1.0);

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
    this->geom_ = std::make_shared<poisson_exp::Cube<dim>>(bcmarks);
}
    
template <int dim>
void PoissonBCExp<dim>::add_case_cmd_args(bpo::options_description& desc) const
{
	desc.add_options()
        ("width", bpo::value<double>(), "Width of each hill");
    constexpr int n_centers = poisson_exp::Params<dim>::n_centers;
    for(int ic = 0; ic < n_centers; ic++) {
        const std::string flag =
            std::string("center") + std::to_string(ic) + "_y";
        const std::string descstr = "y coordinate of " + std::to_string(ic) + "th center";
        desc.add_options()
            (flag.c_str(), bpo::value<double>(), descstr.c_str());
        //eg. centers[0][1] = params["center0_y"].as<double>();
        const std::string coflag = std::string("center") + std::to_string(ic) + "_coeff";
        const std::string codescstr = "Coefficient multiplying the " +
                std::to_string(ic) + "th center";
        // eg. "center1_coeff"
        desc.add_options()
            (coflag.c_str(), bpo::value<double>(), codescstr.c_str());
    }
}

template class PoissonBCExp<2>;

}
}
