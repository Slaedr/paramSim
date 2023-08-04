#include "verify.hpp"

#include <limits>

namespace paramsim {
namespace cases {

using namespace dealii;
   
template <int dim>
const std::array<std::string, 3> PoissonVerify<dim>::dimnames{{"x","y","z"}};

template <int dim>
void PoissonVerify<dim>::initialize(const bpo::variables_map& params)
{
    if(params.count("width")) {
        constexpr int n_centers = poisson_verify::Params<dim>::n_centers;
        std::array<Point<dim>, n_centers> centers;
        std::array<double, n_centers> coeffs;
        for(int ic = 0; ic < n_centers; ic++) {
            for(int idim = 0; idim < dim; idim++) {
                const std::string flag =
                    std::string("center") + std::to_string(ic) + "_" + dimnames[idim];
                centers[ic][idim] = params[flag.c_str()].as<double>();
                //eg. centers[0][0] = params["center0_x"].as<double>();
            }
            const std::string coflag = std::string("center") + std::to_string(ic) + "_coeff";
            // eg. "center1_coeff"
            coeffs[ic] = params[coflag.c_str()].as<double>();
        }
        const double width_sigma = params["width"].as<double>();
        poisson_verify::Params<dim> params(centers, coeffs, width_sigma);
        this->exact_soln_ = std::make_shared<poisson_verify::Solution<dim>>(params);
        this->rhs_ = std::make_shared<poisson_verify::RightHandSide<dim>>(params);
        this->neumann_ = std::make_shared<poisson_verify::Neumann<dim>>(params);
        this->dirichlet_ = this->exact_soln_;

        // Write out params to confirm
        std::cout << "Case 'verify' for Poisson: read parameters:\n";
        std::cout << "  Width = " << width_sigma << std::endl;
        for(int ic = 0; ic < n_centers; ic++) {
            std::cout << "  Center " << ic << ": (";
            for(int idim = 0; idim < dim; idim++) {
                std::cout << centers[ic][idim] << ", ";
            }
            std::cout << "), coeff = " << coeffs[ic] << std::endl;
        }
    } else {
        this->exact_soln_ = std::make_shared<poisson_verify::Solution<dim>>();
        this->rhs_ = std::make_shared<poisson_verify::RightHandSide<dim>>();
        this->neumann_ = std::make_shared<poisson_verify::Neumann<dim>>();
        this->dirichlet_ = this->exact_soln_;
        std::cout << "Case 'verify' for Poisson: default parameters.\n";
    }

    //constexpr double tol = 100*std::numeric_limits<double>::epsilon();
    this->bcid_dirichlet_ = 1;
    this->bcid_neumann_ = 2;
    std::vector<typename DomainGeometry<dim>::bc_mark_desc> bcmarks;
    //bcmarks.push_back(std::make_pair(this->bcid_dirichlet_, 
    //    [](const dealii::Point<dim>& p) {
    //    if(std::abs(p[0] - 1.0) > tol && std::abs(p[1] - 1.0) > tol) {
    //        return true;
    //    } else {
    //        return false;
    //    }
    //    }));
    //bcmarks.push_back(std::make_pair(this->bcid_neumann_, 
    //    [](const dealii::Point<dim>& p) {
    //    if(std::abs(p[0] - 1.0) < tol || std::abs(p[1] - 1.0) < tol) {
    //        return true;
    //    } else {
    //        return false;
    //    }
    //    }));
    bcmarks.push_back(std::make_pair(this->bcid_dirichlet_, 
        [](const dealii::Point<dim>& ) { return true; }));
    this->geom_ = std::make_shared<poisson_verify::Cube<dim>>(bcmarks);
}
    
template <int dim>
void PoissonVerify<dim>::add_case_cmd_args(bpo::options_description& desc) const
{
	desc.add_options()
        ("width", bpo::value<double>(), "Width of each hill");
    constexpr int n_centers = poisson_verify::Params<dim>::n_centers;
    for(int ic = 0; ic < n_centers; ic++) {
        for(int idim = 0; idim < dim; idim++) {
            const std::string flag =
                std::string("center") + std::to_string(ic) + "_" + dimnames[idim];
            const std::string descstr = dimnames[idim] + "th coordinate of " +
                std::to_string(ic) + "th center";
            desc.add_options()
                (flag.c_str(), bpo::value<double>(), descstr.c_str());
            //eg. centers[0][0] = params["center0_x"].as<double>();
        }
        const std::string coflag = std::string("center") + std::to_string(ic) + "_coeff";
        const std::string codescstr = "Coefficient multiplying the " +
                std::to_string(ic) + "th center";
        // eg. "center1_coeff"
        desc.add_options()
            (coflag.c_str(), bpo::value<double>(), codescstr.c_str());
    }
}

template class PoissonVerify<2>;

}
}
