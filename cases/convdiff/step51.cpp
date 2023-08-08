#include "step51.hpp"

namespace paramsim {
namespace cases {

using namespace dealii;
   
template <int dim>
const std::array<std::string, 3> Step51<dim>::dimnames{{"x","y","z"}};

template <int dim>
void Step51<dim>::initialize(const bpo::variables_map& params)
{
    constexpr int n_centers = step51::Solution<dim>::n_centers;
    if(params.count("width")) {
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
        this->exact_soln_ = std::make_shared<cases::step51::Solution<dim>>(centers, coeffs,
                width_sigma);
        this->exact_soln_and_grad_ = std::make_shared<cases::step51::SolutionAndGradient<dim>>(
                centers, coeffs, width_sigma);
        this->rhs_ = std::make_shared<cases::step51::RightHandSide<dim>>(
                centers, coeffs, width_sigma);
        this->neumann_ = std::make_shared<cases::step51::Neumann<dim>>(
                centers, coeffs, width_sigma);
        this->conv_vel_ = std::make_shared<cases::step51::ConvectionVelocity<dim>>();

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
        this->exact_soln_ = std::make_shared<cases::step51::Solution<dim>>();
        this->exact_soln_and_grad_ = std::make_shared<cases::step51::SolutionAndGradient<dim>>();
        this->rhs_ = std::make_shared<cases::step51::RightHandSide<dim>>();
        this->neumann_ = std::make_shared<cases::step51::Neumann<dim>>();
        this->conv_vel_ = std::make_shared<cases::step51::ConvectionVelocity<dim>>();
    }

    this->bc_dirichlet_.push_back(dirichlet_bc<dim>{0, this->exact_soln_});
    this->bcid_neumann_ = 1;
    std::vector<typename DomainGeometry<dim>::bc_mark_desc> bcmarks;
    bcmarks.push_back(std::make_pair(this->bcid_neumann_, 
        [](const dealii::Point<dim>& p) {
            if ((std::fabs(p(0) - (-1)) < 1e-12) ||
                (std::fabs(p(1) - (-1)) < 1e-12))
                return true;
            else
                return false;
        }));
    this->geom_ = std::make_shared<cases::step51::Cube<dim>>(bcmarks);
}
    
template <int dim>
void Step51<dim>::add_case_cmd_args(bpo::options_description& desc) const
{
	desc.add_options()
        ("width", bpo::value<double>(), "Width of each hill");
    constexpr int n_centers = step51::Solution<dim>::n_centers;
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

template class Step51<2>;

}
}
