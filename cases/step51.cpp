#include "step51.hpp"

namespace cases {

using namespace dealii;
   
template <int dim>
const std::array<std::string, 3> Step51<dim>::dimnames{{"x","y","z"}};

template <int dim>
Step51<dim>::Step51(const bpo::variables_map& params)
{
    constexpr int n_centers = step51::Solution<dim>::n_centers;
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
    exact_soln_ = std::make_shared<cases::step51::Solution<dim>>(centers, coeffs, width_sigma);
    exact_soln_grad_ = std::make_shared<cases::step51::SolutionAndGradient<dim>>(
            centers, coeffs, width_sigma);
    rhs_ = std::make_shared<cases::step51::RightHandSide<dim>>(centers, coeffs, width_sigma);
    neumann_ = std::make_shared<cases::step51::Neumann<dim>>(centers, coeffs, width_sigma);
    conv_vel_ = std::make_shared<cases::step51::ConvectionVelocity<dim>>();
}
    
template <int dim>
void Step51<dim>::add_case_cmd_args(bpo::options_description& desc)
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

