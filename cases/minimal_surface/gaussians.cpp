#include "gaussians.hpp"

namespace paramsim {
namespace cases{


template <int dim>
void MinSurfCubeGaussians<dim>::initialize(const bpo::variables_map& params)
{
    std::shared_ptr<minsurf_gauss::Dirichlet<dim>> dirichlet1;
    if(params.count("width")) {
        constexpr int n_centers = minsurf_gauss::Params<dim>::n_centers;
        std::array<double, n_centers> centers;
        std::array<double, n_centers> coeffs;
        for(int ic = 0; ic < n_centers; ic++) {
            const std::string flag =
                std::string("center") + std::to_string(ic) + "_y";
            centers[ic] = params[flag.c_str()].as<double>();
            //eg. centers[0][0] = params["center0_x"].as<double>();
            const std::string coflag = std::string("center") + std::to_string(ic) + "_coeff";
            // eg. "center1_coeff"
            coeffs[ic] = params[coflag.c_str()].as<double>();
        }
        const double width_sigma = params["width"].as<double>();
        minsurf_gauss::Params<dim> params(centers, coeffs, width_sigma);
        dirichlet1 = std::make_shared<minsurf_gauss::Dirichlet<dim>>(params);

        // Write out params to confirm
        std::cout << "Case 'cube_gaussians' for Minimal Surface: read parameters:\n";
        std::cout << "  Width = " << width_sigma << std::endl;
        for(int ic = 0; ic < n_centers; ic++) {
            std::cout << "  Center " << ic << ": " << centers[ic];
            std::cout << ", coeff = " << coeffs[ic] << std::endl;
        }
    } else {
        dirichlet1 = std::make_shared<minsurf_gauss::Dirichlet<dim>>();
        std::cout << "Case 'cube_gaussians' for Minimal Surface: default parameters.\n";
    }

    this->set_geometry_and_boundary(dirichlet1);
}

template <int dim>
void MinSurfCubeGaussians<dim>::add_case_cmd_args(bpo::options_description& desc) const
{
	desc.add_options()
        ("width", bpo::value<double>(), "Width of each hill");
    constexpr int n_centers = minsurf_gauss::Params<dim>::n_centers;
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

template class MinSurfCubeGaussians<2>;
template class MinSurfCubeGaussians<3>;


}
}
