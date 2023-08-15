#include "fourier.hpp"

#include <limits>

namespace paramsim {
namespace cases {

using namespace dealii;

template <int dim>
void PoissonBCFourier<dim>::initialize(const bpo::variables_map& params)
{
    std::shared_ptr<poisson_fourier::DirichletIn<dim>> dirichlet1;
    if(params.count("wavelength")) {
        constexpr int n_modes = poisson_fourier::Params<dim>::n_modes;
        std::array<double, n_modes> as;
        std::array<double, n_modes> bs;
        for(int ic = 1; ic < n_modes+1; ic++) {
            const std::string coflag =
                std::string("a") + std::to_string(ic);
            as[ic-1] = params[coflag.c_str()].as<double>();
            //eg. --a1=0.6 --b1=0.4
            const std::string sflag = std::string("b") + std::to_string(ic);
            bs[ic-1] = params[sflag.c_str()].as<double>();
        }
        const double wavelength = params["wavelength"].as<double>();
        const double a0 = params["a0"].as<double>();
        poisson_fourier::Params<dim> params(as, bs, a0, wavelength);
        dirichlet1 = std::make_shared<poisson_fourier::DirichletIn<dim>>(params);

        // Write out params to confirm
        std::cout << "Case 'bc_fourier' for Poisson: read parameters:\n";
        std::cout << "  Fundamental wavelength = " << params.f_wavelength << std::endl;
        std::cout << "  Constant term = " << params.a0 << std::endl;
        for(int ic = 0; ic < n_modes; ic++) {
            std::cout << "  Modes " << ic << ": (";
            std::cout << as[ic] << ", " << bs[ic] << ")" << std::endl;
        }
    } else {
        dirichlet1 = std::make_shared<poisson_fourier::DirichletIn<dim>>();
        std::cout << "Case 'bc_fourier' for Poisson: default parameters.\n";
    }

    this->rhs_ = std::make_shared<poisson_fourier::RightHandSide<dim>>();

    auto dirichlet2 = std::make_shared<poisson_fourier::DirichletConstant<dim>>(1.0);

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
    this->geom_ = std::make_shared<poisson_fourier::Cube<dim>>(bcmarks);
}
    
template <int dim>
void PoissonBCFourier<dim>::add_case_cmd_args(bpo::options_description& desc) const
{
	desc.add_options()
        ("wavelength", bpo::value<double>(), "The fundamental wavelength for zeroth mode");
	desc.add_options() ("a0", bpo::value<double>(), "Constant term");
    constexpr int n_modes = poisson_fourier::Params<dim>::n_modes;
    for(int ic = 1; ic < n_modes+1; ic++) {
        const std::string coflag =
            std::string("a") + std::to_string(ic);
        const std::string descstr = "Cosine coefficient of " + std::to_string(ic) + "th mode";
        desc.add_options()
            (coflag.c_str(), bpo::value<double>(), descstr.c_str());
        //eg. centers[0][1] = params["center0_y"].as<double>();
        const std::string sflag = std::string("b") + std::to_string(ic);
        // eg. "center1_coeff"
        const std::string sdescstr = "Sine coefficient of " + std::to_string(ic) + "th mode";
        desc.add_options()
            (sflag.c_str(), bpo::value<double>(), sdescstr.c_str());
    }
}

template class PoissonBCFourier<2>;

}
}
