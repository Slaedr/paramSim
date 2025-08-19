#include "case.hpp"

#include <memory>

#include "poisson/verify.hpp"
#include "poisson/exps.hpp"
#include "poisson/fourier.hpp"
#include "poisson/polynomial.hpp"
#include "convdiff/step51.hpp"
#include "minimal_surface/minimal_surface.hpp"
#include "minimal_surface/gaussians.hpp"

namespace paramsim {


template <int dim>
std::unique_ptr<Case<dim>> create_case(const CommonParams& params, const int n_args,
                                       const char* const argv[])
{
    std::unique_ptr<Case<dim>> tcase;
    if(params.case_str == "convdiff_step51") {
        tcase = std::make_unique<cases::Step51<dim>>();
    } else if(params.case_str == "poisson_verify") {
        tcase = std::make_unique<cases::PoissonVerify<dim>>();
    } else if(params.case_str == "poisson_bc_exp") {
        tcase = std::make_unique<cases::PoissonBCExp<dim>>();
    } else if(params.case_str == "poisson_bc_fourier") {
        tcase = std::make_unique<cases::PoissonBCFourier<dim>>();
    } else if(params.case_str == "poisson_bc_polynomial") {
        tcase = std::make_unique<cases::PoissonBCPolynomial<dim>>();
    } else if(params.case_str == "minimal_surface_disk_sinusoidal") {
        tcase = std::make_unique<cases::MinSurfDiskSinusoidal<dim>>();
    } else if(params.case_str == "minimal_surface_cube_sinusoidal") {
        tcase = std::make_unique<cases::MinSurfCubeSinusoidal<dim>>();
    } else if(params.case_str == "minimal_surface_cube_polynomial") {
        tcase = std::make_unique<cases::MinSurfCubePolynomial<dim>>();
    } else if(params.case_str == "minimal_surface_cube_gaussians") {
        tcase = std::make_unique<cases::MinSurfCubeGaussians<dim>>();
    } else {
        throw std::runtime_error("Non-existent case!");
    }
    bpo::options_description case_desc
        (std::string("Solves the case ") + params.case_str + " given one set of parameters.");
    tcase->add_case_cmd_args(case_desc);
    const bpo::variables_map case_cmdmap = get_cmd_args(n_args, argv, case_desc);
    tcase->initialize(case_cmdmap);
    return tcase;
}

template std::unique_ptr<Case<2>> create_case(const CommonParams&, int, const char *const[]);

}
