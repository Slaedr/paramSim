#include "case.hpp"

#include "poisson/verify.hpp"
#include "poisson/exps.hpp"
#include "poisson/fourier.hpp"
#include "poisson/polynomial.hpp"
#include "convdiff/step51.hpp"
#include "minimal_surface/minimal_surface.hpp"
#include "minimal_surface/gaussians.hpp"

namespace paramsim {

template <int dim>
std::unique_ptr<Case<dim>> create_case(const std::string case_str)
{
    if(case_str == "convdiff_step51") {
        return std::make_unique<cases::Step51<dim>>();
    } else if(case_str == "poisson_verify") {
        return std::make_unique<cases::PoissonVerify<dim>>();
    } else if(case_str == "poisson_bc_exp") {
        return std::make_unique<cases::PoissonBCExp<dim>>();
    } else if(case_str == "poisson_bc_fourier") {
        return std::make_unique<cases::PoissonBCFourier<dim>>();
    } else if(case_str == "poisson_bc_polynomial") {
        return std::make_unique<cases::PoissonBCPolynomial<dim>>();
    } else if(case_str == "minimal_surface_disk_sinusoidal") {
        return std::make_unique<cases::MinSurfDiskSinusoidal<dim>>();
    } else if(case_str == "minimal_surface_cube_sinusoidal") {
        return std::make_unique<cases::MinSurfCubeSinusoidal<dim>>();
    } else if(case_str == "minimal_surface_cube_polynomial") {
        return std::make_unique<cases::MinSurfCubePolynomial<dim>>();
    } else if(case_str == "minimal_surface_cube_gaussians") {
        return std::make_unique<cases::MinSurfCubeGaussians<dim>>();
    } else {
        throw std::runtime_error("Non-existent case!");
    }
}

template std::unique_ptr<Case<2>> create_case(const std::string case_str);

}
