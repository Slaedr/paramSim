#include "case.hpp"

#include <memory>

#include "minimal_surface/gaussians.hpp"
#include "minimal_surface/minimal_surface.hpp"
#include "minimal_surface/verify.hpp"
#include "cube/exponential.hpp"
#include "cube/fourier.hpp"
#include "cube/polynomial.hpp"
#include "poisson/verify.hpp"

namespace paramsim {

template <int dim>
std::unique_ptr<Case<dim>> create_case(const CommonParams &params,
                                       const int n_args,
                                       const char *const argv[])
{
    std::unique_ptr<Case<dim>> tcase;
    if (params.case_str == "poisson_verify") {
        tcase = std::make_unique<cases::PoissonVerify<dim>>();
    } else if (params.case_str == "cube_exponential") {
        tcase = std::make_unique<cases::cube::CubeExponential<dim>>();
    } else if (params.case_str == "cube_fourier") {
        tcase = std::make_unique<cases::cube::CubeFourier<dim>>();
    } else if (params.case_str == "cube_polynomial") {
        tcase = std::make_unique<cases::cube::CubePolynomial<dim>>();
    } else if (params.case_str == "minimal_surface_ball_verify") {
        tcase = std::make_unique<cases::MinSurfBallVerify<dim>>();
    } else if (params.case_str == "minimal_surface_cube_verify") {
        tcase = std::make_unique<cases::MinSurfCubeVerify<dim>>();
    } else if (params.case_str == "minimal_surface_disk_sinusoidal") {
        tcase = std::make_unique<cases::MinSurfDiskSinusoidal<dim>>();
    } else if (params.case_str == "minimal_surface_cube_sinusoidal") {
        tcase = std::make_unique<cases::MinSurfCubeSinusoidal<dim>>();
    } else if (params.case_str == "minimal_surface_cube_polynomial") {
        tcase = std::make_unique<cases::MinSurfCubePolynomial<dim>>();
    } else if (params.case_str == "minimal_surface_cube_gaussians") {
        tcase = std::make_unique<cases::MinSurfCubeGaussians<dim>>();
    } else {
        throw std::runtime_error("Non-existent case!");
    }
    bpo::options_description case_desc(std::string("Solves the case ") +
                                       params.case_str +
                                       " given one set of parameters.");
    add_common_options(case_desc, "");
    tcase->add_case_cmd_args(case_desc);
    const bpo::variables_map case_cmdmap =
        get_cmd_args(n_args, argv, case_desc);
    tcase->initialize(case_cmdmap);
    return tcase;
}

template std::unique_ptr<Case<2>> create_case(const CommonParams &, int,
                                              const char *const[]);
template std::unique_ptr<Case<3>> create_case(const CommonParams &, int,
                                              const char *const[]);

} // namespace paramsim
