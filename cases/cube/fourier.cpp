#include "fourier.hpp"

#include "case_parameters.hpp"

#include <limits>

namespace paramsim {
namespace cases {
namespace cube {

using namespace dealii;

namespace fourier {

template <int dim>
Params<dim> read_parameters(const std::string& filename)
{
    ParameterFileReader reader("cube_fourier", filename);
    const unsigned mode_count = reader.read_count("number of modes");
    const auto header =
        reader.read_finite_values(2, "constant and fundamental wavelength");
    if (header[1] <= 0.0) {
        reader.fail(2, "fundamental wavelength must be positive");
    }

    Params<dim> params;
    params.modes.clear();
    params.modes.reserve(mode_count);
    params.constant = header[0];
    params.fundamental_wavelength = header[1];
    for (unsigned index = 0; index < mode_count; ++index) {
        const auto coefficients =
            reader.read_finite_values(2, "mode " + std::to_string(index + 1));
        params.modes.push_back({coefficients[0], coefficients[1]});
    }
    reader.require_end();
    return params;
}

template Params<2> read_parameters<2>(const std::string&);
template Params<3> read_parameters<3>(const std::string&);

} // namespace fourier

template <int dim>
void CubeFourier<dim>::initialize(const bpo::variables_map& params)
{
    std::shared_ptr<fourier::DirichletIn<dim>> dirichlet1;
    if (params.count("case_params_file")) {
        const auto case_params = fourier::read_parameters<dim>(
            params["case_params_file"].as<std::string>());
        dirichlet1 = std::make_shared<fourier::DirichletIn<dim>>(case_params);

        std::cout << "Case 'cube_fourier': read parameters:\n";
        std::cout << "  Fundamental wavelength = "
                  << case_params.fundamental_wavelength << std::endl;
        std::cout << "  Constant term = " << case_params.constant << std::endl;
        for (std::size_t index = 0; index < case_params.modes.size(); ++index) {
            const auto& mode = case_params.modes[index];
            std::cout << "  Mode " << index + 1 << ": ("
                      << mode.cosine_coefficient << ", "
                      << mode.sine_coefficient << ")" << std::endl;
        }
    } else {
        dirichlet1 = std::make_shared<fourier::DirichletIn<dim>>();
        std::cout << "Case 'cube_fourier': default parameters.\n";
    }

    this->rhs_ = std::make_shared<fourier::RightHandSide<dim>>();

    auto dirichlet2 = std::make_shared<fourier::DirichletConstant<dim>>(1.0);

    this->bc_dirichlet_.push_back(dirichlet_bc<dim>{1, dirichlet1});
    this->bc_dirichlet_.push_back(dirichlet_bc<dim>{2, dirichlet2});

    constexpr double tol = 1000 * std::numeric_limits<double>::epsilon();
    std::vector<typename DomainGeometry<dim>::bc_mark_desc> bcmarks;
    bcmarks.push_back(std::make_pair(this->bc_dirichlet_[1].bc_id,
                                     [](const dealii::Point<dim>& p) {
                                         if (std::abs(p[0] - (-1.0)) > tol) {
                                             return true;
                                         } else {
                                             return false;
                                         }
                                     }));
    bcmarks.push_back(std::make_pair(this->bc_dirichlet_[0].bc_id,
                                     [](const dealii::Point<dim>& p) {
                                         if (std::abs(p[0] - (-1.0)) <= tol) {
                                             return true;
                                         } else {
                                             return false;
                                         }
                                     }));
    this->geom_ = std::make_shared<geom::Cube<dim>>(bcmarks);
}

template class CubeFourier<2>;
template class CubeFourier<3>;

} // namespace cube
} // namespace cases
} // namespace paramsim
