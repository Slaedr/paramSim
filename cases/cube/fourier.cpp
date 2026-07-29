#include "fourier.hpp"

#include "case_parameters.hpp"

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
    const auto header = reader.read_finite_values(
        dim + 1, "constant and directional fundamental wavelengths");

    Params<dim> params;
    params.modes.clear();
    params.modes.reserve(mode_count);
    params.constant = header[0];
    for (int direction = 0; direction < dim; ++direction) {
        const double wavelength = header[direction + 1];
        if (wavelength <= 0.0) {
            reader.fail(2, "fundamental wavelengths must be positive");
        }
        params.fundamental_wavelength[direction] = wavelength;
    }
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
        std::cout << "  Fundamental wavelengths = (";
        for (int direction = 0; direction < dim; ++direction) {
            if (direction > 0) {
                std::cout << ", ";
            }
            std::cout << case_params.fundamental_wavelength[direction];
        }
        std::cout << ")" << std::endl;
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

    this->initialize_dirichlet_everywhere(dirichlet1);
}

template class CubeFourier<2>;
template class CubeFourier<3>;

} // namespace cube
} // namespace cases
} // namespace paramsim
