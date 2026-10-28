#include "polynomial.hpp"

#include "case_parameters.hpp"

namespace paramsim {
namespace cases {
namespace cube {

using namespace dealii;

namespace polynomial {

template <int dim>
Params<dim> read_parameters(const std::string& filename)
{
    ParameterFileReader reader("cube_polynomial", filename);
    const unsigned degree_levels = reader.read_count("number of degree levels");
    const auto center_values =
        reader.read_finite_values(3, "polynomial center");

    Params<dim> params;
    for (int coordinate = 0; coordinate < dim; ++coordinate) {
        params.center[coordinate] = center_values[coordinate];
    }
    params.coefficients_by_degree.clear();
    params.coefficients_by_degree.reserve(degree_levels);
    for (unsigned degree = 0; degree < degree_levels; ++degree) {
        const std::size_t coefficient_count =
            degree_exponents<dim>(degree).size();
        params.coefficients_by_degree.push_back(reader.read_finite_values(
            coefficient_count,
            "degree " + std::to_string(degree) + " coefficients"));
    }
    reader.require_end();
    return params;
}

template Params<2> read_parameters<2>(const std::string&);
template Params<3> read_parameters<3>(const std::string&);

} // namespace polynomial

template <int dim>
void CubePolynomial<dim>::initialize(const bpo::variables_map& params)
{
    const auto case_params = params.count("case_params_file")
        ? polynomial::read_parameters<dim>(params["case_params_file"].as<std::string>())
        : polynomial::Params<dim>{};
    auto dirichlet1 = std::make_shared<polynomial::DirichletIn<dim>>(case_params);

    this->rhs_ = std::make_shared<polynomial::RightHandSide<dim>>();

    this->initialize_dirichlet_everywhere(dirichlet1);
}

template class CubePolynomial<2>;
template class CubePolynomial<3>;

} // namespace cube
} // namespace cases
} // namespace paramsim
