#include "polynomial.hpp"

#include "case_parameters.hpp"

#include <limits>

namespace paramsim {
namespace cases {
namespace cube {

using namespace dealii;

namespace polynomial {

template <int dim>
Params<dim> read_parameters(const std::string& filename)
{
    ParameterFileReader reader("cube_polynomial", filename);
    const unsigned degree_levels =
        reader.read_count("number of degree levels");
    const auto center_values =
        reader.read_finite_values(3, "polynomial center");

    Params<dim> params;
    for (std::size_t coordinate = 0; coordinate < dim; ++coordinate) {
        params.center[coordinate] = center_values[coordinate];
    }
    params.coefficients_by_degree.clear();
    params.coefficients_by_degree.reserve(degree_levels);
    for (unsigned degree = 0; degree < degree_levels; ++degree) {
        const std::size_t coefficient_count =
            degree_exponents<dim>(degree).size();
        params.coefficients_by_degree.push_back(
            reader.read_finite_values(
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
    std::shared_ptr<polynomial::DirichletIn<dim>> dirichlet1;
    if (params.count("center_y")) {
        constexpr unsigned legacy_degree_levels = 4;
        std::array<double, legacy_degree_levels> legacy_coefficients{};
        for (unsigned degree = 0; degree < legacy_degree_levels; ++degree) {
            const std::string coefficient_flag =
                "a" + std::to_string(degree);
            legacy_coefficients[degree] =
                params[coefficient_flag].as<double>();
        }

        polynomial::Params<dim> case_params;
        case_params.center.fill(0.0);
        case_params.center[1] = params["center_y"].as<double>();
        case_params.coefficients_by_degree.clear();
        case_params.coefficients_by_degree.reserve(legacy_degree_levels);
        for (unsigned degree = 0; degree < legacy_degree_levels; ++degree) {
            const auto exponents = degree_exponents<dim>(degree);
            std::vector<double> degree_coefficients(exponents.size(), 0.0);
            for (std::size_t term = 0; term < exponents.size(); ++term) {
                if (exponents[term][1] == degree) {
                    degree_coefficients[term] =
                        legacy_coefficients[degree];
                }
            }
            case_params.coefficients_by_degree.push_back(
                std::move(degree_coefficients));
        }
        dirichlet1 =
            std::make_shared<polynomial::DirichletIn<dim>>(case_params);

        std::cout << "Case 'cube_polynomial': read parameters:\n";
        std::cout << "  Center: (";
        for (std::size_t coordinate = 0; coordinate < dim; ++coordinate) {
            if (coordinate > 0) {
                std::cout << ", ";
            }
            std::cout << case_params.center[coordinate];
        }
        std::cout << ")\n";
        for (unsigned degree = 0; degree < legacy_degree_levels; ++degree) {
            const auto exponents = degree_exponents<dim>(degree);
            for (std::size_t term = 0; term < exponents.size(); ++term) {
                std::cout << "  Degree " << degree << " exponents (";
                for (std::size_t coordinate = 0; coordinate < dim;
                     ++coordinate) {
                    if (coordinate > 0) {
                        std::cout << ", ";
                    }
                    std::cout << exponents[term][coordinate];
                }
                std::cout << "): "
                          << case_params.coefficients_by_degree[degree][term]
                          << '\n';
            }
        }
    } else {
        dirichlet1 = std::make_shared<polynomial::DirichletIn<dim>>();
        std::cout << "Case 'cube_polynomial': default parameters.\n";
    }

    this->rhs_ = std::make_shared<polynomial::RightHandSide<dim>>();

    auto dirichlet2 = std::make_shared<polynomial::DirichletConstant<dim>>(1.0);

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
    this->geom_ = std::make_shared<geom::Cube<dim>>(bcmarks);
}

template <int dim>
void CubePolynomial<dim>::add_case_cmd_args(bpo::options_description& desc) const
{
    desc.add_options()(
        "center_y", bpo::value<double>(),
        "The polynomial terms' center or offset");
    constexpr unsigned legacy_degree_levels = 4;
    for (unsigned degree = 0; degree < legacy_degree_levels; ++degree) {
        const std::string coefficient_flag = "a" + std::to_string(degree);
        const std::string description =
            "Coefficient of degree " + std::to_string(degree);
        desc.add_options()(coefficient_flag.c_str(), bpo::value<double>(),
                           description.c_str());
    }
}

template class CubePolynomial<2>;
template class CubePolynomial<3>;

} // namespace cube
} // namespace cases
} // namespace paramsim
