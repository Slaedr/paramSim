#include "exponential.hpp"

#include "case_parameters.hpp"

#include <limits>

namespace paramsim {
namespace cases {
namespace cube {

using namespace dealii;

namespace exponential {

template <int dim>
Params<dim> read_parameters(const std::string& filename)
{
    ParameterFileReader reader("cube_exponential", filename);
    const unsigned center_count = reader.read_count("number of centers");

    Params<dim> params;
    params.centers.clear();
    params.centers.reserve(center_count);
    for (unsigned index = 0; index < center_count; ++index) {
        const auto values =
            reader.read_finite_values(5, "center " + std::to_string(index));

        for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
            if (values[coordinate] < -1.0 || values[coordinate] > 1.0) {
                reader.fail(index + 2,
                            "center coordinates must lie in [-1, 1]");
            }
        }
        if (values[4] <= 0.0) {
            reader.fail(index + 2, "center width must be positive");
        }

        GaussianCenter<dim> center;
        for (int coordinate = 0; coordinate < dim; ++coordinate) {
            center.coordinates[coordinate] = values[coordinate];
        }
        center.coefficient = values[3];
        center.width = values[4];
        params.centers.push_back(center);
    }
    reader.require_end();
    return params;
}

template Params<2> read_parameters<2>(const std::string&);
template Params<3> read_parameters<3>(const std::string&);

} // namespace exponential

template <int dim>
void CubeExponential<dim>::initialize(const bpo::variables_map& params)
{
    std::shared_ptr<exponential::DirichletIn<dim>> dirichlet1;
    if (params.count("case_params_file")) {
        const auto case_params = exponential::read_parameters<dim>(
            params["case_params_file"].as<std::string>());
        dirichlet1 =
            std::make_shared<exponential::DirichletIn<dim>>(case_params);

        std::cout << "Case 'cube_exponential': read parameters:\n";
        for (std::size_t index = 0; index < case_params.centers.size();
             ++index) {
            const auto& center = case_params.centers[index];
            std::cout << "  Center " << index << ": (";
            for (int coordinate = 0; coordinate < dim; ++coordinate) {
                if (coordinate > 0) {
                    std::cout << ", ";
                }
                std::cout << center.coordinates[coordinate];
            }
            std::cout << "), coeff = " << center.coefficient
                      << ", width = " << center.width << std::endl;
        }
    } else {
        dirichlet1 = std::make_shared<exponential::DirichletIn<dim>>();
        std::cout << "Case 'cube_exponential': default parameters.\n";
    }

    this->rhs_ = std::make_shared<exponential::RightHandSide<dim>>();

    auto dirichlet2 =
        std::make_shared<exponential::DirichletConstant<dim>>(1.0);

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

template class CubeExponential<2>;
template class CubeExponential<3>;

} // namespace cube
} // namespace cases
} // namespace paramsim
