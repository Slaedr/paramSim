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
        const auto values = reader.read_finite_values(
            5, "center " + std::to_string(index));

        for (std::size_t coordinate = 0; coordinate < 3; ++coordinate) {
            if (values[coordinate] < -1.0 ||
                values[coordinate] > 1.0) {
                reader.fail(index + 2,
                            "center coordinates must lie in [-1, 1]");
            }
        }
        if (values[4] <= 0.0) {
            reader.fail(index + 2, "center width must be positive");
        }

        GaussianCenter<dim> center;
        for (std::size_t coordinate = 0; coordinate < dim; ++coordinate) {
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
    if (params.count("width")) {
        constexpr std::size_t legacy_center_count = 3;
        const double width = params["width"].as<double>();
        exponential::Params<dim> case_params;
        case_params.centers.clear();
        for (std::size_t index = 0; index < legacy_center_count; ++index) {
            const std::string flag =
                "center" + std::to_string(index) + "_y";
            const std::string coefficient_flag =
                "center" + std::to_string(index) + "_coeff";
            exponential::GaussianCenter<dim> center;
            center.coordinates[0] = -1.0;
            center.coordinates[1] = params[flag].as<double>();
            center.coefficient = params[coefficient_flag].as<double>();
            center.width = width;
            case_params.centers.push_back(center);
        }
        this->rhs_ = std::make_shared<exponential::RightHandSide<dim>>();
        dirichlet1 =
            std::make_shared<exponential::DirichletIn<dim>>(case_params);

        // Write out params to confirm
        std::cout << "Case 'cube_exponential': read parameters:\n";
        for (std::size_t index = 0; index < case_params.centers.size();
             ++index) {
            const auto& center = case_params.centers[index];
            std::cout << "  Center " << index << ": (";
            for (const double coordinate : center.coordinates) {
                std::cout << coordinate << ", ";
            }
            std::cout << "), coeff = " << center.coefficient
                      << ", width = " << center.width << std::endl;
        }
    } else {
        this->rhs_ = std::make_shared<exponential::RightHandSide<dim>>();
        dirichlet1 = std::make_shared<exponential::DirichletIn<dim>>();
        std::cout << "Case 'cube_exponential': default parameters.\n";
    }

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

template <int dim>
void CubeExponential<dim>::add_case_cmd_args(
    bpo::options_description& desc) const
{
    desc.add_options()("width", bpo::value<double>(), "Width of each hill");
    constexpr int legacy_center_count = 3;
    for (int ic = 0; ic < legacy_center_count; ic++) {
        const std::string flag =
            std::string("center") + std::to_string(ic) + "_y";
        const std::string descstr =
            "y coordinate of " + std::to_string(ic) + "th center";
        desc.add_options()(flag.c_str(), bpo::value<double>(), descstr.c_str());
        // eg. centers[0][1] = params["center0_y"].as<double>();
        const std::string coflag =
            std::string("center") + std::to_string(ic) + "_coeff";
        const std::string codescstr =
            "Coefficient multiplying the " + std::to_string(ic) + "th center";
        // eg. "center1_coeff"
        desc.add_options()(coflag.c_str(), bpo::value<double>(),
                           codescstr.c_str());
    }
}

template class CubeExponential<2>;
template class CubeExponential<3>;

} // namespace cube
} // namespace cases
} // namespace paramsim
