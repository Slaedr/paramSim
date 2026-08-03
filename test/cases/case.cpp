#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../cases/case.hpp"
#include "../../cases/cube/exponential.hpp"
#include "../../cases/cube/fourier.hpp"
#include "../../cases/cube/polynomial.hpp"
#include "../../cases/minimal_surface/minimal_surface.hpp"
#include "../../utils/cmdparser.hpp"
#include "../utils/temporary_parameter_file.hpp"

namespace ps = paramsim;
namespace bpo = boost::program_options;
namespace cube = ps::cases::cube;

namespace {
using paramsim::test::TemporaryParameterFile;

template <int dim>
std::unique_ptr<ps::Case<dim>>
create_cube_case(const std::string& case_name,
                 const std::string *parameter_file = nullptr)
{
    ps::CommonParams params{};
    params.case_str = case_name;
    if (parameter_file != nullptr) {
        const char *args[] = {"prog", "--case", case_name.c_str(),
                              "--case_params_file", parameter_file->c_str()};
        return ps::create_case<dim>(params, 5, args);
    }

    const char *args[] = {"prog"};
    return ps::create_case<dim>(params, 1, args);
}

} // namespace

TEST(Cases, CanCreateDefaultMinSurfDiskSinusoidalCase)
{
    const int nargs = 13;
    bpo::options_description common_desc(
        "Solves one problem given one set of parameters.");
    ps::add_common_options(common_desc, "help!");
    char args[][100] = {"prog",
                        "--dimension",
                        "2",
                        "--pde",
                        "minimal_surface",
                        "--refine_levels",
                        "4",
                        "--case",
                        "minimal_surface_disk_sinusoidal",
                        "--fe_degree",
                        "2",
                        "--max_its",
                        "100"};
    const char **argv =
        static_cast<const char **>(std::malloc(nargs * sizeof(char **)));
    for (int i = 0; i < nargs; i++) {
        argv[i] = args[i];
    }
    const bpo::variables_map common_cmdmap =
        ps::get_cmd_args(nargs, argv, common_desc);
    const auto cparams = ps::get_common_params(common_cmdmap);

    std::shared_ptr<ps::Case<2>> case1 =
        ps::create_case<2>(cparams, nargs, argv);

    EXPECT_TRUE(
        std::dynamic_pointer_cast<ps::cases::MinSurfDiskSinusoidal<2>>(case1));
    std::free(argv);
}

TEST(CommonParams, AcceptsSupportedDimensions)
{
    bpo::options_description common_desc("Common options");
    ps::add_common_options(common_desc, "help!");
    const char *args2[] = {"prog",          "--dimension", "2",
                           "--pde",         "poisson_cg",  "--case",
                           "poisson_verify"};
    const char *args3[] = {"prog",          "--dimension", "3",
                           "--pde",         "poisson_cg",  "--case",
                           "poisson_verify"};

    const auto params2 =
        ps::get_common_params(ps::get_cmd_args(7, args2, common_desc));
    const auto params3 =
        ps::get_common_params(ps::get_cmd_args(7, args3, common_desc));

    EXPECT_EQ(params2.dimension, 2);
    EXPECT_EQ(params3.dimension, 3);
}

TEST(CommonParams, RejectsMissingDimension)
{
    bpo::options_description common_desc("Common options");
    ps::add_common_options(common_desc, "help!");
    const char *args[] = {"prog", "--pde", "poisson_cg", "--case",
                          "poisson_verify"};
    const auto common_cmdmap = ps::get_cmd_args(5, args, common_desc);

    EXPECT_THROW(ps::get_common_params(common_cmdmap), std::invalid_argument);
}

TEST(CommonParams, RejectsUnsupportedDimension)
{
    bpo::options_description common_desc("Common options");
    ps::add_common_options(common_desc, "help!");
    const char *args[] = {"prog",   "--dimension",   "4", "--pde", "poisson_cg",
                          "--case", "poisson_verify"};
    const auto common_cmdmap = ps::get_cmd_args(7, args, common_desc);

    EXPECT_THROW(ps::get_common_params(common_cmdmap), std::invalid_argument);
}

TEST(CommonOptions, HasParameterFileOption)
{
    bpo::options_description description("Common options");
    ps::add_common_options(description, "help!");

    bool found = false;
    for (const auto& option : description.options()) {
        found = found || option->long_name() == "case_params_file";
    }
    EXPECT_TRUE(found);
}

TEST(CubeCaseParameters, UsesParameterFileForSelectedCaseAndDimension)
{
    TemporaryParameterFile exponential_file("1\n"
                                            "0 0 0 2 1\n");
    TemporaryParameterFile fourier_file("1\n"
                                        "5 2 3 4\n"
                                        "0 0 0 0 0 0 0 0\n");
    TemporaryParameterFile polynomial_file("2\n"
                                           "1 2 3\n"
                                           "1\n"
                                           "2 3 4\n");
    const std::string exponential_path = exponential_file.path();
    const std::string fourier_path = fourier_file.path();
    const std::string polynomial_path = polynomial_file.path();

    const auto exponential_case =
        create_cube_case<2>("cube_exponential", &exponential_path);
    const auto fourier_case =
        create_cube_case<3>("cube_fourier", &fourier_path);
    const auto polynomial_case =
        create_cube_case<3>("cube_polynomial", &polynomial_path);

    EXPECT_DOUBLE_EQ(exponential_case->get_dirichlet_bcs().front().bc_fn->value(
                         dealii::Point<2>{0.0, 0.0}),
                     1.0 / dealii::numbers::PI);
    EXPECT_DOUBLE_EQ(fourier_case->get_dirichlet_bcs().front().bc_fn->value(
                         dealii::Point<3>{0.0, 0.4, -0.2}),
                     5.0);
    EXPECT_DOUBLE_EQ(polynomial_case->get_dirichlet_bcs().front().bc_fn->value(
                         dealii::Point<3>{2.0, 4.0, 6.0}),
                     21.0);
}

TEST(CubeCaseParameters, UsesBuiltInDefaultsWhenOptionIsOmitted)
{
    const auto exponential_case = create_cube_case<2>("cube_exponential");
    const auto fourier_case = create_cube_case<2>("cube_fourier");
    const auto polynomial_case = create_cube_case<2>("cube_polynomial");
    const dealii::Point<2> point{0.25, -0.4};

    EXPECT_DOUBLE_EQ(
        exponential_case->get_dirichlet_bcs().front().bc_fn->value(point),
        cube::exponential::DirichletIn<2>().value(point));
    EXPECT_DOUBLE_EQ(
        fourier_case->get_dirichlet_bcs().front().bc_fn->value(point),
        cube::fourier::DirichletIn<2>().value(point));
    EXPECT_DOUBLE_EQ(
        polynomial_case->get_dirichlet_bcs().front().bc_fn->value(point),
        cube::polynomial::DirichletIn<2>().value(point));
}

TEST(CubeCaseParameters, HasExactlyOneDirichletRegion)
{
    for (const std::string case_name :
         {"cube_exponential", "cube_fourier", "cube_polynomial"}) {
        const auto case2 = create_cube_case<2>(case_name);
        const auto case3 = create_cube_case<3>(case_name);

        EXPECT_EQ(case2->get_dirichlet_bcs().size(), 1u) << case_name;
        EXPECT_EQ(case3->get_dirichlet_bcs().size(), 1u) << case_name;
    }
}

TEST(CubeCaseParameters, ProfileAppliesToEveryBoundaryFace)
{
    for (const std::string case_name :
         {"cube_exponential", "cube_fourier", "cube_polynomial"}) {
        const auto case2 = create_cube_case<2>(case_name);
        const auto bc_id = case2->get_dirichlet_bcs().front().bc_id;

        dealii::Triangulation<2> tria;
        case2->get_geometry()->generate_grid(tria, 3);
        case2->get_geometry()->set_boundary_ids(tria);

        for (const auto& cell : tria.cell_iterators()) {
            for (const auto& face : cell->face_iterators()) {
                if (!face->at_boundary()) {
                    continue;
                }
                EXPECT_EQ(face->boundary_id(), bc_id) << case_name;
            }
        }
    }
}

TEST(CubeCaseParameters, ReportsMalformedFilesForSelectedCase)
{
    TemporaryParameterFile file("0\n");
    const std::string path = file.path();

    for (const std::string case_name :
         {"cube_exponential", "cube_fourier", "cube_polynomial"}) {
        try {
            create_cube_case<2>(case_name, &path);
            FAIL() << "Expected malformed parameters for " << case_name;
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find(case_name), std::string::npos);
            EXPECT_NE(message.find(path), std::string::npos);
            EXPECT_NE(message.find("line 1"), std::string::npos);
        }
    }
}

TEST(CubeCaseParameters, RejectsUnreadableFiles)
{
    const std::string path = "/path/that/does/not/exist/case_params.txt";

    for (const std::string case_name :
         {"cube_exponential", "cube_fourier", "cube_polynomial"}) {
        EXPECT_THROW(create_cube_case<2>(case_name, &path), std::runtime_error);
    }
}
