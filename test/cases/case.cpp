#include <cstdlib>

#include <gtest/gtest.h>

#include "../../utils/cmdparser.hpp"
#include "../../cases/case.hpp"
#include "../../cases/minimal_surface/minimal_surface.hpp"

namespace ps = paramsim;
namespace bpo = boost::program_options;

TEST(Cases, CanCreateDefaultMinSurfCubeSinusoidalCase)
{
    const int nargs = 11;
    bpo::options_description common_desc
        ("Solves one problem given one set of parameters.");
    ps::add_common_options(common_desc, "help!");
    char args[][100] = {"prog", "--pde", "minimal_surface", "--refine_levels", "4",
        "--case", "minimal_surface_cube_sinusoidal", "--fe_degree", "2", "--max_its", "100"};
    const char **argv = static_cast<const char**>(std::malloc(nargs*sizeof(char**)));
    for(int i = 0; i < nargs; i++) {
        argv[i] = args[i];
    }
    const bpo::variables_map common_cmdmap = ps::get_cmd_args(nargs, argv, common_desc);
    const auto cparams = ps::get_common_params(common_cmdmap);

    std::shared_ptr<ps::Case<2>> case1 = ps::create_case<2>(cparams, nargs, argv);

    EXPECT_TRUE(std::dynamic_pointer_cast<ps::cases::MinSurfCubeSinusoidal<2>>(case1));
    std::free(argv);
}
