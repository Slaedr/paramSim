#include "cmdparser.hpp"

#include <stdexcept>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/parsers.hpp>

namespace paramsim {


void add_common_options(bpo::options_description& desc, const std::string help_msg)
{
    desc.add_options()
        ("help", help_msg.c_str())
        ("case", bpo::value<std::string>(),
         "Name of the PDE case to solve: 'poisson_verify', 'cube_exponential', 'cube_fourier', "
         "'cube_polynomial', 'minimal_surface_ball_verify', 'minimal_surface_cube_verify', "
         "'minimal_surface_disk_sinusoidal'")
        ("pde", bpo::value<std::string>(),
         "Type of PDE solver to use: 'poisson_cg', 'minimal_surface'")
        ("init_pde", bpo::value<std::string>()->default_value(""),
         "PDE solved first on the initial grid to initialize the main solve, using the same case. "
         "Leave empty to use the default for the chosen --pde ('minimal_surface' defaults to "
         "'poisson_cg'); set to 'none' to start the main solve from zero")
        ("init_tolerance", bpo::value<double>()->default_value(1e-4),
         "Tolerance for convergence of the initialization solve")
        ("init_max_its", bpo::value<int>()->default_value(20),
         "Maximum iterations for the initialization solve")
        ("dimension", bpo::value<unsigned int>(),
         "Spatial dimension of the problem (required; supported values: 2, 3)")
        ("case_params_file", bpo::value<std::string>(),
         "Path to the case-specific parameter file")
        ("refine_levels", bpo::value<int>()->default_value(5),
         "Number of times to refine the grid and solve")
        ("initial_resolution", bpo::value<unsigned int>()->default_value(2),
         "Number of cells in first grid")
        ("fe_degree", bpo::value<int>()->default_value(1),
         "Polynomial degree of FEM basis functions to use")
        ("is_adaptive", bpo::value<bool>()->default_value(false),
         "Set to 1 to run adaptive simulation")
        ("max_its", bpo::value<int>()->default_value(10),
         "Maximum solver iterations")
        ("tolerance", bpo::value<double>()->default_value(1e-6),
         "Tolerance for solver convergence")
        ("output_prefix,o", bpo::value<std::string>()->default_value("_"),
         "Output location along with filename prefix for vtk output")
        ;
}

bpo::variables_map get_cmd_args(const int argc, const char *const argv[],
                               const bpo::options_description& desc)
{
    const auto style =
        bpo::command_line_style::default_style &
        ~bpo::command_line_style::allow_guessing;
    const bpo::parsed_options parsed_options =
        bpo::command_line_parser(argc, argv)
            .options(desc)
            .style(style)
            .allow_unregistered()
            .run();
    bpo::variables_map cmdvarmap;
    bpo::store(parsed_options, cmdvarmap);
    bpo::notify(cmdvarmap);

    return cmdvarmap;
}

CommonParams get_common_params(const bpo::variables_map& common_cmdmap)
{
    if(!common_cmdmap.count("dimension")) {
        throw std::invalid_argument(
            "Missing required option '--dimension'. Supported values are 2 and 3.");
    }

    const unsigned int dimension =
        common_cmdmap["dimension"].as<unsigned int>();
    if(dimension != 2 && dimension != 3) {
        throw std::invalid_argument(
            "Unsupported spatial dimension " + std::to_string(dimension) +
            ". Supported values are 2 and 3.");
    }

    // TODO: Use C++20 designated initializers for this
    return CommonParams {
        /*case_str =          */ common_cmdmap["case"].as<std::string>(),
        /*solver_str =        */ common_cmdmap["pde"].as<std::string>(),
        /*dimension =         */ dimension,
        /*refine_levels =     */ common_cmdmap["refine_levels"].as<int>(),
        /*initial_resolution =*/ common_cmdmap["initial_resolution"].as<unsigned>(),
        /*fe_degree =         */ common_cmdmap["fe_degree"].as<int>(),
        /*outpath =           */ common_cmdmap["output_prefix"].as<std::string>(),
        /*is_adaptive =       */ common_cmdmap["is_adaptive"].as<bool>(),
        /*tolerance =         */ common_cmdmap["tolerance"].as<double>(),
        /*max_its =           */ common_cmdmap["max_its"].as<int>(),
        /*init_pde_str =      */ common_cmdmap["init_pde"].as<std::string>(),
        /*init_tolerance =    */ common_cmdmap["init_tolerance"].as<double>(),
        /*init_max_its =      */ common_cmdmap["init_max_its"].as<int>()
    };
}

// Convert cmd line arguments for testing
const char** allocate_setup_args(const int nargs, const char args[][100])
{
    auto argv = static_cast<const char**>(std::malloc(nargs*sizeof(char**)));
    for(int i = 0; i < nargs; i++) {
        argv[i] = args[i];
    }
    return argv;
}

}
