#include "cmdparser.hpp"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/parsers.hpp>

namespace paramsim {


void add_common_options(bpo::options_description& desc, const std::string help_msg)
{
    desc.add_options()
        ("help", help_msg.c_str())
        ("case", bpo::value<std::string>(),
         "Name of the PDE case to solve: 'poisson_verify', 'poisson_bc_exp', 'convdiff_step51'")
        ("solver", bpo::value<std::string>(),
         "Type of PDE solver to use: 'poisson_cg', 'convdiff_hdg'")
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
        ("output_prefix,o", bpo::value<std::string>()->default_value("./case-"),
         "Output location along with filename prefix for vtk output")
        ;
}

bpo::variables_map get_cmd_args(const int argc, const char *const argv[],
                               const bpo::options_description& desc)
{
	bpo::variables_map cmdvarmap;
	bpo::parsed_options parsedopts =
		bpo::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
	bpo::store(parsedopts, cmdvarmap);
	bpo::notify(cmdvarmap);

	return cmdvarmap;
}

CommonParams get_common_params(const bpo::variables_map& common_cmdmap)
{
    // TODO: Use C++20 designated initializers for this
    return CommonParams {
        /*case_str =          */ common_cmdmap["case"].as<std::string>(),
        /*solver_str =        */ common_cmdmap["solver"].as<std::string>(),
        /*refine_levels =     */ common_cmdmap["refine_levels"].as<int>(),
        /*initial_resolution =*/ common_cmdmap["initial_resolution"].as<unsigned>(),
        /*fe_degree =         */ common_cmdmap["fe_degree"].as<int>(),
        /*outpath =           */ common_cmdmap["output_prefix"].as<std::string>(),
        /*is_adaptive =       */ common_cmdmap["is_adaptive"].as<bool>(),
        /*tolerance =         */ common_cmdmap["tolerance"].as<double>(),
        /*max_its =           */ common_cmdmap["max_its"].as<int>()
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
