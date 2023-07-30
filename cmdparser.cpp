#include "cmdparser.hpp"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/parsers.hpp>

namespace convdiff_hdg {

namespace po = boost::program_options;

po::variables_map ensemble_cmd_options(const int argc, const char *const argv[],
                                    po::options_description& desc, const std::string help_msg)
{
	desc.add_options()
		("help", help_msg.c_str())
        ("refine_levels", po::value<int>()->default_value(5),
         "Number of times to refine the grid and solve")
        ("initial_resolution", po::value<unsigned int>()->default_value(2),
         "Number of cells in first grid")
        ;

	po::variables_map cmdvarmap;
	po::parsed_options parsedopts =
		po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
	po::store(parsedopts, cmdvarmap);
	po::notify(cmdvarmap);

	return cmdvarmap;
}

EnsembleParams parse_cmd_options_ensemble(const int argc, const char *const argv[])
{
    EnsembleParams params;
    po::options_description desc
        ("Solves Convection-Diffusion problems under a collection of boundary value instances.");
    const auto map = ensemble_cmd_options(argc, argv, desc, "");
    params.refine_levels = map["refine_levels"].as<int>();
    params.initial_resolution = map["initial_resolution"].as<unsigned int>();
    return params;
}

}
