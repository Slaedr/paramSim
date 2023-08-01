#include "cmdparser.hpp"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/parsers.hpp>

namespace convdiff_hdg {


void add_common_options(bpo::options_description& desc, const std::string help_msg)
{
	desc.add_options()
		("help", help_msg.c_str())
        ("refine_levels", bpo::value<int>()->default_value(5),
         "Number of times to refine the grid and solve")
        ("initial_resolution", bpo::value<unsigned int>()->default_value(2),
         "Number of cells in first grid")
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

}
