#ifndef CONVDIFF_HDG_CMD_PARSER_HPP_
#define CONVDIFF_HDG_CMD_PARSER_HPP_

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

namespace convdiff_hdg {

namespace bpo = boost::program_options;

void add_common_options(bpo::options_description& desc, std::string help_msg);

bpo::variables_map get_cmd_args(int argc, const char *const argv[],
                                const bpo::options_description& desc);

}

#endif
