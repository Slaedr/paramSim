#ifndef CONVDIFF_HDG_CMD_PARSER_HPP_
#define CONVDIFF_HDG_CMD_PARSER_HPP_

#include <string>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

namespace paramsim {

/** Parameters common to all PDEs and cases for one simulation run.
 *
 * If the list below is changed or reordered, get_common_params has to be changed.
 */
struct CommonParams {
    std::string case_str;
    std::string solver_str;
    unsigned int dimension;
    int refine_levels;
    unsigned initial_resolution;
    int fe_degree;
    std::string outpath;
    bool is_adaptive;
    double tolerance;
    int max_outer_its;
};

namespace bpo = boost::program_options;

void add_common_options(bpo::options_description& desc, std::string help_msg);

bpo::variables_map get_cmd_args(int argc, const char *const argv[],
                                const bpo::options_description& desc);

CommonParams get_common_params(const bpo::variables_map& common_map);

/// Convert cmd line arguments specified within the code
const char** allocate_setup_args(const int nargs, const char args[][100]);

}

#endif
