#ifndef CONVDIFF_HDG_CMD_PARSER_HPP_
#define CONVDIFF_HDG_CMD_PARSER_HPP_

namespace convdiff_hdg {

struct EnsembleParams {
    int refine_levels;
    unsigned int initial_resolution;
};

EnsembleParams parse_cmd_options_ensemble(int argc, const char *const argv[]);

}

#endif
