#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../utils/cmdparser.hpp"

namespace ps = paramsim;

namespace {

ps::CommonParams common_params(const std::vector<std::string>& extra = {})
{
    std::vector<std::string> args{"prog", "--case", "poisson_verify", "--pde",
                                  "poisson_cg", "--dimension", "2",
                                  "--initial_resolution", "2"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<const char *> argv;
    for (const auto& arg : args) {
        argv.push_back(arg.c_str());
    }
    ps::bpo::options_description desc;
    ps::add_common_options(desc, "");
    return ps::get_common_params(ps::get_cmd_args(argv.size(), argv.data(), desc));
}

std::map<std::string, std::string> settings(const std::string& text)
{
    std::map<std::string, std::string> result;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        const auto separator = line.find(" = ");
        if (separator != std::string::npos) {
            EXPECT_TRUE(result.emplace(line.substr(2, separator - 2),
                                        line.substr(separator + 3)).second)
                << "Duplicate setting: " << line;
        }
    }
    return result;
}

} // namespace

TEST(Configuration, IncludesEveryCommonSettingAndPreservesStreamFormat)
{
    const auto params = common_params();
    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << std::noboolalpha;
    const auto flags = output.flags();
    ps::print_configuration(output, params, 3);
    EXPECT_EQ(output.flags(), flags);
    EXPECT_EQ(output.precision(), 2);
    const auto values = settings(output.str());
    for (const auto *key : {"case", "pde", "dimension", "case_params_file",
                            "initial_resolution", "fe_degree", "refine_levels",
                            "coarse_output_levels", "coarse_grid_solution",
                            "is_adaptive", "tolerance", "max_its", "output_prefix",
                            "init_pde", "resolved_init_pde", "initialization_enabled",
                            "init_tolerance", "init_max_its", "thread_limit"}) {
        EXPECT_EQ(values.count(key), 1) << key;
    }
    EXPECT_EQ(values.at("case_params_file"), "none");
    EXPECT_EQ(values.at("is_adaptive"), "false");
    EXPECT_EQ(values.at("coarse_grid_solution"), "interpolation");
    EXPECT_EQ(values.at("thread_limit"), "3");
    EXPECT_DOUBLE_EQ(std::stod(values.at("tolerance")), params.tolerance);
    EXPECT_DOUBLE_EQ(std::stod(values.at("init_tolerance")), params.init_tolerance);
    output << 1.234 << ' ' << true;
    EXPECT_EQ(output.str().substr(output.str().size() - 6), "1.23 1");
}

TEST(Configuration, ReportsOverridesAndInitializationBehavior)
{
    auto params = common_params({"--refine_levels", "3", "--fe_degree", "2",
        "--coarse_output_levels", "1", "--coarse_grid_solution", "native_simulation",
        "--is_adaptive", "true", "--max_its", "42", "--tolerance", "0.12345678901234566",
        "--init_tolerance", "0.25", "--init_max_its", "7", "-o", "some prefix",
        "--case_params_file", "some parameters.txt"});
    std::ostringstream output;
    ps::print_configuration(output, params, 1);
    const auto values = settings(output.str());
    EXPECT_EQ(values.at("fe_degree"), "2");
    EXPECT_EQ(values.at("refine_levels"), "3");
    EXPECT_EQ(values.at("coarse_output_levels"), "1");
    EXPECT_EQ(values.at("coarse_grid_solution"), "native_simulation");
    EXPECT_EQ(values.at("is_adaptive"), "true");
    EXPECT_EQ(values.at("max_its"), "42");
    EXPECT_EQ(values.at("init_max_its"), "7");
    EXPECT_EQ(values.at("output_prefix"), "some prefix");
    EXPECT_EQ(values.at("case_params_file"), "some parameters.txt");
    EXPECT_DOUBLE_EQ(std::stod(values.at("tolerance")), params.tolerance);
    EXPECT_EQ(values.at("init_tolerance"), "0.25");

    struct InitCase { std::string pde, request, resolved, enabled; };
    for (const auto& init : std::vector<InitCase>{
             {"poisson_cg", "", "none", "false"},
             {"minimal_surface", "", "poisson_cg", "true"},
             {"minimal_surface", "none", "none", "false"},
             {"minimal_surface", "minimal_surface", "minimal_surface", "false"},
             {"gelfand", "poisson_cg", "poisson_cg", "true"}}) {
        params.solver_str = init.pde;
        params.init_pde_str = init.request;
        std::ostringstream stream;
        ps::print_configuration(stream, params, 1);
        const auto actual = settings(stream.str());
        EXPECT_EQ(actual.at("resolved_init_pde"), init.resolved);
        EXPECT_EQ(actual.at("initialization_enabled"), init.enabled);
        EXPECT_EQ(actual.at("init_pde"), init.request.empty() ? "(default)" : init.request);
    }
}
