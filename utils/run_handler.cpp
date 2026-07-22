#include "run_handler.hpp"

#include <stdexcept>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>

#include "cmdparser.hpp"

namespace paramsim {

template <int dim>
RunData<dim> get_run_data(const int nargs, const char *const argv[])
{
    bpo::options_description common_desc
        ("Solves one problem given one set of parameters.");
    add_common_options(common_desc, "help!");
    const bpo::variables_map common_cmdmap = get_cmd_args(nargs, argv, common_desc);
    const auto common_params = get_common_params(common_cmdmap);
    if(common_params.dimension != dim) {
        throw std::invalid_argument(
            "Requested dimension does not match get_run_data template dimension.");
    }
    std::shared_ptr<const Case<dim>> tcase = create_case<dim>(common_params, nargs, argv);
    PDEParams pde_params;
    pde_params.pde_solver = common_params.solver_str;
    pde_params.fe_degree = common_params.fe_degree,
    pde_params.initial_resolution = common_params.initial_resolution;
    pde_params.refine_levels = common_params.refine_levels;
    pde_params.is_adaptive = common_params.is_adaptive;
    pde_params.output_path = common_params.outpath;
    SolverParams solver_params;
    solver_params.tolerance = common_params.tolerance;
    solver_params.max_its = common_params.max_outer_its;
    return RunData<dim>{pde_params, solver_params, tcase};
}

template RunData<2> get_run_data(int, const char *const[]);
template RunData<3> get_run_data(int, const char *const[]);

}
