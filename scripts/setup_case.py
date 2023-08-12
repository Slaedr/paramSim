import json
import numpy as np

def setup_case(case_data : dict, gen_specs : dict, sim_specs : dict):
    """ Depending on the case type, adds case-specific ensemble run parameters to
        libensemble dicts.

        @param[in] case_data  Dict of ensemble options supplied in the ensemble settings JSON.
        @param[in,out] gen_specs  Parameter bounds are populated in this libEnsemble dict.
    """
    if case_data["case_type"] == "poisson_bc_exp":
        ncenters = 3
        ndim = 1
        # Output of generator include 2 centers, each with x-coord, y-coord and coefficient
        gen_specs["out"].append( ("centers", np.float32, (ncenters, ndim+1)) )
        # ..and width of the hills
        gen_specs["out"].append( ("width", np.float32, (1,)) )
        sim_specs["in"].append("centers")
        sim_specs["in"].append("width")

        cparams = case_data["centers"]
        l_cbounds = np.zeros((ncenters,ndim+1), dtype=np.float32)
        u_cbounds = np.zeros((ncenters,ndim+1), dtype=np.float32)
        for ic in range(ncenters):
            l_cbounds[ic][0] = cparams[ic]["coords_bounds"][0]
            l_cbounds[ic][1] = cparams[ic]["coeff_bounds"][0]
            u_cbounds[ic][0] = cparams[ic]["coords_bounds"][1]
            u_cbounds[ic][1] = cparams[ic]["coeff_bounds"][1]
        gen_specs["user"]["lower"]["centers"] = l_cbounds
        gen_specs["user"]["upper"]["centers"] = u_cbounds
        
        #l_wbound = np.array([case_data["width_bounds"][0]], dtype=np.float32)
        #u_wbound = np.array([case_data["width_bounds"][1]], dtype=np.float32)
        l_wbound = case_data["width_bounds"][0]
        u_wbound = case_data["width_bounds"][1]
        gen_specs["user"]["lower"]["width"] = np.array([l_wbound], dtype=np.float32)
        gen_specs["user"]["upper"]["width"] = np.array([u_wbound], dtype=np.float32)
    else:
        raise "Invalid case type!"

def get_args_str(case_type : str, args):
    """ Given a dictionaries of arguments, generates a command line argument string for `run_case`.

        @param case_type  Paramsim case type.
        @param args  Arguments to `run_case`.
    """
    argstr = " "
    if case_type == "poisson_bc_exp":
        ncenters = 3
        for icenter in range(ncenters):
            argstr += " --center" + str(icenter) + "_y=" + str(args["centers"][icenter][0])
            argstr += " --center" + str(icenter) + "_coeff=" + str(args["centers"][icenter][1])
        argstr += " --width=" + str(args["width"][0])
    else:
        raise "Invalid case type!"

    return argstr
