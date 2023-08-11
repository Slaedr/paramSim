import json
import numpy as np

def setup_case(case_data : dict, gen_specs : dict, sim_specs : dict):
    """ Depending on the case type, adds case-specific ensemble run parameters to
        libensemble dicts.

        @param[in] case_data  Dict of ensemble options supplied in the ensemble settings JSON.
        @param[in,out] gen_specs  Parameter bounds are populated in this libEnsemble dict.
    """
    if case_data["case_type"] == "poisson_bc_exp":
        ncenters = 2
        ndim = 2
        # Output of generator include 2 centers, each with x-coord, y-coord and coefficient
        gen_specs["out"].append( ("centers", np.float32, (ncenters, ndim+1)) )
        # ..and width of the hills
        gen_specs["out"].append( ("width", np.float32, (1,)) )
        sim_specs["in"].append("centers")
        sim_specs["in"].append("width")

        cparams = case_data["centers"]
        l_cbounds = \
        np.array([[ cparams[0]["coords_bounds"][0][0], cparams[0]["coords_bounds"][1][0],
            cparams[0]["coeff_bounds"][0] ],
            [ cparams[1]["coords_bounds"][0][0], cparams[1]["coords_bounds"][1][0],
                cparams[1]["coeff_bounds"][0] ]],
                dtype=np.float32)
        u_cbounds = \
        np.array([[ cparams[0]["coords_bounds"][0][1], cparams[0]["coords_bounds"][1][1],
            cparams[0]["coeff_bounds"][1] ],
            [ cparams[1]["coords_bounds"][0][1], cparams[1]["coords_bounds"][1][1],
                cparams[1]["coeff_bounds"][1] ]],
                dtype=np.float32)
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

def get_args_str(case_type : str, args : dict):
    """ Given a dictionaries of arguments, generates a command line argument string for `run_case`.

        @param case_type  Paramsim case type.
        @param args  Dict of arrays of arguments to `run_case`.
    """
    argstr = " "
    if case_type == "poisson_bc_exp":
        ncenters = 2
        ndim = 2
        dims = ('x', 'y', 'z')
        for icenter in range(ncenters):
            for idim in range(ndim):
                argstr += " --center" + str(icenter) + "_" + dims[idim] + " " \
                    + str(args["centers"][icenter][idim])
            print("Shape of centers array is " + str(args["centers"].shape))
            argstr += " --center" + str(icenter) + "_coeff " + str(args["centers"][icenter][ndim])
        argstr += " --width " + str(args["width"][0])
    else:
        raise "Invalid case type!"

    return argstr
