import json
import numpy as np

#TODO: Replace the if-blocks in this file with a set of classes

def setup_case(case_data : dict, gen_specs : dict, sim_specs : dict):
    """ Depending on the case type, adds case-specific ensemble run parameters to
        libensemble dicts.
        
        For the case poisson_bc_exp, this needs an array "centers" of length 3, each having
        dict "coords_bounds" (lower and upper bounds for y-coordinates, so array of length 2),
        dict "coeff_bounds" (lower and upper bounds for coefficients, so array of length 2).
        In addition, a key "width_bounds" with a 2-array as value, having lower and upper bounds
        for the width of each exponential hill.

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
        
        l_wbound = case_data["width_bounds"][0]
        u_wbound = case_data["width_bounds"][1]
        gen_specs["user"]["lower"]["width"] = np.array([l_wbound], dtype=np.float32)
        gen_specs["user"]["upper"]["width"] = np.array([u_wbound], dtype=np.float32)

    elif case_data["case_type"] == "poisson_bc_polynomial":
        nterms = 4
        ndim = 1
        gen_specs["out"].append( ("coeffs", np.float32, (nterms,)) )
        gen_specs["out"].append( ("center_y", np.float32, (1,)) )
        sim_specs["in"].append("coeffs")
        sim_specs["in"].append("center_y")

        cparams = case_data["coeffs_bounds"]
        l_cbounds = np.zeros((nterms,), dtype=np.float32)
        u_cbounds = np.zeros((nterms,), dtype=np.float32)
        for it in range(nterms):
            l_cbounds[it] = cparams[it][0]
            u_cbounds[it] = cparams[it][1]
        gen_specs["user"]["lower"]["coeffs"] = l_cbounds
        gen_specs["user"]["upper"]["coeffs"] = u_cbounds

        l_ybound = case_data["center_y_bounds"][0]
        u_ybound = case_data["center_y_bounds"][1]
        gen_specs["user"]["lower"]["center_y"] = np.array([l_ybound], dtype=np.float32)
        gen_specs["user"]["upper"]["center_y"] = np.array([u_ybound], dtype=np.float32)

    elif case_data["case_type"] == "poisson_bc_fourier":
        nmodes = 2
        ndim = 1
        # Output of generator include 2 centers, each with x-coord, y-coord and coefficient
        gen_specs["out"].append( ("amplitudes", np.float32, (nmodes, 2)) )
        # ..and width of the hills
        gen_specs["out"].append( ("constant", np.float32, (1,)) )
        gen_specs["out"].append( ("wavelength", np.float32, (1,)) )
        sim_specs["in"].append("amplitudes")
        sim_specs["in"].append("constant")
        sim_specs["in"].append("wavelength")

        cparams = case_data["amplitudes"]
        l_ampbounds = np.zeros((nmodes,2), dtype=np.float32)
        u_ampbounds = np.zeros((nmodes,2), dtype=np.float32)
        for ic in range(nmodes):
            l_ampbounds[ic,0] = cparams[ic]["a_bounds"][0]
            l_ampbounds[ic,1] = cparams[ic]["b_bounds"][0]
            u_ampbounds[ic,0] = cparams[ic]["a_bounds"][1]
            u_ampbounds[ic,1] = cparams[ic]["b_bounds"][1]
        gen_specs["user"]["lower"]["amplitudes"] = l_ampbounds
        gen_specs["user"]["upper"]["amplitudes"] = u_ampbounds

        l_cbound = case_data["constant_bounds"][0]
        u_cbound = case_data["constant_bounds"][1]
        gen_specs["user"]["lower"]["constant"] = np.array([l_cbound], dtype=np.float32)
        gen_specs["user"]["upper"]["constant"] = np.array([u_cbound], dtype=np.float32)

        l_wbound = case_data["wavelength_bounds"][0]
        u_wbound = case_data["wavelength_bounds"][1]
        gen_specs["user"]["lower"]["wavelength"] = np.array([l_wbound], dtype=np.float32)
        gen_specs["user"]["upper"]["wavelength"] = np.array([u_wbound], dtype=np.float32)
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
    elif case_type == "poisson_bc_polynomial":
        nterms = 4
        for it in range(nterms):
            argstr += " --a" + str(it) + "=" + str(args["coeffs"][it])
        argstr += " --center_y=" + str(args["center_y"][0])
    elif case_type == "poisson_bc_fourier":
        nmodes = 2
        argstr += " --wavelength=" + str(args["wavelength"][0]) + " --a0=" + \
            str(args["constant"][0])
        for im in range(nmodes):
            argstr += " --a" + str(im+1) + "=" + str(args["amplitudes"][im][0])
            argstr += " --b" + str(im+1) + "=" + str(args["amplitudes"][im][1])
    else:
        raise "Invalid case type!"

    return argstr
