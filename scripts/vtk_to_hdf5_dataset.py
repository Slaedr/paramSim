import os
import logging

import numpy as np
import meshio as mio
import h5py

logger = logging.getLogger(__name__)

DIM_LABELS = ('x', 'y', 'z', 'u', 'v', 'w')

def get_num_components_and_names(mesh_data : dict):
    """ Get the number of physical variable components in meshio
        point or cell data object, as well as their names.

    Scalar and vector fields are supported (not higher dimensional tensors).
    Scalar fields use their names as is. Vector fields use the name appended
    '_x', '_y', '_z' etc. for the different sub-components.
    """
    num_comps = 0
    names = []
    for name, data in mesh_data.items():
        num_comps += (1 if data.ndim == 1 else data.shape[1])
        if data.ndim == 1:
            names.append(name)
        else:
            for j in range(data.shape[1]):
                names.append(name + "_" + DIM_LABELS[j])
    return num_comps, names

def get_cartesian_meshgrid_shape(coords : np.ndarray, ndim : int, boundary_coord=-1.0):
    """ Computes the N-dimensional shape of a grid given the list of points.

    @param coords  Numpy structured array with coordinates labelled "x", "y", "z".
    @param ndim  Number of spatial dimensions.
    @boundary_coord  Common boundary coordinate of all spatial dimensions.

    @return Tuple of meshgrid shape [n_z, n_y, n_x].
    """
    npoin = coords.shape[0]
    if ndim == 1:
        return (npoin,)

    meshgrid = [0,]*ndim
    for i in range(ndim):
        mask = np.ones_like(npoin, dtype=bool)
        for j in range(1,ndim):
            mask = np.logical_and(mask,
                                  (coords[DIM_LABELS[(i+j)%ndim]] == boundary_coord))
        meshgrid[i] = np.count_nonzero(mask)
    check_npoin = 1
    for i in range(ndim):
        check_npoin *= meshgrid[i]
    assert(npoin == check_npoin)
    # Put x as the fastest changing coordinate
    return tuple(meshgrid[::-1])

def spatial_sort(data : np.ndarray, meshgrid_shape):
    """ Sort along outermost dimesion (z in 3D) globally, followed by y
    in each xy plane, followed by x, the innermost dimension.

    @param data  Structured array with the first ndim columns containing
                 spatial coordinates.
    @param meshgrid_shape  N-dimensional shape of the spatial grid.
    """
    npoin = data.shape[0]
    ndim = len(meshgrid_shape)
    dtypel = data.dtype.descr
    # Sort globally along the outermost dimension
    data.sort(order=dtypel[ndim-1][0])
    if ndim == 3:
        _, npoin_y, npoin_x = meshgrid_shape
        # Order z-planes by y
        for j in range(0,npoin, npoin_x*npoin_y):
            data[j:j+npoin_x*npoin_y].sort(order=dtypel[1][0])
            # Sort y-slices along x
            for i in range(0,npoin_x*npoin_y, npoin_x):
                data[j+i:j+i+npoin_x].sort(order=dtypel[0][0])
    elif ndim == 2:
        _, npoin_x = meshgrid_shape
        # Sort y-slices along x
        for j in range(0,npoin, npoin_x):
            data[j:j+npoin_x].sort(order=dtypel[0][0])
    else:
        raise ValueError("Unsupported dimensions: " + str(ndim))

def reshape_list_to_meshgrid(datalist, nvars, meshgrid_shape):
    ndim = len(meshgrid_shape)
    solution = datalist[:, ndim:]
    points = datalist[:, :ndim]
    dataset = np.moveaxis(solution.reshape(meshgrid_shape + (nvars,)), -1, 0)
    meshpoints = np.moveaxis(points.reshape(meshgrid_shape + (ndim,)), -1, 0)
    return dataset, meshpoints

def cartesian_domain_sort(mesh : mio.Mesh, ndim : int):
    """ Sort cells assuming structured grid and get the data as a structured grid.

    Note that points are sorted such that the x-coordinate is the fastest-changing
    coordinate, while the different physical variables are the outermost index.
    Eg., `data[physical_var_idx, z_idx, y_idx, x_idx]`

    @param mesh  MeshIO mesh containing volume solution values as point data.
    @param ndim  Number of spatial dimensions to use.
    @return soln  Meshgrid representing the grid of solution values in the domain.
        Since numpy arrays are row-major by default, this means the last dimension
        of the this array corresponds to the solution at different x-locations.
    @return points  Meshgrid of the three 3D array representing the mesh points
        with 3 spatial coordinates. Sorted in the same order as the data.
    """
    points = mesh.points
    nvars, varnames = get_num_components_and_names(mesh.point_data)
    datau = np.zeros((points.shape[0], ndim + nvars), dtype=np.float32)
    datau[:,:ndim] = points[:,:ndim]
    ivar = 0
    for _, u_vals in mesh.point_data.items():
        assert(points.shape[0] == u_vals.shape[0])
        nuvars = (1 if u_vals.ndim == 1 else u_vals.shape[1])
        datau[:, ndim+ivar:ndim+ivar+nuvars] = u_vals
        ivar += nuvars
    assert(ivar == nvars)
    dtypes = [*[(label, np.float32) for label in DIM_LABELS[:ndim]],
              *[(name, np.float32) for name in varnames]]
    #data = np.array([tuple(i) for i in datau], dtype = dtypes)
    datas = datau.view(dtypes).reshape(-1)
    assert(datas.shape[0] == points.shape[0])
    assert(len(datas[0]) == ndim+nvars)

    datas = np.unique(datas, axis=0)

    meshgrid_shape = get_cartesian_meshgrid_shape(datas, ndim)

    spatial_sort(datas, meshgrid_shape)

    datau = datas.view((np.float32, len(dtypes)))
    return reshape_list_to_meshgrid(datau, nvars, meshgrid_shape)

def reshape_meshgrid_to_list(values_tensor, coords_tensor):
    """
    @param values_tensor  Physical variable values, shape (nvars, dim_w, ..., dim_x)
    @param coords_tensor  Mesh point coordinates, shape (ndim, dim_w, ..., dim_x)

    @return  List of points with coords and solutions, shape (npoin, ndim + nvars)
    """
    ndim = coords_tensor.shape[0]
    nvars = values_tensor.shape[0]

    # Move the leading axis to the back, then flatten the spatial axes.
    # C-order flattening means dim_x (last spatial axis) varies fastest,
    # which is the standard convention for meshgrid(..., indexing='ij').
    coords_flat = np.moveaxis(coords_tensor, 0, -1).reshape(-1, ndim)
    values_flat = np.moveaxis(values_tensor, 0, -1).reshape(-1, nvars)

    return np.hstack([coords_flat, values_flat])

def ensemble_dir_to_hdf5(path, ndim : int, outpath) -> None:
    """ Reads ensemble of solutions from a directory and writes to HDF5.

    Assumes all the samples come from the same physical mesh.
    """
    nsamples = len([name for name in os.listdir(path)])
    samples = []
    for dire in sorted(os.listdir(path)):
        #sample_dict = {}
        for filename in os.listdir(os.path.join(path, dire)):
            filepath = os.path.join(path,dire,filename)
            if not os.path.isfile(filepath):
                #logger.error("This is not a file!")
                continue
            if not (("vtk" in filepath) or ("vtu" in filepath)):
                continue
            samples.append(mio.read(filepath))
            break
    assert(len(samples) == nsamples)
    _, points = cartesian_domain_sort(samples[0], ndim)
    domain_data = []
    for sample in samples:
        domain_data.append(cartesian_domain_sort(sample, ndim)[0])
    logger.info(" Finished loading data from " + path)
    logger.info(f" Writing data to HDF5 file {outpath}.")
    hfile = h5py.File(outpath, 'w')
    hfile.create_dataset("mesh", data=points)
    for dirid, fields in enumerate(domain_data):
        grp = hfile.create_group("sample" + str(dirid))
        grp.create_dataset("fields", data=fields)
    hfile.close()
