import os
import logging

import numpy as np
import meshio as mio
import h5py

logger = logging.getLogger(__name__)

DIM_LABELS = ('x', 'y', 'z', 'u', 'v', 'w')
DEFAULT_BATCH_SIZE = 8

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

class VTKToHDF5:
    """ Read an ensemble of solutions from a directory
        and write to HDF5.

    Assumes all the samples come from the same physical mesh.
    """
    def __init__(self, ensemble_root_path, hfile, ndim : int,
                 output_sample_count=None):
        """ Prepares to read VTK data from an ensemble tree and write to an
            HDF5 file.

        If the file does not already contain mesh and fields datasets, read
        their shapes and data types from the first input sample and create
        them.

        @param ensemble_root_path  Location of the ensemble directory.
        @param hfile  Open HDF5 file object.
        @param ndim  Number of relevant spatial dimensions.
        @param output_sample_count  Number of samples in the output fields
                                    dataset. Defaults to the number of input
                                    samples.
        """
        self.indirpath = ensemble_root_path
        self.ndim = ndim
        self.hfile = hfile
        self.sample_directories = {}

        for name in os.listdir(self.indirpath):
            directory = os.path.join(self.indirpath, name)
            if not os.path.isdir(directory) or not name.startswith("sim"):
                continue
            sample_id = name[3:]
            if not sample_id.isdigit():
                continue
            sample_index = int(sample_id)
            if sample_index in self.sample_directories:
                raise ValueError(
                    f"Multiple libEnsemble directories represent sample "
                    f"{sample_index}: {self.sample_directories[sample_index]!r} "
                    f"and {directory!r}."
                )
            self.sample_directories[sample_index] = directory

        self.nsamples = len(self.sample_directories)
        if 0 not in self.sample_directories:
            raise ValueError(
                f"No libEnsemble directory found for sample 0 "
                f"in {self.indirpath!r}."
            )

        if output_sample_count is None:
            output_sample_count = self.nsamples
        if output_sample_count < 1:
            raise ValueError("Output sample count must be greater than zero.")
        self.output_sample_count = output_sample_count

        if "mesh" not in self.hfile or "fields" not in self.hfile:
            fields, points = self._read_sample(0)
        if "mesh" not in self.hfile:
            logger.info("  Writing mesh points.")
            self.hfile.create_dataset("mesh", data=points)
        if "fields" not in self.hfile:
            logger.info("  Creating fields dataset.")
            self.hfile.create_dataset(
                "fields",
                shape=(self.output_sample_count, *fields.shape),
                dtype=fields.dtype,
            )
        elif self.hfile["fields"].shape[0] != self.output_sample_count:
            raise ValueError(
                f"Output fields dataset contains "
                f"{self.hfile['fields'].shape[0]} samples; "
                f"{self.output_sample_count} expected."
            )

    def _read_sample(self, sample_index : int):
        """Read and spatially sort one sample's fields and mesh points.

        @param sample_index  Integer sample ID to read from the ensemble tree.

        @return Tuple containing the fields and mesh points arrays.
        """
        if sample_index not in self.sample_directories:
            raise ValueError(
                f"No libEnsemble directory found for sample {sample_index} "
                f"in {self.indirpath!r}."
            )

        directory = self.sample_directories[sample_index]
        for filename in sorted(os.listdir(directory)):
            filepath = os.path.join(directory, filename)
            if not os.path.isfile(filepath):
                continue
            if not filename.lower().endswith((".vtk", ".vtu")):
                continue
            sample = mio.read(filepath)
            print("Reading sample 0 for mesh...")
            return cartesian_domain_sort(sample, self.ndim)

        raise RuntimeError(
            f"No VTK or VTU file found for sample {sample_index} "
            f"in {directory!r}."
        )

    def process_sample(self, in_sample_idx : int, out_sample_idx : int,
                       batch_size : int = 1) -> None:
        """Read and write a batch of consecutive ensemble samples.

        @param in_sample_idx  First sample index from the ensemble to read.
        @param out_sample_idx  First index to use in the output fields dataset.
        @param batch_size  Number of consecutive samples to read and write.
        """
        if batch_size < 1:
            raise ValueError("Batch size must be greater than zero.")

        input_indices = range(in_sample_idx, in_sample_idx + batch_size)
        missing_indices = [
            sample_index for sample_index in input_indices
            if sample_index not in self.sample_directories
        ]
        if missing_indices:
            raise ValueError(
                f"No libEnsemble directories found for sample IDs "
                f"{missing_indices} in {self.indirpath!r}."
            )

        output_end = out_sample_idx + batch_size
        output_fields = self.hfile["fields"]
        if out_sample_idx < 0 or output_end > output_fields.shape[0]:
            raise ValueError(
                f"Output sample range [{out_sample_idx}, {output_end}) "
                f"is outside fields dataset range [0, "
                f"{output_fields.shape[0]})."
            )

        batch_fields = []
        expected_shape = output_fields.shape[1:]
        expected_dtype = output_fields.dtype
        for sample_index in input_indices:
            fields, _ = self._read_sample(sample_index)
            if fields.shape != expected_shape:
                raise ValueError(
                    f"Fields for sample {sample_index} have shape "
                    f"{fields.shape}; expected {expected_shape}."
                )
            if fields.dtype != expected_dtype:
                raise ValueError(
                    f"Fields for sample {sample_index} have dtype "
                    f"{fields.dtype}; expected {expected_dtype}."
                )
            batch_fields.append(fields)

        output_fields[out_sample_idx:output_end] = np.stack(batch_fields)

def ensemble_dir_to_hdf5(path, ndim : int, outpath,
                         batch_size : int = DEFAULT_BATCH_SIZE) -> None:
    """Convert a libEnsemble directory tree to a batched HDF5 dataset.

    @param path  Path to the libEnsemble output directory.
    @param ndim  Number of relevant spatial dimensions.
    @param outpath  Path of the HDF5 file to create.
    @param batch_size  Maximum number of samples per read and write batch.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be greater than zero.")

    logger.info(f" Writing data to HDF5 file {outpath}.")
    with h5py.File(outpath, "w") as hfile:
        simio = VTKToHDF5(path, hfile, ndim)
        for sample_index in range(0, simio.nsamples, batch_size):
            current_batch_size = min(
                batch_size, simio.nsamples - sample_index
            )
            simio.process_sample(
                sample_index, sample_index, current_batch_size
            )
