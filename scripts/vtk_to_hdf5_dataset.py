import os
import logging
import re

import numpy as np
import meshio as mio
import h5py

logger = logging.getLogger(__name__)

DIM_LABELS = ('x', 'y', 'z', 'u', 'v', 'w')
DEFAULT_BATCH_SIZE = 8


def discover_sample_files(ensemble_root_path : str):
    """Discover the VTK or VTU file(s) associated with each ensemble sample.

    Note that each sample can comprise multiple files, one for each resolution.

    @param ensemble_root_path  Location of the libEnsemble directory tree.

    @return Dict of dicts. Mapping from sample ID to {resolution ID: file path}.
    """
    sample_directories = {}
    for name in os.listdir(ensemble_root_path):
        directory = os.path.join(ensemble_root_path, name)
        if not os.path.isdir(directory) or not name.startswith("sim"):
            continue
        if not name[3:].isdigit():
            continue
        sample_index = int(name[3:])
        if sample_index in sample_directories:
            raise ValueError(
                f"Multiple libEnsemble directories represent sample {sample_index}: "
                f"{sample_directories[sample_index]!r} and {directory!r}."
            )
        sample_directories[sample_index] = directory
    if 0 not in sample_directories:
        raise ValueError(
                f"No libEnsemble directory found for sample 0 in {ensemble_root_path!r}.")
    sample_files = {}
    for sample_index, directory in sample_directories.items():
        resolutions = {}
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if not os.path.isfile(filepath) or not filename.lower().endswith((".vtk", ".vtu")):
                continue
            match = re.fullmatch(r".+-([0-9]+)\.(?:vtk|vtu)", filename, re.IGNORECASE)
            if match is None:
                raise ValueError(
                        f"Unparseable resolution filename {filepath!r} in {directory!r}.")
            resolution = int(match[1])
            if resolution in resolutions:
                raise ValueError(f"Duplicate resolution {resolution} for "
                                 f"sample {sample_index} in {directory!r}: "
                                 f"{resolutions[resolution]!r} and {filepath!r}."
                )
            resolutions[resolution] = filepath
        if not resolutions:
            raise ValueError(f"No VTK or VTU file found for sample {sample_index} "
                             f"in {directory!r}.")
        sample_files[sample_index] = resolutions
    return sample_files


def select_samples(input_paths, nsamples=None):
    """Discover inputs and validate requested sample IDs and common resolutions.
    With no count, select all samples (which must have consecutive sample IDs).

    @param input_paths  Paths to the trees to select samples from.
    @param nsamples  Number of samples to select from each tree.

    @return Sample mappings for each tree and numerically ordered resolution IDs.
    """
    if not input_paths:
        raise ValueError("At least one input tree is required.")
    if nsamples is not None and nsamples < 1:
        raise ValueError("Sample count must be greater than zero.")
    inputs = []
    expected = None
    for input_path in input_paths:
        samples = discover_sample_files(input_path)
        count = len(samples) if nsamples is None else nsamples
        missing = sorted(set(range(count)) - samples.keys())
        if missing:
            raise ValueError(f"No libEnsemble directories found for sample IDs {missing}"
                             f" in {input_path!r}.")
        selected = {index: samples[index] for index in range(count)}
        # Now iterate over each sample
        for index, files in selected.items():
            # files is the map from resolution ID to VTK file path for a selected sample,
            #  so set(files) is the set of resolution IDs in this sample.
            actual = set(files)
            if expected is None:
                expected = actual
            if actual != expected:
                raise ValueError(
                    f"Resolution mismatch in tree {input_path!r}, sample {index}: "
                    f"missing resolutions {sorted(expected - actual)}, extra "
                    f"resolutions {sorted(actual - expected)}."
                )
        inputs.append(selected)
    return inputs, sorted(expected)


def resolution_output_paths(output_path, resolution_ids):
    """Preserve a single output filename; append actual IDs for multiple levels."""
    output_path = os.fspath(output_path)
    if len(resolution_ids) == 1:
        return {resolution_ids[0]: output_path}
    prefix = output_path[:-3] if output_path.endswith(".h5") else output_path
    return {resolution: f"{prefix}-{resolution}.h5" for resolution in resolution_ids}


def read_sample_file(sample_path : str, ndim : int):
    """Read and spatially sort one VTK or VTU sample.

    @param sample_path  Path to the VTK or VTU sample file.
    @param ndim  Number of relevant spatial dimensions.

    @return Tuple containing the fields and mesh point arrays.
    """
    sample = mio.read(sample_path)
    return cartesian_domain_sort(sample, ndim)

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
    if npoin != check_npoin:
        raise ValueError(f"Mesh points do not form a Cartesian grid: {npoin} points, grid shape {meshgrid}.")
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
        if points.shape[0] != u_vals.shape[0]:
            raise ValueError(f"Field has {u_vals.shape[0]} points; expected {points.shape[0]}.")
        nuvars = (1 if u_vals.ndim == 1 else u_vals.shape[1])
        datau[:, ndim+ivar:ndim+ivar+nuvars] = u_vals.reshape(points.shape[0], nuvars)
        ivar += nuvars
    dtypes = [*[(label, np.float32) for label in DIM_LABELS[:ndim]],
              *[(name, np.float32) for name in varnames]]
    datas = datau.view(dtypes).reshape(-1)

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
                 output_sample_count=None, resolution_id=None, sample_files=None):
        """ Prepares to read VTK data from an ensemble tree and write to an
            HDF5 file.

        If the file does not already contain mesh and fields datasets, read
        their shapes and data types from the first input sample and create
        them.

        @param ensemble_root_path  Location of the ensemble directory.
        @param hfile  Open HDF5 file object.
        @param ndim  Number of relevant spatial dimensions.
        @param resolution_id  Resolution to select; required for multiple levels.
        @param sample_files  Optional previously validated sample/resolution mapping.
        @param output_sample_count  Number of samples in the output fields
                                    dataset. Defaults to the number of input
                                    samples.
        """
        self.indirpath = ensemble_root_path
        self.ndim = ndim
        self.hfile = hfile
        if sample_files is None:
            inputs, resolutions = select_samples([self.indirpath])
            sample_files = inputs[0]
        else:
            resolutions = sorted(sample_files[0])
        if resolution_id is None:
            if len(resolutions) != 1:
                raise ValueError("Select a resolution_id explicitly for multiple resolutions.")
            resolution_id = resolutions[0]
        if resolution_id not in resolutions:
            raise ValueError(f"Unknown resolution {resolution_id}; available resolutions: {resolutions}.")
        self.resolution_id = resolution_id
        self.sample_files = sample_files
        self.sample_directories = {
            index: os.path.dirname(paths[resolution_id])
            for index, paths in sample_files.items()
        }
        self.nsamples = len(self.sample_files)

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

        self.reference_mesh = self.hfile["mesh"][...]

    def _read_sample(self, sample_index : int):
        """Read and spatially sort one sample's fields and mesh points.

        @param sample_index  Integer sample ID to read from the ensemble tree.

        @return Tuple containing the fields and mesh points arrays.
        """
        if sample_index not in self.sample_files:
            raise ValueError(
                f"No libEnsemble directory found for sample {sample_index} "
                f"in {self.indirpath!r}."
            )

        return read_sample_file(self.sample_files[sample_index][self.resolution_id], self.ndim)

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
            fields, points = self._read_sample(sample_index)
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
            if not np.array_equal(points, self.reference_mesh):
                raise ValueError(f"Mesh for sample {sample_index}, resolution {self.resolution_id} differs from reference mesh.")
            batch_fields.append(fields)

        output_fields[out_sample_idx:output_end] = np.stack(batch_fields)

def ensemble_dir_to_hdf5(path, ndim : int, outpath,
                         batch_size : int = DEFAULT_BATCH_SIZE) -> None:
    """Convert a libEnsemble directory tree to a batched HDF5 dataset.

    @param path  Path to the libEnsemble output directory.
    @param ndim  Number of relevant spatial dimensions.
    @param outpath  Exact filename for one resolution; prefix-ID.h5 for multiple.
    @param batch_size  Maximum number of samples per read and write batch.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be greater than zero.")

    inputs, resolutions = select_samples([path])
    outputs = resolution_output_paths(outpath, resolutions)
    for resolution in resolutions:
        logger.info(f" Writing resolution {resolution} to {outputs[resolution]}.")
        with h5py.File(outputs[resolution], "w") as hfile:
            simio = VTKToHDF5(path, hfile, ndim, resolution_id=resolution,
                              sample_files=inputs[0])
            for sample_index in range(0, simio.nsamples, batch_size):
                simio.process_sample(sample_index, sample_index,
                                     min(batch_size, simio.nsamples - sample_index))
