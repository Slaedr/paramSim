#!/usr/bin/env python3

"""Combine simulation trees with MPI and Parallel HDF5."""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import ex_datacombine as dc
import vtk_to_hdf5_dataset as vh5


def _load_mpi():
    """Import mpi4py only when an MPI conversion is requested.

    @return The mpi4py MPI module.
    """
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError(
            "mpi4py is required for MPI dataset conversion"
        ) from exc
    return MPI


def build_batch_tasks(sample_files_by_input, nsamples, batch_size):
    """Build batch tasks in block-interleaved output order.

    @param sample_files_by_input  Sample-file mappings for each input tree.
                                  This maps resolution IDs to paths for each sample
                                  of each input tree.
    @param nsamples  Number of samples to read from each input tree.
    @param batch_size  Maximum number of samples in each task.

    @return List of tuples containing sample paths and output start indices.
    """
    tasks = []
    output_index = 0
    for sample_index in range(0, nsamples, batch_size):
        current_batch_size = min(batch_size, nsamples - sample_index)
        for sample_files in sample_files_by_input:
            sample_paths = tuple(
                sample_files[index]
                for index in range(
                    sample_index, sample_index + current_batch_size
                )
            )
            # Each sample maps actual resolution IDs to paths.
            tasks.append((sample_paths, output_index))
            output_index += current_batch_size
    return tasks


def _prepare_conversion(input_paths, nsamples, ndim, batch_size, comm_size):
    """Inspect inputs and prepare rank-local task lists on the root rank.

    @param input_paths  Paths to the input libEnsemble directory trees.
    @param nsamples  Number of samples to read from each input tree.
    @param ndim  Number of relevant spatial dimensions.
    @param batch_size  Maximum number of samples in each task.
    @param comm_size  Number of MPI ranks receiving tasks.

    @return Metadata, per-rank tasks, reference sample cache, and meshes.
    """
    sample_files_by_input, resolutions = vh5.select_samples(input_paths, nsamples)
    references = {}
    cached_samples = {}
    meshes = {}
    field_dtype = None
    mesh_dtype = None
    for resolution in resolutions:
        path = sample_files_by_input[0][0][resolution]
        fields, points = vh5.read_sample_file(path, ndim)
        if field_dtype is None:
            field_dtype = fields.dtype.str
            mesh_dtype = points.dtype.str
        references[resolution] = {
            "field_shape": fields.shape, "mesh_shape": points.shape,
            "path": path,
        }
        cached_samples[path] = (fields, points)
        meshes[resolution] = points
    tasks = build_batch_tasks(sample_files_by_input, nsamples, batch_size)
    tasks_by_rank = [[] for _ in range(comm_size)]
    for task_index, task in enumerate(tasks):
        tasks_by_rank[task_index % comm_size].append(task)
    metadata = {
        "resolution_ids": resolutions, "references": references,
        "field_dtype": field_dtype, "mesh_dtype": mesh_dtype,
        "output_sample_count": nsamples * len(input_paths),
        "task_count": len(tasks),
    }
    return metadata, tasks_by_rank, cached_samples, meshes


def _read_batch(sample_paths, ndim, expected_shape, expected_dtype,
                cached_sample, resolution_id, expected_mesh):
    """Read one batch into a preallocated array.

    @param sample_paths  Ordered paths for the batch.
    @param ndim  Number of relevant spatial dimensions.
    @param expected_shape  Required shape of one fields array.
    @param expected_dtype  Required NumPy dtype of one fields array.
    @param cached_sample  Optional mapping containing an already-read sample.
    @param resolution_id  Actual resolution ID to read.
    @param expected_mesh  Reference mesh for this resolution.

    @return Array containing all fields in the batch.
    """
    batch_fields = np.empty(
        (len(sample_paths), *expected_shape), dtype=expected_dtype
    )
    for batch_index, sample_path_res in enumerate(sample_paths):
        sample_path = sample_path_res[resolution_id]
        if sample_path in cached_sample:
            fields, points = cached_sample.pop(sample_path)
        else:
            try:
                fields, points = vh5.read_sample_file(sample_path, ndim)
            except Exception as exc:
                raise ValueError(f"Reading {sample_path!r}: {exc}") from exc
        if fields.shape != expected_shape:
            raise ValueError(
                f"Fields in {sample_path!r} have shape {fields.shape}; "
                f"expected {expected_shape}."
            )
        if fields.dtype != expected_dtype:
            raise ValueError(
                f"Fields in {sample_path!r} have dtype {fields.dtype}; "
                f"expected {expected_dtype}."
            )
        if not np.array_equal(points, expected_mesh):
            raise ValueError(f"Mesh in {sample_path!r} differs from reference mesh "
                             f"for resolution {resolution_id}.")
        batch_fields[batch_index] = fields
    return batch_fields


def write_combined_hdf5_mpi(input_paths, output_prefix, nsamples, ndim,
                            batch_size=vh5.DEFAULT_BATCH_SIZE, comm=None):
    """Combine simulation trees using MPI and Parallel HDF5.

    @param input_paths  Paths to the input libEnsemble directory trees.
    @param output_prefix  Exact filename for one resolution; prefix-ID.h5 for multiple.
    @param nsamples  Number of samples to read from each input tree.
    @param ndim  Number of relevant spatial dimensions.
    @param batch_size  Maximum number of samples in each read/write task.
    @param comm  MPI communicator. Defaults to MPI.COMM_WORLD.
    """
    if not h5py.get_config().mpi:
        raise RuntimeError(
            "The installed h5py was not built with MPI support. Install "
            "h5py against a Parallel HDF5 build."
        )
    if not input_paths:
        raise ValueError("At least one input tree is required.")
    if nsamples < 1:
        raise ValueError("Sample count must be greater than zero.")
    if batch_size < 1:
        raise ValueError("Batch size must be greater than zero.")

    MPI = _load_mpi()
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    metadata = None
    tasks_by_rank = None
    cached_sample = {}
    meshes = None
    setup_error = None
    if rank == 0:
        try:
            metadata, tasks_by_rank, cached_sample, meshes = \
                _prepare_conversion(
                    input_paths, nsamples, ndim, batch_size, comm_size
                )
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"

    setup_error = comm.bcast(setup_error, root=0)
    if setup_error is not None:
        raise RuntimeError(f"Input preparation failed: {setup_error}")

    metadata = comm.bcast(metadata, root=0)
    local_tasks = comm.scatter(tasks_by_rank, root=0)
    expected_dtype = np.dtype(metadata["field_dtype"])
    output_sample_count = metadata["output_sample_count"]
    outputs = vh5.resolution_output_paths(output_prefix, metadata["resolution_ids"])
    if rank == 0:
        print(
            f"Writing {output_sample_count} samples with {comm_size} MPI "
            f"ranks in {metadata['task_count']} batches.",
            flush=True,
        )

    num_resolutions = len(metadata["resolution_ids"])
    for completed_resolutions, resolution in enumerate(metadata["resolution_ids"], 1):
        failure = None
        completed_samples = 0
        reference = metadata["references"][resolution]
        expected_shape = tuple(reference["field_shape"])
        output_path = outputs[resolution]
        # Read a mesh locally instead of broadcasting large arrays. Idle ranks
        # need only participate in collective dataset creation and status checks.
        points = None
        mesh_error = None
        try:
            if rank == 0:
                points = meshes.pop(resolution)
            elif local_tasks:
                reference_fields, points = vh5.read_sample_file(reference["path"], ndim)
                del reference_fields
        except Exception as exc:
            mesh_error = f"rank {rank}, resolution {resolution}, reference {reference['path']!r}: {exc}"
        errors = comm.allgather(mesh_error)
        if any(errors):
            raise RuntimeError("Reference mesh read failed: " + "; ".join(error for error in errors if error))
        with h5py.File(output_path, "w", driver="mpio", comm=comm) as hfile:
            mesh = hfile.create_dataset(
                "mesh",
                shape=tuple(reference["mesh_shape"]),
                dtype=np.dtype(metadata["mesh_dtype"]),
            )
            fields_dataset = hfile.create_dataset(
                "fields",
                shape=(output_sample_count, *expected_shape),
                dtype=expected_dtype,
            )

            mesh_error = None
            if rank == 0:
                try:
                    mesh[...] = points
                except Exception as exc:
                    mesh_error = f"rank {rank}, resolution {resolution}: {type(exc).__name__}: {exc}"
            mesh_errors = comm.allgather(mesh_error)
            if any(error is not None for error in mesh_errors):
                failure = "Mesh write failed: " + "; ".join(
                    error for error in mesh_errors if error is not None
                )

            task_rounds = comm.allreduce(len(local_tasks), op=MPI.MAX)
            for task_round in range(task_rounds):
                if failure is not None:
                    break

                local_completed = 0
                local_error = None
                if task_round < len(local_tasks):
                    sample_paths, output_index = local_tasks[task_round]
                    try:
                        batch_fields = _read_batch(
                            sample_paths,
                            ndim,
                            expected_shape, expected_dtype,
                            cached_sample, resolution, points
                        )
                        output_end = output_index + len(sample_paths)
                        fields_dataset[output_index:output_end] = batch_fields
                        local_completed = len(sample_paths)
                    except Exception as exc:
                        local_error = (
                            f"rank {rank}, resolution {resolution}, output sample {output_index}: "
                            f"{type(exc).__name__}: {exc}"
                        )

                round_status = comm.allgather((local_completed, local_error))
                errors = [error for _, error in round_status if error is not None]
                if errors:
                    failure = "Batch processing failed: " + "; ".join(errors)
                    continue

                completed_samples += sum(count for count, _ in round_status)
                if rank == 0:
                    print(
                        f"Resolution {resolution}: output samples completed: {completed_samples}/"
                        f"{output_sample_count}",
                        flush=True,
                    )
        if failure is not None:
            raise RuntimeError(failure)
        if rank == 0:
            print(f"Mesh resolutions completed: {completed_resolutions}/{num_resolutions}", flush=True)


def main():
    """Parse command-line arguments and run the MPI conversion."""
    parser = argparse.ArgumentParser(
        description=(
            "Interleave simulation trees and write them collectively with "
            "MPI-enabled h5py and Parallel HDF5."
        )
    )
    parser.add_argument(
        "--inpaths",
        nargs="+",
        required=True,
        help="one or more input simulation directory trees",
    )
    parser.add_argument("--outpath", required=True, help="output filename for one resolution; prefix[-ID.h5] for multiple resolutions")
    parser.add_argument(
        "--dimension",
        required=True,
        type=dc.valid_dimension,
        help="number of relevant spatial dimensions",
    )
    parser.add_argument(
        "--nsamples",
        required=True,
        type=dc.positive_integer,
        help="number of samples to take from each input tree",
    )
    parser.add_argument(
        "--batch-size",
        type=dc.positive_integer,
        default=vh5.DEFAULT_BATCH_SIZE,
        help=(
            "maximum samples per input-tree read and write batch "
            f"(default: {vh5.DEFAULT_BATCH_SIZE})"
        ),
    )
    args = parser.parse_args()

    if not h5py.get_config().mpi:
        print(
            "error: the installed h5py was not built with MPI support",
            file=sys.stderr,
        )
        return 2
    try:
        MPI = _load_mpi()
    except RuntimeError as exc:
        print(
            f"error: {exc}",
            file=sys.stderr,
        )
        return 2

    comm = MPI.COMM_WORLD
    try:
        write_combined_hdf5_mpi(
            args.inpaths,
            args.outpath,
            args.nsamples,
            args.dimension,
            args.batch_size,
            comm,
        )
    except Exception as exc:
        if comm.Get_rank() == 0:
            print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
