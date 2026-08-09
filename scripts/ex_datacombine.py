#!/usr/bin/env python3

"""Combine samples from multiple simulation trees into one HDF5 dataset."""

import argparse
import os
import sys

import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import vtk_to_hdf5_dataset as vh5


def positive_integer(value):
    """Parse a positive integer command-line argument."""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result

def valid_dimension(value):
    """Parse the spatial dimensions command-line argument."""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result not in [2,3]:
        raise ValueError("Dimension must be 2 or 3")
    return result


def write_combined_hdf5(input_paths, output_path, nsamples, ndim,
                        batch_size=vh5.DEFAULT_BATCH_SIZE):
    """Write block-interleaved simulation-tree batches to HDF5.

    @param input_paths  Paths to the input libEnsemble directory trees.
    @param output_path  Path of the HDF5 file to create.
    @param nsamples  Number of samples to read from each input tree.
    @param ndim  Number of relevant spatial dimensions.
    @param batch_size  Maximum samples per input-tree read and write batch.
    """
    if not input_paths:
        raise ValueError("At least one input tree is required.")
    if batch_size < 1:
        raise ValueError("Batch size must be greater than zero.")

    output_sample_count = nsamples * len(input_paths)
    with h5py.File(output_path, "w") as hfile:
        simios = [
            vh5.VTKToHDF5(
                input_path,
                hfile,
                ndim,
                output_sample_count=output_sample_count,
            )
            for input_path in input_paths
        ]

        for ip, simio in enumerate(simios):
            available = simio.nsamples
            if nsamples > available:
                raise ValueError(
                    f"Input tree {input_paths[ip]!r} contains {available} samples;"
                    f" {nsamples} requested."
                )

        output_index = 0
        next_progress_index = 0
        for sample_index in range(0, nsamples, batch_size):
            current_batch_size = min(
                batch_size, nsamples - sample_index
            )
            for simio in simios:
                if output_index >= next_progress_index:
                    print(f"Output sample {output_index}")
                    next_progress_index = (
                        (output_index // 100) + 1
                    ) * 100
                simio.process_sample(
                    sample_index, output_index, current_batch_size
                )
                output_index += current_batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Interleave samples from multiple simulation directory trees and "
            "write them to one HDF5 file. "
            "The output HDF5 file has a dataset 'mesh' containing "
            "the common mesh points and a dataset 'fields' with shape "
            "(total samples, fields, spatial dimensions...) containing all "
            "physical variable values. Samples are block-interleaved by "
            "input tree at the requested batch size."
        )
    )
    parser.add_argument(
        "--inpaths",
        nargs="+",
        required=True,
        help="one or more input simulation directory trees",
    )
    parser.add_argument("--outpath", required=True, help="output HDF5 file")
    parser.add_argument("--dimension", required=True, type=valid_dimension,
                        help="number of relevant spatial dimensions")
    parser.add_argument(
        "--nsamples",
        required=True,
        type=positive_integer,
        help="number of samples to take from each input tree",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=vh5.DEFAULT_BATCH_SIZE,
        help=(
            "maximum samples per input-tree read and HDF5 write batch "
            f"(default: {vh5.DEFAULT_BATCH_SIZE})"
        ),
    )
    args = parser.parse_args()

    write_combined_hdf5(
        args.inpaths,
        args.outpath,
        args.nsamples,
        args.dimension,
        args.batch_size,
    )
