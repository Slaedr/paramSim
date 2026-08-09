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


def write_combined_hdf5(input_paths, output_path, nsamples, ndim):
    """Write interleaved samples from multiple simulation trees to HDF5."""
    #datasets = [dc.Sim2DDataSetIO(path) for path in input_paths]
    with h5py.File(output_path, "w") as hfile:
        simios = [vh5.VTKToHDF5(input_path, hfile, ndim) \
                for input_path in input_paths]

        for ip, simio in enumerate(simios):
            available = simio.nsamples
            if nsamples > available:
                raise ValueError(
                    f"Input tree {input_paths[ip]!r} contains {available} samples;"
                    f" {nsamples} requested."
                )

        output_index = 0
        for sample_index in range(nsamples):
            for simio in simios:
                print(f"Output sample {output_index}")
                simio.process_sample(sample_index, output_index)
                output_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Interleave samples from multiple simulation directory trees and "
            "write them to one HDF5 file. "
            "The output HDF5 file has a dataset 'mesh' containing "
            "the common mesh points, and group for every sample, "
            "eg., 'sample0', 'sample1' and so on. Each such sample contains "
            "a dataset 'fields' containing all the physical variable values "
            "at each mesh point."
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
    args = parser.parse_args()

    write_combined_hdf5(args.inpaths, args.outpath, args.nsamples, args.dimension)
