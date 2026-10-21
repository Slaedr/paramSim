import sys
import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from numpy.lib import recfunctions as rfn
import meshio as mio

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from vtk_to_hdf5_dataset import (
    DIM_LABELS,
    reshape_list_to_meshgrid,
    get_cartesian_meshgrid_shape
)

import vtk_to_hdf5_dataset as vh5
import ex_datacombine as dc
import ex_datacombine_mpi as mpi_dc

MPI_TESTS_AVAILABLE = (
    h5py.get_config().mpi
    and importlib.util.find_spec("mpi4py") is not None
    and shutil.which("mpiexec") is not None
)

def open_3d_mesh():
    meshpath = "./data/ex3d.vtk"
    if not os.path.isfile(meshpath):
        meshpath = "./test/scripts/data/ex3d.vtk"
    mesh = mio.read(meshpath)
    return mesh

def pressure_values(i, j, k):
    return i + 10*j + 100.0*k

def temperature_values(i, j, k):
    return 100.0 * i + 10*j + k

def generate_3d_data():
    """ Synthetic 3D data. """
    shape = (5,4,3)
    dtype3d = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
    vardtype = np.dtype([("pressure", np.float32), ("temperature", np.float32)])
    alltypes = np.dtype(dtype3d.descr + vardtype.descr)
    n = shape[0]*shape[1]*shape[2]
    data = np.empty(n, dtype=alltypes)
    x = [-1.0, -0.1, 1.0]
    y = [-1.0, -0.3, 0.2, 1.0]
    z = [-1.0, -0.7, 0.1, 0.5, 1.0]
    points_tensor = np.empty((3, *shape), dtype=np.float32)
    # coords = np.meshgrid(z, y, x, indexing='ij')
    # for idim in range(3):
    #     points_tensor[idim, ...] = coords[idim]
    vars_tensor = np.empty((2, *shape), dtype=np.float32)
    for k in range(shape[0]):
        for j in range(shape[1]):
            for i in range(shape[2]):
                vars_tensor[0, k, j, i] = pressure_values(i, j, k)
                vars_tensor[1, k, j, i] = temperature_values(i, j, k)
                points_tensor[0, k, j, i] = x[i]
                points_tensor[1, k, j, i] = y[j]
                points_tensor[2, k, j, i] = z[k]
    # Now assemble the list
    for k in range(shape[0]):
        for j in range(shape[1]):
            for i in range(shape[2]):
                ipoin = i + j*shape[2] + k*shape[2]*shape[1]
                data[ipoin]['x'] = x[i]
                data[ipoin]['y'] = y[j]
                data[ipoin]['z'] = z[k]
                data[ipoin]['pressure'] = pressure_values(i,j,k)
                data[ipoin]['temperature'] = temperature_values(i,j,k)
    return data, vars_tensor, points_tensor

class ListMeshgrid(unittest.TestCase):
    def setUp(self):
        self.data, self.vars_tensor, self.points_tensor = generate_3d_data()

    def test_list_to_meshgrid_conversion(self):
        datalist = self.data.view((np.float32, len(self.data.dtype)))
        dataset, meshpoints = reshape_list_to_meshgrid(datalist, 2,
                                                       self.points_tensor.shape[1:])
        self.assertTrue(np.all(dataset == self.vars_tensor))
        self.assertTrue(np.all(meshpoints == self.points_tensor))

def assert_spatially_sorted_2x2(data):
    """ Checks whether the given structured array is spatial sorted,
    assuming a 2x2x2 grid.
    """
    ndim = 3 if 'z' in data.dtype.names else 2
    for name in DIM_LABELS[:ndim]:
        dp, idxs = np.unique(data[name], return_index=True)
        # Undo the sort performed by np.unique
        dp = dp[np.argsort(idxs)]
        for i in range(len(dp)-1):
            if dp[i] > dp[i+1]:
                return False
    sorted = True
    for i in range(4):
        sorted = sorted and (data[2*i]['x'] <= data[2*i+1]['x'])
    sorted = sorted and (data[0]['y'] <= data[2]['y']) and \
        (data[1]['y'] <= data[3]['y']) and (data[4]['y'] <= data[6]['y']) and \
        (data[5]['y'] <= data[7]['y'])
    for i in range(4):
        sorted = sorted and (data[i]['z'] <= data[i+4]['z'])
    if not sorted:
        return False
    return True

class VTKProcessing3dSmall(unittest.TestCase):
    def setUp(self):
        self.ndim = 3
        self.nvars = 8
        self.nfields = 4
        self.mesh = open_3d_mesh()
        self.npoin = self.mesh.points.shape[0]
        assert(self.mesh.points.shape[1] == self.ndim)
        self.point_dtypes = np.dtype([*[(label, np.float32) \
                                        for label in DIM_LABELS[:3]]])
        self.var_dtypes = np.dtype([("pressure", "f4"), ("temperature", "f4"),\
                                    ("velocity_x", "f4"), ("velocity_y", "f4"),
                                    ("velocity_z", "f4"), ("heat_flux_x", "f4"),
                                    ("heat_flux_y", "f4"), ("heat_flux_z", "f4")])
        self.alldtypes = np.dtype(self.point_dtypes.descr + self.var_dtypes.descr)
        self.datau = np.empty((self.npoin, self.ndim+self.nvars), dtype=np.float32)
        self.datau[:,:self.ndim] = self.mesh.points[:,:self.ndim]
        assert(len(self.mesh.point_data.items()) == self.nfields)
        self.datau[:, self.ndim] = self.mesh.point_data["temperature"][:,0]
        self.datau[:, self.ndim+1] = self.mesh.point_data["pressure"][:,0]
        self.datau[:, self.ndim+2:self.ndim+5] = self.mesh.point_data["velocity"]
        self.datau[:, self.ndim+5:self.ndim+8] = self.mesh.point_data["heat_flux"]
        self.datas = self.datau.view(self.alldtypes).reshape(-1)

        # Get reference tensors but only for temperature and pressure
        self.points = np.empty((self.ndim, 2, 2, 2), np.float32)
        self.soln = np.empty((self.nvars, 2, 2, 2), np.float32)
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    self.points[0, k, j, i] = -1.0 if i == 0 else 1.0
                    self.points[1, k, j, i] = -1.0 if j == 0 else 1.0
                    self.points[2, k, j, i] = -1.0 if k == 0 else 1.0
        temp = np.array([[[300.0, 305], [302, 310]], [[315, 320], [312, 318]]])
        pres = np.array([[[101325.0, 101320], [101315, 101310]],
                         [[101290.0, 101285], [101288, 101280]]])
        self.soln[0, ...] = temp
        self.soln[1, ...] = pres

    def test_meshgrid_shape(self):
        points_arr = np.unique(self.mesh.points, axis=0)
        points = np.array([tuple(row) for row in points_arr],
                          dtype=self.point_dtypes)

        shape = get_cartesian_meshgrid_shape(points, 3)

        self.assertEqual(shape, (2,2,2))

    def test_np_unique_works(self):
        datas = np.unique(self.datas, axis=0)
        self.assertEqual(datas.shape[0], 8)
        self.assertTrue(np.all(datas["pressure"].sort() == \
                               self.soln[4, ...].ravel().sort()))

    def test_spatial_sort(self):
        datas = np.unique(self.datas, axis=0)
        shape = get_cartesian_meshgrid_shape(datas, self.ndim)

        vh5.spatial_sort(datas, shape)

        self.assertTrue(assert_spatially_sorted_2x2(datas))

    def test_cartesian_domain_sort(self):
        solution, points = vh5.cartesian_domain_sort(self.mesh, 3)

        self.assertTrue(np.all(points == self.points))
        self.assertTrue(np.all(solution[0,...] == self.soln[0,...]))
        self.assertTrue(np.all(solution[1,...] == self.soln[1,...]))


def write_resolution_tree(root, tree_id=0, nsamples=5, resolutions=(2, 10),
                          prefix="field-with-hyphens", extension="vtk"):
    """Create small, distinct Cartesian fixtures with different resolution shapes."""
    root = Path(root)
    for sample in range(nsamples):
        directory = root / f"sim{sample:03}"
        directory.mkdir(parents=True, exist_ok=True)
        for resolution in resolutions:
            level = sorted(resolutions).index(resolution)
            coordinates = np.linspace(-1, 1, level + 2, dtype=np.float32)
            x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
            points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
            values = tree_id * 1000 + sample * 10 + resolution * 100 + x.ravel() + 2 * y.ravel()
            mesh = mio.Mesh(points, [("vertex", np.arange(x.size).reshape(-1, 1))],
                            point_data={"solution": values[:, None].astype(np.float32)})
            mio.write(directory / f"{prefix}-{resolution}.{extension}", mesh)
    return str(root)


def check_resolution_outputs(test, prefix, identities, resolutions=(2, 10)):
    """Verify every coordinate and field value against fixture definitions."""
    for level, resolution in enumerate(resolutions):
        with test.subTest(resolution=resolution):
            output = f"{prefix}.h5" if len(resolutions) == 1 else f"{prefix}-{resolution}.h5"
            with h5py.File(output, "r") as hfile:
                test.assertEqual(set(hfile), {"mesh", "fields"})
                test.assertEqual(hfile["mesh"].dtype, np.dtype("float32"))
                test.assertEqual(hfile["fields"].dtype, np.dtype("float32"))
                coordinates = np.linspace(-1, 1, level + 2, dtype=np.float32)
                x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
                np.testing.assert_array_equal(hfile["mesh"][:], np.stack((x, y)))
                expected = np.stack([tree * 1000 + sample * 10 + resolution * 100 + x + 2 * y
                                     for tree, sample in identities])[:, None]
                np.testing.assert_array_equal(hfile["fields"][:], expected)


class EnsembleDirectoryConversion(unittest.TestCase):
    def test_converts_tinymesh2_to_hdf5(self):
        ensemble_path = (
            Path(__file__).resolve().parent / "data" / "tinymesh2"
        )
        for sample_id in ("sim000", "sim001"):
            domain_path = (
                ensemble_path / sample_id / "test-bc_exp-0-0.vtk"
            )
            domain_mesh = mio.read(domain_path)
            self.assertEqual(
                domain_mesh.point_data["velocity"].shape, (64, 3)
            )

        coordinates = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
        mesh_x, mesh_y = np.meshgrid(coordinates, coordinates, indexing="xy")
        expected_mesh = np.stack((mesh_x, mesh_y))

        expected_solution0 = np.array(
            [[
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.301751, 0.552884, 0.796107, 0.850062, 1.0],
                [0.341951, 0.683350, 1.166260, 0.869299, 1.0],
                [-0.219635, 0.575200, 1.014130, 0.880826, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]],
            dtype=np.float32,
        )
        expected_solution1 = expected_solution0.copy()
        expected_solution1[0, 0, 1] = 0.9
        expected_solution1[0, 1, 0] = 0.322751
        expected_solution1[0, 2, 0] = 0.441951

        expected_velocity0 = np.stack(
            (mesh_x, mesh_y, mesh_x + mesh_y)
        )
        expected_velocity1 = np.stack(
            (mesh_x + 0.25, mesh_y - 0.5, mesh_x - mesh_y)
        )
        expected_fields0 = np.concatenate(
            (expected_solution0, expected_velocity0)
        )
        expected_fields1 = np.concatenate(
            (expected_solution1, expected_velocity1)
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "tinymesh2.h5"
            vh5.ensemble_dir_to_hdf5(
                str(ensemble_path), 2, str(output_path)
            )

            with h5py.File(output_path, "r") as hfile:
                self.assertEqual(set(hfile.keys()), {"mesh", "fields"})
                self.assertEqual(hfile["mesh"].shape, (2, 5, 5))
                self.assertEqual(
                    hfile["mesh"].dtype, np.dtype(np.float32)
                )
                np.testing.assert_allclose(
                    hfile["mesh"][...], expected_mesh
                )

                self.assertEqual(hfile["fields"].shape, (2, 4, 5, 5))
                self.assertEqual(
                    hfile["fields"].dtype, np.dtype(np.float32)
                )
                np.testing.assert_allclose(
                    hfile["fields"][0], expected_fields0
                )
                np.testing.assert_allclose(
                    hfile["fields"][1], expected_fields1
                )

        with tempfile.TemporaryDirectory() as directory:
            tree = write_resolution_tree(Path(directory) / "tree")
            for suffix in ("", ".h5"):
                with self.subTest(multiple_resolutions=True, suffix=suffix):
                    prefix = Path(directory) / "converted"
                    vh5.ensemble_dir_to_hdf5(tree, 2, str(prefix) + suffix, batch_size=2)
                    check_resolution_outputs(self, prefix, [(0, sample) for sample in range(5)])
                    self.assertEqual({p.name for p in Path(directory).glob("converted*")},
                                     {"converted-2.h5", "converted-10.h5"})

    def test_combines_batches_with_block_interleaving(self):
        ensemble_path = (
            Path(__file__).resolve().parent / "data" / "tinymesh2"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "combined.h5"
            dc.write_combined_hdf5(
                [str(ensemble_path), str(ensemble_path)],
                str(output_path),
                2,
                2,
                batch_size=2,
            )

            with h5py.File(output_path, "r") as hfile:
                self.assertEqual(set(hfile.keys()), {"mesh", "fields"})
                self.assertEqual(hfile["fields"].shape, (4, 4, 5, 5))
                np.testing.assert_allclose(
                    hfile["fields"][0], hfile["fields"][2]
                )
                np.testing.assert_allclose(
                    hfile["fields"][1], hfile["fields"][3]
                )
                self.assertFalse(np.allclose(
                    hfile["fields"][0], hfile["fields"][1]
                ))

        with tempfile.TemporaryDirectory() as directory:
            trees = [write_resolution_tree(Path(directory) / f"tree{tree}", tree,
                                           prefix=f"different-prefix-{tree}") for tree in range(2)]
            prefix = Path(directory) / "combined"
            dc.write_combined_hdf5(trees, str(prefix) + ".h5", 5, 2, batch_size=2)
            identities = [(tree, sample) for start in range(0, 5, 2)
                          for tree in range(2) for sample in range(start, min(start + 2, 5))]
            check_resolution_outputs(self, prefix, identities)

    def test_rejects_invalid_batch_ranges_and_shapes(self):
        ensemble_path = (
            Path(__file__).resolve().parent / "data" / "tinymesh2"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "invalid-batches.h5"
            with h5py.File(output_path, "w") as hfile:
                simio = vh5.VTKToHDF5(
                    str(ensemble_path),
                    hfile,
                    2,
                    output_sample_count=2,
                )

                with self.assertRaisesRegex(ValueError, "Batch size"):
                    simio.process_sample(0, 0, 0)
                with self.assertRaisesRegex(ValueError, "sample IDs"):
                    simio.process_sample(1, 0, 2)
                with self.assertRaisesRegex(ValueError, "Output sample range"):
                    simio.process_sample(0, 1, 2)

                original_read_sample = simio._read_sample

                def read_incompatible_sample(sample_index):
                    fields, points = original_read_sample(sample_index)
                    if sample_index == 1:
                        fields = fields[:-1]
                    return fields, points

                simio._read_sample = read_incompatible_sample
                with self.assertRaisesRegex(ValueError, "have shape"):
                    simio.process_sample(0, 0, 2)

        with tempfile.TemporaryDirectory() as directory:
            tree = write_resolution_tree(Path(directory) / "tree", nsamples=2)
            with h5py.File(Path(directory) / "selected.h5", "w") as hfile:
                with self.assertRaisesRegex(ValueError, "explicitly"):
                    vh5.VTKToHDF5(tree, hfile, 2)
                simio = vh5.VTKToHDF5(tree, hfile, 2, resolution_id=10)
                original = simio._read_sample
                for mismatch in ("shape", "mesh"):
                    with self.subTest(mismatch=mismatch):
                        def incompatible(index):
                            fields, points = original(index)
                            if index == 1:
                                if mismatch == "shape":
                                    fields = fields[:, :-1]
                                else:
                                    points = points + 0.25
                            return fields, points
                        simio._read_sample = incompatible
                        with self.assertRaisesRegex(ValueError, "have shape|Mesh"):
                            simio.process_sample(0, 0, 2)

    def test_discovers_resolution_ids_and_validates_requested_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = write_resolution_tree(Path(directory) / "tree", nsamples=3,
                                         resolutions=(10, 2))
            other = write_resolution_tree(Path(directory) / "other", nsamples=2,
                                          prefix="another-prefix", extension="vtu")
            inputs, resolutions = vh5.select_samples([tree, other], 2)
            self.assertEqual(resolutions, [2, 10])
            self.assertEqual(set(inputs[0]), {0, 1})
            # An unrequested sample can contain a different set of levels.
            Path(tree, "sim002", "field-with-hyphens-10.vtk").unlink()
            shutil.copy(Path(tree, "sim002", "field-with-hyphens-2.vtk"),
                        Path(tree, "sim002", "extra-level-30.vtk"))
            metadata, _, _, _ = mpi_dc._prepare_conversion([tree, other], 2, 2, 2, 2)
            self.assertEqual(metadata["resolution_ids"], [2, 10])
            dc.write_combined_hdf5([tree, other], str(Path(directory) / "selected"), 2, 2)
            for location in (tree, other):
                with self.subTest(mismatch_tree=location):
                    path = next(Path(location, "sim001").glob("*-10.*"))
                    renamed = path.with_name(path.name.replace("-10.", "-3."))
                    path.rename(renamed)
                    with self.assertRaisesRegex(
                            ValueError,
                            "sample 1.*missing resolutions.*10.*extra resolutions.*3"):
                        vh5.select_samples([tree, other], 2)
                    with self.assertRaisesRegex(ValueError, "Resolution mismatch"):
                        mpi_dc._prepare_conversion([tree, other], 2, 2, 2, 2)
                    renamed.rename(path)
            missing = Path(tree, "sim001", "field-with-hyphens-10.vtk")
            missing.unlink()
            output = Path(directory) / "invalid.h5"
            with self.assertRaisesRegex(ValueError, "missing resolutions.*10"):
                dc.write_combined_hdf5([tree], output, 2, 2)
            self.assertFalse(list(Path(directory).glob("invalid*")))

    def test_rejects_empty_ambiguous_and_duplicate_samples(self):
        for problem in ("empty", "duplicate", "unparseable", "duplicate_sample", "missing_zero"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                tree = Path(write_resolution_tree(Path(directory) / "tree", nsamples=1,
                                                  resolutions=(2,)))
                sample = tree / "sim000"
                source = sample / "field-with-hyphens-2.vtk"
                if problem == "empty":
                    source.unlink()
                    message = "No VTK or VTU"
                elif problem == "duplicate":
                    shutil.copy(source, sample / "different-prefix-2.vtu")
                    message = "Duplicate resolution 2"
                elif problem == "unparseable":
                    source.rename(sample / "ambiguous.vtk")
                    message = "Unparseable.*ambiguous.vtk"
                elif problem == "duplicate_sample":
                    (tree / "sim0").mkdir()
                    message = "Multiple libEnsemble"
                else:
                    sample.rename(tree / "sim001")
                    message = "sample 0"
                with self.assertRaisesRegex(ValueError, message):
                    vh5.discover_sample_files(tree)


class MPIEnsembleDirectoryConversion(unittest.TestCase):
    def setUp(self):
        self.ensemble_path = (
            Path(__file__).resolve().parent / "data" / "tinymesh2"
        )

    def test_builds_block_interleaved_batch_tasks(self):
        sample_files = [
            {i: {2: f"a{i}-2", 10: f"a{i}-10"} for i in range(3)},
            {i: {2: f"b{i}-2", 10: f"b{i}-10"} for i in range(3)},
        ]

        tasks = mpi_dc.build_batch_tasks(sample_files, 3, 2)

        self.assertEqual(
            tasks,
            [
                ((sample_files[0][0], sample_files[0][1]), 0),
                ((sample_files[1][0], sample_files[1][1]), 2),
                ((sample_files[0][2],), 4),
                ((sample_files[1][2],), 5),
            ],
        )

    def test_rejects_missing_requested_sample(self):
        with self.assertRaisesRegex(ValueError, "sample IDs.*2"):
            mpi_dc._prepare_conversion(
                [str(self.ensemble_path)], 3, 2, 2, 2
            )

    def test_rejects_incompatible_batch_shape(self):
        sample_path = (
            self.ensemble_path / "sim000" / "test-bc_exp-0-0.vtk"
        )

        with self.assertRaisesRegex(ValueError, "have shape"):
            mpi_dc._read_batch(
                ({0: str(sample_path)},),
                2,
                (1, 1, 1),
                np.dtype(np.float32),
                {},
                0,
                np.empty((2, 1, 1)),
            )

        fields, points = vh5.read_sample_file(str(sample_path), 2)
        with self.assertRaisesRegex(ValueError, "Mesh.*differs"):
            mpi_dc._read_batch(({0: str(sample_path)},), 2, fields.shape,
                               fields.dtype, {}, 0, points + 0.25)
        cached = {str(sample_path): (fields, points)}
        result = mpi_dc._read_batch(({0: str(sample_path)},), 2, fields.shape,
                                    fields.dtype, cached, 0, points)
        np.testing.assert_array_equal(result[0], fields)
        self.assertFalse(cached)

    @unittest.skipUnless(
        MPI_TESTS_AVAILABLE,
        "requires mpi4py, MPI-enabled h5py, and mpiexec",
    )
    def test_mpi_output_matches_serial_output(self):
        for nsamples, ranks, resolutions in ((5, 3, (2, 10)), (1, 3, (2, 10)), (2, 2, (7,))):
            with self.subTest(nsamples=nsamples, ranks=ranks), \
                    tempfile.TemporaryDirectory() as directory:
                serial_path = Path(directory) / "serial"
                mpi_path = Path(directory) / "mpi"
                input_paths = [write_resolution_tree(Path(directory) / f"tree{tree}", tree,
                                                     nsamples=nsamples, resolutions=resolutions) \
                               for tree in range(2)]
                dc.write_combined_hdf5(input_paths, str(serial_path) + ".h5", nsamples, 2, batch_size=2)
                command = [shutil.which("mpiexec"), "-n", str(ranks), sys.executable,
                           str(SCRIPTS_DIR / "ex_datacombine_mpi.py"), "--inpaths", *input_paths,
                           "--outpath", str(mpi_path) + ".h5", "--dimension", "2",
                           "--nsamples", str(nsamples), "--batch-size", "2"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0,
                                 msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
                identities = [(tree, sample) for start in range(0, nsamples, 2)
                              for tree in range(2) for sample in range(start, min(start + 2, nsamples))]
                check_resolution_outputs(self, mpi_path, identities, resolutions)
                for resolution in resolutions:
                    suffix = ".h5" if len(resolutions) == 1 else f"-{resolution}.h5"
                    with h5py.File(f"{serial_path}{suffix}", "r") as serial_file, \
                            h5py.File(f"{mpi_path}{suffix}", "r") as mpi_file:
                        for key in ("mesh", "fields"):
                            np.testing.assert_array_equal(serial_file[key][:], mpi_file[key][:])

    @unittest.skipUnless(MPI_TESTS_AVAILABLE, "requires Parallel HDF5 and MPI")
    def test_mpi_batch_failure_stops_before_next_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = write_resolution_tree(Path(directory) / "tree", nsamples=3)
            # Rank one reads sample 1; its first resolution now has a wrong shape.
            path = Path(tree) / "sim001" / "field-with-hyphens-2.vtk"
            mesh = mio.read(Path(tree) / "sim001" / "field-with-hyphens-10.vtk")
            mio.write(path, mesh)
            prefix = Path(directory) / "failed"
            command = [shutil.which("mpiexec"), "-n", "2", sys.executable,
                       str(SCRIPTS_DIR / "ex_datacombine_mpi.py"), "--inpaths", tree,
                       "--outpath", str(prefix), "--dimension", "2", "--nsamples", "3",
                       "--batch-size", "1"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(result.stderr, "rank 1, resolution 2, output sample 1.*have shape")
            self.assertIn(str(path), result.stderr)
            self.assertTrue(Path(f"{prefix}-2.h5").exists())
            self.assertFalse(Path(f"{prefix}-10.h5").exists())


if __name__ == "__main__":
    unittest.main()
