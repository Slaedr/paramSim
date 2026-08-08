import sys
import os
import unittest
from pathlib import Path

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

if __name__ == "__main__":
    unittest.main()
