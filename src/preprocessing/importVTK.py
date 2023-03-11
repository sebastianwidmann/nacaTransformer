#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 8, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
"""

"""
# ---------------------------------------------------------------------------

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np


def convert_vtk_to_numpy(directory):
    """
    Convert datasets from airfoilMNIST in vtk format into numpy arrays

    :param:
        directory: str
            directory to XML file as unstructured grid (.vtu) type
    :return:
        numpy_point_data: numpy.ndarray
            average flow field data in the following column format:
            [x y TMean alphatMean kMean nutMean omegaMean pMean rhoMean
            UxMean UyMean]
    """

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(directory)
    reader.Update()
    vtk_dataset = reader.GetOutput()

    # get points (coordinates) and point data (field variables
    vtk_points = vtk_dataset.GetPoints()
    vtk_point_data = vtk_dataset.GetPointData()

    # read point coordinates into numpy array
    numpy_points = vtk_to_numpy(vtk_points.GetData())
    vtk_coords_number_of_arrays = vtk_points.GetData().GetNumberOfComponents()
    dims = numpy_points.shape

    # read point values from field variables into numpy array
    vtk_number_of_arrays = vtk_point_data.GetNumberOfArrays()
    numpy_point_data = np.zeros(
        [dims[0], vtk_number_of_arrays + 2 + vtk_coords_number_of_arrays])

    # write point coordinates into data array
    numpy_point_data[:, 0:dims[1]] = numpy_points

    for i in range(vtk_number_of_arrays):
        vtk_array_name = vtk_point_data.GetArrayName(i)
        vtk_array = vtk_point_data.GetArray(vtk_array_name)

        idx_start = dims[1] + i
        idx_end = dims[1] + i + vtk_array.GetNumberOfComponents()

        if vtk_array.GetNumberOfComponents() == 1:
            numpy_point_data[:, idx_start:idx_end] = vtk_to_numpy(
                vtk_array).reshape(dims[0], 1)
        else:
            numpy_point_data[:, idx_start:idx_end] = vtk_to_numpy(vtk_array)

    numpy_point_data = np.delete(numpy_point_data[numpy_point_data[:, 2] < 0.5],
                                 [2, -1], axis=1)  # remove z-coord and Uz
    # and only return single plane of points

    return numpy_point_data


def reduce_size(data, xmin=-1, xmax=5, ymin=-1, ymax=1):
    """
    Reduce the size of the computational domain from the initial size of
    (xmin=-10, xmax=30, ymin=-10, ymax=10)

    :param:
        data: numpy.ndarray
            raw average flow field
        xmin: int
            minimum bound upstream of wing geometry (default=-1)
        xmax: int
            maximum bound downstream of wing geometry (default=5)
        ymin: int
            minimum bound below of wing geometry (default=-1)
        ymax: int
            minimum bound above of wing geometry (default=1)
    :return:
        data: numpy.ndarray
            average flow field with new bounds

    """

    assert xmin < xmax
    assert ymin < ymax
    xmask = np.logical_and(data[:, 0] >= xmin, data[:, 0] <= xmax)
    data = data[xmask]
    ymask = np.logical_and(data[:, 1] >= ymin, data[:, 1] <= ymax)
    data = data[ymask]
    return data

# def interpolate_mesh():
