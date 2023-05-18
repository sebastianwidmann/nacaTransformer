# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from stl import mesh
import numpy as np
import tensorflow as tf
from ml_collections import ConfigDict
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from src.preprocessing.interpolation import interpolate


def create_tfExample(x: np.ndarray, y: np.ndarray):
    """ Create example for data sample which will store the necessary
    information about each CFD simulation

    Parameters
    ----------
    y: np.ndarray
            encoder input
    x: np.ndarray
            decoder inpu

    Returns
    -------
    example: tensorflow.train.Example
             returns data in the Example format to write into TFRecord
    """

    feature = {
        'encoder': tf.train.Feature(float_list=tf.train.FloatList(
            value=x.flatten())),
        'decoder': tf.train.Feature(float_list=tf.train.FloatList(
            value=y.flatten())),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def vtk_to_tfTensor(config: ConfigDict, vtu_dir: str, stl_dir: str,
                    mach: float):
    """ Convert datasets from airfoilMNIST into the TFRecord format (more
    information about TFRecords can be found on
    https://www.tensorflow.org/tutorials/load_data/tfrecord)

    Parameters
    ----------
    config: ConfigDict
            configuration parameters
    vtu_dir: str
            raw field data with coordinate locations and field data values
    stl_dir: str
            directory to .stl-file of wing geometry
    mach: float
            Freestream mach number


    Returns
    -------
    target_data: np.ndarray
                 query array with interpolated coordinates and field data
                 values in the following column format: [x y TMean alphatMean
                 kMean nutMean omegaMean pMean rhoMean UxMean UyMean sdf]
    """

    """
    For any NACA .stl-file written with nacaFOAM, the points are read
    into arrays in the order such the side surface at z=1 is written
    first, followed by the side surface at z=-1 and finally the
    airfoil shape surface. The point order is starting from the trailing
    edge, moving along the lower surface towards the leading edge and
    back on the upper surface towards the trailing edge again.

    To create a mask for the interpolated grid, the side surface @
    z=-1 and airfoil shape surface must be removed as these points are
    duplicated.
    """
    # read point coordinates from file
    points = mesh.Mesh.from_file(stl_dir)[:, 0:3]

    # remove duplicate points of side surface at x=-1
    points = points[points[:, 2] > 0][:, 0:2]

    # find index where points of side surface end
    idx = np.where(points == np.max(points[:, 0]))[0][-1]

    # remove redundant points from .stl-file type
    points = points[: idx + 1, :]

    # read fields from vtk files
    fields = vtu_to_numpy(vtu_dir)
    fields = resize(fields, *config.preprocess.dim)

    # interpolate and return equally-spaced fields for encoder and decoder
    x, y = interpolate(config, fields, points, mach)

    # Reshape into correct shapes for VIT
    w, h = config.vit.img_size
    c_encoder, c_decoder = x.shape[1], y.shape[1]
    x = x.reshape([w, h, c_encoder])
    y = y.reshape([w, h, c_decoder])

    return x, y


def vtu_to_numpy(vtu_dir: str):
    """ Convert datasets from airfoilMNIST in vtk format into numpy arrays.

    Parameters
    ----------
    vtu_dir: str
             directory to XML file as UnstructuredGrid (.vtu) type

    Returns
    -------
    numpy_point_data: numpy.ndarray
                      average flow field data in the following column format:
                      [x y TMean alphatMean kMean nutMean omegaMean pMean
                      rhoMean UxMean UyMean]
    """

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_dir)
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

    # Only return pressure, velocity Ux, velocity Uy
    numpy_point_data = np.delete(
        numpy_point_data, [2, 3, 4, 5, 6, 8], axis=1)

    return numpy_point_data


def resize(data: np.ndarray, xmin: float, xmax: float, ymin: float, ymax):
    """ Reduce the size of the computational domain from the initial size of
    (xmin=-10, xmax=30, ymin=-10, ymax=10).

    Parameters
    ----------
    data: numpy.ndarray
          raw average flow field
    xmin: float
          minimum bound upstream of wing geometry
    xmax: float
          maximum bound downstream of wing geometry
    ymin: float
          minimum bound below of wing geometry
    ymax: float
          minimum bound above of wing geometry

    Returns
    -------
    data: numpy.ndarray
          average flow field with new bounds

    """

    xmask = np.logical_and(data[:, 0] >= xmin, data[:, 0] <= xmax)
    data = data[xmask]
    ymask = np.logical_and(data[:, 1] >= ymin, data[:, 1] <= ymax)
    data = data[ymask]
    return data
