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

from interpolation import interpolate


def create_tfExample(x: np.ndarray, y: np.ndarray):
    """
    Create example for data sample which will store the necessary information
    about each CFD simulation

    Parameters
    ----------
    airfoil: str
             shape of NACA airfoil described using a 4- or 5-digit code
    aoa: float
           angle of attack of NACA airfoil
    mach: float
          freestream mach number
    field: np.ndarray
          flow field data in the following column format [x y TMean
          alphatMean kMean nutMean omegaMean pMean rhoMean UxMean UyMean]

    Returns
    -------
    example: tensorflow.train.Example
             returns data in the Example format to write into TFRecord
    """

    feature = {
        'data_encoder': tf.train.Feature(float_list=tf.train.FloatList(
            value=x.flatten())),
        'data_decoder': tf.train.Feature(float_list=tf.train.FloatList(
            value=y.flatten())),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def vtk_to_tfTensor(vtu_dir: str, stl_dir: str, config: ConfigDict,
                    mach: float):
    """
    Convert datasets from airfoilMNIST into the TFRecord format (more
    information about TFRecords can be found on
    https://www.tensorflow.org/tutorials/load_data/tfrecord)

    Parameters
    ----------
    vtu_dir: str
             raw field data with coordinate locations and field data values
    stl_dir: str
             directory to .stl-file of wing geometry
    config: ConfigDict
            configuration parameters for kNN and interpolation

    Returns
    -------
    target_data: np.ndarray
                 query array with interpolated coordinates and field data
                 values in the following column format: [x y TMean alphatMean
                 kMean nutMean omegaMean pMean rhoMean UxMean UyMean sdf]
    """

    # check data_format type
    format_types = ['nacaFOAM']
    # format_types = ['nacaFOAM', 'Selig', 'Lednicer']
    if config.stlformat not in format_types:
        raise ValueError('Invalid format. Expected one of: %s' % format_types)

    if config.stlformat == 'nacaFOAM':
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
        raw_geom = mesh.Mesh.from_file(stl_dir)[:, 0:3]

        # remove duplicate points of side surface at x=-1
        raw_geom = raw_geom[raw_geom[:, 2] > 0][:, 0:2]

        # find index where points of side surface end
        id = np.where(raw_geom == np.max(raw_geom[:, 0]))[0][-1]

        # remove redundant points from .stl-file type
        raw_geom = raw_geom[: id + 1, :]

    raw_data = vtu_to_numpy(vtu_dir)
    raw_data = resize(raw_data, *config.resize)

    x, y = interpolate(raw_data, raw_geom, mach, *config.resize,
                       *config.resolution, config.num_neighbors,
                       config.gpu_id)

    w, h = config.resolution[0], config.resolution[1]
    c_encoder, c_decoder = x.shape[1], y.shape[1]

    x = x.reshape([w, h, c_encoder])
    y = y.reshape([w, h, c_decoder])

    return x, y


def vtu_to_numpy(vtu_dir: str):
    """
    Convert datasets from airfoilMNIST in vtk format into numpy arrays

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

    numpy_point_data = np.delete(numpy_point_data, [2, 3, 4, 5, 6, 8],
                                 axis=1)  # TODO: REMOVE BEFORE FINAL COMMIT

    return numpy_point_data


def resize(data: np.ndarray, xmin: float, xmax: float, ymin: float, ymax):
    """
    Reduce the size of the computational domain from the initial size of
    (xmin=-10, xmax=30, ymin=-10, ymax=10)

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
