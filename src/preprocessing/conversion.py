# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from stl import mesh
import numpy as np
import tensorflow as tf
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# import modules from src
from src.preprocessing.interpolation import interpolate


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfExample(airfoil: str, aoa: float, mach: float, field: tf.Tensor):
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
    field: tf.Tensor
          flow field data in the following column format [x y TMean
          alphatMean kMean nutMean omegaMean pMean rhoMean UxMean UyMean]

    Returns
    -------
    example: tensorflow.train.Example
             returns data in the Example format to write into TFRecord
    """

    feature = {
        'airfoil': _bytes_feature(airfoil.encode('UTF-8')),
        'angle': _float_feature(aoa),
        'mach': _float_feature(mach),
        'data': _bytes_feature(tf.io.serialize_tensor(field)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def vtk_to_tfTensor(vtu_dir: str, stl_dir: str, stl_format: str, xmin: float,
                    xmax: float, ymin: float, ymax: float, nx: int, ny: int,
                    k: int, p: int, gpu_id: int):
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
    stl_format: str
                definition in which format the .stl-file is provided
    xmin: float
          minimum bound upstream of wing geometry
    xmax: float
          maximum bound downstream of wing geometry
    ymin: float
          minimum bound below of wing geometry
    ymax: float
          minimum bound above of wing geometry
    nx: int
        number of interpolation points in x1 direction
    ny: int
        number of interpolation points in x2 direction
    k: int
       number of nearest neighbours
    p: int
       power parameter
    gpu_id: int
            ID of GPU

    Returns
    -------
    target_data: tensorflow.ndarray
                 query array with interpolated coordinates and field data
                 values in the following column format: [x y TMean alphatMean
                 kMean nutMean omegaMean pMean rhoMean UxMean UyMean sdf]
    """

    # TODO add implementation for Selig and Lednicer

    # check data_format type
    format_types = ['nacaFOAM']
    # format_types = ['nacaFOAM', 'Selig', 'Lednicer']
    if stl_format not in format_types:
        raise ValueError('Invalid format. Expected one of: %s' % format_types)

    if stl_format == 'nacaFOAM':
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
    raw_data = reduce_size(raw_data, xmin, xmax, ymin, ymax)

    int_data = interpolate(raw_data, raw_geom, xmin, xmax, ymin, ymax, nx,
                           ny, k, p, gpu_id)

    return tf.convert_to_tensor(int_data, dtype=tf.float32)


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

    return numpy_point_data


def reduce_size(data: np.ndarray, xmin: float, xmax: float, ymin: float, ymax):
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
