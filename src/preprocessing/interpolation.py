# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 8, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import faiss
import numpy as np
import cuspatial, cudf


def interpolate(source_data: np.ndarray, wing_data: np.ndarray, mach: float,
                xmin: float, xmax: float, ymin: float, ymax: float, nx: int,
                ny: int, k: int, gpu_id: int, p: int = 2):
    """
    Interpolate from the base mesh onto a new mesh for the given coordinate
    points and field data values

    Parameters
    ----------
    source_data: numpy.ndarray
                 raw field data with coordinate locations and field data values
    wing_data: numpy.ndarray
               wing geometry in the .stl file format
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
    gpu_id: int
            ID of GPU which shall be used
    p: int (default = 2)
       power parameter

    Returns
    -------
    target_data: numpy.ndarray
                 query array with interpolated coordinates and field data
                 values in the following column format [x y TMean alphatMean
                 kMean nutMean omegaMean pMean rhoMean UxMean UyMean sdf]
    """

    xq = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)].reshape(2, -1).T

    xb = source_data[:, 0:2]
    yb = source_data[:, 2:]

    # find nearest neighbours and indices
    dist, idx = find_knn(xb, xq, k, gpu_id)

    # calculate inverse distance weighting
    weights = np.power(np.reciprocal(dist, out=np.zeros_like(dist),
                                     where=dist != 0), p)

    yq = np.einsum('ij,ijk->ik', weights, yb[idx]) / np.sum(
        weights, axis=1).reshape(xq.shape[0], -1)

    target_data = np.concatenate((xq, yq), axis=1)

    wing_points_idx = find_points_inside(xq, wing_data)

    sdf_dist, _ = find_knn(wing_data, target_data[:, 0:2], 1, gpu_id)

    # set sdf values inside wing equal to minus 1
    sdf_dist[wing_points_idx, :] = -1

    # set values for all fields inside wing geometry to zero
    target_data[wing_points_idx, 2:] = 0

    # split data into arrays for encoder / decoder
    mach_data = np.copy(sdf_dist)
    mach_data = np.where(mach == -1, -1, mach)

    x = np.hstack((sdf_dist, mach_data))
    y = target_data[:, -3:]

    return x, y


def find_knn(xb: np.ndarray, xq: np.ndarray, k: int, gpu_id: int):
    """
    Find k-nearest neighbours for a query vector based on the input coordinates
    using GPU-accelerated kNN algorithm. More information on
    https://github.com/facebookresearch/faiss/wiki

    Parameters
    ----------
    xb: numpy.ndarray
        coordinate points of raw data
    xq: numpy.ndarray
        query vector with interpolation points
    k: int
       number of nearest neighbours
    gpu_id: int
            ID of GPU which shall be used

    Returns
    -------
    (dist, indexes): (numpy.ndarray, numpy.ndarray)
                     distances of k nearest neighbours, the index for
                     the corresponding points in the xb array

    """

    _, d = xq.shape

    xb = np.ascontiguousarray(xb, dtype='float32')
    xq = np.ascontiguousarray(xq, dtype='float32')

    res = faiss.StandardGpuResources()

    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    gpu_index.add(xb)
    distances, neighbors = gpu_index.search(xq, k)

    return distances, neighbors


def find_points_inside(target_points: np.ndarray, wing_points: np.ndarray):
    """

    Parameters
    ----------
    target_points: numpy.ndarray
                   points of interpolation grid
    wing_points: numpy.ndarray
                 points of wing geometry. Must start and end at the same
                 point in clock- or counterclockwise direction

    Returns
    -------
    points_in_wing_idx: numpy.ndarray
                        indexes of target_points array which are inside wing
    """
    # Convert numpy arrays to GeoSeries for points_in_polygon(args)
    pts = cuspatial.GeoSeries.from_points_xy(
        cudf.Series(target_points.flatten()))
    plygon = cuspatial.GeoSeries.from_polygons_xy(
        cudf.Series(wing_points.flatten()).astype(float),
        cudf.Series([0, wing_points.shape[0]]),
        cudf.Series([0, 1]),
        cudf.Series([0, 1])
    )

    # find indexes of points within NACA airfoil shape
    df = cuspatial.point_in_polygon(pts, plygon)
    df.rename(columns={df.columns[0]: "inside"}, inplace=True)

    points_in_wing_idx = df.index[df['inside'] == True].to_numpy()

    return points_in_wing_idx
