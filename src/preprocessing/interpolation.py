# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 8, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
from ml_collections import ConfigDict
import faiss
import numpy as np
import cuspatial, cudf


def interpolate(config: ConfigDict, source_data: np.ndarray,
                wing_data: np.ndarray, mach: float, p: int = 2):
    """
    Interpolate from the base mesh onto a new mesh for the given coordinate
    points and field data values

    Parameters
    ----------
    config: ConfigDict
            configuration parameters
    source_data: numpy.ndarray
            raw field data with coordinate locations and field data values
    wing_data: numpy.ndarray
            wing geometry in the .stl file format
    mach: float
            Freestream mach number
    p: int (default = 2)
        power parameter

    Returns
    -------
    target_data: numpy.ndarray
                 query array with interpolated coordinates and field data
                 values in the following column format [x y TMean alphatMean
                 kMean nutMean omegaMean pMean rhoMean UxMean UyMean sdf]
    """

    xmin, xmax, ymin, ymax = config.preprocess.dim
    nx, ny = config.vit.img_size
    k = config.preprocess.num_neighbors

    xq = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)].reshape(2, -1).T

    xb = source_data[:, 0:2]
    yb = source_data[:, 2:]

    # find nearest neighbours and indices
    dist, idx = find_knn(xb, xq, k, config.preprocess.gpu_id)

    # calculate inverse distance weighting
    weights = np.power(np.reciprocal(dist, out=np.zeros_like(dist),
                                     where=dist != 0), p)

    yq = np.einsum('ij,ijk->ik', weights, yb[idx]) / np.sum(
        weights, axis=1).reshape(xq.shape[0], -1)

    target_data = np.concatenate((xq, yq), axis=1)

    wing_points_idx = find_points_inside(xq, wing_data)

    sdf_dist, _ = find_knn(wing_data, target_data[:, 0:2], 1,
                           config.preprocess.gpu_id)

    # set sdf values inside wing equal to minus 1
    sdf_dist[wing_points_idx, :] = -1

    # set values for all fields inside wing geometry to zero
    target_data[wing_points_idx, 2:] = 0

    # split data into arrays for encoder / decoder
    mach_data = np.copy(sdf_dist)
    mach_data = np.where(mach_data == -1, -1, mach)
    mach_data[wing_points_idx, :] = 0

    # x = np.hstack((sdf_dist, mach_data))
    x = mach_data

    # Delete point coordinates from decoder input
    y = target_data[:, -3:]

    # Define thermodynamic properties of air at ICAO standard atmosphere
    T0 = 288.15  # [K] Total temperature
    p0 = 101325  # [Pa] Total pressure
    gamma = 1.4  # [-] Ratio of specific heats
    R = 287.058  # [J/(kg*K)] Specific gas constant for dry air

    T = T0 / (1 + 0.5 * (gamma - 1) * mach ** 2)
    p_inf = p0 * (1 + 0.5 * (gamma - 1) * mach ** 2) ** (-gamma / (gamma - 1))
    u_inf = mach * np.sqrt(gamma * R * T)

    # Normalise pressure by freestream pressure
    y[:, 0] /= p_inf

    # Normalise velocities by freestream velocity
    y[:, 1] /= u_inf
    y[:, 2] /= u_inf

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
