#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 8, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import faiss
import numpy as np
import cuspatial, cudf


def find_knn(xb, xq, k, gpu_id):
    """
    Find k-nearest neighbours for a query vector based on the input coordinates
    using GPU-accelerated kNN algorithm. More information on
    https://github.com/facebookresearch/faiss/wiki

    :param:
        xb: numpy.ndarray
            coordinate points of raw data
        xq: numpy.ndarray
            query vector with interpolation points
        k: int
            number of nearest neighbours
        gpu_id: int
            ID of GPU which shall be used
    :return:
        (dist, indexes): (numpy.ndarray, numpy.ndarray)
            distances of k nearest neighbours, the index for
            the corresponding points in the xb array

    """

    nq, d = xq.shape

    res = faiss.StandardGpuResources()

    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = gpu_id

    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    index.train(xb)
    index.add(xb)

    dist, indexes = index.search(xq, k=k)

    return np.asarray(dist), np.asarray(indexes)


def interpolate(source_data, wing_data, xmin, xmax, ymin, ymax, nx,
                ny, k, p, gpu_id):
    """
    Interpolate from the base mesh onto a new mesh for the given coordinate
    points and field data values

    :param:
        source_data: numpy.ndarray
            raw field data with coordinate locations and field data values
        wing_data: str
            wing geometry in the .stl file format
        nx: int
            number of interpolation points in x1 direction
        ny: int
            number of interpolation points in x2 direction
        xmin: int
            minimum bound upstream of wing geometry
        xmax: int
            maximum bound downstream of wing geometry
        ymin: int
            minimum bound below of wing geometry
        ymax: int
            minimum bound above of wing geometry
        k: int
            number of nearest neighbours
        p: int
            power parameter
        gpu_id: int
            ID of GPU which shall be used
    :return:
        target_data: numpy.ndarray
            query array with interpolated coordinates and field data values
    """

    xq = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)].reshape(2, -1).T

    # Convert numpy arrays to GeoSeries for points_in_polygon(args)
    pts = cuspatial.GeoSeries.from_points_xy(cudf.Series(xq.flatten()))
    plygon = cuspatial.GeoSeries.from_polygons_xy(
        cudf.Series(wing_data.flatten()).astype(float),
        cudf.Series([0, wing_data.shape[0]]),
        cudf.Series([0, 1]),
        cudf.Series([0, 1])
    )

    # find points within NACA airfoil shape
    df = cuspatial.point_in_polygon(pts, plygon)
    df.rename(columns={df.columns[0]: "inside"}, inplace=True)

    points_in_wing = df.index[df['inside'] == True].to_numpy()

    # delete points inside airfoil shape from interpolation grid
    xq = np.delete(xq, points_in_wing, axis=0)

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

    return target_data
