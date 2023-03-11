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

import faiss
import numpy as np


def find_knn(xb, xq, k, gpu_id=0):
    """
    Reduce the size of the computational domain from the initial size of
    (xmin=-10, xmax=30, ymin=-10, ymax=10)

    :param:
        xb: numpy.ndarray
            coordinate points of raw data
        xq: numpy.ndarray
            query vector with interpolation points
        k: int
            number of nearest neighbours
        gpu_id: int
            ID of GPU which shall be used (default=0)
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


def interpolate(source_data, nx, ny, k, p=2, xmin=-1, xmax=5, ymin=-1, ymax=1):
    """
    Interpolate from the base mesh onto a new mesh for the given coordinate
    points and field data values

    :param:
        source_data: numpy.ndarray
            raw field data with coordinate locations and field data values
        nx: int
            number of interpolation points in x1 direction
        ny: int
            nnumber of interpolation points in x2 direction
        k: int
            number of nearest neighbours
        p: int, optional
            power parameter (default = 2)
        xmin: int, optional
            minimum bound upstream of wing geometry (default=-1)
        xmax: int, optional
            maximum bound downstream of wing geometry (default=5)
        ymin: int, optional
            minimum bound below of wing geometry (default=-1)
        ymax: int, optional
            minimum bound above of wing geometry (default=1)
    :return:
        target_data: numpy.ndarray
            query array with interpolated coordinates and field data values
    """

    xb = source_data[:, 0:2]
    yb = source_data[:, 2:]
    xq = np.mgrid[xmin:xmax:(nx * 1j), ymin:ymax:(ny * 1j)]

    dist, idx = find_knn(xb, xq, k)

    weights = np.power(np.reciprocal(dist, out=np.zeros_like(dist),
                                     where=dist != 0), p)

    yq = np.einsum('ij,ijk->ik', weights, yb[idx]) / np.sum(
        weights, axis=1).reshape(xq.shape[0], -1)

    target_data = np.concatenate((xq, yq), axis=1)

    return target_data
