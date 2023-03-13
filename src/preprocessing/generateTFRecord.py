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

import tensorflow as tf
import numpy as np

from importVTK import convert_vtu_to_numpy, convert_vtp_to_numpy, reduce_size
from interpolateMesh import interpolate


def convert_vtk_to_TFRecords(vtu_dir, vtp_dir, nx, ny, k, p=2):
    """
    Convert datasets from airfoilMNIST into the TFRecord format (more
    information about TFRecords can be found on
    https://www.tensorflow.org/tutorials/load_data/tfrecord)

    :param:
        vtu_dir: str
            raw field data with coordinate locations and field data values
        vtp_dir: str
            directory to XML file as PolyData (.vtp) type
        nx: int
            number of interpolation points in x1 direction
        ny: int
            number of interpolation points in x2 direction
        k: int
            number of nearest neighbours
        p: int, optional
            power parameter (default = 2)
    :return:
        target_data: tensorflow.ndarray
            query array with interpolated coordinates and field data values 
            in the following column format:
            [x y TMean alphatMean kMean nutMean omegaMean pMean rhoMean
            UxMean UyMean]
    """

    raw_data = convert_vtu_to_numpy(vtu_dir)
    raw_data = reduce_size(raw_data)

    raw_geom = convert_vtp_to_numpy(vtp_dir)

    int_data = interpolate(raw_data, raw_geom, nx, ny, k)

    return tf.convert_to_tensor(int_data, dtype=tf.float32)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def createExample(airfoil, aoa, mach, field):
    """
    Create example for data sample which will store the necessary information
    about each CFD simulation

    :param:
        airfoil: str
            shape of NACA airfoil described using a 4- or 5-digit code
        angle: float
            angle of attack of NACA airfoil
        mach: float
            freestream mach number
        data: tensorflow.ndarray
            flow field data in the following column format:
            [x y TMean alphatMean kMean nutMean omegaMean pMean rhoMean
            UxMean UyMean]
    :return:
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
