#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import os
import random
import tensorflow as tf

from src.preprocessing.conversion import vtk_to_tfTensor, \
    create_tfExample


def generate_tfrecords(data_dir, save_dir, stl_format, nsamples, xmin, xmax,
                       ymin, ymax, nx, ny, k, p, gpu_id):
    """
    Convert vtk datasets from airfoilMNIST into the TFRecord format and save
    them as into the TFRecords directory (more
    information about TFRecords can be found on
    https://www.tensorflow.org/tutorials/load_data/tfrecord)

    :param:
        data_dir: str
            input directory of vtk files
        save_dir: str
            output directory of TFRecord files
        stl_format: str
            data format of .stl-file formats
        nsamples: int, optional
            number of samples per .tfrecord file
        xmin: int
            minimum bound upstream of wing geometry
        xmax: int
            maximum bound downstream of wing geometry
        ymin: int
            minimum bound below of wing geometry
        ymax: int
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
            ID of GPU which shall be used
    :return:
        airfoilMNIST_i.tfrecord: tfrecord
            return simple format for storing a sequence of binary records.

            :features:
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
    """

    # check data_format type
    format_types = ['nacaFOAM', 'Selig', 'Lednicer']
    if stl_format not in format_types:
        raise ValueError('Invalid format. Expected one of: %s' % format_types)

    # check if argum

    # check if output directory exists and create dir if necessary
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    vtu_list = [x for x in sorted(os.listdir(data_dir)) if x.endswith('.vtu')]
    stl_list = [x for x in sorted(os.listdir(data_dir)) if x.endswith('.stl')]

    dataset = list(zip(vtu_list, stl_list))

    quotient, remainder = divmod(len(dataset), nsamples)
    n_tfrecords = quotient + remainder

    for i in range(n_tfrecords):
        if remainder != 0 and i == n_tfrecords - 1:
            samples = [dataset.pop(random.randrange(len(dataset))) for _ in
                       range(remainder)]
        else:
            samples = [dataset.pop(random.randrange(len(dataset))) for _ in
                       range(nsamples)]

        file_dir = os.path.join(save_dir, 'airfoilMNIST_{}.tfrecord'.format(i))

        with tf.io.TFRecordWriter(file_dir) as writer:
            for sample in samples:
                airfoil, angle, mach, _ = sample[0].split('_')
                angle, mach = float(angle), float(mach)

                vtu_dir = os.path.join(data_dir, sample[0])
                stl_dir = os.path.join(data_dir, sample[1])

                data = vtk_to_tfTensor(vtu_dir, stl_dir, stl_format, xmin,
                                       xmax, ymin, ymax, nx, ny, k, p, gpu_id)

                example = create_tfExample(airfoil, angle, mach, data)

                writer.write(example.SerializeToString())
