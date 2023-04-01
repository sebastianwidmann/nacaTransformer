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


def generate_tfrecords(data_dir: str, save_dir: str, stl_format: str,
                       nsamples: int, xmin: float, xmax: float, ymin: float,
                       ymax: float, nx: int, ny: int, k: int, p: int,
                       gpu_id: int):
    """
    Convert vtk datasets from airfoilMNIST into the TFRecord format and save
    them as into the TFRecords directory (more information about TFRecords
    can be found on https://www.tensorflow.org/tutorials/load_data/tfrecord)

    Parameters
    ----------
    data_dir : str
               input directory of vtu and stl files. Expects all vtu and stl
               files to be separated into two folders named 'vtu' and 'stl'
               with the respective files in each folder.
    save_dir : str
               output directory of TFRecord files
    stl_format : str
                 data format of .stl-file formats
    nsamples : int
               number of samples per .tfrecord file
    xmin : int
           minimum bound upstream of wing geometry
    xmax : int
           maximum bound downstream of wing geometry
    ymin : int
           minimum bound below of wing geometry
    ymax : int
           minimum bound above of wing geometry
    nx : int
         number of interpolation points in x direction
    ny : int
         number of interpolation points in y direction
    k : int
        number of nearest neighbours
    p : int
        power parameter
    gpu_id : int
             ID of GPU
    """

    # check if output directory exists and create dir if necessary
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    vtu_folder = os.path.join(data_dir, 'vtu')
    stl_folder = os.path.join(data_dir, 'stl')

    vtu_list = [x for x in sorted(os.listdir(vtu_folder)) if x.endswith('.vtu')]
    stl_list = [("_".join(x.split("_", 2)[:2]) + ".stl") for x in vtu_list]

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
                vtu_dir = os.path.join(vtu_folder, sample[0])
                stl_dir = os.path.join(stl_folder, sample[1])

                # split string to extract information about airfoil, angle of
                # attack and Mach number to write as feature into tfrecord
                sim_config = sample[0].rsplit('.', 1)[0]
                sim_config = sim_config.split('_')

                airfoil, angle, mach = sim_config
                angle, mach = float(angle), float(mach)

                data = vtk_to_tfTensor(vtu_dir, stl_dir, stl_format, xmin,
                                       xmax, ymin, ymax, nx, ny, k, p, gpu_id)

                example = create_tfExample(airfoil, angle, mach, data)

                writer.write(example.SerializeToString())
