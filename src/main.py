#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
"""

"""
# ---------------------------------------------------------------------------

from os.path import exists
from sys import exit
from configparser import ConfigParser

# import modules from src
from src.utilities.configuration import generate_config_file
from src.preprocessing.datahandling import generate_tfrecords

# Restore configuration file with default values if file doesn't exist
if not exists('config.ini'):
    generate_config_file()
    exit("Restored config.ini with default values")

# Read configuration file
config = ConfigParser()
config.read("config.ini")

data_dir = config["preprocessing"]["input_directory"]
save_dir = config["preprocessing"]["output_directory"]
stl_format = config["preprocessing"]["stl_format"]
nsamples = int(config["preprocessing"]["nsamples"])
xmin = float(config["preprocessing"]["xmin"])
xmax = float(config["preprocessing"]["xmax"])
ymin = float(config["preprocessing"]["ymin"])
ymax = float(config["preprocessing"]["ymax"])
nx = int(config["preprocessing"]["nx"])
ny = int(config["preprocessing"]["ny"])
k = int(config["preprocessing"]["k"])
p = int(config["preprocessing"]["p"])
gpu_id = int(config["preprocessing"]["gpu_id"])

# Convert airfoilMNIST dataset to TFRecords
generate_tfrecords(data_dir, save_dir, stl_format, nsamples, xmin, xmax,
                   ymin, ymax, nx, ny, k, p, gpu_id)
