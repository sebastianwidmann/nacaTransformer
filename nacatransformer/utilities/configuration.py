#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from configparser import ConfigParser


def generate_config_file():
    config_object = ConfigParser()

    config_object["preprocessing"] = {
        "input_directory": "airfoilMNIST",
        "output_directory": "airfoilMNIST_TFRecords",
        "stl_format": "nacaFOAM",
        "nsamples": "5",
        "xmin": "-1",
        "xmax": "5",
        "ymin": "-1",
        "ymax": "1",
        "nx": "125",
        "ny": "125",
        "k": "5",
        "p": "2",
        "gpu_id": "0"
    }

    config_object["transformer"] = {}

    with open('config.ini', 'w') as conf:
        config_object.write(conf)
