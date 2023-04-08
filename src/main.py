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
import tensorflow as tf

# Hide any GPUs from Tensorflow. Otherwise, TF might reserve memory and block
# it for JAX
tf.config.experimental.set_visible_devices([], 'GPU')


def main():
    return True


if __name__ == '__main__':
    main()
