# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import os


def restructure_database(dir='../airfoilMNIST'):
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name == 'internal.vtu':
                source_path = os.path.join(root, name)
                target_path = os.path.join(
                    '/home/sebastianwidmann/Documents/git/nacaTransformer/airfoilMNIST/' + os.path.basename(
                        root) + '.vtu')

                os.system('mv' + ' ' + source_path + ' ' + target_path)
