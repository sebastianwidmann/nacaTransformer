# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import os
import numpy as np


def restructure_database(src_dir='../database', tgt_dir='../airfoilMNIST'):
    for folder in os.listdir(src_dir):
        folder_dir = os.path.join(src_dir, folder)

        subfolder = [name for name in os.listdir(folder_dir)]

        """
        Some folders have duplicated solutions due to manual error of copying
        the files from the server to the external hard drive.
    
        To remove the incorrect file (i.e. the file which has an older creation
        date), each folder will be checked for duplicates and if duplicates
        exist, the creation date will be checked and the older file is deleted.
        """
        if len(subfolder) == 6:
            duplicates = []
            for i in subfolder:
                dir = os.path.join(folder_dir, i)
                if os.path.isdir(dir):
                    duplicates.append(dir)

            date_created = []
            for j in duplicates:
                date_created.append(os.path.getmtime(j))

            date_created = np.asarray(date_created)

            idx = np.argmax(date_created)

            vtu_dir = duplicates[idx]

            for file in os.listdir(vtu_dir):
                if file == 'internal.vtu':
                    src_path = os.path.join(vtu_dir, file)
                    tgt_path = os.path.join(tgt_dir, os.path.basename(
                        vtu_dir) + '.vtu')

                    os.system('cp -f' + ' ' + src_path + ' ' + tgt_path)

        if len(subfolder) == 4:
            for i in subfolder:
                dir = os.path.join(folder_dir, i)
                if os.path.isdir(dir):
                    for file in os.listdir(dir):
                        if file == 'internal.vtu':
                            src_path = os.path.join(dir, file)
                            tgt_path = os.path.join(tgt_dir, os.path.basename(
                                dir) + '.vtu')

                            os.system('cp -f' + ' ' + src_path + ' ' + tgt_path)


def extract_forceCoeffs_files(src_dir, tgt_dir):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    for root, dirs, files in os.walk(src_dir):
        for name in files:
            if name.endswith('forceCoeffs.dat'):
                source_path = os.path.join(root, name)
                target_path = os.path.join(
                    tgt_dir, os.path.basename(root) + '.dat')

                os.system('cp -f' + ' ' + source_path + ' ' + target_path)
