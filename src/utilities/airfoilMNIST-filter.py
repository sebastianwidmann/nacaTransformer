# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import os
from tqdm import tqdm
import argparse


def restructure_internal(src_dir, tgt_dir='../airfoilMNIST/vtu'):
    """
    Extract internal.vtu with flow field quantities, rename based on
    simulation parameters and write into new directory
    """
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    n_total = int(len(os.listdir(src_dir)))
    source_path = []
    target_path = []

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('internal.vtu'):
                file_path = os.path.join(root, file)
                sim_path = os.path.join(tgt_dir, "_".join(os.path.basename(
                    root).split("_", 3)[:3]) + '.vtu')
                source_path.append(file_path)
                target_path.append(sim_path)

    for i in tqdm(range(n_total)):
        os.system('cp -f' + ' ' + source_path[i] + ' ' + target_path[i])


def restructure_forceCoeffs(src_dir, tgt_dir='../airfoilMNIST/forceCoeffs'):
    """
    Extract forceCoeffs.dat with integral values for lift, drag and
    moments, rename based on simulation parameters and write into new directory
    """
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    n_total = int(len(os.listdir(src_dir)))
    source_path = []
    target_path = []

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('forceCoeffs.dat'):
                file_path = os.path.join(root, file)
                sim_path = os.path.join(tgt_dir, "_".join(os.path.basename(
                    root).split("_", 3)[:3]) + '.dat')
                source_path.append(file_path)
                target_path.append(sim_path)

    for i in tqdm(range(n_total)):
        os.system('cp -f' + ' ' + source_path[i] + ' ' + target_path[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='airfoilMNIST filter',
        description='Restructure database into correct preprocessing format'
    )

    parser.add_argument(
        'type',
        type=str,
        help='Specify filter type. [vtu, forceCoeffs]'
    )
    parser.add_argument(
        'source_path',
        type=str,
        help='Specify load directory.'
    )
    parser.add_argument(
        'target_path',
        type=str,
        help='Specify save directory.'
    )

    args = parser.parse_args()

    if args.type == 'vtu':
        restructure_internal(args.source_path, args.target_path)
    elif args.type == 'forceCoeffs':
        restructure_forceCoeffs(args.source_path, args.target_path)
    else:
        print('No valid type given. Possible choices are [vtu, forceCoeffs]')
