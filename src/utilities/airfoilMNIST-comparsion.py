# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: April 27, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import os
import argparse


def missing_files(src_dir: str, dir1: str, dir2: str):
    dir1 = os.path.join(src_dir, dir1)
    dir2 = os.path.join(src_dir, dir2)

    set1 = [x[:-4] for x in os.listdir(dir1)]
    set2 = [x[:-4] for x in os.listdir(dir2)]

    diff = list(set(set1).difference(set2))

    print(diff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='airfoilMNIST-small comparison',
        description='Check if files are missing between two dirs'
    )

    parser.add_argument(
        'source_path',
        type=str,
        help='Specify load directory.'
    )

    parser.add_argument(
        'folder_name_1',
        type=str,
        help='Specify name of folder.'
    )

    parser.add_argument(
        'folder_name_2',
        type=str,
        help='Specify name of folder.'
    )

    args = parser.parse_args()

    missing_files(args.source_path, args.folder_name_1, args.folder_name_2)
