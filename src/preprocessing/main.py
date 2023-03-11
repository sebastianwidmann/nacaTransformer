import os
import random
import tensorflow as tf


def main(dir='../airfoilMNIST'):
    """

    :param dir: path directory to database
    # :param n: number of observations which are drawn from sample pool
    :return:
    """

    files = tf.data.Dataset.list_files(dir)

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)

    database = os.listdir(dir)
    print(database)
    #
    # samples = [database.pop(random.randrange(len(database))) for _ in range(n)]
    #
    # for sample in samples:
    #     print(sample)


if __name__ == '__main__':
    main()
