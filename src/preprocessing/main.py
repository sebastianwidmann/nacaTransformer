import os
import random
import tensorflow as tf

from generateTFRecord import convert_vtk_to_TFRecords, createExample


def main(nx, ny, data_dir='../airfoilMNIST',
         save_dir='../airfoilMNIST_TFRecords', k=5, nsamples=1):
    """
    Convert vtk datasets from airfoilMNIST into the TFRecord format and save
    them as into the TFRecords directory (more
    information about TFRecords can be found on
    https://www.tensorflow.org/tutorials/load_data/tfrecord)

    :param:
        nx: int
            number of interpolation points in x1 direction
        ny: int
            number of interpolation points in x2 direction
        data_dir: int
            input directory of vtk files
        save_dir: int
            output directory of TFRecord files
        k: int, optional
            number of nearest neighbours (default = 5)
        nsamples: int, optional
            power parameter (default = 2)
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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_dir = '../nacaTest'

    vtu_list = [x for x in sorted(os.listdir(data_dir)) if x.endswith('.vtu')]
    vtp_list = [x for x in sorted(os.listdir(data_dir)) if x.endswith('.vtp')]

    dataset = list(zip(vtu_list, vtp_list))
    n_dataset = len(dataset)

    quotient, remainder = divmod(n_dataset, nsamples)
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
                vtp_dir = os.path.join(data_dir, sample[1])

                data = convert_vtk_to_TFRecords(vtu_dir, vtp_dir, nx,
                                                ny, k)

                example = createExample(airfoil, angle, mach, data)

                writer.write(example.SerializeToString())


if __name__ == '__main__':
    main(nx=125, ny=125)
