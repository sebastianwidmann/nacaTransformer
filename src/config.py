import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.preprocess = ml_collections.ConfigDict()
    config.preprocess.readdir = '../../airfoilMNIST'
    config.preprocess.writedir = '../../airfoilMNIST/tfrecords'
    config.preprocess.nsamples = 10
    config.preprocess.stlformat = 'nacaFOAM'
    config.preprocess.resize = (
        -1, 3, -1, 1)  # resize limits (xmin, xmax, ymin, ymax)
    config.preprocess.resolution = (250, 250)  # new field resolution (nx, ny)
    config.preprocess.num_neighbors = 5
    config.preprocess.gpu_id = 0

    return config
