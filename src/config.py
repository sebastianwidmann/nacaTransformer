import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.batch = 2
    config.epoch = 1
    config.patches = (16, 16)

    config.preprocess = ml_collections.ConfigDict()
    config.preprocess.readdir = '../airfoilMNIST'
    config.preprocess.writedir = '../airfoilMNIST/tfrecords'
    config.preprocess.nsamples = 10
    config.preprocess.stlformat = 'nacaFOAM'
    config.preprocess.resize = (
        -1, 3, -1, 1)  # resize limits (xmin, xmax, ymin, ymax)
    config.preprocess.resolution = (250, 250)  # new field resolution (nx, ny)
    config.preprocess.num_neighbors = 5
    config.preprocess.gpu_id = 0

    config.transformer = ml_collections.ConfigDict()
    config.transformer.readdir = config.preprocess.writedir
    config.transformer.num_layers = 6
    config.transformer.num_heads = 12
    config.transformer.dim_mlp = 3072
    config.transformer.dropout_rate = 0.1
    config.transformer.att_dropout_rate = 0.1

    return config
