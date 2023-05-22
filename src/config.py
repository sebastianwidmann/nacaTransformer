import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.dataset = '../nacaTransformer/airfoilMNIST/'
    config.trainer = 'train'
    # config.trainer = 'preprocess'
    config.train_size = 0.8
    config.batch_size = 2
    config.num_epochs = 50
    config.learning_rate = 0.03
    config.weight_decay = 0.0

    config.vit = ml_collections.ConfigDict()
    config.vit.img_size = (200, 200)
    config.vit.patches = (10, 10)
    config.vit.hidden_size = 300  # num_patches^2 * num_channels
    config.vit.num_layers = 5
    config.vit.num_heads = 10
    config.vit.dim_mlp = 4 * config.vit.hidden_size
    config.vit.dropout_rate = 0.1
    config.vit.att_dropout_rate = 0.0

    config.preprocess = ml_collections.ConfigDict()
    config.preprocess.readdir = '/media/sebastianwidmann/Backup500GB/'
    config.preprocess.writedir = '/media/sebastianwidmann/Backup500GB/airfoilMNIST'
    config.preprocess.nsamples = 128
    config.preprocess.dim = (-0.75, 1.25, -1, 1)
    config.preprocess.num_neighbors = 5
    config.preprocess.gpu_id = 0

    return config
