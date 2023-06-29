import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.dataset = '../nacaTransformer/airfoilMNIST/'
    config.output_dir = '../nacaTransformer/plots'
    config.trainer = 'train'  # Alternative 'preprocess' or 'inference'
    config.num_epochs = 200
    config.batch_size = 88
    config.shuffle_buffer_size = 1024
    config.learning_rate_scheduler = "sgdr"
    config.learning_rate_end_value = 1e-5
    config.sgdr_restarts = int(config.num_epochs / 50)
    config.warmup_fraction = 0.1
    config.weight_decay = 0.1
    config.output_frequency = 25

    config.vit = ml_collections.ConfigDict()
    config.vit.img_size = (200, 200)
    config.vit.patch_size = (10, 10)  # num_patches = (img_size / patch_size)
    config.vit.hidden_size = 300  # patch_size^2 * num_channels
    config.vit.num_layers = 3
    config.vit.num_heads = 3
    config.vit.dim_mlp = 4 * config.vit.hidden_size
    config.vit.dropout_rate = 0.0
    config.vit.att_dropout_rate = 0.0

    config.preprocess = ml_collections.ConfigDict()
    config.preprocess.readdir = ''  # Specify read dir to generate dataset
    config.preprocess.writedir = ''  # Specify write dir to generate dataset
    config.preprocess.train_split = 0.8
    config.preprocess.aoa = (-5, 15)
    config.preprocess.mach = (0, 0.6)
    config.preprocess.nsamples = 1024
    config.preprocess.dim = (-0.75, 1.25, -1, 1)
    config.preprocess.num_neighbors = 5
    config.preprocess.gpu_id = 0

    return config
