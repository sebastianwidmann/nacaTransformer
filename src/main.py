# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import jax
import os
import tensorflow as tf

from src.preprocessing.preprocess import generate_tfds_dataset
from src.train import train_and_evaluate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to hyperparameter configuration',
    lock_config=True,
)
flags.mark_flag_as_required('config')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Hide GPUs from TF. Otherwise, TF might reserve memory and block it for JAX
    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info('JAX process: %d / %d', jax.process_index(),
                 jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    if FLAGS.config.trainer == 'preprocess':
        generate_tfds_dataset(FLAGS.config)
    elif FLAGS.config.trainer == 'train':
        train_and_evaluate(FLAGS.config)
    elif FLAGS.config.trainer == 'inference':
        print('Implement inference')
    else:
        raise app.UsageError(f'Unknown trainer: {FLAGS.config.trainer}')


if __name__ == '__main__':
    jax.config.config_with_absl()
    app.run(main)
