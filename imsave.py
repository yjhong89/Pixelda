# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Evaluates the PIXELDA model.

-- Compiles the model for CPU.
$ bazel build -c opt third_party/tensorflow_models/domain_adaptation/pixel_domain_adaptation:pixelda_eval

-- Compile the model for GPU.
$ bazel build -c opt --copt=-mavx --config=cuda \
    third_party/tensorflow_models/domain_adaptation/pixel_domain_adaptation:pixelda_eval

-- Runs the training.
$ ./bazel-bin/third_party/tensorflow_models/domain_adaptation/pixel_domain_adaptation/pixelda_eval \
    --source_dataset=mnist \
    --target_dataset=mnist_m \
    --dataset_dir=/tmp/datasets/ \
    --alsologtostderr

-- Visualize the results.
$ bash learning/brain/tensorboard/tensorboard.sh \
    --port 2222 --logdir=/tmp/pixelda/
"""
from functools import partial
import math
import sys
import os

# Dependency imports

import tensorflow as tf

import numpy as np
import dataset_factory
import pixelda_model
import pixelda_utils
import pixelda_losses
import scipy.misc
from hparams import create_hparams
import matplotlib.pyplot as plt

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('img_dir','./img_dir3','')
flags.DEFINE_string('checkpoint_dir', '/home/intel/yjhong89/domaina/input_mask/contrast_no_content_entropy',
                    'Directory where the model was written to.')
flags.DEFINE_string('num_readers',4,'')
flags.DEFINE_string('num_preprocessing_threads', 4, '')

# HParams

flags.DEFINE_string('hparams', '', 'Comma separated hyperparameter values')


def run_eval(data_dir, checkpoint_dir, hparams):
  """Runs the eval loop.

  Args:
    run_dir: The directory where eval specific logs are placed
    checkpoint_dir: The directory where the checkpoints are stored
    hparams: The hyperparameters struct.

  Raises:
    ValueError: if hparams.arch is not recognized.
  """
  source_images, source_labels = dataset_factory.provide_batch('source', 'test', 'data_source', FLAGS.num_readers, hparams.batch_size, FLAGS.num_preprocessing_threads)
  target_images, _ = dataset_factory.provide_batch('target', 'test', 'data_target', FLAGS.num_readers, hparams.batch_size, FLAGS.num_preprocessing_threads)
  source_labels = tf.argmax(source_labels['classes'], 1)
  mask_images = source_images[:,:,:,3]
  source_images = source_images[:,:,:,:3]

  if hparams.input_mask:
    mask_images = tf.to_float(tf.greater(mask_images, 0.9))
    source_images = tf.multiply(source_images, tf.tile(tf.expand_dims(mask_images, 3), [1,1,1,3]))
  
  if not os.path.exists(FLAGS.img_dir):
    os.mkdir(FLAGS.img_dir)

  ####################
  # Define the model #
  ####################
  end_points = pixelda_model.create_model(
      hparams, target_images, target_images, 
      source_images=source_images,
      is_training=False, noise=None,
      num_classes=3)

  saver = tf.train.Saver()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess,ckpt.model_checkpoint_path)
      print('loaded')
    else:
      return
    
    for j in range(10):
      source_img, trans_img, source_lbl = sess.run([source_images, end_points['transferred_images'], source_labels])
      for i in range(hparams.batch_size):
        scipy.misc.imsave(os.path.join(FLAGS.img_dir, 'source{}_{}_{}.png'.format(source_lbl[i],j,i)), source_img[i])
        scipy.misc.imsave(os.path.join(FLAGS.img_dir, 'trans{}_{}_{}.png'.format(source_lbl[i],j,i)), trans_img[i])

    coord.request_stop() 
    coord.join(threads)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  hparams = create_hparams(os.path.join(FLAGS.checkpoint_dir, 'hparams.json'))
  run_eval(
      data_dir=FLAGS.img_dir,
      checkpoint_dir=FLAGS.checkpoint_dir,
      hparams=hparams)


if __name__ == '__main__':
  tf.app.run()
