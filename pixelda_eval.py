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

# Dependency imports

import tensorflow as tf

import dataset_factory
import pixelda_model
import pixelda_utils
import pixelda_losses
from hparams import create_hparams

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('checkpoint_dir', '/home/intel/yjhong89/domaina/pixelda4',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/home/intel/yjhong89/domaina/eval',
                    'Directory where the results are saved to.')

flags.DEFINE_integer('eval_interval_secs', 60,
                     'The frequency, in seconds, with which evaluation is run.')

flags.DEFINE_string('target_split_name', 'test',
                    'The name of the train/test split.')
flags.DEFINE_string('source_split_name', 'test', 'Split for source dataset.'
                    ' Defaults to train.')

flags.DEFINE_string('source_dataset', 'source',
                    'The name of the source dataset.')

flags.DEFINE_string('target_dataset', 'target',
                    'The name of the target dataset.')

flags.DEFINE_string(
    'source_dataset_dir',
    'data_source',  
    'The directory where the source datasets can be found.')
flags.DEFINE_string(
    'target_dataset_dir',
    'data_target',
    'The directory where the target datasets can be found.')

flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

flags.DEFINE_integer('num_preprocessing_threads', 4,
                     'The number of threads used to create the batches.')

# HParams

flags.DEFINE_string('hparams', '', 'Comma separated hyperparameter values')


def run_eval(run_dir, checkpoint_dir, hparams):
  """Runs the eval loop.

  Args:
    run_dir: The directory where eval specific logs are placed
    checkpoint_dir: The directory where the checkpoints are stored
    hparams: The hyperparameters struct.

  Raises:
    ValueError: if hparams.arch is not recognized.
  """
  for checkpoint_path in slim.evaluation.checkpoints_iterator(
      checkpoint_dir, FLAGS.eval_interval_secs):
    with tf.Graph().as_default():
      target_images, target_labels = dataset_factory.provide_batch(
          FLAGS.target_dataset, FLAGS.target_split_name, FLAGS.target_dataset_dir,
          FLAGS.num_readers, hparams.batch_size,
          FLAGS.num_preprocessing_threads)
      target_labels['class'] = tf.argmax(target_labels['classes'],1)
      del target_labels['classes']
      target_lateral_labels = (target_labels['class'] % 9) / 3
      target_head_labels    = target_labels['class'] % 3
      #########################
      # Preprocess the inputs #
      #########################
      num_target_classes = 3
      if hparams.arch not in ['dcgan']:
        source_dataset = dataset_factory.get_dataset(
            FLAGS.source_dataset,
            split_name=FLAGS.source_split_name,
            dataset_dir=FLAGS.source_dataset_dir)
        num_source_classes = source_dataset.num_classes
        source_images, source_labels = dataset_factory.provide_batch(
            FLAGS.source_dataset, FLAGS.source_split_name, FLAGS.source_dataset_dir,
            FLAGS.num_readers, hparams.batch_size,
            FLAGS.num_preprocessing_threads)
        source_labels['class'] = tf.argmax(source_labels['classes'], 1)
        del source_labels['classes']
        source_lateral_labels = (source_labels['class'] % 9 ) /3
        source_head_labels    = source_labels['class'] % 3

        mask_images   = source_images[:,:,:,3]
        source_images = source_images[:,:,:,:3]
      else:
        source_images = None
        source_labels = None

      ####################
      # Define the model #
      ####################
      end_points = pixelda_model.create_model(
          hparams, target_images, target_images,
          source_images=source_images,
          is_training=False,
          num_classes=num_target_classes)

      #######################
      # Metrics & Summaries #
      #######################
      names_to_values, names_to_updates = create_metrics(end_points,
                                                         source_lateral_labels,source_head_labels,
                                                         target_lateral_labels,target_head_labels,
                                                         hparams)
      pixelda_utils.summarize_model(end_points)
      pixelda_utils.summarize_transferred_grid(
          end_points['transferred_images'], source_images, name='Transferred')
      if 'source_images_recon' in end_points:
        pixelda_utils.summarize_transferred_grid(
            end_points['source_images_recon'],
            source_images,
            name='Source_Reconstruction')
      pixelda_utils.summarize_images(target_images,'Target')

      for name, value in names_to_values.items():
        tf.summary.scalar(name, value)

      # Use the entire split by default
      num_examples = source_dataset.num_samples

      num_batches = math.ceil(num_examples / float(hparams.batch_size))
      global_step = slim.get_or_create_global_step()

      result = slim.evaluation.evaluate_once(
          master=FLAGS.master,
          checkpoint_path=checkpoint_path,
          logdir=run_dir,
          num_evals=num_batches,
          eval_op=list(names_to_updates.values()),
          final_op=list(names_to_values.values()))


def create_metrics(end_points, source_lateral_labels, source_head_labels, target_lateral_labels, target_head_labels, hparams):
  """Create metrics for the model.

  Args:
    end_points: A dictionary of end point name to tensor
    source_labels: Labels for source images. batch_size x 1
    target_labels: Labels for target images. batch_size x 1
    hparams: The hyperparameters struct.

  Returns:
    Tuple of (names_to_values, names_to_updates), dictionaries that map a metric
    name to its value and update op, respectively

  """
  ###########################################
  # Evaluate the Domain Prediction Accuracy #
  ###########################################
  batch_size = hparams.batch_size

  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      ('eval/Domain_Accuracy-Trasnferred'):
          tf.contrib.metrics.streaming_accuracy(
              tf.to_int32(
                  tf.round(tf.sigmoid(end_points[
                      'transferred_domain_logits']))),
              tf.zeros(batch_size, dtype=tf.int32)),
      ('eval/Domain_Accuracy-Target'):
          tf.contrib.metrics.streaming_accuracy(
              tf.to_int32(
                  tf.round(tf.sigmoid(end_points['target_domain_logits']))),
              tf.ones(batch_size, dtype=tf.int32))
  })
 
  ################################
  # Evaluate the task classifier #
  ################################
  if 'source_lateral_task_logits' in end_points:
    metric_name = 'eval/Lateral_Task_Accuracy-Source'
    names_to_values[metric_name], names_to_updates[
        metric_name] = tf.contrib.metrics.streaming_accuracy(
            tf.argmax(end_points['source_lateral_task_logits'], 1),
            source_lateral_labels)
  if 'source_head_task_logits' in end_points:
    metric_name = 'eval/Head_Task_Accuracy-Source'
    names_to_values[metric_name], names_to_updates[
        metric_name] = tf.contrib.metrics.streaming_accuracy(
            tf.argmax(end_points['source_head_task_logits'], 1),
            source_head_labels)

  metric_name='eval/Lateral_Task_Accuracy-Transferred'
  names_to_values[metric_name], names_to_updates[
      metric_name] = tf.contrib.metrics.streaming_accuracy(
          tf.argmax(end_points['transferred_lateral_task_logits'],1),
          source_lateral_labels)
  metric_name='eval/Head_Task_Accuracy-Transferred'
  names_to_values[metric_name], names_to_updates[
      metric_name] = tf.contrib.metrics.streaming_accuracy(
          tf.argmax(end_points['transferred_head_task_logits'],1),
          source_head_labels)

  metric_name='eval/Lateral_Task_Accuracy-Target'
  names_to_values[metric_name], names_to_updates[
      metric_name] = tf.contrib.metrics.streaming_accuracy(
          tf.argmax(end_points['target_lateral_task_logits'],1),
          target_lateral_labels)
  metric_name='eval/Head_Task_Accuracy-Target'
  names_to_values[metric_name], names_to_updates[
      metric_name] = tf.contrib.metrics.streaming_accuracy(
          tf.argmax(end_points['target_head_task_logits'],1),
          target_head_labels)

  return names_to_values, names_to_updates


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  hparams = create_hparams(FLAGS.hparams)
  run_eval(
      run_dir=FLAGS.eval_dir,
      checkpoint_dir=FLAGS.checkpoint_dir,
      hparams=hparams)


if __name__ == '__main__':
  tf.app.run()
