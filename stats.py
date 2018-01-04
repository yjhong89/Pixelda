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
from hparams import create_hparams
import matplotlib.pyplot as plt

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('checkpoint_dir', '/home/intel/yjhong89/domaina/input_mask_contrast_no_content_entropy',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', 'test1',
                    'Directory where the results are saved to.')

flags.DEFINE_integer('ckpt_num',3807,'ckpt number')
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
      #del target_labels['classes']
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
        #del source_labels['classes']
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
          is_training=False, noise=None,
          num_classes=num_target_classes)

      test_batch_num = int(math.floor(20000 / hparams.batch_size))
      #######################
      # Metrics & Summaries #
      #######################
      target_lateral_one_hot_labels = slim.one_hot_encoding(tf.cast(target_lateral_labels, tf.int64),3)
      target_head_one_hot_labels    = slim.one_hot_encoding(tf.cast(target_head_labels, tf.int64),3)

      lateral_labels_num = tf.reduce_sum(target_lateral_one_hot_labels,0)
      head_labels_num    = tf.reduce_sum(target_head_one_hot_labels,0)
 
      lateral_command = tf.tile(tf.expand_dims(tf.nn.softmax(end_points['target_lateral_task_logits'])[:,2] - tf.nn.softmax(end_points['target_lateral_task_logits'])[:,0],1), [1, 3])
      lateral_cmd_label = tf.multiply(target_lateral_one_hot_labels, lateral_command)
      head_command    = tf.tile(tf.expand_dims(tf.nn.softmax(end_points['target_head_task_logits'])[:,2] - tf.nn.softmax(end_points['target_head_task_logits'])[:,0],1), [1,3])
      head_cmd_label    = tf.multiply(target_head_one_hot_labels, head_command)

      model_lateral_one_hot = slim.one_hot_encoding(tf.argmax(end_points['target_lateral_task_logits'],1),3)
      model_head_one_hot    = slim.one_hot_encoding(tf.argmax(end_points['target_head_task_logits'],1),3)

      lateral_correct_one = tf.reduce_sum(tf.multiply(model_lateral_one_hot,target_lateral_one_hot_labels),0)
      head_correct_one    = tf.reduce_sum(tf.multiply(model_head_one_hot,target_head_one_hot_labels),0)

      target_lateral_one_hot_labels = tf.expand_dims(target_lateral_one_hot_labels,2)
      target_head_one_hot_labels    = tf.expand_dims(target_head_one_hot_labels,2)

      lateral_sort_of_confusion_table = tf.multiply(target_lateral_one_hot_labels, tf.expand_dims(tf.nn.softmax(end_points['target_lateral_task_logits']),1))
      head_sort_of_confusion_table = tf.multiply(target_head_one_hot_labels, tf.expand_dims(tf.nn.softmax(end_points['target_head_task_logits']),1))
      saver = tf.train.Saver()

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.allow_soft_placement = True

      saliency_lateral = compute_saliency_maps(end_points['target_lateral_task_logits'], target_images, tf.cast(target_lateral_labels, tf.int64))
      saliency_head    = compute_saliency_maps(end_points['target_head_task_logits'], target_images, tf.cast(target_head_labels, tf.int64))
      with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            print('loaded')
        else:
            return
           
        np_lateral_confusion = np.zeros((3,3))
        np_head_confusion    = np.zeros((3,3))
        np_lateral_num       = np.zeros(3)
        np_head_num          = np.zeros(3)
        np_lateral_acc       = np.zeros(3)
        np_head_acc          = np.zeros(3)
        np_lateral_cmd       = np.zeros(3)
        np_head_cmd          = np.zeros(3)
        for idx in range(test_batch_num):
          if idx == 0:
            natural_imgs, saliency_lateral_imgs, saliency_head_imgs = sess.run([target_images,saliency_lateral, saliency_head])
            img_num = 10
            plt.rcParams['figure.figsize'] = [48,16]
            for j in range(img_num):
              plt.subplot(3,img_num,j+1)
              plt.imshow(natural_imgs[j])
              plt.axis('off')
              plt.subplot(3,img_num,img_num+j+1)
              plt.imshow(saliency_lateral_imgs[0][j])
              plt.axis('off')
              plt.subplot(3,img_num,2*img_num+j+1)
              plt.imshow(saliency_head_imgs[0][j])
              plt.axis('off')
            plt.savefig('saliency.png')
          x1,x2,y1,y2,z1,z2, a1,a2 = sess.run([tf.reduce_sum(lateral_sort_of_confusion_table,0),tf.reduce_sum(head_sort_of_confusion_table,0),lateral_labels_num, head_labels_num,lateral_correct_one,head_correct_one, tf.reduce_sum(lateral_cmd_label, 0), tf.reduce_sum(head_cmd_label,0)])
          np_lateral_confusion = np_lateral_confusion + x1
          np_head_confusion    = np_head_confusion    + x2
          np_lateral_num = np_lateral_num + y1
          np_head_num    = np_head_num    + y2
          np_lateral_acc = np_lateral_acc + z1
          np_head_acc    = np_head_acc    + z2
          np_lateral_cmd += a1
          np_head_cmd    += a2
          printProgress(idx, test_batch_num, 'Progress:', 'Complete', 1, 50)
      for i in range(3):
        np_lateral_confusion[i,:] = np_lateral_confusion[i,:] / np_lateral_num[i]
        np_head_confusion[i,:]    = np_head_confusion[i,:]    / np_head_num[i]
        np_lateral_acc[i]         = np_lateral_acc[i] / np_lateral_num[i]
        np_head_acc[i]            = np_head_acc[i]    / np_head_num[i]
        np_lateral_cmd[i]         = np_lateral_cmd[i] / np_lateral_num[i]
        np_head_cmd[i]            = np_head_cmd[i]    / np_head_num[i]
      print('lateral_confusion')
      print(np_lateral_confusion)
      print('head_confusion')
      print(np_head_confusion)
      print('lateral accuracy')
      print(np_lateral_acc)
      print('head accuracy')
      print(np_head_acc)
      print('Lateral Command')
      print(np_lateral_cmd)
      print('head Command')
      print(np_head_cmd)
      sys.exit()

def compute_saliency_maps(logits, X, Y):
  correct_scores = tf.gather_nd(logits, tf.stack((tf.cast(tf.range(X.shape[0]),tf.int64),Y),axis=1))
  grad = tf.gradients(correct_scores, X)
  saliency_op = tf.reduce_max(tf.abs(grad), axis=4)
  return saliency_op  

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength= 100):
  formatStr = "{0:." + str(decimals) + "f}"
  percent = formatStr.format(100 * (iteration / float(total)))
  filledLength = int(round(barLength * iteration / float(total)))
  bar = '#' * filledLength + '-' * (barLength - filledLength)
  sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
  if iteration == total:
    sys.stdout.write('\n')
  sys.stdout.flush()

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  hparams = create_hparams(FLAGS.hparams)
  run_eval(
      run_dir=FLAGS.eval_dir,
      checkpoint_dir=FLAGS.checkpoint_dir,
      hparams=hparams)


if __name__ == '__main__':
  tf.app.run()
