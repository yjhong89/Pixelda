
r"""Source simulator data to TFRecords of TF-Example protos.

The script should take about a minute to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import math

# Dependency imports
import numpy as np
from six.moves import urllib
import tensorflow as tf

import dataset_utils

tf.app.flags.DEFINE_string(
    'tosave_dir', '.',
    'The directory where the output TFRecords and temporary files are saved.')
tf.app.flags.DEFINE_string(
    'dataset_dir','data_target',
    'The directory where the data are saved')
tf.app.flags.DEFINE_integer(
    'num_data',2700,
    'data for each directory, 8352 for source dataset, 2700 for target dataset currently')
tf.app.flags.DEFINE_boolean(
    'whether_for_source',False,
    'if true, it is source, else, it is target')
FLAGS = tf.app.flags.FLAGS


# The number of images in the training set for each class.
_NUM_TRAIN_SAMPLES = int(math.floor(FLAGS.num_data*0.9))

# The number of images to be kept from the training set for the validation set for each class.
_NUM_VALIDATION = int(math.floor(FLAGS.num_data*0.01)) 

# The number of images in the test set for each class.
_NUM_TEST_SAMPLES = int(math.floor(FLAGS.num_data*0.09)) 

# Seed for repeatability.
_RANDOM_SEED = 0

# The names of the classes.
_CLASS_NAMES = [
    'left_left',
    'left_striaght',
    'left_right',
    'center_left',
    'center_straight',
    'center_right',
    'right_left',
    'right_straight',
    'right_right',
]


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB PNG data.
    self._decode_im_data = tf.placeholder(dtype=tf.string)
    if FLAGS.whether_for_source:
      self._decode_png = tf.image.decode_png(self._decode_im_data, channels=4)
    else:
      self._decode_jpeg = tf.image.decode_jpeg(self._decode_im_data, channels=3)

  def read_image_dims(self, sess, image_data):
    if FLAGS.whether_for_source:
      image = self.decode_png(sess, image_data)
    else:
      image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(
        self._decode_png, feed_dict={self._decode_im_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 4
    return image

  def decode_jpeg(self, sess, image_data):
    image = sess.run(
        self._decode_jpeg, feed_dict={self._decode_im_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _convert_dataset(split_name, filenames, filename_to_class_id, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'valid'.
    filenames: A list of absolute paths to png images.
    filename_to_class_id: A dictionary from filenames (strings) to class ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  print('Converting the {} split.'.format(split_name))
  png_directory = os.getcwd()

  with tf.Graph().as_default():
    image_reader = ImageReader()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      output_filename = _get_output_filename(dataset_dir, split_name)

      with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for idx in range(len(filenames)):
          # Read the filename:
          imgname  = filenames[idx]
          image_data = tf.gfile.FastGFile(
              os.path.join(png_directory, imgname), 'r').read()
          height, width = image_reader.read_image_dims(sess, image_data)
          class_id = filename_to_class_id[imgname]
          if FLAGS.whether_for_source:
            example = dataset_utils.image_to_tfexample(image_data, 'png', height,
                                                     width, class_id)
          else:
            example = dataset_utils.image_to_tfexample(image_data, 'jpeg', height,
                                                     width, class_id)
          tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _extract_labels(data_filenames):
  """Extract the labels into a dict of filenames to int labels.

    Returns:
    A dictionary of filenames to int labels.
  """
  print('Extracting labels')
  class_to_num = {'left_left' : 0, 'left_straight' : 1, 'left_right' : 2, 'center_left' : 3, 'center_straight' : 4,'center_right' :5, 'right_left' :6, 'right_straight' :7, 'right_right' :8}
  labels = {}
  for filename in data_filenames:
    belonged_class = filename.split('/')[1]
    labels[filename] = class_to_num[belonged_class]
  return labels


def _get_output_filename(dataset_dir, split_name):
  """Creates the output filename.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  if FLAGS.whether_for_source:
    return '%s/source_%s.tfrecord' % (dataset_dir, split_name)
  else:
    return '%s/target_%s.tfrecord' % (dataset_dir, split_name)


def _get_filenames(dataset_dir,start_num,end_num):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set PNG encoded MNIST-M images.

  Returns:
    A list of image file paths, relative to `dataset_dir`.
  """
  dir_list = ['left_left','left_straight','left_right','center_left','center_straight','center_right','right_left','right_straight','right_right']
  photo_filenames = []
  mask_filenames  = []
  for d in dir_list:
    try:
      filelist = os.listdir(os.path.join(dataset_dir,d))
      f_name = []
      list_temp = []
      idx = 0
      for name in filelist:
        if FLAGS.whether_for_source:
          if name[:6] == 'merged':
            list_temp.append(int(name[6:-4]))
            f_name.append(name)
        else:
          list_temp.append(idx)
          f_name.append(name)
      list_num  = np.argsort(list_temp)
      img_list = [f_name[zz] for zz in list_num]
      for idx in range(start_num,end_num):
        photo_filenames.append(os.path.join(dataset_dir,d,img_list[idx]))
    except:
      pass
  return photo_filenames


def run(tosave_dir,dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  train_filename = _get_output_filename(tosave_dir, 'train')
  testing_filename = _get_output_filename(tosave_dir, 'test')

  if tf.gfile.Exists(train_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # TODO(konstantinos): Add download and cleanup functionality

  train_validation_filenames = _get_filenames(dataset_dir,0,_NUM_TRAIN_SAMPLES + _NUM_VALIDATION)

  test_filenames             = _get_filenames(dataset_dir,_NUM_TRAIN_SAMPLES + _NUM_VALIDATION, _NUM_TRAIN_SAMPLES+ _NUM_VALIDATION+ _NUM_TEST_SAMPLES)

  # Divide into train and validation:
  random.seed(_RANDOM_SEED)
  random.shuffle(train_validation_filenames)
  train_filenames = train_validation_filenames[9*_NUM_VALIDATION:]
  validation_filenames = train_validation_filenames[:9*_NUM_VALIDATION]

  train_validation_filenames_to_class_ids = _extract_labels(train_validation_filenames)
  test_filenames_to_class_ids = _extract_labels(test_filenames)

  # Convert the train, validation, and test sets.
  _convert_dataset('train', train_filenames,
                   train_validation_filenames_to_class_ids, dataset_dir)
  _convert_dataset('valid', validation_filenames,
                   train_validation_filenames_to_class_ids, dataset_dir)
  _convert_dataset('test', test_filenames, test_filenames_to_class_ids,
                   dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the MNIST-M dataset!')


def main(_):
  assert FLAGS.dataset_dir
  run(FLAGS.tosave_dir,FLAGS.dataset_dir)


if __name__ == '__main__':
  tf.app.run()
