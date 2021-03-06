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

"""A factory-pattern class which returns image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

import source
import target
import transferred

slim = tf.contrib.slim


def get_dataset(dataset_name,
                split_name,
                dataset_dir,
                file_pattern=None,
                reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    dataset_name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A tf-slim `Dataset` class.

  Raises:
    ValueError: if `dataset_name` isn't recognized.
  """
  dataset_name_to_module = {'source': source, 'target': target, 'transferred': transferred}
  if dataset_name not in dataset_name_to_module:
    raise ValueError('Name of dataset unknown %s.' % dataset_name)

  return dataset_name_to_module[dataset_name].get_split(split_name, dataset_dir,
                                                        file_pattern, reader)


def provide_batch(dataset_name, split_name, dataset_dir, num_readers,
                  batch_size, num_preprocessing_threads):
  """Provides a batch of images and corresponding labels.

    Args:
    dataset_name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    num_readers: The number of readers used by DatasetDataProvider.
    batch_size: The size of the batch requested.
    num_preprocessing_threads: The number of preprocessing threads for
      tf.train.batch.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A batch of
      images: tensor of [batch_size, height, width, channels].
      labels: dictionary of labels.
  """
  if dataset_name == 'transferred':
    image, label = get_dataset(dataset_name, split_name, dataset_dir)

    labels = {}
    images, labels['classes'] = tf.train.shuffle_batch([image,label], batch_size=batch_size,
                                                       capacity=5 * batch_size,
                                                       num_threads = num_preprocessing_threads,
                                                       min_after_dequeue=10)
    labels['classes'] = slim.one_hot_encoding(labels['classes'], 9)
    labels['classes'] = tf.reshape(labels['classes'],[batch_size,9])
    return images, labels
  else:
    dataset = get_dataset(dataset_name, split_name, dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=True,
        num_readers=num_readers,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [image, label] = provider.get(['image', 'label'])
  
    # Convert images to float32 and scale into [-1,1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image -= 0.5
    image *= 2
 
    # Load the data.
    labels = {}
    images, labels['classes'] = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocessing_threads,
        capacity=5 * batch_size)
    labels['classes'] = slim.one_hot_encoding(labels['classes'],
                                              dataset.num_classes)

    #images = tf.image.resize_images(images, [180, 320])
    images = tf.image.resize_images(images, [90,160])
    return images, labels
