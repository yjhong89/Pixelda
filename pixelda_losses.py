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

"""Defines the various loss functions in use by the PIXELDA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

slim = tf.contrib.slim


def add_domain_classifier_losses(end_points, hparams):
  """Adds losses related to the domain-classifier.

  Args:
    end_points: A map of network end point names to `Tensors`.
    hparams: The hyperparameters struct.

  Returns:
    loss: A `Tensor` representing the total task-classifier loss.
  """
  if hparams.domain_loss_weight == 0:
    tf.logging.info(
        'Domain classifier loss weight is 0, so not creating losses.')
    return 0, tf.no_op()

  # The domain prediction loss is minimized with respect to the domain
  # classifier features only. Its aim is to predict the domain of the images.
  # Note: 1 = 'real image' label, 0 = 'fake image' label
  
  # Original GAN objective
  #transferred_domain_loss = tf.losses.sigmoid_cross_entropy(
  #    multi_class_labels=tf.zeros_like(end_points['transferred_domain_logits']),
  #    logits=end_points['transferred_domain_logits'])
  #target_domain_loss = tf.losses.sigmoid_cross_entropy(
  #    multi_class_labels=tf.ones_like(end_points['target_domain_logits']),
  #    logits=end_points['target_domain_logits'])

  # Fisher GAN objective
  alpha = tf.get_variable('fisher_lambda', [], initializer=tf.zeros_initializer)
  
  e_q_f = tf.reduce_mean(end_points['transferred_domain_logits'])
  e_p_f = tf.reduce_mean(end_points['target_domain_logits'])
  e_q_f2 = tf.reduce_mean(tf.square(end_points['transferred_domain_logits']))
  e_p_f2 = tf.reduce_mean(tf.square(end_points['target_domain_logits']))

  constraint = (1 - (0.5*e_p_f2 + 0.5*e_q_f2))

  total_domain_loss = -1.0 * (e_p_f - e_q_f + alpha*constraint - hparams.rho/2 * constraint**2)
  tf.summary.scalar('Domain_loss_total', total_domain_loss)

  alpha_optimizer_op = tf.train.GradientDescentOptimizer(hparams.rho).minimize(-total_domain_loss, var_list=[alpha])


#  # LSGAN objective
#  transferred_domain_loss = tf.reduce_mean(tf.square(tf.sigmoid(end_points['transferred_domain_logits'])))
#  tf.summary.scalar('Domain_loss_transferred', transferred_domain_loss)
#
#  target_domain_loss = tf.reduce_mean(tf.square(1 - tf.sigmoid(end_points['target_domain_logits'])))
#  tf.summary.scalar('Domain_loss_target', target_domain_loss)
#
#  # Compute the total domain loss:
#  total_domain_loss = transferred_domain_loss + target_domain_loss
#  total_domain_loss = total_domain_loss * hparams.domain_loss_weight
#  tf.summary.scalar('Domain_loss_total', total_domain_loss)

  return total_domain_loss, alpha_optimizer_op


def _add_task_specific_losses(end_points, source_lateral_labels, source_head_labels, target_lateral_labels, target_head_labels, num_classes, hparams, add_summaries=False):
  """Adds losses related to the task-classifier.

  Args:
    end_points: A map of network end point names to `Tensors`.
    source_labels: A dictionary of output labels to `Tensors`.
    num_classes: The number of classes used by the classifier.
    hparams: The hyperparameters struct.
    add_summaries: Whether or not to add the summaries.

  Returns:
    loss: A `Tensor` representing the total task-classifier loss.
  """
  # TODO(ddohan): Make sure the l2 regularization is added to the loss

  lateral_one_hot_labels = slim.one_hot_encoding(tf.cast(source_lateral_labels, tf.int64), num_classes)
  head_one_hot_labels    = slim.one_hot_encoding(tf.cast(source_head_labels, tf.int64), num_classes)
  total_loss = 0

  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=lateral_one_hot_labels,
      logits=end_points['source_lateral_task_logits'],
      weights=hparams.source_lateral_task_loss_weight)
  if add_summaries:
    tf.summary.scalar('Lateral_Task_Classifier_Loss_Source', loss)
  total_loss = total_loss + loss
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=head_one_hot_labels,
      logits=end_points['source_head_task_logits'],
      weights=hparams.source_head_task_loss_weight)
  if add_summaries:
    tf.summary.scalar('Head_Task_Classifier_Loss_Source', loss)
  total_loss = total_loss + loss

  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=lateral_one_hot_labels,
    logits=end_points['transferred_lateral_task_logits'],
    weights=hparams.transferred_lateral_task_loss_weight)
  if add_summaries:
    tf.summary.scalar('Lateral_Task_Classfier_Loss_Transferred',loss)
  total_loss = total_loss + loss
  loss = tf.losses.softmax_cross_entropy(
    onehot_labels=head_one_hot_labels,
    logits=end_points['transferred_head_task_logits'],
    weights=hparams.transferred_head_task_loss_weight)
  if add_summaries:
    tf.summary.scalar('Head_Task_Classifier_Loss_Transferred',loss)
  total_loss = total_loss + loss

  if hparams.task_contrast_penalty != 0:
    lateral_one_hot_labels = tf.expand_dims(lateral_one_hot_labels,2)
    head_one_hot_labels    = tf.expand_dims(head_one_hot_labels,2)
    lateral_confusion = tf.multiply(lateral_one_hot_labels, tf.expand_dims(tf.nn.softmax(end_points['transferred_lateral_task_logits']),1))
    head_confusion    = tf.multiply(head_one_hot_labels, tf.expand_dims(tf.nn.softmax(end_points['transferred_head_task_logits']),1))
    loss = tf.reduce_sum(lateral_confusion[:,0,2] + lateral_confusion[:,2,0]) * hparams.task_contrast_penalty
    if add_summaries:
      tf.summary.scalar('lateral_penalty_loss',loss)
    total_loss = total_loss + loss
    loss = tf.reduce_sum(head_confusion[:,0,2] + head_confusion[:,2,0]) * hparams.task_contrast_penalty
    if add_summaries:
      tf.summary.scalar('head_penalty_loss', loss)
    total_loss = total_loss + loss 

  if hparams.task_classifier_entropy_weight != 0:
    loss = tf.reduce_sum(tf.multiply(tf.nn.softmax(end_points['transferred_lateral_task_logits']),tf.log(tf.nn.softmax(end_points['transferred_lateral_task_logits'])))) * hparams.task_classifier_entropy_weight
    if add_summaries:
      tf.summary.scalar('lateral_classifier_entropy_loss',loss)
    total_loss = total_loss + loss
    loss = tf.reduce_sum(tf.multiply(tf.nn.softmax(end_points['transferred_head_task_logits']),tf.log(tf.nn.softmax(end_points['transferred_head_task_logits'])))) * hparams.task_classifier_entropy_weight
    if add_summaries:
      tf.summary.scalar('head_classifier_entropy_loss',loss)
    total_loss = total_loss + loss


  if hparams.target_task_loss_in_d:
    target_lateral_one_hot_labels = slim.one_hot_encoding(tf.cast(target_lateral_labels, tf.int64), num_classes)
    target_head_one_hot_labels = slim.one_hot_encoding(tf.cast(target_head_labels, tf.int64), num_classes)
    loss = tf.losses.softmax_cross_entropy(
     onehot_labels=target_lateral_one_hot_labels,
     logits=end_points['target_lateral_task_logits_w_labels'],
     weights=hparams.target_lateral_task_loss_weight)
    if add_summaries:
      tf.summary.scalar('Lateral_Task_Classifier_Loss_Target', loss)
    total_loss = total_loss + loss

    loss = tf.losses.softmax_cross_entropy(
     onehot_labels=target_head_one_hot_labels,
     logits=end_points['target_head_task_logits_w_labels'],
     weights=hparams.target_head_task_loss_weight)
    if add_summaries:
      tf.summary.scalar('Head_Task_Classifier_Loss_Target', loss)
    total_loss = total_loss + loss
  if add_summaries:
    tf.summary.scalar('Task_Loss_Total', total_loss)

  return total_loss


def _transferred_similarity_loss(reconstructions,
                                 source_images, mask_images,
                                 weight=1.0,
                                 method='mse',
                                 max_diff=0.4,
                                 name='similarity'):
  """Computes a loss encouraging similarity between source and transferred.

  Args:
    reconstructions: A `Tensor` of shape [batch_size, height, width, channels]
    source_images: A `Tensor` of shape [batch_size, height, width, channels].
    weight: Multiple similarity loss by this weight before returning
    method: One of:
      mpse = Mean Pairwise Squared Error
      mse = Mean Squared Error
      hinged_mse = Computes the mean squared error using squared differences
        greater than hparams.transferred_similarity_max_diff
      hinged_mae = Computes the mean absolute error using absolute
        differences greater than hparams.transferred_similarity_max_diff.
    max_diff: Maximum unpenalized difference for hinged losses
    name: Identifying name to use for creating summaries


  Returns:
    A `Tensor` representing the transferred similarity loss.

  Raises:
    ValueError: if `method` is not recognized.
  """
  if weight == 0:
    return 0

  source_channels = source_images.shape.as_list()[-1]
  reconstruction_channels = reconstructions.shape.as_list()[-1]

  # Convert grayscale source to RGB if target is RGB
  if source_channels == 1 and reconstruction_channels != 1:
    source_images = tf.tile(source_images, [1, 1, 1, reconstruction_channels])
  if reconstruction_channels == 1 and source_channels != 1:
    reconstructions = tf.tile(reconstructions, [1, 1, 1, source_channels])

  if method == 'mpse':
    reconstruction_similarity_loss_fn = (
        tf.contrib.losses.mean_pairwise_squared_error)
  elif method == 'masked_mpse':

    def masked_mpse(predictions, labels, weight, mask):
      """Masked mpse assuming we have a depth to create a mask from."""
      mask = tf.to_float(tf.greater(mask,0.99))
      mask = tf.expand_dims(mask,3)
      mask = tf.tile(mask, [1, 1, 1, 3])
      predictions = tf.multiply(predictions, mask)
      labels = tf.multiply(labels, mask)
      tf.summary.image('mask',mask)
      tf.summary.image('masked_pred', predictions)
      tf.summary.image('masked_label', labels)
      return tf.contrib.losses.mean_pairwise_squared_error(
          predictions, labels, weight)

    reconstruction_similarity_loss_fn = masked_mpse
  elif method == 'mse':
    reconstruction_similarity_loss_fn = tf.contrib.losses.mean_squared_error
  elif method == 'hinged_mse':

    def hinged_mse(predictions, labels, weight):
      diffs = tf.square(predictions - labels)
      diffs = tf.maximum(0.0, diffs - max_diff)
      return tf.reduce_mean(diffs) * weight

    reconstruction_similarity_loss_fn = hinged_mse
  elif method == 'hinged_mae':

    def hinged_mae(predictions, labels, weight):
      diffs = tf.abs(predictions - labels)
      diffs = tf.maximum(0.0, diffs - max_diff)
      return tf.reduce_mean(diffs) * weight

    reconstruction_similarity_loss_fn = hinged_mae
  else:
    raise ValueError('Unknown reconstruction loss %s' % method)

  if method == 'masked_mpse':
    reconstruction_similarity_loss = reconstruction_similarity_loss_fn(
        reconstructions, source_images, weight, mask_images)
  else:
    reconstruction_similarity_loss = reconstruction_similarity_loss_fn(
        reconstructions, source_images, weight)

  name = '%s_Similarity_(%s)' % (name, method)
  tf.summary.scalar(name, reconstruction_similarity_loss)
  return reconstruction_similarity_loss


def mask_generated_loss(mask_images, end_points, weight = 1.0):
  reconstruction_similarity_loss_fn = (
    tf.contrib.losses.mean_pairwise_squared_error)

  #source_mask_generated_loss = reconstruction_similarity_loss_fn(
  #    mask_images, tf.squeeze(end_points['source_mask_generated'],3), weight)
  transferred_mask_generated_loss = reconstruction_similarity_loss_fn(
      mask_images, tf.squeeze(end_points['transferred_mask_generated'],3), weight)

  #name_source = 'source_mask_similarity'
  #tf.summary.scalar(name_source, source_mask_generated_loss)
  name_transferred = 'transferred_mask_similarity'
  tf.summary.scalar(name_transferred, transferred_mask_generated_loss)
  return transferred_mask_generated_loss
  #return source_mask_generated_loss + transferred_mask_generated_loss

def g_step_loss(source_images, mask_images, source_lateral_labels, source_head_labels, target_lateral_labels, target_head_labels, end_points, hparams, num_classes):
  """Configures the loss function which runs during the g-step.

  Args:
    source_images: A `Tensor` of shape [batch_size, height, width, channels].
    source_labels: A dictionary of `Tensors` of shape [batch_size]. Valid keys
      are 'class' and 'quaternion'.
    end_points: A map of the network end points.
    hparams: The hyperparameters struct.
    num_classes: Number of classes for classifier loss

  Returns:
    A `Tensor` representing a loss function.

  Raises:
    ValueError: if hparams.transferred_similarity_loss_weight is non-zero but
      hparams.transferred_similarity_loss is invalid.
  """
  generator_loss = 0

  ################################################################
  # Adds a loss which encourages the discriminator probabilities #
  # to be high (near one).
  ################################################################

  # As per the GAN paper, maximize the log probs, instead of minimizing
  # log(1-probs). Since we're minimizing, we'll minimize -log(probs) which is
  # the same thing.
  # Original GAN objective
  #style_transfer_loss = tf.losses.sigmoid_cross_entropy(
  #    logits=end_points['transferred_domain_logits'],
  #    multi_class_labels=tf.ones_like(end_points['transferred_domain_logits']),
  #    weights=hparams.style_transfer_loss_weight)

  style_transfer_loss = -tf.reduce_mean(end_points['transferred_domain_logits'])
  tf.summary.scalar('Style_Transfer_loss', style_transfer_loss)

#  # LSGAN objective
#  style_transfer_loss = -tf.reduce_mean(tf.square(tf.sigmoid(end_points['transferred_domain_logits']))) * hparams.style_transfer_loss_weight
#  tf.summary.scalar('Style_transfer_loss', style_transfer_loss)
  generator_loss += style_transfer_loss

  # Optimizes the style transfer network to produce transferred images similar
  # to the source images.
  generator_loss += _transferred_similarity_loss(
      end_points['transferred_images'],
      source_images, mask_images,
      weight=hparams.transferred_similarity_loss_weight,
      method=hparams.transferred_similarity_loss,
      name='transferred_similarity')

  # Optimizes the style transfer network to maximize classification accuracy.
  if source_lateral_labels is not None and source_head_labels is not None and hparams.task_tower_in_g_step:
    generator_loss += _add_task_specific_losses(
        end_points, source_lateral_labels, source_head_labels, 
        target_lateral_labels, target_head_labels, num_classes,
        hparams) * hparams.task_loss_in_g_weight

  if hparams.another_mask_loss:
    generator_loss += mask_generated_loss(
                               mask_images, end_points) * hparams.another_mask_loss_in_g_weight

  return generator_loss


def d_step_loss(end_points, mask_images, source_lateral_labels, source_head_labels, target_lateral_labels, target_head_labels, num_classes, hparams):
  """Configures the losses during the D-Step.

  Note that during the D-step, the model optimizes both the domain (binary)
  classifier and the task classifier.

  Args:
    end_points: A map of the network end points.
    source_labels: A dictionary of output labels to `Tensors`.
    num_classes: The number of classes used by the classifier.
    hparams: The hyperparameters struct.

  Returns:
    A `Tensor` representing the value of the D-step loss.
  """

  domain_classifier_loss, alpha_train_op = add_domain_classifier_losses(end_points, hparams)

  task_classifier_loss = 0
  if source_lateral_labels is not None and source_head_labels is not None:
    task_classifier_loss = _add_task_specific_losses(
        end_points, source_lateral_labels, source_head_labels, target_lateral_labels, target_head_labels, num_classes, hparams, add_summaries=True)

  if hparams.another_mask_loss:
    mask_loss = mask_generated_loss(mask_images,
                                     end_points) * hparams.another_mask_loss_in_d_weight
    return task_classifier_loss + domain_classifier_loss + mask_loss, alpha_train_op
  else:
    return task_classifier_loss + domain_classifier_loss, alpha_train_op
