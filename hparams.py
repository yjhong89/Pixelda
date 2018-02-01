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

"""Define model HParams."""
import tensorflow as tf


def create_hparams(hparam_string=None):
  """Create model hyperparameters. Parse nondefault from given string."""
  hparams = tf.contrib.training.HParams(
      # The name of the architecture to use.
      arch='resnet',
      lrelu_leakiness=0.2,
      batch_norm_decay=0.9,
      weight_decay=1e-5,
      normal_init_std=0.02,
      generator_kernel_size=3,
      discriminator_kernel_size=3,

      # Stop training after this many examples are processed
      # If none, train indefinitely
      num_training_examples=6400000,

      # Apply data augmentation to datasets
      # Applies only in training job
      augment_source_images=False,
      augment_target_images=False,

      # Discriminator
      # Number of filters in first layer of discriminator
      num_discriminator_filters=64,
      discriminator_conv_block_size=0,  # How many convs to have at each size
      discriminator_filter_factor=2.0,  # Multiply # filters by this each layer
      # Add gaussian noise with this stddev to every hidden layer of D
      discriminator_noise_stddev=0.2,  # lmetz: Start seeing results at >= 0.1
      # If true, add this gaussian noise to input images to D as well
      discriminator_image_noise=False,
      discriminator_first_stride=1,  # Stride in first conv of discriminator
      discriminator_do_pooling=False,  # If true, replace stride 2 with avg pool
      discriminator_dropout_keep_prob=0.35,  # keep probability for dropout

      # DCGAN Generator
      # Number of filters in generator decoder last layer (repeatedly halved
      # from 1st layer)
      num_decoder_filters=64,
      # Number of filters in generator encoder 1st layer (repeatedly doubled
      # after 1st layer)
      num_encoder_filters=64,

      # This is the shape to which the noise vector is projected (if we're
      # transferring from noise).
      # Write this way instead of [4, 4, 64] for hparam search flexibility
      projection_shape_size=12,
      projection_shape_channels=64,

      # Indicates the method by which we enlarge the spatial representation
      # of an image. Possible values include:
      # - resize_conv: Performs a nearest neighbor resize followed by a conv.
      # - conv2d_transpose: Performs a conv2d_transpose.
      upsample_method='resize_conv',

      # Visualization
      summary_steps=500,  # Output image summary every N steps

      input_mask = False,
      #### LOSS RELATED ####
      # g step loss = -logD(G(z)) * style_transfer_loss_weight + similarity_bt_source * transferred_similarity_loss_weight + transferred_task_loss * task_loss_in_g_weight
      # D_step_loss =  source_task_loss * source_task_loss_weight + transferred_task_loss * transferred_task_loss_weight + [-log(1-D(G(z)) - log(D(x))] * domain_loss_weight + target_task_loss * target_task_loss_weight
      rho = 1e-5,
      domain_loss_weight= 0.0, #20.0, #20.888,
      style_transfer_loss_weight = 0.0, #20.0, #20.1,
      source_head_task_loss_weight= 0.0,#12.0,
      source_lateral_task_loss_weight = 0.0, #12.0,
      transferred_head_task_loss_weight= 0.0, #12.0, 
      transferred_lateral_task_loss_weight= 0.0, #12.0,
      #### ADDED ####
      target_task_loss_in_d = True, 
      target_lateral_task_loss_weight = 12.0,
      target_head_task_loss_weight = 12.0,
      another_mask_loss = False,
      another_mask_loss_in_d_weight = 0.0, #0.07,
      another_mask_loss_in_g_weight = 0.0, #0.07,
      task_contrast_penalty = 0, #5,
      task_classifier_entropy_weight = 0.05,
      #### ##### ####
      # If set to True, the style transfer network also attempts to change its
      # weights to maximize the performance of the task tower. If set to False,
      # then the style transfer network only attempts to change its weights to
      # make the transferred images more likely according to the domain
      # classifier.
      task_tower_in_g_step= False,
      task_loss_in_g_weight=0.0, #2.0,  # Weight of task loss in G
      # The weight of the loss function encouraging the source and transferred
      # images to be similar. If set to 0, then the loss function is not used.
      transferred_similarity_loss_weight=0.0,

      # The type of loss used to encourage transferred and source image
      # similarity. Valid values include:
      #   mpse: Mean Pairwise Squared Error
      #   mse: Mean Squared Error
      #   hinged_mse: Computes the mean squared error using squared differences
      #     greater than hparams.transferred_similarity_max_diff
      #   hinged_mae: Computes the mean absolute error using absolute
      #     differences greater than hparams.transferred_similarity_max_diff.
      transferred_similarity_loss='masked_mpse',


      # The maximum allowable difference between the source and target images.
      # This value is used, in effect, to produce a hinge loss. Note that the
      # range of values should be between 0 and 1.
      transferred_similarity_max_diff=0.4,

      ###################################
      # Task Classifier Hyperparameters #
      ###################################

      # Which task-specific prediction tower to use. Possible choices are:
      #  none: No task tower.
      #  doubling_pose_estimator: classifier + quaternion regressor.
      #    [conv + pool]* + FC
      # Classifiers used in DSN paper:
      #  gtsrb: Classifier used for GTSRB
      #  svhn: Classifier used for SVHN
      #  mnist: Classifier used for MNIST
      #  pose_mini: Classifier + regressor used for pose_mini
      task_tower='source',
      weight_decay_task_classifier=1e-5,

      # Number of private layers in doubling_pose_estimator task tower
      num_private_layers=2,


      #########################################
      # 'simple` generator arch model hparams #
      #########################################
      simple_num_conv_layers=1,
      simple_conv_filters=8,

      #########################
      # Resnet Hyperparameters#
      #########################
      resnet_blocks=4,  # Number of resnet blocks
      resnet_filters=64,  # Number of filters per conv in resnet blocks
      # If true, add original input back to result of convolutions inside the
      # resnet arch. If false, it turns into a simple stack of conv/relu/BN
      # layers.
      resnet_residuals=True,

      #######################################
      # The residual / interpretable model. #
      #######################################
      res_int_blocks=2,  # The number of residual blocks.
      res_int_convs=2,  # The number of conv calls inside each block.
      res_int_filters=64,  # The number of filters used by each convolution.

      ####################
      # Latent variables #
      ####################
      # if true, then generate random noise and project to input for generator
      noise_channel=True,
      # The number of dimensions in the input noise vector.
      noise_dims=200,

      # If true, then one hot encode source image class and project as an
      # additional channel for the input to generator. This gives the generator
      # access to the class, which may help generation performance.
      condition_on_source_class=False,

      ########################
      # Loss Hyperparameters #
      ########################

      ########################################################################
      # Encourages the transferred images to be similar to the source images #
      # using a configurable metric.                                         #
      ########################################################################


      ################################
      # Optimization Hyperparameters #
      ################################
      learning_rate=0.0001,
      batch_size=32,
      lr_decay_steps=10000,
      lr_decay_rate=0.8,

      # Recomendation from the DCGAN paper:
      adam_beta1=0.5,
      clip_gradient_norm=5.0,

      # The number of times we run the discriminator train_op in a row.
      discriminator_steps=5,

      # The number of times we run the generator train_op in a row.
      generator_steps=2)

  if hparam_string:
    tf.logging.info('Parsing command line hparams: %s', hparam_string)
    import json
    json_file =  open(hparam_string,'r').read()
    hparams.parse_json(json_file)

  tf.logging.info('Final parsed hparams: %s', hparams.values())
  return hparams
