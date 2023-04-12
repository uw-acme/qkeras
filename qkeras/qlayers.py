# Copyright 2019 Google LLC
#
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
#
# ==============================================================================
"""Definition of quantization package."""

# Some parts of the code were adapted from
#
# https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
#
# "Copyright (c) 2017, Bert Moons" where it applies
#
# and were implemented following several papers.
#
#    https://arxiv.org/pdf/1609.07061.pdf
#    https://arxiv.org/abs/1602.02830
#    https://arxiv.org/abs/1603.05279
#    https://arxiv.org/abs/1605.04711
#    https://ieeexplore.ieee.org/abstract/document/6986082
#    https://ieeexplore.ieee.org/iel4/78/5934/00229903.pdf
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import warnings

import collections
import math
import string

import numpy as np
import six
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.keras.layers import core
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras.utils import tf_utils as tf_utils2
from tensorflow.python.framework import smart_cond as tf_utils

from .quantizers import *
from .quantizers import _get_integer_bits
from .quantizers import get_quantizer
from tensorflow_model_optimization.python.core.sparsity.keras.prunable_layer import PrunableLayer

_CHR_IDX = string.ascii_lowercase


def get_auto_range_constraint_initializer(quantizer, constraint, initializer):
  """Get value range automatically for quantizer.

  Arguments:
   quantizer: A quantizer class in quantizers.py.
   constraint: A tf.keras constraint.
   initializer: A tf.keras initializer.

  Returns:
    a tuple (constraint, initializer), where
      constraint is clipped by Clip class in this file, based on the
      value range of quantizer.
      initializer is initializer contraint by value range of quantizer.
  """
  if quantizer is not None:
    constraint = get_constraint(constraint, quantizer)
    initializer = get_initializer(initializer)

    if initializer and initializer.__class__.__name__ not in ["Ones", "Zeros", 'QInitializer']:
      # we want to get the max value of the quantizer that depends
      # on the distribution and scale
      if not (hasattr(quantizer, "alpha") and
              isinstance(quantizer.alpha, six.string_types)):
        initializer = QInitializer(
            initializer, use_scale=True, quantizer=quantizer)
  return constraint, initializer


class QInitializer(Initializer):
  """Wraps around Keras initializer to provide a fanin scaling factor."""

  def __init__(self, initializer, use_scale, quantizer):
    self.initializer = initializer
    self.use_scale = use_scale
    self.quantizer = quantizer

    try:
      self.is_po2 = "po2" in quantizer.__class__.__name__
    except:
      self.is_po2 = False

  def __call__(self, shape, dtype=None):
    x = self.initializer(shape, dtype)

    max_x = np.max(abs(x))
    std_x = np.std(x)
    delta = self.quantizer.max() * 2**-self.quantizer.bits

    # delta is the minimum resolution of the number system.
    # we want to make sure we have enough values.
    if delta > std_x and hasattr(self.initializer, "scale"):
      q = self.quantizer(x)
      max_q = np.max(abs(q))
      scale = 1.0
      if max_q == 0.0:
        xx = np.mean(x * x)
        scale = self.quantizer.max() / np.sqrt(xx)
      else:
        qx = np.sum(q * x)
        qq = np.sum(q * q)

        scale = qq / qx

      self.initializer.scale *= max(scale, 1)
      x = self.initializer(shape, dtype)

    return np.clip(x, -self.quantizer.max(), self.quantizer.max())

  def get_config(self):
    return {
        "initializer": self.initializer,
        "use_scale": self.use_scale,
        "quantizer": self.quantizer,
    }

  @classmethod
  def from_config(cls, config):
    config = {
      'initializer' : get_initializer(config['initializer']),
      'use_scale'   : config['use_scale'],
      'quantizer'   : get_quantizer(config['quantizer'])}
    return cls(**config)

#
# Because it may be hard to get serialization from activation functions,
# we may be replacing their instantiation by QActivation in the future.
#


class QActivation(Layer, PrunableLayer):
  """Implements quantized activation layers."""

  def __init__(self, activation, **kwargs):

    super(QActivation, self).__init__(**kwargs)

    self.activation = activation

    if not isinstance(activation, six.string_types):
      self.quantizer = activation
      if hasattr(self.quantizer, "__name__"):
        self.__name__ = self.quantizer.__name__
      elif hasattr(self.quantizer, "name"):
        self.__name__ = self.quantizer.name
      elif hasattr(self.quantizer, "__class__"):
        self.__name__ = self.quantizer.__class__.__name__
      return

    self.__name__ = activation

    try:
      self.quantizer = get_quantizer(activation)
    except KeyError:
      raise ValueError("invalid activation '{}'".format(activation))

  def call(self, inputs):
    return self.quantizer(inputs)

  def get_config(self):
    config = {"activation": self.activation}
    base_config = super(QActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return str(self.activation)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_prunable_weights(self):
    return []


class QAdaptiveActivation(Layer, PrunableLayer):
  """[EXPERIMENTAL] Implements an adaptive quantized activation layer using EMA.

  This layer calculates an exponential moving average of min and max of the
  activation values to automatically determine the scale (integer bits) of
  the quantizer used in this layer.
  """

  def __init__(self,
               activation,
               total_bits,
               current_step=None,
               symmetric=True,
               quantization_delay=0,
               ema_freeze_delay=None,
               ema_decay=0.9999,
               per_channel=False,
               po2_rounding=False,
               relu_neg_slope=0.0,
               relu_upper_bound=None,
               **kwargs):
    """Initializes this QAdaptiveActivation layer.

    Args:
      activation: Str. The activation quantizer type to use for this activation
        layer, such as 'quantized_relu'. Should be a string with no params.
      total_bits: Int. The total bits that can be used by the quantizer
      current_step: tf.Variable specifying the current step in training.
        You can find this by passing model.optimizer.iterations
        (see tf.keras.optimizers.Optimizer.iterations). If set to None, the
        layer will attempt to estimate the current step itself, but please note
        that this number may not always match the optimizer step.
      symmetric: Bool. If to enforce symmetry about the origin in the quantized
        bit representation of the value. When using linear activation, this
        should be True for best results.
      quantization_delay: Int. How many training steps to wait until quantizing
        the activation values.
      ema_freeze_delay: Int. Steps to wait until stopping the update of the
        exponential moving average values. Set to None for an infinite delay.
      ema_decay: Float. The decay value used for exponential moving average (see
        tf.keras.backend.moving_average_update)
      per_channel: Bool. If to quantize the activation values on a
        per-channel basis.
      po2_rounding: Bool. If true, the EMA max value is rounded to the nearest
        power-of-2. If false, the EMA max value is rounded up (with ceil) to a
        power-of-two. These power-of-two operations are necessary to calculate
        the number of integer bits used in the quantizer, and the difference
        between using round and ceil trade off the quantizer's range and
        precision.
      relu_neg_slope: Float. Slope of the negative values in relu to enable the
        use of leaky relu. This parameter will only be used with the quantizer
        type quantized_relu. Set to 0.0 to use normal relu.
      relu_upper_bound: Float. The upper bound to use if the activation is set
        to relu. Set to None to not artificially set an upper bound. Pease note
        that this param is ignored if the activation is not quantized_relu
      **kwargs: Args passed to the Layer class.
    """
    super(QAdaptiveActivation, self).__init__(**kwargs)

    self.total_bits = total_bits
    self.symmetric = symmetric
    self.is_estimating_step_count = False  # If the layer should estimate its
                                           # own step count by incrementing it
                                           # every call.
    if isinstance(current_step, tf.Variable):
      self.step = current_step
    elif current_step is None:
      self.step = tf.Variable(-1, dtype=tf.int64)
      self.is_estimating_step_count = True
      print("[WARNING] QAdaptiveActivation is estimating it's own training "
            "step count, which may not always be the same as the true optimizer"
            " training step. To mitigate this, please set the current_step "
            "parameter when initializing QAdaptiveActivation", file=sys.stderr)
    else:
      self.step = tf.Variable(current_step, dtype=tf.int64)
      print("[WARNING] QAdaptiveActivation is disconnected from the optimizer "
            "current step, which may lead to incorrect training. If you wish to"
            " resume training, set this layer's self.step to the optimizer's "
            "tf.Variable current step", file=sys.stderr)
    self.quantization_delay = quantization_delay
    self.ema_freeze_delay = ema_freeze_delay
    self.will_ema_freeze = True if ema_freeze_delay else False
    self.ema_decay = ema_decay
    self.per_channel = per_channel
    self.po2_rounding = po2_rounding
    self.ema_min = None
    self.ema_max = None
    self.relu_neg_slope = relu_neg_slope
    self.relu_upper_bound = relu_upper_bound

    # Verify quantizer type is correct
    self.supported_quantizers = ["quantized_bits", "quantized_relu"]
    if activation not in self.supported_quantizers:
      raise ValueError(("Invalid activation {}. Activation quantizer may NOT "
                        "contain any parameters (they will be set automatically"
                        " by this layer), and only the quantizer types {} are "
                        "supported.").format(activation,
                                             self.supported_quantizers))

    # Get the quantizer associated with the activation
    try:
      self.quantizer = get_quantizer(activation)
    except KeyError:
      raise ValueError("Invalid activation '{}'".format(activation))

    # Check that the quantizer is supported
    if self.quantizer.__class__.__name__ not in self.supported_quantizers:
      raise ValueError("Unsupported activation quantizer '{}'".format(
          self.quantizer.__class__.__name__))

    # Set keep_negative
    if self.quantizer.__class__.__name__ == "quantized_relu":
      self.quantizer.is_quantized_clip = False  # Use relu_upper_bound instead
      if self.relu_upper_bound:
        self.quantizer.relu_upper_bound = self.relu_upper_bound
      self.quantizer.negative_slope = relu_neg_slope
      self.keep_negative = relu_neg_slope != 0.0
      self.quantizer.is_quantized_clip = False  # Use normal relu when qnoise=0
    elif self.quantizer.__class__.__name__ == "quantized_bits":
      self.keep_negative = True
      self.quantizer.keep_negative = True

    # If not using quantization delay, then print warning
    if self.quantization_delay < 1:
      print("[WARNING] If QAdaptiveActivation has the quantization_delay set "
            "to 0, then the moving averages will be heavily biased towards the "
            "initial quantizer configuration, which will likely prevent the "
            "model from converging. Consider a larger quantization_delay.",
            file=sys.stderr)

    self.activation = self.quantizer  # self.activation is used by QTools

  def build(self, input_shape):
    if self.will_ema_freeze:
      self.ema_freeze_delay = tf.constant(self.ema_freeze_delay, dtype=tf.int64)

    self.ema_decay = tf.constant(self.ema_decay, dtype=tf.float32)
    self.is_estimating_step_count = tf.constant(self.is_estimating_step_count,
                                                dtype=tf.bool)

    # Calculate the number of channels
    channel_index = -1 if K.image_data_format() == "channels_last" else 1
    if self.per_channel:
      input_shape_list = list(input_shape) if isinstance(
          input_shape, tuple) else input_shape.as_list()
      num_channels = tf.constant(input_shape_list[channel_index],
                                 shape=(1), dtype=tf.int64)
    else:
      num_channels = tf.constant(1, shape=(1), dtype=tf.int64)

    # Initialize the moving mins and max
    if self.ema_min is None or self.ema_max is None:
      self.ema_min = tf.Variable(tf.zeros(num_channels), name="ema_min",
                                 trainable=False)
      self.ema_max = tf.Variable(tf.zeros(num_channels), name="ema_max",
                                 trainable=False)

    # Determine the parameters for the quantizer
    self.quantizer.bits = self.total_bits

    # Set up the initial integer bits and quantizer params
    self.quantizer.integer = tf.Variable(tf.zeros(num_channels,
                                                  dtype=tf.int32),
                                         name="quantizer_integer_bits",
                                         trainable=False)
    integer_bits = _get_integer_bits(min_value=self.ema_min,
                                     max_value=self.ema_max,
                                     bits=self.total_bits,
                                     symmetric=self.symmetric,
                                     keep_negative=self.keep_negative,
                                     is_clipping=self.po2_rounding)
    self.quantizer.integer.assign(integer_bits)
    self.quantizer.alpha = 1.0  # Setting alpha to 1.0 allows the integer bits
                                # to serve as the scale
    self.quantizer.symmetric = self.symmetric
    self.quantization_delay = tf.constant(self.quantization_delay,
                                          dtype=tf.int64)

  def call(self, inputs, training=False):
    x = inputs
    training = training and self.trainable
    self.will_ema_freeze = self.will_ema_freeze and self.trainable

    # Update the step count if the optimizer step count is unknown
    self.step.assign_add(K.switch(
        tf.math.logical_and(self.is_estimating_step_count, training),
        tf.constant(1, tf.int64), tf.constant(0, tf.int64)))

    # Perform the quantization
    if training:
      # Calculate the qnoise, a scalar from 0 to 1 that represents the level of
      # quantization noise to use. At training start, we want no quantization,
      # so qnoise_factor = 0.0. After quantization_delay steps, we want normal
      # quantization, so qnoise_factor = 1.0.
      qnoise_factor = K.switch(
          tf.greater_equal(self.step, self.quantization_delay),
          lambda: tf.constant(1.0), lambda: tf.constant(0.0))
      self.quantizer.update_qnoise_factor(qnoise_factor)
      qx = self.quantizer(x)

    else:  # If not training, we always want to use full quantization
      self.quantizer.update_qnoise_factor(tf.constant(1.0))
      qx = self.quantizer(x)

    # Calculate the axis along where to find the min and max EMAs
    len_axis = len(x.shape)
    if len_axis > 1:
      if self.per_channel:
        if K.image_data_format() == "channels_last":
          axis = list(range(len_axis - 1))
        else:
          axis = list(range(1, len_axis))
      else:
        axis = list(range(len_axis))
    else:
      axis = [0]

    # Determine if freezing the EMA
    is_ema_training = tf.constant(training, dtype=tf.bool)
    if self.will_ema_freeze:
      is_ema_training = tf.cond(
          tf.greater(self.step, self.ema_freeze_delay),
          lambda: tf.constant(False), lambda: tf.constant(True))

    def update_branch():
      """ Update the moving average when is_ema_training is True."""

      # Set the qnoise factor to 0 to update the EMA using the unquantized input
      prev_qnoise_factor = tf.identity(self.quantizer.qnoise_factor)
      self.quantizer.update_qnoise_factor(tf.constant(0.0))

      # Update the EMA
      act_x = self.quantizer(x)  # act_x is the input after the activation
                                 # function, but before the quantizer. This is
                                 # done by using a qnoise_factor of 0
      new_min = tf.squeeze(K.min(act_x, axis=axis, keepdims=True))
      K.moving_average_update(self.ema_min, new_min, self.ema_decay)
      new_max = tf.squeeze(K.max(act_x, axis=axis, keepdims=True))
      K.moving_average_update(self.ema_max, new_max, self.ema_decay)

      # Reset the qnoise factor to the previous value
      self.quantizer.update_qnoise_factor(prev_qnoise_factor)

    # Update the moving average when is_ema_training is True
    tf_utils.smart_cond(
        is_ema_training, true_fn=update_branch, false_fn=lambda: None)

    # Set the integer bits for the quantizer
    integer_bits = _get_integer_bits(
        min_value=self.ema_min,
        max_value=self.ema_max,
        bits=self.total_bits,
        symmetric=self.symmetric,
        keep_negative=self.keep_negative,
        is_clipping=self.po2_rounding)
    self.quantizer.integer.assign(integer_bits)

    return qx

  # Override get_weights since we do not want ema_min or ema_max to be public
  def get_weights(self):
    return []

  # Override set_weights since we do not want ema_min or ema_max to be public
  def set_weights(self, weights):
    return

  def get_config(self):
    config = {
        "activation": self.quantizer.__class__.__name__,
        "total_bits": self.total_bits,
        "current_step": self.step.numpy(),
        "symmetric": self.symmetric,
        "quantization_delay": np.array(self.quantization_delay),
        "ema_freeze_delay": np.array(self.ema_freeze_delay),
        "ema_decay": np.array(self.ema_decay),
        "per_channel": self.per_channel,
        "po2_rounding": self.po2_rounding,
        "relu_neg_slope": self.relu_neg_slope
    }
    base_config = super(QAdaptiveActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    self.quantizer.integer_bits = np.array(self.quantizer)
    return str(self.quantizer)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_prunable_weights(self):
    return []


#
# Constraint class to clip weights and bias between -1 and 1 so that:
#    1. quantization approximation is symmetric (b = 0).
#    2. max(x) and min(x) are 1 and -1 respectively.
#
class Clip(Constraint):
  """Clips weight constraint."""

  # This function was modified from Keras minmaxconstraints.
  #
  # Constrains the weights to be between min/max values.
  #   min_value: the minimum norm for the incoming weights.
  #   max_value: the maximum norm for the incoming weights.
  #   constraint: previous constraint to be clipped.
  #   quantizer: quantizer to be applied to constraint.

  def __init__(self, min_value=0.0, max_value=1.0,
               constraint=None, quantizer=None):
    """Initializes Clip constraint class."""

    self.min_value = min_value
    self.max_value = max_value
    self.constraint = constraints.get(constraint)
    # Don't wrap yourself
    if isinstance(self.constraint, Clip):
      self.constraint = None
    self.quantizer = get_quantizer(quantizer)

  def __call__(self, w):
    """Clips values between min and max values."""
    if self.constraint:
      w = self.constraint(w)
      if self.quantizer:
        w = self.quantizer(w)
    w = tf.keras.backend.clip(w, self.min_value, self.max_value)
    return w

  def get_config(self):
    """Returns configuration of constraint class."""
    return {"min_value": self.min_value, "max_value": self.max_value}

  @classmethod
  def from_config(cls, config):
    if isinstance(config.get('constraint', None), Clip):
      config['constraint'] = None
    config['constraint'] = constraints.get(config.get('constraint', None))
    config['quantizer'] = get_quantizer(config.get('quantizer', None))
    return cls(**config)

#
# Definition of Quantized NN classes. These classes were copied
# from the equivalent layers in Keras, and we modified to apply quantization.
# Similar implementations can be seen in the references.
#



class QEinsumDense(Layer):
  """A layer that uses tf.einsum as the backing computation.

  This layer can perform einsum calculations of arbitrary dimensionality.

  Args:
    equation: An equation describing the einsum to perform. This equation must
      be a valid einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
      `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum axis
      expression sequence.
    output_shape: The expected shape of the output tensor (excluding the batch
      dimension and any dimensions represented by ellipses). You can specify
      None for any dimension that is unknown or can be inferred from the input
      shape.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied (that is, a "linear" activation: `a(x) = x`).
    bias_axes: A string containing the output dimension(s) to apply a bias to.
      Each character in the `bias_axes` string should correspond to a character
      in the output portion of the `equation` string.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Examples:

  **Biased dense layer with einsums**

  This example shows how to instantiate a standard Keras dense layer using
  einsum operations. This example is equivalent to
  `tf.keras.layers.Dense(64, use_bias=True)`.

  >>> layer = EinsumDense("ab,bc->ac", output_shape=64, bias_axes="c")
  >>> input_tensor = tf.keras.Input(shape=[32])
  >>> output_tensor = layer(input_tensor)
  >>> output_tensor
  <... shape=(None, 64) dtype=...>

  **Applying a dense layer to a sequence**

  This example shows how to instantiate a layer that applies the same dense
  operation to every element in a sequence. Here, the 'output_shape' has two
  values (since there are two non-batch dimensions in the output); the first
  dimension in the output_shape is `None`, because the sequence dimension `b`
  has an unknown shape.

  >>> layer = EinsumDense("abc,cd->abd",
  ...                     output_shape=(None, 64),
  ...                     bias_axes="d")
  >>> input_tensor = tf.keras.Input(shape=[32, 128])
  >>> output_tensor = layer(input_tensor)
  >>> output_tensor
  <... shape=(None, 32, 64) dtype=...>

  **Applying a dense layer to a sequence using ellipses**

  This example shows how to instantiate a layer that applies the same dense
  operation to every element in a sequence, but uses the ellipsis notation
  instead of specifying the batch and sequence dimensions.

  Because we are using ellipsis notation and have specified only one axis, the
  output_shape arg is a single value. When instantiated in this way, the layer
  can handle any number of sequence dimensions - including the case where no
  sequence dimension exists.

  >>> layer = EinsumDense("...x,xy->...y", output_shape=64, bias_axes="y")
  >>> input_tensor = tf.keras.Input(shape=[32, 128])
  >>> output_tensor = layer(input_tensor)
  >>> output_tensor
  <... shape=(None, 32, 64) dtype=...>
  """

  def __init__(self,
               equation,
               output_shape,
               activation=None,
               bias_axes=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_quantizer=None,
               bias_quantizer=None,
               kernel_range=None,
               bias_range=None,
               **kwargs):

    self.equation = equation
    if isinstance(output_shape, int):
      self.partial_output_shape = [output_shape]
    else:
      self.partial_output_shape = list(output_shape)

    """
    Old self variable declaration
    """

    if kernel_range is not None:
      warnings.warn("kernel_range is deprecated in QEinsum_dense layer.")

    if bias_range is not None:
      warnings.warn("bias_range is deprecated in QEinsum_dense layer.")

    self.kernel_range = kernel_range
    self.bias_range = bias_range

    self.bias_axes = bias_axes

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

    self.kernel_constraint, self.kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    if bias_axes is not None:
        self.bias_constraint, self.bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))

    self.activation = None

    # print(activation)
    # if activation is not None:
    #   self.activation = get_quantizer(activation)

    super(QEinsumDense, self).__init__(
      # equation=equation,
      # output_shape=output_shape,
      # activation=activation,
      # bias_axes=bias_axes,
      # kernel_initializer=kernel_initializer,
      # bias_initializer=bias_initializer,
      # kernel_regularizer=kernel_regularizer,
      # bias_regularizer=bias_regularizer,
      # activity_regularizer=activity_regularizer,
      # kernel_constraint=kernel_constraint,
      # bias_constraint=bias_constraint,
      **kwargs)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    shape_data = _analyze_einsum_string(self.equation,
                                        self.bias_axes,
                                        input_shape,
                                        self.partial_output_shape)
    kernel_shape, bias_shape, self.full_output_shape = shape_data
    self.kernel = self.add_weight(
        "kernel",
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)

    if bias_shape is not None:
      self.bias = self.add_weight(
          "bias",
          shape=bias_shape,
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    super(QEinsumDense, self).build(input_shape)

  def compute_output_shape(self, _):
    return tensor_shape.TensorShape(self.full_output_shape)

  def get_config(self):
    config = {
        "output_shape":
            self.partial_output_shape,
        "equation":
            self.equation,
        "activation":
            activations.serialize(self.activation),
        "bias_axes":
            self.bias_axes,
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "kernel_initializer":
            initializers.serialize(self.kernel_initializer),
        "bias_initializer":
            initializers.serialize(self.bias_initializer),
        "kernel_regularizer":
            regularizers.serialize(self.kernel_regularizer),
        "bias_regularizer":
            regularizers.serialize(self.bias_regularizer),
        "activity_regularizer":
            regularizers.serialize(self.activity_regularizer),
        "kernel_constraint":
            constraints.serialize(self.kernel_constraint),
        "bias_constraint":
            constraints.serialize(self.bias_constraint),
        "kernel_range": self.kernel_range,
        "bias_range": self.bias_range
    }
    base_config = super(QEinsumDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel

    ret = special_math_ops.einsum(self.equation, inputs, quantized_kernel)

    if self.bias is not None:
      if self.bias_quantizer:
        ret += self.bias_quantizer_internal(self.bias)
      else:
        ret += self.bias

    # if self.bias is not None:
    #   ret += self.bias

    print(self.activation)
    if self.activation is not None:
      ret = self.activation(ret)

    return ret


def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
  """Analyzes an einsum string to determine the required weight shape."""

  dot_replaced_string = re.sub(r"\.\.\.", "0", equation)

  # This is the case where no ellipses are present in the string.
  split_string = re.match("([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)",
                          dot_replaced_string)
  if split_string:
    return _analyze_split_string(split_string, bias_axes, input_shape,
                                 output_shape)

  # This is the case where ellipses are present on the left.
  split_string = re.match("0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)",
                          dot_replaced_string)
  if split_string:
    return _analyze_split_string(
        split_string, bias_axes, input_shape, output_shape, left_elided=True)

  # This is the case where ellipses are present on the right.
  split_string = re.match("([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0",
                          dot_replaced_string)
  if split_string:
    return _analyze_split_string(split_string, bias_axes, input_shape,
                                 output_shape)

  raise ValueError(
      "Invalid einsum equation '%s'. Equations must be in the form "
      "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...." % equation)


def _analyze_split_string(split_string,
                          bias_axes,
                          input_shape,
                          output_shape,
                          left_elided=False):
  """Analyze an pre-split einsum string to find the weight shape."""
  input_spec = split_string.group(1)
  weight_spec = split_string.group(2)
  output_spec = split_string.group(3)
  elided = len(input_shape) - len(input_spec)

  if isinstance(output_shape, int):
    output_shape = [output_shape]
  else:
    output_shape = list(output_shape)

  output_shape.insert(0, input_shape[0])

  if elided > 0 and left_elided:
    for i in range(1, elided):
      # We already inserted the 0th input dimension at dim 0, so we need to
      # start at location 1 here.
      output_shape.insert(1, input_shape[i])
  elif elided > 0 and not left_elided:
    for i in range(len(input_shape) - elided, len(input_shape)):
      output_shape.append(input_shape[i])

  if left_elided:
    # If we have beginning dimensions elided, we need to use negative indexing
    # to determine where in the input dimension our values are.
    input_dim_map = {
        dim: (i + elided) - len(input_shape) for i, dim in enumerate(input_spec)
    }
    # Because we've constructed the full output shape already, we don't need
    # to do negative indexing.
    output_dim_map = {dim: (i + elided) for i, dim in enumerate(output_spec)}
  else:
    input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
    output_dim_map = {dim: i for i, dim in enumerate(output_spec)}

  for i, dim in enumerate(input_spec):
    input_shape_at_dim = input_shape[i]
    if dim in output_dim_map:
      output_shape_at_dim = output_shape[output_dim_map[dim]]
      if (output_shape_at_dim is not None and
          output_shape_at_dim != input_shape_at_dim):
        raise ValueError(
            "Input shape and output shape do not match at shared "
            "dimension '%s'. Input shape is %s, and output shape "
            "is %s." %
            (dim, input_shape_at_dim, output_shape[output_dim_map[dim]]))

  for dim in output_spec:
    if dim not in input_spec and dim not in weight_spec:
      raise ValueError("Dimension '%s' was specified in the output '%s' but "
                       "has no corresponding dim in the input spec '%s' or "
                       "weight spec '%s.'" % (dim, output_spec, input_spec,
                                              output_spec))

  weight_shape = []
  for dim in weight_spec:
    if dim in input_dim_map:
      weight_shape.append(input_shape[input_dim_map[dim]])
    elif dim in output_dim_map:
      weight_shape.append(output_shape[output_dim_map[dim]])
    else:
      raise ValueError("Weight dimension '%s' did not have a match in either "
                       "the input spec '%s' or the output spec '%s'. For this "
                       "layer, the weight must be fully specified." %
                       (dim, input_spec, output_spec))

  if bias_axes is not None:
    num_left_elided = elided if left_elided else 0
    idx_map = {
        char: output_shape[i + num_left_elided]
        for i, char in enumerate(output_spec)
    }

    for char in bias_axes:
      if char not in output_spec:
        raise ValueError("Bias dimension '%s' was requested, but is not a part "
                         "of the output specification '%s'" %
                         (char, output_spec))

    first_bias_location = min([output_spec.find(char) for char in bias_axes])
    bias_output_spec = output_spec[first_bias_location:]

    bias_shape = [
        idx_map[char] if char in bias_axes else 1 for char in bias_output_spec
    ]

    if not left_elided:
      for _ in range(elided):
        bias_shape.append(1)
  else:
    bias_shape = None

  return weight_shape, bias_shape, output_shape


def _build_attention_equation(rank, attn_axes):
  """Builds einsum equations for the attention computation.

  Query, key, value inputs after projection are expected to have the shape as:
  `(bs, <non-attention dims>, <attention dims>, num_heads, channels)`.
  `bs` and `<non-attention dims>` are treated as `<batch dims>`.

  The attention operations can be generalized:
  (1) Query-key dot product:
  `(<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
  <key attention dims>, num_heads, channels) -> (<batch dims>,
  num_heads, <query attention dims>, <key attention dims>)`
  (2) Combination:
  `(<batch dims>, num_heads, <query attention dims>, <key attention dims>),
  (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
  <query attention dims>, num_heads, channels)`

  Args:
    rank: Rank of query, key, value tensors.
    attn_axes: List/tuple of axes, `[-1, rank)`,
      that attention will be applied to.

  Returns:
    Einsum equations.
  """
  target_notation = _CHR_IDX[:rank]
  # `batch_dims` includes the head dim.
  batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
  letter_offset = rank
  source_notation = ""
  for i in range(rank):
    if i in batch_dims or i == rank - 1:
      source_notation += target_notation[i]
    else:
      source_notation += _CHR_IDX[letter_offset]
      letter_offset += 1

  product_notation = "".join([target_notation[i] for i in batch_dims] +
                             [target_notation[i] for i in attn_axes] +
                             [source_notation[i] for i in attn_axes])
  dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                        product_notation)
  attn_scores_rank = len(product_notation)
  combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                    target_notation)
  return dot_product_equation, combine_equation, attn_scores_rank


def _build_proj_equation(free_dims, bound_dims, output_dims):
  """Builds an einsum equation for projections inside multi-head attention."""
  input_str = ""
  kernel_str = ""
  output_str = ""
  bias_axes = ""
  letter_offset = 0
  for i in range(free_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    output_str += char

  letter_offset += free_dims
  for i in range(bound_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    kernel_str += char

  letter_offset += bound_dims
  for i in range(output_dims):
    char = _CHR_IDX[i + letter_offset]
    kernel_str += char
    output_str += char
    bias_axes += char
  equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

  return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)

class QMultiHeadAttention(Layer):
  """MultiHeadAttention layer.

  This is an implementation of multi-headed attention as described in the paper
  "Attention is all you Need" (Vaswani et al., 2017).
  If `query`, `key,` `value` are the same, then
  this is self-attention. Each timestep in `query` attends to the
  corresponding sequence in `key`, and returns a fixed-width vector.

  This layer first projects `query`, `key` and `value`. These are
  (effectively) a list of tensors of length `num_attention_heads`, where the
  corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
  `(batch_size, <key/value dimensions>, key_dim)`,
  `(batch_size, <key/value dimensions>, value_dim)`.

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor.

  Finally, the result tensor with the last dimension as value_dim can take an
  linear projection and return.

  Examples:

  Performs 1D cross-attention over two sequence inputs with an attention mask.
  Returns the additional attention weights over heads.

  >>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
  >>> target = tf.keras.Input(shape=[8, 16])
  >>> source = tf.keras.Input(shape=[4, 16])
  >>> output_tensor, weights = layer(target, source,
  ...                                return_attention_scores=True)
  >>> print(output_tensor.shape)
  (None, 8, 16)
  >>> print(weights.shape)
  (None, 2, 8, 4)

  Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

  >>> layer = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))
  >>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
  >>> output_tensor = layer(input_tensor, input_tensor)
  >>> print(output_tensor.shape)
  (None, 5, 3, 4, 16)

  Args:
    num_heads: Number of attention heads.
    key_dim: Size of each attention head for query and key.
    value_dim: Size of each attention head for value.
    dropout: Dropout probability.
    use_bias: Boolean, whether the dense layers use bias vectors/matrices.
    output_shape: The expected shape of an output tensor, besides the batch and
      sequence dims. If not specified, projects back to the key feature dim.
    attention_axes: axes over which the attention is applied. `None` means
      attention over all axes, but batch, heads, and features.
    kernel_initializer: Initializer for dense layer kernels.
    bias_initializer: Initializer for dense layer biases.
    kernel_regularizer: Regularizer for dense layer kernels.
    bias_regularizer: Regularizer for dense layer biases.
    activity_regularizer: Regularizer for dense layer activity.
    kernel_constraint: Constraint for dense layer kernels.
    bias_constraint: Constraint for dense layer kernels.

  Call arguments:
    query: Query `Tensor` of shape `(B, T, dim)`.
    value: Value `Tensor` of shape `(B, S, dim)`.
    key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
      `value` for both `key` and `value`, which is the most common case.
    attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
      attention to certain positions. The boolean mask specifies which query
      elements can attend to which key elements, 1 indicates attention and 0
      indicates no attention. Broadcasting can happen for the missing batch
      dimensions and the head dimension.
    return_attention_scores: A boolean to indicate whether the output should
      be attention output if True, or (attention_output, attention_scores) if
      False. Defaults to False.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (no dropout).
      Defaults to either using the training mode of the parent layer/model,
      or False (inference) if there is no parent layer.

  Returns:
    attention_output: The result of the computation, of shape `(B, T, E)`,
      where `T` is for target sequence shapes and `E` is the query input last
      dimension if `output_shape` is `None`. Otherwise, the multi-head outputs
      are project to the shape specified by `output_shape`.
    attention_scores: [Optional] multi-head attention coeffients over
      attention axes.
  """

  def __init__(self,
               num_heads,
               key_dim,
               value_dim=None,
               dropout=0.0,
               use_bias=True,
               output_shape=None,
               attention_axes=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_quantizer=None,
               bias_quantizer=None,
               kernel_range=None,
               bias_range=None,
               **kwargs):


    if kernel_range is not None:
      warnings.warn("kernel_range is deprecated in QEinsum_dense layer.")

    if bias_range is not None:
      warnings.warn("bias_range is deprecated in QEinsum_dense layer.")

    self.kernel_range = kernel_range
    self.bias_range = bias_range

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    self._num_heads = num_heads
    self._key_dim = key_dim
    self._value_dim = value_dim if value_dim else key_dim
    self._dropout = dropout
    self._use_bias = use_bias
    self._output_shape = output_shape
    self._kernel_regularizer = regularizers.get(kernel_regularizer)
    self._bias_regularizer = regularizers.get(bias_regularizer)


    self._kernel_constraint, self._kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    if use_bias:
        self._bias_constraint, self._bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))

    if attention_axes is not None and not isinstance(attention_axes,
                                                     collections.abc.Sized):
      self._attention_axes = (attention_axes,)
    else:
      self._attention_axes = attention_axes
    self._built_from_signature = False
    self._query_shape, self._key_shape, self._value_shape = None, None, None


    super(QMultiHeadAttention, self).__init__(
        # num_heads=num_heads,
        # key_dim=key_dim,
        # value_dim=value_dim,
        # dropout=dropout,
        # use_bias=use_bias,
        # output_shape=output_shape,
        # attention_axes=attention_axes,
        # kernel_initializer=kernel_initializer,
        # bias_initializer=bias_initializer,
        # kernel_regularizer=kernel_regularizer,
        # bias_regularizer=bias_regularizer,
        # activity_regularizer=activity_regularizer,
        # kernel_constraint=kernel_constraint,
        # bias_constraint=bias_constraint,
        **kwargs)


  def get_config(self):
    config = {
        "num_heads": self._num_heads,
        "key_dim": self._key_dim,
        "value_dim": self._value_dim,
        "dropout": self._dropout,
        "use_bias": self._use_bias,
        "output_shape": self._output_shape,
        "attention_axes": self._attention_axes,
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "kernel_initializer":
            initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            constraints.serialize(self._bias_constraint),
        "query_shape": self._query_shape,
        "key_shape": self._key_shape,
        "value_shape": self._value_shape,
        "kernel_range": self.kernel_range,
        "bias_range": self.bias_range
    }
    base_config = super(QMultiHeadAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    # If the layer has a different build() function from the Keras default,
    # we need to trigger the customized build to create weights.
    query_shape = config.pop("query_shape")
    key_shape = config.pop("key_shape")
    value_shape = config.pop("value_shape")
    layer = cls(**config)
    if None in [query_shape, key_shape, value_shape]:
      logging.warning(
          "One of dimensions of the input shape is missing. It should have been"
          " memorized when the layer was serialized. "
          "%s is created without weights.",
          str(cls))
    else:
      layer._build_from_signature(query_shape, value_shape, key_shape)  # pylint: disable=protected-access
    return layer

  def _build_from_signature(self, query, value, key=None):
    """Builds layers and variables.

    Once the method is called, self._built_from_signature will be set to True.

    Args:
      query: Query tensor or TensorShape.
      value: Value tensor or TensorShape.
      key: Key tensor or TensorShape.
    """
    self._built_from_signature = True
    if hasattr(query, "shape"):
      self._query_shape = tensor_shape.TensorShape(query.shape)
    else:
      self._query_shape = tensor_shape.TensorShape(query)
    if hasattr(value, "shape"):
      self._value_shape = tensor_shape.TensorShape(value.shape)
    else:
      self._value_shape = tensor_shape.TensorShape(value)
    if key is None:
      self._key_shape = self._value_shape
    elif hasattr(key, "shape"):
      self._key_shape = tensor_shape.TensorShape(key.shape)
    else:
      self._key_shape = tensor_shape.TensorShape(key)

    common_kwargs = dict(
        activation = None,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint,
        kernel_quantizer=self.kernel_quantizer,
        bias_quantizer=self.bias_quantizer,
        kernel_range=self.kernel_range,
        bias_range=self.bias_range
        )
    # Any setup work performed only once should happen in an `init_scope`
    # to avoid creating symbolic Tensors that will later pollute any eager
    # operations.
    with tf_utils2.maybe_init_scope(self):
      free_dims = self._query_shape.rank - 1
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          free_dims, bound_dims=1, output_dims=2)
      self._query_dense = QEinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._key_dim]),
          bias_axes=bias_axes if self._use_bias else None,
          name="query",
          **common_kwargs)
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          self._key_shape.rank - 1, bound_dims=1, output_dims=2)
      self._key_dense = QEinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._key_dim]),
          bias_axes=bias_axes if self._use_bias else None,
          name="key",
          **common_kwargs)
      einsum_equation, bias_axes, output_rank = _build_proj_equation(
          self._value_shape.rank - 1, bound_dims=1, output_dims=2)
      self._value_dense = QEinsumDense(
          einsum_equation,
          output_shape=_get_output_shape(output_rank - 1,
                                         [self._num_heads, self._value_dim]),
          bias_axes=bias_axes if self._use_bias else None,
          name="value",
          **common_kwargs)

      # Builds the attention computations for multi-head dot product attention.
      # These computations could be wrapped into the keras attention layer once
      # it support mult-head einsum computations.
      self._build_attention(output_rank)
      self._output_dense = self._make_output_dense(
          free_dims, common_kwargs, "attention_output")

  def _make_output_dense(self, free_dims, common_kwargs, name=None):
    """Builds the output projection matrix.

    Args:
      free_dims: Number of free dimensions for einsum equation building.
      common_kwargs: Common keyword arguments for einsum layer.
      name: Name for the projection layer.

    Returns:
      Projection layer.
    """
    if self._output_shape:
      if not isinstance(self._output_shape, collections.abc.Sized):
        output_shape = [self._output_shape]
      else:
        output_shape = self._output_shape
    else:
      output_shape = [self._query_shape[-1]]
    einsum_equation, bias_axes, output_rank = _build_proj_equation(
        free_dims, bound_dims=2, output_dims=len(output_shape))
    return QEinsumDense(
        einsum_equation,
        output_shape=_get_output_shape(output_rank - 1, output_shape),
        bias_axes=bias_axes if self._use_bias else None,
        name=name,
        **common_kwargs)

  def _build_attention(self, rank):
    """Builds multi-head dot-product attention computations.

    This function builds attributes necessary for `_compute_attention` to
    costomize attention computation to replace the default dot-product
    attention.

    Args:
      rank: the rank of query, key, value tensors.
    """
    if self._attention_axes is None:
      self._attention_axes = tuple(range(1, rank - 2))
    else:
      self._attention_axes = tuple(self._attention_axes)
    self._dot_product_equation, self._combine_equation, attn_scores_rank = (
        _build_attention_equation(rank, attn_axes=self._attention_axes))
    norm_axes = tuple(
        range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
    self._softmax = get_quantizer(quantized_softmax(axis=norm_axes))
    self._dropout_layer = core.Dropout(rate=self._dropout)

  def _masked_softmax(self, attention_scores, attention_mask=None):
    # Normalize the attention scores to probabilities.
    # `attention_scores` = [B, N, T, S]
    if attention_mask is not None:
      # The expand dim happens starting from the `num_heads` dimension,
      # (<batch_dims>, num_heads, <query_attention_dims, key_attention_dims>)
      mask_expansion_axes = [-len(self._attention_axes) * 2 - 1]
      for _ in range(len(attention_scores.shape) - len(attention_mask.shape)):
        attention_mask = array_ops.expand_dims(
            attention_mask, axis=mask_expansion_axes)
    return self._softmax(attention_scores, attention_mask)

  def _compute_attention(self,
                         query,
                         key,
                         value,
                         attention_mask=None,
                         training=None):
    """Applies Dot-product attention with query, key, value tensors.

    This function defines the computation inside `call` with projected
    multi-head Q, K, V inputs. Users can override this function for customized
    attention implementation.

    Args:
      query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
      key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
      value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      attention_output: Multi-headed outputs of attention computation.
      attention_scores: Multi-headed attention weights.
    """
    # Note: Applying scalar multiply at the smaller end of einsum improves
    # XLA performance, but may introduce slight numeric differences in
    # the Transformer attention head.
    query = math_ops.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = special_math_ops.einsum(self._dot_product_equation, key,
                                               query)

    attention_scores = self._masked_softmax(attention_scores, attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_scores_dropout = self._dropout_layer(
        attention_scores, training=training)

    # `context_layer` = [B, T, N, H]
    attention_output = special_math_ops.einsum(self._combine_equation,
                                               attention_scores_dropout, value)
    return attention_output, attention_scores

  def call(self,
           query,
           value,
           key=None,
           attention_mask=None,
           return_attention_scores=False,
           training=None):
    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, T, N ,H]
    query = self._query_dense(query)

    # `key` = [B, S, N, H]
    key = self._key_dense(key)

    # `value` = [B, S, N, H]
    value = self._value_dense(value)

    attention_output, attention_scores = self._compute_attention(
        query, key, value, attention_mask, training)
    attention_output = self._output_dense(attention_output)

    if return_attention_scores:
      return attention_output, attention_scores
    return attention_output


class QDense(Dense, PrunableLayer):
  """Implements a quantized Dense layer."""

  # Most of these parameters follow the implementation of Dense in
  # Keras, with the exception of kernel_range, bias_range,
  # kernel_quantizer, bias_quantizer, and kernel_initializer.
  #
  # kernel_quantizer: quantizer function/class for kernel
  # bias_quantizer: quantizer function/class for bias
  # kernel_range/bias_ranger: for quantizer functions whose values
  #   can go over [-1,+1], these values are used to set the clipping
  #   value of kernels and biases, respectively, instead of using the
  #   constraints specified by the user.
  #
  # we refer the reader to the documentation of Dense in Keras for the
  # other parameters.

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer="he_normal",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               kernel_quantizer=None,
               bias_quantizer=None,
               kernel_range=None,
               bias_range=None,
               **kwargs):

    if kernel_range is not None:
      warnings.warn("kernel_range is deprecated in QDense layer.")

    if bias_range is not None:
      warnings.warn("bias_range is deprecated in QDense layer.")

    self.kernel_range = kernel_range
    self.bias_range = bias_range

    self.kernel_quantizer = kernel_quantizer
    self.bias_quantizer = bias_quantizer

    self.kernel_quantizer_internal = get_quantizer(self.kernel_quantizer)
    self.bias_quantizer_internal = get_quantizer(self.bias_quantizer)

    # optimize parameter set to "auto" scaling mode if possible
    if hasattr(self.kernel_quantizer_internal, "_set_trainable_parameter"):
      self.kernel_quantizer_internal._set_trainable_parameter()

    self.quantizers = [
        self.kernel_quantizer_internal, self.bias_quantizer_internal
    ]

    kernel_constraint, kernel_initializer = (
        get_auto_range_constraint_initializer(self.kernel_quantizer_internal,
                                              kernel_constraint,
                                              kernel_initializer))

    if use_bias:
      bias_constraint, bias_initializer = (
          get_auto_range_constraint_initializer(self.bias_quantizer_internal,
                                                bias_constraint,
                                                bias_initializer))
    if activation is not None:
      activation = get_quantizer(activation)

    super(QDense, self).__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

  def call(self, inputs):
    if self.kernel_quantizer:
      quantized_kernel = self.kernel_quantizer_internal(self.kernel)
    else:
      quantized_kernel = self.kernel
    output = tf.keras.backend.dot(inputs, quantized_kernel)
    if self.use_bias:
      if self.bias_quantizer:
        quantized_bias = self.bias_quantizer_internal(self.bias)
      else:
        quantized_bias = self.bias
      output = tf.keras.backend.bias_add(output, quantized_bias,
                                         data_format="channels_last")
    if self.activation is not None:
      output = self.activation(output)
    return output

  def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) >= 2
    assert input_shape[-1]
    output_shape = list(input_shape)
    output_shape[-1] = self.units
    return tuple(output_shape)

  def get_config(self):
    config = {
        "units": self.units,
        "activation": activations.serialize(self.activation),
        "use_bias": self.use_bias,
        "kernel_quantizer":
            constraints.serialize(self.kernel_quantizer_internal),
        "bias_quantizer":
            constraints.serialize(self.bias_quantizer_internal),
        "kernel_initializer":
            initializers.serialize(self.kernel_initializer),
        "bias_initializer":
            initializers.serialize(self.bias_initializer),
        "kernel_regularizer":
            regularizers.serialize(self.kernel_regularizer),
        "bias_regularizer":
            regularizers.serialize(self.bias_regularizer),
        "activity_regularizer":
            regularizers.serialize(self.activity_regularizer),
        "kernel_constraint":
            constraints.serialize(self.kernel_constraint),
        "bias_constraint":
            constraints.serialize(self.bias_constraint),
        "kernel_range": self.kernel_range,
        "bias_range": self.bias_range
    }
    base_config = super(QDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_quantization_config(self):
    return {
        "kernel_quantizer":
            str(self.kernel_quantizer_internal),
        "bias_quantizer":
            str(self.bias_quantizer_internal),
        "activation":
            str(self.activation),
        "units" : str(self.units)
    }

  def get_quantizers(self):
    return self.quantizers

  def get_prunable_weights(self):
    return [self.kernel]


def get_constraint(identifier, quantizer):
  """Gets the initializer.

  Args:
    identifier: A constraint, which could be dict, string, or callable function.
    quantizer: A quantizer class or quantization function

  Returns:
    A constraint class
  """
  if identifier:
    if isinstance(identifier, dict) and identifier['class_name'] == 'Clip':
      return Clip.from_config(identifier['config'])
    else:
      return constraints.get(identifier)
  else:
    max_value = max(1, quantizer.max()) if hasattr(quantizer, "max") else 1.0
    return Clip(-max_value, max_value, identifier, quantizer)

def get_initializer(identifier):
  """Gets the initializer.

  Args:
    identifier: An initializer, which could be dict, string, or callable function.

  Returns:
    A initializer class

  Raises:
    ValueError: An error occurred when quantizer cannot be interpreted.
  """
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    if identifier['class_name'] == 'QInitializer':
      return QInitializer.from_config(identifier['config'])
    else:
      return initializers.get(identifier)
  elif isinstance(identifier, six.string_types):
    return initializers.get(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError("Could not interpret initializer identifier: " +
                     str(identifier))
