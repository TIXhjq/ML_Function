# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Recurrent layers for TF 2.0.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.backend import zeros_like, expand_dims
from tensorflow.keras.backend import reverse
from tensorflow.python.ops import tensor_array_ops,control_flow_util
import uuid
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util.tf_export import keras_export


# The following string constants are used by Defun approach for unified backend
# of LSTM and GRU.
_DEFUN_API_NAME_ATTRIBUTE = 'api_implements'
_DEFUN_DEVICE_ATTRIBUTE = 'api_preferred_device'
_CPU_DEVICE_NAME = 'CPU'
_GPU_DEVICE_NAME = 'GPU'

# The following number constants are used to represent the runtime of the defun
# backend function. Since the CPU/GPU implementation are mathematically same, we
# need some signal for the function to indicate which function is executed. This
# is for testing purpose to verify the correctness of swapping backend function.
_RUNTIME_UNKNOWN = 0
_RUNTIME_CPU = 1
_RUNTIME_GPU = 2


_DEFUN_API_NAME_ATTRIBUTE = 'api_implements'
_DEFUN_DEVICE_ATTRIBUTE = 'api_preferred_device'
_CPU_DEVICE_NAME = 'CPU'
_GPU_DEVICE_NAME = 'GPU'

# The following number constants are used to represent the runtime of the defun
# backend function. Since the CPU/GPU implementation are mathematically same, we
# need some signal for the function to indicate which function is executed. This
# is for testing purpose to verify the correctness of swapping backend function.
_RUNTIME_UNKNOWN = 0
_RUNTIME_CPU = 1
_RUNTIME_GPU = 2


@keras_export('keras.layers.GRUCell', v1=[])
class GRUCell(recurrent.GRUCell):
  """Cell class for the GRU layer.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This class processes one step within the whole time sequence input, whereas
  `tf.keras.layer.GRU` processes the whole sequence.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use. Default: hyperbolic tangent
      (`tanh`). If you pass None, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
      applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the
      linear transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 (default) will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications. Default: 2.
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before",
      True = "after" (default and CuDNN compatible).

  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`.
    states: A 2D tensor with shape of `[batch, units]`, which is the state from
      the previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.

  Examples:

  ```python
  inputs = np.random.random([32, 10, 8]).astype(np.float32)
  rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(4))

  output = rnn(inputs)  # The output has shape `[32, 4]`.

  rnn = tf.keras.layers.RNN(
      tf.keras.layers.GRUCell(4),
      return_sequences=True,
      return_state=True)

  # whole_sequence_output has shape `[32, 10, 4]`.
  # final_state has shape `[32, 4]`.
  whole_sequence_output, final_state = rnn(inputs)
  ```
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=2,
               reset_after=True,
               **kwargs):
    super(GRUCell, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        reset_after=reset_after,
        **kwargs)


@keras_export('keras.layers.GRU', v1=[])
class AUGRU(recurrent.DropoutRNNCellMixin, recurrent.GRU):
  """Gated Recurrent Unit - Cho et al. 2014.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Based on available runtime hardware and constraints, this layer
  will choose different implementations (cuDNN-based or pure-TensorFlow)
  to maximize the performance. If a GPU is available and all
  the arguments to the layer meet the requirement of the CuDNN kernel
  (see below for details), the layer will use a fast cuDNN implementation.

  The requirements to use the cuDNN implementation are:

  1. `activation` == `tanh`
  2. `recurrent_activation` == `sigmoid`
  3. `recurrent_dropout` == 0
  4. `unroll` is `False`
  5. `use_bias` is `True`
  6. `reset_after` is `True`
  7. Inputs are not masked or strictly right padded.

  There are two variants of the GRU implementation. The default one is based on
  [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to hidden
  state before matrix multiplication. The other one is based on
  [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.

  The second variant is compatible with CuDNNGRU (GPU-only) and allows
  inference on CPU. Thus it has separate biases for `kernel` and
  `recurrent_kernel`. To use this variant, set `'reset_after'=True` and
  `recurrent_activation='sigmoid'`.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: sigmoid (`sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
       weights matrix, used for the linear transformation of the recurrent
       state. Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications. Default: 2.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state in addition to the
      output. Default: `False`.
    go_backwards: Boolean (default `False`).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `[timesteps, batch, feature]`, whereas in the False case, it will be
      `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before",
      True = "after" (default and CuDNN compatible).

  Call arguments:
    inputs: A 3D tensor, with shape `[batch, timesteps, feature]`.
    mask: Binary tensor of shape `[samples, timesteps]` indicating whether
      a given timestep should be masked  (optional, defaults to `None`).
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used  (optional, defaults to `None`).
    initial_state: List of initial state tensors to be passed to the first
      call of the cell  (optional, defaults to `None` which causes creation
      of zero-filled initial state tensors).

  Examples:

  ```python
  inputs = np.random.random([32, 10, 8]).astype(np.float32)
  gru = tf.keras.layers.GRU(4)

  output = gru(inputs)  # The output has shape `[32, 4]`.

  gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)

  # whole_sequence_output has shape `[32, 10, 4]`.
  # final_state has shape `[32, 4]`.
  whole_sequence_output, final_state = gru(inputs)
  ```
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=2,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               time_major=False,
               reset_after=True,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.
    self._return_runtime = kwargs.pop('return_runtime', False)

    super(AUGRU, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        time_major=time_major,
        reset_after=reset_after,
        **kwargs)
    # CuDNN uses following setting by default and not configurable.
    self.could_use_cudnn = (
        activation == 'tanh' and recurrent_activation == 'sigmoid' and
        recurrent_dropout == 0 and not unroll and use_bias and
        reset_after and ops.executing_eagerly_outside_functions())

  def build(self, input_shape):
    super(AUGRU, self).build(input_shape)

    if not all(isinstance(v, resource_variable_ops.ResourceVariable)
               for v in self.weights):
      # Non-resource variables, such as DistributedVariables and
      # AutoCastVariables, do not work properly with the implementation
      # selector, which is used when cuDNN is used. However, by chance, such
      # variables happen to work in LSTM, so this check is only needed for GRU.
      # TODO(b/136512020): Make non-resource variables work with the
      # implementation selector.
      self.could_use_cudnn = False

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # The input should be dense, padded with zeros. If a ragged input is fed
    # into the layer, it is padded and the row lengths are used for masking.
    # [inputs,score]=input?s
    inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
    is_ragged_input = (row_lengths is not None)
    self._validate_args_if_ragged(is_ragged_input, mask)

    # GRU does not support constants. Ignore it during process.
    inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

    if isinstance(mask, list):
      mask = mask[0]

    input_shape = K.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]

    if not self.could_use_cudnn:
      kwargs = {'training': training}
      self._maybe_reset_cell_dropout_mask(self.cell)

      def step(cell_inputs, cell_states):
        return self.cell.call(cell_inputs, cell_states, **kwargs)

      last_output, outputs, states = rnn_backend.rnn(
          step,
          inputs,
          initial_state,
          constants=None,
          go_backwards=self.go_backwards,
          mask=mask,
          unroll=self.unroll,
          input_length=row_lengths if row_lengths is not None else timesteps,
          time_major=self.time_major,
          zero_output_for_mask=self.zero_output_for_mask)
      # This is a dummy tensor for testing purpose.
      runtime = _runtime(_RUNTIME_UNKNOWN)
    else:
      last_output, outputs, runtime, states = self._defun_gru_call(
          inputs, initial_state, training, mask, row_lengths)

    if self.stateful:
      updates = [state_ops.assign(self.states[0], states[0])]
      self.add_update(updates)

    if self.return_sequences:
      output = K.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
    else:
      output = last_output

    if self.return_state:
      return [output] + list(states)
    elif self._return_runtime:
      return output, runtime
    else:
      return output

  def _defun_gru_call(self, inputs, initial_state, training, mask,
                      sequence_lengths):
    # Use the new defun approach for backend implementation swap.
    # Note that different implementations need to have same function
    # signature, eg, the tensor parameters need to have same shape and dtypes.

    self.reset_dropout_mask()
    dropout_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    if dropout_mask is not None:
      inputs = inputs * dropout_mask[0]

    cudnn_gru_kwargs = {
        'inputs': inputs,
        'init_h': initial_state[0],
        'kernel': self.cell.kernel,
        'recurrent_kernel': self.cell.recurrent_kernel,
        'bias': self.cell.bias,
        'mask': mask,
        'time_major': self.time_major,
        'go_backwards': self.go_backwards,
        'sequence_lengths': sequence_lengths
    }
    normal_gru_kwargs = cudnn_gru_kwargs.copy()
    normal_gru_kwargs.update({
        'activation': self.activation,
        'recurrent_activation': self.recurrent_activation
    })

    if context.executing_eagerly():
      device_type = _get_context_device_type()
      can_use_gpu = (
          # Either user specified GPU or unspecified but GPU is available.
          (device_type == _GPU_DEVICE_NAME
           or (device_type is None and context.num_gpus() > 0))
          and
          (mask is None or is_sequence_right_padded(mask, self.time_major)))
      # Under eager context, check the device placement and prefer the
      if can_use_gpu:
        last_output, outputs, new_h, runtime = cudnn_gru(**cudnn_gru_kwargs)
      else:
        last_output, outputs, new_h, runtime = standard_gru(**normal_gru_kwargs)
    else:
      last_output, outputs, new_h, runtime = gru_with_backend_selection(
          **normal_gru_kwargs)

    states = [new_h]
    return last_output, outputs, runtime, states


def standard_gru(inputs, init_h, kernel, recurrent_kernel, bias, activation,
                 recurrent_activation, mask, time_major, go_backwards,
                 sequence_lengths):
  """GRU with standard kernel implementation.

  This implementation can be run on all types of hardware.

  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.

  Arguments:
    inputs: Input tensor of GRU layer.
    init_h: Initial state tensor for the cell output.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. The bias contains the
      combined input_bias and recurrent_bias.
    activation: Activation function to use for output.
    recurrent_activation: Activation function to use for hidden recurrent state.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  input_shape = K.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  input_bias, recurrent_bias = array_ops.unstack(bias)

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]

    # inputs projected by all gate matrices at once
    matrix_x = K.dot(cell_inputs, kernel)
    matrix_x = K.bias_add(matrix_x, input_bias)

    x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=1)

    # hidden state projected by all gate matrices at once
    matrix_inner = K.dot(h_tm1, recurrent_kernel)
    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)

    recurrent_z, recurrent_r, recurrent_h = array_ops.split(matrix_inner, 3,
                                                            axis=1)
    z = recurrent_activation(x_z + recurrent_z)
    print('--------gru fix attention begin---------')
    print(z)
    # if score:
    #     z=z*score
    #     print(z)
    print('--------gru fix attention end---------')
    r = recurrent_activation(x_r + recurrent_r)
    hh = activation(x_h + r * recurrent_h)

    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]

  last_output, outputs, new_states = rnn_backend(
      step,
      inputs, [init_h],
      constants=None,
      unroll=False,
      time_major=time_major,
      mask=mask,
      go_backwards=go_backwards,
      input_length=sequence_lengths
      # score=score
      if sequence_lengths is not None else timesteps)
  return last_output, outputs, new_states[0], _runtime(_RUNTIME_CPU)


def cudnn_gru(inputs, init_h, kernel, recurrent_kernel, bias, mask, time_major,
              go_backwards, sequence_lengths):
  """GRU with CuDNN implementation which is only available for GPU."""
  if not time_major and mask is None:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
    seq_axis, batch_axis = (0, 1)
  else:
    seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
  # For init_h, cuDNN expects one more dim of num_layers before or after batch
  # dim for time major or batch major inputs respectively
  init_h = array_ops.expand_dims(init_h, axis=seq_axis)

  weights = array_ops.split(kernel, 3, axis=1)
  weights += array_ops.split(recurrent_kernel, 3, axis=1)
  # Note that the bias was initialized as shape (2, 3 * units), flat it into
  # (6 * units)
  bias = array_ops.split(K.flatten(bias), 6)
  # Note that the gate order for CuDNN is different from the canonical format.
  # canonical format is [z, r, h], whereas CuDNN is [r, z, h]. The swap need to
  # be done for kernel, recurrent_kernel, input_bias, recurrent_bias.
  # z is update gate weights.
  # r is reset gate weights.
  # h is output gate weights.
  weights[0], weights[1] = weights[1], weights[0]
  weights[3], weights[4] = weights[4], weights[3]
  bias[0], bias[1] = bias[1], bias[0]
  bias[3], bias[4] = bias[4], bias[3]

  params = _canonical_to_params(
      weights=weights,
      biases=bias,
      shape=constant_op.constant([-1]),
      transpose_weights=True)

  if mask is not None:
    sequence_lengths = calculate_sequence_by_mask(mask, time_major)

  if sequence_lengths is not None:
    if go_backwards:
      # Three reversals are required. E.g.,
      # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
      # reversed_input_to_cudnn = [3, 2, 1, 0, 0]
      # output_from_cudnn = [6, 5, 4, 0, 0]
      # expected_output = [0, 0, 6, 5 ,4]
      inputs = array_ops.reverse_sequence_v2(
          inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
    outputs, h, _, _, _ = gen_cudnn_rnn_ops.cudnn_rnnv3(
        inputs,
        input_h=init_h,
        input_c=0,
        params=params,
        is_training=True,
        rnn_mode='gru',
        sequence_lengths=sequence_lengths,
        time_major=time_major)
    if go_backwards:
      outputs = array_ops.reverse_sequence_v2(
          outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
      outputs = array_ops.reverse(outputs, axis=[seq_axis])
  else:
    if go_backwards:
      # Reverse axis 0 since the input is already convert to time major.
      inputs = array_ops.reverse(inputs, axis=[0])
    outputs, h, _, _ = gen_cudnn_rnn_ops.cudnn_rnn(
        inputs, input_h=init_h, input_c=0, params=params, is_training=True,
        rnn_mode='gru')

  last_output = outputs[-1]
  if not time_major and mask is None:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = array_ops.squeeze(h, axis=seq_axis)

  # In the case of variable length input, the cudnn kernel will fill zeros for
  # the output, whereas the default keras behavior is to bring over the previous
  # output for t-1, so that in the return_sequence=False case, user can quickly
  # get the final effect output instead just 0s at the last timestep.
  # In order to mimic the default keras behavior, we copy the final h state as
  # the last_output, since it is numerically same as the output.
  if mask is not None:
    last_output = h

  return last_output, outputs, h, _runtime(_RUNTIME_GPU)


def gru_with_backend_selection(inputs, init_h, kernel, recurrent_kernel, bias,
                               mask, time_major, go_backwards, activation,
                               recurrent_activation, sequence_lengths):
  """Call the GRU with optimized backend kernel selection.

  Under the hood, this function will create two TF function, one with the most
  generic kernel and can run on all device condition, and the second one with
  CuDNN specific kernel, which can only run on GPU.

  The first function will be called with normal_lstm_params, while the second
  function is not called, but only registered in the graph. The Grappler will
  do the proper graph rewrite and swap the optimized TF function based on the
  device placement.

  Args:
    inputs: Input tensor of GRU layer.
    init_h: Initial state tensor for the cell output.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    activation: Activation function to use for output.
    recurrent_activation: Activation function to use for hidden recurrent state.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.

  Returns:
    List of output tensors, same as standard_gru.
  """
  params = {
      'inputs': inputs,
      'init_h': init_h,
      'kernel': kernel,
      'recurrent_kernel': recurrent_kernel,
      'bias': bias,
      'mask': mask,
      'time_major': time_major,
      'go_backwards': go_backwards,
      'activation': activation,
      'recurrent_activation': recurrent_activation,
      'sequence_lengths': sequence_lengths
  }

  def cudnn_gru_with_fallback(inputs, init_h, kernel, recurrent_kernel, bias,
                              mask, time_major, go_backwards, activation,
                              recurrent_activation, sequence_lengths):
    """Use CuDNN kernel when mask is none or strictly right padded."""
    if mask is None:
      return cudnn_gru(
          inputs=inputs,
          init_h=init_h,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def input_right_padded():
      return cudnn_gru(
          inputs=inputs,
          init_h=init_h,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def input_not_right_padded():
      return standard_gru(
          inputs=inputs,
          init_h=init_h,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          activation=activation,
          recurrent_activation=recurrent_activation,
          sequence_lengths=sequence_lengths)

    return control_flow_ops.cond(
        is_sequence_right_padded(mask, time_major),
        true_fn=input_right_padded,
        false_fn=input_not_right_padded)

  # Each time a `tf.function` is called, we will give it a unique
  # identifiable API name, so that Grappler won't get confused when it
  # sees multiple GRU layers added into same graph, and it will be able
  # to pair up the different implementations across them.
  api_name = 'gru_' + str(uuid.uuid4())
  defun_standard_gru = _generate_defun_backend(
      api_name, _CPU_DEVICE_NAME, standard_gru)
  defun_cudnn_gru = _generate_defun_backend(
      api_name, _GPU_DEVICE_NAME, cudnn_gru_with_fallback)

  # Call the normal GRU impl and register the CuDNN impl function. The
  # grappler will kick in during session execution to optimize the graph.
  last_output, outputs, new_h, runtime = defun_standard_gru(**params)
  function.register(defun_cudnn_gru, **params)
  return last_output, outputs, new_h, runtime


@keras_export('keras.layers.LSTMCell', v1=[])
class LSTMCell(recurrent.LSTMCell):
  """Cell class for the LSTM layer.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This class processes one step within the whole time sequence input, whereas
  `tf.keras.layer.LSTM` processes the whole sequence.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use. Default: hyperbolic tangent
      (`tanh`). If you pass `None`, no activation is applied (ie. "linear"
      activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs. Default: `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
      the forget gate at initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of smaller dot
      products and additions, whereas mode 2 (default) will batch them into
      fewer, larger operations. These modes will have different performance
      profiles on different hardware and for different applications. Default: 2.

  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`.
    states: List of 2 tensors that corresponding to the cell's units. Both of
      them have shape `[batch, units]`, the first tensor is the memory state
      from previous time step, the second tesnor is the carry state from
      previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.

  Examples:

  ```python
  inputs = np.random.random([32, 10, 8]).astype(np.float32)
  rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4))

  output = rnn(inputs)  # The output has shape `[32, 4]`.

  rnn = tf.keras.layers.RNN(
      tf.keras.layers.LSTMCell(4),
      return_sequences=True,
      return_state=True)

  # whole_sequence_output has shape `[32, 10, 4]`.
  # final_memory_state and final_carry_state both have shape `[32, 4]`.
  whole_sequence_output, final_memory_state, final_carry_state = rnn(inputs)
  ```
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=2,
               **kwargs):
    super(LSTMCell, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        **kwargs)


@keras_export('keras.layers.LSTM', v1=[])
class LSTM(recurrent.DropoutRNNCellMixin, recurrent.LSTM):
  """Long Short-Term Memory layer - Hochreiter 1997.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Based on available runtime hardware and constraints, this layer
  will choose different implementations (cuDNN-based or pure-TensorFlow)
  to maximize the performance. If a GPU is available and all
  the arguments to the layer meet the requirement of the CuDNN kernel
  (see below for details), the layer will use a fast cuDNN implementation.

  The requirements to use the cuDNN implementation are:

  1. `activation` == `tanh`
  2. `recurrent_activation` == `sigmoid`
  3. `recurrent_dropout` == 0
  4. `unroll` is `False`
  5. `use_bias` is `True`
  6. Inputs are not masked or strictly right padded.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
      is applied (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
      applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs. Default: `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
      the forget gate at initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    implementation: Implementation mode, either 1 or 2. Mode 1 will structure
      its operations as a larger number of smaller dot products and additions,
      whereas mode 2 will batch them into fewer, larger operations. These modes
      will have different performance profiles on different hardware and for
      different applications. Default: 2.
    return_sequences: Boolean. Whether to return the last output. in the output
      sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state in addition to the
      output. Default: `False`.
    go_backwards: Boolean (default `False`). If True, process the input sequence
      backwards and return the reversed sequence.
    stateful: Boolean (default `False`). If True, the last state for each sample
      at index i in a batch will be used as initial state for the sample of
      index i in the following batch.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `[timesteps, batch, feature]`, whereas in the False case, it will be
      `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    unroll: Boolean (default `False`). If True, the network will be unrolled,
      else a symbolic loop will be used. Unrolling can speed-up a RNN, although
      it tends to be more memory-intensive. Unrolling is only suitable for short
      sequences.

  Call arguments:
    inputs: A 3D tensor with shape `[batch, timesteps, feature]`.
    mask: Binary tensor of shape `[batch, timesteps]` indicating whether
      a given timestep should be masked (optional, defaults to `None`).
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used (optional, defaults to `None`).
    initial_state: List of initial state tensors to be passed to the first
      call of the cell (optional, defaults to `None` which causes creation
      of zero-filled initial state tensors).

  Examples:

  ```python
  inputs = np.random.random([32, 10, 8]).astype(np.float32)
  lstm = tf.keras.layers.LSTM(4)

  output = lstm(inputs)  # The output has shape `[32, 4]`.

  lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)

  # whole_sequence_output has shape `[32, 10, 4]`.
  # final_memory_state and final_carry_state both have shape `[32, 4]`.
  whole_sequence_output, final_memory_state, final_carry_state = lstm(inputs)
  ```
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=2,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               time_major=False,
               unroll=False,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.
    self.return_runtime = kwargs.pop('return_runtime', False)

    super(LSTM, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        time_major=time_major,
        unroll=unroll,
        **kwargs)

    self.state_spec = [
        InputSpec(shape=(None, dim)) for dim in (self.units, self.units)
    ]
    self.could_use_cudnn = (
        activation == 'tanh' and recurrent_activation == 'sigmoid' and
        recurrent_dropout == 0 and not unroll and use_bias and
        ops.executing_eagerly_outside_functions())

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # The input should be dense, padded with zeros. If a ragged input is fed
    # into the layer, it is padded and the row lengths are used for masking.
    inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
    is_ragged_input = (row_lengths is not None)
    self._validate_args_if_ragged(is_ragged_input, mask)

    # LSTM does not support constants. Ignore it during process.
    inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

    if isinstance(mask, list):
      mask = mask[0]

    input_shape = K.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]

    if not self.could_use_cudnn:
      # Fall back to use the normal LSTM.
      kwargs = {'training': training}
      self._maybe_reset_cell_dropout_mask(self.cell)

      def step(inputs, states):
        return self.cell.call(inputs, states, **kwargs)

      last_output, outputs, states = K.rnn(
          step,
          inputs,
          initial_state,
          constants=None,
          go_backwards=self.go_backwards,
          mask=mask,
          unroll=self.unroll,
          input_length=row_lengths if row_lengths is not None else timesteps,
          time_major=self.time_major,
          zero_output_for_mask=self.zero_output_for_mask)
      runtime = _runtime(_RUNTIME_UNKNOWN)
    else:
      # Use the new defun approach for backend implementation swap.
      # Note that different implementations need to have same function
      # signature, eg, the tensor parameters need to have same shape and dtypes.
      # Since the CuDNN has an extra set of bias, those bias will be passed to
      # both normal and CuDNN implementations.
      self.reset_dropout_mask()
      dropout_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
      if dropout_mask is not None:
        inputs = inputs * dropout_mask[0]
      cudnn_lstm_kwargs = {
          'inputs': inputs,
          'init_h': initial_state[0],
          'init_c': initial_state[1],
          'kernel': self.cell.kernel,
          'recurrent_kernel': self.cell.recurrent_kernel,
          'bias': self.cell.bias,
          'mask': mask,
          'time_major': self.time_major,
          'go_backwards': self.go_backwards,
          'sequence_lengths': row_lengths
      }
      normal_lstm_kwargs = cudnn_lstm_kwargs.copy()
      normal_lstm_kwargs.update({
          'activation': self.activation,
          'recurrent_activation': self.recurrent_activation
      })

      if context.executing_eagerly():
        device_type = _get_context_device_type()
        can_use_gpu = (
            # Either user specified GPU or unspecified but GPU is available.
            (device_type == _GPU_DEVICE_NAME
             or (device_type is None and context.num_gpus() > 0))
            and
            (mask is None or is_sequence_right_padded(mask, self.time_major)))
        # Under eager context, check the device placement and prefer the
        # GPU implementation when GPU is available.
        if can_use_gpu:
          last_output, outputs, new_h, new_c, runtime = cudnn_lstm(
              **cudnn_lstm_kwargs)
        else:
          last_output, outputs, new_h, new_c, runtime = standard_lstm(
              **normal_lstm_kwargs)
      else:
        (last_output, outputs, new_h, new_c,
         runtime) = lstm_with_backend_selection(**normal_lstm_kwargs)

      states = [new_h, new_c]

    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(state_ops.assign(self.states[i], states[i]))
      self.add_update(updates)

    if self.return_sequences:
      output = K.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
    else:
      output = last_output

    if self.return_state:
      return [output] + list(states)
    elif self.return_runtime:
      return output, runtime
    else:
      return output


def _canonical_to_params(weights, biases, shape, transpose_weights=False):
  """Utility function convert variable to CuDNN compatible parameter.

  Note that Keras weights for kernels are different from the CuDNN format. Eg.:

  ```
    Keras                 CuDNN
    [[0, 1, 2],  <--->  [[0, 2, 4],
     [3, 4, 5]]          [1, 3, 5]]
  ```

  If the input weights need to be in a unified format, then set
  `transpose_weights=True` to convert the weights.

  Args:
    weights: list of weights for the individual kernels and recurrent kernels.
    biases: list of biases for individual gate.
    shape: the shape for the converted variables that will be feed to CuDNN.
    transpose_weights: boolean, whether to transpose the weights.

  Returns:
    The converted weights that can be feed to CuDNN ops as param.
  """
  def convert(w):
    return array_ops.transpose(w) if transpose_weights else w

  weights = [array_ops.reshape(convert(x), shape) for x in weights]
  biases = [array_ops.reshape(x, shape) for x in biases]
  return array_ops.concat(weights + biases, axis=0)


def standard_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias,
                  activation, recurrent_activation, mask, time_major,
                  go_backwards, sequence_lengths):
  """LSTM with standard kernel implementation.

  This implementation can be run on all types for hardware.

  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.

  Note that the first half of the bias tensor should be ignored by this impl.
  The CuDNN impl need an extra set of input gate bias. In order to make the both
  function take same shape of parameter, that extra set of bias is also feed
  here.

  Args:
    inputs: input tensor of LSTM layer.
    init_h: initial state tensor for the cell output.
    init_c: initial state tensor for the cell hidden state.
    kernel: weights for cell kernel.
    recurrent_kernel: weights for cell recurrent kernel.
    bias: weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    activation: Activation function to use for output.
    recurrent_activation: Activation function to use for hidden recurrent state.
    mask: Boolean tensor for mask out the steps within sequence.
    time_major: boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    state_1: the cell hidden state, which has same shape as init_c.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  input_shape = K.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]  # previous memory state
    c_tm1 = cell_states[1]  # previous carry state

    z = K.dot(cell_inputs, kernel)
    z += K.dot(h_tm1, recurrent_kernel)
    z = K.bias_add(z, bias)

    z0, z1, z2, z3 = array_ops.split(z, 4, axis=1)

    i = recurrent_activation(z0)
    f = recurrent_activation(z1)
    c = f * c_tm1 + i * activation(z2)
    o = recurrent_activation(z3)

    h = o * activation(c)
    return h, [h, c]

  last_output, outputs, new_states = K.rnn(
      step,
      inputs, [init_h, init_c],
      constants=None,
      unroll=False,
      time_major=time_major,
      mask=mask,
      go_backwards=go_backwards,
      input_length=sequence_lengths
      if sequence_lengths is not None else timesteps)
  return (last_output, outputs, new_states[0], new_states[1],
          _runtime(_RUNTIME_CPU))


def cudnn_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias, mask,
               time_major, go_backwards, sequence_lengths):
  """LSTM with CuDNN implementation which is only available for GPU.

  Note that currently only right padded data is supported, or the result will be
  polluted by the unmasked data which should be filtered.

  Args:
    inputs: Input tensor of LSTM layer.
    init_h: Initial state tensor for the cell output.
    init_c: Initial state tensor for the cell hidden state.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.

  Returns:
    last_output: Output tensor for the last timestep, which has shape
      [batch, units].
    outputs: Output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: The cell output, which has same shape as init_h.
    state_1: The cell hidden state, which has same shape as init_c.
    runtime: Constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should not be used by user.
  """
  if not time_major and mask is None:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
    seq_axis, batch_axis = (0, 1)
  else:
    seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
  # For init_h and init_c, cuDNN expects one more dim of num_layers before or
  # after batch dim for time major or batch major inputs respectively
  init_h = array_ops.expand_dims(init_h, axis=seq_axis)
  init_c = array_ops.expand_dims(init_c, axis=seq_axis)

  weights = array_ops.split(kernel, 4, axis=1)
  weights += array_ops.split(recurrent_kernel, 4, axis=1)
  # CuDNN has an extra set of bias for inputs, we disable them (setting to 0),
  # so that mathematically it is same as the canonical LSTM implementation.
  full_bias = array_ops.concat((array_ops.zeros_like(bias), bias), 0)

  params = _canonical_to_params(
      weights=weights,
      biases=array_ops.split(full_bias, 8),
      shape=constant_op.constant([-1]),
      transpose_weights=True)

  if mask is not None:
    sequence_lengths = calculate_sequence_by_mask(mask, time_major)

  if sequence_lengths is not None:
    if go_backwards:
      # Three reversals are required. E.g.,
      # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
      # reversed_input_to_cudnn = [3, 2, 1, 0, 0]
      # output_from_cudnn = [6, 5, 4, 0, 0]
      # expected_output = [0, 0, 6, 5 ,4]
      inputs = array_ops.reverse_sequence_v2(
          inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
    outputs, h, c, _, _ = gen_cudnn_rnn_ops.cudnn_rnnv3(
        inputs,
        input_h=init_h,
        input_c=init_c,
        params=params,
        is_training=True,
        rnn_mode='lstm',
        sequence_lengths=sequence_lengths,
        time_major=time_major)
    if go_backwards:
      outputs = array_ops.reverse_sequence_v2(
          outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
      outputs = array_ops.reverse(outputs, axis=[seq_axis])
  else:
    # # Fill the array with shape [batch] with value of max timesteps.
    # sequence_length = array_ops.fill([array_ops.shape(inputs)[1]],
    #                                  array_ops.shape(inputs)[0])
    if go_backwards:
      # Reverse axis 0 since the input is already convert to time major.
      inputs = array_ops.reverse(inputs, axis=[0])
    outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
        inputs, input_h=init_h, input_c=init_c, params=params, is_training=True,
        rnn_mode='lstm')

  last_output = outputs[-1]
  if not time_major and mask is None:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = array_ops.squeeze(h, axis=seq_axis)
  c = array_ops.squeeze(c, axis=seq_axis)

  # In the case of variable length input, the cudnn kernel will fill zeros for
  # the output, whereas the default keras behavior is to bring over the previous
  # output for t-1, so that in the return_sequence=False case, user can quickly
  # get the final effect output instead just 0s at the last timestep.
  # In order to mimic the default keras behavior, we copy the final h state as
  # the last_output, since it is numerically same as the output.
  if mask is not None:
    last_output = h
  return last_output, outputs, h, c, _runtime(_RUNTIME_GPU)


def lstm_with_backend_selection(inputs, init_h, init_c, kernel,
                                recurrent_kernel, bias, mask, time_major,
                                go_backwards, activation, recurrent_activation,
                                sequence_lengths):
  """Call the LSTM with optimized backend kernel selection.

  Under the hood, this function will create two TF function, one with the most
  generic kernel and can run on all device condition, and the second one with
  CuDNN specific kernel, which can only run on GPU.

  The first function will be called with normal_lstm_params, while the second
  function is not called, but only registered in the graph. The Grappler will
  do the proper graph rewrite and swap the optimized TF function based on the
  device placement.

  Args:
    inputs: Input tensor of LSTM layer.
    init_h: Initial state tensor for the cell output.
    init_c: Initial state tensor for the cell hidden state.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    activation: Activation function to use for output.
    recurrent_activation: Activation function to use for hidden recurrent state.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.

  Returns:
    List of output tensors, same as standard_lstm.
  """
  params = {
      'inputs': inputs,
      'init_h': init_h,
      'init_c': init_c,
      'kernel': kernel,
      'recurrent_kernel': recurrent_kernel,
      'bias': bias,
      'mask': mask,
      'time_major': time_major,
      'go_backwards': go_backwards,
      'activation': activation,
      'recurrent_activation': recurrent_activation,
      'sequence_lengths': sequence_lengths
  }

  def cudnn_lstm_with_fallback(inputs, init_h, init_c, kernel, recurrent_kernel,
                               bias, mask, time_major, go_backwards, activation,
                               recurrent_activation, sequence_lengths):
    """Use CuDNN kernel when mask is none or strictly right padded."""
    if mask is None:
      return cudnn_lstm(
          inputs=inputs,
          init_h=init_h,
          init_c=init_c,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def input_right_padded():
      return cudnn_lstm(
          inputs=inputs,
          init_h=init_h,
          init_c=init_c,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def input_not_right_padded():
      return standard_lstm(
          inputs=inputs,
          init_h=init_h,
          init_c=init_c,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          activation=activation,
          recurrent_activation=recurrent_activation,
          sequence_lengths=sequence_lengths)

    return control_flow_ops.cond(
        is_sequence_right_padded(mask, time_major),
        true_fn=input_right_padded,
        false_fn=input_not_right_padded)

  # Each time a `tf.function` is called, we will give it a unique
  # identifiable API name, so that Grappler won't get confused when it
  # sees multiple LSTM layers added into same graph, and it will be able
  # to pair up the different implementations across them.
  api_name = 'lstm_' + str(uuid.uuid4())
  defun_standard_lstm = _generate_defun_backend(
      api_name, _CPU_DEVICE_NAME, standard_lstm)
  defun_cudnn_lstm = _generate_defun_backend(
      api_name, _GPU_DEVICE_NAME, cudnn_lstm_with_fallback)

  # Call the normal LSTM impl and register the CuDNN impl function. The
  # grappler will kick in during session execution to optimize the graph.
  last_output, outputs, new_h, new_c, runtime = defun_standard_lstm(
      **params)
  function.register(defun_cudnn_lstm, **params)

  return last_output, outputs, new_h, new_c, runtime


def is_sequence_right_padded(mask, time_major):
  """Check the mask tensor and see if it right padded.

  For CuDNN kernel, it uses the sequence length param to skip the tailing
  timestep. If the data is left padded, or not a strict right padding (has
  masked value in the middle of the sequence), then CuDNN kernel won't be work
  properly in those cases.

  Left padded data: [[False, False, True, True, True]].
  Right padded data: [[True, True, True, False, False]].
  Mixture of mask/unmasked data: [[True, False, True, False, False]].

  Note that for the mixed data example above, the actually data RNN should see
  are those 2 Trues (index 0 and 2), the index 1 False should be ignored and not
  pollute the internal states.

  Args:
    mask: the Boolean tensor with shape [batch, timestep] or [timestep, batch]
      when time_major is True.
    time_major: Boolean, whether the input mask is time major or batch major.

  Returns:
    boolean scalar tensor, whether the mask is strictly right padded.
  """
  if time_major:
    mask = array_ops.transpose(mask)
  max_seq_length = array_ops.shape(mask)[1]
  count_of_true = math_ops.reduce_sum(math_ops.cast(mask, dtypes.int32), axis=1)
  right_padded_mask = array_ops.sequence_mask(
      count_of_true, maxlen=max_seq_length)
  return math_ops.reduce_all(math_ops.equal(mask, right_padded_mask))


def calculate_sequence_by_mask(mask, time_major):
  """Calculate the sequence length tensor (1-D) based on the masking tensor.

  The masking tensor is a 2D boolean tensor with shape [batch, timestep]. For
  any timestep that should be masked, the corresponding field will be False.
  Consider the following example:
    a = [[True, True, False, False],
         [True, True, True, False]]
  It is a (2, 4) tensor, and the corresponding sequence length result should be
  1D tensor with value [2, 3]. Note that the masking tensor must be right
  padded that could be checked by, e.g., `is_sequence_right_padded()`.

  Args:
    mask: Boolean tensor with shape [batch, timestep] or [timestep, batch] if
      time_major=True.
    time_major: Boolean, which indicates whether the mask is time major or batch
      major.
  Returns:
    sequence_length: 1D int32 tensor.
  """
  timestep_index = 0 if time_major else 1
  return math_ops.reduce_sum(math_ops.cast(mask, dtypes.int32),
                             axis=timestep_index)


def _generate_defun_backend(unique_api_name, preferred_device, func):
  function_attributes = {
      _DEFUN_API_NAME_ATTRIBUTE: unique_api_name,
      _DEFUN_DEVICE_ATTRIBUTE: preferred_device,
  }
  return function.defun_with_attributes(func=func,
                                        attributes=function_attributes,
                                        autograph=False)


def _get_context_device_type():
  """Parse the current context and return the device type, eg CPU/GPU."""
  current_device = context.context().device_name
  if current_device is None:
    return None
  return device.DeviceSpec.from_string(current_device).device_type


def _runtime(runtime_name):
  with ops.device('/cpu:0'):
    return constant_op.constant(
        runtime_name, dtype=dtypes.float32, name='runtime')


@keras_export('keras.backend.rnn')
def rnn_backend(step_function,
        inputs,
        initial_states,
        score,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None,
        time_major=False,
        zero_output_for_mask=False, dtypes_module=None):
  """Iterates over the time dimension of a tensor.

  Arguments:
      step_function: RNN step function.
          Args;
              input; Tensor with shape `(samples, ...)` (no time dimension),
                  representing input for the batch of samples at a certain
                  time step.
              states; List of tensors.
          Returns;
              output; Tensor with shape `(samples, output_dim)`
                  (no time dimension).
              new_states; List of tensors, same length and shapes
                  as 'states'. The first state in the list must be the
                  output tensor at the previous timestep.
      inputs: Tensor of temporal data of shape `(samples, time, ...)`
          (at least 3D), or nested tensors, and each of which has shape
          `(samples, time, ...)`.
      initial_states: Tensor with shape `(samples, state_size)`
          (no time dimension), containing the initial values for the states used
          in the step function. In the case that state_size is in a nested
          shape, the shape of initial_states will also follow the nested
          structure.
      go_backwards: Boolean. If True, do the iteration over the time
          dimension in reverse order and return the reversed sequence.
      mask: Binary tensor with shape `(samples, time, 1)`,
          with a zero for every element that is masked.
      constants: List of constant values passed at each step.
      unroll: Whether to unroll the RNN or to use a symbolic `while_loop`.
      input_length: An integer or a 1-D Tensor, depending on whether
          the time dimension is fixed-length or not. In case of variable length
          input, it is used for masking in case there's no mask specified.
      time_major: Boolean. If true, the inputs and outputs will be in shape
          `(timesteps, batch, ...)`, whereas in the False case, it will be
          `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
          efficient because it avoids transposes at the beginning and end of the
          RNN calculation. However, most TensorFlow data is batch-major, so by
          default this function accepts input and emits output in batch-major
          form.
      zero_output_for_mask: Boolean. If True, the output for masked timestep
          will be zeros, whereas in the False case, output from previous
          timestep is returned.

  Returns:
      A tuple, `(last_output, outputs, new_states)`.
          last_output: the latest output of the rnn, of shape `(samples, ...)`
          outputs: tensor with shape `(samples, time, ...)` where each
              entry `outputs[s, t]` is the output of the step function
              at time `t` for sample `s`.
          new_states: list of tensors, latest states returned by
              the step function, of shape `(samples, ...)`.

  Raises:
      ValueError: if input dimension is less than 3.
      ValueError: if `unroll` is `True` but input timestep is not a fixed
      number.
      ValueError: if `mask` is provided (not `None`) but states is not provided
          (`len(states)` == 0).
  """

  def swap_batch_timestep(input_t):
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return array_ops.transpose(input_t, axes)

  if not time_major:
    inputs = tf.nest.map_structure(swap_batch_timestep, inputs)

  flatted_inputs = nest.flatten(inputs)
  time_steps = flatted_inputs[0].shape[0]
  batch = flatted_inputs[0].shape[1]
  time_steps_t = array_ops.shape(flatted_inputs[0])[0]

  for input_ in flatted_inputs:
    input_.shape.with_rank_at_least(3)

  if mask is not None:
    if mask.dtype != dtypes_module.bool:
      mask = math_ops.cast(mask, dtypes_module.bool)
    if len(mask.shape) == 2:
      mask = expand_dims(mask)
    if not time_major:
      mask = swap_batch_timestep(mask)

  if constants is None:
    constants = []

  # tf.where needs its condition tensor to be the same shape as its two
  # result tensors, but in our case the condition (mask) tensor is
  # (nsamples, 1), and inputs are (nsamples, ndimensions) or even more.
  # So we need to broadcast the mask to match the shape of inputs.
  # That's what the tile call does, it just repeats the mask along its
  # second dimension n times.
  def _expand_mask(mask_t, input_t, fixed_dim=1):
    if nest.is_sequence(mask_t):
      raise ValueError('mask_t is expected to be tensor, but got %s' % mask_t)
    if nest.is_sequence(input_t):
      raise ValueError('input_t is expected to be tensor, but got %s' % input_t)
    rank_diff = len(input_t.shape) - len(mask_t.shape)
    for _ in range(rank_diff):
      mask_t = array_ops.expand_dims(mask_t, -1)
    multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
    return array_ops.tile(mask_t, multiples)

  if unroll:
    if not time_steps:
      raise ValueError('Unrolling requires a fixed number of timesteps.')
    states = tuple(initial_states)
    successive_states = []
    successive_outputs = []

    # Process the input tensors. The input tensor need to be split on the
    # time_step dim, and reverse if go_backwards is True. In the case of nested
    # input, the input is flattened and then transformed individually.
    # The result of this will be a tuple of lists, each of the item in tuple is
    # list of the tensor with shape (batch, feature)
    def _process_single_input_t(input_t):
      input_t = array_ops.unstack(input_t)  # unstack for time_step dim
      if go_backwards:
        input_t.reverse()
      return input_t

    if nest.is_sequence(inputs):
      processed_input = nest.map_structure(_process_single_input_t, inputs)
    else:
      processed_input = (_process_single_input_t(inputs),)

    def _get_input_tensor(time):
      inp = [t_[time] for t_ in processed_input]
      return nest.pack_sequence_as(inputs, inp)

    if mask is not None:
      mask_list = array_ops.unstack(mask)
      if go_backwards:
        mask_list.reverse()

    #---------------fix attention----------------------
      for idx_,i in enumerate(range(time_steps)):
        inp = _get_input_tensor(i)
        mask_t = mask_list[i]
        output, new_states = step_function(inp,
                                           tuple(states) + tuple(constants),score=score[idx_])
    # ---------------fix attention----------------------
        tiled_mask_t = _expand_mask(mask_t, output)

        if not successive_outputs:
          prev_output = zeros_like(output)
        else:
          prev_output = successive_outputs[-1]

        output = array_ops.where_v2(tiled_mask_t, output, prev_output)

        flat_states = nest.flatten(states)
        flat_new_states = nest.flatten(new_states)
        tiled_mask_t = tuple(_expand_mask(mask_t, s) for s in flat_states)
        flat_final_states = tuple(
            array_ops.where_v2(m, s, ps)
            for m, s, ps in zip(tiled_mask_t, flat_new_states, flat_states))
        states = nest.pack_sequence_as(states, flat_final_states)

        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

      if zero_output_for_mask:
        last_output = array_ops.where_v2(
            _expand_mask(mask_list[-1], last_output), last_output,
            zeros_like(last_output))
        outputs = array_ops.where_v2(
            _expand_mask(mask, outputs, fixed_dim=2), outputs,
            zeros_like(outputs))

    else:  # mask is None
      for i in range(time_steps):
        inp = _get_input_tensor(i)
        output, states = step_function(inp, tuple(states) + tuple(constants))
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops.stack(successive_outputs)

  else:  # Unroll == False
    states = tuple(initial_states)

    # Create input tensor array, if the inputs is nested tensors, then it will
    # be flattened first, and tensor array will be created one per flattened
    # tensor.
    input_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=inp.dtype,
            size=time_steps_t,
            tensor_array_name='input_ta_%s' % i)
        for i, inp in enumerate(flatted_inputs))
    input_ta = tuple(
        ta.unstack(input_) if not go_backwards else ta
        .unstack(reverse(input_, 0))
        for ta, input_ in zip(input_ta, flatted_inputs))

    # Get the time(0) input and compute the output for that, the output will be
    # used to determine the dtype of output tensor array. Don't read from
    # input_ta due to TensorArray clear_after_read default to True.
    input_time_zero = nest.pack_sequence_as(inputs,
                                            [inp[0] for inp in flatted_inputs])
    # output_time_zero is used to determine the cell output shape and its dtype.
    # the value is discarded.
    output_time_zero, _ = step_function(
        input_time_zero, tuple(initial_states) + tuple(constants))
    output_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=out.dtype,
            size=time_steps_t,
            element_shape=out.shape,
            tensor_array_name='output_ta_%s' % i)
        for i, out in enumerate(nest.flatten(output_time_zero)))

    time = constant_op.constant(0, dtype='int32', name='time')

    # We only specify the 'maximum_iterations' when building for XLA since that
    # causes slowdowns on GPU in TF.
    if (not context.executing_eagerly() and
        control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())):
      max_iterations = math_ops.reduce_max(input_length)
    else:
      max_iterations = None

    while_loop_kwargs = {
        'cond': lambda time, *_: time < time_steps_t,
        'maximum_iterations': max_iterations,
        'parallel_iterations': 32,
        'swap_memory': True,
    }
    if mask is not None:
      if go_backwards:
        mask = reverse(mask, 0)

      mask_ta = tensor_array_ops.TensorArray(
          dtype=dtypes_module.bool,
          size=time_steps_t,
          tensor_array_name='mask_ta')
      mask_ta = mask_ta.unstack(mask)

      def masking_fn(time):
        return mask_ta.read(time)

      def compute_masked_output(mask_t, flat_out, flat_mask):
        tiled_mask_t = tuple(
            _expand_mask(mask_t, o, fixed_dim=len(mask_t.shape))
            for o in flat_out)
        return tuple(
            array_ops.where_v2(m, o, fm)
            for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask))
    elif isinstance(input_length, ops.Tensor):
      if go_backwards:
        max_len = math_ops.reduce_max(input_length, axis=0)
        rev_input_length = math_ops.subtract(max_len - 1, input_length)

        def masking_fn(time):
          return math_ops.less(rev_input_length, time)
      else:

        def masking_fn(time):
          return math_ops.greater(input_length, time)

      def compute_masked_output(mask_t, flat_out, flat_mask):
        return tuple(
            array_ops.where(mask_t, o, zo)
            for (o, zo) in zip(flat_out, flat_mask))
    else:
      masking_fn = None

    if masking_fn is not None:
      # Mask for the T output will be base on the output of T - 1. In the case
      # T = 0, a zero filled tensor will be used.
      flat_zero_output = tuple(array_ops.zeros_like(o)
                               for o in nest.flatten(output_time_zero))
      def _step(time, output_ta_t, prev_output, *states):
        """RNN step function.

        Arguments:
            time: Current timestep value.
            output_ta_t: TensorArray.
            prev_output: tuple of outputs from time - 1.
            *states: List of states.

        Returns:
            Tuple: `(time + 1, output_ta_t, output) + tuple(new_states)`
        """
        current_input = tuple(ta.read(time) for ta in input_ta)
        # maybe set shape.
        current_input = nest.pack_sequence_as(inputs, current_input)
        mask_t = masking_fn(time)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        # mask output
        flat_output = nest.flatten(output)
        flat_mask_output = (flat_zero_output if zero_output_for_mask
                            else nest.flatten(prev_output))
        flat_new_output = compute_masked_output(mask_t, flat_output,
                                                flat_mask_output)

        # mask states
        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
          if isinstance(new_state, ops.Tensor):
            new_state.set_shape(state.shape)
        flat_final_state = compute_masked_output(mask_t, flat_new_state,
                                                 flat_state)
        new_states = nest.pack_sequence_as(new_states, flat_final_state)

        output_ta_t = tuple(
            ta.write(time, out)
            for ta, out in zip(output_ta_t, flat_new_output))
        return (time + 1, output_ta_t,
                tuple(flat_new_output)) + tuple(new_states)

      final_outputs = control_flow_ops.while_loop(
          body=_step,
          loop_vars=(time, output_ta, flat_zero_output) + states,
          **while_loop_kwargs)
      # Skip final_outputs[2] which is the output for final timestep.
      new_states = final_outputs[3:]
    else:
      def _step(time, output_ta_t, *states):
        """RNN step function.

        Arguments:
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.

        Returns:
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
        current_input = tuple(ta.read(time) for ta in input_ta)
        current_input = nest.pack_sequence_as(inputs, current_input)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
          if isinstance(new_state, ops.Tensor):
            new_state.set_shape(state.shape)

        flat_output = nest.flatten(output)
        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, flat_output))
        new_states = nest.pack_sequence_as(initial_states, flat_new_state)
        return (time + 1, output_ta_t) + tuple(new_states)

      final_outputs = control_flow_ops.while_loop(
          body=_step,
          loop_vars=(time, output_ta) + states,
          **while_loop_kwargs)
      new_states = final_outputs[2:]

    output_ta = final_outputs[1]

    outputs = tuple(o.stack() for o in output_ta)
    last_output = tuple(o[-1] for o in outputs)

    outputs = nest.pack_sequence_as(output_time_zero, outputs)
    last_output = nest.pack_sequence_as(output_time_zero, last_output)

  # static shape inference
  def set_shape(output_):
    if isinstance(output_, ops.Tensor):
      shape = output_.shape.as_list()
      shape[0] = time_steps
      shape[1] = batch
      output_.set_shape(shape)
    return output_

  outputs = nest.map_structure(set_shape, outputs)

  if not time_major:
    outputs = nest.map_structure(swap_batch_timestep, outputs)

  return last_output, outputs, new_states
