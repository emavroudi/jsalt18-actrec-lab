from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from numpy import arange, array, max, reshape, zeros
import tensorflow as tf
from tensorflow.python.ops import array_ops, rnn
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import LSTMCell, GRUCell


def count_trues(tensor):
    return tf.reduce_sum(tf.cast(tensor, tf.float32))


def debug_nans(tensor, tensor_name, debug=True):
    if debug:
        return tf.Print(tensor, [count_trues(tf.is_nan(tensor)),
                                 tf.shape(tensor)], tensor_name)
    else:
        return tensor


def trainable_params_info(sess):
    info = defaultdict(defaultdict)
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for variable_name, v in zip(variables_names, values):
        print("Variable name: ", variable_name)
        print("Shape: ", v.shape)
        info[variable_name]['shape'] = v.shape
        info[variable_name]['value'] = v
    return info


def assign_pretrained_weights(sess, pretrained_weights_dict, var_names_map):
    """
    Args:
        sess: session
        pretrained_weights_dict: dictionary whose keys correspond to variable
            names of a trained model.
            dict['var_name']['shape'], dict['var_name']['values']
        var_names_map: dictionary whose keys correspond to variable names
            in the current graph and values have the corresponding names
            in the trained model.
    """
    for new_var_name in var_names_map.keys():
        var = my_get_variable(new_var_name)
        assign_op = tf.assign(
            var, pretrained_weights_dict[var_names_map[new_var_name]]['value'])
        sess.run(assign_op)


def my_get_variable(var_name):
    """
    Args:
        var_name: variable name e.g. "encoder_rnn/encoder_rnn/fw/lstm_cell/
                                      weights:0"
    :return:
    """
    var = [v for v in tf.global_variables() if v.name == var_name]
    if len(var) >= 1:
        var = var[0]
    else:
        print("Not found: ", var_name)
    return var


def get_rnn_cell(params, is_training, inputs_dim=None, layer_name='rnn_cell',
                 input_dropout=-1):
    with tf.variable_scope(layer_name):

        if params['initializer'] is None:
            initializer = None
        elif params['initializer'] == "random_uniform":
            init_scale = params['init_scale'] /params[
                'nb_hidden_units']
            initializer = tf.random_uniform_initializer(
                -init_scale, init_scale)
        elif params['initializer'] == 'random_normal':
            initializer = tf.random_normal_initializer(0, params['init_scale'])
        elif params['initializer'] == 'orthogonal':
            # TODO: check gain
            initializer = tf.orthogonal_initializer(
                gain=params['init_scale'])
        else:
            raise ValueError("Invalid initializer %s",
                             params['initializer'])

        if params['cell_type'] == 'lstm':
            # LSTM Cell
            # Default activation: tanh
            if params['activation'] == 'tanh':
                activation = tf.nn.tanh
            else:
                activation = None

            cell_fw = LSTMCell(num_units=params['nb_hidden_units'],
                               use_peepholes=False,
                               cell_clip=None, initializer=initializer,
                               num_proj=None, proj_clip=None,
                               forget_bias=1.0, state_is_tuple=True,
                               activation=activation)
            cell_bw = LSTMCell(num_units=params['nb_hidden_units'],
                               use_peepholes=False,
                               cell_clip=None, initializer=initializer,
                               num_proj=None, proj_clip=None,
                               forget_bias=1.0, state_is_tuple=True,
                               activation=activation)
        elif params['cell_type'] == 'gru':
            # GRU Cell
            # If bias_initializer is None, then it starts with bias
            # of 1.0
            # to not reset and not update.
            # Default activation: tanh
            if params['activation'] == 'tanh':
                activation = tf.nn.tanh
            else:
                activation = None
            cell_fw = GRUCell(num_units=params['nb_hidden_units'],
                              activation=activation,
                              reuse=None,
                              kernel_initializer=initializer,
                              bias_initializer=None)
            cell_bw = GRUCell(num_units=params['nb_hidden_units'],
                              activation=activation,
                              reuse=None,
                              kernel_initializer=initializer,
                              bias_initializer=None)
        else:
            raise ValueError('Not supported cell type: %s',
                             params['cell_type'])

        # Add dropout
        if input_dropout != -1:
            input_keep_prob = 1 - params['dropout_rate']
            input_size = inputs_dim
        else:
            input_keep_prob = 1.0
            input_size = None

        output_keep_prob = 1 - params['dropout_rate']
        dropout_cell_fw = tf.contrib.rnn.DropoutWrapper(
            cell_fw, input_keep_prob=1.0,
            output_keep_prob=tf.cond(
                is_training,
                lambda: tf.constant(output_keep_prob),
                lambda: tf.constant(1.0)),
            state_keep_prob=1.0,
            variational_recurrent=True, input_size=input_size,
            dtype=tf.float32, seed=None)
        dropout_cell_bw = tf.contrib.rnn.DropoutWrapper(
            cell_bw, input_keep_prob=1.0,
            output_keep_prob=tf.cond(
                is_training,
                lambda: tf.constant(output_keep_prob),
                lambda: tf.constant(1.0)),
            state_keep_prob=1.0,
            variational_recurrent=True, input_size=input_size,
            dtype=tf.float32, seed=None)

        return dropout_cell_fw, dropout_cell_bw


def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    time_major=False,
                                    scope=None):
    """Creates a dynamic bidirectional recurrent neural network.
    Stacks several bidirectional rnn layers. The combined forward and backward
    layer outputs are used as input of the next layer. tf.bidirectional_rnn
    does not allow to share forward and backward information between layers.
    The input_size of the first forward and backward cells must match.
    The initial state for both directions is zero.
    Intermediate states are returned.
    Modification of: r1.4/tensorflow/contrib/rnn/python/ops/rnn.py
    Args:
      cells_fw: List of instances of RNNCell, one per layer,
        to be used for forward direction.
      cells_bw: List of instances of RNNCell, one per layer,
        to be used for backward direction.
      inputs: The RNN inputs. this must be a tensor of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such elements.
      initial_states_fw: (optional) A list of the initial states (one per layer)
        for the forward RNN.
        Each tensor must has an appropriate type and shape
        `[batch_size, cell_fw.state_size]`.
      initial_states_bw: (optional) Same as for `initial_states_fw`, but using
        the corresponding properties of `cells_bw`.
      dtype: (optional) The data type for the initial state.  Required if
        either of the initial states are not provided.
      sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
        containing the actual lengths for each of the sequences.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      time_major: The shape format of the inputs and outputs Tensors. If true,
        these Tensors must be shaped [max_time, batch_size, depth]. If false,
        these Tensors must be shaped [batch_size, max_time, depth]. Using
        time_major = True is a bit more efficient because it avoids transposes at
        the beginning and end of the RNN calculation. However, most TensorFlow
        data is batch-major, so by default this function accepts input and emits
        output in batch-major form.
      scope: VariableScope for the created subgraph; defaults to None.
    Returns:
      A tuple (outputs, output_state_fw, output_state_bw) where:
        outputs: Output `Tensor` shaped:
          `batch_size, max_time, layers_output]`. Where layers_output
          are depth-concatenated forward and backward outputs.
        output_states_fw is the final states, one tensor per layer,
          of the forward rnn.
        output_states_bw is the final states, one tensor per layer,
          of the backward rnn.
    Raises:
      TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
      ValueError: If inputs is `None`.
    """
    if not cells_fw:
        raise ValueError(
            "Must specify at least one fw cell for BidirectionalRNN.")
    if not cells_bw:
        raise ValueError(
            "Must specify at least one bw cell for BidirectionalRNN.")
    if not isinstance(cells_fw, list):
        raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
    if not isinstance(cells_bw, list):
        raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
    if len(cells_fw) != len(cells_bw):
        raise ValueError("Forward and Backward cells must have the same depth.")
    if (initial_states_fw is not None and
            (not isinstance(initial_states_fw, list) or
                     len(initial_states_fw) != len(cells_fw))):
        raise ValueError(
            "initial_states_fw must be a list of state tensors (one per layer).")
    if (initial_states_bw is not None and
            (not isinstance(initial_states_bw, list) or
                     len(initial_states_bw) != len(cells_bw))):
        raise ValueError(
            "initial_states_bw must be a list of state tensors (one per layer).")

    states_fw = []
    states_bw = []
    prev_layer = inputs
    hidden_states_all_layers = []

    with vs.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]

            with vs.variable_scope("cell_%d" % i):
                outputs, (state_fw, state_bw) = rnn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    dtype=dtype,
                    time_major=time_major)
                # Concat the outputs to create the new input.
                prev_layer = array_ops.concat(outputs, 2)
            states_fw.append(state_fw)
            states_bw.append(state_bw)
            hidden_states_all_layers.append(prev_layer)

    return prev_layer, tuple(states_fw), tuple(states_bw), \
        hidden_states_all_layers


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Returns:
        A binary matrix representation of the input.
    Source: r1.8/tensorflow/python/keras/_impl/keras/utils/np_utils.py
    """
    y = array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = max(y) + 1
    n = y.shape[0]
    categorical = zeros((n, num_classes))
    categorical[arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = reshape(categorical, output_shape)

    return categorical


def clip_gradients(grads_and_vars, clip_gradients):
        """Clips gradients by global norm."""
        gradients, variables = zip(*grads_and_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, clip_gradients)
        return list(zip(clipped_gradients, variables))
