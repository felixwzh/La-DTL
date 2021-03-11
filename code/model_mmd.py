# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from utils import gene_embedding_matrix,gene_embedding_matrix_t
from functools import partial
from tensorflow.python.ops.array_ops import sequence_mask, reshape, boolean_mask, expand_dims

import sys
import numpy as np
import six

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# 'Constant' gets imported in the module 'array_ops'.
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.deprecation import deprecated


def Represent_learner_pure(xs, sequence_lengths, max_sequence_length, keep_prob, reuse_flag):
    """
    Shared representation learner, including the embedding lookup table and BiLSTM layer
    :param xs: input sentences
    :param sequence_lengths:
    :param max_sequence_length:
    :param keep_prob:
    :return: hidden state h, size[sample_num,max_sequence_len,args.n_rnn_hidden*2]
    """

    norm_initializer = tf.random_normal_initializer(stddev=0.01)

    # embedding
    with tf.variable_scope('emb', reuse=reuse_flag):
        if args.n_embed_init:
            lookup_table = tf.get_variable('lookup_table', [args.vocabulary_size, args.embedding_size],
                                           initializer=norm_initializer)
        else:

            file_for_gene_dict = [
                '../data/new-weibo/two_domain/all_dict',
            ]
            embedding_file ='../model/embedding/weibo/weibo_100d_emb'

            embedding_matrix = gene_embedding_matrix(embedding_file, *file_for_gene_dict)
            embed_matrix = tf.constant(embedding_matrix)
            print 'generate embed_matrix using embedding file %s.' % embedding_file
            lookup_table = tf.get_variable('lookup_table', initializer=embed_matrix, trainable=True)

        global c_emb
        c_emb = tf.nn.embedding_lookup(lookup_table,
                                       xs)  # c_emb is a tensor with size [batch, max_length, embedding_size]

    # global c_emb
    emb = c_emb

    # dropout
    x_in = tf.nn.dropout(emb, keep_prob)

    # rnn
    with tf.variable_scope('rnn', initializer=norm_initializer, reuse=reuse_flag):
        rnn_cell = tf.contrib.rnn
        fw_cell = rnn_cell.LSTMCell(args.n_rnn_hidden, state_is_tuple=True, reuse=reuse_flag)
        bw_cell = rnn_cell.LSTMCell(args.n_rnn_hidden, state_is_tuple=True, reuse=reuse_flag)
        h, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x_in, dtype=tf.float32,
                                               sequence_length=sequence_lengths)
        h = tf.concat([h[0], h[1]], 2)
        h = tf.reshape(h, [-1, args.n_rnn_hidden * 2])  # final embedding from Bi-LSTM network

    return h



def l2_norm():
    l2 = 0
    for param in tf.trainable_variables():
        if param.name not in['emb/lookup_table:0','emb_t/lookup_table:0','emb_s/lookup_table:0']:
            l2 += tf.nn.l2_loss(param)
    l2 += tf.reduce_mean(tf.reduce_sum(tf.square(c_emb), [1, 2]))
    tf.summary.scalar('l2_loss', l2)
    return l2



def loss_trns_A(s, ys, seq_len, A):
    log_likelihood, A = tf.contrib.crf.crf_log_likelihood(s, ys, seq_len, A)
    loss_ = tf.reduce_mean(-log_likelihood)
    return loss_, A

def mmd_loss_label_aligned(i, h_s, h_t, mask_s, mask_t, mmd_param, label_size, mmd_loss):

    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    mask_s_i = reshape(mask_s[i], [-1])
    mmd_h_s = boolean_mask(h_s, mask_s_i)
    mask_t_i = reshape(mask_t[i], [-1])
    mmd_h_t = boolean_mask(h_t, mask_t_i)

    
    if tf.reduce_sum(tf.to_int32(mask_s_i))!=0 and tf.reduce_sum(tf.to_int32(mask_t_i))!=0:
        loss_value = maximum_mean_discrepancy(mmd_h_s, mmd_h_t, kernel=gaussian_kernel)
        mmd_loss += mmd_param * tf.maximum(1e-4, loss_value)

    return i + 1, h_s, h_t, mask_s, mask_t, mmd_param, label_size, mmd_loss
def train(loss):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step,
                                               args.decay_steps, args.decay_rate, staircase=True)
    optimizer = eval('tf.train.%sOptimizer' % args.optimizer)
    optimizer = optimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op, global_step

def predict(s, A, seq_len):
    yp = []
    for ss, l in zip(s, seq_len):
        ss = ss[:l]
        y, _ = tf.contrib.crf.viterbi_decode(ss, A)
        yp.append(y)
    return yp

def sequence_mask_final(lengths, maxlen=None, dtype=dtypes.bool, name=None):
    """Return a mask tensor representing the first N positions of each row.

    Example:

    ```python
    tf.sequence_mask([1, 3, 2], 5) =
      [[True, False, False, False, False],
       [False,False, False, False, False],
       [False, True, False, False, False]]
    ```

    Args:
      lengths: 1D integer tensor, all its values < maxlen.
      maxlen: scalar integer tensor, maximum length of each row. Default: use
              maximum over lengths.
      dtype: output type of the resulting tensor.
      name: name of the op.
    Returns:
      A 2D mask tensor, as shown in the example above, cast to specified dtype.

    Raises:
      ValueError: if the arguments have invalid rank.
    """
    with ops.name_scope(name, "SequenceMask", [lengths, maxlen]):
        lengths = ops.convert_to_tensor(lengths)
        if lengths.get_shape().ndims != 1:
            raise ValueError("lengths must be 1D for sequence_mask")

        if maxlen is None:
            maxlen = gen_math_ops._max(lengths, [0])
        else:
            maxlen = ops.convert_to_tensor(maxlen)
        if maxlen.get_shape().ndims != 0:
            raise ValueError("maxlen must be scalar for sequence_mask")

        # The basic idea is to compare a range row vector of size maxlen:
        # [0, 1, 2, 3, 4]
        # to length as a matrix with 1 column: [[1], [3], [2]].
        # Because of broadcasting on both arguments this comparison results
        # in a matrix of size (len(lengths), maxlen)
        row_vector = gen_math_ops._range(constant(0, maxlen.dtype),
                                         maxlen,
                                         constant(1, maxlen.dtype))
        # Since maxlen >= max(lengths), it is safe to use maxlen as a cast
        # authoritative type. Whenever maxlen fits into tf.int32, so do the lengths.
        matrix_0 = gen_math_ops.cast(expand_dims(lengths, 1), maxlen.dtype)
        matrix_1 = gen_math_ops.cast(expand_dims(lengths - 1, 1), maxlen.dtype)
        result_0 = (row_vector < matrix_0)
        result_1 = (row_vector >= matrix_1)
        result = tf.logical_and(result_0, result_1)

        if dtype is None or result.dtype.base_dtype == dtype.base_dtype:
            return result
        else:
            return gen_math_ops.cast(result, dtype)


def sequence_mask_mid(lengths, maxlen=None, dtype=dtypes.bool, name=None):
    """Return a mask tensor representing the first N positions of each row.

    Example:

    ```python
    tf.sequence_mask([1, 3, 2], 5) =
      [[True, False, False, False, False],
       [False,False, False, False, False],
       [False, True, False, False, False]]
    ```

    Args:
      lengths: 1D integer tensor, all its values < maxlen.
      maxlen: scalar integer tensor, maximum length of each row. Default: use
              maximum over lengths.
      dtype: output type of the resulting tensor.
      name: name of the op.
    Returns:
      A 2D mask tensor, as shown in the example above, cast to specified dtype.

    Raises:
      ValueError: if the arguments have invalid rank.
    """
    with ops.name_scope(name, "SequenceMask", [lengths, maxlen]):
        lengths = (np.array(lengths)) / 2
        lengths = ops.convert_to_tensor(lengths, dtype=tf.int32)
        # lengths = ops.convert_to_tensor(lengths)
        if lengths.get_shape().ndims != 1:
            raise ValueError("lengths must be 1D for sequence_mask")

        if maxlen is None:
            maxlen = gen_math_ops._max(lengths, [0])
        else:
            maxlen = ops.convert_to_tensor(maxlen)
        if maxlen.get_shape().ndims != 0:
            raise ValueError("maxlen must be scalar for sequence_mask")

        # The basic idea is to compare a range row vector of size maxlen:
        # [0, 1, 2, 3, 4]
        # to length as a matrix with 1 column: [[1], [3], [2]].
        # Because of broadcasting on both arguments this comparison results
        # in a matrix of size (len(lengths), maxlen)
        row_vector = gen_math_ops._range(constant(0, maxlen.dtype),
                                         maxlen,
                                         constant(1, maxlen.dtype))
        # Since maxlen >= max(lengths), it is safe to use maxlen as a cast
        # authoritative type. Whenever maxlen fits into tf.int32, so do the lengths.
        matrix_0 = gen_math_ops.cast(expand_dims(lengths, 1), maxlen.dtype)
        matrix_1 = gen_math_ops.cast(expand_dims(lengths - 1, 1), maxlen.dtype)
        result_0 = (row_vector < matrix_0)
        result_1 = (row_vector >= matrix_1)
        result = tf.logical_and(result_0, result_1)

        if dtype is None or result.dtype.base_dtype == dtype.base_dtype:
            return result
        else:
            return gen_math_ops.cast(result, dtype)


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
        x: a tensor of shape [num_x_samples, num_features]
        y: a tensor of shape [num_y_samples, num_features]
    Returns:
        a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
        ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        sigmas: a tensor of floats which denote the widths of each of the
            gaussians in the kernel.
    Returns:
        A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def set_hyperparams(_args):
    global args  # make args global in this model
    args = _args





def loss(s, ys, seq_len, A):
    log_likelihood, A = tf.contrib.crf.crf_log_likelihood(s, ys, seq_len, A)
    loss_ = tf.reduce_mean(-log_likelihood)
    tf.summary.scalar('c_loss', loss_)
    return loss_, A


