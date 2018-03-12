"""
Unit testing forward-mode automatic differentiation.
"""

from __future__ import (division, print_function, unicode_literals)

import numpy as np
import tensorflow as tf

from tensorflow_forward_ad import logger
from tensorflow_forward_ad.fwgrad import forward_gradients

log = logger.get()
tf.logging.set_verbosity(tf.logging.ERROR)


class BasicFwGradTests(tf.test.TestCase):

  def inner_prod(self, xlist, ylist):
    """Computes inner product of two lists of tensors."""
    return tf.reduce_sum(
        tf.stack([tf.reduce_sum(x * y) for x, y in zip(xlist, ylist)]))

  def make_unit_graph(self, x, y, rnd=None, dtype=tf.float32):
    """Makes a computation graph that computes (J^T r)^T v and r^T J v"""
    if rnd is None:
      rnd = np.random.RandomState(0)
    x_shape = [int(ss) for ss in x.get_shape()]
    v = self.get_random_tensor(x_shape, rnd=rnd)
    y_shape = [int(ss) for ss in y.get_shape()]
    r = self.get_random_tensor(y_shape, rnd=rnd)
    jt_r = tf.gradients(y, [x], r, gate_gradients=True)
    jt_r_t_v = self.inner_prod(jt_r, [v])
    j_v = forward_gradients(y, [x], [v], gate_gradients=True)
    r_t_j_v = tf.reduce_sum(r * j_v)
    return jt_r_t_v, r_t_j_v

  def assert_bw_fw(self, sess, x, y, rnd=None):
    bk, fw = self.make_unit_graph(x, y, rnd=rnd)
    bk_val, fw_val = sess.run([bk, fw])
    np.testing.assert_allclose(bk_val, fw_val, rtol=1e-5)

  def get_random_tensor(self,
                        shape,
                        rnd=None,
                        dtype=tf.float32,
                        scale=(-1.0, 1.0)):
    return tf.constant(rnd.uniform(scale[0], scale[1], shape), dtype=dtype)


class MaxPool_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([2, 5, 5, 3], rnd=rnd)
      y = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class MatMul_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([12, 24], rnd=rnd)
      y = tf.matmul(x, x2)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Conv2D_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([2, 5, 5, 3], rnd=rnd)
      x2 = tf.constant(rnd.uniform(-1.0, 1.0, [3, 3, 3, 4]), dtype=tf.float32)
      y = tf.nn.conv2d(x, x2, [1, 1, 1, 1], "SAME")
      self.assert_bw_fw(sess, x, y, rnd=rnd)

  def test_basic2(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([2, 5, 5, 3], rnd=rnd)
      x2 = tf.constant(rnd.uniform(-1.0, 1.0, [3, 3, 3, 4]), dtype=tf.float32)
      y = tf.nn.conv2d(x, x2, [1, 1, 1, 1], "VALID")
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Conv2DBackpropFilter_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([2, 5, 5, 3], rnd=rnd)
      x2 = tf.constant(rnd.uniform(-1.0, 1.0, [3, 3, 3, 4]), dtype=tf.float32)
      y = tf.gradients(tf.nn.conv2d(x, x2, [1, 1, 1, 1], "SAME"), x2)[0]
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Conv2DBackpropInput_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([5, 5, 4, 3], rnd=rnd)
      x2 = tf.constant(rnd.uniform(-1.0, 1.0, [3, 3, 3, 4]), dtype=tf.float32)
      y = tf.gradients(tf.nn.conv2d(x2, x, [1, 1, 1, 1], "SAME"), x2)[0]
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class MaxPoolGrad_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([2, 5, 5, 3], rnd=rnd)
      y = tf.gradients(
          tf.square(tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")),
          x)[0]
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Square_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      dtype = tf.float32
      x_shape = [18, 12]
      x = tf.constant(rnd.uniform(-1.0, 1.0, x_shape), dtype=dtype, name="x")
      y = tf.square(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Sqrt_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      dtype = tf.float32
      x_shape = [18, 12]
      x = tf.constant(rnd.uniform(0.0, 1.0, x_shape), dtype=dtype, name="x")
      y = tf.sqrt(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Relu_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      dtype = tf.float32
      x_shape = [18, 12]
      x = tf.constant(rnd.uniform(-1.0, 1.0, x_shape), dtype=dtype, name="x")
      y = tf.nn.relu(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class ReluGrad_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.gradients(tf.square(tf.nn.relu(x)), x)[0]
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Maximum_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.maximum(x, x2)
      self.assert_bw_fw(sess, x, y, rnd=rnd)
      self.assert_bw_fw(sess, x2, y, rnd=rnd)


class Minimum_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.minimum(x, x2)
      self.assert_bw_fw(sess, x, y, rnd=rnd)
      self.assert_bw_fw(sess, x2, y, rnd=rnd)


class Add_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      y = x + x2
      self.assert_bw_fw(sess, x, y, rnd=rnd)
      self.assert_bw_fw(sess, x2, y, rnd=rnd)


class Sub_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      y = x - x2
      self.assert_bw_fw(sess, x, y, rnd=rnd)
      self.assert_bw_fw(sess, x2, y, rnd=rnd)


class Neg_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.negative(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class AddN_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.add_n([x, x2])
      self.assert_bw_fw(sess, x, y, rnd=rnd)
      self.assert_bw_fw(sess, x2, y, rnd=rnd)


class Mul_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      y = x * x2
      self.assert_bw_fw(sess, x, y, rnd=rnd)
      self.assert_bw_fw(sess, x2, y, rnd=rnd)


class Div_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      y = x / x2
      self.assert_bw_fw(sess, x, y, rnd=rnd)
      self.assert_bw_fw(sess, x2, y, rnd=rnd)


class Reciprocal_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.reciprocal(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Pow_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.pow(x, x2)
      self.assert_bw_fw(sess, x, y, rnd=rnd)
      self.assert_bw_fw(sess, x2, y, rnd=rnd)


class Identity_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.identity(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Print_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.Print(x, [x])
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Sigmoid_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.sigmoid(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class SigmoidGrad_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.gradients(tf.sigmoid(x), x)[0]
      self.assert_bw_fw(sess, x, y, rnd=rnd)

  def test_basic2(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.gradients(tf.square(tf.sigmoid(x)), x)[0]
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Tanh_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.tanh(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)

    def test_manual(self):
      with tf.Graph().as_default(), tf.device("/cpu:0"):
        with self.test_session() as sess:
          x_val = np.random.uniform(0, 1)
          x = tf.constant(x_val)
          y = tf.tanh(x)
          dy_dx = forward_gradients(y, x, gate_gradients=True)
          dy_dx_tf = sess.run(dy_dx)
          eps = 1e-5
          x_val = x_val - eps
          y_val_1 = np.tanh(x_val)
          x_val = x_val + 2 * eps
          y_val_2 = np.tanh(x_val)
          dy_dx_fd = (y_val_2 - y_val_1) / (2 * eps)
          np.testing.assert_allclose(dy_dx_tf, dy_dx_fd, rtol=1e-5)


class TanhGrad_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.gradients(tf.tanh(x), x)[0]
      self.assert_bw_fw(sess, x, y, rnd=rnd)

  def test_basic2(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.gradients(tf.square(tf.tanh(x)), x)[0]
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Softmax_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.nn.softmax(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class SoftmaxGrad_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.gradients(tf.square(tf.nn.softmax(x)), x)[0]
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Log_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd, scale=(0.1, 10.0))
      y = tf.log(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class SparseSoftmaxCrossEntropyWithLogits_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      labels = tf.constant(
          rnd.uniform(0, 5, [18]).astype(np.int64), dtype=tf.int64)
      if tf.__version__.startswith("0"):
        y = tf.nn.sparse_softmax_cross_entropy_with_logits(x, labels)
      else:
        y = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=x, labels=labels)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Transpose_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.transpose(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Pack_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      if tf.__version__.startswith("0"):
        y = tf.pack([x, x2])
      else:
        y = tf.stack([x, x2])
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Concat_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      for ax in [0, 1]:
        if tf.__version__.startswith("0"):
          y = tf.concat(ax, [x, x2])
        else:
          y = tf.concat([x, x2], axis=ax)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Pack_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      x2 = self.get_random_tensor([18, 12], rnd=rnd)
      if tf.__version__.startswith("0"):
        y = tf.pack([x, x2])
      else:
        y = tf.stack([x, x2])
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Reshape_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.reshape(x, [4, -1])
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class ReduceSum_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.reduce_sum(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)

  def test_basic2(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.reduce_sum(x, [1])
      self.assert_bw_fw(sess, x, y, rnd=rnd)

  def test_basic3(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.reduce_sum(x, [1], keep_dims=True)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class ReduceMean_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.reduce_mean(x)
      self.assert_bw_fw(sess, x, y, rnd=rnd)

  def test_basic2(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.reduce_mean(x, [1])
      self.assert_bw_fw(sess, x, y, rnd=rnd)

  def test_basic3(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([18, 12], rnd=rnd)
      y = tf.reduce_mean(x, [1], keep_dims=True)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class Tile_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([2, 12], rnd=rnd)
      y = tf.tile(x, [4, 1])
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class ExpandDims_FwGradTests(BasicFwGradTests):

  def test_basic(self):
    with tf.Graph().as_default(), self.test_session() as sess:
      rnd = np.random.RandomState(0)
      x = self.get_random_tensor([2, 12], rnd=rnd)
      y = tf.expand_dims(x, 0)
      self.assert_bw_fw(sess, x, y, rnd=rnd)


class ForwardGradGraphTests(tf.test.TestCase):

  def test_convnet(self):
    with tf.Graph().as_default():
      # Define model.
      r = tf.Variable(1.0)
      x = tf.constant(
          np.random.uniform(-1.0, 1.0, [1, 5, 5, 2]), dtype=tf.float32)
      w = tf.constant(
          np.random.uniform(-1.0, 1.0, [2, 2, 2, 3]), dtype=tf.float32)
      h = tf.nn.conv2d(r + x, r * w, [1, 1, 1, 1], "SAME")
      h = tf.nn.max_pool(h, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
      h = tf.nn.relu(h)
      # First branch.
      w2 = tf.constant(np.random.uniform(-1.0, 1.0, [27, 1]), dtype=tf.float32)
      h2 = tf.matmul(tf.reshape(h, [1, -1]), w2)
      y2 = tf.nn.tanh(h2)
      y2 = tf.reduce_sum(y2)
      # We can take a second branch.
      w3 = tf.constant(np.random.uniform(-1.0, 1.0, [27, 1]), dtype=tf.float32)
      h3 = tf.matmul(tf.reshape(h, [1, -1]), w3)
      y3 = tf.nn.sigmoid(h3)
      y3 = tf.reduce_sum(y3)
      # Take gradients of a list of y wrt. scalar r.
      # Returns [grad_y2_r, grad_y3_r].
      grad_fw = forward_gradients([y2, y3], r, gate_gradients=True)
      # Reverse mode implementation from tensorflow.
      grad_bk = [tf.gradients(y2, r)[0], tf.gradients(y3, r)[0]]
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        grad_fw_val = sess.run(grad_fw)
        grad_bk_val = sess.run(grad_bk)
        np.testing.assert_allclose(grad_fw_val, grad_bk_val, rtol=5)

  def test_grad_graph(self):
    with tf.Graph().as_default():

      # Dummy variable.
      r = tf.Variable(1.0)

      # Input.
      x = tf.constant(
          np.random.uniform(-1.0, 1.0, [1, 5, 5, 2]),
          dtype=tf.float32,
          name="x")

      # First convolution.
      v = tf.constant(
          np.random.uniform(-1.0, 1.0, [2, 2, 2, 3]),
          dtype=tf.float32,
          name="v")
      w = tf.constant(
          np.random.uniform(-1.0, 1.0, [2, 2, 2, 3]),
          dtype=tf.float32,
          name="w")
      wv = w + r * v
      h = tf.nn.conv2d(x, wv, [1, 1, 1, 1], "SAME")
      h = tf.nn.max_pool(h, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
      h = tf.nn.relu(h)

      # Second convolution.
      v_ = tf.constant(
          np.random.uniform(-1.0, 1.0, [2, 2, 3, 3]),
          dtype=tf.float32,
          name="v_")
      w_ = tf.constant(
          np.random.uniform(-1.0, 1.0, [2, 2, 3, 3]),
          dtype=tf.float32,
          name="w_")
      w_v = w_ + r * v_
      h = tf.nn.conv2d(h, w_v, [1, 1, 1, 1], "SAME")

      # Fully connected.
      w2 = tf.constant(
          np.random.uniform(-1.0, 1.0, [27, 1]), dtype=tf.float32, name="w2")
      h2 = tf.matmul(tf.reshape(h, [1, -1]), w2)
      y2 = tf.nn.sigmoid(h2)
      y2 = tf.reduce_sum(y2)
      grad_bk = tf.gradients(y2, [w, w_], gate_gradients=True)
      grad_fw = forward_gradients(grad_bk, r, gate_gradients=True)
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(grad_fw)

  def test_op(self):
    with tf.Graph().as_default():
      r = tf.Variable(1.0)
      f = tf.constant(0.0)
      r = tf.add_n([r, f])
      g = tf.get_default_graph()
      node_list = g.as_graph_def().node
      inspect = set(["AddN"])
      node_list = filter(lambda x: x.op in inspect, node_list)

  def test_op_2(self):
    rnd = np.random.RandomState(0)
    with tf.Graph().as_default():
      logits = tf.constant(rnd.uniform(0.0, 1.0, [2, 2]), dtype=tf.float32)
      logits_v = tf.constant(rnd.uniform(0.0, 1.0, [2, 2]), dtype=tf.float32)
      r = tf.Variable(0.0, dtype=tf.float32)
      logits = logits_v * r + logits
      t = tf.constant([1, 0])
      t = tf.one_hot(t, 2, dtype=tf.float32)
      # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      #     logits=logits, labels=y)
      y = tf.nn.softmax(logits)
      loss = t * tf.log(y + 1e-5)
      loss = tf.reduce_sum(loss)

      g = tf.get_default_graph()
      node_list = g.as_graph_def().node
      inspect = set(["Softmax", "SoftmaxGrad"])
      node_list = filter(lambda x: x.op in inspect, node_list)
      grad_fw = forward_gradients(loss, r, gate_gradients=True)
      grad_bk = tf.gradients(loss, r)[0]
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        grad_bk_val = sess.run(grad_bk)
        grad_fw_val = sess.run(grad_fw)
        np.testing.assert_allclose(grad_fw_val, grad_bk_val, rtol=5)

  def test_forward_mode_cnn(self):
    """Test v^T (J v) = (J^T v) ^T v"""
    rnd = np.random.RandomState(0)
    dtype = tf.float32  # Use float64 and CPU for finite difference checking.
    # tf.nn.conv2d and tf.nn.max_pool does not support float64.
    # with tf.Graph().as_default(), tf.device("/cpu:0"):
    with tf.Graph().as_default():
      # Input.
      x = tf.constant(
          rnd.uniform(-1.0, 1.0, [2, 5, 5, 2]), dtype=dtype, name="x")

      # First convolution.
      v = tf.constant(
          rnd.uniform(-1.0, 1.0, [2, 2, 2, 3]), dtype=dtype, name="v")
      w = tf.constant(
          rnd.uniform(-1.0, 1.0, [2, 2, 2, 3]), dtype=dtype, name="w")
      h = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
      h = tf.nn.max_pool(h, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
      h = tf.nn.relu(h)

      # Second convolution.
      v_ = tf.constant(
          rnd.uniform(-0.1, 0.1, [2, 2, 3, 3]), dtype=dtype, name="v_")
      w_ = tf.constant(
          rnd.uniform(-1.0, 1.0, [2, 2, 3, 3]), dtype=dtype, name="w_")
      h = tf.nn.conv2d(h, w_, [1, 1, 1, 1], "SAME")
      h = tf.nn.sigmoid(h)

      # Fully connected.
      dim = 27
      v2 = tf.constant(rnd.uniform(-0.1, 0.1, [dim, 2]), dtype=dtype, name="v2")
      w2 = tf.constant(rnd.uniform(-1.0, 1.0, [dim, 2]), dtype=dtype, name="w2")
      h = tf.reshape(h, [-1, dim])
      y = tf.matmul(h, w2)
      r = tf.constant(rnd.uniform(-1.0, 1.0, [2, 2]), dtype=dtype, name="r")

      w_list = [w, w_, w2]
      v_list = [v, v_, v2]

      # Taking inner product of two list of tensors.
      inner_prod = lambda xlist, ylist: tf.reduce_sum(
        tf.stack([tf.reduce_sum(x * y) for x, y in zip(xlist, ylist)]))

      # J^T r
      jt_r = tf.gradients(y, w_list, r, gate_gradients=True)
      # (J^T r)^T v
      jt_r_t_v = inner_prod(jt_r, v_list)

      # J v
      j_v = forward_gradients(y, w_list, v_list, gate_gradients=True)
      # r^T J v
      r_t_j_v = tf.reduce_sum(r * j_v)

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        bk_val, fw_val = sess.run([jt_r_t_v, r_t_j_v])
        np.testing.assert_allclose(bk_val, fw_val, rtol=1e-5)


if __name__ == "__main__":
  tf.test.main()
