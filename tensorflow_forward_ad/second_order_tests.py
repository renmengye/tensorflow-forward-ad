"""
Unit testing second order matrix vector product.
"""

from __future__ import (division, print_function, unicode_literals)

import numpy as np
import tensorflow as tf

from tensorflow_forward_ad import logger
from tensorflow_forward_ad.fwgrad import forward_gradients
from tensorflow_forward_ad.second_order import (
    hessian_vec_fw, hessian_vec_bk, gauss_newton_vec, fisher_vec_fw,
    fisher_vec_bk, gauss_newton_vec_z, fisher_vec_z)

log = logger.get()


class TestSecondOrderFwGrad(tf.test.TestCase):

  def test_hessian_quadratic(self):
    rnd = np.random.RandomState(0)
    dtype = tf.float64
    with tf.Graph().as_default():
      r = tf.Variable(0.0, dtype=dtype)
      x = tf.constant(rnd.uniform(-1.0, 1.0, [2, 27]), dtype=dtype, name="x")
      w2 = tf.constant(rnd.uniform(-1.0, 1.0, [27, 1]), dtype=dtype, name="w2")
      v2 = tf.constant(rnd.uniform(-1.0, 1.0, [27, 1]), dtype=dtype, name="v2")
      w2v = tf.add(w2, tf.multiply(r, v2))
      h2 = tf.matmul(x, w2v)
      y2 = tf.reduce_sum(h2 * h2)

      grad_w = tf.gradients(y2, w2)
      hv_fw = hessian_vec_fw(y2, [w2v], [v2])
      hv_bk = hessian_vec_bk(y2, [w2], [v2])

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        grad_w = sess.run(grad_w)
        hv_fw_val = sess.run(hv_fw)
        hv_bk_val = sess.run(hv_bk)
        np.testing.assert_allclose(hv_fw_val, hv_bk_val, rtol=1e-5)

  def test_sparse_softmax_with_logits_grad(self):
    rnd = np.random.RandomState(0)
    dtype = tf.float64  # Use float64 and CPU for finite difference checking.
    with tf.Graph().as_default(), tf.device("/cpu:0"):
      r = tf.Variable(0.0, dtype=dtype)
      # Input.
      x = tf.constant(rnd.uniform(-1.0, 1.0, [2, 27]), dtype=dtype, name="x")

      # Fully connected.
      v = tf.constant(rnd.uniform(-0.1, 0.1, [27, 2]), dtype=dtype, name="v")
      w = tf.constant(rnd.uniform(-1.0, 1.0, [27, 2]), dtype=dtype, name="w")
      y = tf.matmul(x, w + r * v)

      label = tf.constant([1, 0], dtype=tf.int32)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=y, labels=label)
      loss = tf.reduce_sum(loss)

      hv_fw = hessian_vec_fw(loss, [w], [v])[0]
      # hv_bk = hessian_vec_bk(loss, [w], [v])[0]
      grad_w = tf.gradients(loss, [w])[0]

      with self.test_session() as sess:
        # Compute Hv with finite difference.
        sess.run(tf.global_variables_initializer())
        eps = 1e-5
        sess.run(tf.assign(r, -eps))
        grad_w_val1 = sess.run(grad_w)
        sess.run(tf.assign(r, eps))
        grad_w_val2 = sess.run(grad_w)
        fndv = (grad_w_val2 - grad_w_val1) / (2 * eps)

        # Compute Hv with forward mode autodiff.
        sess.run(tf.global_variables_initializer())
        fwv = sess.run(hv_fw)

        # # Compute Hv with reverse mode autodiff.
        # bkv = sess.run(hv_bk)

        np.testing.assert_allclose(fndv, fwv, rtol=1e-5)
        # # Expect failure here.
        #np.testing.assert_allclose(fndv, bkv, rtol=1e-5)

  def test_softmax_grad(self):
    rnd = np.random.RandomState(0)
    dtype = tf.float64  # Use float64 and CPU for finite difference checking.
    with tf.Graph().as_default(), tf.device("/cpu:0"):
      r = tf.Variable(0.0, dtype=dtype)
      # Input.
      x = tf.constant(rnd.uniform(-1.0, 1.0, [2, 27]), dtype=dtype, name="x")

      # Fully connected.
      v = tf.constant(rnd.uniform(-0.1, 0.1, [27, 2]), dtype=dtype, name="v")
      w = tf.constant(rnd.uniform(-1.0, 1.0, [27, 2]), dtype=dtype, name="w")
      y = tf.matmul(x, w + r * v)

      label = tf.constant([1, 0], dtype=tf.int32)
      y = tf.nn.softmax(y)
      t = tf.one_hot(label, 2, dtype=dtype)
      loss = tf.log(y + 1e-5) * t
      loss = tf.reduce_sum(loss)

      hv_fw = hessian_vec_fw(loss, [w], [v])[0]
      hv_bk = hessian_vec_bk(loss, [w], [v])[0]
      grad_w = tf.gradients(loss, [w])[0]

      with self.test_session() as sess:
        # Compute Hv with finite difference.
        sess.run(tf.global_variables_initializer())
        eps = 1e-5
        sess.run(tf.assign(r, -eps))
        grad_w_val1 = sess.run(grad_w)
        sess.run(tf.assign(r, eps))
        grad_w_val2 = sess.run(grad_w)
        fndv = (grad_w_val2 - grad_w_val1) / (2 * eps)

        # Compute Hv with forward mode autodiff.
        sess.run(tf.global_variables_initializer())
        fwv = sess.run(hv_fw)

        # Compute Hv with reverse mode autodiff.
        bkv = sess.run(hv_bk)

        np.testing.assert_allclose(fndv, fwv, rtol=1e-5)
        np.testing.assert_allclose(fndv, bkv, rtol=1e-5)
        np.testing.assert_allclose(fwv, bkv, rtol=1e-5)

  def _test_hessian_cnn(self):
    rnd = np.random.RandomState(0)
    dtype = tf.float32  # Use float64 and CPU for finite difference checking.
    # tf.nn.conv2d and tf.nn.max_pool does not support float64.
    # with tf.Graph().as_default(), tf.device("/cpu:0"):
    with tf.Graph().as_default():
      # Input.
      x = tf.constant(
          np.random.uniform(-1.0, 1.0, [2, 5, 5, 2]), dtype=dtype, name="x")

      # First convolution.
      v = tf.constant(
          np.random.uniform(-0.1, 0.1, [2, 2, 2, 3]), dtype=dtype, name="v")
      w = tf.constant(
          np.random.uniform(-1.0, 1.0, [2, 2, 2, 3]), dtype=dtype, name="w")
      h = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME")
      h = tf.nn.max_pool(h, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")
      h = tf.nn.relu(h)

      # Second convolution.
      v_ = tf.constant(
          np.random.uniform(-0.1, 0.1, [2, 2, 3, 3]), dtype=dtype, name="v_")
      w_ = tf.constant(
          np.random.uniform(-1.0, 1.0, [2, 2, 3, 3]), dtype=dtype, name="w_")
      h = tf.nn.conv2d(h, w_, [1, 1, 1, 1], "SAME")
      h = tf.nn.sigmoid(h)

      # Fully connected.
      dim = 27
      v2 = tf.constant(rnd.uniform(-0.1, 0.1, [dim, 2]), dtype=dtype, name="v2")
      w2 = tf.constant(rnd.uniform(-1.0, 1.0, [dim, 2]), dtype=dtype, name="w2")
      w2v = w2 + r * v2
      h = tf.reshape(h, [-1, dim])
      y = tf.matmul(h, w2v)

      label = tf.constant([1, 0], dtype=tf.int32)
      # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      #     logits=y, labels=label)
      # loss = tf.reduce_sum(loss)

      # Use manual cross entropy.
      label_dense = tf.one_hot(label, 2, dtype=dtype)
      y = tf.nn.softmax(y)
      loss = label_dense * tf.log(y + 1e-5)
      loss = tf.reduce_sum(loss)

      # Using explicit R-op is deprecated, due to complex model building.
      hv_fw = hessian_vec_fw(loss, [w2v, w_v, wv], [v2, v_, v])
      hv_bk = hessian_vec_bk(loss, [w2, w_, w], [v2, v_, v])

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        hv_fw_val = sess.run(hv_fw)
        hv_bk_val = sess.run(hv_bk)
        for fwv, bkv in zip(hv_fw_val, hv_bk_val):
          np.testing.assert_allclose(fwv, bkv, rtol=1e-5)

  def test_gauss_newton_quadratic(self):
    rnd = np.random.RandomState(0)
    with tf.Graph().as_default():
      r = tf.Variable(0.0)
      x = tf.constant(
          rnd.uniform(-1.0, 1.0, [2, 27]), dtype=tf.float32, name="x")
      w = tf.constant(
          rnd.uniform(-1.0, 1.0, [27, 3]), dtype=tf.float32, name="w2")
      v = tf.constant(
          rnd.uniform(-1.0, 1.0, [27, 3]), dtype=tf.float32, name="v2")
      wv = tf.add(w, tf.multiply(r, v))
      z = tf.matmul(x, wv)
      y = 0.5 * tf.reduce_sum(z * z)

      # Gauss-Newton, same as Fisher for square loss.
      gv_fw = gauss_newton_vec(y, z, [w], [v])[0]
      # Fisher towards the output layer.
      fv_fw = fisher_vec_fw(z, [w], [v])

      # Fisher using tf.gradients (reverse mode).
      fv_bk = fisher_vec_bk(y, [w], [v])
      # Fisher using forward mode, towards loss function.
      fv_fw_y = fisher_vec_fw(y, [w], [v])

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())

        gv_fw_val = sess.run(gv_fw)
        fv_fw_val = sess.run(fv_fw)
        np.testing.assert_allclose(gv_fw_val, fv_fw_val, rtol=1e-5, atol=1e-5)

        fv_fw_y_val = sess.run(fv_fw_y)
        fv_bk_val = sess.run(fv_bk)
        np.testing.assert_allclose(fv_fw_y_val, fv_bk_val, rtol=1e-5, atol=1e-5)

  def test_gauss_newton_output_quadratic(self):
    rnd = np.random.RandomState(0)
    with tf.Graph().as_default():
      x = tf.constant(
          rnd.uniform(-1.0, 1.0, [27, 2]), dtype=tf.float32, name="x")
      h = tf.constant(
          rnd.uniform(-1.0, 1.0, [3 * 2, 3 * 2]), dtype=tf.float32, name="h")
      h = tf.matmul(tf.transpose(h), h)
      j = tf.constant(
          rnd.uniform(-1.0, 1.0, [3, 27]), dtype=tf.float32, name="j")
      z = tf.matmul(j, x)  # [3, 2]
      z_ = tf.reshape(z, [3 * 2, 1])  # [6, 1]
      y = 0.5 * tf.matmul(tf.matmul(tf.transpose(z_), h), z_)  # [1, 1]
      v = tf.constant(
          rnd.uniform(-1.0, 1.0, [3, 2]), dtype=tf.float32, name="v")
      act_jv = tf.gradients(z, x, v, gate_gradients=True)[0]
      act_hjjv = gauss_newton_vec_z(y, z, x, v)[0]
      exp_jv = tf.matmul(tf.transpose(j), v)  # [27, 2]
      exp_jjv = tf.matmul(j, exp_jv)  # [3, 2]
      exp_jjv_ = tf.reshape(exp_jjv, [3 * 2, 1])
      exp_hjjv = tf.matmul(h, exp_jjv_)  # [6, 1]
      exp_hjjv = tf.reshape(exp_hjjv, [3, 2])

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        h_val = sess.run(h)
        e, _ = np.linalg.eig(h_val)
        self.assertTrue((e > 0).all())
        act_hjjv_val, exp_hjjv_val, act_jv_val, exp_jv_val = sess.run(
            [act_hjjv, exp_hjjv, act_jv, exp_jv])
        np.testing.assert_allclose(act_jv_val, exp_jv_val, rtol=1e-5)
        np.testing.assert_allclose(act_hjjv_val, exp_hjjv_val, rtol=1e-5)

  def test_fisher_output_quadratic(self):
    rnd = np.random.RandomState(0)
    with tf.Graph().as_default():
      x = tf.constant(
          rnd.uniform(-1.0, 1.0, [27, 2]), dtype=tf.float32, name="x")
      h = tf.constant(
          rnd.uniform(-1.0, 1.0, [3 * 2, 3 * 2]), dtype=tf.float32, name="h")
      h = tf.matmul(tf.transpose(h), h)
      j = tf.constant(
          rnd.uniform(-1.0, 1.0, [3, 27]), dtype=tf.float32, name="j")
      z = tf.matmul(j, x)  # [3, 2]
      z_ = tf.reshape(z, [3 * 2, 1])  # [6, 1]
      y = 0.5 * tf.matmul(tf.matmul(tf.transpose(z_), h), z_)  # [1, 1]
      v = tf.constant(
          rnd.uniform(-1.0, 1.0, [3, 2]), dtype=tf.float32, name="v")
      act_jjv = fisher_vec_z(z, x, v)[0]
      exp_jv = tf.matmul(tf.transpose(j), v)  # [27, 2]
      exp_jjv = tf.matmul(j, exp_jv)  # [3, 2]

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        act_jjv_val, exp_jjv_val = sess.run([act_jjv, exp_jjv])
        np.testing.assert_allclose(act_jjv_val, exp_jjv_val, rtol=1e-5)


if __name__ == "__main__":
  tf.test.main()
