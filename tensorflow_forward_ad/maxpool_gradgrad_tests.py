from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gradient_checker
from tensorflow_forward_ad import maxpool_gradgrad


class MaxPoolGradGradTests(tf.test.TestCase):

  def test_basics(self):
    tf.set_random_seed(1234)
    x_shape = [4, 4, 4, 5]
    rnd = np.random.RandomState(0)
    x_np = rnd.uniform(-1.0, 1.0, x_shape).astype(np.float32)

    # test op max_pool_grad
    with tf.Graph().as_default(), tf.Session() as sess:
      x_tf = tf.constant(x_np, name="x")
      y_tf1, _ = tf.nn.max_pool_with_argmax(
          x_tf,
          ksize=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME',
          name="y1")
      y_tf2 = tf.nn.max_pool(
          x_tf,
          ksize=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME',
          name="y2")
      z_tf1 = tf.reduce_sum(tf.square(y_tf1))
      z_tf2 = tf.reduce_sum(tf.square(y_tf2))
      dx1 = tf.gradients(z_tf1, x_tf, name='dx1')[0]
      dx2 = tf.gradients(z_tf2, x_tf, name='dx2')[0]
      err = gradient_checker.compute_gradient_error(
          x_tf, x_shape, dx1, x_shape, delta=1e-3, x_init_value=x_np)
      self.assertTrue(err < 1e-3)
      err = gradient_checker.compute_gradient_error(
          x_tf, x_shape, dx2, x_shape, delta=1e-3, x_init_value=x_np)
      self.assertTrue(err < 1e-3)


if __name__ == "__main__":
  tf.test.main()
