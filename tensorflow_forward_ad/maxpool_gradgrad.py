import tensorflow as tf
from tensorflow.python.framework import ops


@ops.RegisterGradient("MaxPoolGrad")
def _MaxPoolGradGrad(op, grad):
  """The gradients for `MaxPoolGrad`.
    Args:
      op: The `MaxPoolGrad` `Operation` that we are differentiating, which we 
        can use to find the inputs and outputs of the original op.
        op.inputs[0]: x
        op.inputs[1]: y
        op.inputs[2]: dl/dy
        op.outputs[0]: dl/dx
      grad: Gradient with respect to the output of the `MaxPoolGrad` op.
    Returns:
      Gradients with respect to the input of `MaxPoolGrad`.
  """
  ksize = op.get_attr("ksize")
  strides = op.get_attr("strides")
  padding = op.get_attr("padding")

  _, argmax = tf.nn.max_pool_with_argmax(op.inputs[0], ksize, strides, padding)
  grad_flat = tf.reshape(grad, [-1])
  argmax_flat = tf.reshape(argmax, [-1])

  grad_x = tf.zeros_like(op.inputs[0])
  grad_y = tf.zeros_like(op.inputs[1])
  x_shape = tf.cast(tf.shape(op.inputs[0]), argmax.dtype)
  batch_dim = tf.reshape(
      tf.range(x_shape[0], dtype=argmax.dtype), [-1, 1, 1, 1])
  nelem = tf.reduce_prod(x_shape[1:])
  batch_dim *= nelem

  y_zero = tf.zeros_like(op.inputs[1], dtype=argmax.dtype)
  batch_dim += y_zero
  batch_dim = tf.reshape(batch_dim, [-1])

  argmax_flat += batch_dim
  grad_input = tf.gather(grad_flat, argmax_flat)
  grad_input = tf.reshape(grad_input, tf.shape(op.inputs[1]))

  return [None, None, grad_input]
