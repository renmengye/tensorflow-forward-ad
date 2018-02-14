import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


# @ops.RegisterGradient("MaxPoolWithArgmax")
# def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
#   """The gradients for `MaxPoolWithArgmax`.
#   Args:
#     op: The `MaxPoolWithArgmax` `Operation` that we are differentiating, which
#       we can use to find the inputs and outputs of the original op.
#     grad: Gradient with respect to the output of the `MaxPoolWithArgmax` op.
#     op.inputs[0]: x
#     op.outputs[0]: y
#     op.outputs[1]: argmax_in_x
#   Returns:
#     Gradients with respect to the input of `MaxPoolWithArgmax`.
#   """
# 
#   return gen_nn_ops._max_pool_grad_with_argmax(
#       op.inputs[0],
#       grad,
#       op.outputs[1],
#       op.get_attr("ksize"),
#       op.get_attr("strides"),
#       padding=op.get_attr("padding"))


def _max_pool_grad_grad(dy, x, y, ksize, strides, padding, argmax=None):
  """Gradients of MaxPoolGrad."""
  if argmax is None:
    _, argmax = tf.nn.max_pool_with_argmax(x, ksize, strides, padding)
  grad = dy
  grad_flat = tf.reshape(grad, [-1])
  argmax_flat = tf.reshape(argmax, [-1])

  x_shape = tf.cast(tf.shape(x), argmax.dtype)
  batch_dim = tf.reshape(
      tf.range(
          x_shape[0], dtype=argmax.dtype), [-1, 1, 1, 1])
  nelem = tf.reduce_prod(x_shape[1:])
  batch_dim *= nelem

  y_zero = tf.zeros_like(y, dtype=argmax.dtype)
  batch_dim += y_zero
  batch_dim = tf.reshape(batch_dim, [-1])

  argmax_flat += batch_dim
  grad_input = tf.gather(grad_flat, argmax_flat)
  grad_input = tf.reshape(grad_input, tf.shape(y))
  return grad_input


@ops.RegisterGradient("MaxPoolGradWithArgmax")
def _MaxPoolGradWithArgmaxGrad(op, grad):
  """The gradients for `MaxPoolGradWithArgmax`.
  Args:
    op: The `MaxPoolGradWithArgmax` `Operation` that we are differentiating,
      which we can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `MaxPoolGradWithArgmax` op.
    op.inputs[0]: x
    op.inputs[1]: dl/dy
    op.inputs[2]: argmax_in_x
    op.outputs[0]: dl/dx
  Returns:
    Gradients with respect to the input of `MaxPoolGradWithArgmax`.
  """
  ksize = op.get_attr("ksize")
  strides = op.get_attr("strides")
  padding = op.get_attr("padding")
  return [
      None, _max_pool_grad_grad(
          grad,
          op.inputs[0],
          op.inputs[1],
          ksize,
          strides,
          padding,
          argmax=op.inputs[2]), None
  ]


# @ops.RegisterGradient("MaxPoolGrad")
# def _MaxPoolGradGrad(op, grad):
#   """The gradients for `MaxPoolGrad`.
#   Args:
#     op: The `MaxPoolGrad` `Operation` that we are differentiating, which we can use
#       to find the inputs and outputs of the original op.
#       op.inputs[0]: x
#       op.inputs[1]: y
#       op.inputs[2]: dl/dy
#       op.outputs[0]: dl/dx
#     grad: Gradient with respect to the output of the `MaxPoolGrad` op.
#   Returns:
#     Gradients with respect to the input of `MaxPoolGrad`.
#   """
#   ksize = op.get_attr("ksize")
#   strides = op.get_attr("strides")
#   padding = op.get_attr("padding")
#   return [
#       None, None, _max_pool_grad_grad(grad, op.inputs[0], op.inputs[1], ksize,
#                                       strides, padding)
#   ]
