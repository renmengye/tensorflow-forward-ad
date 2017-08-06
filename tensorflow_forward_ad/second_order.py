"""
Implements second-order matrix vector product using R-op (forward-mode
automatic differentiation) in TensorFlow.

1) Hessian
2) Fisher
3) Gauss-Newton
"""
import tensorflow as tf

from tensorflow_forward_ad import logger
from tensorflow_forward_ad.fwgrad import forward_gradients

log = logger.get()


def hessian_vec_fw(ys, xs, vs, grads=None):
  """Implements Hessian vector product using forward on backward AD.

  Args:
    ys: Loss function.
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.

  Returns:
    Hv: Hessian vector product, same size, same shape as xs.
  """
  # Validate the input
  if type(xs) == list:
    if len(vs) != len(xs):
      raise ValueError("xs and vs must have the same length.")

  if grads is None:
    grads = tf.gradients(ys, xs, gate_gradients=True)
  return forward_gradients(grads, xs, vs, gate_gradients=True)


def hessian_vec_bk(ys, xs, vs, grads=None):
  """Implements Hessian vector product using backward on backward AD.

  Args:
    ys: Loss function.
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.

  Returns:
    Hv: Hessian vector product, same size, same shape as xs.
  """
  # Validate the input
  if type(xs) == list:
    if len(vs) != len(xs):
      raise ValueError("xs and vs must have the same length.")

  if grads is None:
    grads = tf.gradients(ys, xs, gate_gradients=True)
  return tf.gradients(grads, xs, vs, gate_gradients=True)


def fisher_vec_bk(ys, xs, vs):
  """Implements Fisher vector product using backward AD.

  Args:
    ys: Loss function, scalar.
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.

  Returns:
    J'Jv: Fisher vector product.
  """
  # Validate the input
  if type(xs) == list:
    if len(vs) != len(xs):
      raise ValueError("xs and vs must have the same length.")

  grads = tf.gradients(ys, xs, gate_gradients=True)
  gradsv = list(map(lambda x: tf.reduce_sum(x[0] * x[1]), zip(grads, vs)))
  jv = tf.add_n(gradsv)
  jjv = list(map(lambda x: x * jv, grads))
  return jjv


def fisher_vec_fw(ys, xs, vs):
  """Implements Fisher vector product using backward and forward AD.

  Args:
    ys: Loss function or output variables.
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.

  Returns:
    J'Jv: Fisher vector product.
  """
  # Validate the input
  if type(xs) == list:
    if len(vs) != len(xs):
      raise ValueError("xs and vs must have the same length.")

  jv = forward_gradients(ys, xs, vs, gate_gradients=True)
  jjv = tf.gradients(ys, xs, jv, gate_gradients=True)
  return jjv


def gauss_newton_vec(ys, zs, xs, vs):
  """Implements Gauss-Newton vector product.

  Args:
    ys: Loss function.
    zs: Before output layer (input to softmax).
    xs: Weights, list of tensors.
    vs: List of perturbation vector for each weight tensor.

  Returns:
    J'HJv: Guass-Newton vector product.
  """
  # Validate the input
  if type(xs) == list:
    if len(vs) != len(xs):
      raise ValueError("xs and vs must have the same length.")

  grads_z = tf.gradients(ys, zs, gate_gradients=True)
  hjv = forward_gradients(grads_z, xs, vs, gate_gradients=True)
  jhjv = tf.gradients(zs, xs, hjv, gate_gradients=True)
  return jhjv, hjv


def fisher_vec_z(ys, xs, vs):
  """Implements JJ'v, where v is on the output space.

  Args:
    ys: Loss function or output variables.
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.

  Returns:
    JJ'v: Fisher vector product on the output space.
  """
  # Validate the input
  if type(ys) == list:
    if len(vs) != len(ys):
      raise ValueError("ys and vs must have the same length.")

  jv = tf.gradients(ys, xs, vs, gate_gradients=True)
  jjv = forward_gradients(ys, xs, jv, gate_gradients=True)
  return jjv


def gauss_newton_vec_z(ys, zs, xs, vs):
  """Implements HJJ'v, where v is on the output space.

  Args:
    ys: Loss function or output variables.
    zs: Before output layer (input to softmax).
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.

  Returns:
    HJJ'v: Gauss-Newton vector product on the output space.
  """
  # Validate the input
  if type(zs) == list:
    if len(vs) != len(zs):
      raise ValueError("zs and vs must have the same length.")

  grads_z = tf.gradients(ys, zs, gate_gradients=True)
  jv = tf.gradients(zs, xs, vs, gate_gradients=True)
  hjjv = forward_gradients(grads_z, xs, jv, gate_gradients=True)
  return hjjv
