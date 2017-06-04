"""
Implements forward-mode automatic differentiation for TensorFlow.

Author: Mengye Ren (mren@cs.toronto.edu)

Usage:

Example 1:
  x = Variable(1.0)
  y = build_model(x)
  grad = forward_gradients(y, x)  # Gradient of y wrt. x, same shape as y.

Example 2::
  # When x is a vector, we compute the gradient of y wrt. a scalar v, with some
  # pre-specified value for the gradient of x wrt. v (by default a 1-vector).
  x = Variable([0.5, 0.8])
  y = build_model(x)
  grad = forward_gradients(y, x, grad_xs=tf.constant([1.0, 2.0]))

Example 3:
  x = [Variable(1.0), Variable(2.0)] # x can be a list of variables.
  y = build_model(x)
  # Grad is still the same dimension of y.
  grad = forward_gradients(y, x, grad_xs=[tf.constant(1.0), tf.constant(2.0)])

Example 4:
  x = Variable(1.0)
  y1 = build_model1(x)
  y2 = build_model2(x)
  # Grad is now a list: [dy1_dx, dy2_dx]
  grad = forward_gradients(y, x)
"""
from __future__ import (division, print_function, unicode_literals)

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gen_nn_ops
from tensorflow_forward_ad.graph import get_path_cover_str_list_list

import logger

log = logger.get()

###############################################################################
# Gradient operator registry utilities.
###############################################################################
FWGRAD_REGISTRY = {}
ELEM_REGISTRY = {}
NODEKW_REGISTRY = {}


def RegisterFwGrad(op, elemwise=False):
  """Registers a forward-mode gradient operator.

  Args:
    op: Name of the op to be differentiated.
    elemwise: Whether the gradient op is element-wise.

  Returns:
    decorator: A function that accepts a function (the gradient op) as input,
    that associates f with the original op.
  """

  def decorator(f):
    FWGRAD_REGISTRY[op] = f
    ELEM_REGISTRY[f] = elemwise

  return decorator


def RegisterNodeKeyword(op):
  """Registers a keyword extractor from NodeDef object.

  Args:
    op: Name of the op to be extracted.

  Returns:
    decorator: A function that accepts a function (the extractor function) as
    input that associates f  with the original op.
  """

  def decorator(f):
    NODEKW_REGISTRY[op] = f

  return decorator


def GetFwGrad(op):
  """Returns the forward mode gradient op that associates with the given op."""
  return FWGRAD_REGISTRY.get(op, None)


def GetNodeKeyword(op):
  """Returns the keyword extractor that associates with the given op."""
  return NODEKW_REGISTRY.get(op, None)


def GetElemwise(f):
  """Returns whether the gradient op elemwise with full forward gradients."""
  return ELEM_REGISTRY.get(f, False)


def NoFwGrad(op):

  def no_grad(*args, **kwargs):
    return None

  FWGRAD_REGISTRY[op] = no_grad


###############################################################################
# Dense operators. Not elemwise.
###############################################################################
@RegisterFwGrad("MatMul", elemwise=False)
def MatMul_FwGrad(op,
                  dx,
                  dw,
                  transpose_a=False,
                  transpose_b=False,
                  adjoint_a=False,
                  adjoint_b=False,
                  _op_table=None,
                  _grad_table=None):
  """Forward gradient operator for matrix multiplication.
  Note: only works when dim(dx) = x, i.e. the variable to be differentiated is
  a scalar.
  """
  x = op.inputs[0]
  w = op.inputs[1]
  if dx is not None:
    dy_x = tf.matmul(
        dx,
        w,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b)
  if dw is not None:
    dy_w = tf.matmul(
        x,
        dw,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b)
  if dx is None and dw is None:
    return None
  elif dx is not None and dw is None:
    return dy_x
  elif dw is not None and dx is None:
    return dy_w
  else:
    return tf.add(dy_x, dy_w)


@RegisterNodeKeyword("MatMul")
def MatMul_NodeKeyword(node):
  kwargs = {}
  kwdict = dict(node.attr)
  if "transpose_a" in kwdict:
    kwargs["transpose_a"] = kwdict["transpose_a"].b
  if "transpose_b" in kwdict:
    kwargs["transpose_b"] = kwdict["transpose_b"].b
  if "adjoint_a" in kwdict:
    kwargs["adjoint_a"] = kwdict["adjoint_a"].b
  if "adjoint_b" in kwdict:
    kwargs["adjoint_b"] = kwdict["adjoint_b"].b
  return kwargs


@RegisterFwGrad("Conv2D", elemwise=False)
def Conv2D_FwGrad(op,
                  dx,
                  dw,
                  strides=[1, 1, 1, 1],
                  padding="SAME",
                  data_format="NHWC",
                  use_cudnn_on_gpu=None,
                  _op_table=None,
                  _grad_table=None):
  """Forward gradient operator for 2D convolution.
  Note: only works when dim(dx) = x, i.e. the variable to be differentiated is
  a scalar.

  y = tf.conv2d(x, w, strides, padding)
  Now computes dy_dv given dx_dv and dw_dv.

  Args:
    x: Input tensor, 4D tensor, [N, H, W, C].
    w: Convolution filters, 4D tensor, [I, J, C, K].
    dx: Gradient of the input tensor, 4D tensor, [N, H, W, C].
    dw: Gradient of the convolution filters, 4D tensor, [I, J, C, K].
    strides: Strides, list of integers.
    padding: Padding, string, "SAME" or "VALID".
    data_format: "NHWC" or "NCHW".
    use_cudnn_on_gpu: Whether to use the cuDNN library on GPU.

  Returns:
    dy_dv: The gradient of y wrt. the variable to be differentiated.
  """
  x = op.inputs[0]
  w = op.inputs[1]
  if dx is not None:
    dy_x = tf.nn.conv2d(
        dx,
        w,
        strides,
        padding,
        data_format=data_format,
        use_cudnn_on_gpu=use_cudnn_on_gpu)
  if dw is not None:
    dy_w = tf.nn.conv2d(
        x,
        dw,
        strides,
        padding,
        data_format=data_format,
        use_cudnn_on_gpu=use_cudnn_on_gpu)
  if dx is None and dw is None:
    return None
  elif dx is not None and dw is None:
    return dy_x
  elif dw is not None and dx is None:
    return dy_w
  else:
    return tf.add(dy_x, dy_w)


@RegisterNodeKeyword("Conv2D")
def Conv2D_NodeKeyword(node):
  kwargs = {}
  kwdict = dict(node.attr)
  kwargs["strides"] = list(kwdict["strides"].list.i)
  kwargs["padding"] = kwdict["padding"].s
  kwargs["data_format"] = kwdict["data_format"].s
  kwargs["use_cudnn_on_gpu"] = kwdict["use_cudnn_on_gpu"].b
  return kwargs


@RegisterFwGrad("Conv2DBackpropFilter", elemwise=False)
def Conv2DBackpropFilter_FwGrad(op,
                                dx,
                                dy,
                                dz,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                data_format="NHWC",
                                use_cudnn_on_gpu=None,
                                _op_table=None,
                                _grad_table=None):
  """Forward gradient operator of conv 2d gradient wrt. filters.
  Args:
    op: Conv2DBackpropFilter operator.
    dx: Forward gradient of the input to Conv2D.
    dy: Forward gradient of the filter shape.
    dz: Forward gradient of the backward gradient of the output of the Conv2D.
  """
  # dx = dx_dr
  # dz = d(dE_dy)_dr
  dE_dy = op.inputs[2]  # dE_dy
  x = op.inputs[0]
  fshape = op.inputs[1]
  if dx is not None:
    dy_x = tf.nn.conv2d_backprop_filter(
        dx,
        fshape,
        dE_dy,
        strides,
        padding,
        data_format=data_format,
        use_cudnn_on_gpu=use_cudnn_on_gpu)
  if dz is not None:
    dy_z = tf.nn.conv2d_backprop_filter(
        x,
        fshape,
        dz,
        strides,
        padding,
        data_format=data_format,
        use_cudnn_on_gpu=use_cudnn_on_gpu)
  if dx is None and dz is None:
    return None
  elif dx is not None and dz is None:
    return dy_x
  elif dz is not None and dx is None:
    return dy_z
  else:
    return tf.add(dy_x, dy_z)


@RegisterNodeKeyword("Conv2DBackpropFilter")
def Conv2DBackpropFilter_NodeKeyword(node):
  kwargs = {}
  kwdict = dict(node.attr)
  kwargs["strides"] = list(kwdict["strides"].list.i)
  kwargs["padding"] = kwdict["padding"].s
  kwargs["data_format"] = kwdict["data_format"].s
  kwargs["use_cudnn_on_gpu"] = kwdict["use_cudnn_on_gpu"].b
  return kwargs


@RegisterFwGrad("Conv2DBackpropInput", elemwise=False)
def Conv2DBackpropInput_FwGrad(op,
                               dx,
                               dy,
                               dz,
                               strides=[1, 1, 1, 1],
                               padding="SAME",
                               data_format="NHWC",
                               use_cudnn_on_gpu=None,
                               _op_table=None,
                               _grad_table=None):
  """Forward gradient operator of conv 2d gradient wrt. input.
  Args:
    op: Conv2DBackpropInput operator.
    dx: Forward gradient of the input shape.
    dy: Forward gradient of the filter.
    dz: Forward gradient of the backward gradient of the output of the Conv2D.
  """
  # dz = d(dE_dy)_dr
  dE_dy = op.inputs[2]  # dE_dy
  f = op.inputs[1]
  df = dy
  xshape = op.inputs[0]
  if df is not None:
    dy_f = tf.nn.conv2d_backprop_input(
        xshape,
        df,
        dE_dy,
        strides,
        padding,
        data_format=data_format,
        use_cudnn_on_gpu=use_cudnn_on_gpu)
  if dz is not None:
    dy_z = tf.nn.conv2d_backprop_input(
        xshape,
        f,
        dz,
        strides,
        padding,
        data_format=data_format,
        use_cudnn_on_gpu=use_cudnn_on_gpu)
  if df is None and dz is None:
    return None
  elif df is not None and dz is None:
    return dy_f
  elif dz is not None and df is None:
    return dy_z
  else:
    return tf.add(dy_f, dy_z)


@RegisterNodeKeyword("Conv2DBackpropInput")
def Conv2DBackpropInput_NodeKeyword(node):
  kwargs = {}
  kwdict = dict(node.attr)
  kwargs["strides"] = list(kwdict["strides"].list.i)
  kwargs["padding"] = kwdict["padding"].s
  kwargs["data_format"] = kwdict["data_format"].s
  kwargs["use_cudnn_on_gpu"] = kwdict["use_cudnn_on_gpu"].b
  return kwargs


@RegisterFwGrad("MaxPool", elemwise=False)
def MaxPool_FwGrad(op,
                   dx,
                   ksize=[1, 2, 2, 1],
                   strides=[1, 2, 2, 1],
                   padding="SAME",
                   _op_table=None,
                   _grad_table=None):
  """Forward gradient operator for max pooling.

  Args:
    x: Input tensor, 4D tensor, [N, H, W, C].
    dx: Gradient of the input tensor, 4D tensor, [N, H, W, C].
    ksize: Kernel size of the max pooling operator, list of integers.
    strides: Strides of the max pooling operator, list of integers.
    padding: Padding, string, "SAME" or "VALID".
    data_format: "NHWC" or "NCHW".
  """
  if dx is None:
    return None
  x = op.inputs[0]
  y = op.outputs[0]
  _, argmax = tf.nn.max_pool_with_argmax(x, ksize, strides, padding)
  dx_flat = tf.reshape(dx, [-1])
  argmax_flat = tf.reshape(argmax, [-1])
  y_zero = tf.zeros_like(y, dtype=argmax.dtype)
  x_shape = tf.cast(tf.shape(x), argmax.dtype)
  batch_dim = tf.reshape(
      tf.range(x_shape[0], dtype=argmax.dtype), [-1, 1, 1, 1])
  nelem = tf.reduce_prod(x_shape[1:])
  batch_dim *= nelem
  batch_dim += y_zero
  batch_dim = tf.reshape(batch_dim, [-1])
  argmax_flat += batch_dim
  dx_sel = tf.gather(dx_flat, argmax_flat)
  dy = tf.reshape(dx_sel, tf.shape(argmax))
  return dy


@RegisterNodeKeyword("MaxPool")
def MaxPool_NodeKeyword(node):
  kwargs = {}
  kwdict = dict(node.attr)
  kwargs["strides"] = list(kwdict["strides"].list.i)
  kwargs["ksize"] = list(kwdict["ksize"].list.i)
  kwargs["padding"] = kwdict["padding"].s
  return kwargs


@RegisterFwGrad("MaxPoolGrad", elemwise=False)
def MaxPoolGrad_FwGrad(op,
                       dx,
                       dy,
                       dz,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       data_format="NHWC",
                       padding="SAME",
                       _op_table=None,
                       _grad_table=None):
  """Forward gradient operator for the backward gradient of max pooling.

  Args:
    op: MaxPoolGrad operator.
    dx: Forward gradient to the input of MaxPool.
    dy: Forward gradient to the output of MaxPool.
    dz: Forward gradient to the backward gradient of the output of MaxPool.
  """
  if dz is None:
    return None
  return gen_nn_ops._max_pool_grad(
      op.inputs[0],
      op.inputs[1],
      dz,
      ksize,
      strides,
      padding,
      data_format=data_format)


@RegisterNodeKeyword("MaxPoolGrad")
def MaxPoolGrad_NodeKeyword(node):
  kwargs = {}
  kwdict = dict(node.attr)
  kwargs["strides"] = list(kwdict["strides"].list.i)
  kwargs["ksize"] = list(kwdict["ksize"].list.i)
  kwargs["padding"] = kwdict["padding"].s
  return kwargs


@RegisterFwGrad("StridedSlice", elemwise=False)
def StridedSlice_FwGrad(op, dx, dy, dz, du, _op_table=None, _grad_table=None):
  if dx is None:
    return None
  y = op.inputs[1]
  z = op.inputs[2]
  u = op.inputs[3]
  return tf.strided_slice(dx, begin=y, end=z, strides=u)


###############################################################################
# Element-wise operators. elemwise.
###############################################################################
@RegisterFwGrad("Relu", elemwise=True)
def Relu_FwGrad(op, dx, _op_table=None, _grad_table=None):
  if dx is None:
    return None
  x = op.inputs[0]
  return tf.multiply(dx, tf.cast(tf.greater(x, 0.0), x.dtype))


@RegisterFwGrad("ReluGrad", elemwise=True)
def ReluGrad_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  """Forward gradient operator for the gradient of ReLU.
  f = ReLU(v)
  forward pass: f(u)
  dx = d(dE_df)_dr
  dy = df_dr
  Want d(dE_dv)_dr

  Args:
    op: ReluGrad op.
    dx: Forward gradient of the backward gradient of the ReLU output.
    dy: Forward gradient of ReLU output.
  """
  if dx is None:
    return None
  x = op.inputs[0]  # Backward gradient of the ReLU output, dE_df.
  y = op.inputs[1]  # ReLU output, f.
  return tf.multiply(dx, tf.cast(tf.greater(y, 0.0), dx.dtype))


@RegisterFwGrad("Add", elemwise=True)
def Add_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  if dx is None and dy is None:
    return None
  else:
    if dx is None:
      x = op.inputs[0]
      dx = tf.zeros_like(x)
    if dy is None:
      y = op.inputs[1]
      dy = tf.zeros_like(y)
    return tf.add(dx, dy)


@RegisterFwGrad("Sub", elemwise=True)
def Sub_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  if dx is None and dy is None:
    return None
  else:
    if dx is None:
      x = op.inputs[0]
      dx = tf.zeros_like(x)
    if dy is None:
      y = op.inputs[1]
      dy = tf.zeros_like(y)
    return tf.subtract(dx, dy)


@RegisterFwGrad("Neg", elemwise=True)
def Neg_FwGrad(op, dx, _op_table=None, _grad_table=None):
  if dx is None:
    return None
  return tf.negative(dx)


@RegisterFwGrad("AddN", elemwise=True)
def AddN_FwGrad(*args, **kwargs):
  no_none = list(filter(lambda x: x is not None, args[1:]))
  if len(no_none) > 0:
    return tf.add_n(no_none)
  else:
    return None


@RegisterFwGrad("Mul", elemwise=True)
def Mul_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  x = op.inputs[0]
  y = op.inputs[1]
  if dx is None and dy is None:
    return None
  elif dx is not None and dy is None:
    return tf.multiply(y, dx)
  elif dy is not None and dx is None:
    return tf.multiply(x, dy)
  else:
    return tf.add(tf.multiply(x, dy), tf.multiply(y, dx))


if tf.__version__.startswith("0"):
  divname = "Div"
else:
  divname = "RealDiv"


@RegisterFwGrad(divname, elemwise=True)
def Div_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  x = op.inputs[0]
  y = op.inputs[1]
  if dx is None and dy is None:
    return None
  elif dx is not None and dy is None:
    return tf.divide(dx, y)
  elif dy is not None and dx is None:
    return tf.divide(x * dy, y**2)
  else:
    return tf.divide(y * dx - x * dy, y**2)


@RegisterFwGrad("Reciprocal", elemwise=True)
def Reciprocal_FwGrad(op, dx, _op_table=None, _grad_table=None):
  x = op.inputs[0]
  if dx is None:
    return None
  return -1 / tf.square(x) * dx


@RegisterFwGrad("Square", elemwise=True)
def Square_FwGrad(op, dx, _op_table=None, _grad_table=None):
  x = op.inputs[0]
  if dx is None:
    return None
  return tf.multiply(tf.multiply(tf.constant(2.0, x.dtype), x), dx)


@RegisterFwGrad("Pow", elemwise=True)
def Pow_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  x = op.inputs[0]
  y = op.inputs[1]
  if dx is None and dy is None:
    return None
  elif dx is not None and dy is None:
    return y * x**(y - 1) * dx
  elif dy is not None and dx is None:
    return x**y * dy
  else:
    return y * x**(y - 1) * dx + x**y * dy


@RegisterFwGrad("Identity", elemwise=True)
def Identity_FwGrad(op, dx, _op_table=None, _grad_table=None):
  return dx


@RegisterFwGrad("PreventGradient", elemwise=True)
def Identity_FwGrad(op, dx, _op_table=None, _grad_table=None):
  return dx


@RegisterFwGrad("Print", elemwise=True)
def Print_FwGrad(*args, **kwargs):
  dx = args[1]
  return dx


@RegisterFwGrad("Sigmoid", elemwise=True)
def Sigmoid_FwGrad(op, dx, _op_table=None, _grad_table=None):
  if dx is None:
    return None
  y = op.outputs[0]
  return tf.multiply(tf.multiply(dx, (1 - y)), y)


@RegisterFwGrad("SigmoidGrad", elemwise=True)
def SigmoidGrad_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  """Forward gradient of SigmoidGrad.
  f = Sigmoid(v).
  dx = df_dr
  dy = d(dE_df)_dr
  Want: d(dE_dv)_dr

  Args:
    op: SigmoidGrad op.
    dx: Forward gradient of the sigmoid.
    dy: Forward gradient of the backward gradient of the sigmoid output.
  """
  f = op.inputs[0]
  f_op = _op_table[f.op.name]
  v = f_op.inputs[0]
  dv_dr = _grad_table[v.name]
  df_dv = f * (1 - f)
  d2E_dfdr = dy

  if dv_dr is None:
    if d2E_dfdr is None:
      return None
    else:
      return d2E_dfdr * df_dv
  else:
    dE_df = op.inputs[1]
    d2f_dv2 = df_dv * (1 - 2 * f)
    df_dr = dx
    d2f_dvdr = df_dr * (1 - 2 * f)
    if d2E_dfdr is None:
      return d2f_dvdr * dE_df
    else:
      return d2E_dfdr * df_dv + d2f_dvdr * dE_df


@RegisterFwGrad("Tanh", elemwise=True)
def Tanh_FwGrad(op, dx, _op_table=None, _grad_table=None):
  y = op.outputs[0]
  if dx is None:
    return None
  return tf.multiply(dx, 1 - y**2)


@RegisterFwGrad("TanhGrad", elemwise=True)
def TanhGrad_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  """Forward gradient of SigmoidGrad.
  f = Sigmoid(v).
  dx = df_dr
  dy = d(dE_df)_dr
  Want: d(dE_dv)_dr

  Args:
    op: TanhGrad op.
    dx: Forward gradient of the sigmoid.
    dy: Forward gradient of the backward gradient of the sigmoid output.
  """
  f = op.inputs[0]
  f_op = _op_table[f.op.name]
  v = f_op.inputs[0]
  dv_dr = _grad_table[v.name]
  df_dv = 1 - tf.square(f)
  d2E_dfdr = dy

  if dv_dr is None:
    if d2E_dfdr is None:
      return None
    else:
      return dy * df_dv
  else:
    dE_df = op.inputs[1]
    d2f_dv_dr = -2 * f * dx
    if d2E_dfdr is None:
      return d2f_dv_dr * dE_df
    else:
      return d2E_dfdr * df_dv + d2f_dv_dr * dE_df


@RegisterFwGrad("Softmax", elemwise=True)
def Softmax_FwGrad(op, dx, _op_table=None, _grad_table=None):
  """Forward gradient operator for softmax."""
  y = op.outputs[0]
  if dx is None:
    return None
  return tf.subtract(
      tf.multiply(y, dx),
      tf.multiply(y, tf.reduce_sum(tf.multiply(dx, y), [1], keep_dims=True)))


@RegisterFwGrad("Log", elemwise=True)
def Log_FwGrad(op, dx, _op_table=None, _grad_table=None):
  x = op.inputs[0]
  if dx is None:
    return None
  return 1 / x * dx


@RegisterFwGrad("SparseSoftmaxCrossEntropyWithLogits", elemwise=True)
def SparseSoftmaxCrossEntropyWithLogits_FwGrad(op,
                                               dx,
                                               dy,
                                               _op_table=None,
                                               _grad_table=None):
  """Forward gradient operator of sparse softmax cross entropy."""
  grad = op.outputs[1]  # This is already computed in the forward pass.
  x = op.inputs[0]
  if dx is None:
    return None
  y = tf.nn.softmax(x)
  grad_grad = tf.subtract(
      tf.multiply(y, dx),
      tf.multiply(y, tf.reduce_sum(tf.multiply(dx, y), [1], keep_dims=True)))
  return tf.reduce_sum(tf.multiply(grad, dx), [1]), grad_grad


@RegisterFwGrad("Transpose", elemwise=True)
def Transpose_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  y = op.inputs[1]
  return tf.transpose(dx, y)


###############################################################################
# Shape sensitive operators. Not elemwise.
###############################################################################
@RegisterFwGrad("Pack", elemwise=True)
def Pack_FwGrad(*args, **kwargs):
  dx = args[1:]
  axis = kwargs["axis"]
  if all(map(lambda x: x is None, dx)):
    log.error("hey")
    return None
  else:
    ### Here we need to fill in zeros.
    def _mapper(_):
      dx = _[0]
      x = _[1]
      return dx if dx is not None else tf.zeros_like(x)

    dx = list(map(_mapper, zip(dx, list(args[0].inputs))))
    if tf.__version__.startswith("0"):
      return tf.pack(dx, axis=axis)
    else:
      return tf.stack(dx, axis=axis)


@RegisterNodeKeyword("Pack")
def PackGrad_NodeKeyword(node):
  kwargs = {}
  kwdict = dict(node.attr)
  kwargs["axis"] = kwdict["axis"].i
  return kwargs


@RegisterFwGrad("Reshape", elemwise=False)
def Reshape_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  shape = op.inputs[1]
  if dx is None:
    return None
  return tf.reshape(dx, shape)


@RegisterFwGrad("Sum", elemwise=False)
def ReduceSum_FwGrad(op,
                     dx,
                     dy,
                     keep_dims=False,
                     _op_table=None,
                     _grad_table=None):
  ridx = op.inputs[1]
  if dx is None:
    return None
  return tf.reduce_sum(dx, keep_dims=keep_dims, reduction_indices=ridx)


@RegisterNodeKeyword("Sum")
def ReduceSum_NodeKeyword(node):
  kwargs = {}
  kwdict = dict(node.attr)
  kwargs["keep_dims"] = kwdict["keep_dims"].b
  return kwargs


@RegisterFwGrad("Mean", elemwise=False)
def ReduceMean_FwGrad(op,
                      dx,
                      dy,
                      keep_dims=False,
                      _op_table=None,
                      _grad_table=None):
  ridx = op.inputs[1]
  if dx is None:
    return None
  return tf.reduce_mean(dx, keep_dims=keep_dims, reduction_indices=ridx)


@RegisterNodeKeyword("Mean")
def ReduceMean_NodeKeyword(node):
  kwargs = {}
  kwdict = dict(node.attr)
  kwargs["keep_dims"] = kwdict["keep_dims"].b
  return kwargs


@RegisterFwGrad("Tile", elemwise=False)
def Tile_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  y = op.inputs[1]
  if dx is None:
    return None
  return tf.tile(dx, y)


@RegisterFwGrad("ExpandDims", elemwise=False)
def ExpandDims_FwGrad(op, dx, dy, _op_table=None, _grad_table=None):
  y = op.inputs[1]
  if dx is None:
    return None
  return tf.expand_dims(dx, y)


###############################################################################
# No gradient operator defined.
###############################################################################
NoFwGrad("NoOp")
NoFwGrad("BroadcastGradientArgs")
NoFwGrad("Const")
NoFwGrad("Shape")
NoFwGrad("Rank")
NoFwGrad("Range")
NoFwGrad("InvertPermutation")


###############################################################################
# Utilities
###############################################################################
def value_or_list(v):
  if type(v) == list:
    return v
  else:
    return [v]


def get_path_cover(ys, xs, node_list):
  ys = value_or_list(ys)
  xs = value_or_list(xs)
  y_name_list = [yy.name for yy in ys]
  x_name_list = [xx.name for xx in xs]
  path_cover_str = get_path_cover_str_list_list(node_list, x_name_list,
                                                y_name_list)
  return path_cover_str


def get_node_kwargs(node):
  """Get additional attributes from the node as keyword arguments to the
  forward gradient operator.

  Args:
    node: NodeDef object.

  Returns:
    kwargs: Keyword arguments dictionary.
  """
  kwargs_fn = GetNodeKeyword(node.op)
  if kwargs_fn is None:
    return {}
  else:
    return kwargs_fn(node)


def get_node_arg_list(op, grad_table):
  # Builds an argument list.
  arg_list = []
  arg_list.append(op)
  for inp in op.inputs:
    if grad_table[inp.name] is None:
      log.warning("Node \"{}\" gradient does not exists.".format(inp.name))
      grad_table[inp.name] = None
    arg_list.append(grad_table[inp.name])
  return arg_list


def store_results(op, grads, grad_table):
  """Stores gradient results to gradient table."""
  if len(op.outputs) == 1:
    grad_table[op.outputs[0].name] = grads
  elif len(op.outputs) > 1:
    if grads is not None:
      for output, grad in zip(op.outputs, grads):
        grad_table[output.name] = grad
    else:
      for output in op.outputs:
        grad_table[output.name] = None


###############################################################################
# Main forward gradient function. Entry point.
###############################################################################
def forward_gradients(ys,
                      xs,
                      grad_xs=None,
                      gate_gradients=False,
                      name="gradients"):
  """Forward-mode automatic differentiation.
  Currently support single variable xs and single/multi variable(s) output ys.

  Args:
    ys: Output variable or a list of output variables.
    xs: Input variable or a list of input variables, v if grad_xs is not
    defined.
    grad_xs: Gradients of x to v, default is the one-vector [1, 1, ...]. This
    is similar to grad_ys in tf.gradients.
    gate_gadients: If True, add a tuple around the results.
    name: Name of the op.

  Returns:
    grad: Gradient of y wrt. v.
  """
  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()
  node_list = graph_def.node
  with tf.variable_scope(name):
    xs = value_or_list(xs)
    ys = value_or_list(ys)
    grad_xs = value_or_list(grad_xs)
    assert len(grad_xs) == len(xs), "Length of grad_xs and xs must equal."

    # Build path cover.
    path_cover_str = get_path_cover(ys, xs, node_list)

    # Initialize node table.
    # From node name to its node def object.
    node_table = dict(map(lambda x: (x.name, x), node_list))

    # Initialize op table.
    # From op name to operation object.
    op_table = dict(map(lambda x: (x.name, x), graph.get_operations()))

    # Initialize tensor table and gradient table.
    tensor_table = dict()  # From tensor name to its tensor object.
    grad_table = dict()  # From tensor name to its forward gradient.
    for node in node_list:
      for output in op_table[node.name].outputs:
        tensor_table[output.name] = output
        grad_table[output.name] = None

    # Set up initial condition.
    inp_names = set(map(lambda x: x.name.split(":")[0], xs))
    for ii, xx in enumerate(xs):
      if grad_xs is None or grad_xs[ii] is None:
        grad_xx = tf.ones_like(xx)
      else:
        grad_xx = grad_xs[ii]
      grad_table[xx.name] = grad_xx

    # Build gradient graph.
    for node_str in path_cover_str:
      if node_str in inp_names:
        continue
      node = node_table[node_str]

      # Build argument list.
      op = op_table[node.name]
      arg_list = get_node_arg_list(op, grad_table)
      kwargs = get_node_kwargs(node)
      kwargs["_op_table"] = op_table
      kwargs["_grad_table"] = grad_table

      # Call forward gradient function.
      fw_grad_fn = GetFwGrad(node.op)
      if fw_grad_fn is None:
        # raise Exception("Op \"{}\" has no FwGrad registered.".format(node.op))
        log.error("Op \"{}\" has no FwGrad registered.".format(node.op))
        node_grads = None
      else:
        # Compute forward gradient.
        node_grads = fw_grad_fn(*arg_list, **kwargs)

      # Store results.
      store_results(op, node_grads, grad_table)

    grads = list(map(lambda x: grad_table[x.name], ys))
    if gate_gradients:
      grads = tf.tuple(grads)
    return grads
