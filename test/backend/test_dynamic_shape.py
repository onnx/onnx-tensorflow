from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import shutil

from onnx import defs
from onnx import helper
from onnx import TensorProto
import numpy as np
import tensorflow as tf

from onnx_tf.backend import onnx_graph_to_tensorflow_rep
from onnx_tf.common.legacy import legacy_opset_pre_ver
from onnx_tf.common.pooling_helper import py_pool


class TestDynamicShape(unittest.TestCase):
  """ Tests for dynamic shape support
  """

  def _get_rnd_float32(self, low=-1.0, high=1.0, shape=None):
    output = np.random.uniform(low, high, shape)
    if shape is None:
      return np.float32(output)
    else:
      return output.astype(np.float32)

  def _get_rnd_int(self, low, high=None, shape=None, dtype=np.int32):
    return np.random.randint(low, high, size=shape, dtype=dtype)

  def test_arg_max(self):
    if legacy_opset_pre_ver(12):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support select_last_index attribute for ArgMax that depends on shape."
          .format(defs.onnx_opset_version()))
    axis = 1
    node_def = helper.make_node("ArgMax",
                                inputs=['X'],
                                outputs=['Y'],
                                axis=axis,
                                keepdims=0,
                                select_last_index=1)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        ])
    x = np.array([[1, 2, 3, 5, 3, 4, 5, 1], [2, 9, 3, 5, 9, 4, 5,
                                             1]]).astype(np.float32)
    # get tf_rep
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/arg_max'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x)
    expected_output = np.argmax(np.flip(x, axis), axis=axis)
    expected_output = x.shape[axis] - expected_output - 1
    np.testing.assert_almost_equal(tf_model_output["Y"], expected_output)

  def test_arg_min(self):
    if legacy_opset_pre_ver(12):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support select_last_index attribute for ArgMin that depends on shape."
          .format(defs.onnx_opset_version()))
    axis = 1
    node_def = helper.make_node("ArgMin",
                                inputs=['X'],
                                outputs=['Y'],
                                axis=axis,
                                keepdims=0,
                                select_last_index=1)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
        ])
    x = np.array([[1, 2, 3, 5, 3, 4, 5, 1], [2, 7, 3, 5, 2, 4, 5,
                                             6]]).astype(np.float32)
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/arg_min'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x)
    expected_output = np.argmin(np.flip(x, axis), axis=axis)
    expected_output = x.shape[axis] - expected_output - 1
    np.testing.assert_almost_equal(tf_model_output["Y"], expected_output)

  def _batch_normalization(self, x, mean, variance, bias, scale,
                           variance_epsilon):
    inv = np.reciprocal(np.sqrt(variance + variance_epsilon))
    if scale is not None:
      inv *= scale
    return x * inv + (bias - mean * inv if bias is not None else -mean * inv)

  def test_batch_normalization(self):
    if legacy_opset_pre_ver(6):
      raise unittest.SkipTest("Backend doesn't support consumed flag")
    node_def = helper.make_node("BatchNormalization",
                                ["X", "scale", "bias", "mean", "var"], ["Y"],
                                epsilon=0.001)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [None]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None])
        ])
    x_shape = [3, 5, 4, 2]
    param_shape = [5]
    _param_shape = [1, 5, 1, 1]
    x = self._get_rnd_float32(0, 1, shape=x_shape)
    m = self._get_rnd_float32(0, 1, shape=param_shape)
    _m = m.reshape(_param_shape)
    v = self._get_rnd_float32(0, 1, shape=param_shape)
    _v = v.reshape(_param_shape)
    scale = self._get_rnd_float32(0, 1, shape=param_shape)
    _scale = scale.reshape(_param_shape)
    bias = self._get_rnd_float32(0, 1, shape=param_shape)
    _bias = bias.reshape(_param_shape)
    golden = self._batch_normalization(x, _m, _v, _bias, _scale, 0.001)
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/batch_normalization'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x, scale=scale, bias=bias, mean=m, var=v)
    np.testing.assert_almost_equal(tf_model_output["Y"], golden, decimal=5)

  def test_compress(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support Compress.".format(
              defs.onnx_opset_version()))
    axis = 1
    node_def = helper.make_node("Compress",
                                inputs=['X', 'condition'],
                                outputs=['Y'],
                                axis=axis)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, [None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None])
        ])
    x = self._get_rnd_float32(shape=[5, 5, 5])
    cond = np.array([1, 0, 1]).astype(np.bool)
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/compress'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x, condition=cond)
    np.testing.assert_almost_equal(tf_model_output["Y"],
                                   np.compress(cond, x, axis=axis))

  def test_conv_transpose(self):
    # test dynamic batch size on transpose of 2d convolution
    pads = [1, 1, 1, 1]
    x_shape = [1, 3, 4, 6]
    x = self._get_rnd_float32(shape=x_shape)
    weight_shape = [3, 5, 2, 2]
    weights = self._get_rnd_float32(shape=weight_shape)

    node_def = helper.make_node("ConvTranspose", ["X", "weights"], ["Y"],
                                pads=pads)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT,
                                          [None, None, None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None])
        ])

    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/conv_transpose'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x, weights=weights)

    padh_left = weight_shape[2] - 1 - pads[0]
    padh_right = weight_shape[2] - 1 - pads[1]
    padw_left = weight_shape[3] - 1 - pads[2]
    padw_right = weight_shape[3] - 1 - pads[3]

    kh = weight_shape[2]
    kw = weight_shape[3]
    outh = x_shape[2] + padh_right + padh_right - (kh - 1)
    outw = x_shape[3] + padw_right + padw_right - (kw - 1)

    out_shape = [x_shape[0], weight_shape[1], outh, outw]

    test_output = np.zeros(out_shape)
    for b in range(0, x_shape[0]):
      for m in range(0, weight_shape[1]):
        for c in range(0, x_shape[1]):
          for h in range(0, outh):
            for w in range(0, outw):
              for k1 in range(h, h + kh):
                for k2 in range(w, w + kw):
                  if (k1 - padh_left >= 0 and k2 - padw_left >= 0):
                    test_output[b][m][h][w] += x[b][c][k1 - padh_left][
                        k2 - padw_left] * weights[c][m][kh + h - 1 -
                                                        k1][kw + w - 1 - k2]

    np.testing.assert_almost_equal(tf_model_output["Y"], test_output, decimal=5)

  def test_depth_to_space(self):
    b, c, h, w = shape = [2, 48, 5, 6]
    blocksize = 4
    x = self._get_rnd_float32(shape=shape)
    node_def = helper.make_node("DepthToSpace", ["X"], ["Y"],
                                blocksize=blocksize,
                                mode="DCR")
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/depth_to_space'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x)
    tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
    tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
    y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
    np.testing.assert_almost_equal(tf_model_output["Y"], y)

  def test_eye_like(self):
    if legacy_opset_pre_ver(9):
      raise unittest.SkipTest("ONNX version {} doesn't support EyeLike.".format(
          defs.onnx_opset_version()))
    shape = [6, 10]
    off_diagonal_offset = -3
    x = self._get_rnd_int(0, 100, shape=shape)
    y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
    node_def = helper.make_node("EyeLike", ["x"], ["y"],
                                dtype=TensorProto.FLOAT,
                                k=off_diagonal_offset)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.INT32, [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/eye_like'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(x=x)
    np.testing.assert_equal(tf_model_output["y"], y)

  def test_flatten(self):
    shape = [2, 3, 4]
    x = self._get_rnd_float32(shape=shape)
    axis = 1
    node_def = helper.make_node("Flatten", ["X"], ["Y"], axis=axis)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None])
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/flatten'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x)
    new_shape = (np.prod(shape[0:axis]).astype(int), -1)
    np.testing.assert_almost_equal(tf_model_output["Y"],
                                   np.reshape(x, new_shape))

  def test_gather_nd(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support GatherND.".format(
              defs.onnx_opset_version()))
    # valid positive and negative indices for elements
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[0, 0], [1, -3]], dtype=np.int64)
    ref_output = np.array([1, 4], dtype=np.int32)
    node_def = helper.make_node("GatherND", ["data", "indices"], ["outputs"])
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.INT32,
                                          [None, None]),
            helper.make_tensor_value_info("indices", TensorProto.INT64,
                                          [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("outputs", TensorProto.INT32, [None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/gather_nd'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(data=data, indices=indices)
    np.testing.assert_almost_equal(tf_model_output["outputs"], ref_output)

  def test_gather_elements(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support GatherND.".format(
              defs.onnx_opset_version()))
    # valid positive and negative indices for elements
    axis = 1
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[0, 0], [1, -3]], dtype=np.int64)
    ref_output = np.array([[1, 1], [5, 4]], dtype=np.int32)
    node_def = helper.make_node("GatherElements", ["data", "indices"],
                                ["outputs"],
                                axis=axis)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.INT32,
                                          [None, None]),
            helper.make_tensor_value_info("indices", TensorProto.INT64,
                                          [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("outputs", TensorProto.INT32, [None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/gather_elements'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(data=data, indices=indices)
    np.testing.assert_almost_equal(tf_model_output["outputs"], ref_output)

  def test_is_inf(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest("ONNX version {} doesn't support IsInf.".format(
          defs.onnx_opset_version()))
    inp = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
                   dtype=np.float32)
    expected_output = np.isinf(inp)
    node_def = helper.make_node("IsInf", ["X"], ["Y"])
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [None]),
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.BOOL, [None])])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/is_inf'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=inp)
    np.testing.assert_equal(tf_model_output["Y"], expected_output)

  def test_matmul_integer(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support MatMulInteger.".format(
              defs.onnx_opset_version()))

    node_def = helper.make_node("MatMulInteger",
                                ["A", "B", "a_zero_point", "b_zero_point"],
                                ["Z"])
    # A & B are 3-D tensor and a_zero_point & b_zero_point are scalar
    A = self._get_rnd_int(-20, 20, shape=(2, 3, 4), dtype=np.int8)
    B = self._get_rnd_int(-20, 20, shape=(2, 4, 6), dtype=np.int8)
    a_zero_point = self._get_rnd_int(-20, 20, dtype=np.int8)
    b_zero_point = self._get_rnd_int(-20, 20, dtype=np.int8)
    A_minus_zero_point = np.subtract(A.astype(np.int32),
                                     a_zero_point.astype(np.int32))
    B_minus_zero_point = np.subtract(B.astype(np.int32),
                                     b_zero_point.astype(np.int32))
    z = np.matmul(A_minus_zero_point, B_minus_zero_point)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("A", TensorProto.INT8,
                                          [None, None, None]),
            helper.make_tensor_value_info("B", TensorProto.INT8,
                                          [None, None, None]),
            helper.make_tensor_value_info("a_zero_point", TensorProto.INT8, []),
            helper.make_tensor_value_info("b_zero_point", TensorProto.INT8, [])
        ],
        outputs=[
            helper.make_tensor_value_info("Z", TensorProto.INT32,
                                          [None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/matmul_integer'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(A=A,
                               B=B,
                               a_zero_point=a_zero_point,
                               b_zero_point=b_zero_point)
    np.testing.assert_almost_equal(tf_model_output["Z"], z)
    # A & B are 4-D tensor and a_zero_point & b_zero_point are 1-D tensor
    A = self._get_rnd_int(-20, 20, shape=(2, 5, 3, 4), dtype=np.int8)
    B = self._get_rnd_int(-20, 20, shape=(2, 1, 4, 6), dtype=np.int8)
    a_zero_point = self._get_rnd_int(-20,
                                     20,
                                     shape=(A.shape[-2]),
                                     dtype=np.int8)
    b_zero_point = self._get_rnd_int(-20,
                                     20,
                                     shape=(B.shape[-1]),
                                     dtype=np.int8)
    a_zero_point_with_reshape = np.reshape(a_zero_point, [A.shape[-2], 1])
    A_minus_zero_point = np.subtract(A.astype(np.int32),
                                     a_zero_point_with_reshape.astype(np.int32))
    B_minus_zero_point = np.subtract(B.astype(np.int32),
                                     b_zero_point.astype(np.int32))
    z = np.matmul(A_minus_zero_point, B_minus_zero_point)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("A", TensorProto.INT8,
                                          [None, None, None, None]),
            helper.make_tensor_value_info("B", TensorProto.INT8,
                                          [None, None, None, None]),
            helper.make_tensor_value_info("a_zero_point", TensorProto.INT8,
                                          [None]),
            helper.make_tensor_value_info("b_zero_point", TensorProto.INT8,
                                          [None])
        ],
        outputs=[
            helper.make_tensor_value_info("Z", TensorProto.INT32,
                                          [None, None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(A=A,
                               B=B,
                               a_zero_point=a_zero_point,
                               b_zero_point=b_zero_point)
    np.testing.assert_almost_equal(tf_model_output["Z"], z)

  def test_non_max_suppression(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support NonMaxSuppression.".format(
              defs.onnx_opset_version()))
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                       [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                       [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0, 101.0]],
                      [[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                       [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                       [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                                101.0]]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
                       [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0], [1, 0, 3],
                                 [1, 0, 0]]).astype(np.int64)
    node_def = helper.make_node("NonMaxSuppression", [
        "boxes", "scores", "max_output_boxes_per_class", "iou_threshold",
        "score_threshold"
    ], ["selected_indices"],
                                center_point_box=0)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT,
                                          [None, None, None]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT,
                                          [None, None, None]),
            helper.make_tensor_value_info("max_output_boxes_per_class",
                                          TensorProto.INT64, [None]),
            helper.make_tensor_value_info("iou_threshold", TensorProto.FLOAT,
                                          [None]),
            helper.make_tensor_value_info("score_threshold", TensorProto.FLOAT,
                                          [None])
        ],
        outputs=[
            helper.make_tensor_value_info("selected_indices", TensorProto.INT64,
                                          [None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/non_max_suppression'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(
        boxes=boxes,
        scores=scores,
        max_output_boxes_per_class=max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold)
    np.testing.assert_almost_equal(tf_model_output["selected_indices"],
                                   selected_indices)

  def test_non_max_suppression_with_if(self):
    # if cond
    #   return NonMaxSuppression suppress by IOU
    # else
    #   return NonNaxSuppression suppress by IOU and score
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                       [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                       [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                                101.0]]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.4]).astype(np.float32)
    selected_indices_1 = np.array([[0, 0, 3], [0, 0, 0], [0, 0,
                                                          5]]).astype(np.int64)
    selected_indices_2 = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)

    boxes_in = helper.make_tensor_value_info("boxes", TensorProto.FLOAT,
                                             [None, None, None])
    scores_in = helper.make_tensor_value_info("scores", TensorProto.FLOAT,
                                              [None, None, None])
    max_output_boxes_per_class_in = helper.make_tensor_value_info(
        "max_output_boxes_per_class", TensorProto.INT64, [None])
    iou_threshold_in = helper.make_tensor_value_info("iou_threshold",
                                                     TensorProto.FLOAT, [None])
    score_threshold_in = helper.make_tensor_value_info("score_threshold",
                                                       TensorProto.FLOAT,
                                                       [None])
    cond_in = helper.make_tensor_value_info('cond', TensorProto.BOOL, [])

    selected_indices_1_out = helper.make_tensor_value_info(
        "selected_indices_1", TensorProto.INT64, [None, None])
    selected_indices_2_out = helper.make_tensor_value_info(
        "selected_indices_2", TensorProto.INT64, [None, None])
    selected_indices_out = helper.make_tensor_value_info(
        "selected_indices", TensorProto.INT64, [None, None])

    non_max_suppression_node_1 = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold"],
        ["selected_indices_1"],
        center_point_box=0,
        name='NonMaxSuppression_1')
    non_max_suppression_node_2 = helper.make_node("NonMaxSuppression", [
        "boxes", "scores", "max_output_boxes_per_class", "iou_threshold",
        "score_threshold"
    ], ["selected_indices_2"],
                                                  center_point_box=0,
                                                  name='NonMaxSuppression_2')

    then_graph = helper.make_graph(nodes=[non_max_suppression_node_1],
                                   name="then_graph",
                                   inputs=[
                                       boxes_in, scores_in,
                                       max_output_boxes_per_class_in,
                                       iou_threshold_in
                                   ],
                                   outputs=[selected_indices_1_out])
    else_graph = helper.make_graph(nodes=[non_max_suppression_node_2],
                                   name="then_graph",
                                   inputs=[
                                       boxes_in, scores_in,
                                       max_output_boxes_per_class_in,
                                       iou_threshold_in, score_threshold_in
                                   ],
                                   outputs=[selected_indices_2_out])
    if_node = helper.make_node('If', ['cond'], ["selected_indices"],
                               then_branch=then_graph,
                               else_branch=else_graph)
    graph_def = helper.make_graph(nodes=[if_node],
                                  name='test_if',
                                  inputs=[
                                      boxes_in, scores_in,
                                      max_output_boxes_per_class_in,
                                      iou_threshold_in, score_threshold_in,
                                      cond_in
                                  ],
                                  outputs=[selected_indices_out])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/non_max_suppression/if'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    for cond, exp in [[True, selected_indices_1], [False, selected_indices_2]]:
      tf_model_output = tf_model(
          boxes=boxes,
          scores=scores,
          max_output_boxes_per_class=max_output_boxes_per_class,
          iou_threshold=iou_threshold,
          score_threshold=score_threshold,
          cond=cond)
      np.testing.assert_almost_equal(tf_model_output["selected_indices"], exp)

  def test_scatter_elements(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ScatterElements.".format(
              defs.onnx_opset_version()))
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    indices = np.array([[1, 3]], dtype=np.int64)
    updates = np.array([[1.1, 2.1]], dtype=np.float32)
    axis = 1
    ref_output = np.array([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=np.float32)
    node_def = helper.make_node("ScatterElements",
                                ["data", "indices", "updates"], ["outputs"],
                                axis=axis)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT,
                                          [None, None]),
            helper.make_tensor_value_info("indices", TensorProto.INT64,
                                          [None, None]),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT,
                                          [None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("outputs", TensorProto.FLOAT,
                                          [None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/scatter_elements'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(data=data, indices=indices, updates=updates)
    np.testing.assert_almost_equal(tf_model_output["outputs"], ref_output)

  def test_scatter_nd(self):
    if legacy_opset_pre_ver(11):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support ScatterND.".format(
              defs.onnx_opset_version()))
    # valid positive and negative indices for slices
    data = np.reshape(np.arange(1, 25, dtype=np.float32), [2, 3, 4])
    indices = np.array([[-1]], dtype=np.int64)
    updates = np.array([[[43, 44, 45, 46], [47, 48, 49, 50], [51, 52, 53, 54]]],
                       dtype=np.float32)
    ref_output = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
         [[43, 44, 45, 46], [47, 48, 49, 50], [51, 52, 53, 54]]],
        dtype=np.float32)
    node_def = helper.make_node("ScatterND", ["data", "indices", "updates"],
                                ["outputs"])
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT,
                                          [None, None, None]),
            helper.make_tensor_value_info("indices", TensorProto.INT64,
                                          [None, None]),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT,
                                          [None, None, None])
        ],
        outputs=[
            helper.make_tensor_value_info("outputs", TensorProto.FLOAT,
                                          [None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/scatter_nd'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(data=data, indices=indices, updates=updates)
    np.testing.assert_almost_equal(tf_model_output["outputs"], ref_output)

  def test_max_pool_2d_dilations_ceil_pads(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations nor ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]
    pads = [1, 1, 2, 2]
    ceil_mode = 1

    input_shape = [10, 3, 23, 23]
    x = self._get_rnd_float32(shape=input_shape)

    test_output = py_pool(x,
                          kernel_shape=kernel_shape,
                          strides=strides,
                          dilations=dilations,
                          padding=pads,
                          ceil_mode=ceil_mode,
                          pooling_type="MAX",
                          include_indices=False)

    node_def = helper.make_node(op_type="MaxPool",
                                inputs=["X"],
                                outputs=["Y"],
                                kernel_shape=kernel_shape,
                                strides=strides,
                                dilations=dilations,
                                pads=pads,
                                ceil_mode=ceil_mode)

    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/max_pool_2d_dilations_ceil_pads'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x)
    np.testing.assert_almost_equal(tf_model_output["Y"], test_output)

  def test_max_pool_with_argmax_2d_dilations_ceil_pads(self):
    if legacy_opset_pre_ver(10):
      raise unittest.SkipTest(
          "ONNX version {} doesn't support dilations nor ceil mode.".format(
              defs.onnx_opset_version()))

    kernel_shape = [3, 3]
    strides = [2, 2]
    dilations = [3, 3]
    pads = [1, 1, 2, 2]
    ceil_mode = True

    input_shape = [10, 3, 23, 23]
    x = self._get_rnd_float32(shape=input_shape) - 2

    node_def = helper.make_node("MaxPool", ["X"], ["Y", "Ind"],
                                kernel_shape=kernel_shape,
                                strides=strides,
                                dilations=dilations,
                                pads=pads,
                                ceil_mode=ceil_mode)

    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None]),
            helper.make_tensor_value_info("Ind", TensorProto.INT64,
                                          [None, None, None, None])
        ])

    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/max_pool_with_argmax_2d_dilations_ceil_pads'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x)

    test_output, test_ind = py_pool(x,
                                    kernel_shape=kernel_shape,
                                    strides=strides,
                                    dilations=dilations,
                                    padding=pads,
                                    ceil_mode=ceil_mode,
                                    pooling_type="MAX")

    np.testing.assert_almost_equal(tf_model_output["Y"], test_output)
    np.testing.assert_almost_equal(tf_model_output["Ind"], test_ind)

  def test_average_pool_2d(self):
    kernel_shape = [1, 2]
    strides = [1, 2]

    input_shape = [10, 10, 4, 4]
    x = self._get_rnd_float32(shape=input_shape)

    test_output = py_pool(x,
                          kernel_shape=kernel_shape,
                          strides=strides,
                          pooling_type="AVG",
                          include_indices=False)

    node_def = helper.make_node(op_type="AveragePool",
                                inputs=["X"],
                                outputs=["Y"],
                                kernel_shape=kernel_shape,
                                strides=strides)

    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/average_pool_2d'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x)
    np.testing.assert_almost_equal(tf_model_output["Y"], test_output)

  def test_max_unpool(self):
    input_shape = [10, 3, 24, 24]
    x = self._get_rnd_float32(shape=input_shape)

    kernel_shape = [2, 2]
    strides = [2, 2]

    maxpool_node_def = helper.make_node(op_type="MaxPool",
                                        inputs=["X"],
                                        outputs=["Pool", "Indices"],
                                        kernel_shape=kernel_shape,
                                        strides=strides)

    maxunpool_node_def = helper.make_node("MaxUnpool", ["Pool", "Indices"],
                                          ["Y"],
                                          kernel_shape=kernel_shape,
                                          strides=strides)

    graph_def = helper.make_graph(
        [maxpool_node_def, maxunpool_node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None, None, None, None]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                          [None, None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/max_unpool'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(X=x)

    test_output = np.zeros(input_shape)
    for i1 in range(0, input_shape[0]):
      for i2 in range(0, input_shape[1]):
        for i3 in range(0, input_shape[2], 2):
          for i4 in range(0, input_shape[3], 2):
            max_val = float('-inf')
            for j1 in range(i3, i3 + 2):
              for j2 in range(i4, i4 + 2):
                if x[i1][i2][j1][j2] > max_val:
                  max_val = x[i1][i2][j1][j2]
                  max_ind = (j1, j2)
            j1, j2 = max_ind
            test_output[i1][i2][j1][j2] = max_val
    np.testing.assert_almost_equal(tf_model_output["Y"], test_output)

  def test_slice(self):
    # test case 1 with normal inputs
    axes = [0, 1, 2]
    starts = [0, 0, 0]
    ends = [2, 2, 2]

    if legacy_opset_pre_ver(10):
      node_def = helper.make_node("Slice", ["X"], ["S"],
                                  axes=axes,
                                  starts=starts,
                                  ends=ends)
      graph_def = helper.make_graph(
          [node_def],
          name="test_unknown_shape",
          inputs=[
              helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                            [None, None, None])
          ],
          outputs=[
              helper.make_tensor_value_info("S", TensorProto.FLOAT,
                                            [None, None, None])
          ])
    else:
      node_def = helper.make_node("Slice", ["X", "starts", "ends", "axes"],
                                  ["S"])
      graph_def = helper.make_graph(
          [node_def],
          name="test_unknown_shape",
          inputs=[
              helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                            [None, None, None]),
              helper.make_tensor_value_info("starts", TensorProto.INT32,
                                            [None]),
              helper.make_tensor_value_info("ends", TensorProto.INT32, [None]),
              helper.make_tensor_value_info("axes", TensorProto.INT32, [None]),
          ],
          outputs=[
              helper.make_tensor_value_info("S", TensorProto.FLOAT,
                                            [None, None, None])
          ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/slice'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)

    if legacy_opset_pre_ver(10):
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      tf_model_output = tf_model(X=x)
      np.testing.assert_almost_equal(tf_model_output["S"], x[0:2, 0:2, 0:2])
    else:
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      tf_model_output = tf_model(X=x, starts=starts, ends=ends, axes=axes)
      np.testing.assert_almost_equal(tf_model_output["S"], x[0:2, 0:2, 0:2])

    # test case 2 with negative, out-of-bound and default inputs
    axes = [0, 2]
    starts = [0, -7]
    ends = [-8, 20]
    steps = [1, 1]

    if legacy_opset_pre_ver(10):
      node_def = helper.make_node("Slice", ["X"], ["S"],
                                  axes=axes,
                                  starts=starts,
                                  ends=ends)
      graph_def = helper.make_graph(
          [node_def],
          name="test_unknown_shape",
          inputs=[
              helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                            [None, None, None])
          ],
          outputs=[
              helper.make_tensor_value_info("S", TensorProto.FLOAT,
                                            [None, None, None])
          ])
    else:
      node_def = helper.make_node("Slice",
                                  ["X", "starts", "ends", "axes", "steps"],
                                  ["S"])
      graph_def = helper.make_graph(
          [node_def],
          name="test_unknown_shape",
          inputs=[
              helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                            [None, None, None]),
              helper.make_tensor_value_info("starts", TensorProto.INT32,
                                            [None]),
              helper.make_tensor_value_info("ends", TensorProto.INT32, [None]),
              helper.make_tensor_value_info("axes", TensorProto.INT32, [None]),
              helper.make_tensor_value_info("steps", TensorProto.INT32, [None]),
          ],
          outputs=[
              helper.make_tensor_value_info("S", TensorProto.FLOAT,
                                            [None, None, None])
          ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/slice'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)

    if legacy_opset_pre_ver(10):
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      tf_model_output = tf_model(X=x)
      np.testing.assert_almost_equal(tf_model_output[0], x[0:-8, :, -7:20])
    else:
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      tf_model_output = tf_model(X=x,
                                 starts=starts,
                                 ends=ends,
                                 axes=axes,
                                 steps=steps)
      np.testing.assert_almost_equal(tf_model_output["S"], x[0:-8, :, -7:20])

    # test case 3 with non-default steps
    axes = [0, 1, 2]
    starts = [0, 0, 0]
    ends = [2, 2, 2]
    steps = [2, -2, -1]

    if not legacy_opset_pre_ver(10):
      x = self._get_rnd_float32(shape=[1000]).reshape([10, 10, 10])
      tf_model_output = tf_model(X=x,
                                 starts=starts,
                                 ends=ends,
                                 axes=axes,
                                 steps=steps)
      np.testing.assert_almost_equal(tf_model_output["S"], x[0:2:2, 0:2:-2,
                                                             0:2:-1])

  def test_split(self):
    shape = [12, 12]
    axis = 0
    output_count = 3
    node_def = helper.make_node("Split", ["X"],
                                ["Z%i" % i for i in range(output_count)],
                                axis=axis)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                          [None] * len(shape))
        ],
        outputs=[
            helper.make_tensor_value_info("Z%i" % i, TensorProto.FLOAT,
                                          [None] * len(shape))
            for i in range(output_count)
        ])

    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/split'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    x = self._get_rnd_float32(shape=shape)
    tf_model_output = tf_model(X=x)

    per_part = shape[axis] // output_count
    split = [per_part] * output_count
    for i in range(output_count):
      o = tf_model_output["Z%i" % i]
      np.testing.assert_almost_equal(o, np.split(x, np.cumsum(split))[i])

  def test_trilu(self):
    if legacy_opset_pre_ver(14):
      raise unittest.SkipTest("ONNX version {} doesn't support Trilu.".format(
          defs.onnx_opset_version()))
    shape = [2, 4, 6]
    k = np.array(1).astype(np.int64)
    x = self._get_rnd_int(0, 100, shape=shape)
    y = np.triu(x, k)
    node_def = helper.make_node("Trilu",
                                inputs=["x", "k"],
                                outputs=['y'],
                                upper=1)
    graph_def = helper.make_graph(
        [node_def],
        name="test_unknown_shape",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.INT32,
                                          [None, None, None]),
            helper.make_tensor_value_info("k", TensorProto.INT64, [])
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT,
                                          [None, None, None])
        ])
    tf_rep = onnx_graph_to_tensorflow_rep(graph_def)
    # export to tf.saved_model
    model_path = 'test_dynamic_shape/trilu'
    tf_rep.export_graph(model_path)
    # load the saved_model back
    tf_model = tf.saved_model.load(model_path)
    # run the model
    tf_model_output = tf_model(x=x, k=k)
    np.testing.assert_equal(tf_model_output["y"], y)

    k = np.array(7).astype(np.int64)
    y = np.triu(x, k)
    tf_model_output = tf_model(x=x, k=k)
    np.testing.assert_equal(tf_model_output["y"], y)

    k = np.array(-7).astype(np.int64)
    y = np.triu(x, k)
    tf_model_output = tf_model(x=x, k=k)
    np.testing.assert_equal(tf_model_output["y"], y)

  @classmethod
  def tearDownClass(cls):
    # clean up saved model folder
    try:
      model_path = 'test_dynamic_shape'
      shutil.rmtree(model_path)
    except FileNotFoundError:
      # the model folder doesn't exist
      pass


if __name__ == '__main__':
  unittest.main()
