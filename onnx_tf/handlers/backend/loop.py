import numpy as np
import tensorflow as tf

import onnx_tf
from onnx.helper import make_opsetid
from onnx_tf.common import data_type
from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("Loop")
class Loop(BackendHandler):

  @classmethod
  def get_initializer_from_subgraph(cls, node, init_dict, callback_func):
    return callback_func(node.attrs["body"], init_dict)

  @classmethod
  def create_variables(cls, handlers, node, init_dict, var_dict, callback_func):
    return callback_func(handlers, node.attrs["body"], init_dict, var_dict)

  @classmethod
  def _common(cls, node, **kwargs):
    body = node.attrs["body"]
    tensor_dict = kwargs["tensor_dict"]
    M = tensor_dict[node.inputs[0]] if node.inputs[0] != "" else None
    M = tf.where(tf.greater(M, tf.int32.max), tf.constant(
        tf.int32.max, tf.int32), tf.cast(M, tf.int32)) if M is not None else M
    cond_init = tf.cast(tensor_dict[node.inputs[1]],
                        tf.bool) if node.inputs[1] != "" else None
    v_init = [tensor_dict[graph_input] for graph_input in node.inputs[2:]]
    v_shapes = [
        tf.TensorShape([None for i in range(v.shape.rank)]) for v in v_init
    ]
    iter_cnt_init = np.int64(0)
    current_opset = [make_opsetid(cls.DOMAIN, cls.VERSION)]
    # outputs of the body will be in this format:
    # (condition, loop carried dependencies..., scan_outputs...)
    scan_outputs_start_index = 1 + len(v_init)
    scan_outputs_init = [
        tf.TensorArray(dtype=data_type.onnx2tf(
            body.output[i].type.tensor_type.elem_type),
                       size=0,
                       dynamic_size=True)
        for i in range(scan_outputs_start_index, len(body.output))
    ]
    scan_outputs_shapes = [tf.TensorShape(None) for o in scan_outputs_init]

    def run_subgraph(iter_cnt, cond, v, scan_outputs):
      subgraph_tensor_dict = dict(tensor_dict)
      subgraph_tensor_dict[body.input[0].name] = iter_cnt
      subgraph_tensor_dict[body.input[1].name] = cond
      for i in range(2, len(body.input)):
        subgraph_tensor_dict[body.input[i].name] = v[i - 2]
      subgraph_tensor_dict = onnx_tf.backend.onnx_graph_to_tensorflow_ops(
          subgraph=body, tensor_dict=subgraph_tensor_dict, opset=current_opset)
      outputs = [subgraph_tensor_dict[output.name] for output in body.output]
      for i in range(scan_outputs_start_index, len(outputs)):
        s_index = i - scan_outputs_start_index
        insert_index = scan_outputs[s_index].size()
        scan_outputs[s_index] = scan_outputs[s_index].write(
            insert_index, outputs[i])
      iter_cnt += 1
      return iter_cnt, outputs[0], outputs[
          1:scan_outputs_start_index], scan_outputs

    # for loop
    if M is not None and cond_init is None:
      condition = lambda iter_cnt, cond, v, scan_outputs: True
      iter_cnt_final, _, v_final, scan_outputs_final = tf.while_loop(
          cond=condition,
          body=run_subgraph,
          loop_vars=[iter_cnt_init, "", v_init, scan_outputs_init],
          shape_invariants=[
              tf.TensorShape([]),
              tf.TensorShape(None), v_shapes, scan_outputs_shapes
          ],
          maximum_iterations=M)
    # while and do-while loop
    elif M is None and cond_init is not None:
      condition = lambda iter_cnt, cond, v, scan_outputs: tf.reduce_all(
          tf.equal(cond, True))
      iter_cnt_final, cond_final, v_final, scan_outputs_final = tf.while_loop(
          cond=condition,
          body=run_subgraph,
          loop_vars=[iter_cnt_init, cond_init, v_init, scan_outputs_init],
          shape_invariants=[
              tf.TensorShape([]),
              tf.TensorShape(None), v_shapes, scan_outputs_shapes
          ])
    # combine for loop and while loop together
    elif M is not None and cond_init is not None:
      condition = lambda iter_cnt, cond, v, scan_outputs: tf.reduce_all(
          tf.equal(cond, True))
      iter_cnt_final, cond_final, v_final, scan_outputs_final = tf.while_loop(
          cond=condition,
          body=run_subgraph,
          loop_vars=[iter_cnt_init, cond_init, v_init, scan_outputs_init],
          shape_invariants=[
              tf.TensorShape([]),
              tf.TensorShape(None), v_shapes, scan_outputs_shapes
          ],
          maximum_iterations=M)
    else:  # M is None and cond is None
      exception.OP_UNSUPPORTED_EXCEPT(
          "Both M and cond in Loop are not set at the same time",
          "Tensorflow.(PS. if you want to create a do-while loop " +
          "then please set cond to True or 1)")

    if scan_outputs_start_index == len(body.output):
      # there is no scan_output in the body graph
      return v_final
    else:
      # if the loop has run >= 1 time then do nothing
      def true_fn():
        return scan_outputs_final

      # if the loop didn't run at all then recreate the scan_outputs'
      # TensorArray and set the element_shape to [0].
      # Then tensorflow will allow to append the empty tensor
      # to v_final
      def false_fn():
        new_scan_outputs = []
        for i in range(scan_outputs_start_index, len(body.output)):
          exp_elem_shape = scan_outputs_init[
              i - scan_outputs_start_index].element_shape
          elem_shape = []
          for j in range(exp_elem_shape.rank):
            shape_j = 0 if exp_elem_shape[j] is None else exp_elem_shape[j]
            elem_shape.append(shape_j)
          new_scan_outputs.append(
              tf.TensorArray(dtype=data_type.onnx2tf(
                  body.output[i].type.tensor_type.elem_type),
                             size=0,
                             element_shape=tf.TensorShape(elem_shape)))
        return new_scan_outputs

      scan_out_final = tf.cond(tf.greater(iter_cnt_final, 0), true_fn, false_fn)
      scan_outputs_tensors = [o.stack() for o in scan_out_final]
      return v_final + scan_outputs_tensors

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
    return cls._common(node, **kwargs)
