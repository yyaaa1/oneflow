"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np

from onnx import numpy_helper
import tensorflow as tf

import oneflow as flow
from oneflow.python.ops import get_variable
from oneflow.python.onnx import util
from oneflow.python.onnx.load.handlers.backend_handler import BackendHandler
from oneflow.python.onnx.handler import onnx_op
from oneflow.python.onnx.handler import tf_func

import os


@onnx_op("Constant")
@tf_func(get_variable.api_get_variable)
class Constant(BackendHandler):
    @classmethod
    def _common(cls, node, **kwargs):
        attr_value = node.attrs["value"]
        dtype = util.Onnx2FlowDtype(attr_value.data_type)
        shape = numpy_helper.to_array(attr_value).shape
        # we do not support 0d tensor
        if len(shape) == 0:
            shape = (1,)
        return [
            cls.run_onnx_node(
                node,
                # inputs=[value],
                # attrs={"dtype": dtype}
                name=node.output_tensor_names[0],
                attrs={
                    "dtype": dtype,
                    "trainable": False,
                    "shape": shape,
                    "initializer": flow.zeros_initializer(),
                },
            )
        ]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_9(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_11(cls, node, **kwargs):
        # either value or sparse_value
        if "value" in node.attrs:
            return cls._common(node, **kwargs)
        else:
            sparse_value = node.attrs["sparse_value"]
            indices = numpy_helper.to_array(sparse_value.indices)
            values = numpy_helper.to_array(sparse_value.values)
            shape = np.array(sparse_value.dims)
        return [tf.SparseTensor(indices, values, shape)]

    @classmethod
    def version_12(cls, node, **kwargs):
        if "value" in node.attrs or "sparse_value" in node.attrs:
            return cls.version_11(node, **kwargs)
        elif "value_float" in node.attrs:
            value = node.attrs["value_float"]
            dtype = tf.float32
        elif "value_floats" in node.attrs:
            value = node.attrs["value_floats"]
            dtype = tf.float32
        elif "value_int" in node.attrs:
            value = node.attrs["value_int"]
            dtype = tf.int64
        elif "value_ints" in node.attrs:
            value = node.attrs["value_ints"]
            dtype = tf.int64
        elif "value_string" in node.attrs:
            value = node.attrs["value_string"]
            dtype = tf.string
        elif "value_strings" in node.attrs:
            value = node.attrs["value_strings"]
            dtype = tf.string
        return [
            cls.run_onnx_node(node, inputs=[value], attrs={"dtype": dtype})
        ]