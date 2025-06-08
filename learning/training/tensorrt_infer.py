#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import os
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# If you face the following issue:
#  "pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
#  Add "import pycuda.autoinit", this is needed to initialize cuda!
import pycuda.autoinit

TRT_DYNAMIC_DIM = -1


class HostDeviceMem(object):
    """Simple helper data class to store Host and Device memory."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine: trt.ICudaEngine, batch_size: int) -> [list, list, list]:
    """
    Function to allocate buffers and bindings for TensorRT inference.

    Args:
        engine (trt.ICudaEngine):
        batch_size (int): batch size to be used during inference.

    Returns:
        inputs (List): list of input buffers.
        outputs (List): list of output buffers.
        dbindings (List): list of device bindings.
    """
    inputs = []
    outputs = []
    dbindings = []

    for binding in engine:
        binding_shape = engine.get_tensor_shape(binding)
        if binding_shape[0] == TRT_DYNAMIC_DIM:  # dynamic shape
            size = batch_size * abs(trt.volume(binding_shape))
        else:
            size = abs(trt.volume(binding_shape))
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings
        dbindings.append(int(device_mem))

        # Append to the appropriate list (input/output)
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, dbindings


def infer(
    engine_path: str,
    batch_size: int = 1,
) -> None:
    """
    Performs inference in TensorRT engine.

    Args:
        engine_path (str): path to the TensorRT engine.
        val_batches (tf.data.Dataset): validation dataset (batches).
        batch_size (int): batch size used for inference and dataset batch splitting.
        top_k_value (int): value of `K` for the top K predictions used in the accuracy calculation.

    Raises:
        RuntimeError: raised when loading images in the host fails.
    """

    def override_shape(shape: tuple) -> tuple:
        """Overrides batch dimension if dynamic."""
        if TRT_DYNAMIC_DIM in shape:
            shape = tuple(
                [batch_size if dim == TRT_DYNAMIC_DIM else dim for dim in shape]
            )
        return shape

    # Open engine as runtime
    with open(engine_path, "rb") as f, trt.Runtime(
        trt.Logger(trt.Logger.ERROR)
    ) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        # Allocate buffers and create a CUDA stream.
        inputs, outputs, dbindings = allocate_buffers(engine, batch_size)

        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Resolves dynamic shapes in the context
            for binding in engine:
                binding_shape = engine.get_tensor_shape(binding)
                if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    binding_shape = override_shape(binding_shape)
                    # context.set_input_shape('input_tensor_name', shape)
                    context.set_input_shape(binding, binding_shape)


            data = np.random.randn(batch_size, 6, 160, 160).astype(np.float32).ravel() # TODO: random input
            pagelocked_buffer = inputs[0].host
            np.copyto(pagelocked_buffer, data)
            inp = inputs[0]
            # Transfer input data from Host to Device (GPU)
            cuda.memcpy_htod(inp.device, inp.host)
            # Run inference
            context.execute_v2(dbindings)
            # Transfer predictions back to Host from GPU
            out = outputs[0]
            cuda.memcpy_dtoh(out.host, out.device)

            output = np.array(out.host)
            print(output)



if __name__ == "__main__":
    # engine_path = "/home/ksachdev/repos/FoundationPose/weights/2023-10-28-18-33-37/model_best_dynamic.plan"
    engine_path = "/home/ksachdev/repos/FoundationPose/weights/2023-10-28-18-33-37/model_best.plan"
    infer(engine_path, batch_size=1)
