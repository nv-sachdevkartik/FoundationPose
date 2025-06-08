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
import sys
import torch
import os
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from typing import List

# If you face the following issue:
#  "pycuda._driver.LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
#  Add "import pycuda.autoinit", this is needed to initialize cuda!
import pycuda.autoinit

TRT_DYNAMIC_DIM = -1


class HostDeviceMem(object):
    """Simple helper data class to store Host and Device memory."""

    def __init__(self, host_mem, device_mem, binding):
        self.host = host_mem
        self.device = device_mem
        self.binding = binding

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device) + "\nBinding:\n" + str(self.binding)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine: trt.ICudaEngine, batch_size: int, verbose: bool = False) -> [list, list, list]:
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
        if verbose:
            print(f"binding: {binding}")
        binding_shape = engine.get_tensor_shape(binding)
        if binding_shape[0] == TRT_DYNAMIC_DIM:  # dynamic shape
            size = batch_size * abs(trt.volume(binding_shape))
        else:
            size = abs(trt.volume(binding_shape))
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        dbindings.append(int(device_mem))

        # Append to the appropriate list (input/output)
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem, binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, binding))

    return inputs, outputs, dbindings


class TensorRTInfer:
    def __init__(self, engine_path: str, batch_size: int = 1, verbose: bool = False):
        """
        Initialize TensorRT inference engine.

        Args:
            engine_path (str): Path to the TensorRT engine file
            batch_size (int): Batch size for inference
        """
        self.batch_size = batch_size
        self.engine_path = engine_path
        self.verbose = verbose

        # Load engine
        with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.ERROR)) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        # Allocate buffers
        self.inputs, self.outputs, self.dbindings = allocate_buffers(self.engine, batch_size, verbose=self.verbose)
        if self.verbose:
            print(f"num of inputs: {len(self.inputs)}")
            print(f"num of outputs: {len(self.outputs)}")
            print(f"num of dbindings: {len(self.dbindings)}")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Set input shapes
        for binding in self.engine:
            binding_shape = self.engine.get_tensor_shape(binding)
            if self.verbose:
                print(f"binding: {binding}, shape: {binding_shape}")

            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                if TRT_DYNAMIC_DIM in binding_shape:
                    binding_shape = tuple([batch_size if dim == TRT_DYNAMIC_DIM else dim for dim in binding_shape])
                self.context.set_input_shape(binding, binding_shape)

    def infer(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Run inference on two input images.

        Args:
            image1 (np.ndarray): First input image of shape (batch_size, 3, 160, 160)
            image2 (np.ndarray): Second input image of shape (batch_size, 3, 160, 160)

        Returns:
            np.ndarray: Model output
        """
        
        # Copy input data to host buffer
        for i in range(len(self.inputs)):
            np.copyto(self.inputs[i].host, images[i].ravel())
        
        # Transfer input to device
        for i in range(len(self.inputs)):
            cuda.memcpy_htod(self.inputs[i].device, self.inputs[i].host)

        # Run inference
        self.context.execute_v2(self.dbindings)
        
        # Transfer output back to host
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[i].host, self.outputs[i].device)
        output = [np.array(self.outputs[i].host) for i in range(len(self.outputs))]

        return output

    def __del__(self):
        """Cleanup CUDA resources"""
        del self.context
        del self.engine


import time

if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    engine_paths = [
        f"{code_dir}/../../weights/2023-10-28-18-33-37/model_best_dynamic.plan",
        f"{code_dir}/../../weights/2024-01-11-20-02-45/model_best_dynamic.plan"
    ]

    batch_size = 1
    num_inferences_per_engine = 1000

    for engine_path in engine_paths:
        trt_infer = TensorRTInfer(engine_path, batch_size=batch_size, verbose=False)
        for i in range(num_inferences_per_engine):
            image1 = np.random.randn(batch_size, 6, 160, 160).astype(np.float32)
            image2 = np.random.randn(batch_size, 6, 160, 160).astype(np.float32)
            start_time = time.time()
            output = trt_infer.infer([image1, image2])
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"Running inference {i+1}/{num_inferences_per_engine} | Inference time: {inference_time:.2f} ms")
            # print(f"Output: {output}")
