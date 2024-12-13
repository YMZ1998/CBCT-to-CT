# tensorrt版本不对
# 可尝试 https://github.com/onnx/onnx-tensorrt
# import onnx
# import onnx_tensorrt.backend as backend
# import numpy as np
#
# model = onnx.load("/path/to/model.onnx")
# engine = backend.prepare(model, device='CUDA:1')
# input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
# output_data = engine.run(input_data)[0]
# print(output_data)
# print(output_data.shape)
