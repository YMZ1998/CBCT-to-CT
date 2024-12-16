# tensorrt版本不对
# 可尝试 https://github.com/onnx/onnx-tensorrt
# Error Code 9: API Usage Error (Target GPU SM 61 is not supported by this TensorRT release.)
# 硬件设备可能不支持，编译好的模型对不同架构上可能存在差异，需要自行编译。
# https://forums.developer.nvidia.com/t/unsupported-sm-0x601/289377

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
import tensorrt as trt
import onnx
import sys

# 加载 ONNX 模型
onnx_model_path = "model.onnx"
onnx_model = onnx.load(onnx_model_path)
# 使用 onnx.checker.check_model 来验证模型的有效性
try:
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")
except onnx.onnx_cpp2py_export.checker.ValidationError as e:
    print("ONNX model is invalid.")
    print(e)

# 创建 TensorRT 解析器
logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(flags=(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)))

# 通过 ONNX 解析器加载模型
onnx_parser = trt.OnnxParser(network, logger)
with open(onnx_model_path, 'rb') as f:
    if not onnx_parser.parse(f.read()):
        print("ERROR: Failed to parse the ONNX model.")
        for error in range(onnx_parser.num_errors):
            print(onnx_parser.get_error(error))
        sys.exit(1)

# 创建构建配置
config = builder.create_builder_config()

# 设置最大工作区大小（例如：1GB）
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
# 设置网络的最大批量大小，直接在网络定义中设置
network.get_input(0).shape = [1, 5, 320, 320]
network.get_input(1).shape = [1, 1, 320, 320]
# 如果模型的输入批量大小是动态的，可以使用以下方法
profile = builder.create_optimization_profile()
profile.set_shape("input0", min=(1, 5, 320, 320), opt=(1, 5, 320, 320), max=(32, 5, 320, 320))
profile.set_shape("input1", min=(1, 1, 320, 320), opt=(1, 1, 320, 320), max=(32, 1, 320, 320))

config.add_optimization_profile(profile)

# 使用 build_engine_with_config 构建引擎
engine = builder.build_engine_with_config(network, config)

# 检查引擎是否成功构建
if engine is not None:
    # 保存引擎
    with open("model.trt", "wb") as f:
        f.write(engine.serialize())
    print("TensorRT engine saved successfully!")
else:
    print("Failed to build the engine.")
