"""
Transformer, linear with dynamic quantization 
CNN with static quantization 

"""

import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

fp32_path = './deploy/onnx_models/model_batch.onnx'

dynamic_quant_path = './deploy/onnx_models/dynamic_quant.onnx'
dynamic_quantized_model = quantize_dynamic(fp32_path, dynamic_quant_path)

# static_quant_path = './deploy/onnx_models/static_quant.onnx'
# static_quantized_model = quantize_static(fp32_path, static_quant_path, per_channel=False,
#         weight_type=QuantType.QInt8,
#         optimize_model=False)

