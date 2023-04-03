import onnx
import onnxruntime as ort


def print_input_shape(model_path):
    model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # model = ort.InferenceSession(model_path, providers=['Te','CPUExecutionProvider'])
    input_shape = model.get_inputs()[0].shape
    output_shape = model.get_outputs()[0].shape
    print("Input shape: {}".format(input_shape))
    print("Output shape: {}".format(output_shape))
    
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    print("Input name:", input_name)
    print("Output name:", output_name)
    
    
def print_model_graph(model_path): 
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))
    
    
if __name__=="__main__": 
    model_path = "./deploy/onnx_models/model_batch.onnx"
    
    print_model_graph(model_path)
    print("====================")
    print_input_shape(model_path)
    
    