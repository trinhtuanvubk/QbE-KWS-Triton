### Streamlit Demo

###### Run classify demo 
```bash
python3 -m scenario.classify_demo 
```

###### Run query by example demo 
```bash
python3 -m scenario.qbe_demo 
```

### Serve API Service with Triton 

###### Convert model to onnx 
- To convert 
```bash
python3 -m onnx_opt.convert
```
- To quantize (optional)
```bash
python3 -m onnx.quantize 
```
- To move onnx model to model_repository
```bash
mv deploy/onnx_models/model_batch.onxx deploy/model_repository/1/model.onnx
```

###### Run triton server 
- To run server 
```bash
docker run --gpus=all -itd \
-p8000:8000 -p8001:8001 -p8002:8002 \
-v$(pwd)/deploy/model_repository/:/models \
--name kws_triton \
nvcr.io/nvidia/tritonserver:22.01-py3 bash
```
### TODO
###### Write API 

###### Run API service

###### Remove unused libs and update requirements 

