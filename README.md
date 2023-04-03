### Streamlit Demo

###### Run classify demo 
```bash
python3 -m scenario.classify_demo 
```

###### Run query by example demo 
```bash
python3 -m scenario.qbe_demo 
```

### Serve API Model 

###### Convert model to onnx 
```bash
python3 -m onnx_opt.convert
```

###### Quantize model (optional)
```bash
python3 -m onnx.quantize 
```

###### Move model to model_repository 
```bash
mv deploy/onnx_models/model_batch.onxx deploy/model_repository/1/
```

### TODO
###### Run triton server 

###### Write API 

###### Run API service

