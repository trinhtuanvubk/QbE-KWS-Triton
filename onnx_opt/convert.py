import torch
import os
from utils.args import get_args 
import models 

args = get_args()

def load_model(ckpt_path,args):
    model = models.get_model(args)
    ############
    model = model.to(args.device)
    ############
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model



def convert_torch_to_onnx_batch(model, output_path, dummy_input, device=None):

    input_names = ["input_audio"]
    output_names = ["output_embedding"]
    
    if device!=None:
        model = model.to(device)
        dummy_input = dummy_input.to(device)
    
    torch.onnx.export(model, 
                 dummy_input, 
                 output_path, 
                 verbose=True, 
                 input_names=input_names, 
                 output_names=output_names,
                 dynamic_axes={'input_audio' : {0: 'batch_size'},    # variable length axes
                               'output_asr' : {0:'batch_size'}})
    
if __name__=="__main__":

    device = torch.device('cuda:0')
    args.device = device
    args.metric = 'softmax'
    args.n_keyword = 35
    print(device)
    output_path = "./deploy/onnx_models/model_batch.onnx"

    ckpt_path = "./ckpt/bcres_softmax_ce_35/checkpoints/model.ckpt"
    model = models.get_model(args)
    model = model.to(args.device)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.embedding 
 
    dummy_input = torch.rand([3,101,40])
  
    
    convert_torch_to_onnx_batch(model,output_path,dummy_input,device=device)