import torch 
import torchaudio
import models
import numpy as np 
import argparse
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def get_args():
    # create args parser
    parser = argparse.ArgumentParser(description='Kws Trainer')

    # parameter for dataset
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--no_extract', action='store_false')

    # parameter for model
    parser.add_argument('--model', type=str, default='bcres')
    parser.add_argument('--metric', type=str, default='cosface')
    parser.add_argument('--loss', type=str, default='ce')

    # parameter for model's hyper parameters
    parser.add_argument('--n_mels', type=int, default=40)
    parser.add_argument('--n_fft', type=int, default=400)
    parser.add_argument('--cnn_channel', type=str, default='512,512,512,512,1500')
    parser.add_argument('--cnn_kernel', type=str, default='5,3,3,1,1')
    parser.add_argument('--cnn_dilation', type=str, default='1,2,3,1,1')
    parser.add_argument('--n_embed', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_keyword', type=int, default=12)

    # parameter for loss's hyper parameters
    parser.add_argument('--m', type=float, default=0.5)
    parser.add_argument('--s', type=float, default=64)
    parser.add_argument('--plus', action='store_true')
    parser.add_argument('--std', type=float, default=0.0125)

    # parameter for scenario
    parser.add_argument('--scenario', type=str, default='train')

    # parameter for training
    parser.add_argument('--no_shuffle', action='store_false')
    parser.add_argument('--no_evaluate', action='store_false')
    parser.add_argument('--clear_cache', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--no_pin_memory', action='store_false')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)
    parser.add_argument('--limit_train_batch', type=int, default=-1)
    parser.add_argument('--limit_val_batch', type=int, default=-1)


    # parse args
    args = parser.parse_args()
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = torch.device("cpu")

    return args

def load_audio(file_path):
    audio, _ = torchaudio.load(file_path)
    # length = torch.tensor(audio.shape[1] / pad_audio.shape[1])
    # print("dmcmcm",audio.shape[0])
    # if audio.shape[0]>1: 
        # audio = torch.unsqueeze(audio[1],0)
    return audio
    
def padding(x):
    if x.shape[1] >= 16000:
        x = x[:,:16000]
    else:
        pad = torch.zeros(1,16000)
        pad[:,:x.shape[1]] = x
        x = pad
    return x

def load_model(ckpt_path,args):
    model = models.get_model(args)
    ############
    model = model.to(args.device)
    ############
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def init_embedding(model,args,audio):
    model=model.embedding
    feature = models.MelSpectrogram(args).feature(audio)
    print("feature",feature.shape)
    # print(feature)
    output = model(feature)
    output_torch = output.cpu().detach()
    output_np =  output.cpu().detach().numpy()
    return output_np, output_torch 

def average_embedding(path,keyword,ckpt_path,args):
    folder_path = f"{path}/{keyword}/"
    # folder_path = f"./data/{keyword}/"
    model = load_model(ckpt_path,args)

    embeddings =[]
    embeddings_torch = []
    for file in os.listdir(folder_path)[:5]:
        file_path = os.path.join(folder_path,file)
        audio = load_audio(file_path)
        pad_audio = padding(audio)
        embed, embed_torch = init_embedding(model,args,pad_audio.to(args.device))
        # print("embed:",embed)
        embeddings.append(embed)
        embeddings_torch.append(embed_torch)
    average = sum(embeddings_torch)/len(embeddings_torch)
    return average, embeddings_torch
    # return average, embeddings

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)

def compute_threshold(vectors:list):
    print("vecter1",vectors[1].shape)
    dist = []
    for i in range(len(vectors)-1):
        cos = torch.nn.CosineSimilarity(dim=0)
        d = cos(vectors[i],vectors[i+1])
        dist.append(d)
    return sum(dist)/len(dist)

if __name__=="__main__":
    args = get_args()
    args.device = torch.device("cpu")
    keyword = "hello"
    keyword = "turnofftv"
    path = "./data"
    path = "./sc_smarthome2"
    ckpt_path = "./ckpt/bcres_cosface_ce_12_aug_95/checkpoints/model.ckpt"
    print(args.device)
    # file = "./data/hello/hello_2022_06_06-08:58:22_PM_50.wav"
    file = "./sc_smarthome2/closedoor/1555950235754.wav"
    # audio, _ = torchaudio.load(file)
    audio = load_audio(file)
    print(audio.shape)
    print("au:",audio)
    pad_audio = padding(audio)
    x = pad_audio.squeeze(0).to(args.device)
    x = pad_audio.to(args.device)
    print("pad:",x)
    print(x.shape)
    # length = torch.tensor(audio.shape[1] / pad_audio.shape[1])
    model = load_model(ckpt_path,args)
    embed,embed_torch = init_embedding(model,args,x)
    # print(embed)
    print(embed.shape)
    # print(len(embed[1]))
    print("OK")
    average, embedding = average_embedding(path,keyword,ckpt_path,args)
    # print(average)
    print(average.shape)

    thres = compute_threshold(embedding)
    print(thres)