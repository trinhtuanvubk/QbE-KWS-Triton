
from .metrics import *
from .model import *
from .melspectrogram import *
from .resnet import *

def get_embedding_model(args):
    models = {
        'bcres': BCResNet(args).to(args.device),
        'resnet': resnet18().to(args.device),
    }
    try:
        return models[args.model]
    except:
        raise NotImplementedError

def get_model(args):
    models = {
        'softmax': SoftMax(get_embedding_model(args), args).to(args.device),
        # 'adacos': AdaCos(get_embedding_model(args), args).to(args.device),
        'arcface': ArcFace(get_embedding_model(args), args).to(args.device),
        'cosface': CosFace(get_embedding_model(args), args).to(args.device),
        # 'sphereface': SphereFace(get_embedding_model(args), args).to(args.device),
        # 'elasticarc': ElasticArcFace(get_embedding_model(args), args).to(args.device),
        'elasticcos': ElasticCosFace(get_embedding_model(args),args).to(args.device),
    }
    try:
        return models[args.metric]
    except:
        raise NotImplementedError