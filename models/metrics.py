import torch.nn.functional as F
import torch.nn as nn
import torch
import math


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class SoftMax(nn.Module):

    def __init__(self, model, args):
        super().__init__()

        # embedding model
        self.embedding = model

        # classifier model
        self.classifier = nn.Linear(args.n_embed, args.n_keyword)

    def forward(self, x):
        x = self.embedding(x)
        x = self.classifier(x)

        return x

class CosFace(nn.Module):
    def __init__(self, model, args):
        super(CosFace, self).__init__()
        self.embedding = model
        self.s = args.s
        self.m = args.m
        self.W = nn.Parameter(torch.FloatTensor(args.n_keyword, args.n_embed))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        # compute embedding
        x = self.embedding(x)
        # normalize features
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)

        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output
class ElasticCosFace(nn.Module):
    def __init__(self, model,args, plus=False):
        super(ElasticCosFace, self).__init__()
        self.embbedings = model
        self.s = args.s
        self.m = args.m
        self.W = nn.Parameter(torch.FloatTensor(args.n_keyword, args.n_embed))
        nn.init.xavier_uniform_(self.W)
        self.std=args.std
        self.plus=plus

    def forward(self, x, label):
        x = self.embbedings(x)
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        cos_theta = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7) 
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device)  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret



class ArcFace(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, model, args):
        super(ArcFace, self).__init__()
        self.embedding = model
        self.classnum = args.n_keyword
        self.kernel = nn.Parameter(torch.Tensor(args.n_embed,args.n_keyword))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = args.m # the margin value, default is 0.5
        self.s = args.s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(args.m)
        self.sin_m = math.sin(args.m)
        self.mm = self.sin_m * args.m  # issue 1
        self.threshold = math.cos(math.pi - args.m)
    def forward(self, x, label):
        # weights norm
        embbedings = self.embedding(x)
        embbedings = l2_norm(embbedings, axis=0)
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2 + 1e-7)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output
