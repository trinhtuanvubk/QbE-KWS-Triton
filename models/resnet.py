import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(1, 64, 3, stride=(2,1), padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(8, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc5 = nn.Linear(512 * 14, 512)
        self.conv_out = nn.Conv2d(512*8*8,512,1)
        self.conv_1x1 = nn.Conv2d(160, 256, 1)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1,2).unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = nn.AvgPool2d(kernel_size=x.size()[2:])(x)
        # x = self.avgpool(x)


        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc5(x)
        # x = x.mean(-1).unsqueeze(-1)
        # print(x.shape)
        # x = self.conv_out(x)
        # x = torch.squeeze(x)


        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


# =========================================================================



# import torch
# from nemo.backends.pytorch import TrainableNM
# from nemo.core.neural_types import *
# from nemo.utils.decorators import add_port_docs
# from torch import nn
# from torch.nn import functional as F


# class Swish(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i * torch.sigmoid(i)
#         ctx.save_for_backward(i)
#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         i = ctx.saved_variables[0]
#         sigmoid_i = torch.sigmoid(i)
#         return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


# class CustomSwish(nn.Module):
#     def forward(self, input_tensor):
#         return Swish.apply(input_tensor)


# class Res15(TrainableNM):
#     @property
#     @add_port_docs()
#     def input_ports(self):
#         """Returns definitions of module input ports.
#         """
#         return {
#             "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
#             "length": NeuralType(tuple('B'), LengthsType()),
#         }

#     @property
#     @add_port_docs()
#     def output_ports(self):
#         """Returns definitions of module output ports.
#         """
#         return {
#             "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
#             "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
#         }

#     def __init__(self, n_maps):
#         super().__init__()
#         n_maps = n_maps
#         self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
#         self.n_layers = n_layers = 13
#         dilation = True
#         if dilation:
#             self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2 ** (i // 3)), dilation=int(2 ** (i // 3)),
#                                     bias=False) for i in range(n_layers)]
#         else:
#             self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
#                                     bias=False) for _ in range(n_layers)]
#         for i, conv in enumerate(self.convs):
#             self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
#             self.add_module("conv{}".format(i + 1), conv)

#     def forward(self, audio_signal, length=None):
#         x = audio_signal.unsqueeze(1)
#         for i in range(self.n_layers + 1):
#             y = F.relu(getattr(self, "conv{}".format(i))(x))
#             if i == 0:
#                 if hasattr(self, "pool"):
#                     y = self.pool(y)
#                 old_x = y
#             if i > 0 and i % 2 == 0:
#                 x = y + old_x
#                 old_x = x
#             else:
#                 x = y
#             if i > 0:
#                 x = getattr(self, "bn{}".format(i))(x)
#         x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
#         x = torch.mean(x, 2)
#         return x.unsqueeze(-2), length


# class Res8(TrainableNM):
#     @property
#     @add_port_docs()
#     def input_ports(self):
#         """Returns definitions of module input ports.
#         """
#         return {
#             "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
#             "length": NeuralType(tuple('B'), LengthsType()),
#         }

#     @property
#     @add_port_docs()
#     def output_ports(self):
#         """Returns definitions of module output ports.
#         """
#         return {
#             "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
#             "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
#         }

#     def __init__(self, hidden_size):
#         super().__init__()
#         n_maps = hidden_size
#         self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
#         self.pool = nn.AvgPool2d((3, 4))  # flipped -- better for 80 log-Mels

#         self.n_layers = n_layers = 6
#         self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, bias=False) for _ in range(n_layers)]
#         for i, conv in enumerate(self.convs):
#             self.add_module(f'bn{i + 1}', nn.BatchNorm2d(n_maps, affine=False))
#             self.add_module(f'conv{i + 1}', conv)

#     def forward(self, audio_signal, length=None):
#         x = audio_signal.unsqueeze(1)
#         x = x.permute(0, 1, 3, 2).contiguous()  # Original res8 uses (time, frequency) format
#         for i in range(self.n_layers + 1):
#             y = F.relu(getattr(self, f'conv{i}')(x))
#             if i == 0:
#                 if hasattr(self, 'pool'):
#                     y = self.pool(y)
#                 old_x = y
#             if i > 0 and i % 2 == 0:
#                 x = y + old_x
#                 old_x = x
#             else:
#                 x = y
#             if i > 0:
#                 x = getattr(self, f'bn{i}')(x)
#         x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
#         x = torch.mean(x, 2)
#         return x.unsqueeze(-2)