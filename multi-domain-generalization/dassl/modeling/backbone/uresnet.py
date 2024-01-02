from itertools import permutations
from turtle import forward
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from torch.nn import functional as F

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
        dim   (int): dimension of feature map channels

    """

    def __init__(self, p=0.5, eps=1e-8, dim=-1):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x


'''
class CorrelatedDistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
        dim   (int): dimension of feature map channels
    """

    def __init__(self, p=0.5, eps=1e-6, alpha=0.3):
        super(CorrelatedDistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        B, C = x.size(0), x.size(1)
        mu = torch.mean(x, dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        # mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        factor = self.beta.sample((B, 1, 1, 1)).to(x.device)

        mu_squeeze = torch.squeeze(mu)
        mean_mu = torch.mean(mu_squeeze, dim=0, keepdim=True)
        correlation_mu = (mu_squeeze-mean_mu).T @ (mu_squeeze-mean_mu) / B

        sig_squeeze = torch.squeeze(sig)
        mean_sig = torch.mean(sig_squeeze, dim=0, keepdim=True)
        correlation_sig = (sig_squeeze.T-mean_sig.T) @ (sig_squeeze-mean_sig) / B

        with torch.no_grad():
            try:
                mu_norm = torch.diag(1./torch.sqrt(torch.diag(correlation_mu)+self.eps))
                _, mu_eng_vector = torch.linalg.eigh(mu_norm@correlation_mu@mu_norm.T + 0.0001*torch.eye(C, device=x.device))
                # mu_corr_matrix = mu_eng_vector @ torch.sqrt(torch.diag(torch.clip(mu_eng_value, min=1e-10))) @ (mu_eng_vector.T)
            except:
                mu_eng_vector = torch.eye(C, device=x.device)
            
            if not torch.all(torch.isfinite(mu_eng_vector)) or torch.any(torch.isnan(mu_eng_vector)):
                mu_eng_vector = torch.eye(C, device=x.device)
            
            try:
                sig_norm = torch.diag(1./torch.sqrt(torch.diag(correlation_sig)+self.eps))
                _, sig_eng_vector = torch.linalg.eigh(sig_norm@correlation_sig@sig_norm.T + 0.0001*torch.eye(C, device=x.device))
                # sig_corr_matrix = sig_eng_vector @ torch.sqrt(torch.diag(torch.clip(sig_eng_value, min=1e-10))) @ (sig_eng_vector.T)
            except:
                sig_eng_vector = torch.eye(C, device=x.device)
            
            if not torch.all(torch.isfinite(sig_eng_vector )) or torch.any(torch.isnan(sig_eng_vector)):
                sig_eng_vector = torch.eye(C, device=x.device)


        mu_corr_matrix = mu_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((mu_eng_vector.T)@ correlation_mu @ mu_eng_vector),min=1e-12))) @ (mu_eng_vector.T)
        sig_corr_matrix = sig_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((sig_eng_vector.T)@ correlation_sig @ sig_eng_vector), min=1e-12))) @ (sig_eng_vector.T)

        gaussian_mu = (torch.randn(B, 1, C, device=x.device) @ mu_corr_matrix)
        gaussian_mu = torch.reshape(gaussian_mu, (B, C, 1, 1))

        gaussian_sig = (torch.randn(B, 1, C, device=x.device) @ sig_corr_matrix)
        gaussian_sig = torch.reshape(gaussian_sig, (B, C, 1, 1))

        mu_mix = mu + factor*gaussian_mu
        sig_mix = sig + factor*gaussian_sig

        return x_normed * sig_mix + mu_mix
'''



class CorrelatedDistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
        dim   (int): dimension of feature map channels

    """

    def __init__(self, p=0.5, eps=1e-6, alpha=0.3):
        super(CorrelatedDistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.alpha = alpha
        self.beta = torch.distributions.Beta(alpha, alpha)
    
    def __repr__(self):
        return f'CorrelatedDistributionUncertainty with p {self.p} and alpha {self.alpha}'

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        B, C = x.size(0), x.size(1)
        mu = torch.mean(x, dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        # mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        factor = self.beta.sample((B, 1, 1, 1)).to(x.device)

        mu_squeeze = torch.squeeze(mu)
        mean_mu = torch.mean(mu_squeeze, dim=0, keepdim=True)
        correlation_mu = (mu_squeeze-mean_mu).T @ (mu_squeeze-mean_mu) / B

        sig_squeeze = torch.squeeze(sig)
        mean_sig = torch.mean(sig_squeeze, dim=0, keepdim=True)
        correlation_sig = (sig_squeeze.T-mean_sig.T) @ (sig_squeeze-mean_sig) / B

        with torch.no_grad():
            try:
                _, mu_eng_vector = torch.linalg.eigh(C*correlation_mu+self.eps*torch.eye(C, device=x.device))
                # mu_corr_matrix = mu_eng_vector @ torch.sqrt(torch.diag(torch.clip(mu_eng_value, min=1e-10))) @ (mu_eng_vector.T)
            except:
                mu_eng_vector = torch.eye(C, device=x.device)
            
            if not torch.all(torch.isfinite(mu_eng_vector)) or torch.any(torch.isnan(mu_eng_vector)):
                mu_eng_vector = torch.eye(C, device=x.device)

            try:
                _, sig_eng_vector = torch.linalg.eigh(C*correlation_sig+self.eps*torch.eye(C, device=x.device))
                # sig_corr_matrix = sig_eng_vector @ torch.sqrt(torch.diag(torch.clip(sig_eng_value, min=1e-10))) @ (sig_eng_vector.T)
            except:
                sig_eng_vector = torch.eye(C, device=x.device)

            if not torch.all(torch.isfinite(sig_eng_vector )) or torch.any(torch.isnan(sig_eng_vector)):
                sig_eng_vector = torch.eye(C, device=x.device)

        mu_corr_matrix = mu_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((mu_eng_vector.T)@ correlation_mu @ mu_eng_vector),min=1e-12))) @ (mu_eng_vector.T)
        sig_corr_matrix = sig_eng_vector @ torch.diag(torch.sqrt(torch.clip(torch.diag((sig_eng_vector.T)@ correlation_sig @ sig_eng_vector), min=1e-12))) @ (sig_eng_vector.T)

        gaussian_mu = (torch.randn(B, 1, C, device=x.device) @ mu_corr_matrix)
        gaussian_mu = torch.reshape(gaussian_mu, (B, C, 1, 1))

        gaussian_sig = (torch.randn(B, 1, C, device=x.device) @ sig_corr_matrix)
        gaussian_sig = torch.reshape(gaussian_sig, (B, C, 1, 1))

        mu_mix = mu + factor*gaussian_mu
        sig_mix = sig + factor*gaussian_sig

        return x_normed * sig_mix + mu_mix



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
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


class UResNet(Backbone):

    def __init__(
            self, block, layers, pertubration=None, uncertainty=0.0, **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pertubration0 = pertubration(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration1 = pertubration(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration2 = pertubration(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration3 = pertubration(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration4 = pertubration(p=uncertainty) if pertubration else nn.Identity()
        self.pertubration5 = pertubration(p=uncertainty) if pertubration else nn.Identity()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion


        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.pertubration0(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.pertubration1(x)
        x = self.layer1(x)
        x = self.pertubration2(x)
        x = self.layer2(x)
        x = self.pertubration3(x)
        x = self.layer3(x)
        x = self.pertubration4(x)
        x = self.layer4(x)
        x = self.pertubration5(x)

        return x

    def forward(self, x, label=None):
        if label == None:
            f = self.featuremaps(x)
        else:
            f = self.featuremaps(x, label)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)



class CUResNet(Backbone):

    def __init__(
            self, block, layers, pertubration_list:list=['layer1'], uncertainty=0.0, alpha=0.3, **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.pertubration_list = pertubration_list
        self.pertubration0 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer0' in pertubration_list else nn.Identity()
        self.pertubration1 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer1' in pertubration_list else nn.Identity()
        self.pertubration2 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer2' in pertubration_list else nn.Identity()
        self.pertubration3 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer3' in pertubration_list else nn.Identity()
        self.pertubration4 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer4' in pertubration_list else nn.Identity()
        self.pertubration5 = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha) if 'layer5' in pertubration_list else nn.Identity()
        '''
        self.total_list = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']
        self.pertubration_dict = nn.ModuleDict({})
        
        for layer in self.pertubration_list:
            self.pertubration_dict[layer] = CorrelatedDistributionUncertainty(p=uncertainty, alpha=alpha)
        '''
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion


        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.pertubration0(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.pertubration1(x)
        x = self.layer1(x)
        x = self.pertubration2(x)
        x = self.layer2(x)
        x = self.pertubration3(x)
        x = self.layer3(x)
        x = self.pertubration4(x)
        x = self.layer4(x)
        x = self.pertubration5(x)
        return x

    def forward(self, x, label=None):
        if label == None:
            f = self.featuremaps(x)
        else:
            f = self.featuremaps(x, label)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""
"""
Standard residual networks
"""


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


@BACKBONE_REGISTRY.register()
def uresnet18(pretrained=True, uncertainty=0.0, pos=[], is_corr:bool=True,**kwargs):
    if is_corr:
        permutation = CorrelatedDistributionUncertainty
        # permutation = DistributionUncertainty
    else:
        permutation = DistributionUncertainty

    model = UResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration=permutation, uncertainty=uncertainty, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_dsu(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    permutation = DistributionUncertainty

    model = UResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration=permutation, uncertainty=uncertainty, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_compare(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def uresnet18_a012345d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a012345d1(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.1, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a012345d2(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.2, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a012345d4(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.4, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a012345d5(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.5, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a012345d7(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.7, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a012345d9(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.9, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a01d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a12d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=[ 'layer1', 'layer2'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a23d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=[ 'layer2', 'layer3'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a34d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=[ 'layer3', 'layer4'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a45d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=[ 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a012d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a123d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=[ 'layer1', 'layer2', 'layer3'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a234d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=[ 'layer2', 'layer3', 'layer4'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a345d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=[ 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def uresnet18_a0123d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a1234d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer1', 'layer2', 'layer3', 'layer4'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a2345d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def uresnet18_a01234d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer0', 'layer1', 'layer2', 'layer3', 'layer4'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def uresnet18_a12345d3(pretrained=True, uncertainty=0.0, pos=[], **kwargs):
    model = CUResNet(block=BasicBlock, layers=[2, 2, 2, 2],
                    pertubration_list=['layer1', 'layer2', 'layer3', 'layer4', 'layer5'], uncertainty=uncertainty, alpha=0.3, pos=pos)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model
