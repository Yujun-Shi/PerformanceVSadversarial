import torch
import torch.nn as nn
import math

__all__=['resnet20x1_cifar', 'resnet20x5_cifar', 'resnet20x10_cifar', 'resnet32x1_cifar', 'resnet32x5_cifar', 'resnet32x10_cifar']

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion=1

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

class ResNet_Cifar(nn.Module):

    def __init__(self, block, wfactor, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16*wfactor, layers[0])
        self.layer2 = self._make_layer(block, 32*wfactor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64*wfactor, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64*wfactor*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet20x1_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, wfactor=1, layers=[3, 3, 3], **kwargs)
    return model

def resnet20x5_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, wfactor=5, layers=[3, 3, 3], **kwargs)
    return model

def resnet20x10_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, wfactor=10, layers=[3, 3, 3], **kwargs)
    return model

def resnet32x1_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, wfactor=1, layers=[5, 5, 5], **kwargs)
    return model

def resnet32x5_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, wfactor=5, layers=[5, 5, 5], **kwargs)
    return model

def resnet32x10_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, wfactor=10, layers=[5, 5, 5], **kwargs)
    return model
