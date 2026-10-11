"""ResNet18 without pretrained weights, with a CIFAR 32x32 stem."""
from torch import nn
from torchvision.models import resnet18


def resnet18_cifar() -> nn.Module:
    model = resnet18(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
