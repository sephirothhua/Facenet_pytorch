import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.nn.modules.distance import PairwiseDistance

class FaceNetModel(nn.Module):
    def __init__(self, embedding_size, pretrained=False):
        super(FaceNetModel, self).__init__()

        self.model = resnet18(pretrained)
        self.embedding_size = embedding_size
        self.model.fc = nn.Linear(2048, self.embedding_size)

    def l2_norm(self, input):
        # input_size = input.size()
        buffer     = torch.pow(input, 2)
        normp      = torch.sum(buffer, 1,keepdim=True)
        norm       = torch.sqrt(normp)
        # norm       = norm.expand_as(input)
        _output    = torch.div(input, norm)
        output     = _output.view(-1, 128)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x).view(-1, 2048)
        # print(x.size)
        x = self.model.fc(x)

        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        self.features = self.features * alpha

        return self.features





if __name__ == '__main__':
    model = FaceNetModel(128,22)
