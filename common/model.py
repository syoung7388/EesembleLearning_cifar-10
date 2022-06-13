from cgitb import reset
from xml.parsers.expat import model
import torch.nn as nn
import torchvision
import torch

class ResNet(nn.Module):

    def __init__(self, model_name):
        super(ResNet, self).__init__()
        if model_name == 'resnet34':
            self.model = torchvision.models.resnet34(pretrained=True)
        elif model_name == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            self.model = torchvision.models.resnet101(pretrained = True)
        elif model_name == 'resnet152':
            self.model = torchvision.models.resnet152(pretrained = True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        torch.nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, inputs):
        return self.model(inputs)



class Efficientnet(nn.Module):

    def __init__(self, model_name):
        super(Efficientnet, self).__init__()
        if model_name == 'efficientnet_b1':
            self.model = torchvision.models.efficientnet_b1(pretrained=True)
        elif model_name == 'efficientnet_b5':
            self.model = torchvision.models.efficientnet_b5(pretrained=True)
        elif model_name == 'efficientnet_b3':
            self.model = torchvision.models.efficientnet_b3(pretrained = True)
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, 10)
        torch.nn.init.xavier_uniform_(self.model.classifier[1].weight)
    def forward(self, inputs):
        return self.model(inputs)



