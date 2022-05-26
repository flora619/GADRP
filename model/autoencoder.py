import os
import torch.nn as nn
class Auto_Encoder(nn.Module):
    def __init__(self, device,indim, outdim=400):
        super(Auto_Encoder, self).__init__()
        self.encoder = Encoder(device=device,indim=indim,outdim=outdim)
        self.decoder = Decoder(device=device,outdim=indim,indim= outdim)
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    def output(self, x):
        return self.encoder(x)


class Encoder(nn.Module):
    def __init__(self,device, indim, outdim=400):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(indim, 2048,device=device)
        self.linear2 = nn.Linear(2048, 1024,device=device)
        self.linear3 = nn.Linear(1024, outdim,device=device)
    def forward(self, x):
        x = nn.SELU()(self.linear1(x))
        x = nn.SELU()(self.linear2(x))
        x = nn.Sigmoid()(self.linear3(x))
        return x


class Decoder(nn.Module):
    def __init__(self,device,outdim, indim=400):
        super(Decoder, self).__init__()
        self.linear3 = nn.Linear(indim, 1024,device=device)
        self.linear2 = nn.Linear(1024,2048,device=device)
        self.linear1 = nn.Linear(2048,outdim,device=device)
    def forward(self, x):

        x = nn.SELU()(self.linear3(x))
        x = nn.SELU()(self.linear2(x))
        x = nn.Sigmoid()(self.linear1(x))
        return x
