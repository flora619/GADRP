import datetime
import gc
import os

import torch
import torch.nn as nn
from torch.nn import Parameter

from model.drug_cell_encoder import Drug_cell_encoder
from torch_geometric.nn import GCNConv

class GADRP_Net(torch.nn.Module):
    def __init__(self, device):
        super(GADRP_Net, self).__init__()
        self.device=device
        self.drugfc1 = nn.Linear(881, 200)
        self.cellfc1 = nn.Linear(800, 200)
        self.embedding = Drug_cell_encoder(400,device=device)
        self.att = Parameter(torch.tensor([1/1,1/1,1/1,1/1,1/1]) )         # in_features × out_features

        self.fc1 = nn.Linear(400,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, drug_feature, cell_feature1, cell_feature2, edge_idx, drug_cell_label):
        # drug_feature
        drug_feature=self.relu(self.drugfc1(drug_feature))
        drug_feature = self.dropout(drug_feature)
        # cell_feature
        cell_feature = torch.cat((cell_feature1, cell_feature2), dim=1)
        cell_feature = self.relu(self.cellfc1(cell_feature))
        cell_feature = self.dropout(cell_feature)
        # concatenate drug+cell
        drug_num = drug_feature.shape[0]
        cell_num = cell_feature.shape[0]
        list_drug = (torch.arange(0, drug_num).reshape(-1, 1) * torch.ones(size=(1, cell_num))).reshape(-1).long()
        list_cell = (torch.arange(0, cell_num) * torch.ones(size=(drug_num, 1))).reshape(-1).long()
        drug_cell_pair_feature = torch.cat((drug_feature[list_drug], cell_feature[list_cell]), 1)

        # pair_feature
        drug_cell_pair_feature1,drug_cell_pair_feature2,\
        drug_cell_pair_feature3,drug_cell_pair_feature4,drug_cell_pair_feature5=self.embedding(drug_cell_pair_feature,edge_idx)
        cell_num = cell_feature.shape[0]
        feature1 = drug_cell_pair_feature1[(drug_cell_label[:, 1] * cell_num + drug_cell_label[:, 0]).long()]
        feature2 = drug_cell_pair_feature2[(drug_cell_label[:, 1] * cell_num + drug_cell_label[:, 0]).long()]
        feature3 = drug_cell_pair_feature3[(drug_cell_label[:, 1] * cell_num + drug_cell_label[:, 0]).long()]
        feature4 = drug_cell_pair_feature4[(drug_cell_label[:, 1] * cell_num + drug_cell_label[:, 0]).long()]
        feature5 = drug_cell_pair_feature5[(drug_cell_label[:, 1] * cell_num + drug_cell_label[:, 0]).long()]
        #
        #
        feature=self.att[0]*feature1
        feature=feature+self.att[1]*feature2
        feature=feature+self.att[2]*feature3
        feature=feature+self.att[3]*feature4
        feature=feature+self.att[4]*feature5
        feature=self.dropout(feature)
        output = self.fc1(feature)
        output = self.relu(output)
        output=self.dropout(output)

        output = self.fc2(output)
        output = self.relu(output)
        output=self.dropout(output)

        output = self.fc3(output)
        output = self.sigmoid(output)
        return output
