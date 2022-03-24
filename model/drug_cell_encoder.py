from model.IRGCN import GraphConvolution
import torch
from torch import nn


class Drug_cell_encoder(torch.nn.Module):
    def __init__(self,indim,device):
        self.device=device
        super(Drug_cell_encoder, self).__init__()
        self.gcn=GraphConvolution(indim,indim,device=device)
        self.relu = nn.ReLU(inplace=True)
        self.dropout=nn.Dropout(0.1)
    def forward(self,drug_cell_pair_feature,edge_idx):
        output=self.relu(self.gcn(drug_cell_pair_feature,drug_cell_pair_feature,edge_idx,0.1))
        output=self.relu(self.gcn(drug_cell_pair_feature,output,edge_idx,0.1))
        output = self.relu(self.gcn(drug_cell_pair_feature, output,edge_idx, 0.1))
        output = self.relu(self.gcn(drug_cell_pair_feature, output,edge_idx, 0.1))
        output = self.relu(self.gcn(drug_cell_pair_feature, output,edge_idx, 0.1))

        return output
