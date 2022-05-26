from model.IRGCN import GraphConvolution
import torch
from torch import nn


class Drug_cell_encoder(torch.nn.Module):
    def __init__(self,indim,device):
        self.device=device
        super(Drug_cell_encoder, self).__init__()
        self.gcn=GraphConvolution(indim,indim,device=device)
        self.relu = nn.ReLU(inplace=True)
        self.dropout=nn.Dropout(0.2)
    def forward(self,drug_cell_pair_feature,edge_idx):
        output1=self.relu(self.gcn(drug_cell_pair_feature,drug_cell_pair_feature,edge_idx,0.1))
        output1=self.dropout(output1)
        output2=self.relu(self.gcn(drug_cell_pair_feature,output1,edge_idx,0.1))
        output3 = self.relu(self.gcn(drug_cell_pair_feature, output2,edge_idx, 0.1))
        output3 = self.dropout(output3)
        output4 = self.relu(self.gcn(drug_cell_pair_feature, output3,edge_idx, 0.1))
        output5 = self.relu(self.gcn(drug_cell_pair_feature, output4,edge_idx, 0.1))
        output5 = self.dropout(output5)


        return output1,output2,output3,output4,output5
