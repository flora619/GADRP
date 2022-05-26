import csv
import os
import random

from torch import nn as nn
from model.autoencoder import Auto_Encoder
import torch
from sklearn.preprocessing import scale, MinMaxScaler
import pandas as pd
import torch.utils.data as Data
import datetime
# cell
cell_index_file = "./data/cell_line/cell_index.csv"
cell_RNAseq_file="./data/cell_line/RNAseq_462cell_48392dim.csv"
cell_copynumber_file = "./data/cell_line/copynumber_461cell_23316dim.csv"

cell_RNAseq_ae="./data/cell_line/cell_RNAseq400_ae.pt"
cell_copynumber_ae="./data/cell_line/cell_copynumber400_ae.pt"
device="cuda"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def train_ae(model,trainLoader,test_feature):
    start = datetime.datetime.now()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    best_model=model
    best_loss=100
    for epoch in range(1, 2500 + 1 ):
        for x in trainLoader:
            y=x
            encoded, decoded = model(x)
            train_loss = loss_func(decoded, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        with torch.no_grad():
            y = test_feature
            encoded, decoded = model(test_feature)
            test_loss = loss_func(decoded, y)
        if (test_loss.item() < best_loss):
            best_loss = test_loss
            best_model = model
        if epoch%100==0:
            end = datetime.datetime.now()
            print('epoch:' ,epoch, 'train loss = ' ,train_loss.item(),"test loss:",test_loss.item(), "time:",(end - start).seconds)
    return best_model

lr=0.0001
batch_size=388
def main():
    random.seed(4)
    # load  gene expression data, and DNA copy number data of cell line
    RNAseq_feature = pd.read_csv(cell_RNAseq_file, sep=',', header=None, index_col=[0], skiprows=2)
    copynumber_feature = pd.read_csv(cell_copynumber_file, sep=',', header=None, index_col=[0], skiprows=5)

    cell_index = pd.read_csv(cell_index_file, sep=',', header=None, index_col=[0])

    RNAseq_feature = RNAseq_feature.loc[list(cell_index.index)].values
    copynumber_feature = copynumber_feature.loc[list(cell_index.index)].values

    #normalization
    min_max = MinMaxScaler()
    RNAseq_feature = torch.tensor(min_max.fit_transform(RNAseq_feature)).float().to(device)
    copynumber_feature = torch.tensor(min_max.fit_transform(copynumber_feature)).float().to(device)


    RNAseq_indim = RNAseq_feature.shape[-1]
    copynumber_indim = copynumber_feature.shape[-1]
    print(RNAseq_indim)
    print(copynumber_indim)

    # dimension reduction(gene expression data)
    RNAseq_ae = Auto_Encoder(device,RNAseq_indim, 400)
    train_list = random.sample((RNAseq_feature).tolist(), int(0.9 * len(RNAseq_feature)))
    test_list = [item for item in (RNAseq_feature).tolist() if item not in train_list]
    train=torch.tensor(train_list).float().to(device)
    test = torch.tensor(test_list).float().to(device)
    data_iter = Data.DataLoader(train, batch_size, shuffle=True)
    best_model=train_ae(RNAseq_ae,data_iter,test)
    torch.save(best_model.output(RNAseq_feature),cell_RNAseq_ae)

    # dimension reduction(DNA copy number data)
    copynumber_ae = Auto_Encoder(device,copynumber_indim, 400)
    train_list = random.sample((copynumber_feature).tolist(), int(0.9 * len(copynumber_feature)))
    test_list = [item for item in (copynumber_feature).tolist() if item not in train_list]
    train = torch.tensor(train_list).float().to(device)
    test = torch.tensor(test_list).float().to(device)
    data_iter = Data.DataLoader(train, batch_size, shuffle=True)
    best_model = train_ae(copynumber_ae, data_iter, test)
    torch.save(best_model.output(copynumber_feature), cell_copynumber_ae)





if __name__ == '__main__':
    main()
