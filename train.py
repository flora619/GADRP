import datetime
import os
import random

import torch
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from model.GADRP import GADRP_Net
import torch.utils.data as Data
import torch.optim as optim



# drug
drug_fingerprint_file= "./data/drug/881_dim_fingerprint.csv"
# cell
cell_index_file = "./data/cell_line/cell_index.csv"

cell_RNAseq_ae="./data/cell_line/cell_RNAseq400_ae.pt"
cell_copynumber_ae = "./data/cell_line/cell_copynumber400_ae.pt"

# drug_cell pair
edge_idx_file="./data/pair/edge_idx_file.pt"

# label
drug_cell_label_index_file = "./data/pair/drug_cell_label.pt"

lr = 0.0001
num_epoch =130
batch_size =1024

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
def data_process():
    # load molecular substructure fingerprints of drugs
    fingerprint = pd.read_csv(drug_fingerprint_file, sep=',', header=0, index_col=[0])
    fingerprint = torch.from_numpy(fingerprint.values).float().to(device)

    # load gene expression data and DNA copy number data of cell lines
    cell_index = pd.read_csv(cell_index_file, sep=',', header=None, index_col=[0])
    RNAseq_feature=torch.load(cell_RNAseq_ae).to(device)
    copynumber_feature = torch.load(cell_copynumber_ae).to(device)

    # load drug-cell line similarity matrix
    edge_idx=torch.load(edge_idx_file).to(device)
    drug_cell_label = torch.load(drug_cell_label_index_file).to(device)
    return fingerprint, \
           list(cell_index.index), RNAseq_feature,copynumber_feature, \
             edge_idx, drug_cell_label


def split(drug_cell_label, cell_num):
    """
    :param drug_cell_label:
    :param cell_index:
    :return:
    """
    drug_cell_label = drug_cell_label
    train_index1=torch.tensor([],dtype=torch.long).to(device)
    train_index2 = torch.tensor([], dtype=torch.long).to(device)
    train_index3 = torch.tensor([], dtype=torch.long).to(device)
    train_index4 = torch.tensor([], dtype=torch.long).to(device)
    train_index5 = torch.tensor([], dtype=torch.long).to(device)

    text_index1 = torch.tensor([], dtype=torch.long).to(device)
    text_index2 = torch.tensor([], dtype=torch.long).to(device)
    text_index3 = torch.tensor([], dtype=torch.long).to(device)
    text_index4 = torch.tensor([], dtype=torch.long).to(device)
    text_index5 = torch.tensor([], dtype=torch.long).to(device)
    for cell in range(cell_num):
        drug_subcell_label_index = torch.where(drug_cell_label[:,0]==cell)[0]
        drug_subcell_label_index=drug_subcell_label_index[torch.randperm(drug_subcell_label_index.size(0))]
        i1,i2,i3,i4,i5=drug_subcell_label_index.chunk(5,0)
        train_index1=torch.cat((train_index1,i2,i3,i4,i5),0)
        train_index2 = torch.cat((train_index2, i1, i3, i4, i5), 0)
        train_index3 = torch.cat((train_index3, i2, i1, i4, i5), 0)
        train_index4 = torch.cat((train_index4, i2, i3, i1, i5), 0)
        train_index5 = torch.cat((train_index5, i2, i3, i4, i1), 0)

        text_index1 = torch.cat((text_index1, i1), 0)
        text_index2 = torch.cat((text_index2, i2), 0)
        text_index3 = torch.cat((text_index3, i3), 0)
        text_index4 = torch.cat((text_index4, i4), 0)
        text_index5 = torch.cat((text_index5, i5), 0)

    train_index=[train_index1,train_index2,train_index3,train_index4,train_index5]
    test_index=[text_index1,text_index2,text_index3,text_index4,text_index5]
    return train_index,test_index


def training(model, drug_feature
             , cell_feature1, cell_feature2,
             edge_idx, data_iter, features_test, labels_test):
    # loss function
    loss = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    train_ls, test_ls, test_pearson = [], [], []
    start = datetime.datetime.now()
    print(optimizer)
    best_test_loss = torch.tensor(100)

    for epoch in range(1, num_epoch + 1):
        model.train()
        batch = 0
        for X, y in data_iter:
            y_pre= model(drug_feature, cell_feature1, cell_feature2,
                         edge_idx, X)
            l = loss(y_pre, y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_ls.append(l)
            batch += 1
        model.eval()
        with torch.no_grad():

            end = datetime.datetime.now()
            y_pre= model(drug_feature, cell_feature1, cell_feature2,
                         edge_idx, features_test)
            l_test = loss(y_pre, labels_test.view(-1, 1)).sum()
            temp_y_pre = y_pre
            p = pearsonr(temp_y_pre.view(-1).tolist(), labels_test.view(-1).tolist())[0]
            s = spearmanr(temp_y_pre.view(-1).tolist(), labels_test.view(-1).tolist())[0]
            r2 = r2_score(temp_y_pre.view(-1).tolist(), labels_test.view(-1).tolist(),multioutput="raw_values")
            print("e:", epoch, "tr_loss:", l.item(), "te_loss:", l_test.item(),
                  " te_pearson:", p," test_spearson:",s," test_r2:",r2,"t:", (end - start).seconds)
        if l_test.item() < best_test_loss.item():
            best_test_loss = torch.tensor(l_test.item())


def main():
    random.seed(2)
    fingerprint, \
    cell_index,RNAseq_feature, copynumber_feature, \
    edge_idx, drug_cell_label = data_process()



    # Stratified sampling
    train_index_all, test_index_all=split(drug_cell_label,cell_num=388)

    for i in range(5):
        train_index=train_index_all[i]
        test_index=test_index_all[i]
        print("train set", len(train_index))
        print("test set", len(test_index))

        features = drug_cell_label[train_index, :2]
        labels = drug_cell_label[train_index, 2]
        features_test = drug_cell_label[test_index, :2]
        labels_test = drug_cell_label[test_index, 2]
        dataset = Data.TensorDataset(features, labels)

        data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
        model = GADRP_Net(device).to(device)
        training(model, fingerprint
                 , copynumber_feature,RNAseq_feature,
                 edge_idx, data_iter, features_test, labels_test)



if __name__ == '__main__':
    main()
