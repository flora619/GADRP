import datetime
import os
import random

import torch
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score
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
drug_cell_label_index_file = "/home/wh/python_file/gnnbased/data/pair/drug_cell_label.pt"


lr = 0.0001
num_epoch =100
batch_size =512

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:2" if (torch.cuda.is_available()) else "cpu")

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
    # 对细胞系进行筛选
    return fingerprint, \
           list(cell_index.index), RNAseq_feature,copynumber_feature, \
           edge_idx, drug_cell_label


def split(drug_cell_label, cell_num, ratio=0.9):
    """
    :param drug_cell_label:
    :param cell_index:
    :param ratio:
    :return:
    """
    drug_cell_label = drug_cell_label.tolist()
    drug_cell_label_train = []
    drug_cell_label_test = []
    for cell in range(cell_num):
        drug_subcell_label = [item for item in drug_cell_label if item[0] == cell]
        train_list = random.sample(drug_subcell_label, int(ratio * len(drug_subcell_label)))
        test_list = [item for item in drug_subcell_label if item not in train_list]
        drug_cell_label_train += train_list
        drug_cell_label_test += test_list
    return torch.tensor(drug_cell_label_train).to(device), torch.tensor(drug_cell_label_test).to(device)



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
    random.seed(3)
    fingerprint, \
    cell_index,RNAseq_feature, copynumber_feature, \
    edge_idx, drug_cell_label = data_process()



    # Stratified sampling
    drug_cell_label_train, drug_cell_label_test = split(drug_cell_label, len(cell_index))
    print("train set", len(drug_cell_label_train))  # 221092
    print("test set", len(drug_cell_label_test))  # 24749


    features = drug_cell_label_train[:, :2]
    labels = drug_cell_label_train[:, 2]
    features_test = drug_cell_label_test[:, :2]
    labels_test = drug_cell_label_test[:, 2]
    dataset = Data.TensorDataset(features, labels)


    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    model = GADRP_Net(device).to(device)
    training(model, fingerprint
             , copynumber_feature,RNAseq_feature,
             edge_idx, data_iter, features_test, labels_test)


if __name__ == '__main__':
    main()
