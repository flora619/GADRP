import datetime
import math
import os

import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

# drug_sim
from sklearn.preprocessing import MinMaxScaler
drug_physicochemical_file = "../data/drug/269_dim_physicochemical.csv"
drug_sim_top10_file = "../data/drug/drug_sim_top10.pt"
drug_sim_file = "../data/drug/drug_sim.pt"
cell_sim_top10_file = "../data/cell_line/cell_sim_top10.pt"
cell_sim_file = "../data/cell_line/cell_sim.pt"

# drug_cell_pair_index
cell_index_file = "../data/cell_line/cell_index.csv"

edge_idx_file = "../data/pair/edge_idx_file.pt"
drug_cell_sim_file = "../data/pair/drug_cell_sim.pt"
drug_cell_sim_matrix_txt = "../data/pair/drug_cell_sim_matrix.txt"
drug_cell_sim_top100_index_file = "../data/pair/drug_cell_sim_top100_index.pt"

drug_cell_label_file = "../data/pair/drug_cell_response.csv"
drug_cell_label_index_file = "../data/pair/drug_cell_label.pt"


# 计算药物和细胞系对的相似性矩阵
def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    # Transform the adjacency matrix into a symmetric matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tocoo()
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, - 0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # ndarray类型
    # toarray returns an ndarray; todense returns a matrix. If you want a matrix, use todense otherwise, use toarray
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).tocoo()


def main():
    device = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #药物和细胞系的相似性矩阵
    drug_sim=torch.load(drug_sim_file).to(device)
    drug_num=drug_sim.shape[0]
    cell_sim=torch.load(cell_sim_file).to(device)
    cell_num=cell_sim.shape[0]

    physicochemical_feature = pd.read_csv(drug_physicochemical_file, sep=',', header=0, index_col=[0])
    drug_index = list(physicochemical_feature.index)
    cell_index=pd.read_csv(cell_index_file, sep=',', header=None, index_col=[0]).index
    drug_cell_pair_index=[]
    for i in range(drug_num):
        for j in range(cell_num):
            pair_temp=[]
            pair_temp.append(drug_index[i])
            pair_temp.append(cell_index[j])
            pair_temp.append(i)
            pair_temp.append(j)
            drug_cell_pair_index.append(pair_temp)


    #calculate drug cell line pair similarity matrix
    drug_sim_top10=torch.load(drug_sim_top10_file).to(device)
    cell_sim_top10=torch.load(cell_sim_top10_file).to(device)
    list_drug = (torch.arange(0, drug_num).reshape(-1, 1) * torch.ones(size=(1, cell_num))).reshape(-1).long().to(device)
    list_cell = (torch.arange(0, cell_num) * torch.ones(size=(drug_num, 1))).reshape(-1).long().to(device)

    list_drug100 = (torch.arange(0, 10).reshape(-1, 1) * torch.ones(size=(1, 10))).reshape(-1).long()
    list_cell100 = (torch.arange(0, 10) * torch.ones(size=(10, 1))).reshape(-1).long()
    drug_sim_top10_indexlist = drug_sim_top10[:, list_drug100]
    cell_sim_top10_indexlist = cell_sim_top10[:, list_cell100]
    drug_sim_top10_indexlist = drug_sim_top10_indexlist[list_drug].to(device)
    cell_sim_top10_indexlist = cell_sim_top10_indexlist[list_cell].to(device)
    drug_cell_sim_index = drug_sim_top10_indexlist*cell_num + cell_sim_top10_indexlist

    drug_j = (drug_cell_sim_index // 388).long()
    cell_j = (drug_cell_sim_index % 388).long()
    drug_i = torch.zeros_like(drug_j)
    cell_i = torch.zeros_like(cell_j)
    drug_i[:, :] = list_drug.reshape(-1, 1).long()
    cell_i[:, :] = list_cell.reshape(-1, 1).long()
    drug_cell_sim = torch.zeros(size=(len(drug_cell_pair_index), 100)).to(device)
    for i in range(100):
        drug_cell_sim[:,i]=drug_sim[drug_i[:,i],drug_j[:,i]]+cell_sim[cell_i[:,i],cell_j[:,i]]
    drug_cell_sim/=2
    #sort and select top10 DCPs for each DCP
    drug_cell_sim, index = torch.sort(drug_cell_sim, descending=True, dim=1)
    pair_num = drug_cell_sim_index.shape[0]
    for i in range(pair_num):
        drug_cell_sim_index[i] = drug_cell_sim_index[i][index[i]]
    neighbor_num = 10
    index = range(neighbor_num)
    drug_cell_sim = drug_cell_sim[:, index]
    drug_cell_sim_index = drug_cell_sim_index[:, index]

    #Sparse matrix
    row = (torch.arange(0, pair_num).reshape(-1, 1) * torch.ones(size=(1, neighbor_num))).reshape(-1).long()
    col = drug_cell_sim_index.reshape(-1).cpu()
    values =drug_cell_sim.reshape(-1).cpu()
    edge_idx = coo_matrix((values, (row, col)), shape=(pair_num, pair_num))
    edge_idx = sym_adj(edge_idx)
    values = edge_idx.data
    index = torch.vstack((torch.LongTensor(edge_idx.row), torch.LongTensor(edge_idx.col)))
    edge_idx = torch.sparse_coo_tensor(index, values, size=(pair_num, pair_num))
    torch.save(edge_idx, edge_idx_file)



    # ic50
    drug_index_dict = dict(zip(drug_index, range(len(drug_index))))
    cell_index_dict = dict(zip(cell_index, range(len(cell_index))))

    drug_cell_label = pd.read_csv(drug_cell_label_file, sep=',', header=0, index_col=None,
                                  usecols=["ccle_name", "pubchem_cid", "ic50"])
    # dropna 309933*31
    drug_cell_label = drug_cell_label.dropna(axis=0)
    print("after dropna:", drug_cell_label.shape)

    # filter  249810*3
    drug_cell_label = drug_cell_label[drug_cell_label.ccle_name.isin(list(cell_index))]
    print("cell_index select", drug_cell_label.shape)
    #          249801*3
    drug_cell_label = drug_cell_label.loc[drug_cell_label["ic50"] > 0]
    print("after ic50 > 0", drug_cell_label.shape)

    temp = drug_cell_label.values[:, 2]
    temp = np.log10(temp.astype(float))
    drug_cell_label["ic50"] = temp


    drug_cell_label = drug_cell_label.sort_values(by=["ic50"])

    # box-plot  249810*3
    Q1_index = math.ceil(len(drug_cell_label) / 4)
    Q2_index = math.ceil(len(drug_cell_label) / 4 * 2)
    Q3_index = math.ceil(len(drug_cell_label) / 4 * 3)
    Q1 = drug_cell_label.values[Q1_index][2]
    Q2 = drug_cell_label.values[Q2_index][2]
    Q3 = drug_cell_label.values[Q3_index][2]
    print("Q1,Q2,Q3", Q1, Q2, Q3)
    IQR = Q3 - Q1
    LOW = Q1 - 1.5 * IQR
    HIGH = Q3 + 1.5 * IQR
    print("IQR, LOW, HIGH", IQR, LOW, HIGH)
    drug_cell_label = drug_cell_label.loc[drug_cell_label["ic50"] >= LOW]
    drug_cell_label = drug_cell_label.loc[drug_cell_label["ic50"] <= HIGH].values
    print("after box-polt", len(drug_cell_label))

    min_max = MinMaxScaler()
    temp = drug_cell_label[:, 2].reshape(-1, 1)
    temp = min_max.fit_transform(temp)
    drug_cell_label[:, 2] = temp.reshape(-1)

    drug_cell_label = drug_cell_label.tolist()
    for i in range(len(drug_cell_label)):
        drug_cell_label[i][0] = cell_index_dict[drug_cell_label[i][0]]
        drug_cell_label[i][1] = drug_index_dict[drug_cell_label[i][1]]

    drug_cell_label = torch.tensor(drug_cell_label).to(device)
    torch.save(drug_cell_label, drug_cell_label_index_file)


if __name__ == '__main__':
    main()
