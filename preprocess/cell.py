
import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import torch
import heapq



cell_miRNA_file = '../data/cell_line/miRNA_470cell_734dim.csv'
cell_CpG_file = '../data/cell_line/CpG_407cell_69641dim.csv'
cell_id_file="../data/cell_line/cell_index.csv"
cell_sim_file="../data/cell_line/cell_sim.pt"
cell_sim_top10_file="../data/cell_line/cell_sim_top10.pt"



def main():
    """
    caculate cell line similarity matrix
    :return:
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #load microRNA expression data, DNA methylation data of cell line
    miRNA_feature = pd.read_csv(cell_miRNA_file, sep=',', header=None, index_col=[0])
    CpG_feature = pd.read_csv(cell_CpG_file, sep=',', header=None, index_col=[0], skiprows=2)

    cell_id_set = pd.read_csv(cell_id_file, sep=',', header=None, index_col=[0])

    miRNA_feature = miRNA_feature.loc[list(cell_id_set.index)].values
    CpG_feature = CpG_feature.loc[list(cell_id_set.index)].values


    # Normalization
    min_max=MinMaxScaler()
    miRNA_feature = min_max.fit_transform(miRNA_feature)
    CpG_feature = min_max.fit_transform(CpG_feature)


    #calculate miRNA_sim  CpG_sim
    miRNA_sim = torch.zeros(size=(len(miRNA_feature), len(miRNA_feature)))
    CpG_sim = torch.zeros(size=(len(CpG_feature), len(CpG_feature)))


    for i in range(len(miRNA_feature)):
        print(i)
        for j in range(len(miRNA_feature)):
            temp_sim = pearsonr(miRNA_feature[i, :], miRNA_feature[j, :])
            miRNA_sim[i][j] = np.abs(temp_sim[0])

            temp_sim = pearsonr(CpG_feature[i, :], CpG_feature[j, :])
            CpG_sim[i][j] = np.abs(temp_sim[0])

    #calculate cell line similarity matrix
    cell_sim=(miRNA_sim + CpG_sim)/2


    cell_sim_top10 = torch.zeros(size=(388, 10), dtype=torch.int).to(device)
    for i in range(388):
        celli_list = list(cell_sim[i])
        cell_sim_top10[i] = torch.tensor(list(map(celli_list.index, heapq.nlargest(10, celli_list))),
                                         dtype=torch.int)

    torch.save(cell_sim, cell_sim_file)
    torch.save(cell_sim_top10, cell_sim_top10_file)





if __name__ == '__main__':
    main()