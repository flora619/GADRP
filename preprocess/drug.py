import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

drug_physicochemical_file = "../data/drug/269_dim_physicochemical.csv"
drug_sim_file= "../data/drug/drug_sim.pt"
drug_sim_top10_file="../data/drug/drug_sim_top10.pt"


def main():


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load  physicochemical_feature of drugs
    physicochemical_feature = pd.read_csv(drug_physicochemical_file, sep=',', header=0, index_col=[0])
    drugset = list(physicochemical_feature.index)
    # normalization
    min_max = MinMaxScaler()
    physicochemical_feature = min_max.fit_transform(physicochemical_feature)


    drugset=np.array(drugset)
    drug_num=len(drugset)


    drug_physicochemical_sim = torch.zeros(size=(drug_num, drug_num)).to(device)

    for i in range(len(drugset)):
        print(i)
        for j in range(len(drugset)):
            temp_sim = pearsonr(physicochemical_feature[i, :], physicochemical_feature[j, :])
            drug_physicochemical_sim[i][j] = np.abs(temp_sim[0])


    #drug_sim_top10: The subscript corresponding to the most similar drug
    drug_sim=drug_physicochemical_sim
    _,drug_sim_top10 = torch.topk(drug_sim, 10,dim=1)
    torch.save(drug_sim, drug_sim_file)
    torch.save(drug_sim_top10,drug_sim_top10_file)



if __name__ == '__main__':
    main()