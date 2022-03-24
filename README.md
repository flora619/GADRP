# GADRP
Source code and data for "GADRP: Graph Neural Network and Autoencoder for Drug Response Prediction"
## Data
* 269-dim-physicochemical.csv - Physicochemical properties of drugs
* 881-dim-fingerprint.csv - Molecular fingerprint of drugs
* miRNA_470cell_734dim.csv - MicroRNA expression data of cell lines
* CpG_407cell_69641dim.csv - DNA methylation data of cell lines
* RNAseq_462cell_48392dim.csv - Gene expression data of cell lines
* copynumber_461cell_23316dim.csv - DNA copy number data of cell lines
* drug_cell_response.csv - response data between drugs and cell lines
* cell_name.csv - Names of 388 cell lines with four cellline characteristics
## Requirements
* Python == 3.7.10
* PyTorch == 1.9.0
* sklearn == 0.24.2
* Numpy == 1.19.2
* Pandas == 1.3.4
## Installation
git clone https://github.com/flora619/GADRP
## Operation steps
1. python cell_ae.py
2. python train.py
