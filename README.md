# GADRP
Source code and data for "GADRP: graph convolutional networks and autoencoders for cancer drug response prediction"
## Data
* 269-dim-physicochemical.csv - Physicochemical properties of drugs
* 881-dim-fingerprint.csv - Molecular fingerprint of drugs
* miRNA_470cell_734dim.csv - MicroRNA expression data of cell lines
* CpG_407cell_69641dim.csv - DNA methylation data of cell lines
* RNAseq_462cell_48392dim.csv - Gene expression data of cell lines
* copynumber_461cell_23316dim.csv - DNA copy number data of cell lines
* drug_cell_response.csv - response data between drugs and cell lines
* cell_name.csv - Names of 388 cell lines with four cell line characteristics
## Source codes
* drug.py: generate drug similarity matrix according to physicochemical properties of drugs
* cell.py: generate cell line similarity matrix according to microRNA expression and DNA methylation of cell lines
* drug_cell.py: generate drug cell line pairs similarity matrix 
* cell_ae.py: learn low_dimensional representations from high-dimensional cell line features
* train.py: train the model and make predictions
* GADRP.py: details of GADRP model
## Requirements
* Python == 3.7.10
* PyTorch == 1.9.0
* sklearn == 0.24.2
* Numpy == 1.19.2
* Pandas == 1.3.4
## Operation steps
1. Install dependencies, including torch1.9, sklearn, numpy and pandas
2. run drug.py and cell.py to generate drug and cell line similarity matrices
3. run drug_cell.py to generate drug cell line pair similarity matrix
4. run cell_ae.py to generate low-dimensional representations of cell line omics characteristics
5. run python train.py for training and prediction
## Installation
git clone https://github.com/flora619/GADRP
