# UGG-Unbiased-Graph-Generator

This is the code for the paper "Unbiased Graph Generator: Treatment-Intervened Graph Generation with Latent Diffusion Models". 

## Requirements

Code is tested in Python 3.8 and PyTorch 2.0.1. Some major requirements are listed below:
```
cuda~=11.8
numpy~=1.22.2
```

## Datasets

We include four datasets: **NBA**, **German**, **Pokec-n** and **Pokec-z**. Raw data are uploaded to ```./raw_data``` folder. We have constructed subgraphs from the raw data and subgraph data files are stored in ```./data``` folder. You can also reconstruct subgraph data by running ```python ./raw_data/subgraph.py --dataset_name```, and the results will be automatically stored in ```./data/dataset_name``` folder. 

## Run the Code

## Citation
Please check our paper for technical details and full results. If you find this repo useful for your research, please consider citing the paper. 
