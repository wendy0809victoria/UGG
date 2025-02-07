# UGG-Unbiased-Graph-Generator

This is the code for the paper "Mitigating Topology Bias in Graph Diffusion via Counterfactual Intervention". 

## Requirements

Code is tested in Python 3.8 and PyTorch 2.0.1. Some major requirements are listed below:
```
cuda~=11.8
numpy~=1.22.2
```

## Datasets

We include four datasets: **NBA**, **German**, **Pokec-n** and **Pokec-z**. Raw data are uploaded to ```./raw_data``` folder. We have constructed subgraphs from the raw data and subgraph data files are stored in ```./data``` folder. You can also reconstruct subgraph data by running ```python ./raw_data/subgraph.py --dataset_name```, and the results will be automatically stored in ```./data/dataset_name``` folder. 

## Run the Code
### Train FairGDiff (Default: NBA dataset)
```
python main.py
```

### Generate Synthetic Graph (Default: NBA dataset)
```
python nba-gen-whole.py
```

### Evaluate Synthetic Graphs via Node Classification (Default: NBA dataset)
Under /src directory
```
python train_fairGNN.py \
        --seed=42 \
        --epochs=2000 \
        --model=GCN \
        --dataset=nba \
        --num-hidden=128 \
        --acc=0.65 \
        --roc=0.60 \
        --alpha=10 \
        --beta=1
```

## Citation
Please check our paper for technical details and full results. If you find this repo useful for your research, please consider citing the paper. 
