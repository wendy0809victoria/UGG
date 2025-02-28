import argparse
import os
import random
import scipy as sp
import pickle as pk
import torch.nn.functional as F
import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import AutoEncoder, VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample, p_sample_loop
from utils import create_dataset, CustomDataset, linear_beta_schedule, read_stats, eval_autoencoder, construct_nx_from_adj, store_stats, gen_stats, calculate_mean_std, evaluation_metrics, z_score_norm

from torch.utils.data import Subset
np.random.seed(13)

# TODO: check/count number of all parameters

def calc_disc(disc_func, z, nodepairs_f, nodepairs_cf):
    X_f = torch.cat((z[nodepairs_f.T[0]], z[nodepairs_f.T[1]]), axis=1)
    X_cf = torch.cat((z[nodepairs_cf.T[0]], z[nodepairs_cf.T[1]]), axis=1)
    if disc_func == 'lin':
        mean_f = X_f.mean(0)
        mean_cf = X_cf.mean(0)
        loss_disc = torch.sqrt(F.mse_loss(mean_f, mean_cf) + 1e-6)
    elif disc_func == 'kl':
        # TODO: kl divergence
        pass
    elif disc_func == 'w':
        # Wasserstein distance
        dist = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        loss_disc = dist(X_cf, X_f)
    else:
        raise Exception('unsupported distance function for discrepancy loss')
    return loss_disc

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--batch-size', type=int, default=5)
parser.add_argument('--epochs-autoencoder', type=int, default=10)
parser.add_argument('--hidden-dim-encoder', type=int, default=128)
parser.add_argument('--hidden-dim-decoder', type=int, default=256)
parser.add_argument('--latent-dim', type=int, default=32) 
parser.add_argument('--n-max-nodes', type=int, default=100)
parser.add_argument('--n-layers-encoder', type=int, default=2)
parser.add_argument('--n-layers-decoder', type=int, default=3)
parser.add_argument('--spectral-emb-dim', type=int, default=94)
parser.add_argument('--variational-autoencoder', action='store_true', default=True)
parser.add_argument('--epochs-denoise', type=int, default=90)
parser.add_argument('--timesteps', type=int, default=500)
parser.add_argument('--hidden-dim-denoise', type=int, default=512)
parser.add_argument('--n-layers_denoise', type=int, default=3)
parser.add_argument('--train-autoencoder', action='store_false', default=True)
parser.add_argument('--train-denoiser', action='store_true', default=True)
parser.add_argument('--n-properties', type=int, default=15)
parser.add_argument('--dim-condition', type=int, default=128)
parser.add_argument('--gamma1', type=int, default=5)
parser.add_argument('--gamma2', type=int, default=0.2)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

graph_types  = ["barabasi_albert", "cycle", "dual_barabasi_albert", "extended_barabasi_albert", "fast_gnp","ladder", "lobster", "lollipop","newman_watts_strogatz","regular", "partition","path", "powerlaw","star","stochastic","watts_strogatz","wheel"]

gr2id = {graph_types[i]:i for i in range(len(graph_types))}

data_lst = []
filename = f'processed_data/NBA.pt'

if os.path.isfile(filename):
    data_lst = torch.load(filename)
    for i in range(len(data_lst)):
        data_lst[i].edge_index = data_lst[i].edge_index.long()
        data_lst[i].edge_index_cf = data_lst[i].edge_index_cf.long()
        data_lst[i].T_cf_edge_index = data_lst[i].T_cf_edge_index.long()
        data_lst[i].T_edge_index = data_lst[i].T_edge_index.long()
    print(f'length: {len(data_lst)}')
else:
    print('No such dataset. Please check again. ')

# Slit into training, validation and test sets
idx = np.random.permutation(len(data_lst))
train_size = int(0.8*idx.size)
val_size = int(0.1*idx.size)

train_idx = [int(i) for i in idx[:train_size]]
val_idx = [int(i) for i in idx[train_size:train_size + val_size]]
test_idx = [int(i) for i in idx[train_size + val_size:]]

train_loader = DataLoader([data_lst[i] for i in train_idx], batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader([data_lst[i] for i in val_idx], batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader([data_lst[i] for i in test_idx], batch_size=args.batch_size, shuffle=False)

if args.variational_autoencoder:
    autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)
else:
    autoencoder = AutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

trainable_params_autoenc = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print("Number of Autoencoder's trainable parameters: "+str(trainable_params_autoenc))

# Train autoencoder
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        if args.variational_autoencoder:
            train_loss_all_recon = 0
            train_loss_all_kld = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            if args.variational_autoencoder:
                loss, recon, kld  = autoencoder.loss_function(data)
                train_loss_all_recon += recon.item()
                train_loss_all_kld += kld.item()
            else:
                loss = autoencoder.loss_function(data)#*data.x.size(0)
            loss.backward()
            if args.variational_autoencoder:
                train_loss_all += loss.item()
            else:
                train_loss_all += (torch.max(data.batch)+1) * loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        if args.variational_autoencoder:
            val_loss_all_recon = 0
            val_loss_all_kld = 0

        for data in val_loader:
            data = data.to(device)
            if args.variational_autoencoder:
                loss, recon, kld = autoencoder.loss_function(data)
                val_loss_all_recon += recon.item()
                val_loss_all_kld += kld.item()
            else:
                loss = autoencoder.loss_function(data)#*data.x.size(0)
            if args.variational_autoencoder:
                val_loss_all += loss.item()
            else:
                val_loss_all += torch.max(data.batch)+1 * loss.item()
            val_count += torch.max(data.batch)+1
            
        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if args.variational_autoencoder:
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/train_count, train_loss_all_recon/train_count, train_loss_all_kld/train_count, val_loss_all/val_count, val_loss_all_recon/val_count, val_loss_all_kld/val_count))
            else:
                print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder.pth.tar')
else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()
eval_autoencoder(test_loader, test_loader, autoencoder, args.n_max_nodes, device) # add also mse (loss that we use generally)


# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

denoise_model = DenoiseNN(feat_num=args.spectral_emb_dim+1, input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_properties, d_cond=args.dim_condition).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

trainable_params_diff = sum(p.numel() for p in denoise_model.parameters() if p.requires_grad)
print("Number of Diffusion model's trainable parameters: "+str(trainable_params_diff))

if args.train_denoiser:
    # Train denoising model
    best_val_loss = np.inf
    gamma_1 = args.gamma1
    gamma_2 = args.gamma2
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        
        for data in train_loader:
            
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            data.edge_index = data.edge_index_cf
            data.A = data.A_cf
            x_g_cf = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            
            loss_f, noise = p_losses(denoise_model, x_g, t, data.T_edge_index, data.x, data, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")   
            loss_cf, noise_cf = p_losses(denoise_model, x_g_cf, t, data.T_cf_edge_index, data.x, data, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber") 
            
            loss = gamma_1 * loss_f + gamma_2 * loss_cf
            
            loss.backward()
            train_loss_all += x_g.size(0) * loss.item()
            train_count += x_g.size(0)
            optimizer.step()

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
            loss, noise = p_losses(denoise_model, x_g, t, data.T_cf_edge_index, data.x, data, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
            val_loss_all += x_g.size(0) * loss.item()
            val_count += x_g.size(0)

        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])

denoise_model.eval()


ground_truth = []
pred = []
adjs = []

for k, data in enumerate(tqdm(train_loader, desc='Processing train set',)):
    data = data.to(device)
    stat = data.stats
    bs = stat.size(0)
    samples = sample(denoise_model, data.T_cf_edge_index, data.x, data, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
    x_sample = samples[-1]
    adj = autoencoder.decode_mu(x_sample)
    adjs.append(adj.cpu())

for k, data in enumerate(tqdm(val_loader, desc='Processing val set',)):
    data = data.to(device)
    stat = data.stats
    bs = stat.size(0)
    samples = sample(denoise_model, data.T_cf_edge_index, data.x, data, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
    x_sample = samples[-1]
    adj = autoencoder.decode_mu(x_sample)
    adjs.append(adj.cpu())

for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
    data = data.to(device)
    stat = data.stats
    bs = stat.size(0)
    samples = sample(denoise_model, data.T_cf_edge_index, data.x, data, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
    x_sample = samples[-1]
    adj = autoencoder.decode_mu(x_sample)
    adjs.append(adj.cpu())
    
torch.save(adjs, f'adj-gen-NBA-FairGDiff-{gamma_1}-{gamma_2}.pt')
