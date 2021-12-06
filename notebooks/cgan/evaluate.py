import logging
import os
import numpy as np
import torch
from torch import nn
import tqdm

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import pickle

from models import Generator, Discriminator
from utils import load_model_by_name, load_svd_model

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)

# +
BATCH_SIZE = 128

data_path = '../data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# -

run = 1

## VIASH START
# Anything within this block will be removed by `viash` and will be
# replaced with the parameters as specified in your config.vsh.yaml.
par = {
    'input_train_mod1': os.path.join(data_path, 'multiome/multiome_gex_processed_training.h5ad'),
    'input_train_mod2': os.path.join(data_path, 'multiome/multiome_atac_processed_training.h5ad'),
    'distance_method': 'minkowski',
    'output': 'output.h5ad',
    'n_pcs': 100,
}
## VIASH END

# TODO: change this to the name of your method
method_id = "c-gan"

# # Load and preprocess data

logging.info('Reading `h5ad` files...')
input_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_mod2 = ad.read_h5ad(par['input_train_mod2'])

n_obs = input_mod1.n_obs
dim1 = input_mod1.n_vars
dim2 = input_mod2.n_vars
print(f"n_obs: {n_obs}, gex_dim: {dim1}, atac_dim: {dim2}")

# print(input_mod1.obs.keys())
# print(input_mod1.obs_names)
# print(input_mod1.obs["batch"])

# Do PCA on the input data
logging.info('Performing dimensionality reduction on modality 1 values...')
embedder_mod1 = load_svd_model(run, 1)
mod1_pca = embedder_mod1.transform(input_mod1.X)

logging.info('Performing dimensionality reduction on modality 2 values...')
embedder_mod2 = load_svd_model(run, 2)
# mod2_pca = embedder_mod2.transform(input_mod2.X)

n = mod1_pca.shape[0]

# +
# This will get passed to the method
# input_train_mod1 = mod1_pca[input_mod1.obs["batch"] != "s1d2"]
# input_train_mod2 = mod2_pca[input_mod1.obs["batch"] != "s1d2"]
input_test_mod1 = mod1_pca[input_mod1.obs["batch"] == "s1d2"][:n, :]

# This will get passed to the metric
true_test_mod2 = input_mod2[input_mod1.obs["batch"] == "s1d2"].X[:n, :].toarray()
# -

# assert len(input_train_mod1) + len(input_test_mod1) == len(mod1_pca)
assert input_test_mod1.shape[0] == true_test_mod2.shape[0]

# # Set up test

tensor_mod1 = torch.Tensor(input_test_mod1)
tensor_mod2 = torch.Tensor(true_test_mod2)

test_dataset = torch.utils.data.TensorDataset(tensor_mod1, tensor_mod2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

mse_loss_fn = torch.nn.MSELoss(reduction='mean')


def mse_loss(predicted, true):
    return mse_loss_fn(predicted, true)


def test(model, test_loader, device, tqdm, save_preds=False):
    with tqdm(total=len(test_dataset) // BATCH_SIZE) as pbar:
        model.eval()
        preds = []
        #         mse = 0
        for i, (mod1, mod2) in enumerate(test_loader):
            pred = model(mod1)
            #             print(type(preds), type(embedder_mod2.components_))
            #             print(preds.shape, embedder_mod2.components_.shape)
            pred = pred.detach().numpy() @ embedder_mod2.components_
            #             mse += float(mse_loss(torch.Tensor(pred), mod2))
            #             pbar.set_postfix(loss='{:.2e}'.format(rmse))
            preds.append(torch.Tensor(pred))
            pbar.update(1)
        preds = torch.Tensor(torch.concat(preds, axis=0))
        rmse = float(np.sqrt(mse_loss(torch.Tensor(preds), tensor_mod2)))

        if save_preds:
            return rmse, preds
        del pred
    return rmse, None


losses = {i: {} for i in range(0, 10)}

pca_dim1 = mod1_pca.shape[1]
# pca_dim2 = mod2_pca.shape[1]

r = run
ss = []
for i in range(38, 1000000):
    s = 100 * i
    if s <= 8500:
        ss.append(s)
    else:
        break
loss_list = []

for i, s in enumerate(ss):
    generator = Generator(pca_dim1, pca_dim1)
    load_model_by_name(generator, global_step=s, run=r, device=device)
    rmse, preds = test(generator, test_loader, device, tqdm.tqdm, save_preds=False)
    losses[r][s] = rmse
    loss_list.append(rmse)
    print(s, rmse, i / len(ss))
    del generator
plt.plot(loss_list)
plt.title("Conditional-GAN Loss")
plt.ylabel("RMSE")
plt.xlabel("Epochs")

print(loss_list, losses)
min_key = min(losses[run], key=losses[run].get)
print(min(loss_list), min_key)

plt.plot(list(losses[run].keys()), list(losses[run].values()))
plt.title("Conditional-GAN Loss")
plt.ylabel("RMSE")
plt.xlabel("Epochs")

# # Visualise predictions

s = min_key

generator = Generator(pca_dim1, pca_dim1)
load_model_by_name(generator, global_step=s, run=run, device=device)
rmse, preds = test(generator, test_loader, device, tqdm.tqdm, save_preds=True)
del generator

pred_adata = sc.AnnData(preds.detach().numpy())
print(pred_adata.X.shape, true_test_mod2.shape)
pred_adata.obs = input_mod2[input_mod1.obs["batch"] == "s1d2"].obs

sc.pp.neighbors(pred_adata)
sc.tl.umap(pred_adata)
sc.pl.umap(pred_adata, color=['cell_type'])

orig = input_mod2[input_mod1.obs["batch"] == "s1d2"]
sc.pp.neighbors(orig)
sc.tl.umap(orig)
sc.pl.umap(pred_adata, color=['cell_type'])


