import logging
import os
import numpy as np
import torch
from torch import nn
import tqdm

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

from models import Generator, Discriminator
from utils import save_model_by_name, save_svd_model

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)

# +
BATCH_SIZE = 1024

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
    'n_pcs': 50,
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

print(input_mod1.obs.keys())
print(input_mod1.obs_names)
# print(input_mod1.obs["batch"])

# Do PCA on the input data
logging.info('Performing dimensionality reduction on modality 1 values...')
embedder_mod1 = TruncatedSVD(n_components=par['n_pcs'])
mod1_pca = embedder_mod1.fit_transform(input_mod1.X)
save_svd_model(embedder_mod1, run, 1)

logging.info('Performing dimensionality reduction on modality 2 values...')
embedder_mod2 = TruncatedSVD(n_components=par['n_pcs'])
mod2_pca = embedder_mod2.fit_transform(input_mod2.X)
save_svd_model(embedder_mod2, run, 2)

# This will get passed to the method
input_train_mod1 = mod1_pca[input_mod1.obs["batch"] != "s1d2"]
input_train_mod2 = mod2_pca[input_mod1.obs["batch"] != "s1d2"]
input_test_mod1 = mod1_pca[input_mod1.obs["batch"] == "s1d2"]
# This will get passed to the metric
true_test_mod2 = mod2_pca[input_mod1.obs["batch"] == "s1d2"]

assert len(input_train_mod1) + len(input_test_mod1) == len(mod1_pca)

# # Set up training

tensor_mod1 = torch.Tensor(input_train_mod1)
tensor_mod2 = torch.Tensor(input_train_mod2)

train_dataset = torch.utils.data.TensorDataset(tensor_mod1, tensor_mod2)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

bce = torch.nn.BCELoss()


def generator_loss(label, output):
    gen_loss = bce(label, output)
    return gen_loss


def discriminator_loss(label, output):
    disc_loss = bce(label, output)
    return disc_loss


def train(generator, discriminator, train_loader, device, tqdm, lr=1e-5, iter_max=np.inf, iter_save=np.inf,
          num_epochs=2, run=0):
    D_optimizer = torch.optim.SGD(discriminator.parameters(), lr=lr)
    G_optimizer = torch.optim.SGD(generator.parameters(), lr=lr)

    if iter_max is None:
        iter_max = int(num_epochs * len(train_dataset) / BATCH_SIZE)

    i = 1
    with tqdm(total=iter_max) as pbar:
        for epoch in range(1, num_epochs + 1):
            D_loss_list, G_loss_list = [], []

            for index, (gex, adt) in enumerate(train_loader):
                D_optimizer.zero_grad()
                gex = gex.to(device)
                adt = adt.to(device)

                real_target = torch.autograd.Variable(torch.ones(adt.size(0), 1).to(device))
                fake_target = torch.autograd.Variable(torch.zeros(adt.size(0), 1).to(device))

                D_real_loss = discriminator_loss(discriminator((gex, adt)), real_target)
                # print(discriminator(real_images))
                # D_real_loss.backward()

                generated_adt = generator(gex)
                output = discriminator((gex, generated_adt.detach()))
                D_fake_loss = discriminator_loss(output, fake_target)

                # train with fake
                # D_fake_loss.backward()

                D_total_loss = (D_real_loss + D_fake_loss) / 2
                D_loss_list.append(D_total_loss)

                D_total_loss.backward()
                D_optimizer.step()

                # Train generator with real labels
                G_optimizer.zero_grad()
                disc_gen = discriminator((gex, generated_adt))
                G_loss = generator_loss(disc_gen, real_target)
                G_loss_list.append(G_loss)

                G_loss.backward()
                G_optimizer.step()

                pbar.set_postfix(
                    d_loss='{:.2e}'.format(D_total_loss),
                    g_loss='{:.2e}'.format(G_loss))
                pbar.update(1)

                # Log summaries
                #                 if i % 50 == 0: ut.log_summaries(writer, summaries, i)

                del G_loss
                del D_fake_loss
                del D_real_loss
                del D_total_loss
                del output
                del disc_gen
                del generated_adt
                del real_target
                del fake_target

            # Save model
            if i % iter_save == 0:
                save_model_by_name(generator, i, run)
                save_model_by_name(discriminator, i, run)

            if i == iter_max:
                return

            i += 1


pca_dim1 = mod1_pca.shape[1]
pca_dim2 = mod2_pca.shape[1]

generator = Generator(pca_dim1, pca_dim2).to(device)
discriminator = Discriminator(pca_dim1, pca_dim2).to(device)

train(generator, discriminator, train_loader, device, tqdm.tqdm, lr=1e-5, iter_max=None,
      iter_save=100, num_epochs=10000, run=run)



