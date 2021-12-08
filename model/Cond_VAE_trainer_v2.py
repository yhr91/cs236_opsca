"""
Script for training a VAE for GEX --> ATAC conversion

Cond VAE v2

"""

import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import torch.nn as nn


def full_block(in_features, out_features, p_drop):
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.LayerNorm(out_features),
            nn.ELU(),
            nn.Dropout(p=p_drop),
        )


class encoder(nn.Module):
    def __init__(self, x1_dim, hid_dim=64, z_dim=64,
                 p_drop=0, y_dim=0):
        super(encoder, self).__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            full_block(x1_dim + y_dim, hid_dim, p_drop),
            full_block(hid_dim, z_dim, p_drop),
        )

    def forward(self, x, y):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.encoder(xy)
        encoded = self.encoder(xy)
        return encoded


class decoder(nn.Module):
    def __init__(self, x2_dim, hid_dim=64, z_dim=64,
                 p_drop=0, y_dim=0):
        super(decoder, self).__init__()
        self.z_dim = z_dim

        self.decoder = nn.Sequential(
            full_block(z_dim + y_dim, hid_dim, p_drop),
            full_block(hid_dim, x2_dim, p_drop),
        )

    def forward(self, z, y):
        zy = z if y is None else torch.cat((z, y), dim=1)
        decoded = self.decoder(zy)
        return decoded


class Classifier(nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, y_dim)
        )

    def forward(self, x):
        return self.net(x)


class VAE(pl.LightningModule):
    def __init__(self, x1_dim=50, x2_dim=512,
                 enc_out_dim=512, latent_dim=256, gen_weight=1, class_weight=100, num_classes=21):
        super().__init__()

        self.save_hyperparameters()

        self.gen_weight = gen_weight
        self.class_weight = class_weight
        self.cls_1 = Classifier(x1_dim, num_classes)
        self.cls_2 = Classifier(x2_dim, num_classes)
        self.y_dim = num_classes

        # encoder, decoder
        self.encoder = encoder(
            x1_dim, latent_dim, enc_out_dim, p_drop=0, y_dim=num_classes)
        self.decoder = decoder(x2_dim, enc_out_dim,
                               latent_dim, p_drop=0, y_dim=num_classes)

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        # Set prior as fixed parameter attached to Module TODO: quick hack
        self.z_prior_m = torch.nn.Parameter(
            torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        # return log_pxz.sum()
        return log_pxz.sum(-1)
        #return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    # def classification_cross_entropy_1(self, x, y):
    #     y_logits = self.cls_1(x)
    #     return F.cross_entropy(y_logits, y)

    # def classification_cross_entropy_2(self, x, y):
    #     y_logits = self.cls_2(x)
    #     return F.cross_entropy(y_logits, y)

    def training_step(self, batch, batch_idx):
        x1, x2, y_label = batch
        batch_size = x1.shape[0]
        # q(y|x)
        y_logits = self.cls_1(x1)
        y_logprob = F.log_softmax(y_logits, dim=1)
        y_prob = torch.softmax(y_logprob, dim=1)  # (batch, y_dim)

        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), batch_size)   # (batch*self.y_dim,)
        y = x1.new(np.eye(self.y_dim)[y])    # (batch*self.y_dim, self.y_dim)
        x1 = duplicate(x1, self.y_dim)     # (batch*self.y_dim, x_dim)
        x2 = duplicate(x2, self.y_dim)     # (batch*self.y_dim, x_dim)
        # kl_y
        kl_y = kl_cat(y_prob, y_logprob, -np.log(self.y_dim)
                         * torch.ones_like(y_prob)).mean()
        # q(z|x, y)
        # encode x to get the mu and variance parameters
        x1_encoded = self.encoder(x1, y)
        mu, log_var = self.fc_mu(x1_encoded), self.fc_var(x1_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z_y = q.rsample()

        # kl_z_y: (batch, y_dim)
        kl_z_y = (kl_normal(
            mu, std**2, *self.z_prior).reshape(self.y_dim, -1)).transpose(1, 0)
        kl_z = (kl_z_y * y_prob).sum(-1).mean()

        # decoded
        x_hat = self.decoder(z_y, y)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x2).reshape(self.y_dim, -1).transpose(1, 0)

        # elbo
        kl = kl_z + kl_y
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        ce1 = F.cross_entropy(y_logits, y_label)
        ce2 = F.cross_entropy(self.cls_2(batch[1]), y_label)
        ce = (ce1 + ce2) * 0.5

        loss = self.gen_weight * elbo + self.class_weight * ce

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'kl': kl.mean(),
            'ce': ce
        })

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y_label = batch
        b_size = x1.shape[0]
        # q(y|x)
        y_logits = self.cls_1(x1)
        y_logprob = F.log_softmax(y_logits, dim=1)
        y_prob = torch.softmax(y_logprob, dim=1)  # (batch, y_dim)

        y = np.repeat(np.arange(self.y_dim), b_size)   # (batch*self.y_dim,)
        y = x1.new(np.eye(self.y_dim))[y]    # (batch*self.y_dim, self.y_dim)
        x1 = duplicate(x1, self.y_dim)     # (batch*self.y_dim, x_dim)
        x2 = duplicate(x2, self.y_dim)     # (batch*self.y_dim, x_dim)

        x_hat = self.transform(x1, y)
        # reconstruction loss
        # recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x2)
        logprob = self.gaussian_likelihood(x_hat, self.log_scale, x2).reshape(self.y_dim, -1).transpose(1, 0)
        recon_loss = -(logprob * y_prob).sum(-1).mean()

        return recon_loss

    def transform(self, x, y):
        """
        Performs multi modal domain alignment after training
        """
        x_encoded = self.encoder(x, y)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z, y)
        return x_hat


class multimodal(Dataset):
    def __init__(self, gex1, atac):

        self.gex1 = torch.Tensor(gex1.X.toarray())
        self.atac = torch.Tensor(atac.X.toarray())

        obs_cell_types = atac.obs['cell_type']
        classes = obs_cell_types.unique()
        t2idx = {t: int(i) for i, t in enumerate(classes)}
        self.y = torch.tensor(
            [t2idx[t] for t in obs_cell_types], dtype=torch.long)

    def __len__(self):
        return self.gex1.shape[0]

    def __getitem__(self, idx):
        return self.gex1[idx], self.atac[idx], self.y[idx]

def create_dataloaders(gex1, atac):
    print("Setting up dataloaders")

    idx = np.arange(len(gex1))
    trainval, test_idx = train_test_split(idx, test_size=0.10, shuffle=True)
    train_idx, val_idx = train_test_split(trainval, test_size=0.10, shuffle=True)

    gex1_train = gex1[train_idx]
    gex1_val = gex1[val_idx]
    gex1_test = gex1[test_idx]

    atac_train = atac[train_idx]
    atac_val = atac[val_idx]
    atac_test = atac[test_idx]

    train_dl = DataLoader(multimodal(gex1_train, atac_train), batch_size=64, shuffle=False)
    val_dl = DataLoader(multimodal(gex1_val, atac_val), batch_size=64, shuffle=False)
    test_dl = DataLoader(multimodal(gex1_test, atac_test), batch_size=64, shuffle=False)

    return train_dl, val_dl, test_dl


def kl_cat(q, log_q, log_p):
    """
    Computes the KL divergence between two categorical distributions

    Args:
        q: tensor: (batch, dim): Categorical distribution parameters
        log_q: tensor: (batch, dim): Log of q
        log_p: tensor: (batch, dim): Log of p

    Return:
        kl: tensor: (batch,) kl between each sample
    """
    element_wise = (q * (log_q - log_p))
    kl = element_wise.sum(-1)
    return kl


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) +
                          qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def duplicate(x, rep):
    """
    Duplicates x along dim=0

    Args:
        x: tensor: (batch, ...): Arbitrary tensor
        rep: int: (): Number of replicates. Setting rep=1 returns orignal x

    Returns:
        _: tensor: (batch * rep, ...): Arbitrary replicated tensor
    """
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])


def main():
    print("Reading in data")
    # Gene Expression Dataset 1 (GEX1)
    gex1 = sc.read_h5ad('/data/single_cell/multiome/multiome_gex_processed_training.h5ad')

    # DNA Accessibility Dataset (ATAC)
    atac = sc.read_h5ad('/data/single_cell/multiome/multiome_atac_processed_training.h5ad')

    pl.seed_everything(1234)
    train_dl, val_dl, test_dl = create_dataloaders(gex1, atac)

    vae = VAE(x1_dim=gex1.shape[1], x2_dim=atac.shape[1])

    trainer = pl.Trainer(gpus=1, max_epochs=1000, progress_bar_refresh_rate=10,
                        default_root_dir='./checkpoints/')

    trainer.fit(vae, train_dl, val_dl)


if __name__ == "__main__":
    main()
