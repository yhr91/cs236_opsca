import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import scanpy as sc

# from dataloader import NucleiDatasetNew as NucleiDataset
#from dataloader import ATAC_Dataset
import model as AENet

import argparse
import numpy as np
import sys
import os
#import imageio

# adapted from pytorch/examples/vae and ethanluoyc/pytorch-vae

# parse arguments
def setup_args():

    options = argparse.ArgumentParser()

    options.add_argument('--save-dir', action="store", default="./", dest="save_dir")
    options.add_argument('-pt', action="store", dest="pretrained_file", default=None)
    options.add_argument('-bs', action="store", dest="batch_size", default = 32, type = int)
    options.add_argument('-ds', action="store", dest="datadir", default = "data/nuclear_crops_all_experiments/")

    options.add_argument('-iter', action="store", dest="max_iter", default = 100, type = int)
    options.add_argument('-lr', action="store", dest="lr", default=1e-3, type = float)
    options.add_argument('-nz', action="store", dest="nz", default=128, type = int)
    options.add_argument('-lamb', action="store", dest="lamb", default=0.0000001, type = float)
    options.add_argument('-lamb2', action="store", dest="lamb2", default=0.001, type = float)
    options.add_argument('--conditional', default=False, action="store_false")

    return options.parse_args()

args = setup_args()
os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, "log_ae_atac.txt"), 'w') as f:
    print(args, file=f)

# retrieve dataloader

"""
dataset = ATAC_Dataset(datadir="data/atac_seq_data/")
test_size = len(dataset) // 10
trainset = ATAC_Dataset(datadir="data/atac_seq_data/", start_idx=test_size+1)
testset = ATAC_Dataset(datadir="data/atac_seq_data/", end_idx=test_size)

train_loader = DataLoader(trainset, batch_size=args.batch_size, drop_last=False, shuffle=True)
test_loader = DataLoader(testset, batch_size=2, drop_last=False, shuffle=False)
"""

class scDataset(Dataset):
    def __init__(self, adata):
        self.X = torch.Tensor(adata.X.toarray())
        #self.labels = torch.Tensor(adata.obs.cell_type)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        atac_sample = self.X[idx]
        return {'tensor': atac_sample.float()}
        """
        cluster = self.labels[idx]
        return {'tensor': atac_sample.float(), 'binary_label': int(cluster)}
        """

def create_dataloaders(adata=None):
    print("Setting up dataloaders")

    idx = np.arange(len(adata))
    trainval, test_idx = train_test_split(idx, test_size=0.10, shuffle=True)
    train_idx, val_idx = train_test_split(trainval, test_size=0.10, shuffle=True)

    adata_train = adata[train_idx]
    adata_val = adata[val_idx]
    adata_test = adata[test_idx]

    train_dl = DataLoader(scDataset(adata_train), batch_size=args.batch_size, shuffle=False)
    val_dl = DataLoader(scDataset(adata_val), batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(scDataset(adata_test), batch_size=args.batch_size, shuffle=False)

    return train_dl, val_dl, test_dl

atac = sc.read_h5ad('../multiome/multiome_atac_processed_training.h5ad')
train_loader, val_loader, test_loader = create_dataloaders(atac)


print('Data loaded')

model = AENet.FC_VAE(n_input=atac.shape[1], nz=128)
if args.conditional:
    netCondClf = AENet.Simple_Classifier(nz=args.nz)

if args.pretrained_file is not None:
    model.load_state_dict(torch.load(args.pretrained_file))
    print("Pre-trained model loaded")
    sys.stdout.flush()

CE_weights = torch.FloatTensor([4.5, 0.5])

if torch.cuda.is_available():
    print('Using GPU')
    model.cuda()
    CE_weights = CE_weights.cuda()
    if args.conditional:
        netCondClf.cuda()

CE = nn.CrossEntropyLoss(CE_weights)

if args.conditional:
    optimizer = optim.Adam(list(model.parameters())+list(netCondClf.parameters()), lr = args.lr)
else:
    optimizer = optim.Adam([{'params': model.parameters()}], lr = args.lr)

def loss_function(recon_x, x, mu, logvar, latents):
    MSE = nn.MSELoss()
    lloss = MSE(recon_x,x)

    if args.lamb>0:
        KL_loss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        lloss = lloss + args.lamb*KL_loss

    return lloss

def train(epoch):
    model.train()
    if args.conditional:
        netCondClf.train()

    train_loss = 0
    total_clf_loss = 0

    for batch_idx, samples in enumerate(train_loader):

        inputs = Variable(samples['tensor'])
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        optimizer.zero_grad()
        recon_inputs, latents, mu, logvar = model(inputs)
        loss = loss_function(recon_inputs, inputs, mu, logvar, latents)
        train_loss += loss.data.item() * inputs.size(0)

        """
        if args.conditional:
            targets = Variable(samples['binary_label'])
            if torch.cuda.is_available():
                targets = targets.cuda()
            clf_outputs = netCondClf(latents)
            class_clf_loss = CE(clf_outputs, targets.view(-1).long())
            loss += args.lamb2 * class_clf_loss
            total_clf_loss += class_clf_loss.data.item() * inputs.size(0)
        """
        
        loss.backward()
        optimizer.step()

    with open(os.path.join(args.save_dir, "log.txt"), 'a') as f:
        print('Epoch: {} Average loss: {:.15f} Clf loss: {:.15f} '.format(epoch, train_loss / len(train_loader.dataset), total_clf_loss / len(train_loader.dataset)), file=f)

def test(epoch):
    model.eval()
    if args.conditional:
        netCondClf.eval()

    test_loss = 0
    total_clf_loss = 0

    for i, samples in enumerate(test_loader):

        inputs = Variable(samples['tensor'])
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        recon_inputs, latents, mu, logvar = model(inputs)

        loss = loss_function(recon_inputs, inputs, mu, logvar, latents)
        test_loss += loss.data.item() * inputs.size(0)

        """
        if args.conditional:
            targets = Variable(samples['binary_label'])
            if torch.cuda.is_available():
                targets = targets.cuda()
            clf_outputs = netCondClf(latents)
            class_clf_loss = CE(clf_outputs, targets.view(-1).long())
            total_clf_loss += class_clf_loss.data.item() * inputs.size(0)
        """

    test_loss /= len(test_loader.dataset)
    total_clf_loss /= len(test_loader.dataset)

    with open(os.path.join(args.save_dir, "log.txt"), 'a') as f:
        print('Test set loss: {:.15f} Test clf loss: {:.15f}'.format(test_loss, total_clf_loss), file=f)

    return test_loss


def save(epoch):
    model_dir = os.path.join(args.save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, str(epoch)+".pth"))
    if torch.cuda.is_available():
        model.cuda()

# main training loop
save(0)

_ = test(0)

for epoch in range(args.max_iter):
    print(epoch)
    train(epoch)
    _ = test(epoch)

    if epoch % 10 == 1:
        save(epoch)
