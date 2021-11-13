import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#from dataloader import RNA_Dataset, ATAC_Dataset
#from dataloader import NucleiDatasetNew as NucleiDataset
from model import FC_Autoencoder, FC_Classifier, VAE, FC_VAE, Simple_Classifier

#from dataloader import create_dataloaders

import os
import argparse
import numpy as np
#import imageio
import scanpy as sc

torch.manual_seed(1)

#============ PARSE ARGUMENTS =============

def setup_args():

    options = argparse.ArgumentParser()

    # save and directory options
    options.add_argument('--save-dir', action="store", default="./", dest="save_dir")
    options.add_argument('--save-freq', action="store", dest="save_freq", default=20, type=int)
    options.add_argument('--pretrained-file', default="./models/pretrain/91.pth",
                         action="store")

    # training parameters
    options.add_argument('-bs', '--batch-size', action="store", dest="batch_size", default=32, type=int)
    options.add_argument('-w', '--num-workers', action="store", dest="num_workers", default=10, type=int)
    options.add_argument('-lrAE', '--learning-rate-AE', action="store", dest="learning_rate_AE", default=1e-4, type=float)
    options.add_argument('-lrD', '--learning-rate-D', action="store", dest="learning_rate_D", default=1e-4, type=float)
    options.add_argument('-e', '--max-epochs', action="store", dest="max_epochs", default=100, type=int)
    options.add_argument('-wd', '--weight-decay', action="store", dest="weight_decay", default=0, type=float)
    options.add_argument('--train-atacnet', action="store_true")
    options.add_argument('--conditional', action="store_true")
    options.add_argument('--conditional-adv', action="store_true")

    # hyperparameters
    options.add_argument('--alpha', action="store", default=0.1, type=float)
    options.add_argument('--beta', action="store", default=1., type=float)
    options.add_argument('--lamb', action="store", default=0.00000001, type=float)
    options.add_argument('--latent-dims', action="store", default=128, type=int)

    # gpu options
    options.add_argument('-gpu', '--use-gpu', action="store_false", dest="use_gpu")

    return options.parse_args()

torch.cuda.set_device(4)
args = setup_args()
if not torch.cuda.is_available():
    args.use_gpu = False

os.makedirs(args.save_dir, exist_ok=True)

#============= TRAINING INITIALIZATION ==============

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

def create_dataloaders(gex, atac):
    print("Setting up dataloaders")

    idx = np.arange(len(gex))
    trainval, test_idx = train_test_split(idx, test_size=0.10, shuffle=True)
    train_idx, val_idx = train_test_split(trainval, test_size=0.10, shuffle=True)

    gex_train = gex[train_idx]
    gex_val = gex[val_idx]
    gex_test = gex[test_idx]
    
    atac_train = atac[train_idx]
    atac_val = atac[val_idx]
    atac_test = atac[test_idx]

    atac_loader = DataLoader(scDataset(atac_train), batch_size=args.batch_size, shuffle=False)
    atac_loader_val = DataLoader(scDataset(atac_val), batch_size=args.batch_size, shuffle=False)
    atac_loader_test = DataLoader(scDataset(atac_test), batch_size=args.batch_size, shuffle=False)
    
    genomics_loader = DataLoader(scDataset(gex_train), batch_size=args.batch_size, shuffle=False)
    genomics_loader_val = DataLoader(scDataset(gex_val), batch_size=args.batch_size, shuffle=False)
    genomics_loader_test = DataLoader(scDataset(gex_test), batch_size=args.batch_size, shuffle=False)

    return (atac_loader, atac_loader_val, atac_loader_test),\
           (genomics_loader, genomics_loader_val, genomics_loader_test)

gex1 = sc.read_h5ad('../multiome/multiome_gex_processed_training.h5ad')
atac = sc.read_h5ad('../multiome/multiome_atac_processed_training.h5ad')
atac_loaders, genomics_loaders = create_dataloaders(gex1, atac)
atac_loader, atac_loader_val, atac_loader_test = atac_loaders
genomics_loader, genomics_loader_val, genomics_loader_test = genomics_loaders


# initialize autoencoder
netRNA = FC_VAE(n_input=gex1.shape[1], nz=args.latent_dims)

netATAC = FC_VAE(n_input=atac.shape[1], nz=args.latent_dims)
netATAC.load_state_dict(torch.load(args.pretrained_file))
print("Pre-trained model loaded from %s" % args.pretrained_file)

if args.conditional_adv:
    netClf = FC_Classifier(nz=args.latent_dims+10)
    assert(not args.conditional)
else:
    netClf = FC_Classifier(nz=args.latent_dims)

if args.conditional:
    netCondClf = Simple_Classifier(nz=args.latent_dims)

if args.use_gpu:
    netRNA.cuda()
    netATAC.cuda()
    netClf.cuda()
    if args.conditional:
        netCondClf.cuda()

        
# load data
"""
genomics_dataset = RNA_Dataset(datadir="data/nCD4_gene_exp_matrices/")
atac_dataset = ATAC_Dataset(datadir="data/atac_seq_data/")

atac_loader = torch.utils.data.DataLoader(atac_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
genomics_loader = torch.utils.data.DataLoader(genomics_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
"""

# Our data
# Gene Expression Dataset 1 (GEX1)
# gex1 = sc.read_h5ad('/data/single_cell/multiome/multiome_gex_processed_training.h5ad')
# # DNA Accessibility Dataset (ATAC)
# atac = sc.read_h5ad('/data/single_cell/multiome/multiome_atac_processed_training.h5ad')
# train_dl, val_dl, test_dl = create_dataloaders(gex1, atac)


# setup optimizer
opt_netRNA = optim.Adam(list(netRNA.parameters()), lr=args.learning_rate_AE)
opt_netClf = optim.Adam(list(netClf.parameters()), lr=args.learning_rate_D, weight_decay=args.weight_decay)
opt_netATAC = optim.Adam(list(netATAC.parameters()), lr=args.learning_rate_AE)

if args.conditional:
    opt_netCondClf = optim.Adam(list(netCondClf.parameters()), lr=args.learning_rate_AE)

# loss criteria
criterion_reconstruct = nn.MSELoss()
criterion_classify = nn.CrossEntropyLoss()

# setup logger
with open(os.path.join(args.save_dir, 'log_rna_atac.txt'), 'w') as f:
    print(args, file=f)
    print(netRNA, file=f)
    print(netATAC, file=f)
    print(netClf, file=f)
    if args.conditional:
        print(netCondClf, file=f)

# define helper train functions
def compute_KL_loss(mu, logvar):
    if args.lamb>0:
        KLloss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return args.lamb * KLloss
    return 0

def train_autoencoders(rna_inputs, atac_inputs, rna_class_labels=None, atac_class_labels=None):
    netRNA.train()
    if args.train_atacnet:
        netATAC.train()
    else:
        netATAC.eval()
    netClf.eval()
    if args.conditional:
        netCondClf.train()

    # process input data
    rna_inputs, atac_inputs = Variable(rna_inputs), Variable(atac_inputs)

    if args.use_gpu:
        rna_inputs, atac_inputs = rna_inputs.cuda(), atac_inputs.cuda()

    # reset parameter gradients
    netRNA.zero_grad()

    # forward pass
    rna_recon, rna_latents, rna_mu, rna_logvar = netRNA(rna_inputs)
    atac_recon, atac_latents, atac_mu, atac_logvar = netATAC(atac_inputs)

    if args.conditional_adv:
        rna_class_labels, atac_class_labels = rna_class_labels.cuda(), atac_class_labels.cuda()
        rna_scores = netClf(torch.cat((rna_latents, rna_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
        atac_scores = netClf(torch.cat((atac_latents, atac_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
    else:
        rna_scores = netClf(rna_latents)
        atac_scores = netClf(atac_latents)

    rna_labels = torch.zeros(rna_scores.size(0),).long()
    atac_labels = torch.ones(atac_scores.size(0),).long()

    if args.conditional:
        rna_class_scores = netCondClf(rna_latents)
        atac_class_scores = netCondClf(atac_latents)

    if args.use_gpu:
        rna_labels, atac_labels = rna_labels.cuda(), atac_labels.cuda()
        if args.conditional:
            rna_class_labels, atac_class_labels = rna_class_labels.cuda(), atac_class_labels.cuda()

    # compute losses
    rna_recon_loss = criterion_reconstruct(rna_inputs, rna_recon)
    atac_recon_loss = criterion_reconstruct(atac_inputs, atac_recon)
    kl_loss = compute_KL_loss(rna_mu, rna_logvar) + compute_KL_loss(atac_mu, atac_logvar)
    clf_loss = 0.5*criterion_classify(rna_scores, atac_labels) + 0.5*criterion_classify(atac_scores, rna_labels)
    loss = args.alpha*(rna_recon_loss + atac_recon_loss) + clf_loss + kl_loss

    if args.conditional:
        clf_class_loss = 0.5*criterion_classify(rna_class_scores, rna_class_labels) + 0.5*criterion_classify(atac_class_scores, atac_class_labels)
        loss += args.beta*clf_class_loss

    # backpropagate and update model
    loss.backward()
    opt_netRNA.step()
    if args.conditional:
        opt_netCondClf.step()

    # if args.train_imagenet:
    if args.train_atacnet:
        opt_netATAC.step()

    summary_stats = {'rna_recon_loss': rna_recon_loss.item()*rna_scores.size(0),
            'atac_recon_loss': atac_recon_loss.item()*atac_scores.size(0),
            'clf_loss': clf_loss.item()*(rna_scores.size(0)+atac_scores.size(0))}

    if args.conditional:
        summary_stats['clf_class_loss'] = clf_class_loss.item()*(rna_scores.size(0)+atac_scores.size(0))

    return summary_stats

def train_classifier(rna_inputs, atac_inputs, rna_class_labels=None, atac_class_labels=None):

    netRNA.eval()
    netATAC.eval()
    netClf.train()

    # process input data
    rna_inputs, atac_inputs = Variable(rna_inputs), Variable(atac_inputs)

    if args.use_gpu:
        rna_inputs, atac_inputs = rna_inputs.cuda(), atac_inputs.cuda()

    # reset parameter gradients
    netClf.zero_grad()

    # forward pass
    _, rna_latents, _, _ = netRNA(rna_inputs)
    _, atac_latents, _, _ = netATAC(atac_inputs)

    if args.conditional_adv:
        rna_class_labels, atac_class_labels = rna_class_labels.cuda(), atac_class_labels.cuda()
        rna_scores = netClf(torch.cat((rna_latents, rna_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
        atac_scores = netClf(torch.cat((atac_latents, atac_class_labels.float().view(-1,1).expand(-1,10)), dim=1))
    else:
        rna_scores = netClf(rna_latents)
        atac_scores = netClf(atac_latents)

    rna_labels = torch.zeros(rna_scores.size(0),).long()
    atac_labels = torch.ones(atac_scores.size(0),).long()

    if args.use_gpu:
        rna_labels, atac_labels = rna_labels.cuda(), atac_labels.cuda()

    # compute losses
    clf_loss = 0.5*criterion_classify(rna_scores, rna_labels) + 0.5*criterion_classify(atac_scores, atac_labels)

    loss = clf_loss

    # backpropagate and update model
    loss.backward()
    opt_netClf.step()

    summary_stats = {'clf_loss': clf_loss*(rna_scores.size(0)+atac_scores.size(0)), 'rna_accuracy': accuracy(rna_scores, rna_labels), 'rna_n_samples': rna_scores.size(0),
            'atac_accuracy': accuracy(atac_scores, atac_labels), 'atac_n_samples': atac_scores.size(0)}

    return summary_stats

def accuracy(output, target):
    pred = output.argmax(dim=1).view(-1)
    correct = pred.eq(target.view(-1)).float().sum().item()
    return correct


def generate_atac(epoch):
    atac_dir = os.path.join(args.save_dir, "atac")
    os.makedirs(atac_dir, exist_ok=True)
    netRNA.eval()
    netATAC.eval()

    for i in range(5):
        samples = genomics_loader.dataset[np.random.randint(30)]
        rna_inputs = samples['tensor']
        rna_inputs = Variable(rna_inputs.unsqueeze(0))
        samples = atac_loader.dataset[np.random.randint(30)]
        atac_inputs = samples['tensor']
        atac_inputs = Variable(atac_inputs.unsqueeze(0))

        if torch.cuda.is_available():
            rna_inputs = rna_inputs.cuda()
            atac_inputs = atac_inputs.cuda()

        _, rna_latents, _, _ = netRNA(rna_inputs)
        recon_inputs = netATAC.decode(rna_latents)
        recon_atac, _, _, _ = netATAC(atac_inputs)


### main training loop
for epoch in range(args.max_epochs):
    print(epoch)

    if epoch % args.save_freq == 0:
        generate_atac(epoch)

    recon_rna_loss = 0
    recon_atac_loss = 0
    clf_loss = 0
    clf_class_loss = 0
    AE_clf_loss = 0

    n_rna_correct = 0
    n_rna_total = 0
    n_atac_correct = 0
    n_atac_total = 0

    for idx, (rna_samples, atac_samples) in enumerate(zip(genomics_loader, atac_loader)):
        rna_inputs = rna_samples['tensor']
        atac_inputs = atac_samples['tensor']

        if args.conditional or args.conditional_adv:
            rna_labels = rna_samples['binary_label']
            atac_labels = atac_samples['binary_label']
            out = train_autoencoders(rna_inputs, atac_inputs, rna_labels, atac_labels)
        else:
            out = train_autoencoders(rna_inputs, atac_inputs)

        recon_rna_loss += out['rna_recon_loss']
        recon_atac_loss += out['atac_recon_loss']
        AE_clf_loss += out['clf_loss']

        if args.conditional:
            clf_class_loss += out['clf_class_loss']

        if args.conditional_adv:
            out = train_classifier(rna_inputs, atac_inputs, rna_labels, atac_labels)
        else:
            out = train_classifier(rna_inputs, atac_inputs)

        clf_loss += out['clf_loss']
        n_rna_correct += out['rna_accuracy']
        n_rna_total += out['rna_n_samples']
        n_atac_correct += out['atac_accuracy']
        n_atac_total += out['atac_n_samples']

    recon_rna_loss /= n_rna_total
    clf_loss /= n_rna_total+n_atac_total
    AE_clf_loss /= n_rna_total+n_atac_total

    if args.conditional:
        clf_class_loss /= n_rna_total + n_atac_total

    with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
        print('Epoch: ', epoch, ', rna recon loss: %.8f' % float(recon_rna_loss), ', atac recon loss: %.8f' % float(recon_atac_loss),
                ', AE clf loss: %.8f' % float(AE_clf_loss), ', clf loss: %.8f' % float(clf_loss), ', clf class loss: %.8f' % float(clf_class_loss),
                ', clf accuracy RNA: %.4f' % float(n_rna_correct / n_rna_total), ', clf accuracy ATAC: %.4f' % float(n_atac_correct / n_atac_total), file=f)

    # save model
    if epoch % args.save_freq == 0:
        torch.save(netRNA.cpu().state_dict(), os.path.join(args.save_dir,"netRNA_%s.pth" % epoch))
        torch.save(netATAC.cpu().state_dict(), os.path.join(args.save_dir,"netATAC_%s.pth" % epoch))
        torch.save(netClf.cpu().state_dict(), os.path.join(args.save_dir,"netClf_%s.pth" % epoch))
        if args.conditional:
            torch.save(netCondClf.cpu().state_dict(), os.path.join(args.save_dir,"netCondClf_%s.pth" % epoch))

    if args.use_gpu:
        netRNA.cuda()
        netClf.cuda()
        netATAC.cuda()
        if args.conditional:
            netCondClf.cuda()

