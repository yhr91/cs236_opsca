import torch
from torch import nn


class Generator(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super(Generator, self).__init__()
        self.name = "generator"

        self.model = nn.Sequential(nn.Linear(dim1, 64 * 4),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(64 * 4, 64 * 2),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(0.4),
                                   nn.Linear(64 * 2, dim2),
                                   nn.ReLU(inplace=True),)

    def forward(self, gex_vector):
        adt = self.model(gex_vector)
        return adt


class Discriminator(nn.Module):
    def __init__(self, dim1, dim2):
        super(Discriminator, self).__init__()
        self.name = "discriminator"

        self.model = nn.Sequential(nn.Linear(dim1 + dim2, 64 * 8),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(64 * 8, 64 * 4),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(64 * 4, 64 * 2),
                                   nn.Dropout(0.4),
                                   nn.Linear(64 * 2, 1),
                                   nn.Sigmoid(),)

    def forward(self, inputs):
        gex, adt = inputs
        concat = torch.cat((gex, adt), dim=1)
        output = self.model(concat)
        return output
