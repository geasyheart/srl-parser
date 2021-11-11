# -*- coding: utf-8 -*-

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, x):

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
