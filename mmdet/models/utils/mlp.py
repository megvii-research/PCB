# Copyright (c) 2022 megvii-model. All Rights Reserved.
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, use_layer_norm=False):
        super().__init__()
        self.layers = nn.Sequential()
        self.dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        for i in range(n_layers):
            self.layers.add_module("fc{}".format(i+1), nn.Linear(self.dims[i], self.dims[i+1]))
            if i < n_layers - 1:
                self.layers.add_module("relu{}".format(i+1), nn.ReLU())
        
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.layers(x)
        return x