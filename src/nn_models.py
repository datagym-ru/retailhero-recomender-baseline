import torch
from torch import nn


class UserModel(nn.Module):
    def __init__(self, num_products, embedding_dim):
        super(UserModel, self).__init__()
        self._model = nn.Sequential(nn.Linear(num_products, embedding_dim),)

    def forward(self, x):
        return self._model(x)


class ItemModel(nn.Module):
    def __init__(self, num_products, embedding_dim):
        super(ItemModel, self).__init__()
        self._embeds = nn.Embedding(num_products, embedding_dim)

    def forward(self, x):
        return self._embeds(x)
