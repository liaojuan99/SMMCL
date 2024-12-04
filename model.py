# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.spmm(adj, x)
        out = self.linear(out)
        return out

class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_behaviors):
        super(RecommenderModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.behavior_embedding = nn.Embedding(num_behaviors, embedding_dim)
        self.gcn_layer = GCNLayer(embedding_dim, embedding_dim)

    def forward(self, user, item, behavior, adj):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        behavior_emb = self.behavior_embedding(behavior)
        x = user_emb + item_emb + behavior_emb
        x = self.gcn_layer(x, adj)
        return x