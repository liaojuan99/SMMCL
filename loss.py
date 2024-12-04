# loss.py
import torch
import torch.nn.functional as F

def contrastive_loss(user_emb, pos_item_emb, neg_item_emb, temperature=0.1):
    pos_score = torch.exp(torch.sum(user_emb * pos_item_emb, dim=-1) / temperature)
    neg_score = torch.exp(torch.sum(user_emb * neg_item_emb, dim=-1) / temperature)
    loss = -torch.log(pos_score / (pos_score + neg_score)).mean()
    return loss

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.sum(user_emb * pos_item_emb, dim=-1)
    neg_score = torch.sum(user_emb * neg_item_emb, dim=-1)
    loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
    return loss