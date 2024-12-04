# train.py
import torch
from torch.utils.data import DataLoader
from dataset import InteractionDataset
from model import RecommenderModel
from loss import contrastive_loss, bpr_loss


def train_model(data_path, embedding_dim, batch_size, epochs, lr, num_users, num_items, num_behaviors, adj):

    interactions = [(0, 1, 0), (0, 2, 1), (1, 2, 0), (1, 3, 1)]  # (user, item, behavior) pairs
    dataset = InteractionDataset(interactions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RecommenderModel(num_users, num_items, embedding_dim, num_behaviors)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for user, pos_item, behavior in dataloader:
            neg_item = torch.randint(0, num_items, pos_item.shape)  # 随机选择负样本
            user_emb = model(user, pos_item, behavior, adj)
            pos_item_emb = model(user, pos_item, behavior, adj)
            neg_item_emb = model(user, neg_item, behavior, adj)

            cl_loss = contrastive_loss(user_emb, pos_item_emb, neg_item_emb)
            bpr_loss_value = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
            loss = cl_loss + bpr_loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')