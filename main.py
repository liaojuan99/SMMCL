# main.py
import argparse
import torch
from train import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contrastive Learning for Recommendation')
    parser.add_argument('--data_path', type=str, default='./data/interactions.csv', help='Path to the interaction data')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Dimension of embeddings')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_users', type=int, default=100, help='Number of users')
    parser.add_argument('--num_items', type=int, default=100, help='Number of items')
    parser.add_argument('--num_behaviors', type=int, default=5, help='Number of behaviors')
    args = parser.parse_args()


    adj = torch.eye(args.num_users + args.num_items)

    train_model(args.data_path, args.embedding_dim, args.batch_size, args.epochs, args.lr, args.num_users, args.num_items, args.num_behaviors, adj)