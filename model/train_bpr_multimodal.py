# BPR with SBERT-enhanced metadata features

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import numpy as np
import random
import os

class RatingsDatasetBPR(Dataset):
    def __init__(self, df, num_items):
        self.user_item_pairs = df[['user_id', 'item_id']].values
        self.num_items = num_items
        self.user_positive_items = df.groupby('user_id')['item_id'].apply(set).to_dict()

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, pos_item = self.user_item_pairs[idx]
        while True:
            neg_item = random.randint(1, self.num_items)
            if neg_item not in self.user_positive_items.get(user, set()):
                break
        return torch.tensor(user), torch.tensor(pos_item), torch.tensor(neg_item)

class BPRMultimodalRecommender(nn.Module):
    def __init__(self, num_users, item_text_embeddings):
        super().__init__()
        self.num_items, self.embedding_dim = item_text_embeddings.shape
        self.user_embedding = nn.Embedding(num_users + 1, self.embedding_dim)

        # Freeze item embeddings
        self.item_embedding = nn.Embedding.from_pretrained(item_text_embeddings, freeze=True)

        nn.init.normal_(self.user_embedding.weight, std=0.1)

    def forward(self, user, pos_item, neg_item):
        user_vec = self.user_embedding(user)
        pos_vec = self.item_embedding(pos_item)
        neg_vec = self.item_embedding(neg_item)
        pos_score = (user_vec * pos_vec).sum(1)
        neg_score = (user_vec * neg_vec).sum(1)
        return pos_score, neg_score

def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

def split_train_test(df):
    df_sorted = df.sort_values(by=["user_id", "timestamp"])
    test = df_sorted.groupby("user_id").tail(1)
    train = df_sorted.drop(test.index)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    return train, test

def train_model():
    wandb.init(project="recommender-system")
    config = wandb.config

    df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    train_df, _ = split_train_test(df)

    num_users = df["user_id"].max()

    # Load precomputed SBERT-based item embeddings
    item_text_embeddings = np.load("data/item_text_embeddings.npy")

    item_text_embeddings = np.vstack([
        np.zeros((1, item_text_embeddings.shape[1])),
        item_text_embeddings
    ])

    item_text_tensor = torch.tensor(item_text_embeddings, dtype=torch.float)

    dataset = RatingsDatasetBPR(train_df, item_text_tensor.shape[0] - 1)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = BPRMultimodalRecommender(num_users, item_text_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for user, pos_item, neg_item in loader:
            optimizer.zero_grad()
            pos_scores, neg_scores = model(user, pos_item, neg_item)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        wandb.log({"epoch": epoch + 1, "bpr_loss": total_loss})
        print(f"Epoch {epoch+1}, BPR (multimodal) Loss: {total_loss:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            model_path = f"model/recommender_multimodal_{wandb.run.id}.pt"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)

    print("Multimodal BPR training complete.")
    return model

if __name__ == "__main__":
    sweep_defaults = {
        "lr": 0.005,
        "batch_size": 512,
        "weight_decay": 1e-4,
        "epochs": 15
    }
    wandb.init(config=sweep_defaults, project="recommender-system")
    train_model()
