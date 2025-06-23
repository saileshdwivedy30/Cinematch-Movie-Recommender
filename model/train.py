# bpr + bias + wb sweep + embedding + epoc and other hyper params

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import random

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

class BPRRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_size)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_size)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

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
    sweep_config_defaults = {
        "lr": 0.005,
        "embedding_size": 100,
        "batch_size": 512,
        "weight_decay": 1e-4,
        "epochs": 10
    }
    wandb.init(project="recommender-system", config=sweep_config_defaults)
    config = wandb.config

    df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    train_df, _ = split_train_test(df)

    num_users = df["user_id"].max()
    num_items = df["item_id"].max()

    dataset = RatingsDatasetBPR(train_df, num_items)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = BPRRecommender(num_users, num_items, embedding_size=config.embedding_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_loss = float("inf")
    patience = 3
    no_improve_epochs = 0

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for user, pos_item, neg_item in loader:
            optimizer.zero_grad()
            pos_scores, neg_scores = model(user, pos_item, neg_item)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        wandb.log({"epoch": epoch + 1, "bpr_loss": total_loss})
        print(f"Epoch {epoch+1}, BPR Loss: {total_loss:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            no_improve_epochs = 0
            model_path = f"model/recommender_{wandb.run.id}.pt"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    return model, num_users, num_items

if __name__ == "__main__":
    train_model()

