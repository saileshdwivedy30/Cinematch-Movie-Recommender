import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import random
import os
import pickle


class RatingsDatasetNeuMF(Dataset):
    def __init__(self, df, num_items, num_negatives=4):
        self.user_item_pairs = df[['user_id', 'item_id']].values
        self.num_items = num_items
        self.user_positive_items = df.groupby('user_id')['item_id'].apply(set).to_dict()
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.user_item_pairs) * (1 + self.num_negatives)

    def __getitem__(self, idx):
        base_idx = idx // (1 + self.num_negatives)
        user, pos_item = self.user_item_pairs[base_idx]

        if idx % (1 + self.num_negatives) == 0:
            return torch.tensor(user), torch.tensor(pos_item), torch.tensor(pos_item)
        else:
            while True:
                neg_item = random.randint(1, self.num_items)
                if neg_item not in self.user_positive_items.get(user, set()):
                    break
            return torch.tensor(user), torch.tensor(pos_item), torch.tensor(neg_item)


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=128):
        super().__init__()
        self.user_embed_GMF = nn.Embedding(num_users + 1, embedding_size)
        self.item_embed_GMF = nn.Embedding(num_items + 1, embedding_size)

        self.user_embed_MLP = nn.Embedding(num_users + 1, embedding_size)
        self.item_embed_MLP = nn.Embedding(num_items + 1, embedding_size)

        self.mlp_layers = nn.Sequential(
            nn.Linear(2 * embedding_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(embedding_size + 64, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)

    def forward(self, user, item):
        gmf_user = self.user_embed_GMF(user)
        gmf_item = self.item_embed_GMF(item)
        gmf_output = gmf_user * gmf_item

        mlp_user = self.user_embed_MLP(user)
        mlp_item = self.item_embed_MLP(item)
        mlp_input = torch.cat((mlp_user, mlp_item), dim=1)
        mlp_output = self.mlp_layers(mlp_input)

        final_input = torch.cat((gmf_output, mlp_output), dim=1)
        logits = self.output_layer(final_input)
        return logits.squeeze()

    def predict(self, user, item):
        self.eval()
        with torch.no_grad():
            return self.forward(user, item)


def bpr_loss(pos_preds, neg_preds):
    return -torch.mean(torch.log(torch.sigmoid(pos_preds - neg_preds) + 1e-8))


def split_train_val_test(df, val_ratio=0.1):
    df_sorted = df.sort_values(by=["user_id", "timestamp"])
    test = df_sorted.groupby("user_id").tail(1)
    rest = df_sorted.drop(test.index)
    val_size = int(len(rest) * val_ratio)
    val = rest.iloc[:val_size]
    train = rest.iloc[val_size:]
    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    return train, val, test


def evaluate_hit_ndcg(model, df, num_users, num_items, k=5):
    from model.inference import get_top_n
    hit_total = 0
    ndcg_total = 0.0
    users = df["user_id"].unique()

    for user_id in users:
        true_items = df[df["user_id"] == user_id]["item_id"].tolist()
        top_n = get_top_n(model, user_id, num_users, num_items, N=k)
        hit_total += int(any(item in top_n for item in true_items))
        for i, item in enumerate(top_n):
            if item in true_items:
                ndcg_total += 1.0 / torch.log2(torch.tensor(i + 2, dtype=torch.float)).item()
                break
    return hit_total / len(users), ndcg_total / len(users)


def train_model():
    wandb.init(project="recommender-system", config={
        "embedding_size": 128,
        "lr": 0.001,
        "batch_size": 256,
        "epochs": 300,
        "weight_decay": 1e-5,
        "patience": 20,
        "num_negatives": 4
    })
    config = wandb.config

    df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    df["user_id"], user_index = pd.factorize(df["user_id"])
    df["item_id"], item_index = pd.factorize(df["item_id"])

    # Save mapping for evaluation
    with open(f"model/mapping_{wandb.run.id}.pkl", "wb") as f:
        pickle.dump({"user_index": user_index, "item_index": item_index}, f)

    train_df, val_df, _ = split_train_val_test(df)

    num_users = df["user_id"].max()
    num_items = df["item_id"].max()

    dataset = RatingsDatasetNeuMF(train_df, num_items, config.num_negatives)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = NeuMF(num_users, num_items, embedding_size=config.embedding_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_ndcg = -1
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for user, pos_item, neg_item in loader:
            optimizer.zero_grad()
            pos_preds = model(user, pos_item)
            neg_preds = model(user, neg_item)
            loss = bpr_loss(pos_preds, neg_preds)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        hit, ndcg = evaluate_hit_ndcg(model, val_df, num_users, num_items, k=5)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss,
            "val_Hit@5": hit,
            "val_NDCG@5": ndcg
        })
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Hit@5: {hit:.4f} | NDCG@5: {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            epochs_no_improve = 0
            model_path = f"model/recommender_neumf_{wandb.run.id}.pt"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.patience:
            print(f"Early stopping at epoch {epoch+1} — no NDCG@5 improvement.")
            break

    print("NeuMF training complete.")
    return model


if __name__ == "__main__":
    train_model()

