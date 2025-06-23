import pandas as pd
import torch
from model.train import BPRRecommender

# Load popular items for cold start fallback
df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
popular_items = df.groupby("item_id")["rating"].count().sort_values(ascending=False).index.tolist()

def load_model(num_users, num_items, embedding_size=64):
    embedding_size=64
    model = BPRRecommender(num_users, num_items, embedding_size)
    model.load_state_dict(torch.load("model/recommender_np84a2fd.pt"))
    model.eval()
    return model

def get_top_n(model, user_id, num_users, num_items, N=5):
    if user_id > num_users or user_id < 1:
        return popular_items[:N]  # Cold start fallback

    user_tensor = torch.tensor([user_id] * num_items)
    item_tensor = torch.tensor(range(1, num_items + 1))

    with torch.no_grad():
        user_vec = model.user_embedding(user_tensor)
        item_vec = model.item_embedding(item_tensor)
        scores = (user_vec * item_vec).sum(1)

    top_indices = torch.topk(scores, N).indices
    return item_tensor[top_indices].tolist()
