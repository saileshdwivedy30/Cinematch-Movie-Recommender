import pandas as pd
import torch
from model.train import BPRRecommender
from model.inference import get_top_n
import wandb

def hit_at_k(true_items, predicted_items, k):
    return int(any(item in predicted_items[:k] for item in true_items))

def ndcg_at_k(true_items, predicted_items, k):
    for i, item in enumerate(predicted_items[:k]):
        if item in true_items:
            return 1.0 / torch.log2(torch.tensor(i + 2, dtype=torch.float)).item()
    return 0.0

def evaluate_model():
    wandb.init(project="recommender-system", name="eval-bpr-v2-heldout-mf")
    config = wandb.config or {"embedding_size": 100}  # fallback

    test_df = pd.read_csv("data/test.csv")
    all_df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    num_users = all_df["user_id"].max()
    num_items = all_df["item_id"].max()

    model = BPRRecommender(num_users, num_items, embedding_size=config.get("embedding_size", 100))
    model.load_state_dict(torch.load("model/recommender.pt"))
    model.eval()
    print("BPR model loaded successfully for evaluation.")

    hit_total, ndcg_total = 0.0, 0.0
    test_users = test_df["user_id"].unique()

    for user_id in test_users:
        true_items = test_df[test_df["user_id"] == user_id]["item_id"].tolist()
        top_n = get_top_n(model, user_id, num_users, num_items, N=5)
        hit_total += hit_at_k(true_items, top_n, 5)
        ndcg_total += ndcg_at_k(true_items, top_n, 5)

    hit_rate = hit_total / len(test_users)
    ndcg_score = ndcg_total / len(test_users)

    print(f"Hit@5: {hit_rate:.4f}")
    print(f"NDCG@5: {ndcg_score:.4f}")

    wandb.log({"Hit@5": hit_rate, "NDCG@5": ndcg_score})
    wandb.finish()

if __name__ == "__main__":
    evaluate_model()
