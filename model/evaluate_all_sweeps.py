import os
import torch
import pandas as pd
import yaml
import wandb
from model.train import BPRRecommender
from model.inference import get_top_n

def hit_at_k(true_items, predicted_items, k):
    return int(any(item in predicted_items[:k] for item in true_items))

def ndcg_at_k(true_items, predicted_items, k):
    for i, item in enumerate(predicted_items[:k]):
        if item in true_items:
            return 1.0 / torch.log2(torch.tensor(i + 2, dtype=torch.float)).item()
    return 0.0

def find_local_config(run_id):
    for root, dirs, files in os.walk("wandb"):
        for file in files:
            if file == "config.yaml" and run_id in root:
                return os.path.join(root, file)
    return None

def load_embedding_size_from_yaml(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        raw_val = config.get("embedding_size", 100)
        # Fix for W&B sweeps: might be {"value": 128}
        if isinstance(raw_val, dict) and "value" in raw_val:
            return raw_val["value"]
        return raw_val


def evaluate_model(model_path, embedding_size):
    test_df = pd.read_csv("data/test.csv")
    all_df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    num_users = all_df["user_id"].max()
    num_items = all_df["item_id"].max()

    model = BPRRecommender(num_users, num_items, embedding_size=embedding_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    hit_total, ndcg_total = 0.0, 0.0
    test_users = test_df["user_id"].unique()

    for user_id in test_users:
        true_items = test_df[test_df["user_id"] == user_id]["item_id"].tolist()
        top_n = get_top_n(model, user_id, num_users, num_items, N=5)
        hit_total += hit_at_k(true_items, top_n, 5)
        ndcg_total += ndcg_at_k(true_items, top_n, 5)

    hit_rate = hit_total / len(test_users)
    ndcg_score = ndcg_total / len(test_users)

    return hit_rate, ndcg_score

def run_all():
    results = []

    for file in os.listdir("model"):
        if file.startswith("recommender_") and file.endswith(".pt"):
            run_id = file.replace("recommender_", "").replace(".pt", "")
            config_path = find_local_config(run_id)

            if not config_path:
                print(f"Skipping {file} — config not found locally")
                continue

            embedding_size = load_embedding_size_from_yaml(config_path)
            model_path = os.path.join("model", file)

            wandb.init(project="recommender-system", name=f"eval-{run_id}", config={"model_path": model_path, "embedding_size": embedding_size})
            hit, ndcg = evaluate_model(model_path, embedding_size)
            wandb.log({"Hit@5": hit, "NDCG@5": ndcg})
            wandb.finish()

            print(f"Evaluated {file} | Hit@5: {hit:.4f} | NDCG@5: {ndcg:.4f}")
            results.append((file, hit, ndcg))

    if results:
        best = max(results, key=lambda x: x[1])
        print(f"\n Best Model: {best[0]} — Hit@5: {best[1]:.4f}, NDCG@5: {best[2]:.4f}")
    else:
        print("No models evaluated.")

if __name__ == "__main__":
    run_all()
