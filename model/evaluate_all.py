import os
import sys
import torch
import pandas as pd
import yaml
import wandb
from model.baseline import MostPopularRecommender
from model.inference import get_top_n, load_model

def hit_at_k(true_items, predicted_items, k):
    return int(any(item in predicted_items[:k] for item in true_items))

def ndcg_at_k(true_items, predicted_items, k):
    for i, item in enumerate(predicted_items[:k]):
        if item in true_items:
            return 1.0 / torch.log2(torch.tensor(i + 2, dtype=torch.float)).item()
    return 0.0

def find_local_config(run_id):
    for root, dirs, files in os.walk("wandb"):
        if root.endswith(f"{run_id}/files") and "config.yaml" in files:
            return os.path.join(root, "config.yaml")
    return None


def load_embedding_size_from_yaml(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        val = config.get("embedding_size", 100)
        return val["value"] if isinstance(val, dict) else val

def evaluate_model(model_path, embedding_size):
    import pickle

    run_id = model_path.split("_")[-1].replace(".pt", "")
    mapping_path = f"model/mapping_{run_id}.pkl"
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    user_index = mapping["user_index"]
    item_index = mapping["item_index"]

    test_df = pd.read_csv("data/test.csv")
    test_df["user_id"] = test_df["user_id"].map(lambda x: user_index.get_loc(x) if x in user_index else -1)
    test_df["item_id"] = test_df["item_id"].map(lambda x: item_index.get_loc(x) if x in item_index else -1)
    test_df = test_df[(test_df["user_id"] >= 0) & (test_df["item_id"] >= 0)]

    num_users = max(user_index.get_indexer(user_index))  # same as before training
    num_items = max(item_index.get_indexer(item_index))

    model = load_model(num_users, num_items, embedding_size, model_path)
    hit_total, ndcg_total = 0.0, 0.0
    test_users = test_df["user_id"].unique()

    for user_id in test_users:
        true_items = test_df[test_df["user_id"] == user_id]["item_id"].tolist()
        top_n = get_top_n(model, user_id, num_users, num_items, N=5)
        hit_total += hit_at_k(true_items, top_n, 5)
        ndcg_total += ndcg_at_k(true_items, top_n, 5)

    return hit_total / len(test_users), ndcg_total / len(test_users)

def evaluate_baseline():
    df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    test_df = pd.read_csv("data/test.csv")
    recommender = MostPopularRecommender(df)

    hit_total, ndcg_total = 0.0, 0.0
    test_users = test_df["user_id"].unique()

    for user_id in test_users:
        true_items = test_df[test_df["user_id"] == user_id]["item_id"].tolist()
        pred = recommender.recommend(user_id, N=5)
        hit_total += hit_at_k(true_items, pred, 5)
        ndcg_total += ndcg_at_k(true_items, pred, 5)

    return hit_total / len(test_users), ndcg_total / len(test_users)

def run_all():
    results = []

    # Evaluate trained models
    for file in os.listdir("model"):
        if file.startswith("recommender_") and file.endswith(".pt"):
            # Updated to extract only the run_id suffix
            parts = file.replace("recommender_", "").replace(".pt", "").split("_")
            run_id = parts[-1]

            config_path = find_local_config(run_id)
            if not config_path:
                print(f"ERROR: config.yaml not found for {file} (run_id: {run_id})")
                print(" Make sure the config.yaml is present in the corresponding wandb/run-* folder.")
                sys.exit(1)

            embedding_size = load_embedding_size_from_yaml(config_path)
            model_path = os.path.join("model", file)

            wandb.init(project="recommender-system", name=f"eval-{run_id}", config={
                "model_path": model_path,
                "embedding_size": embedding_size
            })
            hit, ndcg = evaluate_model(model_path, embedding_size)
            wandb.log({"Hit@5": hit, "NDCG@5": ndcg})
            wandb.finish()

            print(f"{file} | Hit@5: {hit:.4f} | NDCG@5: {ndcg:.4f}")
            results.append((file, hit, ndcg))

    # Evaluate baseline
    wandb.init(project="recommender-system", name="baseline-mostpopular")
    hit, ndcg = evaluate_baseline()
    wandb.log({"Hit@5": hit, "NDCG@5": ndcg})
    wandb.finish()
    results.append(("Baseline (MostPopular)", hit, ndcg))
    print(f"Baseline | Hit@5: {hit:.4f} | NDCG@5: {ndcg:.4f}")

    best = max(results, key=lambda x: x[1])
    print(f"\n Best Model: {best[0]} — Hit@5: {best[1]:.4f}, NDCG@5: {best[2]:.4f}")

if __name__ == "__main__":
    run_all()
