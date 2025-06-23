import pandas as pd
import wandb
from model.baseline import MostPopularRecommender

def hit_at_k(true_items, predicted_items, k):
    return int(any(item in predicted_items[:k] for item in true_items))

def ndcg_at_k(true_items, predicted_items, k):
    for i, item in enumerate(predicted_items[:k]):
        if item in true_items:
            return 1.0 / (i + 2)**0.63  # approximate log2(i+2)
    return 0.0

def evaluate():
    wandb.init(project="recommender-system", name="baseline-mostpopular")

    test_df = pd.read_csv("data/test.csv")
    full_df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    model = MostPopularRecommender(full_df)

    hit_total, ndcg_total = 0.0, 0.0
    test_users = test_df["user_id"].unique()

    for user_id in test_users:
        true_items = test_df[test_df["user_id"] == user_id]["item_id"].tolist()
        top_n = model.recommend(user_id, N=5)
        hit_total += hit_at_k(true_items, top_n, 5)
        ndcg_total += ndcg_at_k(true_items, top_n, 5)

    hit_rate = hit_total / len(test_users)
    ndcg_score = ndcg_total / len(test_users)

    print(f"Baseline Hit@5: {hit_rate:.4f}")
    print(f"Baseline NDCG@5: {ndcg_score:.4f}")

    wandb.log({"Hit@5": hit_rate, "NDCG@5": ndcg_score})
    wandb.finish()

if __name__ == "__main__":
    evaluate()
