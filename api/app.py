from fastapi import FastAPI, Request
from model.inference import load_model, get_top_n
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

app = FastAPI()

df = pd.read_csv("data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
num_users = df["user_id"].max()
num_items = df["item_id"].max()
valid_user_ids = set(df["user_id"].unique())

EMBEDDING_SIZE = 100  # Make this configurable if needed

try:
    model = load_model(num_users, num_items, embedding_size=EMBEDDING_SIZE)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

@app.get("/recommend")
def recommend(user_id: int):
    if user_id not in valid_user_ids:
        logging.warning(f"Unknown user_id {user_id}, returning fallback items.")
        return {"user_id": user_id, "recommended_item_ids": [50, 258, 100, 181, 294]}
    try:
        top_items = get_top_n(model, user_id, num_users, num_items, N=5)
        logging.info(f"Recommendations for user_id {user_id}: {top_items}")
        return {"user_id": user_id, "recommended_item_ids": top_items}
    except Exception as e:
        logging.error(f"Failed to generate recommendations for user_id {user_id}: {e}")
        return {"error": "Failed to generate recommendations."}
