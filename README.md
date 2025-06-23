
# CineMatch - Personalized Movie Recommender System

This project implements an end-to-end recommender system using collaborative filtering and deep learning models. It supports training, evaluation, W&B sweep optimization, and real-time inference via a FastAPI service.

---

## 🔧 Features

- BPR (Bayesian Personalized Ranking)
- NeuMF (Neural Collaborative Filtering)
- BPR Multimodal Recommender (combining title + genre embedding using Sentence BERT)
- Offline evaluation with Hit@5, NDCG@5
- W&B sweep integration for hyperparameter tuning
- FastAPI-based REST service
- Dockerized for local deployment

---

## 📦 Setup

```bash
# Clone the repository
git clone https://github.com/saileshdwivedy30/cinematch.git
cd cinematch

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🚀 Training

### Train BPR Model

```bash
python model/train.py
```
### Train Mutlimodal BPR Model

```bash
python model/train_bpr_multimodal.py
```

---
### Train NeuMF Model

```bash
python model/train_neuf.py
```

---

## 🔁 Run W\&B Sweep

```bash
wandb sweep sweep.yaml
wandb agent <your_sweep_id>
```

---

## 🧪 Evaluation

```bash
python evaluate.py                   # Evaluate saved model
python evaluate_baseline.py         # Evaluate MostPopular baseline
python evaluate_all_sweeps.py       # Batch-evaluate sweep runs
```

---

## 🌐 API Usage

### Start locally:

```bash
uvicorn api.app:app --reload
```

### Sample request:

```http
GET /recommend?user_id=15
```

### Sample response:

```json
{
  "user_id": 15,
  "recommended_item_ids": [50, 258, 100, 181, 294]
}
```

---

## 🐳 Docker Support

```bash
docker-compose build
docker-compose up
```
### Sample request:

```http
 curl "http://localhost:8000/recommend?user_id=42"
```

The API will be live at:
[http://localhost:8000/recommend?user\_id=10](http://localhost:8000/recommend?user_id=10)

---

## 📊 Models Implemented

* `BPRRecommender`: Pairwise ranking model
* `BPR Multimodal Recommender`: BPR model using precomputed SBERT item embeddings for multimodal recommendation
* `NeuMF`: Hybrid MF + MLP neural model
* `MostPopularRecommender`: Fallback baseline

Model files are saved in `/model/recommender_<run_id>.pt`.
---

## 📚 Dataset

MovieLens 100K
[https://grouplens.org/datasets/movielens/100k](https://grouplens.org/datasets/movielens/100k)

---

## 📬 Contact

**LinkedIn**: [https://www.linkedin.com/in/saileshdwivedy/](https://www.linkedin.com/in/saileshdwivedy/)  
**Portfolio**: [https://saileshdwivedy30.github.io/](https://saileshdwivedy30.github.io/)  
**Email**: [sadw2186@colorado.edu](mailto:sadw2186@colorado.edu)



