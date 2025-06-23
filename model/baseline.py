import pandas as pd
from collections import Counter

class MostPopularRecommender:
    def __init__(self, df):
        self.popular_items = self._get_popular_items(df)

    def _get_popular_items(self, df):
        return df["item_id"].value_counts().index.tolist()

    def recommend(self, user_id, N=5):
        return self.popular_items[:N]
