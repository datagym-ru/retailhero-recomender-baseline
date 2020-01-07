import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse as sp

from utils import ProductEncoder, make_coo_row


class ItemToItemPredictor:
    def __init__(self, product_csv_path, model_pickled_path):
        self.product_encoder = ProductEncoder(product_csv_path)
        self.model = pickle.load(open(model_pickled_path, "rb"))

    def predict(self, transactions_history):
        row = make_coo_row(transactions_history, self.product_encoder).tocsr()
        raw_recs = self.model.recommend(
            userid=0, user_items=row, N=30, filter_already_liked_items=False, recalculate_user=True
        )
        return self.product_encoder.toPid([idx for (idx, score) in raw_recs])
