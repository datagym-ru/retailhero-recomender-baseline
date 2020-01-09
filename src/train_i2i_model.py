import glob
import json
import os
import pickle
import sys

import implicit
import numpy as np
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm

import config as cfg
from utils import (
    ProductEncoder,
    TrainingSample,
    coo_to_pytorch_sparse,
    get_shard_path,
    make_coo_row,
    normalized_average_precision,
)

if __name__ == "__main__":

    product_encoder = ProductEncoder(cfg.PRODUCT_CSV_PATH)

    rows = []
    for i in range(15):
        for js in tqdm((json.loads(s) for s in open(get_shard_path(i)))):
            rows.append(make_coo_row(js["transaction_history"], product_encoder))
    train_mat = sp.vstack(rows)

    model = implicit.nearest_neighbours.CosineRecommender(K=1)
    model.fit(train_mat.T)
    out_dir = "../tmp/implicit_cosine1/"
    os.makedirs(out_dir, exist_ok=True)
    print("Dump model to " + out_dir)
    pickle.dump(model, open(out_dir + "/model.pkl", "wb"))

    print("Estimate quiality...")
    scores = []
    for js in tqdm((json.loads(s) for s in open(get_shard_path(15)))):
        row = make_coo_row(js["transaction_history"], product_encoder).tocsr()
        raw_recs = model.recommend(
            userid=0, user_items=row, N=30, filter_already_liked_items=False, recalculate_user=True
        )
        recommended_items = product_encoder.toPid([idx for (idx, score) in raw_recs])
        gt_items = js["target"][0]["product_ids"]
        ap = normalized_average_precision(gt_items, recommended_items)
        scores.append(ap)
    print("\tmap: {}".format(np.mean(scores)))
