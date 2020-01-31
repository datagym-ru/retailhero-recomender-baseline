import hashlib
from typing import List, Set
import numpy as np
import pandas as pd
from scipy import sparse as sp

import src.config as cfg


class ProductEncoder:
    def __init__(self, product_csv_path):
        self.product_idx = {}
        self.product_pid = {}
        for idx, pid in enumerate(pd.read_csv(product_csv_path).product_id.values):
            self.product_idx[pid] = idx
            self.product_pid[idx] = pid

    def toIdx(self, x):
        if type(x) == str:
            pid = x
            return self.product_idx[pid]
        return [self.product_idx[pid] for pid in x]

    def toPid(self, x):
        if type(x) == int:
            idx = x
            return self.product_pid[idx]
        return [self.product_pid[idx] for idx in x]

    @property
    def num_products(self):
        return len(self.product_idx)


def make_coo_row(transaction_history, product_encoder: ProductEncoder):
    idx = []
    values = []

    items = []
    for trans in transaction_history:
        items.extend([i["product_id"] for i in trans["products"]])
    n_items = len(items)

    for pid in items:
        idx.append(product_encoder.toIdx(pid))
        values.append(1.0 / n_items)

    return sp.coo_matrix(
        (np.array(values).astype(np.float32), ([0] * len(idx), idx)), shape=(1, product_encoder.num_products),
    )


def np_normalize_matrix(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norm


def get_shard_path(n_shard, jsons_dir=cfg.JSONS_DIR):
    return "{}/{:02d}.jsons.splitted".format(jsons_dir, n_shard)


def md5_hash(x):
    return int(hashlib.md5(x.encode()).hexdigest(), 16)
