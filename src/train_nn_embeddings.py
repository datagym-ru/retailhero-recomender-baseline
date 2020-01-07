import itertools
import json
import os
from collections import defaultdict
from typing import List, Set

import numpy as np
import torch
from scipy import sparse as sp
from torch import nn
from tqdm import tqdm

import config as cfg
from nn_models import ItemModel, UserModel
from utils import (
    ProductEncoder,
    TrainingSample,
    coo_to_pytorch_sparse,
    get_shard_path,
    make_coo_row,
    normalized_average_precision,
)

MAX_COVERAGE = 0
MAX_AP = 0


def collect_train_data(jsons: List[str], product_encoder: ProductEncoder) -> List[TrainingSample]:
    samples = []
    for js_path in jsons:
        print("Load samples from {}".format(js_path))
        for js in tqdm((json.loads(s) for s in open(js_path))):
            samples.append(
                TrainingSample(
                    row=make_coo_row(js["transaction_history"], product_encoder),
                    target_items=set(product_encoder.toIdx(js["target"][0]["product_ids"])),
                    client_id=js["client_id"],
                )
            )
    return samples


def sample_aux_batch(batch: List[TrainingSample], neg_rate: int = 2, max_id: int = 43038):
    batch_indices = []
    batch_target = []
    batch_repeat_users = []
    for sample in batch:
        cur_repeat = 0
        if len(sample.target_items) > 0:
            positive_ids = sample.target_items
            num_positive = len(positive_ids)
            batch_indices.extend(positive_ids)
            batch_target.extend([1.0,] * num_positive)

            neg_candidates = np.random.choice(max_id, num_positive * neg_rate)
            negative_ids = [idx for idx in neg_candidates if idx not in positive_ids]
            num_negative = len(negative_ids)
            batch_indices.extend(negative_ids)
            batch_target.extend([-1.0,] * num_negative)
            cur_repeat = num_positive + num_negative
        batch_repeat_users.append(cur_repeat)

    return (
        torch.LongTensor(batch_repeat_users).cuda(),
        torch.LongTensor(batch_indices).cuda(),
        torch.FloatTensor(batch_target).cuda(),
    )


def evaluate(user_model: UserModel, item_model: ItemModel, valid_data: List[TrainingSample]):
    from sklearn.neighbors import NearestNeighbors

    user_vectors = (
        user_model(coo_to_pytorch_sparse(sp.vstack([sample.row for sample in valid_data])).cuda())
        .data.cpu()
        .numpy()
    )
    item_vectors = item_model._embeds.weight.data.cpu().numpy()
    knn = NearestNeighbors(n_neighbors=30, metric="cosine")
    knn.fit(item_vectors)
    preds = knn.kneighbors(user_vectors, n_neighbors=30, return_distance=False)

    coverage = []
    for i, recommended in enumerate(preds):
        gt = valid_data[i].target_items
        coverage.append(len(gt.intersection(recommended)) / len(gt))

    ap = []
    for i, recommended in enumerate(preds):
        gt = valid_data[i].target_items
        ap.append(normalized_average_precision(gt, recommended))

    return np.mean(coverage), np.mean(ap)


def eval_and_dump(
    batch_cnt: int,
    user_model: UserModel,
    item_model: ItemModel,
    valid_data: List[TrainingSample],
    output_root: str,
):
    global MAX_COVERAGE
    global MAX_AP
    coverage, ap = evaluate(user_model, item_model, valid_data)
    stats = {"batch_cnt": batch_cnt, "coverage": coverage, "ap": ap}
    print("[eval] {}".format(stats))
    if coverage > MAX_COVERAGE:
        MAX_COVERAGE = coverage
        print("\t This is a new COVERAGE leader!")
        output_dir = output_root + "/cov/"
        os.makedirs(output_dir, exist_ok=True)
        print("\t Save state to {}".format(output_dir))
        torch.save(user_model.state_dict(), output_dir + "/user_model.pth")
        torch.save(item_model.state_dict(), output_dir + "/item_model.pth")
        json.dump(stats, open(output_dir + "/stats.json", "w"))
    if ap > MAX_AP:
        MAX_AP = ap
        print("\t This is a new AP leader!")
        output_dir = output_root + "/ap/"
        os.makedirs(output_dir, exist_ok=True)
        print("\t Save state to {}".format(output_dir))
        torch.save(user_model.state_dict(), output_dir + "/user_model.pth")
        torch.save(item_model.state_dict(), output_dir + "/item_model.pth")
        json.dump(stats, open(output_dir + "/stats.json", "w"))


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(43)

    product_encoder = ProductEncoder(cfg.PRODUCT_CSV_PATH)
    train_samples = collect_train_data([get_shard_path(i) for i in range(8)], product_encoder,)
    valid_samples = collect_train_data([get_shard_path(15)], product_encoder)

    dim = 512
    user_model = UserModel(product_encoder.num_products, dim).cuda()
    item_model = ItemModel(product_encoder.num_products, dim).cuda()

    criterion = nn.CosineEmbeddingLoss(margin=0.01).cuda()
    optimizer = torch.optim.Adam(list(user_model.parameters()) + list(item_model.parameters()), lr=0.01)

    epoches = (
        [{"num_batches": 512, "batch_size": 32, "neg_rate": 32}]
        + [
            {"num_batches": 256, "batch_size": 64, "neg_rate": 256},
            {"num_batches": 256, "batch_size": 128, "neg_rate": 128},
        ]
        * 30
        + [{"num_batches": 256, "batch_size": 64, "neg_rate": 256}] * 20
    )

    batch_cnt = 0
    for epoch in epoches:
        for _ in tqdm(range(epoch["num_batches"])):
            batch_cnt += 1

            optimizer.zero_grad()

            batch_samples = np.random.choice(train_samples, epoch["batch_size"], replace=False)
            _input = coo_to_pytorch_sparse(sp.vstack([sample.row for sample in batch_samples])).cuda()
            _repeat, _idx, _target = sample_aux_batch(
                batch=batch_samples, neg_rate=epoch["neg_rate"], max_id=product_encoder.num_products
            )

            raw_users = user_model.forward(_input)
            repeated_users = torch.repeat_interleave(raw_users, _repeat, dim=0)
            repeated_items = item_model.forward(_idx)

            loss = criterion(repeated_items, repeated_users, _target)
            loss.backward()
            optimizer.step()
        eval_and_dump(batch_cnt, user_model, item_model, valid_samples, "../tmp/embds_d{}/".format(dim))
