# if you are confused with `artifact` in filename, please read
# https://english.stackexchange.com/questions/37903/difference-between-artifact-and-artefact

import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp
from torch import nn
from tqdm import tqdm
from tqdm.notebook import tqdm

import config as cfg
from nn_models import ItemModel, UserModel
from utils import ProductEncoder, TrainingSample, coo_to_pytorch_sparse, normalized_average_precision

if __name__ == "__main__":

    product_encoder = ProductEncoder(cfg.PRODUCT_CSV_PATH)
    dim = 128

    input_dir = "../tmp/embds_d{}/".format(dim)
    output_dir = "../artifacts/embds_d{}/".format(dim)

    os.makedirs(output_dir, exist_ok=True)

    # load models
    user_model = UserModel(product_encoder.num_products, dim)
    item_model = ItemModel(product_encoder.num_products, dim)
    user_model.load_state_dict(torch.load(input_dir + "/user_model.pth"))
    item_model.load_state_dict(torch.load(input_dir + "/item_model.pth"))

    # conver user model to cpu
    user_model = user_model.cpu()
    torch.save(user_model.state_dict(), output_dir + "/user_model_cpu.pth")

    # export normalized item vectors
    item_vectors = item_model._embeds.weight.data.cpu().numpy()
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)
    np.save(output_dir + "/item_vectors.npy", item_vectors)

    # export knn index (compression and speed-up by FAISS, with Inner Product as distance)
    import faiss

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, 128, 16, 8)
    index.train(item_vectors)
    index.add(item_vectors)
    faiss.write_index(index, output_dir + "/knn.idx")
