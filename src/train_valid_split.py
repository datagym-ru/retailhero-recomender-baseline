import glob
import json
import random
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

import config as cfg


def transaction_to_target(transaction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tid": transaction["tid"],
        "datetime": transaction["datetime"],
        "product_ids": [e["product_id"] for e in transaction["products"]],
        "store_id": transaction["store_id"],
    }


def get_client_info(client_data_path: str) -> Dict[str, Dict]:
    client_info = {}
    for row in pd.read_csv(client_data_path).itertuples():
        client_info[row.client_id] = {
            "age": row.age,
            "gender": row.gender,
            "client_id": row.client_id,
        }
    return client_info


if __name__ == "__main__":
    random.seed(43)  # lets be special

    client_csv_path = cfg.CLIENT_CSV_PATH
    jsons_root = cfg.JSONS_DIR

    client_info = get_client_info(client_csv_path)

    print("process shards")
    for js_path in tqdm(sorted(glob.glob(jsons_root + "/*.jsons"))):
        fout = open(js_path + ".splitted", "w")
        for js in (json.loads(s) for s in open(js_path)):
            sorted_transactions = sorted(js["transaction_history"], key=lambda x: x["datetime"])
            split_candidates = [
                t["datetime"] for t in sorted_transactions if t["datetime"] > cfg.BASE_SPLIT_POINT
            ]
            if len(split_candidates) == 0:
                # no transactions after split points - so we cannot validates on this sample, skip it.
                continue
            split_point = random.choice(split_candidates)
            train_transactions = [t for t in sorted_transactions if t["datetime"] < split_point]
            test_transactons = [t for t in sorted_transactions if t["datetime"] >= split_point]

            # copy info about client% client_id, age, gender
            sample = {**client_info[js["client_id"]]}
            sample["transaction_history"] = train_transactions
            sample["target"] = [transaction_to_target(x) for x in test_transactons]

            fout.write(json.dumps(sample) + "\n")
        fout.close()
