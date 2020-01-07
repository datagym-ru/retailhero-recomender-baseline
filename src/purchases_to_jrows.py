import json
import os

import pandas as pd
from tqdm import tqdm

import config as cfg
from utils import md5_hash


class Transaction:
    def __init__(self, transaction_id, transaction_datetime, **kwargs):
        self.data = {
            **{"tid": transaction_id, "datetime": transaction_datetime, "products": [],},
            **kwargs,
        }

    def add_item(
        self, product_id: str, product_quantity: float, trn_sum_from_iss: float, trn_sum_from_red: float,
    ) -> None:
        p = {
            "product_id": product_id,
            "quantity": product_quantity,
            "s": trn_sum_from_iss,
            "r": "0" if trn_sum_from_red is None or pd.isna(trn_sum_from_red) else trn_sum_from_red,
        }
        self.data["products"].append(p)

    def as_dict(self,):
        return self.data

    def transaction_id(self,):
        return self.data["tid"]


class ClientHistory:
    def __init__(
        self, client_id,
    ):
        self.data = {
            "client_id": client_id,
            "transaction_history": [],
        }

    def add_transaction(
        self, transaction,
    ):
        self.data["transaction_history"].append(transaction)

    def as_dict(self,):
        return self.data

    def client_id(self,):
        return self.data["client_id"]


class RowSplitter:
    def __init__(
        self, output_path, n_shards=16,
    ):
        self.n_shards = n_shards
        os.makedirs(
            output_path, exist_ok=True,
        )
        self.outs = []
        for i in range(self.n_shards):
            self.outs.append(open(output_path + "/{:02d}.jsons".format(i), "w",))
        self._client = None
        self._transaction = None

    def finish(self,):
        self.flush()
        for outs in self.outs:
            outs.close()

    def flush(self,):
        if self._client is not None:
            self._client.add_transaction(self._transaction.as_dict())
            # rows are sharded by cliend_id
            shard_idx = md5_hash(self._client.client_id()) % self.n_shards
            data = self._client.as_dict()
            self.outs[shard_idx].write(json.dumps(data) + "\n")

            self._client = None
            self._transaction = None

    def consume_row(
        self, row,
    ):
        if self._client is not None and self._client.client_id() != row.client_id:
            self.flush()

        if self._client is None:
            self._client = ClientHistory(client_id=row.client_id)

        if self._transaction is not None and self._transaction.transaction_id() != row.transaction_id:
            self._client.add_transaction(self._transaction.as_dict())
            self._transaction = None

        if self._transaction is None:
            self._transaction = Transaction(
                transaction_id=row.transaction_id,
                transaction_datetime=row.transaction_datetime,
                rpr=row.regular_points_received,
                epr=row.express_points_received,
                rps=row.regular_points_spent,
                eps=row.express_points_spent,
                sum=row.purchase_sum,
                store_id=row.store_id,
            )

        self._transaction.add_item(
            product_id=row.product_id,
            product_quantity=row.product_quantity,
            trn_sum_from_iss=row.trn_sum_from_iss,
            trn_sum_from_red=row.trn_sum_from_red,
        )


def split_data_to_chunks(
    input_path, output_dir, n_shards=16,
):
    splitter = RowSplitter(output_path=output_dir, n_shards=n_shards,)
    print("split_data_to_chunks: {} -> {}".format(input_path, output_dir,))
    for df in tqdm(pd.read_csv(input_path, chunksize=500000,)):
        for row in df.itertuples():
            splitter.consume_row(row)
    splitter.finish()


def calculate_unique_clients_from_input(input_path,):
    client_set = set()
    print("calculate_unique_clients_from: {}".format(input_path))
    for df in tqdm(pd.read_csv(input_path, chunksize=500000,)):
        client_set.update(set([row.client_id for row in df.itertuples()]))
    return len(client_set)


def calculate_unique_clients_from_output(output_dir,):
    import glob

    client_cnt = 0
    print("calculate_unique_clients_from: {}".format(output_dir))
    for js_file in glob.glob(output_dir + "/*.jsons"):
        for _ in open(js_file):
            client_cnt += 1
    return client_cnt


if __name__ == "__main__":
    purchases_csv_path = cfg.PURCHASE_CSV_PATH
    output_jsons_dir = cfg.JSONS_DIR

    split_data_to_chunks(
        purchases_csv_path, output_jsons_dir, n_shards=16,
    )

    # check splitting for correctness
    _from_input = calculate_unique_clients_from_input(purchases_csv_path)
    _from_output = calculate_unique_clients_from_output(output_jsons_dir)
    assert _from_input == _from_output
