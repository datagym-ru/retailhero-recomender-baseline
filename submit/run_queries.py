import sys
import json
import requests


def average_precision(actual, recommended, k=30):
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id is not None and product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / k


def normalized_average_precision(actual, recommended, k=30):
    actual = set(actual)
    if len(actual) == 0:
        return 0.0
    
    ap = average_precision(actual, recommended, k=k)
    ap_ideal = average_precision(actual, list(actual)[:k], k=k)
    return ap / ap_ideal


def run_queries(url, queryset_file):
    ap_values = []
    
    with open(queryset_file) as fin:
        for line in fin:
            query_data, next_transaction = line.strip().split('\t')
            query_data = json.loads(query_data)
            next_transaction = json.loads(next_transaction)
            
            resp = requests.post(url, json=query_data, timeout=1)
            resp.raise_for_status()
            resp_data = resp.json()
            
            assert len(resp_data['recommended_products']) <= 30
            
            ap = normalized_average_precision(next_transaction['product_ids'], resp_data['recommended_products'])
            ap_values.append(ap)
            
    map_score = sum(ap_values) / len(ap_values)
    return map_score


if __name__ == '__main__':
    url = sys.argv[1] # 'http://localhost:8000/recommend'
    queryset_file = sys.argv[2] # 'data/check_queries.tsv'
    score = run_queries(url, queryset_file)
    print('Score:', score)
