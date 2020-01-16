from flask import Flask, jsonify, request
from collections import defaultdict

app = Flask(__name__)

default_best = [
	'4009f09b04', '15ccaa8685', 'bf07df54e1', '3e038662c0', '4dcf79043e',
    'f4599ca21a', '5cb93c9bc5', '4a29330c8d', '439498bce2', '343e841aaa',
    '0a46068efc', 'dc2001d036', '31dcf71bbd', '5645789fdf', '113e3ace79',
    'f098ee2a85', '53fc95e177', '080ace8748', '4c07cb5835', 'ea27d5dc75',
    'cbe1cd3bb3', '1c257c1a1b', 'f5e18af323', '5186e12ff4', '6d0f84a0ac',
    'f95785964a', 'ad865591c6', 'ac81544ebc', 'de25bccdaf', 'f43c12d228',
]


def make_recommendations_from_history(transaction_history, k=30):
    cnt = defaultdict(int)
    for trans in transaction_history:
        for item in trans["products"]:
              cnt[item["product_id"]] += 1

    # take only items with 1+ occurence
    cnt = {k:v for k,v in cnt.items() if v > 1}
    recs = sorted(list(cnt.keys()), key=lambda x: -cnt[x])[:k]
    
    # add items from global top if needed
    for item in default_best:
        if len(recs) >= k:
            break
        if item not in recs:
                recs.append(item)
 
    return recs


@app.route('/ready')
def ready():
    return "OK"


@app.route('/recommend', methods=['POST'])
def recommend():
    r = request.json
    return jsonify({
        'recommended_products': make_recommendations_from_history(r.get("transaction_history", []))
    })


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=8000)
