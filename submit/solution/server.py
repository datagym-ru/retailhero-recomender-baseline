import sys
sys.path.append("src")

from flask import Flask, jsonify, request
from item_to_item_predictor import ItemToItemPredictor


app = Flask(__name__)
PREDICTOR = ItemToItemPredictor(
    product_csv_path="assets/products.csv", 
    model_pickled_path="assets/model.pkl"
)


@app.route("/ready")
def ready():
    return "OK"


@app.route("/recommend", methods=["POST"])
def recommend():
    r = request.json
    result = PREDICTOR.predict(r.get("transaction_history", []))[:30]
    return jsonify({"recommended_products": result})


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=8000)
