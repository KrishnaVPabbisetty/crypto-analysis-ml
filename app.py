import random
from flask import Flask, request, jsonify

app = Flask(__name__)


# POST endpoint for decision, accuracy, and pair handling
@app.route("/prediction", methods=["POST"])
def submit():
    # Get the JSON data from the request body
    data = request.get_json()

    # Check if the 'pair' key exists in the request
    if not data or "pair" not in data:
        return jsonify({"error": "Missing required parameter: pair"}), 400

    # Extract the pair value from the request
    pair = data["pair"]

    # Create an array with 4 random vectors
    vectors = [random.uniform(1, 100) for _ in range(4)]

    # Divide each vector by the corresponding "accuracy" values
    accuracies = [75, 74, 77, 78]
    processed_vectors = [v / a for v, a in zip(vectors, accuracies)]

    # Randomly return Buy, Sell, or Hold
    decision = random.choice(["Buy", "Sell", "Hold"])

    # Create a response
    response = {
        "pair": pair,
        "processed_vectors": processed_vectors,
        "decision": decision,
    }

    # Return the response as JSON
    return jsonify(response), 200


# Main entry point
if __name__ == "__main__":
    app.run(debug=True)
