from flask import Flask, render_template, request, jsonify
from pickle import load

# Load the trained model
with open("rent_model.pkl", "rb") as f:
    model = load(f)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    br = int(data.get("br"))
    fs = int(data.get("fs"))
    loc = int(data.get("loc"))

    # Bedroom encoding
    if br == 1:
        d1 = [1, 0, 0, 0]
    elif br == 2:
        d1 = [0, 1, 0, 0]
    elif br == 3:
        d1 = [0, 0, 1, 0]
    else:
        d1 = [0, 0, 0, 1]

    # Furnishing encoding
    if fs == 1:
        d2 = [1, 0, 0]
    elif fs == 2:
        d2 = [0, 1, 0]
    else:
        d2 = [0, 0, 1]

    # Location encoding
    if loc == 1:
        d3 = [1, 0, 0]
    elif loc == 2:
        d3 = [0, 1, 0]
    else:
        d3 = [0, 0, 1]

    d = [d1 + d2 + d3]
    ans = model.predict(d)

    return jsonify({"rent": round(ans[0], 2)})

if __name__ == "__main__":
    app.run(debug=True)
