from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


def ohe(location):
    locs = ["chicago", "new_york", "texas"]
    return [1 if location == loc else 0 for loc in locs]


def log_transform(val):
    return np.log(val) if val > 0 else 0


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    bed = float(request.form.get("bed"))
    bath = float(request.form.get("bath"))
    sqft = float(request.form.get("sqft"))
    location = request.form.get("location")

    ohe_loc = ohe(location)

    input_data = np.array([[
        log_transform(bed),
        log_transform(bath),
        sqft,
        *ohe_loc
    ]])

    prediction = model.predict(input_data)[0]
    return render_template(
        "index.html", prediction=f"Predicted Price: ${prediction:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)
