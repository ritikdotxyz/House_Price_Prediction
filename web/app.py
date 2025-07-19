from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    bed = float(request.form['bed'])
    bath = float(request.form['bath'])
    sqft = float(request.form['sqft'])

    input_data = np.array([[bed, bath, sqft]])

    prediction = model.predict(input_data)[0]
    return render_template('index.html', prediction=f"Predicted Price: ${prediction:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)
