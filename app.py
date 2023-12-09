from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    variables = [x for x in request.form.values()]
    features = np.array(variables).reshape(1,-1)
    predictions_label = model.predict(features)
    predictions_probability = model.predict_proba(features)

    return render_template('predict.html', prediction_text=f'Predicted Crop is {predictions_label[0]} and with certanity of {max(predictions_probability)}')


if __name__ == '__main__':
    app.run(debug=True)
