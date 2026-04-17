from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import spam
import util

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load training data and train model once at startup
train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
dictionary = spam.create_dictionary(train_messages)
train_matrix = spam.transform_text(train_messages, dictionary)
model = spam.fit_naive_bayes_model(train_matrix, train_labels)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    message = ''
    if request.method == 'POST':
        # Get message from HTML form and turn it into feature vector
        message = request.form.get('message', '')
        matrix = spam.transform_text([message], dictionary)

        # Run Naive Bayes prediction
        label = spam.predict_from_naive_bayes_model(model, matrix)[0]
        prediction = 'Spam' if label == 1 else 'Not Spam'

    return render_template('index.html', message=message, prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict_api():
    # API receives JSON input instead of form data
    data = request.get_json(force=True)
    msg = data.get('message', '')

    # Transform and predict (just like the form route)
    matrix = spam.transform_text([msg], dictionary)
    label = spam.predict_from_naive_bayes_model(model, matrix)[0]

    return jsonify({ 'prediction': 'Spam' if label == 1 else 'Not Spam' })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2025, debug=True)
