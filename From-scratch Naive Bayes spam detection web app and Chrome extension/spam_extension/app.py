import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import util
import spam

# Load dataset and train Naive Bayes model once at startup
train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
train_labels = np.array(train_labels)

dictionary = spam.create_dictionary(train_messages)
train_matrix = spam.transform_text(train_messages, dictionary)
nb_model = spam.fit_naive_bayes_model(train_matrix, train_labels)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # API receives JSON input containing the message to classify
    data = request.get_json(force=True)
    message = data.get('message', '')

    # Convert message to feature vector and run Naive Bayes prediction
    message_matrix = spam.transform_text([message], dictionary)
    pred = spam.predict_from_naive_bayes_model(nb_model, message_matrix)

    label = 'Spam' if pred[0] == 1 else 'Not Spam'
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2026, debug=True)
