from flask import Flask
from flask import render_template
from flask import request
from flask import escape
from flask import Markup
from transformers import pipeline

classifier = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis", top_k=None)

app = Flask(__name__)

def format_classifier_output(testObj):
    test_dict = testObj[0]
    list_emotions = [test_dict[i]["label"] for i in range(0, 3)]
    output = [f'Positive: {test_dict[list_emotions.index("POS")]["score"]}',
    f'Neutral: {test_dict[list_emotions.index("NEU")]["score"]}',
    f'Negative: {test_dict[list_emotions.index("NEG")]["score"]}']
    return output

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        escaped_input = escape(request.form["emailtext"])
        testObj = classifier(escaped_input)
        text = ["No input found"]
        if testObj:
            format_output = format_classifier_output(testObj)
        else:
            format_output = text
        return render_template('results.html', format_output=format_output)
    else:
        return render_template('index.html')