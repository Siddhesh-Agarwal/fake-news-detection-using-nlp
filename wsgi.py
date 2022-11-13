import requests
from flask import Flask, render_template, request
from keras.models import load_model
import pandas as pd

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/news")
def news():
    catagories = [
        "All",
        "Automobile",
        "Business",
        "Entertainment",
        "Hatke",
        "National",
        "Politics",
        "Science",
        "Sports",
        "Startup",
        "Technology",
        "World",
        "Miscellaneous",
    ]
    return render_template("news.html", catagories=catagories)


@app.route("/news/<category>")
def results(category):
    news = requests.get(f"https://inshorts.deta.dev/news?category={category}").json()
    articles = news["data"]
    return render_template("results.html", articles=articles[:24], category=category)


@app.route("/verify")
def verify_page():
    return render_template("verify.html", results="None", show_results=False)


@app.route("/verify-results", methods=["POST"])
def verify():
    from keras.preprocessing.text import one_hot
    from keras_preprocessing.sequence import pad_sequences
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    import re

    title = request.form.get("title").lower()
    author = request.form.get("author").lower()
    text = request.form.get("text").lower()
    features = pd.DataFrame(
        {
            "title": [title],
            "author": [author],
            "text": [text],
        }
    )
    features["total"] = features["title"] + " " + features["author"]
    msg_test = features.copy()

    ps = PorterStemmer()
    corpus_test = [
        ps.stem(word)
        for word in re.sub("[^a-zA-Z]", " ", msg_test["total"][0])
        if not word in stopwords.words("english")
    ]
    onehot_rep_test = [one_hot(words, 5000) for words in corpus_test]
    embedded_docs_test = pad_sequences(onehot_rep_test, padding="pre", maxlen=25)
    model = load_model("./model/model.h5")
    results = model.predict(embedded_docs_test)
    result = not not results[0][0]
    return render_template(
        "verify.html",
        result=result,
        show_results=True,
        color="green" if result else "red",
    )


if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
