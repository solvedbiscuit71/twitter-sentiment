from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.special import softmax

os.makedirs("static/tweets")
os.makedirs("static/plots")

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def predict(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    predict_scores = {}
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        predict_scores[l] = np.round(float(s), 4)
    return predict_scores

def predict_bulk(sents):
    predict_scores = {}
    for i, sent in enumerate(sents):
        scores = predict(sent)
        scores["text"] = sent
        predict_scores[f"Sentence {i+1}"] = scores
    return predict_scores

def create_barplot(sentiment_probabilities, filename="bar.jpeg"):
    plt.figure(figsize=(10, 6))
    sentences = list(sentiment_probabilities.keys())
    positive_probs = [sentiment_probabilities[sentence]['positive'] for sentence in sentences]
    neutral_probs = [sentiment_probabilities[sentence]['neutral'] for sentence in sentences]
    negative_probs = [sentiment_probabilities[sentence]['negative'] for sentence in sentences]

    bar_width = 0.25
    index = range(len(sentences))
    plt.bar(index, positive_probs, width=bar_width, color='lightgreen', label='Positive')
    plt.bar([i + bar_width for i in index], neutral_probs, width=bar_width, color='lightskyblue', label='Neutral')
    plt.bar([i + 2 * bar_width for i in index], negative_probs, width=bar_width, color='lightcoral', label='Negative')

    plt.xlabel('Sentences')
    plt.ylabel('Sentiment Probability')
    plt.title('Sentiment Analysis Results')
    plt.xticks([i + bar_width for i in index], sentences, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)


def create_piechart(sentiment_probabilities, filename="pie.jpeg"):
    plt.figure(figsize=(6, 6))
    sentences = list(sentiment_probabilities.keys())
    positive_probs = [sentiment_probabilities[sentence]['positive'] for sentence in sentences]
    neutral_probs = [sentiment_probabilities[sentence]['neutral'] for sentence in sentences]
    negative_probs = [sentiment_probabilities[sentence]['negative'] for sentence in sentences]

    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [sum(positive_probs), sum(neutral_probs), sum(negative_probs)]
    colors = ['lightgreen', 'lightskyblue', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    plt.axis('equal')
    plt.savefig(filename)

def read_csv(path):
    df = pd.read_csv(path)
    return df["Tweet"].tolist()

# prediction = predict_bulk(read_csv("input2.csv"))
# create_barplot(prediction)
# create_piechart(prediction)
