from flask import Flask, render_template, request
from datetime import datetime
from sentiment import predict_bulk, read_csv, create_barplot, create_piechart

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("upload.html")

@app.post("/upload")
def upload_csv():
    now = datetime.now()
    time = now.strftime('%Y_%m_%d_%H_%M_%S')

    tweet_path = f"./static/tweets/{time + '.csv'}"
    bar_path = f"./static/plots/{time + '_bar.jpeg'}"
    pie_path = f"./static/plots/{time + '_pie.jpeg'}"

    file = request.files["file"]
    file.save(tweet_path)

    sents = read_csv(tweet_path)
    scores = predict_bulk(sents)

    create_barplot(scores, filename=bar_path)
    create_piechart(scores, filename=pie_path)

    return render_template("result.html", bar_path=bar_path, pie_path=pie_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
