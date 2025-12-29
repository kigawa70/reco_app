from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# CSV 読み込み
movies = pd.read_csv("movies_100k.csv", sep="|", encoding="latin-1")
movies = movies.rename(columns={"movie_id": "movieId", "movie_title": "title"})

ratings = pd.read_csv("ratings_100k.csv")
ratings = ratings.rename(columns={"movie_id": "movieId", "user_id": "userId"})

# 評価行列（映画 × ユーザ）
movie_user = ratings.pivot_table(
    index="movieId",
    columns="userId",
    values="rating",
    fill_value=0
)

# コサイン類似度
matrix = movie_user.values
norm = np.linalg.norm(matrix, axis=1)
norm[norm == 0] = 1e-9
cos_sim = matrix @ matrix.T / (norm[:, None] * norm[None, :])

movie_index = {m: i for i, m in enumerate(movie_user.index)}
index_movie = {i: m for m, i in movie_index.items()}


@app.route("/")
def index():
    return render_template("index.html", movies=movies.head(300))


@app.route("/recommend", methods=["POST"])
def recommend():
    selected = request.form.getlist("movie")

    # -------------------
    # 未選択の場合
    # -------------------
    if len(selected) == 0:
        avg = ratings.groupby("movieId")["rating"].mean().reset_index()
        result = avg.merge(movies, on="movieId") \
                    .sort_values("rating", ascending=False) \
                    .head(5)

        return render_template(
            "recommend.html",
            movies=result["title"].tolist()
        )

    # -------------------
    # 選択した場合（協調フィルタリング）
    # -------------------
    score = np.zeros(len(movie_user))

    for m in selected:
        mid = int(m)
        if mid in movie_index:
            score += cos_sim[movie_index[mid]]

    # 選択済み映画は除外
    for m in selected:
        if int(m) in movie_index:
            score[movie_index[int(m)]] = -1

    top_index = np.argsort(-score)[:5]
    top_movie_ids = [index_movie[i] for i in top_index]

    result = movies[movies["movieId"].isin(top_movie_ids)]["title"].tolist()

    return render_template("recommend.html", movies=result)


if __name__ == "__main__":
    app.run(debug=True)
