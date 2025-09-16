
# 🎬 Movie Recommendation System

A **Movie Recommendation System** built with **Python, scikit-learn, and Streamlit**, using the **MovieLens dataset**.  
This mini-project demonstrates **Content-Based** and **Collaborative Filtering** approaches, deployed with **Streamlit Community Cloud**.

---

## 📌 Features
- 🔍 **Content-Based Filtering** – recommends movies similar to a chosen one (using TF-IDF + cosine similarity).  
- 👥 **Collaborative Filtering** – recommends movies based on user ratings (using TruncatedSVD + Nearest Neighbors).  
- ⭐ **Popularity-Based** – fallback recommendations based on most-rated movies.  
- 🎨 **Interactive Streamlit App** – simple web UI for exploration.  

---

## 📂 Dataset
We use the [MovieLens](https://grouplens.org/datasets/movielens/) dataset:

- `movies.csv` → contains movie titles & genres  
- `ratings.csv` → contains user ratings  

For quick testing, download **MovieLens Latest Small (100k ratings)**:  
👉 [Download Here](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

Place the files inside the `data/` folder:

```

movie-recommender/
├─ data/
│   ├─ movies.csv
│   └─ ratings.csv
├─ src/
│   ├─ recommender.py
│   └─ utils.py
├─ app.py
├─ requirements.txt
└─ README.md

````

---

## ⚙️ Installation & Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/sreyadevarapalli/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
````

Run locally:

```bash
streamlit run app.py
```

---

## 🚀 Deployment (Streamlit Community Cloud)

1. Push this project to GitHub ✅
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Click **Create App** → Connect your repo → Select `app.py` as entrypoint
4. Deploy → Your app will be live at:

   ```
   https://<your-app-name>.streamlit.app
   ```

---

## 📊 Tech Stack

* **Python 3**
* **pandas, numpy** – data handling
* **scikit-learn** – ML models (TF-IDF, SVD, NearestNeighbors)
* **Streamlit** – interactive web app

---

## 🔮 Future Improvements

* ✅ Add **Hybrid Recommender** (combine content & collaborative).
* 🎭 Show **movie posters & metadata** (via TMDB API).
* 📈 Evaluate with **Precision\@K / Recall\@K**.
* 💾 Use **larger MovieLens datasets (1M/20M)** for better accuracy.

---

## 🙌 Acknowledgements

* Dataset: [MovieLens](https://grouplens.org/datasets/movielens/)
* Built with ❤️ by [Sreya Devarapalli](https://github.com/sreyadevarapalli)


