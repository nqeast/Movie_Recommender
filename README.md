# **Movie Recommendation System**

A Python-based movie recommendation system built using collaborative filtering with the **MovieLens 100k Dataset**. This project demonstrates how to implement a recommendation engine using the **Surprise library**, leveraging matrix factorization techniques such as **SVD (Singular Value Decomposition)**.

---

## **Features**
- Collaborative filtering-based movie recommendation engine.
- Predict ratings for movies a user hasn’t watched.
- Ranked recommendations tailored to individual users.
- Preprocessing of the MovieLens 100k dataset with `pandas`.
- Evaluation of recommendation accuracy using metrics such as **RMSE** and **MAE**.

---

## **Tech Stack**
- **Programming Language**: Python
- **Libraries**:
  - [Pandas](https://pandas.pydata.org/) for data manipulation.
  - [Surprise](https://surprise.readthedocs.io/) for recommendation algorithms.
  - [Matplotlib](https://matplotlib.org/) for data visualization.
- **Dataset**: [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/)

---

## **Usage**
- **Generate Recommendations**
- The system predicts movie ratings for a specific user. Follow these steps:

- Train the SVD model on the dataset.
- Find movies the user hasn’t rated.
- Predict ratings for these unrated movies.
- Display top recommendations based on predicted ratings.

## **Future Enhancements**
- Implement content-based filtering using movie metadata.
- Add a hybrid recommendation system combining collaborative and content-based approaches.
- Deploy the system with a user-friendly web interface using Streamlit
