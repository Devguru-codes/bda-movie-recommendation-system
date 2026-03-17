# 🎬 Movie Recommendation System using ALS (Apache Spark - Java)

## 📌 Project Overview

This project implements a **Movie Recommendation System** using the **Alternating Least Squares (ALS)** algorithm in **Apache Spark**.
It analyzes user-movie ratings and predicts personalized movie recommendations.

The system is designed as a **Big Data Analytics (BDA)** project using distributed computing.

---

## 🎯 Problem Statement

With a large number of movies available, users find it difficult to discover relevant content.
The goal of this project is to:

* Predict user preferences
* Recommend top movies to users
* Handle large-scale data using distributed processing

---

## 🛠️ Technologies Used

* Java
* Apache Spark (MLlib)
* ALS Algorithm
* WSL (Ubuntu)
* Git & GitHub

---

## 📂 Dataset

The project uses the **MovieLens dataset**, which contains:

* `ratings.csv` → user ratings for movies
* `movies.csv` → movie titles and genres
* `tags.csv`, `links.csv` → additional metadata

### Dataset Size

* ~100,000 ratings
* ~9,000 movies

---

## ⚙️ Project Architecture

```
MovieLens Dataset
        ↓
Data Preprocessing (Spark DataFrame)
        ↓
Train/Test Split
        ↓
ALS Model Training
        ↓
Model Evaluation (RMSE)
        ↓
Top-N Movie Recommendations
```

---

## 🧠 Algorithm Used: ALS

**Alternating Least Squares (ALS)** is a collaborative filtering algorithm that:

* Factorizes the user-item matrix
* Learns hidden features of users and movies
* Predicts missing ratings

---

## 📊 Model Evaluation

The model is evaluated using:

```
RMSE (Root Mean Square Error)
```

Lower RMSE indicates better prediction accuracy.

---

## 🚀 How to Run the Project

### 1️⃣ Navigate to project folder

```bash
cd ~/bda_project
```

### 2️⃣ Compile Java code

```bash
javac -cp "$SPARK_HOME/jars/*" MovieRecommender.java
```

### 3️⃣ Run the program

```bash
java --add-exports java.base/sun.nio.ch=ALL-UNNAMED \
     -cp ".:$SPARK_HOME/jars/*" \
     MovieRecommender
```

---

## 📈 Sample Output

```
RMSE = 0.87

User Recommendations:
User 1 → [3022, 33649, 5013...]
User 2 → [33649, 2936, 167746...]
```

These represent **top recommended movies** for each user.

---

## ⭐ Features

* Distributed data processing using Spark
* Collaborative filtering using ALS
* Model evaluation using RMSE
* Scalable for large datasets
* Java-based implementation

---

## 📌 Applications

* Netflix-style recommendation systems
* E-commerce product recommendations
* Personalized content delivery

---

## ⚠️ Challenges

* Cold start problem (new users/items)
* Data sparsity
* Hyperparameter tuning

---

## 🔮 Future Improvements

* Add movie titles in recommendations
* Build web UI using Streamlit or Java frontend
* Use hybrid recommendation (content + collaborative)
* Real-time recommendation system

---

## 👨‍💻 Author

**Devguru Tiwari**
B.Tech CSE (Big Data Analytics Project)

---

## 📎 GitHub Repository

[Project Link](https://github.com/Devguru-codes/bda-movie-recommendation-system)

---
