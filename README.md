# Iris Flower Classification using K-Nearest Neighbors (KNN)

A simple machine learning project that classifies Iris flowers into three species
(Setosa, Versicolor, Virginica) using the K-Nearest Neighbors algorithm.

## Overview

This project uses the classic Iris dataset (150 samples, 4 features: sepal length,
sepal width, petal length, petal width) to build and evaluate a KNN classifier.

## What it does

- Loads and explores the Iris dataset
- Scales features using `StandardScaler` (important for distance-based algorithms like KNN)
- Splits data into training (80%) and testing (20%) sets
- Finds the optimal value of `k` by testing k = 1 to 20 and picking the one with
  the lowest error rate (elbow method)
- Trains the final KNN model and evaluates it using accuracy, F1 score, and a
  confusion matrix
- Generates 4 visualizations:
  - Elbow curve for k selection
  - Confusion matrix heatmap
  - Per-class F1 scores
  - Feature space plot (petal length vs. petal width) with correct/incorrect
    predictions highlighted

## Tech stack

- Python 3
- NumPy
- Matplotlib / Seaborn
- scikit-learn

## How to run

```bash
pip install numpy matplotlib seaborn scikit-learn
python project2_knn.py
```

The script prints dataset stats, model performance, and a classification report
to the console, and saves a combined results figure as `project2_results.png`.

## Results

The model typically achieves high accuracy (>95%) since the Iris dataset is
small, clean, and well-separated by class.
