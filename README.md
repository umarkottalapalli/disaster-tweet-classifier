# Disaster Tweet Classifier

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify tweets as disaster-related or not.

The model analyzes tweet text and predicts whether the tweet describes a real disaster event.

## Features
- Classifies tweets as disaster or non-disaster
- Uses TF-IDF for text vectorization
- Machine Learning model using Naive Bayes
- Accuracy evaluation on test data

## Technologies Used
- Python
- Scikit-learn
- Pandas
- TF-IDF Vectorization
- Naive Bayes Algorithm

## Dataset
The model is trained using a dataset of disaster-related tweets.

Columns in the dataset:
- `text` → Tweet content
- `target` → Label (1 = Disaster, 0 = Not Disaster)

## How It Works
1. Load tweet dataset
2. Clean and preprocess text
3. Convert text into numerical features using TF-IDF
4. Train a Naive Bayes classifier
5. Predict whether a new tweet is disaster-related

## Example

Input:
