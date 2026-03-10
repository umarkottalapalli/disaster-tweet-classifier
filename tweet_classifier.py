import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("train.csv")

# Input and output
X = data["text"].fillna("")
y = data["target"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")
X_vector = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vector, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Test custom tweet
tweet = input("Enter a tweet: ")
tweet_vec = vectorizer.transform([tweet])
prediction = model.predict(tweet_vec)

if prediction[0] == 1:
    print("This tweet is about a disaster.")
else:
    print("This tweet is NOT about a disaster.")