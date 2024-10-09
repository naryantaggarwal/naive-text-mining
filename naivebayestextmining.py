import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Define the categories of interest
categories = ['rec.autos', 'rec.motorcycles', 'sci.crypt', 'sci.electronics']

# Load the data using fetch_20newsgroups
newsgroups_data = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups_data.data, newsgroups_data.target, test_size=0.2, random_state=42)

##############################################################################
# Try running with different values of max_df between 0.0 and 1.0
##############################################################################
max_df = 0.9

# Tokenize the text data using CountVectorizer
vectorizer = CountVectorizer(max_df=max_df)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Use the trained classifier to predict the categories of the testing data
y_pred = clf.predict(X_test_vec)

# Evaluate the performance of the classifier using metrics such as accuracy
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy:.5f}")
print("Confusion matrix:")
print(confusion)

# Check what words were algorithmically identified as stop words
print("Stop words:", vectorizer.stop_words_)
