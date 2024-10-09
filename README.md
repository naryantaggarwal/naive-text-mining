# Text Classification Using Naive Bayes

This Python program performs text classification using a **Naive Bayes classifier** on the popular **20 Newsgroups dataset**. It specifically focuses on four categories: `rec.autos`, `rec.motorcycles`, `sci.crypt`, and `sci.electronics`. The main goal is to train the classifier to predict the category of new, unseen text data based on the learned patterns.

### Dataset
The **20 Newsgroups dataset** is a collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups. The subset used in this project includes four categories from the `rec` and `sci` groups:
- rec.autos
- rec.motorcycles
- sci.crypt
- sci.electronics

### Workflow

1. **Data Loading:**
   - The data is loaded using the `fetch_20newsgroups` function, specifying the categories of interest.
   - The data is then split into training and testing sets using `train_test_split`. The testing set is 20% of the total data.

2. **Text Vectorization:**
   - The `CountVectorizer` from `sklearn.feature_extraction.text` is used to tokenize the text data into word counts. 
   - The `max_df` parameter is used to control which words are ignored (those that appear in more than `max_df` fraction of documents). 
   - The vectorizer creates a bag-of-words representation of the text, which is then used for training the model.

3. **Naive Bayes Classifier:**
   - The **Multinomial Naive Bayes (MultinomialNB)** classifier from `sklearn.naive_bayes` is trained on the vectorized training data.
   - The model is trained to classify text documents into one of the four categories.

4. **Prediction and Evaluation:**
   - After training, the model is used to predict the categories of the test data.
   - The performance of the classifier is evaluated using **accuracy** and a **confusion matrix**.

5. **Stop Words:**
   - The program outputs the words that were automatically identified as stop words by the `CountVectorizer`. These are words that were ignored during vectorization because they appear too frequently across the documents.

### Installation

Ensure you have the required libraries installed before running the program:

```bash
pip install scikit-learn pandas
```

### How to Run

1. Load the script in a Python environment or Jupyter notebook.
2. Execute the script to load the dataset, train the model, and evaluate its performance.
3. Adjust the `max_df` value to observe its impact on the model's performance. `max_df` controls the upper bound for term frequency, helping to ignore very common words.

### Example Output

```bash
Accuracy: 0.92584
Confusion matrix:
[[165   4   2   2]
 [  3 161   1   2]
 [  3   0 159   4]
 [  1   2   6 165]]
Stop words: {'to', 'from', 'subject', 'edu', 'com'}
```

### Parameters

- **`max_df`**: A floating-point value between 0.0 and 1.0. This parameter allows control over how frequent words should be handled. If a word appears in more than `max_df` fraction of the documents, it is ignored as it might be too common to help classification.

### Dependencies

- `pandas`: For data manipulation.
- `scikit-learn`: For machine learning algorithms, data preprocessing, and evaluation metrics.

### License

This project is open-source and available under the MIT License.
