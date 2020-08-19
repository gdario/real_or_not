from pathlib import Path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection
from sklearn import preprocessing

BASE_DIR = Path('/Users/dariog/Learning/Kaggle/real_or_not')
INPUT_DIR = BASE_DIR/'data/input'
OUTPUT_DIR = BASE_DIR/'data/output'

train_df = pd.read_csv(INPUT_DIR/'train.csv')
test_df = pd.read_csv(INPUT_DIR/'test.csv')

# Preprocessing
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df["text"])

# Classifier
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"],
                                         cv=3, scoring="f1")
clf.fit(train_vectors, train_df["target"])

# We reuse the sample submission replacing our own predictions
sample_submission = pd.read_csv(INPUT_DIR/'sample_submission.csv')
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.to_csv(OUTPUT_DIR/'submission.csv', index=False)