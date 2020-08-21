from pathlib import Path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE_DIR = Path('/Users/dariog/Learning/Kaggle/real_or_not')
INPUT_DIR = BASE_DIR/'data/input'
OUTPUT_DIR = BASE_DIR/'data/output'

train_df = pd.read_csv(INPUT_DIR/'train.csv')
test_df = pd.read_csv(INPUT_DIR/'test.csv')

pipe = Pipeline([
    ('cvec', CountVectorizer()),
    ('ridge', RidgeClassifier())
])

# classifier
scores = cross_val_score(pipe, train_df['text'], train_df["target"],
                         cv=3, scoring="f1")

kfold = StratifiedKFold()

# We reuse the sample submission replacing our own predictions
sample_submission = pd.read_csv(INPUT_DIR/'sample_submission.csv')
# sample_submission["target"] = clf.predict(test_vectors)
# sample_submission.to_csv(OUTPUT_DIR/'submission.csv', index=False)
