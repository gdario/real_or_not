from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE_DIR = Path('/home/giovenko/Projects/real_or_not')
INPUT_DIR = BASE_DIR/'data/input'
OUTPUT_DIR = BASE_DIR/'data/output'

train_df = pd.read_csv(INPUT_DIR/'train.csv')
test_df = pd.read_csv(INPUT_DIR/'test.csv')

train_df['keyword'] = train_df.keyword.fillna('missing')
train_df['str_target'] = train_df['target'].apply(lambda x: str(x))
train_df['keyword_target'] = train_df.keyword.str.cat(train_df.str_target)


pipe = Pipeline([
    ('cvec', CountVectorizer()),
    ('ridge', RidgeClassifier())
])

# CV iterator
skf = StratifiedKFold(n_splits=5)
# cv = skf.split(train_df, train_df.keyword_target)
cv = skf.split(train_df, train_df.keyword)

scores = cross_val_score(pipe, train_df['text'], train_df["target"],
                         cv=cv, scoring="f1")

# We reuse the sample submission replacing our own predictions
sample_submission = pd.read_csv(INPUT_DIR/'sample_submission.csv')
# sample_submission["target"] = clf.predict(test_vectors)
# sample_submission.to_csv(OUTPUT_DIR/'submission.csv', index=False)
