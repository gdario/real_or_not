import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

HOME_DIR = Path.home()
BASE_DIR = Path(HOME_DIR/'Projects/real_or_not')
INPUT_DIR = BASE_DIR/'data/input'
OUTPUT_DIR = BASE_DIR/'data/output'

train_df = pd.read_csv(INPUT_DIR/'train.csv')
test_df = pd.read_csv(INPUT_DIR/'test.csv')

train_df['keyword'] = train_df.keyword.fillna('missing')
train_df['str_target'] = train_df.target.apply(lambda x: str(x))
train_df['keyword_target'] = train_df.keyword.str.cat(train_df.str_target)
test_df['keyword'] = test_df.keyword.fillna('missing')

pipe = Pipeline([('cvec', TfidfVectorizer(stop_words='english')),
                 ('ridge', RidgeClassifier())])

skf = StratifiedKFold(n_splits=5)  # , shuffle=True, random_state=42)
cv = skf.split(train_df, train_df.keyword_target)

scores = cross_val_score(
    pipe, train_df['text'], train_df['target'], scoring='f1', cv=cv)
print('score: {:.3f} +/- {:.3f}'.format(scores.mean(), 2*scores.std()))

pipe.fit(train_df['text'], train_df['target'])
y_pred = pipe.predict(test_df['text'])

# sample_submission = pd.read_csv(INPUT_DIR/'sample_submission.csv')
# sample_submission['target'] = y_pred
# sample_submission.to_csv(OUTPUT_DIR/'submission_tfidf_keyword_target.csv',
#                          index=False)
