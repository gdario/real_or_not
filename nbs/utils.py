import re
import pandas as pd
from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_multiple_whitespaces


def process_text(txt):
    txt = re.sub('@[^ ]+', '', txt)
    txt = re.sub('https?[^ ]+', '', txt)
    txt = txt.replace('#', '')
    txt = re.sub('^ ', '', txt)
    txt = strip_non_alphanum(txt)
    txt = strip_multiple_whitespaces(txt)
    return txt.lower()


def clean_df(df):
    df['keyword'] = df.keyword.fillna('missing')
    if 'target' in df.columns:
        df['str_target'] = df.target.apply(lambda x: str(x))
        df['keyword_target'] = df.keyword.str.cat(df.str_target)
    df['clean_text'] = df.text.apply(process_text)
    return df


if __name__ == '__main__':
    train_df = pd.read_csv('data/input/train.csv')
    test_df = pd.read_csv('data/input/test.csv')
    for k, v in {'train': train_df, 'test': test_df}.items():
        cleaned = clean_df(v)
        out_name = 'data/output/cleaned_' + k + '_df.csv'
        print(out_name)
        cleaned.to_csv(out_name)
