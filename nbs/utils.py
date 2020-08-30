import re
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


def add_keyword_target(df):
    df['keyword2'] = df.keyword.fillna('missing')
    df['str_target'] = df.target.apply(lambda x: str(x))
    df['keyword_target'] = df.keyword2.str.cat(df.str_target)
    return df.drop('keyword2', axis=1)


def paste_keyword(df):
    df['keyword3'] = df.keyword.fillna('').str.replace('%20', ' ')
    df['text'] = df.keyword3.str.cat(df.text, sep=' ')
    return df.drop('keyword3', axis=1)


def clean_df(df, keep_unique=True, paste_kw=True, add_kw_tgt=True):
    if add_kw_tgt:
        df = add_keyword_target(df)
    if paste_kw:
        df = paste_keyword(df)
    df['clean_text'] = df.text.apply(process_text)
    if keep_unique:
        df = df.drop(['id', 'location', 'text'], axis=1)
        if 'target' in df.columns:
            df = df.drop(['str_target'], axis=1)
        df = df[~df.duplicated()]
    return df
