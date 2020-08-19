# Analysis Log

## Basic analysis

1. Run a 10-fold CV and check whether the result is consistent with the public leaderboard.
2. Use If-Tdf with the same model (Ridge regression) and see if it makes a difference.
3. Try a GBM: learn how to play with the parameters.

## More advanced analysis

1. Try FastText (read the documentation first).
2. Try a model from Huggingface (this should perform best).

## Investigate the difference between CV and LD score

When running 10 fold CV on the training set I get a score of < 0.7, but when I submit to the LD the score is around 0.8. Understand what causes this difference.