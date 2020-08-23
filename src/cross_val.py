import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

df = pd.DataFrame({'x': np.arange(12),
                   'g': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                   'y': [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]})

skf = StratifiedKFold()
gen = skf.split(df['x'], df['y'], groups=df['y'])
