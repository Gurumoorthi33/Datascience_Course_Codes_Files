import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

"""pip install Bayesian-Optimization"""

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import warnings
warnings.simplefilter("ignore", category = FutureWarning)

df = pd.read_csv("flight_delays_train.csv")
print(df)
print(df.info())

dc = pd.read_csv("flight_delays_test.csv")
print(df.head())

# encoder = LabelEncoder()
# df["dep_delayed_15min"] = encoder.fit_transform(df["dep_delayed_15min"])
# print(df["dep_delayed_15min"])

df["dep_delayed_15min"] = df["dep_delayed_15min"].map(df["dep_delayed_15min"].value_counts(normalize = True).to_dict())
print(df["dep_delayed_15min"])

df["hour"] = df["DepTime"] // 100
print((df["hour"]))
# https://www.kaggle.com/code/rohitgr/hyperparameter-tuning-with-bayesian-optimization

df.loc[df['hour'] == 24, 'hour'] = 0
df.loc[df['hour'] == 25, 'hour'] = 1
df['minute'] = df['DepTime'] % 100

dc['hour'] = dc['DepTime'] // 100
dc.loc[dc['hour'] == 24, 'hour'] = 0
dc.loc[dc['hour'] == 25, 'hour'] = 1
dc['minute'] = dc['DepTime'] % 100

# Season
df['summer'] = (df['Month'].isin([6, 7, 8])).astype(np.int32)
df['autumn'] = (df['Month'].isin([9, 10, 11])).astype(np.int32)
df['winter'] = (df['Month'].isin([12, 1, 2])).astype(np.int32)
df['spring'] = (df['Month'].isin([3, 4, 5])).astype(np.int32)

dc['summer'] = (dc['Month'].isin([6, 7, 8])).astype(np.int32)
dc['autumn'] = (dc['Month'].isin([9, 10, 11])).astype(np.int32)
dc['winter'] = (dc['Month'].isin([12, 1, 2])).astype(np.int32)
dc['spring'] = (dc['Month'].isin([3, 4, 5])).astype(np.int32)


# Daytime
df['daytime'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 23], include_lowest=True)
df['daytime'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 23], include_lowest=True)

# Extract the labels
train_y = df.pop('dep_delayed_15min')
train_y = train_y.map({'N': 0, 'Y': 1})

# Concatenate for preprocessing
train_split = df.shape[0]
full_df = pd.concat((df, dc))
full_df['Distance'] = np.log(full_df['Distance'])

# String to numerical
for col in ['Month', 'DayofMonth', 'DayOfWeek']:
    full_df[col] = full_df[col].apply(
        lambda x: x.split('-')[1]).astype(np.int32) - 1

# Label Encoding
for col in ['Origin', 'Dest', 'UniqueCarrier', 'daytime']:
    full_df[col] = pd.factorize(full_df[col])[0]

# Categorical columns
cat_cols = ['Month', 'DayofMonth', 'DayOfWeek', 'Origin', 'Dest',
            'UniqueCarrier', 'hour', 'summer', 'autumn', 'winter', 'spring', 'daytime']

# Converting categorical columns to type 'category' as required by LGBM
for c in cat_cols:
    full_df[c] = full_df[c].astype('category')

# Split into train and test
train_df, test_df = full_df.iloc[:train_split], full_df.iloc[train_split:]
print(train_df.shape, train_y.shape, test_df.shape)



