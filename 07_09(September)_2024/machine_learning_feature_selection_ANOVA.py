import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    "feature1": [2.5, 3.6, 1.8, 2.0, 3.2],
    "feature2": [1.2, 2.3, 1.5, 2.2, 2.6],
    "feature3": [4.4, 3.2, 2.9, 3.6, 4.0],
    "target": [0, 1, 0, 1, 0]
})

x = df.drop('target', axis = 1)
y = df['target']

chi_square_feature_selection = SelectKBest(f_classif, k = 2)
best_feature = chi_square_feature_selection.fit_transform(x,y)
print(best_feature)

selected_feature = chi_square_feature_selection.get_support(indices = True)
print(selected_feature)

final_df = pd.DataFrame(best_feature, columns = x.columns[selected_feature])
print(final_df)

