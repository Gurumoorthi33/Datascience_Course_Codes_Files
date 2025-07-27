'''If axis = 1, then it column wise operation(based on column name)
If axis = 2 , then row-wise operation(based on index label).'''

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    "feature1": ["a","b", "c", "a","b"],
    "feature2": ["x","y","x","x","y"],
    "feature3": ["p", "q", "p", 'q','p'],
    "target": [0, 1, 0, 1, 0]
})

'''Below line is not possible because we can't do the label encoding for more than one column at once'''
# l_encoder = LabelEncoder()
# encoded_data = l_encoder.fit_transform(df[["feature1", "feature2", "feature3"]])
# print(encoded_data)

'''This uses the method called apply to apply functionality to entire dataframe '''
l_encoder = LabelEncoder()
encoded_df = df[["feature1", "feature2", "feature3"]].apply(l_encoder.fit_transform)
print(encoded_df)
'''------------------------------------------------------------------------------------------------------'''
# ohe = OneHotEncoder(sparse=False)
# encoded_array = ohe.fit_transform(df[["feature1", "feature2", "feature3"]])
# print(encoded_array)

'''------------------------------------------------------------------------------------------------------'''

x = encoded_df
y = df['target']

chi_square_feature_selection = SelectKBest(chi2, k = 2)
best_feature =chi_square_feature_selection.fit_transform(x,y)
print(best_feature)

selected_feature = chi_square_feature_selection.get_support(indices = True)
print(selected_feature)

final_df = pd.DataFrame(best_feature, columns = x.columns[selected_feature])
print(final_df)

