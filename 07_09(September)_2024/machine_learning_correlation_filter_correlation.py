import pandas as pd
import numpy as np

# df = pd.DataFrame({"f1": [1,2,3,4,5],
#                    "f2": [2,4,6,8,10],
#                    "output": [0,1,1,0,1]})
#
# correlation_finder = df.corr()
# print(correlation_finder)
#
# threshold_value = 0.5
#
# target_correlation = correlation_finder["output"].drop("output")
# print(target_correlation)
#
# selected_feature  = target_correlation[abs(target_correlation) > threshold_value]
# print(selected_feature)

df = pd.DataFrame({"f1": [1, 2, 3, 4, 5],
                   "f2": [2, 4, 6, 8, 10],
                   "f3": [-5, -5, 10, 31, 42],
                   "output": [10, 21, 31, 40, 20]})

correlation_finder = df.corr()
print(correlation_finder)

threshold_value = 0.5

target_correlation = correlation_finder["output"].drop("output")
print(target_correlation)

selected_feature  = target_correlation[(target_correlation) > threshold_value].index.tolist()
print(selected_feature)
print(type(selected_feature))

final_df = df[selected_feature + ["output"]]
print(final_df)
