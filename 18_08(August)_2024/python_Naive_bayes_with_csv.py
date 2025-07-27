import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("adult.csv")
print(data.columns)
print(data.shape)

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship','race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
# my_columns = col_names.split() ---> It splits the spaced words into the separate into list.
data.columns = col_names

# find categorical variables
'''------------------------------------Method-1--------------------------------------------------'''
categorical = [var for var in data.columns if data[var].dtype == 'O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)

# step by step.
'''-------------------------------------Method-2-----------------------------------------------'''
# categorical = []  # Initialize an empty list to store names of categorical columns
# for var in data.columns:  # Loop over each column name in the DataFrame 'df'
#     if data[var].dtype == 'O':  # Check if the column’s data type is 'object' (typically strings or categorical)
#         categorical.append(var)  # Add it to the list if it's categorical

print(data[categorical].head())

null_cat = data[categorical].isnull().sum()
print(null_cat)

# view frequency counts of values in categorical variables

for var in categorical:
    print(data[var].value_counts())

# view frequency distribution of categorical variables

for var in categorical:
    print(data[var].value_counts() / np.float64(len(data)))

# check labels in workclass variable

unique_category_count = data.workclass.unique()
print(unique_category_count)

# check frequency distribution of values in workclass variable

freq_check_work = data.workclass.value_counts()
print(freq_check_work)
index_check = freq_check_work.index
print(index_check)

#Pie-Chart
plt.figure(figsize = (10,8))
plt.pie(freq_check_work, labels = freq_check_work.index, autopct = "%5.2f%%", startangle = 180)
plt.grid(True)
plt.legend()
plt.show()

data["workclass"] = data["workclass"].replace(" ?", np.nan)  # ------->  Method-1
# data.replace({"workclass": " ?"}, np nan, inplace = True) # -------->  Method-2
print(data["workclass"].value_counts())

unique_occupation = data.occupation.unique()
print(unique_occupation)

nunique_occupation = data.occupation.nunique()
print(nunique_occupation)

freq_check_occupation = data.occupation.value_counts()
print(freq_check_occupation)

data["occupation"] = data["occupation"].replace(" ?", np.nan)
print(data["occupation"].value_counts())

unique_native_country = data.native_country.unique()
print(unique_native_country)

nunique_native_country = data.native_country.nunique()
print(nunique_native_country)

data["native_country"] = data["native_country"].replace(" ?", np.nan)
print(data.native_country.value_counts())

"""continueeeeeeeeeee------------"""







