import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

# for reading the csv file we are using the read csv code

df = pd.read_csv("../Gurumoorthi_A_Project/HousePricePrediction.csv")

# for checking the files have been fetching the first five data correctly
df.head(5)
df.tail(5)
print(df)

#  SAINTY CHECK OF THE DATASET PROVIDED
# getting the overall values of the data set
# as we have imported the data. so the data shape method will show us the dimension of the data set

#  **** Data Preprocessing **** first method

# information of the dataset provided

df.info()

# Finding Null values
# There is no Null values in the dataset
df.isnull().sum()

# finding duplicates
df.duplicated().sum()

# To check the descriptive overview of the dataset.
df.describe()

# **** Data preprocessing second method *******

obj=(df.dtypes=='object')
object_cols=list(obj [obj].index)
print("Categorical variables:",len(object_cols))

int=(df.dtypes=='int')
num_cols=list(int [int].index)
print("Integer variables:",len(num_cols))

fl=(df.dtypes=='float')
fl_cols=list(fl[fl].index)
print("float variables:",len(fl_cols))


# ***** EDA ( Expolratory Data Analysis ) *****

#  it helps to deep analysis of data so to discover different patterns and spot anomalies

numerical_dataset = df.select_dtypes(include=['number'])
plt.figure(figsize=(12,6))
sns.heatmap(numerical_dataset.corr(),cmap='BrBG',fmt='.2f',linewidths=2,annot=True)
plt.show()

# to analyze the different categorical features we are drawing the barplot

unique_values=[]
for col in object_cols:
 unique_values.append(df[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No.unique values of categorical features')
sns.barplot(x=object_cols,y=unique_values)
plt.show()

# the plt shows the exterior1st has around 16 unique categories and other features
# have around 6 unique categories


# ** To findout the actual count of each category we can plot the bargraph
# of each four features separtely ***

plt.figure(figsize=(18,36))
plt.title('Categorical Features Distrubution')
index=1
for col in object_cols:
    y=df[col].value_counts()
    plt.plot(11,4,index)
    sns.barplot(x=list(y.index), y=y)
    index+=1
    plt.show()


#     ****** Data Cleaning ***

#  to improvise the data or remove incorrect and  corrupted data

# as Id column will not be of any use so we can drop it

df.drop(['Id'],axis=1,inplace=True)

#  replacing salesprice empty values with mean values to make the data distrubution symmetric

df['SalePrice']=df['SalePrice'].fillna(df['SalePrice'].mean())

# drop records with null values

new_df = df.dropna()

# checking features which have null values in the new dataframe

new_df.isnull().sum()

print(new_df)

# *** One hot encoder *****

#  By using One hot encoder  we can easily convert object into int
#  for that we have to collect all the feature which have the object data type
s=(new_df.dtypes == 'object')
object_cols=list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No.of categorical features:',len(object_cols))

# applying One Hot encoder
OH_encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_df[object_cols]))
OH_cols.index = new_df.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_df.drop(object_cols,axis=1)
df_final=pd.concat([df_final,OH_cols],axis=1)

print(df_final)


# **** Spliting dataset into Training and Testing *****

X = df_final.drop(['SalePrice'] , axis =1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y,train_size=0.8,test_size=0.2,random_state=0)


# **** Training the Model for accuracy ***
# **** we are using Random Forest regressor and linear regressor ****

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train,Y_train)
Y_pred = model_RFR.predict(X_valid)
print("Random forest regressor ")
print(mean_absolute_percentage_error(Y_valid, Y_pred))


# **** Linear Regression *******

model_LR = LinearRegression()
model_LR.fit(X_train,Y_train)
Y_pred = (model_LR.predict(X_valid))
print("Linear Regression")
print(mean_absolute_percentage_error(Y_valid, Y_pred))


#  conclusion

#  Between the two models linear aggression gives 0.18 approx than the random tree regressor








