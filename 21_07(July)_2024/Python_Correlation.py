import pandas as pd

data = {
    "Age" : [25,30,35,40,45],
    "Height" : [150,168,170,190,180],
    "weight" : [55,65,75,85,95]
}
df = pd.DataFrame(data)

#print(df.corr())

'''----------------------------------Negative Correlation---------------------------------------------------'''
data = {
    "Age" : [25,30,35,40,45],
    "Height" : [150,168,170,190,180],
    "weight" : [55,50,45,40,35]
}
df = pd.DataFrame(data)

specific_corr = df[["Age","weight"]].corr()
print(specific_corr)