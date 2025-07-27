import pandas as pd


data = {
    "Age" : [25,30,35,40,45],
    "Height" : [150,168,170,190,180],
    "weight" : [55,65,75,85,95]
}
df = pd.DataFrame(data)
print(df)

total_var = df.var()
print(total_var)

weight_variance = df['weight'].var()
print(weight_variance)