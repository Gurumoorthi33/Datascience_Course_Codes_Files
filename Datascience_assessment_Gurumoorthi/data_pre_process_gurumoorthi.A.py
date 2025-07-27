import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

##a = pd.read_csv('games_gurumoorthi.csv')
##print(a)
##print(a.head())
##print(a.tail())
##print(a['title'])
##print(a['title'].head())
##print(a[['critic_score','title']].head())
##print(a.columns)
##print(a.index)
##print(a.values)
##print(pd.DataFrame(a))
##b = {'name':['GTA','NFS','ROBLOX'],'date':[2002,2005,2016]}
##c = pd.DataFrame(b)
##print(c)
##c.to_csv('Gamespot.csv')
##print(a.ndim)
##print(a.shape)
##print(a.info())
##print(a.isnull().sum().sum())
##print(a.describe())
##print(a.iloc[1:3,6])
##print(a.loc[1:3,['critic_score','jp_sales']])
##print(a.backfill)
##print(a.isnull())
##a.fillna(550, inplace = True)
##print(a)
##
##c = pd.read_csv('games_gurumoorthi.csv')
##c['last_update'].fillna(19_7_2024,inplace = True)
##print(c.head(5))
##
##d = pd.read_csv('games_gurumoorthi.csv')
##d['img'] = SimpleImputer(strategy = 'mean').fit_transform(d[['critic_score']])
##d['title'] = SimpleImputer(strategy = 'median').fit_transform(d[['jp_sales']])
##d.fillna({'critic_score':450,'last_update':250},inplace = True)
##print(d)
##
##f = pd.read_csv('games_gurumoorthi.csv')
##f.fillna({'jp_sales':f['critic_score'].median(),'other_sales':f['critic_score'].mean()},inplace = True)
##print(f[['jp_sales','other_sales']])
##
##n = pd.read_csv('games_gurumoorthi.csv')
##n.ffill(inplace = True)
##print(n[['critic_score','other_sales']])
##
##m = pd.read_csv('games_gurumoorthi.csv')
##m.bfill(inplace = True)header = 
##print(m[['jp_sales','other_sales']])
##
##v = pd.read_csv('games_gurumoorthi.csv')
##v.dropna(inplace = True)
##print(v[['critic_score','jp_sales']])

a = pd.read_csv('games_gurumoorthi.csv')
df  = pd.DataFrame(a)
print(df)
print(df.ndim)
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum().sum())
print(df.loc[1:3,['genre','console']])

##finding_null = df.isnull()
##print(finding_null.head(20))
##print(df.fillna(1000))
##(df['last_update'].fillna(5000, inplace  = True))
##print(df)
##
##df.fillna({'genre':df['other_sales'].mean(), 'console':df['other_sales'].median()}, inplace = True)
##print(df)

b = SimpleImputer(strategy = 'most_frequent')
df['last_update'] = b.fit_transform(df[['last_update']])
print(df)

df.dropna(inplace = True)
print(df) 










