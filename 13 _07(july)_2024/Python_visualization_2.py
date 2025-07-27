import pandas as pd
import matplotlib.pyplot as plt
#
data = pd.read_csv("elections.csv")

party_count = data["Party"].value_counts()
#
print(party_count)
#
plt.figure(figsize=(10,8))
#
plt.pie(party_count, labels = party_count.index, autopct = "%5.3f%%", startangle = 180)
#
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
#
# data = pd.read_csv("elections.csv")
#
# party_count = data["Party"].value_counts()
#
# print(party_count)
#
# plt.figure(figsize=(10,8))
#
# plt.pie(party_count, labels = party_count.index, autopct = "%5.3f%%", startangle = 180)
#
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
#
# data = {'Name':['a','b','c','d'],'Rank':[22,23,24,25]}
# df = pd.DataFrame(data)
# print(df)
# explode_data = (0,0,0.5,0)
# data_color = ('yellow','blue','red','purple')
#
# plt.figure(figsize = (10,8))
# plt.pie(df['Rank'],labels = df['Name'],explode = explode_data,colors = data_color, autopct = '%5.2f%%',startangle = 120)
# plt.title('Welcome To My Emperor')
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# good_data = np.random.normal(size = 100)
# good_df = pd.DataFrame(good_data,columns = ['Good_Data_Performance'])
# print(good_df)
#
# plt.boxplot(good_df['Good_Data_Performance'],vert = False)
# plt.title('Good data')
# plt.show()

'''Any input that has mean() value around 0 and standard deviation around 1 is called normally
    distributed data'''
# good_data = np.random.normal(loc=50,scale = 5,size = 100)
# good_df = pd.DataFrame(good_data,columns = ['Good_Data_Performance'])
#
# print(good_df)
# print(good_df.std())
# print(good_df.mean())
#
#
# plt.boxplot(good_df['Good_Data_Performance'],vert = False)
# plt.title('Good data')
# plt.show()


# import numpy as np
#
# input_data = [ 0.496714, -0.138264,0.64768,1.523030, -0.234153 ]
# final = np.array(input_data)
# print(final)
#
# mean_data =final.mean()
# print(mean_data)
#
# std_data = final.std()
# print(std_data)

# bad_data = np.random.normal(loc=50,scale = 5,size = 100)
# bad_data = np.append(bad_data,[80,90,100])
# bad_df = pd.DataFrame(bad_data,columns = ['Bad_Data_Performance'])
#
# print(bad_df)
# print(bad_df.std())
# print(bad_df.mean())
#
#
# plt.boxplot(bad_df['Bad_Data_Performance'],vert = False)
# plt.title('Bad data')
# plt.show()

'''creating subplot for boxplot'''

# good_data = np.random.normal(size = 100)
# good_df = pd.DataFrame(good_data,columns = ['Good_Data_Performance'])
# print(good_df)
#
# plt.subplot(1,2,1)
# plt.boxplot(good_df['Good_Data_Performance'],vert = False)
# plt.title('Good data')
#
# bad_data = np.random.normal(loc=50,scale = 5,size = 100)
# bad_data = np.append(bad_data,[80,90,100])
# bad_df = pd.DataFrame(bad_data,columns = ['Bad_Data_Performance'])
#
# print(bad_df)
# print(bad_df.std())
# print(bad_df.mean())
#
# plt.subplot(1,2,2)
# plt.boxplot(bad_df['Bad_Data_Performance'],vert = False)
# plt.title('Bad data')
# plt.tight_layout()
# plt.show()

'''--------------------------------------------------------------------------------------------'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


good_data = np.random.normal(size = 100)
good_df = pd.DataFrame(good_data, columns = ['Good_Data_Performance'])
print(good_df)

bad_data = np.random.normal(size = 100)
bad_df = np.append(bad_data, [25,50,76,100])
df = pd.DataFrame(bad_df, columns = ['Bad_Data_Performance'])




plt.subplot(1,2,1)
plt.boxplot(good_df['Good_Data_Performance'],vert = False)
plt.title('good data')

plt.subplot(1,2,2)
plt.boxplot(df['Bad_Data_Performance'],vert = False)
plt.title('bad data')

plt.tight_layout()
plt.show()




