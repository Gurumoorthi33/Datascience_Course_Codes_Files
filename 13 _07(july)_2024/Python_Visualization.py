import matplotlib
import seaborn as ss
import pandas as pd
import matplotlib.pyplot as plt

# x = [1,2,3,4,5]
# y = [2,4,6,8,10]
#
# plt.scatter(x,y)
# plt.show()

# x = [1,2,3,4,5]
# y = [2,4,6,8,10]
#
# plt.plot(x,y)
# plt.show()

# x = [1,2,3,4,5]
# y = ['g','u','r','u','a']
#
# plt.plot(x,y)
# plt.show()

# x = ['a','b','c','d','e']
#
# plt.plot(x)
# plt.show()

# x = [1,2,3,4,5]
# y = [2,4,6,8,10]
# df = pd.DataFrame({'in':x,'out':y})
# plt.plot(df['in'],df['out'],color = 'yellow',marker = 'x') #marker = 'o' or 'x'
# plt.show()

x = [1,2,3,4,5]  # it works on the tuples also
y = [2,4,6,8,10]

df = pd.DataFrame({'in':x,'out':y})
ss.scatterplot(x= 'in',y = 'out',data = df)
plt.xlabel('x data')
plt.ylabel('y data')
plt.grid(True)
plt.show()

'''---------------------------------------------------------------------------------------------'''


