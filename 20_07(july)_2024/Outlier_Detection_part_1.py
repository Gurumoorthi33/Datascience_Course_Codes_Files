import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

data = np.random.normal(100, 20, 100)

df = pd.DataFrame(data, columns=['values'])
print(df)

q1 = df['values'].quantile(0.25)  #25% data haven
print(q1)
q3 = df['values'].quantile(0.75)
print(q3)
q2 = df['values'].quantile(0.5)
print(q2)

IQR = q3 - q1  # inter quartile range
print(IQR)

lower_bound = q1 - 1.5 * IQR
print(lower_bound)

upper_bound = q3 + 1.5 * IQR
print(upper_bound)

plt.figure(figsize=(10, 6))
box = plt.boxplot(df['values'], vert=False, patch_artist=True)  #patch_artist = color settings true vaa aakanu
plt.title('Testing IQR')

for patch in box['boxes']:
    patch.set_facecolor('red')
    patch.set_edgecolor('yellow')
    patch.set_color("violet")

plt.text(q1, 1.1, f'q1:{q1:.2f}', horizontalalignment='center', color='blue')
plt.text(q3, 1.1, f'q3:{q3:.2f}', horizontalalignment='center', color='blue')

plt.text((q1 + q3) / 2, 1.1, f'IQR : {IQR :.2f}', horizontalalignment='center', color='red')

plt.axvline(x=lower_bound, color='red', linestyle='--')
plt.axvline(x=upper_bound, color='red', linestyle='--')

plt.text(lower_bound, 1.1, f'lower_bound : {lower_bound:.2f}', horizontalalignment='center', color='black')
plt.text(upper_bound, 1.1, f'upper_bound : {upper_bound:.2f}', horizontalalignment='center', color='black')
plt.show()

# f = format string  = to set an variable for the given indicator

# Interview repeated Questons based on the Outlier detection












