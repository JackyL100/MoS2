# compare the peak differences for training_data.csv and real_training_data.xlsx
import pandas as pd
import numpy as np
import openpyxl
import scipy.stats as stats
import statsmodels.api as sm
import pylab

real_data = pd.read_excel("real_training_data.xlsx")
my_data = pd.read_csv("training_data.csv")

# remove WS2 samples
real_data.drop([i for i in range(27, len(real_data))], inplace=True) 
locations = real_data["Unnamed: 3"]
flake = real_data["Unnamed: 2"]
sample = [real_data.iloc[0]["Sample"]]

# get corresponding sample number
for i in range(1, len(real_data)):
    if pd.notna(real_data.iloc[i]["Sample"]):
        sample.append(real_data.iloc[i]["Sample"])
    else:
        sample.append(sample[-1])

# combine the columns to get a list of the available files
files = [sample[i] + "/MoS2_" + str(flake.iloc[i]) + "_"+str(locations.iloc[i]) for i in range(len(sample))]
real_data["Image"] = files
# 8 and 9 bc no data, 0 bc my data has outlier at 0
bad_idxs = [0,8,9]
for idx in sorted(bad_idxs, reverse=True):
    del files[idx]

real_peak_diff = np.array([real_data.loc[real_data["Image"].str.contains(files[i])]["Peak difference"].values[0] for i in range(len(files))])
my_peak_diff = np.array([my_data.loc[my_data["Image"].str.contains(files[i] + ".jpg")]["Peak Difference"].values[0] for i in range(len(files))])
# print(real_peak_diff)
# print(my_peak_diff)

# hypothesis testing

# check if variances are equal
S1 = real_peak_diff.std()
S2 = my_peak_diff.std()
n1 = len(real_peak_diff)
n2 = len(my_peak_diff)
f = S1 ** 2 / S2 ** 2
p_value = stats.f.cdf(f, n1 - 1 , n2 - 1)
print(f'p-value for variance: {p_value}')

# check if means are equal
u1 = real_peak_diff.mean()
u2 = my_peak_diff.mean()
Sp2 = ((n1 - 1) * (S1 ** 2) + (n2 - 1) * (S2 ** 2)) / (n1 + n2 - 2)
t = (u1 - u2) / np.sqrt(Sp2 * (1 / n1 + 1 / n2))
p_value = 2 * stats.t.cdf(-1 * abs(t), n1 + n2 - 2)
print(f'p-value for population means: {p_value}')

sm.qqplot(real_peak_diff)
pylab.show()
sm.qqplot(my_peak_diff)
pylab.show()