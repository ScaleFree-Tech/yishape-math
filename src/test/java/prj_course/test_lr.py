import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import UestcLinearRegression as lr

path = r"E:\项目管理案例分析\data\boston_housing.csv"
df = pd.read_csv(path)

feature = df.iloc[:,0:-1]
labels = df.iloc[:,-1]

feature = feature.apply(lambda s: (s - s.mean()) / s.std())

print(feature)
print(labels)

lr_model = lr.UestcLinearRegression()
lr_model.fit(feature.values,labels.values)

residuals = lr_model.residuals
r2 = lr_model.r2
alpha = lr_model.alpha
beta = lr_model.beta
print(r2,alpha,beta)
