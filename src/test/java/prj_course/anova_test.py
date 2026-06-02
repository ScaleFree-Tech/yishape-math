import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anova_util as anova

path = r"E:\项目管理案例分析\data\iris.csv"
df = pd.read_csv(path)


sample1 = df[df['class']=='Iris-setosa']
sample2 = df[df['class']=='Iris-versicolor']
sample = pd.concat([sample1,sample2])
sns.violinplot(data=sample,x="sepalwidth",hue="class")
plt.show()
res = anova.anova_test(df=sample,nominal_feature_name="class",numeric_feature_name="sepalwidth")
print(res)