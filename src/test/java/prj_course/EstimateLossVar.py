
import scipy.stats as stats
import numpy as np
# 市场调查出的风险损失
losses = np.array([4.0,5.8,3.2,5.6,9.4,5.6])
# 样本均值
x_bar = losses.mean()
# 样本标准差
s2 = losses.var(ddof = 1)
# 样本量
n = len(losses)
# 标准chi2分布中95%置信度情况下的上下界
chi2_lower = stats.chi2.ppf(0.025,n-1)
chi2_upper = stats.chi2.ppf(0.975,n-1)
# 转换回原问题的上下界
sigma2_lower = (n-1)*s2/chi2_upper
sigma2_upper = (n-1)*s2/chi2_lower
sigma_lower = sigma2_lower**0.5
sigma_upper = sigma2_upper**0.5
print("estimated sigma interval:",sigma_lower,sigma_upper)