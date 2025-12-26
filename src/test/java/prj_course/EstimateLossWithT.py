
import scipy.stats as stats
import numpy as np
# 市场调查出的风险损失
losses = np.array([4.0,5.8,3.2,5.6,9.4,5.6])
# 样本均值
x_bar = losses.mean()
# 样本标准差
s = losses.std(ddof = 1)
# 样本量
n = len(losses)
# 标准t分布中95%置信度情况下的上下界
t_lower = stats.t.ppf(0.025,n-1)
t_upper = stats.t.ppf(0.975,n-1)
# 转换回原问题的上下界
u_lower = x_bar - t_upper*s/(n**0.5)
u_upper = x_bar - t_lower*s/(n**0.5)
print("estimated interval:",u_lower,u_upper)

