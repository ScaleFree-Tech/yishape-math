# 参考答案 / Exercise Answers

> **⚠️ 使用说明**：建议读者先独立完成练习，再对照参考答案验证。如果思路不同但结果正确，不必强求与参考答案一致。
> 
> 参考答案按章节组织，与教材练习题目一一对应。

---


## 1.1 向量 / Vector
> 📍 对应练习：[1.1.Vector.md](Chapter1_LinearAlgebra/1.1.Vector.md)


## 第 1 题：向量范数的几何意义（画图题）

**参考答案**：

在二维平面中，向量 $\mathbf{a} = (3, 4)$ 和 $\mathbf{b} = (-2, 1)$ 的端点标出后：

- $\ell_1$ 范数（曼哈顿距离）：$\|\mathbf{a}\|_1 = |3| + |4| = 7$，是从原点到 $(3,4)$ 沿坐标轴走的总距离
- $\ell_2$ 范数（欧氏距离）：$\|\mathbf{a}\|_2 = \sqrt{3^2 + 4^2} = 5$（著名的 3-4-5 直角三角形）
- $\ell_\infty$ 范数：$\|\mathbf{a}\|_\infty = \max(|3|, |4|) = 4$

**几何直觉**：$\ell_1$ 范数的「等高线」是正方形（旋转 45°）；$\ell_2$ 范数的等高线是同心圆；$\ell_\infty$ 的等高线是轴对齐正方形。

---

## 第 2 题：内积与夹角推导（证明题）

**参考答案**：

由内积定义 $\langle\mathbf{a}, \mathbf{b}\rangle = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$，得：

$$\cos\theta = \frac{\langle\mathbf{a}, \mathbf{b}\rangle}{\|\mathbf{a}\|\|\mathbf{b}\|} = \frac{3\times(-2) + 4\times1}{\sqrt{25}\sqrt{5}} = \frac{-2}{5\sqrt{5}}$$

---

## 第 3 题：向量投影计算（计算题）

**参考答案**：

$\mathbf{a}$ 在 $\mathbf{b}$ 方向上的投影：

$$\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{\langle\mathbf{a},\mathbf{b}\rangle}{\langle\mathbf{b},\mathbf{b}\rangle}\mathbf{b} = \frac{-2}{5}((-2, 1)) = \left(\frac{4}{5}, -\frac{2}{5}\right)$$

---

## 第 4 题：线性相关判断（概念题）

**参考答案**：

向量组 $\{(1,2), (2,4), (3,5)\}$ 线性相关，因为存在不全为零的系数使得 $c_1(1,2) + c_2(2,4) + c_3(3,5) = (0,0)$。

事实上 $(2,4) = 2(1,2)$，即第二个向量是第一个的倍数。


## 4.1 统计推断 / Statistical Inference
> 📍 对应练习：[4.1.md](Chapter4_Statistics/4.1.md)


## 第 1 题：中心极限定理验证（模拟题）

**参考答案**：

```java
import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.linalg.Linalg;

public class CLTVerification {
    public static void main(String[] args) {
        // 重复1000次抽样，每次n=30
        int n = 30, trials = 1000;
        var sampleMeans = Linalg.vector(new double[1000]);
        
        for (int i = 0; i < trials; i++) {
            // 从指数分布抽取30个样本
            var sample = Stats.exponential(2.0, n);
            sampleMeans.set(i, sample.mean());
        }
        
        // CLT预测：样本均值近似 N(μ, σ²/n)
        double theoreticalSE = 2.0 / Math.sqrt(30);
        double empiricalSE = sampleMeans.std();
        
        System.out.printf("理论标准误: %.4f%n", theoreticalSE);
        System.out.printf("经验标准误: %.4f%n", empiricalSE);
        // 两者应接近（约0.36）
    }
}
```

**答案**：理论 SE ≈ 0.365，模拟结果应接近该值。样本量越大，吻合越好。

---

## 第 2 题：标准误差计算（计算题）

**参考答案**：

已知总体标准差 $\sigma = 15$，样本量 $n = 50$：

$$\text{SE} = \frac{\sigma}{\sqrt{n}} = \frac{15}{\sqrt{50}} \approx 2.12$$

---

## 第 3 题：LLN vs CLT（概念题）

**参考答案**：

| | 大数定律（LLN） | 中心极限定理（CLT） |
|--|--|--|
| 描述 | 样本均值趋近于总体均值 | 样本均值趋近于正态分布 |
| 关注点 | 收敛性（趋近哪个值） | 分布形状（趋近什么分布） |
| 条件 | 独立同分布，期望存在 | 独立同分布，方差存在 |

LLN 告诉我们「会收敛到哪里」，CLT 告诉我们「收敛的速度和分布形状」。


## 4.2 概率分布族 / Probability Distribution Families
> 📍 对应练习：[4.2.md](Chapter4_Statistics/4.2.md)


## 第 1 题：置信区间的频率学派解释（概念题）

**参考答案**：

置信水平 95% 的含义：**如果重复抽样 100 次，每次构造一个 95% 置信区间，则约有 95 个区间会包含真实参数值。**

⚠️ **常见误解**：不能说「真实参数有 95% 概率落在这个特定区间内」——真实参数是固定值，要么在区间内（概率1）要么不在（概率0）。

---

## 第 2 题：MLE 推导：指数分布参数（推导题）

**参考答案**：

设样本 $x_1, ..., x_n \sim \text{Exp}(\lambda)$，似然函数：

$$\mathcal{L}(\lambda) = \lambda^n e^{-\lambda \sum x_i}$$

对数似然：$\ell(\lambda) = n\log\lambda - \lambda\sum x_i$

求导：$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum x_i = 0$

解得：$\hat{\lambda} = \frac{n}{\sum x_i} = \frac{1}{\bar{x}}$

---

## 第 3 题：MME vs MLE 对比（计算题）

**参考答案**：

| | MLE（最大似然估计） | MME（矩估计） |
|--|--|--|
| 正态分布 $\mu$ | $\bar{x}$ | $\bar{x}$ |
| 正态分布 $\sigma^2$ | $\frac{1}{n}\sum(x_i - \bar{x})^2$（有偏）| $\frac{1}{n}\sum(x_i - \bar{x})^2$（相同）|
| 指数分布 $\lambda$ | $1/\bar{x}$ | $1/\bar{x}$（相同）|

两者在正态 $\sigma^2$ 上的区别在于除以 $n$ 还是 $n-1$（MLE 用 $n$，样本方差用 $n-1$）。


## 4.3 假设检验 / Hypothesis Testing
> 📍 对应练习：[4.3.md](Chapter4_Statistics/4.3.md)


## 第 1 题：p 值的正确理解（选择题）

**参考答案**：

正确答案：**C**（若选 A/B/D 则误解了 p 值的含义）

p 值 = 0.03 的正确解读：
- 如果原假设（H0：硬币公平）为真，则观察到 58 次正面的概率为 3%
- 这是一个「小概率事件」，我们有理由拒绝 H0
- ⚠️ p=0.03 ≠ 「H0 为真的概率是 3%」（那是贝叶斯观点）

常见错误选项：
- A（3%是H0为真的概率）→ 这是对 p 值的常见误解
- B（硬币被动过手脚的概率）→ 需要贝叶斯计算才能得到
- D（3%的概率不是随机波动）→ 需要先验才能量化

---

## 第 2 题：两类错误辨析（计算题）

**参考答案**：

| | H0 为真（硬币公平） | H0 为假（硬币有偏）|
|--|--|--|
| **拒绝 H0** | I 类错误（假阳性）α | 正确（1-β）|
| **不拒绝 H0** | 正确（1-α）| II 类错误（假阴性）β |

$\alpha = P(\text{拒绝}H_0|H_0\text{为真})$，由研究者事先设定（通常 0.05）
$\beta = P(\text{不拒绝}H_0|H_0\text{为假})$，与效应量和样本量相关

---

## 第 3 题：配对 t 检验 vs 独立双样本 t 检验（应用题）

**参考答案**：

使用配对 t 检验（paired t-test）的场景：
- 同一受试者接受前后两次测量（治疗前/后）
- 双胞胎/兄弟姐妹配对研究
- 同一批样品分两种方法测定

关键区别：配对设计消除了个体间差异，检验效能更高。

---

## 第 4 题：Bonferroni 校正（计算题）

**参考答案**：

原始 $\alpha = 0.05$，共进行 $m = 3$ 次两两比较：

$$\alpha_{\text{校正}} = \frac{0.05}{3} \approx 0.0167$$

因此每次比较的显著性阈值变为 0.0167（而非 0.05）。

⚠️ 若不校正：$\binom{3}{2}=3$ 次比较，整体 I 类错误膨胀至约 $1-(1-0.05)^3 = 14.3\%$。


## 4.5 贝叶斯统计 / Bayesian Statistics
> 📍 对应练习：[4.5.md](Chapter4_Statistics/4.5.md)


## 第 1 题：贝叶斯定理实战：医学检测（应用题）

**参考答案**：

已知：
- $P(\text{阳性}|\text{患病}) = 99.9%$（灵敏度）
- $P(\text{阳性}|\text{健康}) = 1\%$（假阳性率）
- $P(\text{患病}) = 0.1\%$（患病率）

求 $P(\text{患病}|\text{阳性})$：

$$P(\text{患病}|\text{阳性}) = \frac{0.999 \times 0.001}{0.999 \times 0.001 + 0.01 \times 0.999} \approx 9.1\%$$

⚠️ **直觉陷阱**：尽管检测很准确，但阳性结果更可能是假阳性（因为健康人群远大于患病人群）。

---

## 第 2 题：共轭先验的推导（计算题）

**参考答案**：

以 Beta-Binomial 为例：

先验：$\theta \sim \text{Beta}(\alpha, \beta)$
似然：$X|\theta \sim \text{Bin}(n, \theta)$

后验 $\propto$ 先验 $\times$ 似然：

$$P(\theta|X) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1} \cdot \theta^x(1-\theta)^{n-x} = \theta^{\alpha+x-1}(1-\theta)^{\beta+n-x-1}$$

即后验仍为 $\text{Beta}(\alpha+x, \beta+n-x)$，与先验同分布——共轭性得证。


## 5.2 分类 / Classification
> 📍 对应练习：[5.2. Classification.md](Chapter5_Machine_learning/5.2. Classification.md)


## 第 1 题：Softmax 梯度推导（推导题）

**参考答案**：

Softmax: $P(y=k|\mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$

对数似然 $\ell_k = \log P(y=k) = z_k - \log\sum_j e^{z_j}$

对 $z_j$ 求偏导（分两种情况）：

- 当 $j = k$ 时：$\frac{\partial\ell_k}{\partial z_j} = 1 - \frac{e^{z_j}}{\sum e^{z_j}} = 1 - P_j$
- 当 $j \neq k$ 时：$\frac{\partial\ell_k}{\partial z_j} = -\frac{e^{z_j}}{\sum e^{z_j}} = -P_j$

即：$\frac{\partial\ell_k}{\partial z_j} = \mathbb{1}(j=k) - P_j$，向量化实现时即 `pred - one_hot(label)`。

---

## 第 2 题：类别不平衡下的评估指标（应用题）

**参考答案**：

| 指标 | 公式 | 适用场景 |
|------|------|---------|
| 准确率 Accuracy | $(TP+TN)/(TP+TN+FP+FN)$ | 平衡数据集 |
| 精确率 Precision | $TP/(TP+FP)$ | 成本高的假阳性（如垃圾邮件误判）|
| 召回率 Recall | $TP/(TP+FN)$ | 漏检代价高（如癌症筛查）|
| F1 分数 | $2PR/(P+R)$ | 不平衡数据的综合评估 |
| AUC-ROC | ROC 曲线下面积 | 不关心阈值时的全局评估 |

在 99:1 不平衡数据中，准确率 99% 的分类器可能全是负类预测，实际无价值——此时应看 AUC-ROC 或 F1。

---

## 第 3 题：KNN vs 逻辑回归（概念题）

**参考答案**：

| | KNN | 逻辑回归 |
|--|--|--|
| 学习方式 | 惰性学习（记忆训练集）| 显式学习权重 $\mathbf{w}$ |
| 决策边界 | 非线性（取决于 k 和距离度量）| 线性（通过 sigmoid 变换）|
| 预测速度 | 慢（需遍历全部训练集）| 快（只需计算 $\mathbf{w}^T\mathbf{x}$）|
| 可解释性 | 一般 | 高（系数直接反映特征重要性）|

KNN 在小数据集、高维稀疏数据上表现差；逻辑回归是基线模型，适合特征-标签关系近似线性的场景。

---

## 第 4 题：Pipeline 编码题（实操题）

**参考答案**：

```java
var scaler = new StandardScaler();
var XTrainScaled = scaler.fitTransform(XTrain);
var XTestScaled = scaler.transform(XTest);

var model = ML.logisticRegression()
    .setLambda(0.1)
    .setMaxIter(500)
    .setThreshold(0.5)
    .fit(XTrainScaled, yTrain);

var predictions = model.predict(XTestScaled);
var auc = ML.auc(yTest, predictions);
System.out.printf("Test AUC: %.4f%n", auc);
```

⚠️ 注意：测试集必须用 `transform()`（用训练集的均值/标准差），而非 `fitTransform()`，否则数据泄露。

---

## 第 5 题：Softmax 过拟合（推导题）

**参考答案**：

当类别数 $K$ 很大时，Softmax 容易过拟合——每个类别仅有少量样本。

**解决方案**：
1. **标签平滑**（Label Smoothing）：将 one-hot 标签 $(0,0,...,1,...,0)$ 替换为 $(\epsilon/K, ..., 1-\epsilon+\epsilon/K, ..., \epsilon/K)$
2. **权重正则化**：对权重 $\mathbf{W}$ 加 L2 惩罚 $\lambda\|\mathbf{W}\|_F^2$
3. **数据增强**：同类样本混叠（mixup）

---

## 第 6 题：AUC-ROC 解读（应用题）

**参考答案**：

AUC = 0.85 的含义：**随机抽取一个正样本和一个负样本，分类器给正样本打分高于负样本的概率是 85%。**

- AUC = 0.5：等价于随机猜测
- AUC = 0.85：在大多数阈值下都有较好的区分能力
- AUC = 1.0：完美分类器（实际不存在）

⚠️ 注意：AUC 不受阈值选择影响，但无法反映正负样本不平衡程度——一个 99:1 的数据集，AUC=0.9 可能掩盖了极低的召回率。


## 5.4 降维 / Dimensionality Reduction
> 📍 对应练习：[5.4. Dimension reduction.md](Chapter5_Machine_learning/5.4. Dimension reduction.md)


## 第 1 题：PCA 几何意义（概念题）

**参考答案**：

PCA 找的是使数据投影后方差最大的正交方向：

1. 第一个主成分 $\mathbf{u}_1 = \arg\max_{\|\mathbf{u}\|=1} \text{Var}(\mathbf{X}\mathbf{u})$
2. 第二个主成分 $\mathbf{u}_2 = \arg\max \text{Var}(\mathbf{X}\mathbf{u}_2)$，且 $\mathbf{u}_2 \perp \mathbf{u}_1$

几何解释：把高维数据投影到低维子空间，同时尽可能保留原始数据的信息（方差）。

---

## 第 2 题：SVD→PCA 等价性证明

**参考答案**：

对数据矩阵 $\mathbf{X} \in \mathbb{R}^{n \times p}$ 做 SVD：$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$

则 $\mathbf{X}^T\mathbf{X} = \mathbf{V}\boldsymbol{\Sigma}^2\mathbf{V}^T$，即 $\mathbf{V}$ 的列是 $\mathbf{X}^T\mathbf{X}$ 的特征向量（也是协方差矩阵 $\frac{1}{n}\mathbf{X}^T\mathbf{X}$ 的特征向量）。

数据在前 $k$ 个主成分上的投影：$\mathbf{Z} = \mathbf{X}\mathbf{V}_k = \mathbf{U}\boldsymbol{\Sigma}_k$

因此 SVD 和 PCA 得到完全相同的结果，但 SVD 更数值稳定。

---

## 第 3 题：PCA 实战编码（实操题）

**参考答案**：

```java
// 1. 标准化（PCA必须先标准化）
var scaler = new StandardScaler();
var XStd = scaler.fitTransform(X);

// 2. PCA降维到k维
int k = 10;
var pca = new PCA(k);
var Z = pca.fitTransform(XStd);

// 3. 方差解释比例
var explainedVar = pca.explainedVarianceRatio();
System.out.printf("前10主成分累计解释方差: %.2f%%%n", 
    Arrays.stream(explainedVar).sum() * 100);
```

---

## 第 4 题：t-SNE vs PCA（比较题）

**参考答案**：

| | PCA | t-SNE |
|--|--|--|
| 类型 | 线性降维 | 非线性降维 |
| 目标 | 保留全局方差结构 | 保留局部邻域结构 |
| 计算 | 闭式解，快 | 迭代优化，慢 |
| 超参数 | 仅 k（维度数）| perplexity, learning rate, iterations |
| 适用 | 预处理、去噪 | 可视化（2-3维）|

⚠️ t-SNE 的缺点：随机性高（不同运行结果不同）、无法处理新数据（不是变换，而是嵌入）、对 perplexity 敏感。

---

## 第 5 题：维度灾难（概念题）

**参考答案**：

设均匀分布于 $p$ 维单位超立方体，分布在中心单位球内：

- 中心附近密度：$\propto n / (2^{2p})$ → 指数衰减
- 最近邻距离：$\approx (\frac{1}{n})^{1/p}$ → 高维需要极多样本才能找到近邻

**应对策略**：
1. 降维（PCA、t-SNE）
2. 特征选择（去掉无关特征）
3. 正则化（增加样本有效密度）
4. 稀疏建模（假设有效维度远低于原始维度）


## 5.5 树集成与 Boosting
> 📍 对应练习：[5.5. Tree ensembles and boosting.md](Chapter5_Machine_learning/5.5. Tree ensembles and boosting.md)


## 第 1 题：Bootstrap / OOB 概念题

**参考答案**：

- **Bootstrap**：有放回抽样 $n$ 次，构造 $B$ 个自助样本
- **OOB（Out-of-Bag）**：每个样本约有 1/3 没被抽中，可作为该样本的验证集
- **OOB 误差估计**：对每个样本，用未包含它的树组成的森林进行预测，误差率即 OOB 误差

优势：无需单独留出验证集，可利用全部数据训练同时估计泛化误差。

---

## 第 2 题：XGBoost 二阶导数推导

**参考答案**：

XGBoost 目标函数在第 $t$ 轮：

$$\mathcal{L}^{(t)} = \sum_{i=1}^n [g_i f_t(\mathbf{x}_i) + \frac{1}{2}h_i f_t^2(\mathbf{x}_i)] + \Omega(f_t)$$

其中 $g_i = \partial_{\hat{y}^{(t-1)}} \ell(y_i, \hat{y}^{(t-1)})$（一阶梯度），$h_i = \partial^2_{\hat{y}^{(t-1)}} \ell(y_i, \hat{y}^{(t-1)})$（二阶梯度）。

叶节点权重：$w_j^* = -\frac{\sum_{i \in R_j} g_i}{\sum_{i \in R_j} h_i + \lambda}$

二阶信息使 XGBoost 比只用梯度的 GBDT 收敛更快更准。

---

## 第 3 题：RF vs XGBoost 调参对比

**参考答案**：

| | Random Forest | XGBoost |
|--|--|--|
| 树数量 | 越多越好（不易过拟合）| 适中（过多会过拟合）|
| 树深度 | 通常较深（6-20）| 通常较浅（3-10）|
| 正则化 | 隐式（树多样性）| 显式（λ, α, γ）|
| 缺失值 | 随机森林可处理 | XGBoost 可学习最优方向 |
| 调参重点 | mtry（特征采样比）| learning_rate + n_estimators |

工业场景：先尝试 XGBoost（有正则化）；数据噪声大时优先 Random Forest（更鲁棒）。

---

## 第 4 题：AdaBoost 权重更新机制

**参考答案**：

每轮权重更新：

$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t \cdot y_i \cdot h_t(\mathbf{x}_i))$$

其中 $\alpha_t = \frac{1}{2}\log\frac{1-\epsilon_t}{\epsilon_t}$（分类误差越低，$\alpha_t$ 越大）

- 误分类样本：权重指数增长
- 正确分类样本：权重指数衰减

最终分类器：$H(\mathbf{x}) = \text{sign}(\sum_t \alpha_t h_t(\mathbf{x}))$，加权投票。

---

## 第 5 题：三大分类器综合对比

**参考答案**：

| | 逻辑回归 | 随机森林 | XGBoost |
|--|--|--|--|
| 决策边界 | 线性 | 非线性分段常数 | 非线性分段常数 |
| 训练方式 | 凸优化（全局最优）| Bagging（方差削减）| Boosting（偏差削减）|
| 可解释性 | 高（系数）| 中（特征重要性）| 低（黑盒）|
| 抗过拟合 | 需正则化 | 较强 | 需正则化 |
| 适合场景 | 基线模型 | 高方差数据 | 结构化数据 |
| 超参数数量 | 少 | 中 | 多 |



---

## 其他练习答案（陆续更新）

以下章节的练习答案正在编写中：

- Ch2 DataFrame（2.1-2.6）：基础操作题，略
- Ch3 可视化（3.1-3.6）：图表选择实操题，略
- Ch6 优化（6.1-6.4）：建模推导题
- Ch7 时序信号（7.1-7.4）：频谱分析题

如有疑问，欢迎通过 GitHub Issues 提问。

---

*最后更新：2026 年 4 月*

---

## 2.5 数据分组与聚合 / Grouping and Aggregation

### 练习答案

**练习1（分组统计量）**

```java
// 按部门统计平均工资
var deptGroups = df.groupBy("department");
var avgSalary = deptGroups.agg("salary", "mean");
```

**练习2（HAVING子句等价）**

```java
// 筛选平均工资 > 8000 的部门
var highDept = deptGroups.agg("salary", "mean")
    .filter(row -> row.getDouble("mean_salary") > 8000);
```

**练习3（多级分组）**

```java
// 按「部门+年份」分组统计
var result = df.groupBy(Arrays.asList("department", "year"))
    .agg("sales", "sum");
```

---

## 3.6 本章小结 / Summary

*注：本节为综述，无独立练习，内容与「典型数据处理流水线」练习共用。*

---

## 4.4 方差分析 / ANOVA

### 练习答案

**练习1（单因素ANOVA）**

```java
var anovaResult = Stats.anova.performOneWayANOVA(groups);
// anovaResult.pValue() < 0.05 → 组间差异显著
```

**练习2（F统计量计算）**

```java
// 手动计算F = SS_between / SS_within
double fStat = anovaResult.F();
double pValue = anovaResult.pValue();
```

**练习3（效应量η²）**

```java
double etaSquared = anovaResult.SSBetween() / anovaResult.SSTotal();
// η² > 0.14 → 大效应
```

---

## 4.6 统计质量控制 / Quality Control

### 练习答案

**练习1（控制图判断）**  
判断以下数据点是否超出3σ控制限：

```java
var qc = new QualityControl(limits);
boolean outOfControl = qc.isOutOfControl(dataPoint);
```

**练习2（Cpk计算）**

```java
// Cpk = min((USL-μ)/(3σ), (μ-LSL)/(3σ))
double cpk = Stats.qc().cpk(data, lowerLimit, upperLimit);
```

**练习3（EWMA控制图）**

```java
var ewma = Stats.qc().ewma(data, lambda=0.2, l=3);
// EWMA对微小漂移比传统控制图更敏感
```

---

## 5.1 回归 / Regression

### 练习答案

**练习1（多元线性回归）**

```java
var X = df.toMatrix("sqft", "bedrooms", "bathrooms");
var y = df.toVector("price");
var result = ML.linearRegression(X, y);
// result.coefficients() → [intercept, β_sqft, β_bedrooms, β_bathrooms]
```

**练习2（Ridge回归防过拟合）**

```java
// λ=1.0 时 L2 正则化
var ridgeResult = ML.linearRegression(X, y, lambda=1.0);
```

**练习3（预测与残差分析）**

```java
var predictions = ridgeResult.predict(X);
var residuals = predictions.subtract(y);
double rmse = Stats.math().rmse(residuals);
```

---

## 5.3 聚类 / Clustering

### 练习答案

**练习1（K-Means实战）**

```java
var X = df.toMatrix("sepal_length", "sepal_width", "petal_length", "petal_width");
var kmeans = ML.clustering().kMeans(X, k=3, seed=42);
var labels = kmeans.labels();
```

**练习2（确定最优K）**

```java
// 手肘法：绘制K从2到10的SSE
for (int k = 2; k <= 10; k++) {
    var result = ML.clustering().kMeans(X, k, seed=0);
    double sse = result.withinSS();
    System.out.println(k + "	" + sse);
}
```

**练习3（GMM概率聚类）**

```java
// GMM 返回每个样本属于各簇的概率
var gmm = ML.clustering().gmm(X, k=3);
var probs = gmm.predictProba(X); // shape: [n, k]
```


---

## 4.8 马尔可夫链蒙特卡洛方法

### 练习答案

**练习1（蒙特卡洛积分估计）**

```java
// 估计 ∫₀¹ e^{-x²} dx
int N = 10000;
double sum = 0;
var rng = new java.util.Random(42);
for (int i = 0; i < N; i++) {
    double x = rng.nextDouble();  // [0,1]均匀抽样
    sum += Math.exp(-x * x);
}
double estimate = sum / N;  // (b-a)/N = 1/N when a=0, b=1
System.out.println("估计值: " + estimate);
System.out.println("真实值(误差函数): " + 0.746824); // ∫₀¹ e^{-x²}dx
```

**练习2（MH接受率计算）**

```java
// 在 x=0.3 处
double pi_star = Math.pow(0.3, 2) * Math.pow(1-0.3, 3); // ∝ x²(1-x)³
// q(x*|xₜ) = N(xₜ, 0.1²)，对称 → q比率=1
double alpha = Math.min(1, pi_star / Math.pow(0.3, 2) * Math.pow(0.7, 3));
System.out.println("接受率: " + alpha);
```

**练习3（Gibbs vs MH对比）**

```java
// 二维高斯相关抽样
var gibbs = Stats.bayes().gibbsSampler(rho = 0.9);
var mh = Stats.bayes().metropolisHastings(target = bivariateGaussian);

// 比较ESS
double gibbsESS = Stats.bayes().effectiveSampleSize(gibbs.sample(5000));
double mhESS = Stats.bayes().effectiveSampleSize(mh.sample(5000));
System.out.println("Gibbs ESS: " + gibbsESS);
System.out.println("MH ESS: " + mhESS);
```

**练习4（HMC参数调优）**

```java
// β-Binomial 的HMC
var posterior = Stats.bayes().posterior(data, prior = "Beta(1,1)");
var hmc = Stats.bayes().hamiltonianMonteCarlo(posterior);

for (double stepSize : new double[]{0.01, 0.05, 0.1}) {
    var sampler = hmc.stepSize(stepSize).numLeapfrog(10);
    double ess = Stats.bayes().effectiveSampleSize(sampler.sample(5000));
    System.out.println("stepSize=" + stepSize + ": ESS=" + ess);
}
```

---

## 5.6 模型持久化

### 练习答案

**练习1（保存与加载）**

```java
// 训练
var model = ML.logisticRegression(Xtrain, ytrain, lambda = 0.01);

// 保存
ML.saveClassifier(model, "spam_lr.yishape");

// 加载并验证
var loaded = ML.loadClassifier("spam_lr.yishape");
var predictions1 = model.predict(Xtest);
var predictions2 = loaded.predict(Xtest);

// 比较
boolean consistent = Arrays.equals(predictions1, predictions2);
System.out.println("预测一致: " + consistent);
```

**练习2（模型元数据）**

```java
var modelInfo = new java.util.HashMap<String, Object>();
modelInfo.put("version", "1.0");
modelInfo.put("trainingDate", LocalDate.now().toString());
modelInfo.put("accuracy", 0.952);
modelInfo.put("dataset", "spam_v3.csv");

ML.saveClassifier(model, "spam_v3.yishape", modelInfo);

// 加载后提取
var loaded = ML.loadClassifier("spam_v3.yishape");
var info = loaded.getMetadata();
System.out.println("Version: " + info.get("version"));
System.out.println("Accuracy: " + info.get("accuracy"));
```

**练习3（大模型分片）**

```java
// 保存
var forest = ML.randomForest(Xtrain, ytrain, nTrees = 1000);
forest.saveSharded("rf_model_part", numShards = 4);
// 生成: rf_model_part_0.yishape, rf_model_part_1.yishape, ...

// 加载
var loaded = RandomForest.loadSharded("rf_model_part", 4);

// 时间对比
long t1 = System.currentTimeMillis();
var single = ML.loadClassifier("rf_1000trees.yishape");
long t2 = System.currentTimeMillis();
System.out.println("单文件: " + (t2-t1) + "ms");
```

---

## 7.6 卡尔曼滤波

### 练习答案

**练习1（一维卡尔曼滤波计算）**

```java
double[] z = {1.0, 1.1, 0.9, 1.05};
double Q = 0.01, R = 0.1;
double xhat = 0, P = 1;

for (double measurement : z) {
    // 预测
    double xhat_minus = xhat;
    double P_minus = P + Q;
    
    // 更新
    double K = P_minus / (P_minus + R);
    xhat = xhat_minus + K * (measurement - xhat_minus);
    P = (1 - K) * P_minus;
    
    System.out.println("测量: " + measurement + ", 估计: " + xhat);
}
```

**练习2（参数敏感性）**

```java
// Q/R 比值很大（过程稳定）
var stable = Signals.kalman(measurements, Q = 0.001, R = 1.0);
// → 滤波器响应慢，平滑效果好

// Q/R 比值很小（过程敏捷）
var agile = Signals.kalman(measurements, Q = 1.0, R = 0.001);
// → 滤波器响应快，但噪声大
```

**练习3（GPS定位应用）**

```java
// 生成含噪位置数据
int N = 200;
double[] truePos = new double[N];
double[] measured = new double[N];
var rng = new java.util.Random(42);

for (int i = 0; i < N; i++) {
    truePos[i] = Math.sin(0.1 * i);  // 真实：正弦+趋势
    measured[i] = truePos[i] + rng.nextGaussian() * 0.1; // 加噪声
}

// 卡尔曼滤波
var filtered = Signals.kalman(measured, Q = 0.01, R = 0.1);

// 对比
double[] ma = Signals.movingAverage(measured, 5);

// 计算RMSE
double rmseKalman = Stats.math().rmse(filtered, truePos);
double rmseMA = Stats.math().rmse(ma, truePos);
System.out.println("Kalman RMSE: " + rmseKalman);
System.out.println("MA(5) RMSE: " + rmseMA);
```

---

## 2.1 DataFrame 的构造与读取

### 练习答案

**练习1（CSV读取）**

```java
// 自动推断类型
var df = DataFrame.readCsv("data.csv");
System.out.println(df.describe());

// 手动指定类型
var df2 = DataFrame.readCsv("data.csv", 
    DataType.forName("double"),
    DataType.forName("int"),
    DataType.forName("String"));
```

**练习2（对比验证）**

自动推断和手动指定的结果应该完全一致。如果不一致，可能是：
- 自动推断把数值列识别成了字符串
- CSV中有缺失值被自动处理成了NaN

---

## 2.2 属性查询与数据访问

### 练习答案

**练习1（按名称取列）**

```java
// 按名称取 setosa 的花瓣长度
var col = df.col("petal_length");
var setosa = df.filter(row -> row.getString("species").equals("setosa"));
var petalLens = setosa.col("petal_length");
double mean = petalLens.toVector().mean();
System.out.println("Setosa平均花瓣长度: " + mean);
```

**练习2（负数索引验证）**

```java
var last = df.iloc(-1);        // 最后一列
var secondLast = df.iloc(-2);  // 倒数第二列
System.out.println(last.name());  // 验证列名
```

---

## 2.3 数据切分与过滤

### 练习答案

**练习1（训练/测试分割）**

```java
var X = df.toMatrix("sepal_length", "sepal_width", "petal_length", "petal_width");
var labels = df.col("species").toVector();

// 80/20 分割
var trainX = X.sliceRows(0, (int)(X.rowCount() * 0.8));
var testX = X.sliceRows((int)(X.rowCount() * 0.8), X.rowCount());

// 70/30 分割
var trainX70 = X.sliceRows(0, (int)(X.rowCount() * 0.7));
var testX30 = X.sliceRows((int)(X.rowCount() * 0.7), X.rowCount());

System.out.println("80/20: train=" + trainX.rowCount() + ", test=" + testX.rowCount());
System.out.println("70/30: train=" + trainX70.rowCount() + ", test=" + testX30.rowCount());
```

**练习2（过滤器组合）**

```java
var filtered = df.filter(row -> 
    row.getDouble("petal_length") > 3.0 && 
    row.getString("species").equals("versicolor")
);
System.out.println("符合条件的样本数: " + filtered.rowCount());
```

---

## 2.4 DataFrame 到 ML 流水线

### 练习答案

**练习1（相关性矩阵）**

```java
var X = df.toMatrix("sepal_length", "sepal_width", "petal_length", "petal_width");
var corr = MatrixStats.corr(X);

// 找最大相关性对
double maxCorr = 0;
int maxI = 0, maxJ = 0;
for (int i = 0; i < corr.rowCount(); i++) {
    for (int j = i+1; j < corr.colCount(); j++) {
        if (Math.abs(corr.get(i, j)) > maxCorr) {
            maxCorr = Math.abs(corr.get(i, j));
            maxI = i; maxJ = j;
        }
    }
}
System.out.println("最大相关: [" + maxI + "," + maxJ + "] = " + maxCorr);
```

**练习2（toMatrix维度问题）**

```java
// 混合类型DataFrame调用toMatrix会丢弃非数值列
var dfMixed = DataFrame.readCsv("mixed.csv");  // 含数值和字符串
var X = dfMixed.toMatrix();  // 只保留数值列，字符串列被丢弃
// 建议：先filter筛选数值列，或用select指定列
```

---

## 7.5 z 变换与离散时间系统

### 练习答案

**练习1（收敛域判断）**

序列 x[n] = (0.5)^n u[n] + (2)^n u[-n-1]：

- 第一项 (0.5)^n u[n]：右侧序列，|z| > 0.5
- 第二项 (2)^n u[-n-1]：左侧序列，|z| < 2
- 共同收敛域：0.5 < |z| < 2（环状区域）

**练习2（稳定性分析）**

H(z) = z/(z-0.8)，极点在 z=0.8（单位圆内） -> 系统稳定（因果条件下）

**练习3（频率响应）**

H(z) = (z+1)/(z-0.5)，令 z=e^{j\omega}：
- 判断低通/高通：看 \omega=0（直流）和 \omega=\pi（高频）的增益
- 低频增益 > 高频增益 -> 低通特性

**练习4（逆变换）**

由 X(z) = 2z/(z-0.5) + z/(z-2)：
- |z| > 0.5 -> 右侧序列 -> x1[n] = 2*(0.5)^n u[n]
- |z| < 2 -> 左侧序列 -> x2[n] = -(2)^n u[-n-1]
- 总序列为两者之和

**练习5（差分方程）**

y[n] = 0.7y[n-1] + x[n]：
- 两边取z变换：Y(z)(1-0.7z^{-1}) = X(z)
- H(z) = Y(z)/X(z) = 1/(1-0.7z^{-1}) = z/(z-0.7)
- h[n] = (0.7)^n u[n]

---

## 6.5 并行优化

### 练习答案

**练习1（并行度选择）**

- 95% 可并行：理论加速比上限 = 1/(0.05 + 0.95/10) ≈ 1/(0.05 + 0.095) ≈ 6.9x
- 80% 可并行：理论加速比上限 = 1/(0.2 + 0.8/10) = 1/(0.2 + 0.08) ≈ 3.6x

**练习2（K-Means并行化）**

```java
ExecutorService exec = Executors.newFixedThreadPool(numThreads);
List<Future<double[]>> futures = new ArrayList<>();

for (int k = 0; k < K; k++) {
    final int clusterId = k;
    futures.add(exec.submit(() -> {
        return computeCentroid(data, assignments, clusterId);
    }));
}

double[][] newCentroids = new double[K][];
for (int k = 0; k < K; k++) {
    newCentroids[k] = futures.get(k).get();
}
exec.shutdown();
```

---

## 2.6 DataFrame 本章小结

### 练习答案

**1. 完整数据流水线（综合题）**

```java
var df = DataFrame.readCsv("sales.csv");
var cleaned = df.fillna("revenue", 0.0)  // 缺失值填0
               .filter(row -> row.getDouble("revenue") > 0);  // 过滤异常
var monthly = cleaned.groupBy("month").agg("revenue", "sum");
System.out.println("月销售额:\\n" + monthly);
```

**2. 性能对比思考（思考题）**

pandas（Python）在数据框操作上生态更成熟，API 丰富；但 Java 的 DataFrame 在与 Java 企业系统（Spring Boot）集成时有天然优势，且 YiShape Math 的 DataFrame 与 Linalg、ML 等库无缝衔接，适合 Java 技术栈的数据团队。

---

## 4.7 统计质量控制

### 练习答案

**1. 控制图判断（应用题）**

数据点：10.05, 10.08, 10.12, 10.15, 10.20

判断：
- 前4点在控制限内
- 10.20 > UCL(10.3) 且呈上升趋势
- 最后一点超出UCL，**工序失控**，需查找特殊原因

**2. Cpk计算（计算题）**

USL=10.2, LSL=9.8, μ=10.0, σ=0.05

Cpk = min((10.2-10.0)/(3*0.05), (10.0-9.8)/(3*0.05))
    = min(0.2/0.15, 0.2/0.15) = min(1.33, 1.33) = **1.33**

Cpk = 1.33，满足 Cpk > 1.33 的要求，工序能力合格。

---

## 6.1 优化基础

### 练习答案

**1. 梯度下降实现（编程题）**

```java
var x = Linalg.vector(new double[]{0, 0});
double lr = 0.1;
for (int i = 0; i < 1000; i++) {
    var grad = gradient(f, x);        // 梯度
    x = x.subtract(grad.multiply(lr));  // x_{k+1} = x_k - lr * grad
    if (grad.norm() < 1e-6) break;   // 收敛判断
}
System.out.println("最优解: " + x);
System.out.println("最优值: " + f(x));
```

**2. 全局最优 vs 局部最优（思考题）**

凸优化：局部最优即全局最优（非凸函数不保证）
判断方法：检查 Hessian 矩阵是否正定，或使用多个初始点验证

---

## 6.2 线性规划

### 练习答案

**1. 图解法（计算题）**

约束条件：
- x + y <= 8
- 2x + y <= 12
- x >= 0, y >= 0

图解：
- 可行域为四边形顶点 (0,0), (0,8), (4,4), (6,0)
- 代入目标函数 z = 2x + 3y：
  - (0,0): z=0, (0,8): z=24, (4,4): z=20, (6,0): z=12
- 最优解: x=0, y=8, z_max=24

**2. Simplex算法（编程题）**

```java
var c = Linalg.vector(new double[]{-40, -30});  // 最大化转最小化
var A = Linalg.matrix(new double[][]{
    new double[]{1, 0},
    new double[]{0, 1},
    new double[]{2, 3}
});
var b = Linalg.vector(new double[]{8, 6, 18});
var solver = Opts.simplexLinProgSolver();
var result = solver.solve(c, A, b);
System.out.println("最优解: x=" + result.x().get(0) + ", y=" + result.x().get(1));
System.out.println("最优值: " + -result.value());
```

---

## 6.3 整数规划

### 练习答案

**1. 分支定界思路（分析题）**

分支：x1=0 或 x1=1 两个分支
定界：每分支求 LP 松弛得到上界
剪枝：下界比上界还大则剪枝
迭代至所有分支剪枝，找到最优整数解。

**2. 0-1背包建模（应用题）**

```java
var items = new double[]{10, 20, 15, 8};  // 价值
var weights = new double[]{2, 4, 3, 1};  // 重量
double capacity = 7;

// 建模
var c = Linalg.vector(items);  // 最大化价值
var A = Linalg.matrix(new double[][]{weights});
var b = Linalg.vector(new double[]{capacity});
var solver = Opts.intLinProgSolver();
var result = solver.solve(c, A, b);  // 整数规划求解
System.out.println("选中物品: " + result.x());
```

---

## 6.4 非凸优化

### 练习答案

**1. 多起点策略（应用题）**

```java
var starts = new double[][]{
    new double[]{0, 0},
    new double[]{5, 5},
    new double[]{-3, 2}
};
var bestX = starts[0];
var bestVal = Double.MAX_VALUE;
for (var x0 : starts) {
    var result = Opts.steepestDescent().optimize(f, Linalg.vector(x0), 0.01, 1000);
    if (result.value() < bestVal) {
        bestVal = result.value();
        bestX = result.x();
    }
}
System.out.println("全局最优近似: " + bestX + " = " + bestVal);
```

**2. 鞍点判断（计算题）**

鞍点条件：梯度为0，但 Hessian 矩阵特征值有正有负。
判断方法：计算 Hessian 矩阵的特征值 λ1...λn，若全正=局部最小，全负=局部最大，有正有负=鞍点。

---

## 7.1 时间序列分析

### 练习答案

**1. AR(1)建模（计算题）**

x_t = 0.7*x_{t-1} + ε_t

参数含义：
- AR系数 0.7：前一期对当期影响为70%
- 序列自相关：ρ1 ≈ 0.7，ρ2 ≈ 0.49

平稳性判断：|0.7| < 1，**序列平稳**

**2. ADF检验应用（编程题）**

```java
var ts = TimeSeries.createTimeSeries(prices);
var adfResult = ts.adfTest();
System.out.println("ADF统计量: " + adfResult.statistic());
System.out.println("p值: " + adfResult.pValue());
if (adfResult.pValue() < 0.05) {
    System.out.println("拒绝原假设：序列平稳");
} else {
    System.out.println("接受原假设：序列非平稳，需差分");
}
```

**3. STL分解（编程题）**

```java
var ts = TimeSeries.createTimeSeries(data);
var decomp = ts.seasonalDecompose(seasonalPeriod=12);
// 提取趋势、季节、残差
var trend = decomp.trend();
var seasonal = decomp.seasonal();
var residual = decomp.residual();
Plots.plot TimeSeries(new TimeSeries[]{trend, seasonal, residual}, 
    new String[]{"趋势", "季节", "残差"});
```

---

## 7.2 信号处理

### 练习答案

**1. 采样定理验证（计算题）**

信号频率 f_max = 100Hz，临界采样率 f_s = 200Hz

- f_s > 2*f_max，可完全重建
- f_s = 2*f_max，刚好满足奈奎斯特频率

**2. 频谱分析（编程题）**

```java
var signal = Signals.chirpSignal(0, 1, 100, 500);  // 扫频信号
var spectrum = signal.fft();  // FFT得到频谱
var psd = spectrum.powerSpectrum();  // 功率谱密度

// 找主频率
int peakIdx = spectrum.magnitude().argmax();
double peakFreq = spectrum.frequency(peakIdx);
System.out.println("主频率: " + peakFreq + " Hz");
```

**3. 滤波器设计（设计题）**

设计一个截止频率 200Hz 的低通滤波器：
```java
var lpf = Signals.lowpass(cutoffFreq=200, sampleRate=1000, order=4);
var filtered = lpf.apply(noisySignal);
```

---

## 7.3 音频处理

### 练习答案

**1. MFCC提取（编程题）**

```java
var audio = Signals.readAudio("speech.wav");
var mfcc = audio.extractMFCC(numCoefs=13);
System.out.println("MFCC形状: " + mfcc.rows() + " x " + mfcc.cols());
// 绘制时间-频率热力图
Plots.heatmap(mfcc.T(), "时间", "MFCC系数", "MFCC时频图");
```

**2. 频谱特征计算（计算题）**

```java
var audio = Signals.readAudio("piano.wav");
var spectrum = audio.fft();
double centroid = spectrum.spectralCentroid();
double rolloff = spectrum.spectralRolloff(0.85);
System.out.println("频谱质心: " + centroid + " Hz");
System.out.println("频谱滚降点(85%): " + rolloff + " Hz");
// 频谱质心越高，音色越明亮
```

---

## 7.4 音乐挖掘

### 练习答案

**1. 节拍检测（编程题）**

```java
var music = Signals.readAudio("music.wav");
var tempo = music.tempoDetection();
System.out.println("估计 BPM: " + tempo.bpm());

// 或用 librosa 方式
var onsetEnv = music.onsetEnvelope();
var beats = onsetEnv.beatTrack();
```

**2. 和弦识别思路（分析题）**

1. 分帧：将音频切为短时帧（如 4096 点/帧）
2. FFT：每帧做快速傅里叶变换得到频谱
3. 音高映射：频谱峰值对应音高频率（C, C#, D...）
4. 和弦匹配：将音高集合与已知和弦模板匹配
5. 平滑：动态规划或维特比算法平滑时间轴上的和弦序列
