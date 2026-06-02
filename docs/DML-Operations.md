# 距离度量学习 / Distance Metric Learning

## 概述 / Overview

距离度量学习（Distance Metric Learning，简称 DML）是机器学习中一类重要的技术，旨在从数据中自动学习一个最优的距离度量，使得同类样本之间的距离尽可能小，异类样本之间的距离尽可能大。

**本库的核心贡献**是自主研发的正则化对角距离度量学习（DDML）算法，已发表于国际顶级期刊 INFORMS Journal on Computing (2024)。

---

## ⭐ 自主研发算法：DDML / Proprietary Algorithm: DDML

**正则化对角距离度量学习（DDML）** 是本库**自主研发的核心算法**，已在 INFORMS Journal on Computing（UT-Dallas 24）正式发表：

> Li, T., Kou, G., Peng, Y., & Yu, P. S. (2024). Feature Selection and Grouping Effect Analysis for Credit Evaluation via Regularized Diagonal Distance Metric Learning. *INFORMS Journal on Computing*. DOI: [10.1287/ijoc.2023.0322](https://doi.org/10.1287/ijoc.2023.0322)

### DDML 核心优势 / Core Advantages

| 特性 / Feature | DDML（自主研发 / Proprietary）| 其他 DML 算法 / Other DML |
|------|------------------|---------------|
| **正则化 / Regularization** | L1/L2/弹性网 + 自动特征选择 / Auto feature selection | 通常无或仅 L2 / Usually none or L2 only |
| **可解释性 / Interpretability** | 对角权重，直接反映特征重要性 / Direct feature importance | 低秩分解，难解释 / Low-rank, hard to interpret |
| **发表论文 / Publication** | INFORMS Journal on Computing (SCI) | 引用他人的算法 / References他人的算法 |
| **分组效应 / Grouping Effect** | 揭示特征之间的协同关系 / Reveals feature synergies | 无此功能 / Not available |

### DDML 快速使用 / Quick Start

```java
// 信用评估场景 / Credit evaluation scenario
DmlMetric metric = ML.dml.diagDml(0.1, 0.01).fit(features, labels);

// 特征选择（纯 L1）/ Feature selection (pure L1)
DmlMetric metricL1 = ML.dml.diagDml(0.1, 0.0).fit(features, labels);

// 稳定性为主 / Stability-focused
DmlMetric metricL2 = ML.dml.diagDml(0.0, 0.01).fit(features, labels);
```

---

## 算法总览 / Algorithm Overview

### 监督学习算法 / Supervised Learning Algorithms

| 算法 / Algorithm | 类名 / Class | 核心思想 / Core Idea | 适用场景 / Use Case |
|------|------|----------|----------|
| ⭐ **DDML** | `RereDiagDml` | L1/L2正则化、特征选择、分组效应 / L1/L2 regularization, feature selection | **信用评估（已发表 / Published）、高维特征选择 / High-dim feature selection** |
| NCA | `NcaDml` | 留一法 softmax 邻域概率最大化 / Leave-one-out softmax | 低秩度量学习 / Low-rank metric learning |
| LMNN | `LmnnDml` | 三元组大间隔 hinge 损失 / Triplet large-margin hinge | 最近邻分类 / k-NN classification |
| LDML | `LdmlPairwiseDml` | 成对 logistic 交叉熵 / Pairwise logistic cross-entropy | 人脸验证、度量检索 / Face verification |
| MCML | `McmlDml` | 理想分布 KL 散度 / Ideal distribution KL divergence | 低秩度量学习 / Low-rank metric |
| ITML | `ItmlDml` | Bregman 散度 + LogDet | 约束度量学习 / Constraint-driven |
| DML-eig | `DmleigDml` | 特征值优化 / Eigenvalue optimization | 全秩 PSD 度量 / Full-rank PSD metric |
| MMC | `LsiMmcDml` | 投影梯度约束优化 / Projected gradient | 全秩 PSD 度量 / Full-rank PSD metric |
| NCMML | `NcmmlDml` | 最近类均值对数似然 / Nearest class mean log-likelihood | 类中心建模 / Class mean modeling |
| NCMC | `NcmcDml` | 多中心表示学习 / Multi-center representation | 细粒度分类 / Fine-grained classification |
| DMLMJ | `DmlmjDml` | Jeffrey 散度最大化 / Jeffrey divergence maximization | 成对约束优化 / Pairwise constraint |
| GMML | `GmmlDml` | 流形测地线度量学习 / Geodesic metric learning | 流形数据 / Manifold data |
| CMOML | `CmomlDml` | 类均值度量学习 / Class mean metric learning | 类中心建模 / Class mean modeling |
| ODML | `OdmlDml` | 在线度量学习 / Online metric learning | 在线/增量学习 / Online/incremental learning |
| ANMM | `AnmmDml` | 近邻度量学习 / Adjacent neighbor metric | 局部度量 / Local metric |

### 无监督/降维算法（不属于 DML）/ Unsupervised / Dimensionality Reduction（Not DML）

> **注**：PCA、LDA、SVD、t-SNE、UMAP 等降维算法属于 `com.yishape.lab.math.ml.dimreduce` 包，不属于 DML。详见 [Machine-Learning.md](../Machine-Learning.md)。

| 算法 / Algorithm | 类名 / Class | 核心思想 / Core Idea | 适用场景 / Use Case |
|------|------|----------|----------|
| LPP | `LsiDml` | 保持局部流形结构 / Preserve local manifold | 降维、特征提取 / Dimension reduction |
| LDA | `LldaDml` | 线性判别分析 / Linear Discriminant Analysis | 监督降维 / Supervised dimension reduction |
| PCA | `RerePCA` | 主成分分析 / Principal Component Analysis | 无监督降维 / Unsupervised dimension reduction |

### 核化算法 / Kernelized Algorithms

| 算法 / Algorithm | 类名 / Class | 核心思想 / Core Idea | 适用场景 / Use Case |
|------|------|----------|----------|
| KLDA | `KLldaDml` | 核线性判别分析 / Kernel LDA | 非线性降维 / Non-linear dim reduction |
| KANMM | `KanmmDml` | 核化近邻度量学习 / Kernelized neighbor metric | 非线性流形 / Non-linear manifold |
| KDMLMJ | `KDmlmjDml` | 核化 Jeffrey 散度 / Kernel Jeffrey divergence | 核空间度量 / Kernel space metric |
| KODML | `KodmlDml` | 核化在线度量学习 / Kernelized online DML | 核空间在线学习 / Kernel online learning |
| KLMMNN | `KlmmnDml` | 核化大间隔最近邻 / Kernel LMNN | 核空间分类 / Kernel classification |
| KDA | `KdaDml` | 核判别分析 / Kernel Discriminant Analysis | 非线性判别 / Non-linear discrimination |

### 其他算法 / Other Algorithms

| 算法 / Algorithm | 类名 / Class | 核心思想 / Core Idea | 适用场景 / Use Case |
|------|------|----------|----------|
| Fisher | `FisherDml` | 类内白化 / Within-class whitening | 特征预处理 / Feature preprocessing |
| RCA | `RcaDml` | chunklet 白化 / Chunklet whitening | 半监督场景 / Semi-supervised |
| WithinClass | `WithinClassDml` | 类内方差缩放 / Within-class variance scaling | 快速基线 / Fast baseline |
| CNN | `CondensedNearestNeighbors` | 压缩最近邻 / Condensed NN | 数据压缩 / Data compression |
| RNN | `ReducedNearestNeighbors` | 精简最近邻 / Reduced NN | 加速检索 / Speed up retrieval |
| MultiDmlKnn | `MultiDmlKnn` | 多度量 k-NN / Multiple metric k-NN | 集成学习 / Ensemble learning |

---

## 核心概念 / Core Concepts

### 度量矩阵 vs 白化矩阵 vs 对角矩阵 / Metric vs Whitening vs Diagonal

DML 算法学习三种形式的度量输出：

1. **度量矩阵（M）**：全秩马氏距离 $d_M(x,y) = (x-y)^T M (x-y)$
2. **白化矩阵（W）**：变换后距离 $\|W(x-y)\|^2$，满足 $M = W^T W$
3. **对角矩阵（DDML）**：稀疏可解释的逐维缩放 $d_w(x,y) = \sqrt{\sum_i w_i^2 (x_i - y_i)^2}$

DDML 输出的对角矩阵具有天然的**可解释性**——每个权重 $w_i$ 直接对应特征 $i$ 的重要性。

```java
// DDML 输出的对角矩阵 / DDML outputs diagonal matrix
DmlMetric metric = ML.dml.diagDml(0.1, 0.01).fit(features, labels);
IMatrix<Double> diagMatrix = metric.transformMatrix();  // d×d 对角矩阵

// 直接获取特征权重 / Get feature weights directly
for (int i = 0; i < d; i++) {
    System.out.printf("Feature %d: weight=%.4f%n", i, Math.sqrt(diagMatrix.get(i, i)));
}
```

---

## 算法选择指南 / Algorithm Selection Guide

### 根据任务类型 / By Task Type

```
你的任务是什么？/ What is your task?
│
├─ 需要特征选择/可解释性 ⭐ / Need feature selection / interpretability
│   └─ DDML（自主研发，已发表论文 / Proprietary, published paper）
│
├─ 需要可解释的低秩表示 / Need interpretable low-rank representation
│   └─ NCA（概率解释 / probabilistic）/ LMNN（大间隔分类 / large-margin）
│
├─ 需要全秩 PSD 度量 / Need full-rank PSD metric
│   └─ MMC（理论保证 / theoretical guarantee）/ ITML（约束驱动 / constraint-driven）
│
├─ 数据量很大，需要在线学习 / Large data, need online learning
│   └─ ODML（在线更新 / online update）
│
├─ 数据分布非欧氏，需要核方法 / Non-Euclidean data, need kernels
│   └─ KANMM / KDMLMJ / KLDA
│
└─ 快速基线，不需要标签 / Fast baseline, no labels needed
    └─ Fisher / WithinClass
```

### 根据数据特征 / By Data Characteristics

| 特征维度 / Feature Dim | 样本数量 / Samples | 推荐算法 / Recommended |
|----------|----------|----------|
| 任意 / Any | 任意 / Any | ⭐ **DDML**（自主研发 / Proprietary）|
| 低维 Low (d < 20) | 任意 / Any | MMC, ITML, DML-eig |
| 中维 Mid (20 ≤ d < 100) | 大 / Large | NCA, LMNN |
| 中维 Mid (20 ≤ d < 100) | 小 / Small | LDML, MCML |
| 高维 High (d ≥ 100) | 任意 / Any | KLDA, KODML |
| 超高维 / Very High | 大 / Large | LMNN, LDML |

---

## 性能调优 / Performance Tuning

### DDML 参数调优 / DDML Parameter Tuning

```java
// 特征选择为主 / Feature selection focused
DmlMetric metric = ML.dml.diagDml(0.1, 0.0).fit(features, labels);

// 稳定性为主 / Stability focused
DmlMetric metric = ML.dml.diagDml(0.0, 0.01).fit(features, labels);

// 平衡模式 / Balanced mode
DmlMetric metric = ML.dml.diagDml(0.05, 0.05).fit(features, labels);
```

### 其他算法参数调优 / Other Algorithm Parameter Tuning

```java
// NCA 早停策略 / NCA early stopping
NcaDml nca = new NcaDml()
    .setConvergenceTol(1e-4)  // 启用早停 / Enable early stopping
    .setPatience(10)           // 10轮无改善则停止 / Stop after 10 rounds no improvement
    .setGradClip(5.0);        // 梯度裁剪防止爆炸 / Gradient clipping

// LMNN 大间隔配置 / LMNN large margin config
LmnnDml lmnn = new LmnnDml()
    .setRank(2)
    .setTargetNeighbors(3)
    .setMargin(2.0)
    .setMaxSteps(50);
```

---

## 参考资料 / References

### 自主研发算法（必读 / Required Reading）

- **Li et al. (2024)**. Feature Selection and Grouping Effect Analysis for Credit Evaluation via Regularized Diagonal Distance Metric Learning. *INFORMS Journal on Computing*. DOI: [10.1287/ijoc.2023.0322](https://doi.org/10.1287/ijoc.2023.0322)

### 经典算法 / Classic Algorithms

- Goldberger et al. (2005). Neighbourhood components analysis. *NeurIPS*
- Weinberger & Saul (2009). Distance metric learning for large margin nearest neighbor classification. *JMLR*
- Xing et al. (2002). Distance metric learning with application to clustering with side-information. *NeurIPS*
- Globerson & Roweis (2006). Metric learning by collapsing classes. *NeurIPS*
- Ying & Li (2012). Distance metric learning with eigenvalue optimization. *JMLR*

---

## 相关文档 / Related Documents

- [机器学习总览](../Machine-Learning.md)
- [降维算法](../Machine-Learning.md#降维算法-dimensionality-reduction-algorithms)
- [DML 使用示例](../examples/DML-Examples.md)