# 第 5 章总结 / Chapter 5 Summary

## 这一章解决了什么问题

机器学习干的事情，从本质上说只有三件：**预测一个数值（回归）、预测一个类别（分类）、发现数据里的自然群组（聚类）**。

学完这章，你应该能：

**拿到一个业务问题，判断它属于哪类 ML 任务，然后选择合适的算法，从数据预处理到模型评估独立完成。**

---

## 核心概念回顾

| 任务 | 算法 | 回答什么问题 | YiShape-Math API |
|------|------|-------------|-----------------|
| **回归** | 线性回归 + 正则化 | 多少？（广告预算 → 订单量） | `ML.reg.linear()` |
| **回归** | 岭回归/Lasso | 多少？（正则化防过拟合） | `ML.reg.linear(alpha, lambda)` |
| **二分类** | 逻辑回归 | 是还是否？（贷款违约/不违约） | `ML.clf.logisticRegression(l1, l2)` |
| **二分类** | K 近邻 | 像不像？（用户行为相似 → 同类推荐） | `ML.clf.kNN(k)` |
| **二分类** | 决策树 | 怎么判断？（可解释的规则） | `ML.clf.decisionTree()` |
| **二分类** | 线性 SVM | 最优分隔？（最大间隔分类） | `ML.clf.linearSvm(C)` |
| **多分类** | 逻辑回归扩展 | 哪个？（图片识别：猫/狗/车） | 同上 |
| **聚类** | K-Means++ | 怎么分群？（用户行为分群） | `ML.clu.kMeans(k)` |
| **聚类** | GMM | 概率分群？（软聚类） | `ML.clu.gmm(k)` |
| **降维** | PCA | 能不能压缩？（50个特征 → 5个主成分） | `ML.dr.pca(k)` |
| **降维** | t-SNE/UMAP | 高维数据怎么可视化？（2D 散点图） | `ML.dr.tsne(k)` / `ML.dr.umap(k)` |
| **降维** | 核 PCA | 非线性降维？ | `ML.dr.kernelPca(k)` |
| **集成** | 随机森林 | 怎么减少过拟合？（多棵树投票） | `ML.clf.randomForest()` |
| **集成** | XGBoost | 精度怎么再提升？（梯度提升） | `ML.clf.xGboost()` |
| **预处理** | 标准化 | 特征尺度统一 | `ML.preproc.standardScaler()` |
| **预处理** | 归一化 | 特征缩放到 [0,1] | `ML.preproc.minMaxScaler()` |
| **预处理** | 鲁棒缩放 | 异常值场景 | `ML.preproc.robustScaler()` |
| **预处理** | 独热编码 | 分类变量 → 数值 | `ML.preproc.oneHotEncoder()` |
| **预处理** | 多项式特征 | 非线性特征 | `ML.preproc.polynomialFeatures(degree)` |
| **度量学习** | Fisher 白化 | 最大化类间分离 | `ML.dml.fisherWhitening()` |
| **度量学习** | LMNN | KNN 改进首选 | `ML.dml.lmnn(rank, k, margin)` |
| **度量学习** | NCA | 概率化距离学习 | `ML.dml.nca()` |

---

## 模型选择指南（实践建议）

```
先建立逻辑回归基线（可解释、快速）
    ↓
再试随机森林（调参少、鲁棒）
    ↓
最后试 XGBoost（精度最高，但需更多调参）
```

**记住**：不是越复杂的模型越好。逻辑回归在大量实际场景中（尤其是特征有明确业务含义时）反而是最可靠的选择。

---

## 评估指标速查

```java
// 分类
ClassificationMetrics.accuracy(labels, predictions);   // 准确率
ClassificationMetrics.precision(labels, predictions);  // 精确率
ClassificationMetrics.recall(labels, predictions);   // 召回率
ClassificationMetrics.f1(labels, predictions);       // F1
ClassificationMetrics.auc(labels, probabilityScores); // AUC（ROC 曲线下面积，Area Under Curve，最常用）

// 回归
RegressionMetrics.mse(yTrue, yPred);  // 均方误差
RegressionMetrics.rmse(yTrue, yPred); // RMSE（均方根误差，Root Mean Square Error，常用）
RegressionMetrics.mae(yTrue, yPred);  // 平均绝对误差
RegressionMetrics.r2(yTrue, yPred);  // R²（模型解释力）
```

---

## 与其他章节的联系

- **第 4 章统计学**：正则化（岭/Lasso）≈ 贝叶斯先验；交叉验证来自统计检验思想；偏差-方差分解是模型选择理论基础
- **第 1 章线性代数**：SVD/PCA 的数学基础；矩阵分解是协同过滤推荐系统的核心
- **第 6 章最优化**：所有 ML 模型训练 = 优化问题；梯度下降、L-BFGS 是求解引擎
- **第 3 章可视化**：混淆矩阵、ROC 曲线、Precision-Recall 曲线是模型评估的必备图表

---

## 常见误区

1. **不 train/test 分割就报告准确率**：在训练集上评估 = 自我欺骗；永远留出独立的测试集
2. **不处理类别不平衡**：99% 的用户不违约，1% 违约——直接跑模型，准确率 99% 但毫无意义
3. **用测试集调超参**：测试集只能评估，不能用来调参；调参用验证集（或交叉验证）
4. **把相关性当因果**：模型告诉你「买过帐篷的人也买登山鞋」，不意味着「买了帐篷导致买登山鞋」

---

*第 5 章的核心只有一句话：**机器学习是从数据中找规律；规律找得靠不靠谱，要靠验证集来检验。***
