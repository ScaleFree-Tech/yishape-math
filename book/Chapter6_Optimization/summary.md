# 第 6 章总结 / Chapter 6 Summary

## 这一章解决了什么问题

**机器学习模型的训练，和高德地图导航，背后是同一类数学问题。**

那就是：在所有可能的选项中，找到使某个目标最大（或最小）的那一个——优化。

学完这章，你应该理解：

- **为什么梯度下降能找到「好一点」的解，但不一定是最优解**
- **为什么线性规划（LP）有高效算法，而整数规划（IP）几乎不可能高效求解**
- **为什么神经网络的训练是一个非凸优化问题，以及为什么我们仍然能训练出有用的模型**

---

## 核心概念回顾

| 优化类型 | 适用场景 | 典型算法 | YiShape-Math API |
|---------|---------|---------|-----------------|
| 无约束凸优化 | 平滑的凸问题 | L-BFGS（推荐）、共轭梯度法 | `Opts.lbfgs()` |
| 在线优化 | 大规模/流式数据、神经网络 | SGD、Adam | `new RereOnlineSGD(lr)` |
| 线性规划 | 资源分配、调度 | 单纯形法 | `Opts.simplexLinProgSolver()` |
| 整数规划 | 0-1 决策、选址、排班 | 分支定界 | `Opts.intLinProgSolver()` |
| 非凸优化 | 神经网络、混合整数 | SGD、Adam（启发式） | `new RereOnlineAdam(lr, beta1, beta2)` |

---

## 本章 API 速查

```java
// 无约束优化
IOptimizer opt = Opts.lbfgs();  // 推荐用于平滑凸问题
OptResult result = opt.optimize(initialPoint, objective, gradient);

// 线性规划（等式约束为主）
ILinProgSolver lp = Opts.simplexLinProgSolver();
OptResult result = lp.solveEq(c, A_eq, b_eq);
var solution = result.getOptimalPoint();
double optValue = result.getOptimalValue();

// 整数规划
IIntegerProg ip = Opts.intLinProgSolver();
OptResult result = ip.solve(c, A, b);

// SGD / Adam（神经网络训练）
IOnlineOptimizer sgd = new RereOnlineSGD(0.01);
IOnlineOptimizer adam = new RereOnlineAdam(0.001, 0.9, 0.999);
```

> **注意**：`OptResult` 字段是 `getOptimalPoint()`（不是 `.getOptimalValue()`），`isConverged()` 检查是否收敛。

---

## 与其他章节的联系

- **第 5 章机器学习**：所有 ML 模型的训练 = 在 loss function 上跑优化算法；线性回归用 L-BFGS，神经网络用 SGD/Adam
- **第 1 章线性代数**：约束矩阵 $\mathbf{A}$ 和右端向量 $\mathbf{b}$ 是 LP 的核心；正规方程是最小二乘优化的闭式解
- **第 4 章统计学**：最大似然估计（MLE）是优化问题；EM 算法是含隐变量的优化

---

## 常见误区

1. **把收敛当最优**：优化算法报告「收敛」只说明梯度接近零，不说明找到了全局最优——非凸问题尤其如此
2. **步长（learning rate）设太大**：SGD/Adam 的步长过大时会overshoot（跳过最优解来回震荡），设太小则跑不动
3. **忽视约束条件**：LP 里 $\mathbf{x} \geq \mathbf{0}$ 是约束，不是目标函数的一部分——漏写约束会得到完全不同的解
4. **整数规划当连续解**：LP 松弛解 $x = 0.7$ 在整数约束下可能是 $x = 0$ 或 $x = 1$——二者的目标值差距可能很大（optimality gap）

---

*第 6 章的核心只有一句话：**优化是把「想要什么」翻译成「怎么搜索」的工具——理解它的结构和限制，比记住一堆算法更重要。***
