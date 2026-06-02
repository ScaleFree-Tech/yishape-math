# 多目标线性规划 / Multi-Criteria Linear Programming (MCLP)

## 概述 / Overview

多目标线性规划（Multi-Criteria Linear Programming，MCLP）是一类优化问题，其中同时存在多个需要优化的目标函数，而这些目标函数之间可能相互冲突。MCLP 广泛应用于资源分配、投资组合优化、工程设计等需要权衡多个目标的场景。

Multi-Criteria Linear Programming (MCLP) is a class of optimization problems where multiple objective functions need to be optimized simultaneously, and these objectives may conflict with each other. MCLP is widely used in resource allocation, portfolio optimization, engineering design, and other scenarios requiring trade-offs between multiple objectives.

**入口类 / Entry Point**: `Opts.mclp`（通过 `MclpWrapper` 提供所有 MCLP 求解器）

---

## 核心接口 / Core Interface

### IMclpSolver

```java
public interface IMclpSolver extends Serializable {
    /**
     * 求解多目标线性规划问题
     * Solve multi-criteria linear programming problem
     */
    MclpResult solve(IVector[] c, IMatrix A_ub, IVector b_ub,
                     IMatrix A_eq, IVector b_eq, IVector initX);

    /**
     * 获取求解器类型
     * Get solver type
     */
    MclpSolverType getSolverType();
}
```

### MclpResult

```java
public class MclpResult implements Serializable {
    // 解集信息
    List<IVector> getSolutions();           // 最优解集
    List<double[]> getObjectiveValues();     // 各目标函数值
    double[] getWeights();                  // 偏好权重
    int[] getPriorityOrder();               // 优先级顺序

    // 理想点与反理想点
    double[] getIdealPoint();               // 各目标的最优值
    double[] getNadirPoint();              // 各目标的最差值

    // Pareto 分析
    List<Integer> getNonDominatedSolutions(); // 非支配解索引
    boolean isParetoOptimal(int index);     // 验证 Pareto 最优性

    // 最终选择
    IVector getSelectedSolution();           // 最终选择的解
    double[] getSelectedObjectiveValues();   // 最终解的目标值

    // 摘要与报告
    String getSummary();                    // 结果摘要
    String getDetailedReport();             // 详细报告
}
```

---

## 算法总览 / Algorithm Overview

### 支持的求解器 / Supported Solvers

| 求解器 / Solver | 类名 / Class | 核心思想 / Core Idea | 适用场景 / Use Case |
|------|------|----------|----------|
| **加权求和法** | `RereWeightedSum` | 线性加权和转化为单目标 | 凸 Pareto 前沿，权重已知 |
| **字典序法** | `RereLexicographic` | 按优先级逐个优化 | 目标有明确优先级 |
| **目标规划法** | `RereGoalProgramming` | 最小化与目标值的偏差 | 目标值可设定 |
| **Pareto 最优法** | `RereParetoOptimal` | 生成完整 Pareto 前沿 | 需要了解全局权衡 |
| **层次分析法 (AHP)** | `RereAHP` | 成对比较矩阵计算权重 | 复杂决策问题 |
| **TOPSIS 法** | `RereTopsis` | 理想解/负理想解距离排序 | 多属性决策 |
| **交互式 STEM 法** | `RereInteractive` | 决策者逐步反馈 | 需要人工干预偏好 |

---

## 求解器详解 / Solver Details

### 1. 加权求和法 / Weighted Sum Method

**数学表述**：

$$\min \sum_{i=1}^k w_i f_i(x) \quad \text{s.t.} \quad Ax \leq b$$

其中 $w_i \geq 0$ 且 $\sum w_i = 1$。

**特点**：
- 简单直观，计算效率高
- 适用于凸 Pareto 前沿
- 无法找到非凸区域的最优解

```java
// 基本用法 / Basic usage
IMclpSolver solver = Opts.mclp.weightedSumMclp();

// 带权重 / With weights
IMclpSolver solver = Opts.mclp.weightedSumMclp(new double[]{0.5, 0.3, 0.2});
```

### 2. 字典序法 / Lexicographic Method

**核心思想**：按优先级逐个优化目标，前一个目标达到最优后才考虑下一个。

**特点**：
- 适用于目标有明确优先级顺序的场景
- 不会牺牲高优先级目标来改善低优先级目标
- 需要决策者明确优先级

```java
// 按优先级优化 / Optimize by priority
IMclpSolver solver = Opts.mclp.lexicographicMclp(new int[]{0, 2, 1});
// 先优化目标 0，再优化目标 2，最后优化目标 1
```

### 3. 目标规划法 / Goal Programming

**数学表述**：

$$\min \sum_{i=1}^k d_i^+ + d_i^- \quad \text{s.t.} \quad f_i(x) + d_i^+ - d_i^- = g_i$$

其中 $g_i$ 是目标值，$d_i^+, d_i^-$ 是正负偏差变量。

**特点**：
- 允许决策者为每个目标设定目标值
- 最小化实际值与目标值的偏差
- 支持加权偏差和优先级

```java
// 带目标值 / With goals
IMclpSolver solver = Opts.mclp.goalProgrammingMclp(new double[]{100, 200, 150});

// 带目标值和权重 / With goals and weights
IMclpSolver solver = Opts.mclp.goalProgrammingMclp(
    new double[]{100, 200, 150},  // 目标值 / Goals
    new double[]{1.0, 0.8, 0.5}  // 权重 / Weights
);
```

### 4. Pareto 最优法 / Pareto Optimal Method

**核心思想**：系统探索权重空间，生成完整的 Pareto 前沿，帮助决策者了解各目标之间的权衡关系。

**特点**：
- 提供全局视角的最优解集
- 支持决策者选择满意解
- 计算量较大

```java
// 生成 Pareto 前沿 / Generate Pareto frontier
IMclpSolver solver = Opts.mclp.paretoMclp(100);  // 采样 100 个点

// 默认采样点数 / Default sampling
IMclpSolver solver = Opts.mclp.paretoMclp();
```

### 5. 层次分析法 (AHP) / Analytic Hierarchy Process

**核心思想**：通过成对比较矩阵计算目标权重，基于 Saaty 的 1-9 标度。

**特点**：
- 适合复杂决策问题
- 考虑目标间的相对重要性
- 需要决策者进行成对比较

```java
// 基本用法 / Basic usage
IMclpSolver solver = Opts.mclp.ahpMclp();

// 带比较矩阵 / With comparison matrix
// 比较矩阵：c[i][j] 表示目标 i 相对于目标 j 的重要性
IMatrix comparisonMatrix = IMatrix.of(new double[][]{
    {1.0, 3.0, 5.0},
    {1.0/3.0, 1.0, 2.0},
    {1.0/5.0, 1.0/2.0, 1.0}
});
IMclpSolver solver = Opts.mclp.ahpMclp(comparisonMatrix);
```

### 6. TOPSIS 法 / Technique for Order Preference by Similarity to Ideal Solution

**核心思想**：选择距离理想解最近且距离负理想解最远的方案。

**特点**：
- 基于几何距离的排序方法
- 直观易懂
- 需要权重信息

```java
// 基本用法 / Basic usage
IMclpSolver solver = Opts.mclp.topsisMclp();

// 带权重 / With weights
IMclpSolver solver = Opts.mclp.topsisMclp(new double[]{0.4, 0.3, 0.3});
```

### 7. 交互式 STEM 法 / STEM Interactive Method

**核心思想**：决策者逐步交互，通过反馈调整偏好找到满意解。

**特点**：
- 迭代式交互过程
- 决策者全程参与
- 适合复杂偏好结构

```java
// 带决策者回调 / With decision maker callback
IMclpSolver solver = Opts.mclp.interactiveMclp(decisionMaker -> {
    // 决策者返回下一步的权重调整
    return new double[]{0.6, 0.4};
});

// 基本用法 / Basic usage
IMclpSolver solver = Opts.mclp.interactiveMclp();
```

---

## 算法选择指南 / Algorithm Selection Guide

```
你的决策场景是什么？/ What is your decision scenario?
│
├─ 目标有明确优先级顺序 / Clear priority order
│   └─ 字典序法 / Lexicographic Method
│
├─ 知道目标权重或可以确定 / Know weights or can determine
│   │
│   ├─ 权重精确已知 / Weights precisely known
│   │   └─ 加权求和法 / Weighted Sum Method
│   │
│   └─ 需要通过比较确定权重 / Need pairwise comparison
│       └─ AHP（层次分析法）/ AHP
│
├─ 有明确的目标值 / Have specific target values
│   └─ 目标规划法 / Goal Programming
│
├─ 需要了解全局权衡关系 / Need global trade-off view
│   └─ Pareto 最优法 / Pareto Optimal Method
│
├─ 基于距离理想解排序 / Rank by distance to ideal
│   └─ TOPSIS 法 / TOPSIS Method
│
└─ 需要人工交互干预 / Need human interaction
    └─ 交互式 STEM 法 / Interactive STEM Method
```

---

## 性能调优 / Performance Tuning

### Pareto 前沿采样

```java
// 增加采样点以获得更完整的 Pareto 前沿
// Increase samples for more complete Pareto frontier
IMclpSolver solver = Opts.mclp.paretoMclp(500);  // 500 个采样点
```

### AHP 一致性

```java
// 确保成对比较矩阵具有合理的一致性
// Ensure reasonable consistency in comparison matrix
// Saaty 建议 CR < 0.1
```

---

## 参考资料 / References

- Saaty, T. L. (1980). The Analytic Hierarchy Process. *McGraw-Hill*.
- Hwang, C.L. & Yoon, K. (1981). Multiple Attribute Decision Making. *Springer*.
- Steuer, R. E. (1986). Multiple Criteria Optimization. *John Wiley & Sons*.

---

## 相关文档 / Related Documents

- [MCLP 使用示例](../examples/MCLP-Examples.md)
- [优化算法总览](../Optimization-Algorithms.md)