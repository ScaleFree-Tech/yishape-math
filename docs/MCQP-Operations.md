# 多目标二次规划 / Multi-Criteria Quadratic Programming (MCQP)

## 概述 / Overview

多目标二次规划（Multi-Criteria Quadratic Programming，MCQP）是多目标线性规划（MCLP）的扩展，目标函数为二次形式：

$$\min \; f_i(x) = \frac{1}{2} x^T Q_i x + c_i^T x \quad (i = 1, \dots, k)$$

其中 $Q_i$ 为对称半正定矩阵，$c_i$ 为线性项系数向量。MCQP 广泛应用于投资组合优化（马科维茨均值-方差模型）、工程结构设计、资源调度等需要考虑二次代价或风险度量的场景。

Multi-Criteria Quadratic Programming (MCQP) extends MCLP to quadratic objective functions. It is widely used in portfolio optimization (Markowitz mean-variance), engineering design, and resource scheduling where quadratic cost or risk measures are involved.

**入口类 / Entry Point**: `Opts.mcqp`（通过 `McqpWrapper` 提供所有 MCQP 求解器）

---

## 核心接口 / Core Interface

### IMcqpSolver

```java
public interface IMcqpSolver extends Serializable {
    /**
     * 求解多目标二次规划问题
     * Solve multi-criteria quadratic programming problem
     */
    McqpResult solve(IMatrix[] Q, IVector[] c,
                     IMatrix A_ub, IVector b_ub,
                     IMatrix A_eq, IVector b_eq);

    /**
     * 获取求解器类型
     * Get solver type
     */
    McqpSolverType getSolverType();
}
```

### McqpResult

```java
public class McqpResult implements Serializable {
    // 解集信息 / Solution set
    List<IVector> getSolutions();           // 最优解集
    List<double[]> getObjectiveValues();     // 各目标函数值
    double[] getWeights();                  // 偏好权重
    int[] getPriorityOrder();               // 优先级顺序

    // 理想点与反理想点 / Ideal and nadir points
    double[] getIdealPoint();               // 各目标的最优值
    double[] getNadirPoint();              // 各目标的最差值

    // Pareto 分析 / Pareto analysis
    List<Integer> getNonDominatedSolutions(); // 非支配解索引
    boolean isParetoOptimal(int index);     // 验证 Pareto 最优性

    // 最终选择 / Final selection
    IVector getSelectedSolution();           // 最终选择的解
    double[] getSelectedObjectiveValues();   // 最终解的目标值

    // 摘要与报告 / Summary and report
    String getSummary();                    // 结果摘要
    String getDetailedReport();             // 详细报告
}
```

---

## 算法总览 / Algorithm Overview

### 支持的求解器 / Supported Solvers

MCQP 复用了 MCLP 的全部求解策略，针对二次目标函数进行适配：

| 求解器 / Solver | 类名 / Class | 核心思想 / Core Idea | 适用场景 / Use Case |
|------|------|----------|----------|
| **加权求和法** | `RereWeightedSumQp` | 线性加权和转化为单目标二次规划 | 凸 Pareto 前沿，权重已知 |
| **字典序法** | `RereLexicographicQp` | 按优先级逐个优化 | 目标有明确优先级 |
| **目标规划法** | `RereGoalProgrammingQp` | 最小化与目标值的偏差 | 目标值可设定 |
| **Pareto 最优法** | `RereParetoQp` | 生成完整 Pareto 前沿 | 需要了解全局权衡 |
| **层次分析法 (AHP)** | `RereAhpQp` | 成对比较矩阵计算权重 | 复杂决策问题 |
| **TOPSIS 法** | `RereTopsisQp` | 理想解/负理想解距离排序 | 多属性决策 |
| **交互式 STEM 法** | `RereInteractiveQp` | 决策者逐步反馈 | 需要人工干预偏好 |

---

## 快速使用 / Quick Start

### 双目标快速求解 / Bi-Objective Quick Solve

对于最常见的双目标二次规划，可直接使用便捷方法：

```java
// 求解双目标二次规划（加权求和法）
// Solve bi-objective QP (weighted sum)
McqpResult result = Opts.mcqp.solveBiObjective(
    Q1, c1,          // 第一个目标的二次项和线性项 / 1st objective quadratic & linear terms
    Q2, c2,          // 第二个目标的二次项和线性项 / 2nd objective quadratic & linear terms
    A_ub, b_ub,      // 不等式约束 / Inequality constraints
    0.5              // 权重（0 到 1 之间）/ Weight between 0 and 1
);
```

### 通用求解器 / General Solver

```java
// 加权求和法 / Weighted sum
IMcqpSolver solver = Opts.mcqp.weightedSumQp(new double[]{0.5, 0.3, 0.2});

// 字典序法 / Lexicographic
IMcqpSolver solver = Opts.mcqp.lexicographicQp(new int[]{0, 2, 1});

// 目标规划法 / Goal programming
IMcqpSolver solver = Opts.mcqp.goalProgrammingQp(
    new double[]{100, 200, 150},  // 目标值 / Goals
    new double[]{1.0, 0.8, 0.5}  // 权重 / Weights
);

// Pareto 最优法 / Pareto optimal
IMcqpSolver solver = Opts.mcqp.paretoQp(100);  // 采样 100 个点 / 100 samples

// AHP
IMatrix comparisonMatrix = Linalg.matrix(new double[][]{
    {1.0, 3.0, 5.0},
    {1.0/3.0, 1.0, 2.0},
    {1.0/5.0, 1.0/2.0, 1.0}
});
IMcqpSolver solver = Opts.mcqp.ahpQp(comparisonMatrix);

// TOPSIS
IMcqpSolver solver = Opts.mcqp.topsisQp(new double[]{0.4, 0.3, 0.3});

// 交互式 STEM
IMcqpSolver solver = Opts.mcqp.interactiveQp(10);  // 最大 10 轮迭代
```

### 执行求解 / Execute Solving

```java
IMatrix[] Q = new IMatrix[]{Q1, Q2, Q3};
IVector[] c = new IVector[]{c1, c2, c3};

McqpResult result = solver.solve(Q, c, A_ub, b_ub, A_eq, b_eq);

// 获取最终选择的解
IVector bestSolution = result.getSelectedSolution();
double[] objectives = result.getSelectedObjectiveValues();
```

---

## 与 MCLP 的关系 / Relationship with MCLP

MCQP 与 MCLP 共享相同的求解策略框架，区别在于：

| 特性 / Feature | MCLP | MCQP |
|------|------|------|
| 目标函数 / Objective | 线性 $c^T x$ | 二次 $\frac{1}{2} x^T Q x + c^T x$ |
| 应用场景 / Use Case | 资源分配、生产计划 | 投资组合、风险优化、结构力学 |
| 入口 / Entry | `Opts.mclp` | `Opts.mcqp` |
| 求解器策略 / Solvers | 7 种 / 7 methods | 7 种（复用 MCLP 策略）/ 7 methods (same strategies) |

当 $Q_i = 0$ 时，MCQP 退化为 MCLP。/ When $Q_i = 0$, MCQP degenerates to MCLP.

---

## 算法选择指南 / Algorithm Selection Guide

MCQP 的算法选择与 MCLP 完全一致，参见 [MCLP/README.md](../MCLP/README.md) 的「算法选择指南」章节。

---

## 相关文档 / Related Documents

- [MCLP 多目标线性规划](../MCLP/README.md)
- [优化算法总览](../Optimization-Algorithms.md)
