# 多目标线性规划使用示例 / Multi-Criteria Linear Programming Examples

## 概述 / Overview

本文档提供多目标线性规划（MCLP）算法的详细使用示例。矩阵与向量推荐通过 `Linalg.matrix` / `Linalg.vector` 创建，与库内其余文档一致。

This document provides detailed usage examples for Multi-Criteria Linear Programming (MCLP) algorithms. Prefer `Linalg.matrix` / `Linalg.vector` for matrix/vector construction, consistent with other library documentation.

---

## 完整工作流示例 / Complete Workflow Examples

### 1. 加权求和法 / Weighted Sum Method

```java
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.optimize.mclp.IMclpSolver;
import com.yishape.lab.math.optimize.mclp.MclpResult;

// 定义目标函数系数（3个目标）
// Define objective coefficients (3 objectives)
IVector[] c = new IVector[]{
    Linalg.vector(new double[]{1.0, 2.0, 3.0}),  // 目标1: min x1 + 2*x2 + 3*x3
    Linalg.vector(new double[]{2.0, 1.0, 2.0}),  // 目标2: min 2*x1 + x2 + 2*x3
    Linalg.vector(new double[]{1.0, 1.0, 1.0})   // 目标3: min x1 + x2 + x3
};

// 定义不等式约束 Ax <= b
// Define inequality constraints Ax <= b
IMatrix A_ub = Linalg.matrix(new double[][]{
    {1.0, 1.0, 1.0},   // x1 + x2 + x3 <= 10
    {2.0, 1.0, 0.0},   // 2*x1 + x2 <= 8
    {0.0, 1.0, 1.0}    // x2 + x3 <= 7
});
IVector b_ub = Linalg.vector(new double[]{10, 8, 7});

// 创建加权求和求解器（权重：0.5, 0.3, 0.2）
// Create weighted sum solver (weights: 0.5, 0.3, 0.2)
IMclpSolver solver = Opts.mclp.weightedSumMclp(new double[]{0.5, 0.3, 0.2});

// 求解
// Solve
MclpResult result = solver.solve(c, A_ub, b_ub);

// 输出结果
// Output result
System.out.println(result.getSummary());
IVector selectedSolution = result.getSelectedSolution();
double[] objValues = result.getSelectedObjectiveValues();
System.out.println("Solution: " + selectedSolution);
System.out.println("Objective Values: " + Arrays.toString(objValues));
```

### 2. 字典序法 / Lexicographic Method

```java
// 定义目标函数
// Define objectives
IVector[] c = new IVector[]{
    Linalg.vector(new double[]{1.0, 2.0}),   // 目标1（最高优先级）
    Linalg.vector(new double[]{0.0, 1.0}),   // 目标2（次优先级）
    Linalg.vector(new double[]{1.0, 1.0})    // 目标3（最低优先级）
};

// 定义约束
// Define constraints
IMatrix A_ub = Linalg.matrix(new double[][]{
    {1.0, 0.0},
    {0.0, 1.0},
    {1.0, 1.0}
});
IVector b_ub = Linalg.vector(new double[]{10, 8, 12});

// 创建字典序求解器：先优化目标0，再优化目标1，最后优化目标2
// Create lexicographic solver: optimize obj0, then obj1, then obj2
IMclpSolver solver = Opts.mclp.lexicographicMclp(new int[]{0, 1, 2});

// 求解
// Solve
MclpResult result = solver.solve(c, A_ub, b_ub);

System.out.println(result.getSummary());
System.out.println("Selected Solution: " + result.getSelectedSolution());
```

### 3. 目标规划法 / Goal Programming

```java
// 定义目标函数
// Define objectives
IVector[] c = new IVector[]{
    Linalg.vector(new double[]{1.0, 2.0}),
    Linalg.vector(new double[]{2.0, 1.0})
};

// 定义约束
// Define constraints
IMatrix A_ub = Linalg.matrix(new double[][]{
    {1.0, 1.0},
    {2.0, 1.0}
});
IVector b_ub = Linalg.vector(new double[]{10, 15});

// 设置目标值和权重
// Set goal values and weights
double[] goals = new double[]{5.0, 8.0};      // 目标值 / Target values
double[] weights = new double[]{1.0, 0.5};    // 权重 / Weights

// 创建目标规划求解器
// Create goal programming solver
IMclpSolver solver = Opts.mclp.goalProgrammingMclp(goals, weights);

// 求解
// Solve
MclpResult result = solver.solve(c, A_ub, b_ub);

System.out.println("Goal Programming Result:");
System.out.println("Goals: " + Arrays.toString(result.getGoals()));
System.out.println("Solution: " + result.getSelectedSolution());
System.out.println("Objective Values: " + Arrays.toString(result.getSelectedObjectiveValues()));
```

---

## Pareto 前沿分析 / Pareto Frontier Analysis

### 生成和可视化 Pareto 前沿

```java
// 创建 Pareto 最优求解器（采样 100 个点）
// Create Pareto optimal solver (sample 100 points)
IMclpSolver solver = Opts.mclp.paretoMclp(100);

// 定义双目标问题
// Define bi-objective problem
IVector[] c = new IVector[]{
    Linalg.vector(new double[]{1.0, 1.0}),   // 目标1
    Linalg.vector(new double[]{1.0, 2.0})    // 目标2
};

IMatrix A_ub = Linalg.matrix(new double[][]{
    {1.0, 0.0},
    {0.0, 1.0},
    {1.0, 1.0}
});
IVector b_ub = Linalg.vector(new double[]{8, 6, 10});

// 求解
// Solve
MclpResult result = solver.solve(c, A_ub, b_ub);

// 获取 Pareto 前沿
// Get Pareto frontier
List<double[]> paretoFront = result.getObjectiveValues();
List<Integer> nonDominatedIndices = result.getNonDominatedSolutions();

System.out.println("=== Pareto Frontier Analysis ===");
System.out.println("Total Solutions: " + result.getNumSolutions());
System.out.println("Non-dominated Solutions: " + nonDominatedIndices.size());

// 打印 Pareto 前沿点
// Print Pareto frontier points
for (int idx : nonDominatedIndices) {
    double[] objVals = result.getObjectiveValues(idx);
    System.out.printf("Pareto Point: [%.3f, %.3f]%n", objVals[0], objVals[1]);
}

// 获取理想点和反理想点
// Get ideal and nadir points
double[] ideal = result.getIdealPoint();
double[] nadir = result.getNadirPoint();
System.out.println("Ideal Point: " + Arrays.toString(ideal));
System.out.println("Nadir Point: " + Arrays.toString(nadir));
```

---

## AHP 层次分析法 / AHP Method

### 使用 AHP 确定权重

```java
// 创建成对比较矩阵（ Saaty 1-9 标度）
// Create pairwise comparison matrix (Saaty 1-9 scale)
// 目标1 vs 目标2: 同等重要 (1)
// 目标1 vs 目标3: 稍微重要 (3)
// 目标2 vs 目标3: 稍微重要 (2)
IMatrix comparisonMatrix = Linalg.matrix(new double[][]{
    {1.0, 1.0, 3.0},
    {1.0, 1.0, 2.0},
    {1.0/3.0, 1.0/2.0, 1.0}
});

// 使用 AHP 求解器
// Use AHP solver
IMclpSolver solver = Opts.mclp.ahpMclp(comparisonMatrix);

// 定义 MCLP 问题
// Define MCLP problem
IVector[] c = new IVector[]{
    Linalg.vector(new double[]{1.0, 2.0, 1.0}),
    Linalg.vector(new double[]{2.0, 1.0, 1.0}),
    Linalg.vector(new double[]{1.0, 1.0, 2.0})
};

IMatrix A_ub = Linalg.matrix(new double[][]{
    {1.0, 1.0, 1.0},
    {2.0, 1.0, 0.0}
});
IVector b_ub = Linalg.vector(new double[]{10, 8});

// 求解
// Solve
MclpResult result = solver.solve(c, A_ub, b_ub);

System.out.println("AHP-based MCLP Result:");
System.out.println("Solution: " + result.getSelectedSolution());
System.out.println("Weights (from AHP): " + Arrays.toString(result.getWeights()));
```

---

## TOPSIS 方法 / TOPSIS Method

### 多属性决策

```java
// 定义目标函数
// Define objectives
IVector[] c = new IVector[]{
    Linalg.vector(new double[]{100.0, 50.0, 30.0}),   // 成本（最小化）
    Linalg.vector(new double[]{80.0, 90.0, 70.0}),   // 质量（最大化）
    Linalg.vector(new double[]{20.0, 30.0, 25.0})    // 时间（最小化）
};

// 定义约束（资源限制）
// Define constraints (resource limits)
IMatrix A_ub = Linalg.matrix(new double[][]{
    {1.0, 1.0, 1.0},   // 总资源 <= 100
    {2.0, 1.0, 1.0}    // 主要资源 <= 80
});
IVector b_ub = Linalg.vector(new double[]{100, 80});

// 权重
// Weights
double[] weights = new double[]{0.4, 0.4, 0.2};  // 成本、质量、时间

// TOPSIS 求解
// TOPSIS solver
IMclpSolver solver = Opts.mclp.topsisMclp(weights);
MclpResult result = solver.solve(c, A_ub, b_ub);

System.out.println("TOPSIS Result:");
System.out.println("Selected Solution: " + result.getSelectedSolution());
System.out.println("Objective Values: " + Arrays.toString(result.getSelectedObjectiveValues()));
```

---

## 交互式 STEM 方法 / Interactive STEM Method

### 决策者反馈循环

```java
import com.yishape.lab.math.optimize.mclp.RereInteractive;

// 定义问题
// Define problem
IVector[] c = new IVector[]{
    Linalg.vector(new double[]{1.0, 2.0}),
    Linalg.vector(new double[]{2.0, 1.0})
};

IMatrix A_ub = Linalg.matrix(new double[][]{
    {1.0, 1.0},
    {2.0, 1.0}
});
IVector b_ub = Linalg.vector(new double[]{10, 15});

// 创建交互式求解器
// Create interactive solver
IMclpSolver solver = Opts.mclp.interactiveMclp(decisionMaker -> {
    // 获取当前解
    IVector currentSolution = decisionMaker.getCurrentSolution();
    double[] currentObjValues = decisionMaker.getCurrentObjectiveValues();

    System.out.println("Current solution: " + currentSolution);
    System.out.println("Current objective values: " + Arrays.toString(currentObjValues));

    // 决策者返回新的权重
    // Decision maker returns new weights
    // 这里演示自动调整权重
    double[] newWeights = new double[]{0.6, 0.4};
    return newWeights;
});

// 求解
// Solve
MclpResult result = solver.solve(c, A_ub, b_ub);

System.out.println("Final Result:");
System.out.println(result.getSummary());
```

---

## 结果分析与可视化 / Result Analysis and Visualization

### 计算多种指标

```java
// 获取所有解及其目标值
// Get all solutions and objective values
List<IVector> solutions = result.getSolutions();
List<double[]> objectiveValues = result.getObjectiveValues();

System.out.println("=== Solution Analysis ===");
System.out.println("Number of solutions: " + solutions.size());
System.out.println("Number of objectives: " + result.getNumObjectives());

// 计算每个解的标准化目标值
// Compute normalized objective values for each solution
for (int i = 0; i < solutions.size(); i++) {
    double[] objs = objectiveValues.get(i);
    IVector sol = solutions.get(i);
    boolean isPareto = result.isParetoOptimal(i);

    System.out.printf("Solution %d:%n", i);
    System.out.printf("  x = %s%n", sol);
    System.out.printf("  f(x) = %s%n", Arrays.toString(objs));
    System.out.printf("  Pareto Optimal: %s%n", isPareto);
}

// 理想点和反理想点
// Ideal and nadir points
double[] ideal = result.computeIdealPoint();
double[] nadir = result.computeNadirPoint();
System.out.println("Ideal Point: " + Arrays.toString(ideal));
System.out.println("Nadir Point: " + Arrays.toString(nadir));

// 多样性指标
// Diversity metric
System.out.printf("Diversity Metric: %.4f%n", result.getDiversityMetric());
System.out.printf("Hypervolume: %.4f%n", result.getHypervolume());
```

### 打印详细报告

```java
// 打印摘要
// Print summary
System.out.println(result.getSummary());

// 打印详细报告
// Print detailed report
System.out.println(result.getDetailedReport());
```

---

## 常见问题处理 / Troubleshooting

### 1. 约束条件冲突 / Conflicting Constraints

```java
// 问题：无可行解
// Problem: No feasible solution

// 解决方案：放宽约束或检查约束设置
// Solution: Relax constraints or check constraint settings
IMatrix A_ub = Linalg.matrix(new double[][]{
    {1.0, 1.0},    // 原始约束太严格
    {2.0, 1.5}     // 调整为更宽松的值
});
IVector b_ub = Linalg.vector(new double[]{15, 20});  // 放宽右端项
```

### 2. 权重选择困难 / Difficulty in Weight Selection

```java
// 问题：不确定如何设置权重
// Problem: Uncertain how to set weights

// 解决方案：使用 AHP 或 Pareto 方法
// Solution: Use AHP or Pareto method
IMclpSolver solver = Opts.mclp.ahpMclp();  // AHP 自动计算权重
// 或 / or
IMclpSolver solver = Opts.mclp.paretoMclp(200);  // 生成 200 个候选解
```

### 3. 结果解释困难 / Difficulty Interpreting Results

```java
// 使用 Pareto 前沿分析
// Use Pareto frontier analysis
MclpResult result = Opts.mclp.paretoMclp(100).solve(c, A_ub, b_ub);

List<Integer> paretoIndices = result.getNonDominatedSolutions();
System.out.println("Pareto Optimal Solutions:");
for (int idx : paretoIndices) {
    double[] objs = result.getObjectiveValues(idx);
    System.out.printf("Index %d: [%s]%n", idx, Arrays.toString(objs));
}
```

---

## 相关文档 / Related Documents

- [MCLP 总览](../MCLP/README.md)
- [优化算法总览](../Optimization-Algorithms.md)