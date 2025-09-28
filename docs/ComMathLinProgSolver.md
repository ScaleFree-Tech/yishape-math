# ComMathLinProgSolver - 基于Apache Commons Math3的线性规划求解器

## 简介

`ComMathLinProgSolver` 是一个基于 Apache Commons Math3 库实现的线性规划求解器。它实现了 `ILinProgSolver` 接口，可以求解带有非负约束的线性规划问题。

## 功能特点

- 基于 Apache Commons Math3 的 Simplex 算法
- 支持等式约束的线性规划问题
- 自动处理非负约束 (x >= 0)
- 返回详细的优化结果信息

## 使用方法

### 基本用法

```java
// 创建求解器实例
ComMathLinProgSolver solver = new ComMathLinProgSolver();

// 定义目标函数系数 (最小化 c^T * x)
IVector c = Linalg.vector(new double[]{1.0, 1.0});

// 定义等式约束 A_eq * x = b_eq
IMatrix A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});
IVector b_eq = Linalg.vector(new double[]{2.0});

// 求解
OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);

// 检查结果
if (result != null && result.isConverged()) {
    System.out.println("最优值: " + result.getOptimalValue());
    System.out.println("最优解: " + result.getOptimalPoint());
}
```

### 方法签名

```java
public OptResult solveWithNonNegativeEqualConstraints(
    IVector c,        // 目标函数系数向量
    IMatrix A_eq,     // 等式约束矩阵 (可为null)
    IVector b_eq,     // 等式约束右侧向量 (可为null)
    IVector initX     // 初始解向量 (可为null)
)
```

## 示例

### 示例1: 简单线性规划问题

```java
// 最小化: x1 + x2
// 约束: x1 + x2 = 2
//       x1, x2 >= 0

IVector c = Linalg.vector(new double[]{1.0, 1.0});
IMatrix A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});
IVector b_eq = Linalg.vector(new double[]{2.0});

ComMathLinProgSolver solver = new ComMathLinProgSolver();
OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
```

### 示例2: 复杂线性规划问题

```java
// 最小化: 2*x1 + 3*x2
// 约束: x1 + 2*x2 = 4
//       2*x1 + x2 = 5
//       x1, x2 >= 0

IVector c = Linalg.vector(new double[]{2.0, 3.0});
IMatrix A_eq = Linalg.matrix(new double[][]{
    {1.0, 2.0},
    {2.0, 1.0}
});
IVector b_eq = Linalg.vector(new double[]{4.0, 5.0});

ComMathLinProgSolver solver = new ComMathLinProgSolver();
OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
```

## 依赖

- Apache Commons Math3 3.6.1

## 注意事项

1. 该求解器只处理最小化问题
2. 自动应用非负约束 (x >= 0)
3. 只支持等式约束，不直接支持不等式约束
4. 对于无解或不可行的问题，可能返回null或不收敛的结果