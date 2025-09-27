# 优化算法示例 (Optimization Algorithm Examples)

## 概述 / Overview

本文档提供了 `yishape-math` 包中优化算法的详细使用示例，包括L-BFGS优化器、在线优化器（SGD、Adam）和线搜索算法。

## 基础优化示例 / Basic Optimization Examples

### L-BFGS优化器基础使用 / Basic L-BFGS Optimizer Usage

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;

public class BasicLBFGSExample {
    public static void main(String[] args) {
        System.out.println("=== L-BFGS基础优化示例 / Basic L-BFGS Optimization Example ===");
        
        // 1. 创建L-BFGS优化器 / Create L-BFGS optimizer
        RereLBFGS optimizer = new RereLBFGS();
        
        // 2. 配置优化器参数 / Configure optimizer parameters
        optimizer.setMaxIterations(1000);      // 最大迭代次数 / Maximum iterations
        optimizer.setTolerance(1e-6);          // 收敛容差 / Convergence tolerance
        optimizer.setM(10);                    // 内存大小 / Memory size
        
        System.out.println("优化器配置完成 / Optimizer configuration completed");
        
        // 3. 定义目标函数：Rosenbrock函数 / Define objective function: Rosenbrock function
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                // f(x1, x2) = (1-x1)² + 100(x2-x1²)²
                return Math.pow(1 - x1, 2) + 100 * Math.pow(x2 - x1 * x1, 2);
            }
        };
        
        // 4. 定义梯度函数 / Define gradient function
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double[] grad = new double[2];
                // ∂f/∂x1 = -2(1-x1) - 400x1(x2-x1²)
                grad[0] = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
                // ∂f/∂x2 = 200(x2-x1²)
                grad[1] = 200 * (x2 - x1 * x1);
                
                return Linalg.vector(grad);
            }
        };
        
        // 5. 设置初始点 / Set initial point
        IVector initX = Linalg.vector(new double[]{-1.0, -1.0});
        System.out.println("初始点: " + initX);
        System.out.println("初始函数值: " + objFun.computeObjective(initX));
        
        // 6. 执行优化 / Execute optimization
        System.out.println("\n开始优化... / Starting optimization...");
        
        Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        // 7. 输出结果 / Output results
        System.out.println("优化完成! / Optimization completed!");
        System.out.println("最优值: " + optimalValue);
        System.out.println("最优点: " + optimalPoint);
        System.out.println("理论最优点: [1.0, 1.0] / Theoretical optimal point: [1.0, 1.0]");
        
        // 8. 验证结果 / Verify results
        double error = (Double) optimalPoint.subtract(Linalg.vector(new double[]{1.0, 1.0})).norm2();
        System.out.println("与理论最优点的误差: " + error);
    }
}
```

### 在线SGD优化器使用 / Online SGD Optimizer Usage

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereOnlineSGD;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;

public class OnlineSGDExample {
    public static void main(String[] args) {
        System.out.println("=== 在线SGD优化示例 / Online SGD Optimization Example ===");
        
        // 1. 创建在线SGD优化器 / Create online SGD optimizer
        RereOnlineSGD optimizer = new RereOnlineSGD(0.1, 0.9, 0.0001);  // 学习率、动量、权重衰减
        
        // 2. 配置优化器参数 / Configure optimizer parameters
        optimizer.setVerbose(true);                    // 详细输出 / Verbose output
        optimizer.setLrDecayRate(0.1);                // 学习率衰减率 / Learning rate decay rate
        optimizer.setLrDecayStep(1000);               // 学习率衰减步长 / Learning rate decay step
        
        // 3. 定义目标函数：二次函数 / Define objective function: quadratic function
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                // f(x1, x2) = (x1-2)² + (x2-3)²
                return Math.pow(x1 - 2.0, 2) + Math.pow(x2 - 3.0, 2);
            }
        };
        
        // 4. 定义梯度函数 / Define gradient function
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double[] grad = new double[2];
                // ∂f/∂x1 = 2(x1-2)
                grad[0] = 2.0 * (x1 - 2.0);
                // ∂f/∂x2 = 2(x2-3)
                grad[1] = 2.0 * (x2 - 3.0);
                
                return Linalg.vector(grad);
            }
        };
        
        // 5. 初始化优化器 / Initialize optimizer
        IVector initialParams = Linalg.vector(new double[]{0.0, 0.0});
        optimizer.initialize(initialParams);
        
        System.out.println("初始参数: " + initialParams);
        System.out.println("初始损失: " + objFun.computeObjective(initialParams));
        
        // 6. 在线学习循环 / Online learning loop
        System.out.println("\n开始在线学习... / Starting online learning...");
        
        int maxIterations = 100;
        double tolerance = 1e-6;
        
        for (int i = 0; i < maxIterations; i++) {
            // 计算梯度和损失 / Compute gradient and loss
            IVector currentParams = optimizer.getCurrentParams();
            IVector gradient = grdFun.computeGradient(currentParams);
            double loss = objFun.computeObjective(currentParams);
            
            // 执行一步优化 / Perform one optimization step
            IVector updatedParams = optimizer.step(gradient, loss);
            
            // 检查收敛 / Check convergence
            if (loss < tolerance) {
                System.out.println("在线SGD在第" + (i+1) + "步收敛");
                break;
            }
            
            // 每10步输出一次进度 / Output progress every 10 steps
            if ((i + 1) % 10 == 0) {
                System.out.printf("步骤 %d: 损失 = %.6f, 学习率 = %.6f\n", 
                                 i + 1, loss, optimizer.getCurrentLearningRate());
            }
        }
        
        // 7. 输出最终结果 / Output final results
        IVector finalParams = optimizer.getCurrentParams();
        double finalLoss = objFun.computeObjective(finalParams);
        
        System.out.println("\n在线学习完成! / Online learning completed!");
        System.out.println("最终参数: " + finalParams);
        System.out.println("最终损失: " + finalLoss);
        System.out.println("总步数: " + optimizer.getCurrentStep());
        System.out.println("理论最优解: [2.0, 3.0] / Theoretical optimal solution: [2.0, 3.0]");
        
        // 8. 验证结果 / Verify results
        double error = (Double) finalParams.subtract(Linalg.vector(new double[]{2.0, 3.0})).norm2();
        System.out.println("与理论最优解的误差: " + error);
    }
}
```

### 在线Adam优化器使用 / Online Adam Optimizer Usage

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereOnlineAdam;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;

public class OnlineAdamExample {
    public static void main(String[] args) {
        System.out.println("=== 在线Adam优化示例 / Online Adam Optimization Example ===");
        
        // 1. 创建在线Adam优化器 / Create online Adam optimizer
        RereOnlineAdam optimizer = new RereOnlineAdam(0.001, 0.9, 0.999, 1e-8, 0.0001);
        
        // 2. 配置优化器参数 / Configure optimizer parameters
        optimizer.setVerbose(true);                    // 详细输出 / Verbose output
        optimizer.setAmsgrad(false);                   // 是否使用AMSGrad变体 / Whether to use AMSGrad variant
        optimizer.setLrDecayRate(0.1);                // 学习率衰减率 / Learning rate decay rate
        optimizer.setLrDecayStep(1000);               // 学习率衰减步长 / Learning rate decay step
        
        // 3. 定义目标函数：Rosenbrock函数 / Define objective function: Rosenbrock function
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                // f(x1, x2) = (1-x1)² + 100(x2-x1²)²
                return Math.pow(1 - x1, 2) + 100 * Math.pow(x2 - x1 * x1, 2);
            }
        };
        
        // 4. 定义梯度函数 / Define gradient function
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double[] grad = new double[2];
                // ∂f/∂x1 = -2(1-x1) - 400x1(x2-x1²)
                grad[0] = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
                // ∂f/∂x2 = 200(x2-x1²)
                grad[1] = 200 * (x2 - x1 * x1);
                
                return Linalg.vector(grad);
            }
        };
        
        // 5. 初始化优化器 / Initialize optimizer
        IVector initialParams = Linalg.vector(new double[]{-1.0, -1.0});
        optimizer.initialize(initialParams);
        
        System.out.println("初始参数: " + initialParams);
        System.out.println("初始损失: " + objFun.computeObjective(initialParams));
        
        // 6. 在线学习循环 / Online learning loop
        System.out.println("\n开始在线学习... / Starting online learning...");
        
        int maxIterations = 200;
        double tolerance = 1e-6;
        
        for (int i = 0; i < maxIterations; i++) {
            // 计算梯度和损失 / Compute gradient and loss
            IVector currentParams = optimizer.getCurrentParams();
            IVector gradient = grdFun.computeGradient(currentParams);
            double loss = objFun.computeObjective(currentParams);
            
            // 执行一步优化 / Perform one optimization step
            IVector updatedParams = optimizer.step(gradient, loss);
            
            // 检查收敛 / Check convergence
            if (loss < tolerance) {
                System.out.println("在线Adam在第" + (i+1) + "步收敛");
                break;
            }
            
            // 每20步输出一次进度 / Output progress every 20 steps
            if ((i + 1) % 20 == 0) {
                System.out.printf("步骤 %d: 损失 = %.6f, 学习率 = %.6f\n", 
                                 i + 1, loss, optimizer.getCurrentLearningRate());
            }
        }
        
        // 7. 输出最终结果 / Output final results
        IVector finalParams = optimizer.getCurrentParams();
        double finalLoss = objFun.computeObjective(finalParams);
        
        System.out.println("\n在线学习完成! / Online learning completed!");
        System.out.println("最终参数: " + finalParams);
        System.out.println("最终损失: " + finalLoss);
        System.out.println("总步数: " + optimizer.getCurrentStep());
        System.out.println("理论最优解: [1.0, 1.0] / Theoretical optimal solution: [1.0, 1.0]");
        
        // 8. 验证结果 / Verify results
        double error = (Double) finalParams.subtract(Linalg.vector(new double[]{1.0, 1.0})).norm2();
        System.out.println("与理论最优解的误差: " + error);
    }
}
```

### 多峰函数优化 / Multi-modal Function Optimization

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;

public class MultiModalOptimizationExample {
    public static void main(String[] args) {
        System.out.println("=== 多峰函数优化示例 / Multi-modal Function Optimization Example ===");
        
        // 定义多峰函数：Himmelblau函数 / Define multi-modal function: Himmelblau function
        // f(x,y) = (x²+y-11)² + (x+y²-7)²
        IObjectiveFunction himmelblauFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double term1 = x1 * x1 + x2 - 11;
                double term2 = x1 + x2 * x2 - 7;
                
                return term1 * term1 + term2 * term2;
            }
        };
        
        IGradientFunction himmelblauGrad = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double[] grad = new double[2];
                
                // ∂f/∂x1 = 2(x1²+x2-11)(2x1) + 2(x1+x2²-7)
                grad[0] = 2 * (x1 * x1 + x2 - 11) * (2 * x1) + 2 * (x1 + x2 * x2 - 7);
                // ∂f/∂x2 = 2(x1²+x2-11) + 2(x1+x2²-7)(2x2)
                grad[1] = 2 * (x1 * x1 + x2 - 11) + 2 * (x1 + x2 * x2 - 7) * (2 * x2);
                
                return Linalg.vector(grad);
            }
        };
        
        // Himmelblau函数的四个全局最小值点 / Four global minima of Himmelblau function
        double[][] globalMinima = {
            {3.0, 2.0},    // 全局最小值点1 / Global minimum point 1
            {-2.805118, 3.131312},  // 全局最小值点2 / Global minimum point 2
            {-3.779310, -3.283186}, // 全局最小值点3 / Global minimum point 3
            {3.584428, -1.848126}   // 全局最小值点4 / Global minimum point 4
        };
        
        RereLBFGS optimizer = new RereLBFGS();
        optimizer.setMaxIterations(500);
        optimizer.setTolerance(1e-5);
        
        // 从不同初始点开始优化 / Start optimization from different initial points
        double[][] initialPoints = {
            {0.0, 0.0},    // 原点 / Origin
            {5.0, 5.0},    // 右上角 / Upper right
            {-5.0, -5.0},  // 左下角 / Lower left
            {5.0, -5.0},   // 右下角 / Lower right
            {-5.0, 5.0}    // 左上角 / Upper left
        };
        
        System.out.println("从不同初始点优化Himmelblau函数 / Optimize Himmelblau function from different initial points");
        System.out.println("理论全局最小值: 0.0 / Theoretical global minimum: 0.0\n");
        
        for (int i = 0; i < initialPoints.length; i++) {
            IVector initPoint = Linalg.vector(initialPoints[i]);
            System.out.println("初始点 " + (i + 1) + ": " + initPoint);
            
            Tuple2<Double, IVector> result = optimizer.optimize(initPoint, himmelblauFun, himmelblauGrad);
            
            double optimalValue = result._1;
            IVector optimalPoint = result._2;
            
            System.out.println("  最优点: " + optimalPoint);
            System.out.println("  最优值: " + optimalValue);
            
            // 找到最近的全局最小值点 / Find nearest global minimum point
            double minDistance = Double.MAX_VALUE;
            int nearestMinIndex = -1;
            
            for (int j = 0; j < globalMinima.length; j++) {
                double distance = (Double) optimalPoint.subtract(Linalg.vector(globalMinima[j])).norm2();
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestMinIndex = j;
                }
            }
            
            System.out.println("  收敛到全局最小值点 " + (nearestMinIndex + 1) + 
                             " (距离: " + minDistance + ")");
            System.out.println();
        }
    }
}
```

## 高级优化示例 / Advanced Optimization Examples

### 拉格朗日乘数法约束优化 / Lagrange Multiplier Constrained Optimization

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.constraint.LagrangeMultiplierSolver;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;

public class LagrangeMultiplierExample {
    public static void main(String[] args) {
        System.out.println("=== 拉格朗日乘数法约束优化示例 / Lagrange Multiplier Constrained Optimization Example ===");
        
        // 定义等式约束：x1 + x2 = 1 / Define equality constraint: x1 + x2 = 1
        IMatrix A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});
        IVector b_eq = Linalg.vector(new double[]{1.0});
        
        // 创建拉格朗日乘数求解器 / Create Lagrange multiplier solver
        LagrangeMultiplierSolver solver = new LagrangeMultiplierSolver(A_eq, b_eq);
        
        // 可选：设置参数 / Optional: Set parameters
        solver.setPenaltyFactor(1000.0);      // 惩罚因子 / Penalty factor
        solver.setMaxPenaltyIterations(100);  // 最大惩罚迭代次数 / Maximum penalty iterations
        solver.setPenaltyIncreaseRate(10.0);  // 惩罚增长率 / Penalty increase rate
        
        // 定义目标函数：最小化 x1² + x2² / Define objective function: minimize x1² + x2²
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                return x1 * x1 + x2 * x2;
            }
        };
        
        // 定义梯度函数 / Define gradient function
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                return Linalg.vector(new double[]{2 * x1, 2 * x2});
            }
        };
        
        // 初始点 / Initial point
        IVector initX = Linalg.vector(new double[]{0.5, 0.5});
        
        System.out.println("目标函数: minimize x1² + x2²");
        System.out.println("约束条件: x1 + x2 = 1");
        System.out.println("初始点: " + initX);
        System.out.println("初始函数值: " + objFun.computeObjective(initX));
        
        // 执行约束优化 / Execute constrained optimization
        System.out.println("\n开始约束优化... / Starting constrained optimization...");
        
        Tuple2<Double, IVector> result = solver.optimize(initX, objFun, grdFun);
        
        double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("\n约束优化完成! / Constrained optimization completed!");
        System.out.println("最优值: " + optimalValue);
        System.out.println("最优点: " + optimalPoint);
        System.out.println("理论最优解: [0.5, 0.5] / Theoretical optimal solution: [0.5, 0.5]");
        System.out.println("理论最优值: 0.5 / Theoretical optimal value: 0.5");
        
        // 验证约束满足情况 / Verify constraint satisfaction
        double constraintValue = optimalPoint.get(0).doubleValue() + optimalPoint.get(1).doubleValue();
        System.out.println("约束验证 x1 + x2 = " + constraintValue + " (应该等于1)");
        
        // 验证结果 / Verify results
        double error = Math.abs(optimalValue - 0.5);
        System.out.println("与理论最优值的误差: " + error);
    }
}
```

### 线性规划示例 / Linear Programming Examples


**重要提示 / Important Note:**
- 本库的线性规划求解器求解的是**最小化问题** / The linear programming solver in this library solves **minimization problems**
- 如果要求解最大化问题，需要将目标函数系数取负数 / To solve maximization problems, negate the objective function coefficients
- 求解器返回的最优值也需要取负数才是真正的最大值 / The optimal value returned by the solver also needs to be negated to get the true maximum value

#### 单纯形法示例 / Simplex Method Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.linpg.SimplexLinProgSolver;

public class SimplexLinProgExample {
    public static void main(String[] args) {
        System.out.println("=== 单纯形法线性规划示例 / Simplex Method Linear Programming Example ===");
        
        // 创建单纯形法求解器 / Create simplex solver
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // 定义线性规划问题 / Define linear programming problem
        // minimize 2x1 + 3x2
        // subject to x1 + x2 = 5, x1 ≥ 0, x2 ≥ 0
        IVector c = Linalg.vector(new double[]{2.0, 3.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});
        IVector b_eq = Linalg.vector(new double[]{5.0});
        
        System.out.println("线性规划问题:");
        System.out.println("minimize 2x1 + 3x2");
        System.out.println("subject to x1 + x2 = 5, x1 ≥ 0, x2 ≥ 0");
        
        // 求解 / Solve
        System.out.println("\n开始单纯形法求解... / Starting simplex method solving...");
        
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        double optimalValue = result._1;
        IVector solution = result._2;
        
        System.out.println("\n单纯形法求解完成! / Simplex method solving completed!");
        System.out.println("最优解: " + solution);
        System.out.println("最优值: " + optimalValue);
        System.out.println("理论最优解: [5.0, 0.0] / Theoretical optimal solution: [5.0, 0.0]");
        System.out.println("理论最优值: 10.0 / Theoretical optimal value: 10.0");
        
        // 验证约束满足情况 / Verify constraint satisfaction
        double constraintValue = solution.get(0).doubleValue() + solution.get(1).doubleValue();
        System.out.println("约束验证 x1 + x2 = " + constraintValue + " (应该等于5)");
        System.out.println("非负约束验证: x1 = " + solution.get(0) + " ≥ 0, x2 = " + solution.get(1) + " ≥ 0");
    }
}
```

#### 内点法示例 / Interior Point Method Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.linpg.InteriorPointLinProgSolver;

public class InteriorPointLinProgExample {
    public static void main(String[] args) {
        System.out.println("=== 内点法线性规划示例 / Interior Point Method Linear Programming Example ===");
        
        // 创建内点法求解器 / Create interior point solver
        InteriorPointLinProgSolver solver = new InteriorPointLinProgSolver();
        
        // 定义线性规划问题 / Define linear programming problem
        // 将不等式约束转换为等式约束形式
        // minimize 2x1 + 3x2
        // subject to x1 + x2 = 5, x1 ≥ 0, x2 ≥ 0
        IVector c = Linalg.vector(new double[]{2.0, 3.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});
        IVector b_eq = Linalg.vector(new double[]{5.0});
        
        System.out.println("线性规划问题:");
        System.out.println("minimize 2x1 + 3x2");
        System.out.println("subject to x1 + x2 = 5, x1 ≥ 0, x2 ≥ 0");
        
        // 求解 / Solve
        System.out.println("\n开始内点法求解... / Starting interior point method solving...");
        
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        double optimalValue = result._1;
        IVector solution = result._2;
        
        System.out.println("\n内点法求解完成! / Interior point method solving completed!");
        System.out.println("最优解: " + solution);
        System.out.println("最优值: " + optimalValue);
        System.out.println("理论最优解: [5.0, 0.0] / Theoretical optimal solution: [5.0, 0.0]");
        System.out.println("理论最优值: 10.0 / Theoretical optimal value: 10.0");
        
        // 验证约束满足情况 / Verify constraint satisfaction
        double constraint = solution.get(0).doubleValue() + solution.get(1).doubleValue();
        System.out.println("约束验证:");
        System.out.println("  x1 + x2 = " + constraint + " (应等于 5.0)");
        System.out.println("  x1 = " + solution.get(0) + " ≥ 0, x2 = " + solution.get(1) + " ≥ 0");
    }
}
```

### 整数规划示例 / Integer Programming Examples

#### 纯整数规划示例 / Pure Integer Programming Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.linpg.RereIntegerProg;
import com.reremouse.lab.math.optimize.linpg.SimplexLinProgSolver;
import com.reremouse.lab.util.Tuple2;

public class PureIntegerProgrammingExample {
    public static void main(String[] args) {
        // 问题描述 / Problem Description:
        // 最大化 / Maximize: 3x1 + 2x2
        // 约束条件 / Subject to:
        //   x1 + x2 ≤ 4
        //   2x1 + x2 ≤ 6
        //   x1, x2 ≥ 0 且为整数 / x1, x2 ≥ 0 and integers
        
        // 构造目标函数 / Construct objective function
        IVector c = Linalg.vector(new double[]{3.0, 2.0});
        
        // 构造约束矩阵 / Construct constraint matrix
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {1.0, 1.0},  // x1 + x2 ≤ 4
            {2.0, 1.0}   // 2x1 + x2 ≤ 6
        });
        
        // 构造约束右端向量 / Construct constraint right-hand side
        IVector b_ub = Linalg.vector(new double[]{4.0, 6.0});
        
        // 创建整数规划求解器 / Create integer programming solver
        RereIntegerProg solver = new RereIntegerProg(new SimplexLinProgSolver());
        
        // 设置所有变量为整数变量 / Set all variables as integer variables
        solver.setAllVariablesInteger();
        
        // 设置算法参数 / Set algorithm parameters
        solver.setMaxDepth(20);
        solver.setGapTolerance(1e-6);
        solver.setTolerance(1e-9);
        
        // 求解 / Solve
        Tuple2<Double, IVector> result = solver.solve(c, A_ub, b_ub, null, null);
        IVector solution = result.getSecond();
        double optimalValue = result.getFirst();
        
        // 输出结果 / Output results
        System.out.println("=== 纯整数规划示例 / Pure Integer Programming Example ===");
        System.out.println("最优解 / Optimal solution: " + solution);
        System.out.println("最优值 / Optimal value: " + optimalValue);
        System.out.println("理论最优解 / Theoretical optimal solution: [2.0, 2.0]");
        System.out.println("理论最优值 / Theoretical optimal value: 10.0");
        
        // 验证整数约束 / Verify integer constraints
        System.out.println("\n约束验证 / Constraint verification:");
        for (int i = 0; i < solution.length(); i++) {
            double value = (Double)solution.get(i);
            boolean isInteger = Math.abs(value - Math.round(value)) < 1e-9;
            System.out.println("  x" + (i+1) + " = " + value + " (整数: " + isInteger + ")");
        }
        
        // 验证线性约束 / Verify linear constraints
        IVector Ax = A_ub.mmul(solution);
        System.out.println("\n线性约束验证 / Linear constraint verification:");
        for (int i = 0; i < Ax.length(); i++) {
            boolean satisfied = (Double)Ax.get(i) <= (Double)b_ub.get(i) + 1e-9;
            System.out.println("  约束 " + (i+1) + ": " + Ax.get(i) + " ≤ " + b_ub.get(i) + " (" + satisfied + ")");
        }
    }
}
```

#### 混合整数规划示例 / Mixed Integer Programming Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.linpg.RereIntegerProg;
import com.reremouse.lab.math.optimize.linpg.InteriorPointLinProgSolver;
import com.reremouse.lab.util.Tuple2;

public class MixedIntegerProgrammingExample {
    public static void main(String[] args) {
        // 问题描述 / Problem Description:
        // 最大化 / Maximize: 4x1 + 3x2 + 2x3
        // 约束条件 / Subject to:
        //   2x1 + x2 + x3 ≤ 8
        //   x1 + 2x2 + x3 ≤ 7
        //   x1, x2 ≥ 0 且为整数 / x1, x2 ≥ 0 and integers
        //   x3 ≥ 0 (连续变量 / continuous variable)
        
        // 构造目标函数 / Construct objective function
        IVector c = Linalg.vector(new double[]{4.0, 3.0, 2.0});
        
        // 构造约束矩阵 / Construct constraint matrix
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {2.0, 1.0, 1.0},  // 2x1 + x2 + x3 ≤ 8
            {1.0, 2.0, 1.0}   // x1 + 2x2 + x3 ≤ 7
        });
        
        // 构造约束右端向量 / Construct constraint right-hand side
        IVector b_ub = Linalg.vector(new double[]{8.0, 7.0});
        
        // 创建整数规划求解器（使用内点法作为基础求解器）
        // Create integer programming solver (using interior-point method as base solver)
        RereIntegerProg solver = new RereIntegerProg(new InteriorPointLinProgSolver());
        
        // 只设置前两个变量为整数变量，第三个变量为连续变量
        // Set only the first two variables as integer variables, the third as continuous
        solver.addIntegerVariables(0, 1);
        
        // 设置算法参数 / Set algorithm parameters
        solver.setMaxDepth(25);
        solver.setGapTolerance(1e-8);
        solver.setTolerance(1e-10);
        
        // 求解 / Solve
        Tuple2<Double, IVector> result = solver.solve(c, A_ub, b_ub, null, null);
        IVector solution = result.getSecond();
        double optimalValue = result.getFirst();
        
        // 输出结果 / Output results
        System.out.println("=== 混合整数规划示例 / Mixed Integer Programming Example ===");
        System.out.println("最优解 / Optimal solution: " + solution);
        System.out.println("最优值 / Optimal value: " + optimalValue);
        
        // 验证变量类型 / Verify variable types
        System.out.println("\n变量类型验证 / Variable type verification:");
        System.out.println("  x1 = " + solution.get(0) + " (整数变量 / Integer variable)");
        System.out.println("  x2 = " + solution.get(1) + " (整数变量 / Integer variable)");
        System.out.println("  x3 = " + solution.get(2) + " (连续变量 / Continuous variable)");
        
        // 验证整数约束 / Verify integer constraints
        System.out.println("\n整数约束验证 / Integer constraint verification:");
        for (int i = 0; i < 2; i++) {  // 只检查前两个变量 / Only check first two variables
            double value = (Double)solution.get(i);
            boolean isInteger = Math.abs(value - Math.round(value)) < 1e-9;
            System.out.println("  x" + (i+1) + " 是整数: " + isInteger);
        }
        
        // 验证线性约束 / Verify linear constraints
        IVector Ax = A_ub.mmul(solution);
        System.out.println("\n线性约束验证 / Linear constraint verification:");
        for (int i = 0; i < Ax.length(); i++) {
            boolean satisfied = (Double)Ax.get(i) <= (Double)b_ub.get(i) + 1e-9;
            System.out.println("  约束 " + (i+1) + ": " + Ax.get(i) + " ≤ " + b_ub.get(i) + " (" + satisfied + ")");
        }
    }
}
```

#### 0-1整数规划示例 / Binary Integer Programming Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.linpg.RereIntegerProg;
import com.reremouse.lab.math.optimize.linpg.SimplexLinProgSolver;
import com.reremouse.lab.util.Tuple2;

public class ComplexIntegerProgrammingExample {
    public static void main(String[] args) {
        System.out.println("=== 0-1整数规划示例 / Complex Integer Programming Example ===");
        
        // 简化的背包问题示例
        // 物品价值: [3, 4, 5]
        // 物品重量: [2, 3, 4]
        // 背包容量: 5
        // 目标：最大化价值
        
        // 构造目标函数（最大化价值转换为最小化问题）
        // 由于求解器是最小化问题，需要将价值系数取负数
        IVector c = Linalg.vector(new double[]{-3.0, -4.0, -5.0});
        
        // 构造约束矩阵（重量约束）
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {2.0, 3.0, 4.0}  // 重量约束
        });
        
        // 构造约束右端向量（背包容量）
        IVector b_ub = Linalg.vector(new double[]{5.0});
        
        // 创建整数规划求解器
        RereIntegerProg solver = new RereIntegerProg();
        
        // 设置所有变量为整数变量（0-1变量）
        solver.setAllVariablesBinary();
        
        // 设置算法参数
        solver.setMaxDepth(20);
        solver.setGapTolerance(1e-6);
        solver.setTolerance(1e-9);
        
        System.out.println("正在求解整数规划问题...");
        System.out.println("目标函数: maximize 3*x1 + 4*x2 + 5*x3");
        System.out.println("约束条件: 2*x1 + 3*x2 + 4*x3 <= 5");
        System.out.println("变量约束: x1, x2, x3 为 0-1 变量");
        
        // 求解
        Tuple2<Double, IVector> result = solver.solve(c, A_ub, b_ub, null, null);
        
        if (result == null) {
            System.out.println("未找到可行解");
            return;
        }
        
        IVector solution = result.getSecond();
        double optimalValue = -result.getFirst(); // 转换回正值
        
        // 输出结果
        System.out.println("最优解: " + solution);
        System.out.println("最优价值: " + optimalValue);
        
        // 分析解的含义
        System.out.println("\n解的分析:");
        String[] items = {"物品1", "物品2", "物品3"};
        double[] values = {3.0, 4.0, 5.0};
        double[] weights = {2.0, 3.0, 4.0};
        
        double totalWeight = 0;
        double totalValue = 0;
        
        for (int i = 0; i < solution.length(); i++) {
            int selected = (int) Math.round((Double)solution.get(i));
            if (selected == 1) {
                System.out.println("  选择 " + items[i] + " (价值: " + values[i] + ", 重量: " + weights[i] + ")");
                totalWeight += weights[i];
                totalValue += values[i];
            }
        }
        
        System.out.println("\n总重量: " + totalWeight + " <= 5");
        System.out.println("总价值: " + totalValue);
        
        // 验证约束
        IVector Ax = A_ub.mmul(solution);
        boolean feasible = (Double)Ax.get(0) <= (Double)b_ub.get(0) + 1e-9;
        System.out.println("约束满足: " + feasible);
        
        // 验证0-1约束
        System.out.println("\n0-1约束验证:");
        boolean allBinary = true;
        for (int i = 0; i < solution.length(); i++) {
            double value = (Double)solution.get(i);
            boolean isBinary = Math.abs(value) < 1e-9 || Math.abs(value - 1.0) < 1e-9;
            allBinary &= isBinary;
            System.out.println("  x" + (i+1) + " = " + value + " (0-1变量: " + isBinary + ")");
        }
        System.out.println("所有变量都是0-1变量: " + allBinary);
    }
}
```

#### 求解器性能比较示例 / Solver Performance Comparison Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.linpg.RereIntegerProg;
import com.reremouse.lab.math.optimize.linpg.SimplexLinProgSolver;
import com.reremouse.lab.math.optimize.linpg.InteriorPointLinProgSolver;
import com.reremouse.lab.util.Tuple2;

public class SolverPerformanceComparisonExample {
    public static void main(String[] args) {
        // 定义测试问题 / Define test problem
        IVector c = Linalg.vector(new double[]{5.0, 3.0, 4.0, 2.0});
        IMatrix A = Linalg.matrix(new double[][]{
            {2.0, 1.0, 1.0, 3.0},
            {1.0, 3.0, 2.0, 1.0},
            {3.0, 1.0, 2.0, 2.0}
        });
        IVector b = Linalg.vector(new double[]{12.0, 11.0, 15.0});
        int[] integerVars = {0, 1, 2, 3};
        
        System.out.println("=== 求解器性能比较示例 / Solver Performance Comparison Example ===");
        
        // 测试单纯形法作为基础求解器 / Test Simplex method as base solver
        testSolver("单纯形法 / Simplex Method", 
                  new RereIntegerProg(new SimplexLinProgSolver()), 
                  c, A, b, integerVars);
        
        System.out.println();
        
        // 测试内点法作为基础求解器 / Test Interior-point method as base solver
        testSolver("内点法 / Interior-Point Method", 
                  new RereIntegerProg(new InteriorPointLinProgSolver()), 
                  c, A, b, integerVars);
    }
    
    private static void testSolver(String solverName, RereIntegerProg solver, 
                                 IVector c, IMatrix A, IVector b, int[] integerVars) {
        System.out.println("--- " + solverName + " ---");
        
        // 设置整数变量 / Set integer variables
        solver.addIntegerVariables(integerVars);
        
        // 设置算法参数 / Set algorithm parameters
        solver.setMaxDepth(25);
        solver.setGapTolerance(1e-6);
        solver.setTolerance(1e-9);
        
        // 记录开始时间 / Record start time
        long startTime = System.nanoTime();
        
        try {
            // 求解 / Solve
            Tuple2<Double, IVector> result = solver.solve(c, A, b);
            IVector solution = result.getSecond();
            double optimalValue = result.getFirst();
            
            // 记录结束时间 / Record end time
            long endTime = System.nanoTime();
            double elapsedTime = (endTime - startTime) / 1_000_000.0; // 转换为毫秒 / Convert to milliseconds
            
            // 输出结果 / Output results
            System.out.println("求解时间 / Solving time: " + String.format("%.2f", elapsedTime) + " ms");
            System.out.println("最优解 / Optimal solution: " + solution);
            System.out.println("最优值 / Optimal value: " + optimalValue);
            
            // 验证解的质量 / Verify solution quality
            boolean isInteger = true;
            for (int i : integerVars) {
                double value = solution.get(i).doubleValue();
                if (Math.abs(value - Math.round(value)) > 1e-9) {
                    isInteger = false;
                    break;
                }
            }
            System.out.println("整数约束满足 / Integer constraints satisfied: " + isInteger);
            
        } catch (Exception e) {
            long endTime = System.nanoTime();
            double elapsedTime = (endTime - startTime) / 1_000_000.0;
            System.out.println("求解失败 / Solving failed: " + e.getMessage());
            System.out.println("失败时间 / Failure time: " + String.format("%.2f", elapsedTime) + " ms");
        }
    }
}
```

### 共轭梯度法示例 / Conjugate Gradient Method Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereConjugateGradient;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;

public class ConjugateGradientExample {
    public static void main(String[] args) {
        System.out.println("=== 共轭梯度法优化示例 / Conjugate Gradient Method Optimization Example ===");
        
        // 创建共轭梯度优化器 / Create conjugate gradient optimizer
        RereConjugateGradient optimizer = new RereConjugateGradient();
        
        // 设置参数（可选）/ Set parameters (optional)
        // optimizer.setMaxIterations(1000);     // 最大迭代次数 / Maximum iterations
        // optimizer.setTolerance(1e-8);         // 收敛容差 / Convergence tolerance
        // optimizer.setRestartThreshold(0.5);   // 重启阈值 / Restart threshold
        
        // 定义正定矩阵A / Define positive definite matrix A
        IMatrix A = Linalg.matrix(new double[][]{
            {4.0, 1.0},
            {1.0, 3.0}
        });
        
        // 定义向量b / Define vector b
        IVector b = Linalg.vector(new double[]{1.0, 2.0});
        
        // 定义二次目标函数：f(x) = 0.5 * x^T * A * x - b^T * x
        // Define quadratic objective function: f(x) = 0.5 * x^T * A * x - b^T * x
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                IVector Ax = A.mmul(x);
                return 0.5 * x.innerProduct(Ax) - b.innerProduct(x);
            }
        };
        
        // 定义梯度函数：∇f(x) = A * x - b / Define gradient function: ∇f(x) = A * x - b
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                return A.mmul(x).subtract(b);
            }
        };
        
        // 初始点 / Initial point
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        
        System.out.println("二次目标函数: f(x) = 0.5 * x^T * A * x - b^T * x");
        System.out.println("矩阵A: " + A);
        System.out.println("向量b: " + b);
        System.out.println("初始点: " + initX);
        System.out.println("初始函数值: " + objFun.computeObjective(initX));
        
        // 执行优化 / Execute optimization
        System.out.println("\n开始共轭梯度法优化... / Starting conjugate gradient optimization...");
        
        Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("\n共轭梯度法优化完成! / Conjugate gradient optimization completed!");
        System.out.println("最优值: " + optimalValue);
        System.out.println("最优点: " + optimalPoint);
        
        // 理论最优解：x* = A^(-1) * b / Theoretical optimal solution: x* = A^(-1) * b
        IVector theoreticalSolution = A.inverse().mmul(b);
        double theoreticalValue = objFun.computeObjective(theoreticalSolution);
        
        System.out.println("理论最优解: " + theoreticalSolution);
        System.out.println("理论最优值: " + theoreticalValue);
        
        // 验证结果 / Verify results
        double error = (Double) optimalPoint.subtract(theoreticalSolution).norm2();
        System.out.println("与理论最优解的误差: " + error);
    }
}
```

### DFP算法示例 / DFP Algorithm Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereDFP;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;

public class DFPExample {
    public static void main(String[] args) {
        System.out.println("=== DFP算法优化示例 / DFP Algorithm Optimization Example ===");
        
        // 创建DFP优化器 / Create DFP optimizer
        RereDFP optimizer = new RereDFP();
        
        // 设置参数（可选）/ Set parameters (optional)
        // optimizer.setMaxIterations(1000);     // 最大迭代次数 / Maximum iterations
        // optimizer.setTolerance(1e-6);         // 收敛容差 / Convergence tolerance
        
        // 定义Rosenbrock函数 / Define Rosenbrock function
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                return (1 - x1) * (1 - x1) + 100 * (x2 - x1 * x1) * (x2 - x1 * x1);
            }
        };
        
        // 定义梯度函数 / Define gradient function
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double[] grad = new double[2];
                grad[0] = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
                grad[1] = 200 * (x2 - x1 * x1);
                
                return Linalg.vector(grad);
            }
        };
        
        // 初始点 / Initial point
        IVector initX = Linalg.vector(new double[]{-1.0, -1.0});
        
        System.out.println("Rosenbrock函数: f(x1, x2) = (1-x1)² + 100(x2-x1²)²");
        System.out.println("初始点: " + initX);
        System.out.println("初始函数值: " + objFun.computeObjective(initX));
        
        // 执行优化 / Execute optimization
        System.out.println("\n开始DFP算法优化... / Starting DFP algorithm optimization...");
        
        Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("\nDFP算法优化完成! / DFP algorithm optimization completed!");
        System.out.println("最优值: " + optimalValue);
        System.out.println("最优点: " + optimalPoint);
        System.out.println("理论最优解: [1.0, 1.0] / Theoretical optimal solution: [1.0, 1.0]");
        System.out.println("理论最优值: 0.0 / Theoretical optimal value: 0.0");
        
        // 验证结果 / Verify results
        double error = (Double) optimalPoint.subtract(Linalg.vector(new double[]{1.0, 1.0})).norm2();
        System.out.println("与理论最优解的误差: " + error);
    }
}
```

### 最速下降法示例 / Steepest Descent Method Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereSteepestDescent;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;

public class SteepestDescentExample {
    public static void main(String[] args) {
        System.out.println("=== 最速下降法优化示例 / Steepest Descent Method Optimization Example ===");
        
        // 创建最速下降优化器 / Create steepest descent optimizer
        RereSteepestDescent optimizer = new RereSteepestDescent();
        
        // 设置参数（可选）/ Set parameters (optional)
        // optimizer.setMaxIterations(10000);    // 最大迭代次数 / Maximum iterations
        // optimizer.setTolerance(1e-6);         // 收敛容差 / Convergence tolerance
        // optimizer.setInitialStepSize(0.01);   // 初始步长 / Initial step size
        
        // 定义二次函数 / Define quadratic function
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                return (x1 - 2) * (x1 - 2) + (x2 - 3) * (x2 - 3);
            }
        };
        
        // 定义梯度函数 / Define gradient function
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double[] grad = new double[2];
                grad[0] = 2 * (x1 - 2);
                grad[1] = 2 * (x2 - 3);
                
                return Linalg.vector(grad);
            }
        };
        
        // 初始点 / Initial point
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        
        System.out.println("二次函数: f(x1, x2) = (x1-2)² + (x2-3)²");
        System.out.println("初始点: " + initX);
        System.out.println("初始函数值: " + objFun.computeObjective(initX));
        
        // 执行优化 / Execute optimization
        System.out.println("\n开始最速下降法优化... / Starting steepest descent optimization...");
        
        Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("\n最速下降法优化完成! / Steepest descent optimization completed!");
        System.out.println("最优值: " + optimalValue);
        System.out.println("最优点: " + optimalPoint);
        System.out.println("理论最优解: [2.0, 3.0] / Theoretical optimal solution: [2.0, 3.0]");
        System.out.println("理论最优值: 0.0 / Theoretical optimal value: 0.0");
        
        // 验证结果 / Verify results
        double error = (Double) optimalPoint.subtract(Linalg.vector(new double[]{2.0, 3.0})).norm2();
        System.out.println("与理论最优解的误差: " + error);
    }
}
```

### 混合优化示例 / Mixed Optimization Examples

#### 多目标优化示例 / Multi-objective Optimization Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;

public class MultiObjectiveOptimizationExample {
    public static void main(String[] args) {
        System.out.println("=== 多目标优化示例 / Multi-objective Optimization Example ===");
        
        // 创建L-BFGS优化器 / Create L-BFGS optimizer
        RereLBFGS optimizer = new RereLBFGS();
        // 设置参数（可选）/ Set parameters (optional)
        // optimizer.setMaxIterations(1000);
        // optimizer.setTolerance(1e-6);
        
        // 权重系数 / Weight coefficients
        double w1 = 0.6;  // 第一个目标的权重 / Weight for first objective
        double w2 = 0.4;  // 第二个目标的权重 / Weight for second objective
        
        // 定义多目标优化问题：最小化 w1*(x1-1)² + w2*(x2-2)²
        // Define multi-objective optimization: minimize w1*(x1-1)² + w2*(x2-2)²
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double obj1 = (x1 - 1) * (x1 - 1);  // 第一个目标 / First objective
                double obj2 = (x2 - 2) * (x2 - 2);  // 第二个目标 / Second objective
                
                return w1 * obj1 + w2 * obj2;  // 加权组合 / Weighted combination
            }
        };
        
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double grad1 = w1 * 2 * (x1 - 1);
                double grad2 = w2 * 2 * (x2 - 2);
                
                return Linalg.vector(new double[]{grad1, grad2});
            }
        };
        
        // 初始点 / Initial point
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        
        System.out.println("多目标优化问题:");
        System.out.println("目标1: (x1-1)², 权重: " + w1);
        System.out.println("目标2: (x2-2)², 权重: " + w2);
        System.out.println("初始点: " + initX);
        
        // 执行优化 / Execute optimization
        Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("\n多目标优化完成! / Multi-objective optimization completed!");
        System.out.println("最优点: " + optimalPoint);
        System.out.println("加权目标函数值: " + optimalValue);
        System.out.println("理论最优解: [1.0, 2.0] / Theoretical optimal solution: [1.0, 2.0]");
    }
}
```

#### 鲁棒优化示例 / Robust Optimization Example

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;
import java.util.Random;

public class RobustOptimizationExample {
    private static final Random random = new Random(42);
    
    public static void main(String[] args) {
        System.out.println("=== 鲁棒优化示例 / Robust Optimization Example ===");
        
        // 创建L-BFGS优化器 / Create L-BFGS optimizer
        RereLBFGS optimizer = new RereLBFGS();
        // 设置参数（可选）/ Set parameters (optional)
        // optimizer.setMaxIterations(1000);
        // optimizer.setTolerance(1e-6);
        
        // 不确定性参数 / Uncertainty parameters
        double noiseLevel = 0.1;
        int numScenarios = 100;
        
        // 定义鲁棒优化问题：最小化期望目标函数值
        // Define robust optimization: minimize expected objective value
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double expectedValue = 0.0;
                
                // 蒙特卡洛采样计算期望值 / Monte Carlo sampling for expected value
                for (int i = 0; i < numScenarios; i++) {
                    // 添加噪声 / Add noise
                    double noise1 = random.nextGaussian() * noiseLevel;
                    double noise2 = random.nextGaussian() * noiseLevel;
                    
                    double x1_noisy = x1 + noise1;
                    double x2_noisy = x2 + noise2;
                    
                    // 计算带噪声的目标函数值 / Calculate noisy objective value
                    double scenarioValue = (x1_noisy - 1) * (x1_noisy - 1) + (x2_noisy - 1) * (x2_noisy - 1);
                    expectedValue += scenarioValue;
                }
                
                return expectedValue / numScenarios;
            }
        };
        
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double grad1 = 0.0;
                double grad2 = 0.0;
                
                // 数值梯度计算 / Numerical gradient calculation
                double h = 1e-6;
                
                IVector xPlus1 = Linalg.vector(new double[]{x1 + h, x2});
                IVector xMinus1 = Linalg.vector(new double[]{x1 - h, x2});
                grad1 = (objFun.computeObjective(xPlus1) - objFun.computeObjective(xMinus1)) / (2 * h);
                
                IVector xPlus2 = Linalg.vector(new double[]{x1, x2 + h});
                IVector xMinus2 = Linalg.vector(new double[]{x1, x2 - h});
                grad2 = (objFun.computeObjective(xPlus2) - objFun.computeObjective(xMinus2)) / (2 * h);
                
                return Linalg.vector(new double[]{grad1, grad2});
            }
        };
        
        // 初始点 / Initial point
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        
        System.out.println("鲁棒优化问题:");
        System.out.println("目标函数: E[(x1+ε1-1)² + (x2+ε2-1)²]");
        System.out.println("噪声水平: " + noiseLevel);
        System.out.println("场景数量: " + numScenarios);
        System.out.println("初始点: " + initX);
        
        // 执行优化 / Execute optimization
        System.out.println("\n开始鲁棒优化... / Starting robust optimization...");
        
        Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("\n鲁棒优化完成! / Robust optimization completed!");
        System.out.println("最优点: " + optimalPoint);
        System.out.println("期望目标函数值: " + optimalValue);
        System.out.println("理论最优解: [1.0, 1.0] / Theoretical optimal solution: [1.0, 1.0]");
    }
}
```

### 约束优化问题 / Constrained Optimization Problems

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;

public class ConstrainedOptimizationExample {
    public static void main(String[] args) {
        System.out.println("=== 约束优化问题示例 / Constrained Optimization Example ===");
        
        // 定义带约束的目标函数 / Define constrained objective function
        // 最小化 f(x,y) = x² + y²
        // 约束条件: x + y >= 1, x >= 0, y >= 0
        IObjectiveFunction constrainedFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                // 基础目标函数 / Basic objective function
                double objective = x1 * x1 + x2 * x2;
                
                // 惩罚函数方法处理约束 / Penalty function method for constraints
                double penalty = 0;
                
                // 约束1: x + y >= 1 / Constraint 1: x + y >= 1
                if (x1 + x2 < 1) {
                    penalty += 1000 * (1 - x1 - x2) * (1 - x1 - x2);
                }
                
                // 约束2: x >= 0 / Constraint 2: x >= 0
                if (x1 < 0) {
                    penalty += 1000 * x1 * x1;
                }
                
                // 约束3: y >= 0 / Constraint 3: y >= 0
                if (x2 < 0) {
                    penalty += 1000 * x2 * x2;
                }
                
                return objective + penalty;
            }
        };
        
        IGradientFunction constrainedGrad = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                
                double[] grad = new double[2];
                
                // 基础梯度 / Basic gradient
                grad[0] = 2 * x1;
                grad[1] = 2 * x2;
                
                // 约束梯度 / Constraint gradients
                if (x1 + x2 < 1) {
                    grad[0] += 2000 * (x1 + x2 - 1);
                    grad[1] += 2000 * (x1 + x2 - 1);
                }
                
                if (x1 < 0) {
                    grad[0] += 2000 * x1;
                }
                
                if (x2 < 0) {
                    grad[1] += 2000 * x2;
                }
                
                return Linalg.vector(grad);
            }
        };
        
        RereLBFGS optimizer = new RereLBFGS();
        // 设置参数（可选）/ Set parameters (optional)
        // optimizer.setMaxIterations(1000);
        // optimizer.setTolerance(1e-6);
        
        // 从可行域内的点开始优化 / Start optimization from feasible point
        IVector initPoint = Linalg.vector(new double[]{0.5, 0.5});
        System.out.println("初始点: " + initPoint);
        System.out.println("初始函数值: " + constrainedFun.computeObjective(initPoint));
        
        Tuple2<Double, IVector> result = optimizer.optimize(initPoint, constrainedFun, constrainedGrad);
        
        double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("\n优化结果 / Optimization Results:");
        System.out.println("最优点: " + optimalPoint);
        System.out.println("最优值: " + optimalValue);
        
        // 验证约束满足情况 / Verify constraint satisfaction
        System.out.println("\n约束验证 / Constraint Verification:");
        System.out.println("x + y >= 1: " + (optimalPoint.get(0).doubleValue() + optimalPoint.get(1).doubleValue()) + " >= 1");
        System.out.println("x >= 0: " + optimalPoint.get(0).doubleValue() + " >= 0");
        System.out.println("y >= 0: " + optimalPoint.get(1).doubleValue() + " >= 0");
        
        // 理论最优解: (0.5, 0.5) / Theoretical optimal solution: (0.5, 0.5)
        double theoreticalOptimal = 0.5 * 0.5 + 0.5 * 0.5;
        System.out.println("理论最优值: " + theoreticalOptimal);
        System.out.println("误差: " + Math.abs(optimalValue - theoreticalOptimal));
    }
}
```

### 大规模优化问题 / Large-scale Optimization Problems

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.newton.RereLBFGS;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.util.Tuple2;

public class LargeScaleOptimizationExample {
    public static void main(String[] args) {
        System.out.println("=== 大规模优化问题示例 / Large-scale Optimization Example ===");
        
        // 定义大规模二次函数 / Define large-scale quadratic function
        // f(x) = 0.5 * x^T * A * x + b^T * x
        int dimension = 1000;
        
        // 创建随机正定矩阵A / Create random positive definite matrix A
        double[][] A = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (i == j) {
                    A[i][j] = 1.0 + Math.random() * 9.0; // 对角线元素 / Diagonal elements
                } else {
                    A[i][j] = Math.random() * 0.1 - 0.05; // 非对角线元素 / Off-diagonal elements
                }
            }
        }
        
        // 创建随机向量b / Create random vector b
        double[] b = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            b[i] = Math.random() * 2.0 - 1.0;
        }
        
        IMatrix AMatrix = Linalg.matrix(A);
        IVector bVector = Linalg.vector(b);
        
        // 定义目标函数 / Define objective function
        IObjectiveFunction largeScaleFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                // f(x) = 0.5 * x^T * A * x + b^T * x
                IVector Ax = AMatrix.mmul(x);
                double quadraticTerm = (Double) x.innerProduct(Ax) * 0.5;
                double linearTerm = (Double) bVector.innerProduct(x);
                
                return quadraticTerm + linearTerm;
            }
        };
        
        // 定义梯度函数 / Define gradient function
        IGradientFunction largeScaleGrad = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                // ∇f(x) = A * x + b
                return AMatrix.mmul(x).add(bVector);
            }
        };
        
        // 创建优化器 / Create optimizer
        RereLBFGS optimizer = new RereLBFGS();
        // 设置参数（可选）/ Set parameters (optional)
        // optimizer.setMaxIterations(2000);
        // optimizer.setTolerance(1e-8);
        // optimizer.setM(20); // 增加内存大小以提高性能 / Increase memory size for better performance
        
        // 随机初始点 / Random initial point
        double[] initArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            initArray[i] = Math.random() * 2.0 - 1.0;
        }
        IVector initPoint = Linalg.vector(initArray);
        
        System.out.println("问题维度: " + dimension);
        System.out.println("初始函数值: " + largeScaleFun.computeObjective(initPoint));
        System.out.println("开始大规模优化... / Starting large-scale optimization...");
        
        long startTime = System.currentTimeMillis();
        Tuple2<Double, IVector> result = optimizer.optimize(initPoint, largeScaleFun, largeScaleGrad);
        long endTime = System.currentTimeMillis();
        
        double optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("优化完成! / Optimization completed!");
        System.out.println("最优值: " + optimalValue);
        System.out.println("优化时间: " + (endTime - startTime) + " ms");
        
        // 验证最优性条件 / Verify optimality conditions
        IVector gradient = largeScaleGrad.computeGradient(optimalPoint);
        double gradientNorm = (Double) gradient.norm2();
        System.out.println("梯度范数: " + gradientNorm);
        System.out.println("是否满足一阶最优性条件: " + (gradientNorm < 1e-6));
    }
}
```

## 线搜索算法示例 / Line Search Algorithm Examples

### Armijo线搜索 / Armijo Line Search

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.RereLineSearch;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;

public class LineSearchExample {
    public static void main(String[] args) {
        System.out.println("=== 线搜索算法示例 / Line Search Algorithm Example ===");
        
        // 创建线搜索器 / Create line searcher
        RereLineSearch lineSearcher = new RereLineSearch(1e-4, 0.9, 1.0);  // c1, c2, initialStepSize
        
        // 定义搜索方向 / Define search direction
        IVector searchDirection = Linalg.vector(new double[]{-1.0, -1.0});
        
        // 定义目标函数和梯度函数 / Define objective and gradient functions
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public double computeObjective(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                return x1 * x1 + x2 * x2;
            }
        };
        
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x1 = x.get(0).doubleValue();
                double x2 = x.get(1).doubleValue();
                return Linalg.vector(new double[]{2 * x1, 2 * x2});
            }
        };
        
        // 当前点 / Current point
        IVector currentPoint = Linalg.vector(new double[]{2.0, 2.0});
        IVector currentGradient = grdFun.computeGradient(currentPoint);
        
        System.out.println("当前点: " + currentPoint);
        System.out.println("当前梯度: " + currentGradient);
        System.out.println("搜索方向: " + searchDirection);
        
        // 执行线搜索 / Execute line search
        double stepSize = lineSearcher.search(currentPoint, searchDirection, 
                                            objFun, grdFun, currentGradient);
        
        System.out.println("最优步长: " + stepSize);
        
        // 计算新点 / Calculate new point
        IVector newPoint = currentPoint.add(searchDirection.multiplyScalar(stepSize));
        System.out.println("新点: " + newPoint);
        System.out.println("函数值变化: " + (objFun.computeObjective(newPoint) - objFun.computeObjective(currentPoint)));
    }
}
```

## 总结 / Summary

本文档展示了 `yishape-math` 优化算法包的各种使用示例，包括：

### 1. 基础优化算法 / Basic Optimization Algorithms
- **L-BFGS**：适用于大规模无约束优化问题，具有超线性收敛速度
- **Online SGD**：适用于在线学习和大数据场景，支持动量和权重衰减
- **Online Adam**：结合动量和自适应学习率的在线优化算法

### 2. 约束优化算法 / Constrained Optimization Algorithms
- **拉格朗日乘数法**：处理等式约束优化问题的经典方法
- **惩罚函数方法**：将约束优化转换为无约束优化问题

### 3. 线性规划算法 / Linear Programming Algorithms
- **单纯形法**：求解线性规划问题的经典算法
- **内点法**：现代线性规划求解器的核心算法

### 4. 牛顿类方法 / Newton-type Methods
- **共轭梯度法**：适用于大规模二次优化问题
- **DFP算法**：拟牛顿方法，适用于中等规模非线性优化
- **最速下降法**：简单的一阶优化方法

### 5. 高级优化技术 / Advanced Optimization Techniques
- **多目标优化**：处理多个目标函数的优化问题
- **鲁棒优化**：考虑不确定性的优化方法
- **大规模优化**：处理高维优化问题
- **线搜索**：精确控制步长选择

### 6. 算法选择指南 / Algorithm Selection Guide
- **无约束优化**：L-BFGS（中大规模）、共轭梯度法（大规模二次问题）、DFP（中等规模）
- **约束优化**：拉格朗日乘数法（等式约束）、惩罚函数法（一般约束）
- **线性规划**：单纯形法（小中规模）、内点法（大规模）
- **在线优化**：Online SGD（简单快速）、Online Adam（自适应学习率）

### 7. 实际应用建议 / Practical Application Recommendations

#### 算法选择 / Algorithm Selection
- 根据问题特性（规模、约束类型、函数性质）选择最适合的优化算法
- 对于大规模问题，优先考虑内存效率高的算法（如L-BFGS、共轭梯度法）
- 对于非凸问题，尝试多个不同的初始点

#### 参数设置 / Parameter Setting
- 合理设置学习率、容差、最大迭代次数等参数
- 对于在线学习场景，使用增量式算法（Online SGD、Online Adam）
- 根据问题的数值特性调整收敛容差

#### 约束处理 / Constraint Handling
- 对于约束优化问题，选择合适的约束处理方法
- 等式约束优先使用拉格朗日乘数法
- 复杂约束可以考虑惩罚函数方法

#### 收敛监控 / Convergence Monitoring
- 监控优化过程，确保算法收敛到合理解
- 注意数值稳定性和条件数问题
- 选择合适的初始点可以显著影响优化效果

### 8. 性能优化提示 / Performance Optimization Tips
- 对于大规模问题，使用内存高效的算法
- 对于线性规划问题，内点法通常比单纯形法更快
- 对于二次规划问题，共轭梯度法是最佳选择
- 对于一般非线性优化，L-BFGS是首选算法

这些示例为用户提供了完整的优化算法使用指南，涵盖了从基础到高级的各种优化场景，帮助解决各种实际优化问题。

---

**优化算法示例** - 从基础到高级，掌握数学优化的精髓！
