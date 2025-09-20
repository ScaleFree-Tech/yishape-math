# 优化算法示例 (Optimization Algorithm Examples)

## 概述 / Overview

本文档提供了 `yishape-math` 包中优化算法的详细使用示例，包括L-BFGS优化器、在线优化器（SGD、Adam）和线搜索算法。

## 基础优化示例 / Basic Optimization Examples

### L-BFGS优化器基础使用 / Basic L-BFGS Optimizer Usage

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.RereLBFGS;
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
import com.reremouse.lab.math.optimize.RereOnlineSGD;
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
import com.reremouse.lab.math.optimize.RereOnlineAdam;
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
import com.reremouse.lab.math.optimize.RereLBFGS;
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

### 约束优化问题 / Constrained Optimization Problems

```java
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.RereLBFGS;
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
        optimizer.setMaxIterations(1000);
        optimizer.setTolerance(1e-6);
        
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
import com.reremouse.lab.math.optimize.RereLBFGS;
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
        optimizer.setMaxIterations(2000);
        optimizer.setTolerance(1e-8);
        optimizer.setM(20); // 增加内存大小以提高性能 / Increase memory size for better performance
        
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

本文档展示了优化算法的各种使用方法。建议在实际使用中：

1. **选择合适的优化算法** / **Choose appropriate optimization algorithm**
2. **合理设置参数** / **Set parameters reasonably**
3. **注意约束处理** / **Pay attention to constraint handling**
4. **监控收敛性** / **Monitor convergence**

---

**优化算法示例** - 从基础到高级，掌握数学优化的精髓！
