# 优化算法示例 (Optimization Algorithm Examples)

## 概述 / Overview

本文档提供了 `yishape-math` 包中优化算法的详细使用示例，包括L-BFGS优化器和线搜索算法。

## 基础优化示例 / Basic Optimization Examples

### L-BFGS优化器基础使用 / Basic L-BFGS Optimizer Usage

```java
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.optimize.RereLBFGS;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;

public class BasicLBFGSExample {
    public static void main(String[] args) {
        System.out.println("=== L-BFGS基础优化示例 / Basic L-BFGS Optimization Example ===");
        
        // 1. 创建L-BFGS优化器 / Create L-BFGS optimizer
        RereLBFGS optimizer = new RereLBFGS();
        
        // 2. 配置优化器参数 / Configure optimizer parameters
        optimizer.setMaxIterations(1000);      // 最大迭代次数 / Maximum iterations
        optimizer.setTolerance(1e-6f);         // 收敛容差 / Convergence tolerance
        optimizer.setMemorySize(10);           // 内存大小 / Memory size
        optimizer.setLineSearchType(LineSearchType.ARMIJO); // 线搜索类型 / Line search type
        
        System.out.println("优化器配置: " + optimizer);
        
        // 3. 定义目标函数：Rosenbrock函数 / Define objective function: Rosenbrock function
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public float compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                // f(x1, x2) = (1-x1)² + 100(x2-x1²)²
                return (1 - x1) * (1 - x1) + 100 * (x2 - x1 * x1) * (x2 - x1 * x1);
            }
        };
        
        // 4. 定义梯度函数 / Define gradient function
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                
                float[] grad = new float[2];
                // ∂f/∂x1 = -2(1-x1) - 400x1(x2-x1²)
                grad[0] = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
                // ∂f/∂x2 = 200(x2-x1²)
                grad[1] = 200 * (x2 - x1 * x1);
                
                return IVector.of(grad);
            }
        };
        
        // 5. 设置初始点 / Set initial point
        IVector initX = IVector.of(new float[]{-1.0f, -1.0f});
        System.out.println("初始点: " + initX);
        System.out.println("初始函数值: " + objFun.compute(initX));
        
        // 6. 执行优化 / Execute optimization
        System.out.println("\n开始优化... / Starting optimization...");
        
        Tuple2<Float, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        float optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        // 7. 输出结果 / Output results
        System.out.println("优化完成! / Optimization completed!");
        System.out.println("最优值: " + optimalValue);
        System.out.println("最优点: " + optimalPoint);
        System.out.println("理论最优点: [1.0, 1.0] / Theoretical optimal point: [1.0, 1.0]");
        
        // 8. 验证结果 / Verify results
        float error = optimalPoint.subtract(IVector.of(new float[]{1.0f, 1.0f})).norm2();
        System.out.println("与理论最优点的误差: " + error);
    }
}
```

### 多峰函数优化 / Multi-modal Function Optimization

```java
public class MultiModalOptimizationExample {
    public static void main(String[] args) {
        System.out.println("=== 多峰函数优化示例 / Multi-modal Function Optimization Example ===");
        
        // 定义多峰函数：Himmelblau函数 / Define multi-modal function: Himmelblau function
        // f(x,y) = (x²+y-11)² + (x+y²-7)²
        IObjectiveFunction himmelblauFun = new IObjectiveFunction() {
            @Override
            public float compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                
                float term1 = x1 * x1 + x2 - 11;
                float term2 = x1 + x2 * x2 - 7;
                
                return term1 * term1 + term2 * term2;
            }
        };
        
        IGradientFunction himmelblauGrad = new IGradientFunction() {
            @Override
            public IVector compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                
                float[] grad = new float[2];
                
                // ∂f/∂x1 = 2(x1²+x2-11)(2x1) + 2(x1+x2²-7)
                grad[0] = 2 * (x1 * x1 + x2 - 11) * (2 * x1) + 2 * (x1 + x2 * x2 - 7);
                // ∂f/∂x2 = 2(x1²+x2-11) + 2(x1+x2²-7)(2x2)
                grad[1] = 2 * (x1 * x1 + x2 - 11) + 2 * (x1 + x2 * x2 - 7) * (2 * x2);
                
                return IVector.of(grad);
            }
        };
        
        // Himmelblau函数的四个全局最小值点 / Four global minima of Himmelblau function
        float[][] globalMinima = {
            {3.0f, 2.0f},    // 全局最小值点1 / Global minimum point 1
            {-2.805118f, 3.131312f},  // 全局最小值点2 / Global minimum point 2
            {-3.779310f, -3.283186f}, // 全局最小值点3 / Global minimum point 3
            {3.584428f, -1.848126f}   // 全局最小值点4 / Global minimum point 4
        };
        
        RereLBFGS optimizer = new RereLBFGS();
        optimizer.setMaxIterations(500);
        optimizer.setTolerance(1e-5f);
        
        // 从不同初始点开始优化 / Start optimization from different initial points
        float[][] initialPoints = {
            {0.0f, 0.0f},    // 原点 / Origin
            {5.0f, 5.0f},    // 右上角 / Upper right
            {-5.0f, -5.0f},  // 左下角 / Lower left
            {5.0f, -5.0f},   // 右下角 / Lower right
            {-5.0f, 5.0f}    // 左上角 / Upper left
        };
        
        System.out.println("从不同初始点优化Himmelblau函数 / Optimize Himmelblau function from different initial points");
        System.out.println("理论全局最小值: 0.0 / Theoretical global minimum: 0.0\n");
        
        for (int i = 0; i < initialPoints.length; i++) {
            IVector initPoint = IVector.of(initialPoints[i]);
            System.out.println("初始点 " + (i + 1) + ": " + initPoint);
            
            Tuple2<Float, IVector> result = optimizer.optimize(initPoint, himmelblauFun, himmelblauGrad);
            
            float optimalValue = result._1;
            IVector optimalPoint = result._2;
            
            System.out.println("  最优点: " + optimalPoint);
            System.out.println("  最优值: " + optimalValue);
            
            // 找到最近的全局最小值点 / Find nearest global minimum point
            float minDistance = Float.MAX_VALUE;
            int nearestMinIndex = -1;
            
            for (int j = 0; j < globalMinima.length; j++) {
                float distance = optimalPoint.subtract(IVector.of(globalMinima[j])).norm2();
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
public class ConstrainedOptimizationExample {
    public static void main(String[] args) {
        System.out.println("=== 约束优化问题示例 / Constrained Optimization Example ===");
        
        // 定义带约束的目标函数 / Define constrained objective function
        // 最小化 f(x,y) = x² + y²
        // 约束条件: x + y >= 1, x >= 0, y >= 0
        IObjectiveFunction constrainedFun = new IObjectiveFunction() {
            @Override
            public float compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                
                // 基础目标函数 / Basic objective function
                float objective = x1 * x1 + x2 * x2;
                
                // 惩罚函数方法处理约束 / Penalty function method for constraints
                float penalty = 0;
                
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
            public IVector compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                
                float[] grad = new float[2];
                
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
                
                return IVector.of(grad);
            }
        };
        
        RereLBFGS optimizer = new RereLBFGS();
        optimizer.setMaxIterations(1000);
        optimizer.setTolerance(1e-6f);
        
        // 从可行域内的点开始优化 / Start optimization from feasible point
        IVector initPoint = IVector.of(new float[]{0.5f, 0.5f});
        System.out.println("初始点: " + initPoint);
        System.out.println("初始函数值: " + constrainedFun.compute(initPoint));
        
        Tuple2<Float, IVector> result = optimizer.optimize(initPoint, constrainedFun, constrainedGrad);
        
        float optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("\n优化结果 / Optimization Results:");
        System.out.println("最优点: " + optimalPoint);
        System.out.println("最优值: " + optimalValue);
        
        // 验证约束满足情况 / Verify constraint satisfaction
        System.out.println("\n约束验证 / Constraint Verification:");
        System.out.println("x + y >= 1: " + (optimalPoint.get(0) + optimalPoint.get(1)) + " >= 1");
        System.out.println("x >= 0: " + optimalPoint.get(0) + " >= 0");
        System.out.println("y >= 0: " + optimalPoint.get(1) + " >= 0");
        
        // 理论最优解: (0.5, 0.5) / Theoretical optimal solution: (0.5, 0.5)
        float theoreticalOptimal = 0.5f * 0.5f + 0.5f * 0.5f;
        System.out.println("理论最优值: " + theoreticalOptimal);
        System.out.println("误差: " + Math.abs(optimalValue - theoreticalOptimal));
    }
}
```

### 大规模优化问题 / Large-scale Optimization Problems

```java
public class LargeScaleOptimizationExample {
    public static void main(String[] args) {
        System.out.println("=== 大规模优化问题示例 / Large-scale Optimization Example ===");
        
        // 定义大规模二次函数 / Define large-scale quadratic function
        // f(x) = 0.5 * x^T * A * x + b^T * x
        int dimension = 1000;
        
        // 创建随机正定矩阵A / Create random positive definite matrix A
        float[][] A = new float[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (i == j) {
                    A[i][j] = 1.0f + (float) Math.random() * 9.0f; // 对角线元素 / Diagonal elements
                } else {
                    A[i][j] = (float) (Math.random() * 0.1f - 0.05f); // 非对角线元素 / Off-diagonal elements
                }
            }
        }
        
        // 创建随机向量b / Create random vector b
        float[] b = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            b[i] = (float) (Math.random() * 2.0f - 1.0f);
        }
        
        IMatrix AMatrix = IMatrix.of(A);
        IVector bVector = IVector.of(b);
        
        // 定义目标函数 / Define objective function
        IObjectiveFunction largeScaleFun = new IObjectiveFunction() {
            @Override
            public float compute(IVector x) {
                // f(x) = 0.5 * x^T * A * x + b^T * x
                IVector Ax = AMatrix.mmul(x);
                float quadraticTerm = x.dot(Ax) * 0.5f;
                float linearTerm = bVector.dot(x);
                
                return quadraticTerm + linearTerm;
            }
        };
        
        // 定义梯度函数 / Define gradient function
        IGradientFunction largeScaleGrad = new IGradientFunction() {
            @Override
            public IVector compute(IVector x) {
                // ∇f(x) = A * x + b
                return AMatrix.mmul(x).add(bVector);
            }
        };
        
        // 创建优化器 / Create optimizer
        RereLBFGS optimizer = new RereLBFGS();
        optimizer.setMaxIterations(2000);
        optimizer.setTolerance(1e-8f);
        optimizer.setMemorySize(20); // 增加内存大小以提高性能 / Increase memory size for better performance
        
        // 随机初始点 / Random initial point
        float[] initArray = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            initArray[i] = (float) (Math.random() * 2.0f - 1.0f);
        }
        IVector initPoint = IVector.of(initArray);
        
        System.out.println("问题维度: " + dimension);
        System.out.println("初始函数值: " + largeScaleFun.compute(initPoint));
        System.out.println("开始大规模优化... / Starting large-scale optimization...");
        
        long startTime = System.currentTimeMillis();
        Tuple2<Float, IVector> result = optimizer.optimize(initPoint, largeScaleFun, largeScaleGrad);
        long endTime = System.currentTimeMillis();
        
        float optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("优化完成! / Optimization completed!");
        System.out.println("最优值: " + optimalValue);
        System.out.println("优化时间: " + (endTime - startTime) + " ms");
        
        // 验证最优性条件 / Verify optimality conditions
        IVector gradient = largeScaleGrad.compute(optimalPoint);
        float gradientNorm = gradient.norm2();
        System.out.println("梯度范数: " + gradientNorm);
        System.out.println("是否满足一阶最优性条件: " + (gradientNorm < 1e-6f));
    }
}
```

## 线搜索算法示例 / Line Search Algorithm Examples

### Armijo线搜索 / Armijo Line Search

```java
import com.reremouse.lab.math.optimize.RereLineSearch;
import com.reremouse.lab.math.optimize.LineSearchType;

public class LineSearchExample {
    public static void main(String[] args) {
        System.out.println("=== 线搜索算法示例 / Line Search Algorithm Example ===");
        
        // 创建线搜索器 / Create line searcher
        RereLineSearch lineSearcher = new RereLineSearch();
        lineSearcher.setLineSearchType(LineSearchType.ARMIJO);
        lineSearcher.setMaxIterations(50);
        lineSearcher.setTolerance(1e-6f);
        
        // 定义搜索方向 / Define search direction
        IVector searchDirection = IVector.of(new float[]{-1.0f, -1.0f});
        
        // 定义目标函数和梯度函数 / Define objective and gradient functions
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public float compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                return x1 * x1 + x2 * x2;
            }
        };
        
        IGradientFunction grdFun = new IGradientFunction() {
            @Override
            public IVector compute(IVector x) {
                float x1 = x.get(0);
                float x2 = x.get(1);
                return IVector.of(new float[]{2 * x1, 2 * x2});
            }
        };
        
        // 当前点 / Current point
        IVector currentPoint = IVector.of(new float[]{2.0f, 2.0f});
        IVector currentGradient = grdFun.compute(currentPoint);
        
        System.out.println("当前点: " + currentPoint);
        System.out.println("当前梯度: " + currentGradient);
        System.out.println("搜索方向: " + searchDirection);
        
        // 执行线搜索 / Execute line search
        float stepSize = lineSearcher.search(currentPoint, searchDirection, 
                                          currentGradient, objFun, grdFun);
        
        System.out.println("最优步长: " + stepSize);
        
        // 计算新点 / Calculate new point
        IVector newPoint = currentPoint.add(searchDirection.multiplyByScalar(stepSize));
        System.out.println("新点: " + newPoint);
        System.out.println("函数值变化: " + (objFun.compute(newPoint) - objFun.compute(currentPoint)));
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
