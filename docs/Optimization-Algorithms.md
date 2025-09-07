# 优化算法 (Optimization Algorithms)

## 概述 / Overview

优化算法包提供了各种数学最优化求解器的实现，包括L-BFGS优化器、线搜索等。这些算法为机器学习模型训练提供了强大的优化支持，能够高效地求解各种目标函数的最优解。

The optimization algorithms package provides implementations of various mathematical optimization solvers including L-BFGS optimizer, line search, and more. These algorithms provide powerful optimization support for machine learning model training, efficiently solving optimal solutions for various objective functions.

## 核心接口 / Core Interfaces

### IOptimizer 接口 / IOptimizer Interface

```java
public interface IOptimizer {
    /**
     * 求解数学最优化问题 / Solve mathematical optimization problem
     * @param initX 初始点 / Initial point
     * @param objFun 目标函数计算法 / Objective function calculator
     * @param grdFun 梯度计算法 / Gradient calculator
     * @return 返回最优值及最优点的变量值（向量）/ Returns optimal value and optimal point variable values (vector)
     */
    public Tuple2<Float, IVector> optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun);
}
```

### IObjectiveFunction 接口 / IObjectiveFunction Interface

```java
public interface IObjectiveFunction {
    /**
     * 计算目标函数值 / Calculate objective function value
     * @param x 变量值 / Variable values
     * @return 目标函数值 / Objective function value
     */
    public float compute(IVector x);
}
```

### IGradientFunction 接口 / IGradientFunction Interface

```java
public interface IGradientFunction {
    /**
     * 计算梯度 / Calculate gradient
     * @param x 变量值 / Variable values
     * @return 梯度向量 / Gradient vector
     */
    public IVector compute(IVector x);
}
```

## 主要算法 / Main Algorithms

### 1. L-BFGS优化器 / L-BFGS Optimizer

#### 算法原理 / Algorithm Principles

L-BFGS（Limited-memory BFGS）是一种拟牛顿法优化算法，它通过近似Hessian矩阵的逆矩阵来更新搜索方向，具有以下特点：

- **内存效率** / **Memory Efficient**: 只存储有限数量的向量对，而不是完整的Hessian矩阵
- **收敛速度** / **Convergence Speed**: 超线性收敛，比梯度下降法快
- **适用性** / **Applicability**: 适用于大规模优化问题

#### 数学原理 / Mathematical Principles

L-BFGS算法基于BFGS公式的有限内存版本：

1. **搜索方向计算** / **Search Direction Calculation**:
   ```
   p_k = -H_k * ∇f(x_k)
   ```

2. **BFGS更新公式** / **BFGS Update Formula**:
   ```
   H_{k+1} = (I - ρ_k * s_k * y_k^T) * H_k * (I - ρ_k * y_k * s_k^T) + ρ_k * s_k * s_k^T
   ```
   其中：
   - `s_k = x_{k+1} - x_k`
   - `y_k = ∇f(x_{k+1}) - ∇f(x_k)`
   - `ρ_k = 1 / (y_k^T * s_k)`

3. **有限内存实现** / **Limited Memory Implementation**:
   - 只存储最近的m个向量对 `(s_i, y_i)`
   - 通过双循环算法高效计算搜索方向

#### 使用示例 / Usage Examples

```java
// 创建L-BFGS优化器 / Create L-BFGS optimizer
RereLBFGS optimizer = new RereLBFGS();

// 设置参数 / Set parameters
optimizer.setMaxIterations(1000);        // 最大迭代次数 / Maximum iterations
optimizer.setTolerance(1e-6f);           // 收敛容差 / Convergence tolerance
optimizer.setMemorySize(10);             // 内存大小（向量对数量）/ Memory size (number of vector pairs)

// 定义目标函数 / Define objective function
IObjectiveFunction objFun = new IObjectiveFunction() {
    @Override
    public float compute(IVector x) {
        // 示例：Rosenbrock函数 / Example: Rosenbrock function
        float x1 = x.get(0);
        float x2 = x.get(1);
        return (1 - x1) * (1 - x1) + 100 * (x2 - x1 * x1) * (x2 - x1 * x1);
    }
};

// 定义梯度函数 / Define gradient function
IGradientFunction grdFun = new IGradientFunction() {
    @Override
    public IVector compute(IVector x) {
        float x1 = x.get(0);
        float x2 = x.get(1);
        
        float[] grad = new float[2];
        grad[0] = -2 * (1 - x1) - 400 * x1 * (x2 - x1 * x1);
        grad[1] = 200 * (x2 - x1 * x1);
        
        return IVector.of(grad);
    }
};

// 初始点 / Initial point
IVector initX = IVector.of(new float[]{-1.0f, -1.0f});

// 执行优化 / Execute optimization
Tuple2<Float, IVector> result = optimizer.optimize(initX, objFun, grdFun);

// 获取结果 / Get results
float optimalValue = result._1;
IVector optimalPoint = result._2;

System.out.println("最优值: " + optimalValue);
System.out.println("最优点: " + optimalPoint);
```

### 2. 线搜索 / Line Search

#### 算法原理 / Algorithm Principles

线搜索是一种一维搜索方法，用于确定沿搜索方向的最优步长。常用的线搜索方法包括：

- **Armijo线搜索** / **Armijo Line Search**: 基于函数值下降的线搜索
- **Wolfe线搜索** / **Wolfe Line Search**: 同时考虑函数值下降和梯度条件的线搜索
- **Goldstein线搜索** / **Goldstein Line Search**: 基于函数值下降的简单线搜索

#### 数学原理 / Mathematical Principles

线搜索的目标是找到满足以下条件的步长α：

1. **Armijo条件** / **Armijo Condition**:
   ```
   f(x + αp) ≤ f(x) + c₁α∇f(x)^T p
   ```

2. **Wolfe条件** / **Wolfe Condition**:
   ```
   f(x + αp) ≤ f(x) + c₁α∇f(x)^T p
   ∇f(x + αp)^T p ≥ c₂∇f(x)^T p
   ```
   其中 `0 < c₁ < c₂ < 1`

#### 使用示例 / Usage Examples

```java
// 创建线搜索优化器 / Create line search optimizer
RereLineSearch optimizer = new RereLineSearch();

// 设置参数 / Set parameters
optimizer.setMaxIterations(100);         // 最大迭代次数 / Maximum iterations
optimizer.setTolerance(1e-6f);           // 收敛容差 / Convergence tolerance
optimizer.setC1(1e-4f);                  // Armijo条件参数 / Armijo condition parameter
optimizer.setC2(0.9f);                   // Wolfe条件参数 / Wolfe condition parameter

// 定义目标函数 / Define objective function
IObjectiveFunction objFun = new IObjectiveFunction() {
    @Override
    public float compute(IVector x) {
        // 示例：二次函数 / Example: quadratic function
        float x1 = x.get(0);
        float x2 = x.get(1);
        return x1 * x1 + x2 * x2;
    }
};

// 定义梯度函数 / Define gradient function
IGradientFunction grdFun = new IGradientFunction() {
    @Override
    public IVector compute(IVector x) {
        float x1 = x.get(0);
        float x2 = x.get(1);
        
        float[] grad = new float[2];
        grad[0] = 2 * x1;
        grad[1] = 2 * x2;
        
        return IVector.of(grad);
    }
};

// 初始点和搜索方向 / Initial point and search direction
IVector initX = IVector.of(new float[]{1.0f, 1.0f});
IVector searchDir = IVector.of(new float[]{-1.0f, -1.0f});

// 执行线搜索 / Execute line search
float optimalStep = optimizer.lineSearch(initX, searchDir, objFun, grdFun);

System.out.println("最优步长: " + optimalStep);
```

## 高级特性 / Advanced Features

### 1. 参数配置 / Parameter Configuration

#### L-BFGS参数 / L-BFGS Parameters

```java
RereLBFGS optimizer = new RereLBFGS();

// 优化参数 / Optimization parameters
optimizer.setMaxIterations(1000);        // 最大迭代次数 / Maximum iterations
optimizer.setTolerance(1e-6f);           // 收敛容差 / Convergence tolerance
optimizer.setMemorySize(10);             // 内存大小 / Memory size
optimizer.setLineSearchTolerance(1e-4f); // 线搜索容差 / Line search tolerance
optimizer.setMaxLineSearchIterations(20); // 最大线搜索迭代次数 / Maximum line search iterations

// 输出控制 / Output control
optimizer.setVerbose(true);              // 详细输出 / Verbose output
optimizer.setPrintInterval(100);         // 打印间隔 / Print interval
```

#### 线搜索参数 / Line Search Parameters

```java
RereLineSearch optimizer = new RereLineSearch();

// 线搜索参数 / Line search parameters
optimizer.setMaxIterations(100);         // 最大迭代次数 / Maximum iterations
optimizer.setTolerance(1e-6f);           // 收敛容差 / Convergence tolerance
optimizer.setC1(1e-4f);                  // Armijo条件参数 / Armijo condition parameter
optimizer.setC2(0.9f);                   // Wolfe条件参数 / Wolfe condition parameter
optimizer.setInitialStep(1.0f);          // 初始步长 / Initial step size
optimizer.setStepReduction(0.5f);        // 步长缩减因子 / Step reduction factor
```

### 2. 收敛监控 / Convergence Monitoring

```java
// 创建优化器 / Create optimizer
RereLBFGS optimizer = new RereLBFGS();
optimizer.setVerbose(true);

// 自定义收敛监控 / Custom convergence monitoring
optimizer.setConvergenceChecker(new ConvergenceChecker() {
    @Override
    public boolean isConverged(OptimizationState state) {
        // 检查梯度范数 / Check gradient norm
        float gradNorm = state.getGradientNorm();
        if (gradNorm < 1e-6f) {
            System.out.println("梯度收敛 / Gradient converged");
            return true;
        }
        
        // 检查函数值变化 / Check function value change
        float valueChange = Math.abs(state.getCurrentValue() - state.getPreviousValue());
        if (valueChange < 1e-8f) {
            System.out.println("函数值收敛 / Function value converged");
            return true;
        }
        
        return false;
    }
});

// 执行优化 / Execute optimization
Tuple2<Float, IVector> result = optimizer.optimize(initX, objFun, grdFun);
```

### 3. 自定义目标函数 / Custom Objective Functions

#### 线性回归目标函数 / Linear Regression Objective Function

```java
public class LinearRegressionObjective implements IObjectiveFunction, IGradientFunction {
    private IMatrix X;
    private IVector y;
    private float lambda1;  // L1正则化系数 / L1 regularization coefficient
    private float lambda2;  // L2正则化系数 / L2 regularization coefficient
    
    public LinearRegressionObjective(IMatrix X, IVector y, float lambda1, float lambda2) {
        this.X = X;
        this.y = y;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
    }
    
    @Override
    public float compute(IVector w) {
        // 计算预测值 / Calculate predictions
        IVector yPred = X.mmul(w);
        
        // 计算残差 / Calculate residuals
        IVector residuals = yPred.subtract(y);
        
        // 计算均方误差 / Calculate mean squared error
        float mse = residuals.squre().sum() / (2 * X.getRowNum());
        
        // 添加正则化项 / Add regularization terms
        float l1Reg = lambda1 * w.abs().sum();
        float l2Reg = lambda2 * w.squre().sum() / 2;
        
        return mse + l1Reg + l2Reg;
    }
    
    @Override
    public IVector compute(IVector w) {
        // 计算预测值 / Calculate predictions
        IVector yPred = X.mmul(w);
        
        // 计算残差 / Calculate residuals
        IVector residuals = yPred.subtract(y);
        
        // 计算梯度 / Calculate gradient
        IVector grad = X.transpose().mmul(residuals).divideByScalar(X.getRowNum());
        
        // 添加正则化梯度 / Add regularization gradients
        if (lambda1 > 0) {
            IVector l1Grad = w.sign().multiplyByScalar(lambda1);
            grad = grad.add(l1Grad);
        }
        
        if (lambda2 > 0) {
            IVector l2Grad = w.multiplyByScalar(lambda2);
            grad = grad.add(l2Grad);
        }
        
        return grad;
    }
}

// 使用自定义目标函数 / Use custom objective function
LinearRegressionObjective objFun = new LinearRegressionObjective(X, y, 0.01f, 0.1f);
RereLBFGS optimizer = new RereLBFGS();
Tuple2<Float, IVector> result = optimizer.optimize(initW, objFun, objFun);
```

## 性能特性 / Performance Features

### 收敛性能 / Convergence Performance
- **L-BFGS**: 超线性收敛，适合大规模问题 / **L-BFGS**: Superlinear convergence, suitable for large-scale problems
- **线搜索**: 保证函数值下降，提高优化稳定性 / **Line search**: Ensures function value decrease, improves optimization stability
- **内存效率**: 有限内存实现，适合高维问题 / **Memory efficiency**: Limited memory implementation, suitable for high-dimensional problems

### 数值稳定性 / Numerical Stability
- 梯度裁剪防止梯度爆炸 / Gradient clipping prevents gradient explosion
- 步长控制保证收敛性 / Step size control ensures convergence
- 条件数检查提高稳定性 / Condition number checking improves stability

### 并行化支持 / Parallelization Support
- 向量运算并行化 / Vector operation parallelization
- 矩阵运算优化 / Matrix operation optimization
- 多线程支持（未来版本）/ Multi-threading support (future version)

## 注意事项 / Notes

1. **初始点选择** / **Initial Point Selection**: 选择合适的初始点有助于快速收敛 / Choosing appropriate initial points helps with fast convergence
2. **参数调优** / **Parameter Tuning**: 根据问题特点调整优化参数 / Adjust optimization parameters based on problem characteristics
3. **收敛判断** / **Convergence Judgment**: 合理设置收敛条件 / Reasonably set convergence conditions
4. **数值精度** / **Numerical Precision**: 注意浮点数精度问题 / Pay attention to floating-point precision issues

## 扩展性 / Extensibility

优化算法包设计支持扩展：
The optimization algorithms package is designed to support extensions:
- 新的优化算法（如Adam、RMSprop等）/ New optimization algorithms (such as Adam, RMSprop, etc.)
- 自适应参数调整 / Adaptive parameter adjustment
- 分布式优化支持 / Distributed optimization support
- GPU加速优化 / GPU-accelerated optimization

## 应用场景 / Application Scenarios

### 机器学习 / Machine Learning
- 模型参数优化 / Model parameter optimization
- 损失函数最小化 / Loss function minimization
- 特征选择优化 / Feature selection optimization

### 科学计算 / Scientific Computing
- 函数最优化 / Function optimization
- 参数估计 / Parameter estimation
- 数值求解 / Numerical solving

### 工程优化 / Engineering Optimization
- 设计参数优化 / Design parameter optimization
- 控制系统优化 / Control system optimization
- 资源分配优化 / Resource allocation optimization

---

**优化算法** - 数学优化的核心，让求解更高效！

**Optimization Algorithms** - The core of mathematical optimization, making solving more efficient!
