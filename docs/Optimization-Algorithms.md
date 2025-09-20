# 优化算法 (Optimization Algorithms)

## 概述 / Overview

优化算法包提供了各种数学最优化求解器的实现，包括L-BFGS优化器、在线优化器（SGD、Adam）、线搜索等。这些算法为机器学习模型训练提供了强大的优化支持，能够高效地求解各种目标函数的最优解。

The optimization algorithms package provides implementations of various mathematical optimization solvers including L-BFGS optimizer, online optimizers (SGD, Adam), line search, and more. These algorithms provide powerful optimization support for machine learning model training, efficiently solving optimal solutions for various objective functions.

## 核心接口 / Core Interfaces

### IOptimizer 接口 / IOptimizer Interface

批量优化器接口，用于一次性优化问题。

```java
public interface IOptimizer {
    /**
     * 求解数学最优化问题 / Solve mathematical optimization problem
     * @param initX 初始点 / Initial point
     * @param objFun 目标函数计算法 / Objective function calculator
     * @param grdFun 梯度计算法 / Gradient calculator
     * @return 返回最优值及最优点的变量值（向量）/ Returns optimal value and optimal point variable values (vector)
     */
    public Tuple2<Double, IVector> optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun);
}
```

### IOnlineOptimizer 接口 / IOnlineOptimizer Interface

在线优化器接口，用于流式数据处理和增量学习。

```java
public interface IOnlineOptimizer {
    /**
     * 初始化优化器 / Initialize the optimizer
     * @param initialParams 初始参数向量 / Initial parameter vector
     */
    void initialize(IVector initialParams);
    
    /**
     * 执行一步优化更新 / Perform one step of optimization update
     * @param gradient 当前梯度向量 / Current gradient vector
     * @return 更新后的参数向量 / Updated parameter vector
     */
    IVector step(IVector gradient);
    
    /**
     * 执行一步优化更新（带损失值）/ Perform one step of optimization update (with loss value)
     * @param gradient 当前梯度向量 / Current gradient vector
     * @param loss 当前损失值 / Current loss value
     * @return 更新后的参数向量 / Updated parameter vector
     */
    IVector step(IVector gradient, double loss);
    
    /**
     * 获取当前参数 / Get current parameters
     * @return 当前参数向量 / Current parameter vector
     */
    IVector getCurrentParams();
    
    /**
     * 设置当前参数 / Set current parameters
     * @param params 参数向量 / Parameter vector
     */
    void setCurrentParams(IVector params);
    
    /**
     * 获取当前学习率 / Get current learning rate
     * @return 当前学习率 / Current learning rate
     */
    double getCurrentLearningRate();
    
    /**
     * 设置学习率 / Set learning rate
     * @param learningRate 学习率 / Learning rate
     */
    void setLearningRate(double learningRate);
    
    /**
     * 获取当前步数 / Get current step number
     * @return 当前步数 / Current step number
     */
    int getCurrentStep();
    
    /**
     * 重置优化器状态 / Reset optimizer state
     */
    void reset();
    
    /**
     * 检查是否已初始化 / Check if initialized
     * @return 是否已初始化 / Whether initialized
     */
    boolean isInitialized();
    
    /**
     * 克隆优化器 / Clone optimizer
     * @return 克隆的优化器 / Cloned optimizer
     */
    IOnlineOptimizer clone();
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
    public double computeObjective(IVector x);
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
    public IVector computeGradient(IVector x);
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

        System.out.println("最优值: " + optimalValue); // Optimal value
        System.out.println("最优点: " + optimalPoint); // Optimal point
```

### 2. 在线随机梯度下降优化器 / Online Stochastic Gradient Descent Optimizer

#### 算法原理 / Algorithm Principles

在线SGD（Online Stochastic Gradient Descent）是一种适用于流式数据处理和增量学习的优化算法，具有以下特点：

- **在线学习** / **Online Learning**: 支持逐步接收数据样本并更新模型参数
- **内存效率** / **Memory Efficient**: 维护内部状态，支持流式数据处理
- **动量支持** / **Momentum Support**: 支持动量机制加速收敛
- **权重衰减** / **Weight Decay**: 支持L2正则化

#### 数学原理 / Mathematical Principles

在线SGD算法的更新规则：

1. **基础SGD更新** / **Basic SGD Update**:
   ```
   θ_{t+1} = θ_t - α * ∇f(θ_t)
   ```

2. **带动量的SGD** / **SGD with Momentum**:
   ```
   v_t = μ * v_{t-1} + ∇f(θ_t)
   θ_{t+1} = θ_t - α * v_t
   ```

3. **带权重衰减的SGD** / **SGD with Weight Decay**:
   ```
   θ_{t+1} = θ_t - α * (∇f(θ_t) + λ * θ_t)
   ```

其中：
- `α` 是学习率
- `μ` 是动量系数
- `λ` 是权重衰减系数

#### 使用示例 / Usage Examples

```java
// 创建在线SGD优化器 / Create online SGD optimizer
RereOnlineSGD optimizer = new RereOnlineSGD(0.01, 0.9, 0.0001);  // 学习率、动量、权重衰减

// 设置参数 / Set parameters
optimizer.setVerbose(true);                    // 详细输出 / Verbose output
optimizer.setLrDecayRate(0.1);                // 学习率衰减率 / Learning rate decay rate
optimizer.setLrDecayStep(1000);               // 学习率衰减步长 / Learning rate decay step

// 初始化优化器 / Initialize optimizer
IVector initialParams = Linalg.vector(new double[]{0.0, 0.0});
optimizer.initialize(initialParams);

// 在线学习循环 / Online learning loop
for (int i = 0; i < numIterations; i++) {
    // 计算梯度 / Compute gradient
    IVector gradient = computeGradient(optimizer.getCurrentParams());
    
    // 计算损失 / Compute loss
    double loss = computeLoss(optimizer.getCurrentParams());
    
    // 执行一步优化 / Perform one optimization step
    IVector updatedParams = optimizer.step(gradient, loss);
    
    // 检查收敛 / Check convergence
    if (loss < tolerance) {
        break;
    }
}

// 获取最终参数 / Get final parameters
IVector finalParams = optimizer.getCurrentParams();
```

### 3. 在线Adam优化器 / Online Adam Optimizer

#### 算法原理 / Algorithm Principles

在线Adam（Adaptive Moment Estimation）是一种结合了动量和自适应学习率的优化算法，具有以下特点：

- **自适应学习率** / **Adaptive Learning Rate**: 为每个参数维护独立的学习率
- **动量机制** / **Momentum Mechanism**: 使用一阶矩估计（梯度的指数移动平均）
- **二阶矩估计** / **Second Moment Estimation**: 使用二阶矩估计（梯度平方的指数移动平均）
- **偏差修正** / **Bias Correction**: 对一阶和二阶矩估计进行偏差修正

#### 数学原理 / Mathematical Principles

Adam算法的更新规则：

1. **计算梯度** / **Compute Gradient**:
   ```
   g_t = ∇f(θ_t)
   ```

2. **更新一阶矩估计** / **Update First Moment Estimate**:
   ```
   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
   ```

3. **更新二阶矩估计** / **Update Second Moment Estimate**:
   ```
   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
   ```

4. **偏差修正** / **Bias Correction**:
   ```
   m̂_t = m_t / (1 - β₁ᵗ)
   v̂_t = v_t / (1 - β₂ᵗ)
   ```

5. **参数更新** / **Parameter Update**:
   ```
   θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
   ```

其中：
- `α` 是学习率
- `β₁` 是一阶矩衰减率（通常为0.9）
- `β₂` 是二阶矩衰减率（通常为0.999）
- `ε` 是数值稳定性常数（通常为1e-8）

#### 使用示例 / Usage Examples

```java
// 创建在线Adam优化器 / Create online Adam optimizer
RereOnlineAdam optimizer = new RereOnlineAdam(0.001, 0.9, 0.999, 1e-8, 0.0001);

// 设置参数 / Set parameters
optimizer.setVerbose(true);                    // 详细输出 / Verbose output
optimizer.setAmsgrad(false);                   // 是否使用AMSGrad变体 / Whether to use AMSGrad variant
optimizer.setLrDecayRate(0.1);                // 学习率衰减率 / Learning rate decay rate
optimizer.setLrDecayStep(1000);               // 学习率衰减步长 / Learning rate decay step

// 初始化优化器 / Initialize optimizer
IVector initialParams = Linalg.vector(new double[]{0.0, 0.0});
optimizer.initialize(initialParams);

// 在线学习循环 / Online learning loop
for (int i = 0; i < numIterations; i++) {
    // 计算梯度 / Compute gradient
    IVector gradient = computeGradient(optimizer.getCurrentParams());
    
    // 计算损失 / Compute loss
    double loss = computeLoss(optimizer.getCurrentParams());
    
    // 执行一步优化 / Perform one optimization step
    IVector updatedParams = optimizer.step(gradient, loss);
    
    // 检查收敛 / Check convergence
    if (loss < tolerance) {
        break;
    }
}

// 获取最终参数 / Get final parameters
IVector finalParams = optimizer.getCurrentParams();
```

### 4. 线搜索 / Line Search

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
// 创建线搜索器 / Create line searcher
RereLineSearch lineSearcher = new RereLineSearch(1e-4, 0.9, 1.0);  // c1, c2, initialStepSize

// 定义目标函数 / Define objective function
IObjectiveFunction objFun = new IObjectiveFunction() {
    @Override
    public double computeObjective(IVector x) {
        // 示例：二次函数 / Example: quadratic function
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
        
        double[] grad = new double[2];
        grad[0] = 2 * x1;
        grad[1] = 2 * x2;
        
        return Linalg.vector(grad);
    }
};

// 当前点和搜索方向 / Current point and search direction
IVector currentPoint = Linalg.vector(new double[]{2.0, 2.0});
IVector searchDirection = Linalg.vector(new double[]{-1.0, -1.0});
IVector currentGradient = grdFun.computeGradient(currentPoint);

// 执行线搜索 / Execute line search
double optimalStep = lineSearcher.search(currentPoint, searchDirection, objFun, grdFun, currentGradient);

System.out.println("最优步长: " + optimalStep); // Optimal step size

// 计算新点 / Calculate new point
IVector newPoint = currentPoint.add(searchDirection.multiplyScalar(optimalStep));
System.out.println("新点: " + newPoint);
System.out.println("函数值变化: " + (objFun.computeObjective(newPoint) - objFun.computeObjective(currentPoint)));
```

## 高级特性 / Advanced Features

### 1. 参数配置 / Parameter Configuration

#### L-BFGS参数 / L-BFGS Parameters

```java
RereLBFGS optimizer = new RereLBFGS();

// 优化参数 / Optimization parameters
optimizer.setMaxIterations(1000);        // 最大迭代次数 / Maximum iterations
optimizer.setTolerance(1e-6);            // 收敛容差 / Convergence tolerance
optimizer.setM(10);                      // 存储的历史信息对数 / Number of stored history pairs
```

#### 在线SGD参数 / Online SGD Parameters

```java
RereOnlineSGD optimizer = new RereOnlineSGD(0.01, 0.9, 0.0001);

// 基础参数 / Basic parameters
optimizer.setLearningRate(0.01);         // 学习率 / Learning rate
optimizer.setMomentum(0.9);              // 动量系数 / Momentum coefficient
optimizer.setWeightDecay(0.0001);        // 权重衰减 / Weight decay

// 学习率调度 / Learning rate scheduling
optimizer.setLrDecayRate(0.1);           // 学习率衰减率 / Learning rate decay rate
optimizer.setLrDecayStep(1000);          // 学习率衰减步长 / Learning rate decay step

// 输出控制 / Output control
optimizer.setVerbose(true);              // 详细输出 / Verbose output
```

#### 在线Adam参数 / Online Adam Parameters

```java
RereOnlineAdam optimizer = new RereOnlineAdam(0.001, 0.9, 0.999, 1e-8, 0.0001);

// 基础参数 / Basic parameters
optimizer.setLearningRate(0.001);        // 学习率 / Learning rate
optimizer.setBeta1(0.9);                 // 一阶矩衰减率 / First moment decay rate
optimizer.setBeta2(0.999);               // 二阶矩衰减率 / Second moment decay rate
optimizer.setEpsilon(1e-8);              // 数值稳定性常数 / Numerical stability constant
optimizer.setWeightDecay(0.0001);        // 权重衰减 / Weight decay

// 高级参数 / Advanced parameters
optimizer.setAmsgrad(false);             // 是否使用AMSGrad变体 / Whether to use AMSGrad variant

// 学习率调度 / Learning rate scheduling
optimizer.setLrDecayRate(0.1);           // 学习率衰减率 / Learning rate decay rate
optimizer.setLrDecayStep(1000);          // 学习率衰减步长 / Learning rate decay step

// 输出控制 / Output control
optimizer.setVerbose(true);              // 详细输出 / Verbose output
```

#### 线搜索参数 / Line Search Parameters

```java
RereLineSearch lineSearcher = new RereLineSearch(1e-4, 0.9, 1.0);

// 线搜索参数 / Line search parameters
// c1: Armijo条件参数 / Armijo condition parameter (默认: 1e-4)
// c2: Wolfe条件参数 / Wolfe condition parameter (默认: 0.9)
// initialStepSize: 初始步长 / Initial step size (默认: 1.0)
```

### 2. 收敛监控 / Convergence Monitoring

#### L-BFGS收敛监控 / L-BFGS Convergence Monitoring

```java
// 创建L-BFGS优化器 / Create L-BFGS optimizer
RereLBFGS optimizer = new RereLBFGS();
optimizer.setMaxIterations(1000);
optimizer.setTolerance(1e-6);

// 执行优化并监控收敛 / Execute optimization and monitor convergence
Tuple2<Double, IVector> result = optimizer.optimize(initX, objFun, grdFun);

// 检查最终梯度范数 / Check final gradient norm
IVector finalGradient = grdFun.computeGradient(result._2);
double gradientNorm = (Double) finalGradient.norm2();
System.out.println("最终梯度范数: " + gradientNorm);
System.out.println("是否收敛: " + (gradientNorm < 1e-6));
```

#### 在线优化器收敛监控 / Online Optimizer Convergence Monitoring

```java
// 创建在线优化器 / Create online optimizer
RereOnlineAdam optimizer = new RereOnlineAdam(0.001);
optimizer.setVerbose(true);

// 初始化 / Initialize
optimizer.initialize(initialParams);

// 在线学习循环 / Online learning loop
double previousLoss = Double.MAX_VALUE;
for (int i = 0; i < maxIterations; i++) {
    // 计算梯度和损失 / Compute gradient and loss
    IVector gradient = computeGradient(optimizer.getCurrentParams());
    double currentLoss = computeLoss(optimizer.getCurrentParams());
    
    // 执行一步优化 / Perform one optimization step
    optimizer.step(gradient, currentLoss);
    
    // 检查收敛 / Check convergence
    if (Math.abs(currentLoss - previousLoss) < 1e-8) {
        System.out.println("损失收敛在第 " + i + " 步");
        break;
    }
    
    if (currentLoss < 1e-6) {
        System.out.println("达到目标损失在第 " + i + " 步");
        break;
    }
    
    previousLoss = currentLoss;
}
```

### 3. 自定义目标函数 / Custom Objective Functions

#### 线性回归目标函数 / Linear Regression Objective Function

```java
public class LinearRegressionObjective implements IObjectiveFunction, IGradientFunction {
    private IMatrix X;
    private IVector y;
    private double lambda1;  // L1正则化系数 / L1 regularization coefficient
    private double lambda2;  // L2正则化系数 / L2 regularization coefficient
    
    public LinearRegressionObjective(IMatrix X, IVector y, double lambda1, double lambda2) {
        this.X = X;
        this.y = y;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
    }
    
    @Override
    public double computeObjective(IVector w) {
        // 计算预测值 / Calculate predictions
        IVector yPred = X.mmul(w);
        
        // 计算残差 / Calculate residuals
        IVector residuals = yPred.subtract(y);
        
        // 计算均方误差 / Calculate mean squared error
        double mse = residuals.square().sum() / (2 * X.getRowNum());
        
        // 添加正则化项 / Add regularization terms
        double l1Reg = lambda1 * w.abs().sum();
        double l2Reg = lambda2 * w.square().sum() / 2;
        
        return mse + l1Reg + l2Reg;
    }
    
    @Override
    public IVector computeGradient(IVector w) {
        // 计算预测值 / Calculate predictions
        IVector yPred = X.mmul(w);
        
        // 计算残差 / Calculate residuals
        IVector residuals = yPred.subtract(y);
        
        // 计算梯度 / Calculate gradient
        IVector grad = X.transpose().mmul(residuals).divideByScalar(X.getRowNum());
        
        // 添加正则化梯度 / Add regularization gradients
        if (lambda1 > 0) {
            IVector l1Grad = w.sign().multiplyScalar(lambda1);
            grad = grad.add(l1Grad);
        }
        
        if (lambda2 > 0) {
            IVector l2Grad = w.multiplyScalar(lambda2);
            grad = grad.add(l2Grad);
        }
        
        return grad;
    }
}

// 使用自定义目标函数 / Use custom objective function
LinearRegressionObjective objFun = new LinearRegressionObjective(X, y, 0.01, 0.1);
RereLBFGS optimizer = new RereLBFGS();
Tuple2<Double, IVector> result = optimizer.optimize(initW, objFun, objFun);
```

## 性能特性 / Performance Features

### 收敛性能 / Convergence Performance
- **L-BFGS**: 超线性收敛，适合大规模问题 / **L-BFGS**: Superlinear convergence, suitable for large-scale problems
- **在线SGD**: 线性收敛，适合在线学习 / **Online SGD**: Linear convergence, suitable for online learning
- **在线Adam**: 自适应收敛，适合非凸优化 / **Online Adam**: Adaptive convergence, suitable for non-convex optimization
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
- 在线学习 / Online learning
- 增量学习 / Incremental learning
- 流式数据处理 / Streaming data processing

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
