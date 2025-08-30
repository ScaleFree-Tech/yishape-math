# 机器学习示例 (Machine Learning Examples)

## 概述 / Overview

本文档提供了 `com.reremouse.lab.math` 包中机器学习算法的详细使用示例。

## 线性回归示例 / Linear Regression Examples

### 基本线性回归 / Basic Linear Regression

```java
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.ml.lr.RereLinearRegression;
import com.reremouse.lab.math.ml.lr.RegressionResult;

public class BasicLinearRegressionExample {
    public static void main(String[] args) {
        // 准备训练数据 / Prepare training data
        float[][] featureData = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        float[] labelData = {14, 32, 50};
        
        IMatrix features = IMatrix.of(featureData);
        IVector labels = IVector.of(labelData);
        
        // 创建和训练模型 / Create and train model
        RereLinearRegression lr = new RereLinearRegression();
        RegressionResult result = lr.fit(features, labels);
        
        // 获取结果 / Get results
        IVector weights = result.getWeights();
        float loss = result.getLoss();
        float r2Score = result.getR2Score();
        
        System.out.println("权重: " + weights);
        System.out.println("损失: " + loss);
        System.out.println("R²分数: " + r2Score);
        
        // 预测新样本 / Predict new sample
        IVector newFeatures = IVector.of(new float[]{2, 3, 4});
        float prediction = lr.predict(newFeatures);
        System.out.println("预测值: " + prediction);
    }
}
```

### 带正则化的线性回归 / Linear Regression with Regularization

```java
public class RegularizedLinearRegressionExample {
    public static void main(String[] args) {
        // 创建带L2正则化的模型 / Create model with L2 regularization
        RereLinearRegression lr = new RereLinearRegression();
        lr.setRegularizationType(RegularizationType.L2);
        lr.setLambda2(0.1f);
        
        // 训练模型 / Train model
        RegressionResult result = lr.fit(features, labels);
        
        // 查看正则化效果 / View regularization effects
        System.out.println("L2正则化系数: " + lr.getLambda2());
        System.out.println("最终损失: " + result.getLoss());
    }
}
```

## 优化算法示例 / Optimization Algorithm Examples

### L-BFGS优化器应用 / L-BFGS Optimizer Application

```java
import com.reremouse.lab.math.optimize.RereLBFGS;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IGradientFunction;

public class LBFGSExample {
    public static void main(String[] args) {
        // 创建L-BFGS优化器 / Create L-BFGS optimizer
        RereLBFGS optimizer = new RereLBFGS();
        optimizer.setMaxIterations(1000);
        optimizer.setTolerance(1e-6f);
        
        // 定义目标函数 / Define objective function
        IObjectiveFunction objFun = new IObjectiveFunction() {
            @Override
            public float compute(IVector x) {
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
        
        // 执行优化 / Execute optimization
        IVector initX = IVector.of(new float[]{-1.0f, -1.0f});
        Tuple2<Float, IVector> result = optimizer.optimize(initX, objFun, grdFun);
        
        float optimalValue = result._1;
        IVector optimalPoint = result._2;
        
        System.out.println("最优值: " + optimalValue);
        System.out.println("最优点: " + optimalPoint);
    }
}
```

## 总结 / Summary

本文档展示了机器学习算法的基本使用方法。建议在实际使用中：

1. 根据数据特点选择合适的正则化方法
2. 使用交叉验证评估模型性能
3. 合理设置优化算法参数
4. 注意数据预处理和特征工程

---

**机器学习示例** - 从理论到实践，掌握机器学习的精髓！
