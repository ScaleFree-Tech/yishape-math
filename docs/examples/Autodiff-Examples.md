# 自动微分示例 (Autodiff Examples)

## 概述 / Overview

本文档提供 `AD` 从基础到高级的完整示例，涵盖反向/前向模式 AD、混合模式（HVP/Hessian）、VJP/vmap、ML 训练、算子融合、Neural ODE、混合精度、稀疏/复数/Tensor autodiff 等。

## 基础示例 / Basic Examples

### 反向模式梯度计算 / Reverse-mode Gradient Computation

```java
import com.yishape.lab.math.autodiff.AD;

public class BasicAutodiffExample {
    public static void main(String[] args) {
        // 创建可微变量
        var x = AD.vector(new double[]{3.0, 4.0});

        // 定义计算：f(x) = x₀² + x₁²
        var loss = x.pow(2).sum();

        // 反向传播梯度
        loss.backward();

        // 获取梯度：∂f/∂x = 2x = [6.0, 8.0]
        var grad = x.getGradient();
        System.out.println("梯度 / Gradient: " + grad);
    }
}
```

### 简单梯度下降 / Simple Gradient Descent

```java
import com.yishape.lab.math.autodiff.AD;

public class GradientDescentAD {
    public static void main(String[] args) {
        // 最小化 f(x) = x² - 2，理论最优 x=0, f=-2
        var x = AD.vector(new double[]{3.0});
        for (int i = 0; i < 50; i++) {
            x.zeroGradient();
            var loss = x.pow(2).sub(2);       // f(x) = x² - 2
            loss.backward();                   // ∂f/∂x = 2x → [6.0] → [0.0]
            double step = 0.1 * x.getGradient().get(0);
            x = AD.vector(x.getValue().get(0) - step);
        }
        System.out.println("最优 x: " + x.getValue().get(0));    // → ~0
    }
}
```

### 使用 reuseNode 避免重建图 / reuseNode for In-Place Updates

```java
import com.yishape.lab.math.autodiff.AD;

public class ReuseNodeExample {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{3.0, 4.0});

        for (int i = 0; i < 100; i++) {
            x.zeroGradient();
            var loss = x.pow(2).sum();
            loss.backward();

            double[] grad = x.getGradient().toDoubleArray();
            double[] newData = new double[x.getValue().size()];
            for (int j = 0; j < newData.length; j++) {
                newData[j] = x.getValue().get(j) - 0.01 * grad[j];
            }

            // 原地更新叶子数据，避免重建计算图节点
            x = AD.reuseNode(x, newData);
        }
        System.out.println("收敛值: " + x.getValue());
    }
}
```

## 高级激活函数 / Advanced Activation Functions

```java
import com.yishape.lab.math.autodiff.AD;

public class AdvancedActivationsAD {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{-2.0, -0.5, 0.0, 0.5, 2.0});

        // GELU: x * Φ(x)（高斯误差线性单元）
        var geluOut = x.gelu();
        geluOut.sum().backward();
        System.out.println("GELU 梯度: " + x.getGradient());

        x.zeroGradient();
        // SiLU / Swish: x * σ(x)
        var siluOut = x.silu();
        siluOut.sum().backward();
        System.out.println("SiLU 梯度: " + x.getGradient());

        x.zeroGradient();
        // Mish: x * tanh(softplus(x))
        var mishOut = x.mish();
        mishOut.sum().backward();
        System.out.println("Mish 梯度: " + x.getGradient());

        x.zeroGradient();
        // Leaky ReLU: max(αx, x)
        var lreluOut = x.leakyRelu(0.01);
        lreluOut.sum().backward();
        System.out.println("LeakyReLU 梯度: " + x.getGradient());

        x.zeroGradient();
        // ELU: x if x>0 else α(exp(x)-1)
        var eluOut = x.elu(1.0);
        eluOut.sum().backward();
        System.out.println("ELU 梯度: " + x.getGradient());

        x.zeroGradient();
        // Softplus: log(1+exp(βx))/β
        var spOut = x.softplus(1.0);
        spOut.sum().backward();
        System.out.println("Softplus 梯度: " + x.getGradient());

        x.zeroGradient();
        // SELU: scaled ELU
        var seluOut = x.selu();
        seluOut.sum().backward();
        System.out.println("SELU 梯度: " + x.getGradient());

        x.zeroGradient();
        // Clamp
        var clampOut = x.clamp(-0.5, 0.5);
        clampOut.sum().backward();
        System.out.println("Clamp 梯度: " + x.getGradient());
    }
}
```

### 融合层归一化 / Fused Layer Normalization

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.linalg.Linalg;

public class LayerNormAD {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{1.0, 2.0, 3.0, 4.0});
        var gamma = AD.vector(new double[]{1.0, 1.0, 1.0, 1.0});
        var beta  = AD.vector(new double[]{0.0, 0.0, 0.0, 0.0});

        // Fused LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
        var normed = x.layerNorm(gamma, beta, 1e-5);
        var loss = normed.square().sum();
        loss.backward();

        System.out.println("x 梯度: " + x.getGradient());
        System.out.println("gamma 梯度: " + gamma.getGradient());
        System.out.println("beta 梯度: " + beta.getGradient());
    }
}
```

## ML 训练：一行代码替代手写梯度 / ML Training: One-Liner

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.Opts;

public class MLAutodiffExample {
    public static void main(String[] args) {
        // 生成数据
        var X = Linalg.randn(100, 5);
        var y = Linalg.randn(100);
        // 加 bias 列
        var Xc = X.addColumn(Linalg.ones(100));

        // 包装为 autodiff 类型
        var Xm = AD.matrix((IDoubleMatrix) Xc);
        var yConst = AD.constant((IDoubleVector) y);

        var w0 = Linalg.zeros(6);
        // 只需定义 loss，无需手写梯度！
        var result = AD.optimize(w0,
            w -> Xm.matmul(w).sub(yConst).square().mean(),  // MSE
            Opts.lbfgs()
        );

        System.out.println("最优参数: " + result.getOptimalPoint());
        System.out.println("最优损失: " + result.getOptimalValue());
    }
}
```

### 带正则化的逻辑回归 / Logistic Regression with Regularization

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.Opts;

public class LogisticRegressionAD {
    public static IDiffVector sigmoid(IDiffVector z) {
        return z.mul(-1).exp().add(1).rdiv(1);  // 1 / (1 + exp(-z))
    }

    public static void main(String[] args) {
        var X = Linalg.randn(200, 10);
        double[] yRaw = new double[200];
        for (int i = 0; i < 200; i++) yRaw[i] = Math.random() < 0.5 ? 0.0 : 1.0;
        var y = Linalg.vector(yRaw);

        var Xc = X.addColumn(Linalg.ones(200));
        var Xm = AD.matrix((IDoubleMatrix) Xc);
        var yConst = AD.constant((IDoubleVector) y);

        var w0 = Linalg.zeros(11);
        double lambda = 0.01;

        var result = AD.optimize(w0, w -> {
            var logits = Xm.matmul(w);
            var probs = sigmoid(logits);
            // Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))
            var bce = yConst.mul(probs.log())
                .add(yConst.rsub(1).mul(probs.rsub(1).log()))
                .mean().mul(-1);
            return bce.add(w.pow(2).sum().mul(lambda));  // + L2 penalty
        }, Opts.lbfgs());

        System.out.println("训练完成");
        System.out.println("最优参数: " + result.getOptimalPoint());
    }
}
```

### Softmax 交叉熵分类 / Softmax CrossEntropy Classification

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.Opts;

public class SoftmaxCEExample {
    public static void main(String[] args) {
        // 数据：100 样本 × 5 特征 → 3 类
        var X = Linalg.randn(100, 5);
        var Xc = ((IDoubleMatrix) X).addColumn(Linalg.ones(100));
        var Xm = AD.matrix(Xc);

        // One-hot 标签
        double[][] Ydata = new double[100][3];
        for (int i = 0; i < 100; i++) Ydata[i][(int)(Math.random() * 3)] = 1.0;
        var Y = AD.constant(Linalg.matrix(Ydata));

        var W0 = Linalg.zeros(6, 3);  // 6 特征 → 3 类
        var result = AD.optimize(Linalg.flatten(W0), w -> {
            var wMat = w.reshape(6, 3);
            var logits = Xm.matmul(wMat);              // [100, 3]
            // 融合 Softmax + CrossEntropy（数值稳定）
            return logits.softmaxCrossEntropy(Y);
        }, Opts.lbfgs());

        System.out.println("最优损失: " + result.getOptimalValue());
    }
}
```

## 前向模式 AD / Forward-mode AD

```java
import com.yishape.lab.math.autodiff.AD;

public class ForwardModeADExample {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{1.0, 2.0, 3.0});

        // 完整 Jacobian：J ∈ R³ˣ³ where Jᵢⱼ = ∂fᵢ/∂xⱼ
        var J = AD.jacobian(
            z -> z.mul(z).exp(),  // f(z) = exp(z⊙z)
            x
        );
        System.out.println("Jacobian:\n" + J);

        // 梯度检查
        boolean ok = AD.checkGradient(
            z -> z.pow(3).sum(), x, 1e-4
        );
        System.out.println("梯度正确: " + ok);

        // 详细梯度检查
        var detail = AD.checkGradientDetailed(
            z -> z.pow(3).sum(), x, 1e-4
        );
        System.out.println(detail.detailedReport());
    }
}
```

## 混合模式 AD：HVP 与 Hessian / Mixed-Mode AD: HVP & Hessian

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.MixedMode;
import com.yishape.lab.math.linalg.Linalg;

public class MixedModeExample {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{1.0, 2.0, 3.0});

        // Hessian-vector product: H·v（tape-of-tape 实现）
        var v = Linalg.vector(new double[]{1.0, 0.0, 0.0});
        double[] hvp = MixedMode.hvp(
            z -> z.pow(3).sum(),  // f(z) = z₀³ + z₁³ + z₂³
            x, v
        );
        // H = diag(6x₀, 6x₁, 6x₂), H·v = [6x₀, 0, 0]
        System.out.println("H·v: [" + hvp[0] + ", " + hvp[1] + ", " + hvp[2] + "]");

        // Jacobian-vector product: J·v（前向模式 AD）
        double[] jvp = MixedMode.jvp(
            z -> z.pow(2),  // f(z) = z²
            x, v
        );
        // J = diag(2x₀, 2x₁, 2x₂), J·v = [2x₀, 0, 0]
        System.out.println("J·v: [" + jvp[0] + ", " + jvp[1] + ", " + jvp[2] + "]");

        // Vector-Jacobian product: J^T·g
        var g = Linalg.vector(new double[]{1.0, 1.0, 1.0});
        double[] vjp = MixedMode.vjp(z -> z.pow(2), x, g);
        System.out.println("J^T·g: [" + vjp[0] + ", " + vjp[1] + ", " + vjp[2] + "]");

        // 完整 Hessian（仅小维度，n<100）
        var H = MixedMode.hessian(z -> z.pow(3).sum(), x);
        System.out.println("Hessian:\n" + H);

        // 完整 Jacobian
        var Jfull = MixedMode.jacobianFull(z -> z.pow(2), x);
        System.out.println("Full Jacobian:\n" + Jfull);
    }
}
```

## 自定义操作 / Custom Operations

### CustomOp 示例

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.CustomOp;
import com.yishape.lab.math.linalg.IDoubleVector;

public class CustomOpExample {
    public static void main(String[] args) {
        // 定义自定义操作：f(x) = clip(x, -1, 1) 的平滑近似
        var smoothClip = new CustomOp() {
            @Override
            protected ForwardResult forward(IDoubleVector[] rawInputs) {
                var x = rawInputs[0];
                double[] data = x.toDoubleArray();
                double[] out = new double[data.length];
                for (int i = 0; i < data.length; i++) {
                    out[i] = Math.tanh(data[i]);  // 平滑 clip
                }
                // context 存储输出值供 backward 使用
                return new ForwardResult(
                    com.yishape.lab.math.linalg.Linalg.vector(out),
                    out  // context = 前向输出
                );
            }

            @Override
            protected IDoubleVector[] backward(IDoubleVector gradOutput, Object ctx) {
                double[] fwdOut = (double[]) ctx;
                double[] gradIn = gradOutput.toDoubleArray();
                double[] gradX = new double[fwdOut.length];
                for (int i = 0; i < fwdOut.length; i++) {
                    // d/dx tanh(x) = 1 - tanh²(x) = 1 - f²
                    gradX[i] = gradIn[i] * (1.0 - fwdOut[i] * fwdOut[i]);
                }
                return new IDoubleVector[]{
                    com.yishape.lab.math.linalg.Linalg.vector(gradX)
                };
            }
        };

        var x = AD.vector(new double[]{-2.0, -0.5, 0.0, 0.5, 2.0});
        var y = AD.op(smoothClip, x);
        var loss = y.square().sum();
        loss.backward();

        System.out.println("x 梯度: " + x.getGradient());
    }
}
```

## VJP 与 vmap / VJP & vmap

### VJP：可复用梯度算子

```java
import com.yishape.lab.math.autodiff.AD;

public class VjpExample {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{1.0, 2.0, 3.0});

        // 计算 VJP 变换，返回可复用算子
        var result = AD.vjp(z -> z.pow(2).exp(), x);

        System.out.println("前向输出 y: " + result.y().getValue());

        // 用不同上游梯度反复调用 vjpFn，无需重建图
        var grad1 = result.vjpFn().apply(AD.vector(new double[]{1.0, 1.0, 1.0}));
        System.out.println("J^T·[1,1,1]: " + grad1.getValue());

        var grad2 = result.vjpFn().apply(AD.vector(new double[]{1.0, 0.0, 0.0}));
        System.out.println("J^T·[1,0,0]: " + grad2.getValue());
    }
}
```

### 批量 VJP

```java
import com.yishape.lab.math.autodiff.AD;
import java.util.List;

public class BatchVjpExample {
    public static void main(String[] args) {
        var x1 = AD.vector(new double[]{1.0, 2.0});
        var x2 = AD.vector(new double[]{3.0, 4.0});
        var x3 = AD.vector(new double[]{5.0, 6.0});

        var batchResult = AD.batchVjp(
            z -> z.pow(2),
            List.of(x1, x2, x3)
        );

        System.out.println("批量大小: " + batchResult.batchSize());

        // 所有样本用同一上游梯度
        var upstream = AD.vector(new double[]{1.0, 1.0});
        var grads = batchResult.applyAll(upstream);
        for (int i = 0; i < grads.length; i++) {
            System.out.println("样本" + i + " 梯度: " + grads[i].getValue());
        }

        // 梯度累加
        var sumGrad = batchResult.sumGradients(upstream);
        System.out.println("梯度累加: " + sumGrad.getValue());

        // 梯度平均
        var meanGrad = batchResult.meanGradients(upstream);
        System.out.println("梯度平均: " + meanGrad.getValue());
    }
}
```

### vmap：自动批处理

```java
import com.yishape.lab.math.autodiff.AD;
import java.util.List;

public class VmapExample {
    public static void main(String[] args) {
        var xs = List.of(
            AD.vector(new double[]{1.0, 2.0, 3.0}),
            AD.vector(new double[]{4.0, 5.0, 6.0}),
            AD.vector(new double[]{7.0, 8.0, 9.0})
        );

        // 对每个样本独立执行 fn（独立计算图）
        var results = AD.vmap(
            z -> z.pow(2).exp().sum(),
            xs
        );
        for (int i = 0; i < results.length; i++) {
            System.out.println("样本 " + i + ": " + results[i].getValue());
        }

        // vmap + 求和：自动累加损失
        var totalLoss = AD.vmapSum(
            z -> z.pow(2).exp().sum(),
            xs
        );
        System.out.println("总损失: " + totalLoss.getValue());
    }
}
```

## 算子融合 / Operator Fusion

### 显式融合链

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.linalg.Linalg;

public class FusedOpsExample {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{1.0, 2.0, 3.0});
        var y = AD.vector(new double[]{0.5, 1.0, 1.5});

        // 显式融合链：exp(x) * y + 2 → tanh → 单次前向 + 单次反向
        var fused = AD.fuse(x)
            .exp()              // exp(x)
            .mul(y)             // * y
            .add(2.0)           // + 2
            .tanh()             // tanh
            .compute();         // 物化为单个图节点

        var loss = fused.sum();
        loss.backward();
        System.out.println("x 梯度: " + x.getGradient());
        System.out.println("y 梯度: " + y.getGradient());
    }
}
```

### 自动融合

```java
import com.yishape.lab.math.autodiff.AD;

public class AutoFusionExample {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{1.0, 2.0, 3.0});

        // AD.elementwise 尝试追踪并融合，不可融合时回退 eager
        var y = AD.elementwise(x, z ->
            z.exp().mul(2).add(1).tanh()
        );

        var loss = y.sum();
        loss.backward();
        System.out.println("x 梯度: " + x.getGradient());
    }
}
```

## Neural ODE 示例 / Neural ODE Example

```java
import com.yishape.lab.math.autodiff.AD;

public class NeuralODEExample {
    public static void main(String[] args) {
        // 定义动力学 dz/dt = tanh(z)
        var z0 = AD.vector(new double[]{1.0, 0.5});

        // 通过 ODE 积分 + adjoint 反向传播
        var z1 = AD.odeint(
            z -> z.tanh(),   // dz/dt = tanh(z)
            z0, 0.0, 2.0, 0.1
        );

        var loss = z1.square().sum();
        loss.backward();  // 梯度自动穿过 ODE 求解器（伴随法）
        var grad = z0.getGradient();
        System.out.println("dL/dz₀: " + grad);
    }
}
```

## 高阶微分 / Higher-Order Differentiation

```java
import com.yishape.lab.math.autodiff.AD;

public class HigherOrderAD {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{2.0, 3.0});

        // 一阶梯度（符号形式，可用微节点）
        var y = x.pow(3).sum();               // f(x) = x₀³ + x₁³
        var grads = AD.grad(y, x);            // [3x₀², 3x₁²] 作为可微节点

        // 二阶梯度：对一阶梯度再求导
        var grad0 = grads[0];                 // 3x₀²
        var secondGrads = AD.grad(grad0, x);  // [6x₀, 0]
        System.out.println("∂²f/∂x₀²: " + secondGrads[0].getValue());
        System.out.println("∂²f/∂x₀∂x₁: " + secondGrads[1].getValue());
    }
}
```

## 矩阵 Autodiff 进阶 / Matrix AD Advanced

### 矩阵乘法与轴归约

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.linalg.Linalg;

public class MatrixADAdvanced {
    public static void main(String[] args) {
        // 矩阵乘法 + bias 广播
        var W = AD.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        var x = AD.vector(new double[]{0.5, -0.5});
        var b = AD.vector(new double[]{0.1, 0.2, 0.3});

        var z = W.matmul(x).add(b);  // Wx + b
        var loss = z.square().sum();
        loss.backward();

        System.out.println("W 梯度:\n" + W.getGradient());
        System.out.println("x 梯度: " + x.getGradient());
        System.out.println("b 梯度: " + b.getGradient());
    }

    public static void axisReductions() {
        var M = AD.matrix(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});

        // 沿轴归约
        var colSums = M.sum(0);  // [5, 7, 9] — 列方向求和
        var rowSums = M.sum(1);  // [6, 15]  — 行方向求和

        colSums.square().sum().backward();
        System.out.println("列求和梯度:\n" + M.getGradient());
    }

    public static void broadcastArithmetic() {
        var M = AD.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        var bias = AD.vector(new double[]{0.1, 0.2});

        // 沿 axis=0 广播减法（每列减对应 bias）
        var centered = M.sub(bias, 0);
        var loss = centered.square().sum();
        loss.backward();

        System.out.println("M 梯度:\n" + M.getGradient());
        System.out.println("bias 梯度: " + bias.getGradient());
    }
}
```

## 稀疏 Autodiff 示例 / Sparse Autodiff Example

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.linalg.Linalg;

public class SparseAutodiffExample {
    public static void main(String[] args) {
        // 创建稀疏矩阵
        int[] rows = {0, 1, 2};
        int[] cols = {0, 1, 2};
        double[] vals = {1.0, 2.0, 3.0};
        var sparse = Linalg.sparseFromCOO(rows, cols, vals, 3, 3);

        var A = AD.sparse(sparse);
        var x = AD.vector(new double[]{1.0, 2.0, 3.0});

        var y = A.matmul(x);  // 稀疏 @ 稠密
        var loss = y.square().sum();
        loss.backward();

        System.out.println("稀疏梯度: " + A.getGradient());
    }
}
```

## 复数 Autodiff 示例 / Complex Autodiff Example

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.linalg.Linalg;

public class ComplexAutodiffExample {
    public static void main(String[] args) {
        // 创建复数变量
        var real = new double[]{1.0, 2.0};
        var imag = new double[]{0.5, -0.3};
        var complexVec = Linalg.complexVector(real, imag);

        var z = AD.complex(complexVec);

        // Wirtinger 导数自动处理
        var w = z.exp().mul(z.conjugate());
        var loss = w.sum();
        loss.backward();

        System.out.println("复数梯度: " + z.getGradient());
    }
}
```

## 混合精度 FP32 示例 / Mixed Precision Example

```java
import com.yishape.lab.math.autodiff.AD;

public class MixedPrecisionExample {
    public static void main(String[] args) {
        // FP32 前向 + FP64 梯度累积
        var x = AD.diffFloat(new float[]{1.5f, 2.5f, 3.5f});

        var loss = x.square().sum();
        loss.backward();

        // 梯度以 FP64 精度累积
        System.out.println("FP32 输入 / FP64 梯度: " + x.getGradient());
    }
}
```

## Tensor Autodiff 示例 / Tensor Autodiff Example

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.linalg.Linalg;

public class TensorAutodiffExample {
    public static void main(String[] args) {
        // 创建 2×3 可微张量
        var t = IDiffTensor.fromTensor(
            Linalg.tensor(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}),
            true  // requiresGrad
        );

        // Softmax 沿最后一维
        var probs = t.softmax(1);
        // 对数似然损失
        var loss = probs.log().sumAll();
        loss.backward();

        System.out.println("梯度:\n" + t.grad());

        // 矩阵乘法（可微）
        t.zeroGradient();
        var W = IDiffTensor.fromTensor(
            Linalg.tensor(new double[][]{{0.1, -0.1}, {0.2, -0.2}, {0.3, -0.3}}),
            true
        );
        var out = t.mmul(W);  // [2,3] @ [3,2] = [2,2]
        var loss2 = out.square().sumAll();
        loss2.backward();

        System.out.println("t 梯度 (mmul):\n" + t.grad());
        System.out.println("W 梯度 (mmul):\n" + W.grad());

        // 批量矩阵乘法
        var batch = IDiffTensor.fromTensor(
            Linalg.tensor(new double[][][]{
                {{1,2},{3,4},{5,6}},
                {{7,8},{9,10},{11,12}}
            }),
            true
        );
        var batchOut = batch.bmm(
            Linalg.tensor(new double[][][]{
                {{0.1,-0.1},{0.2,-0.2}},
                {{0.3,-0.3},{0.4,-0.4}}
            }).detach()
        );
        System.out.println("Batch MM 输出形状: " + java.util.Arrays.toString(batchOut.shape()));
    }
}
```

## 图工具示例 / Graph Tools Example

```java
import com.yishape.lab.math.autodiff.AD;

public class GraphToolsExample {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{1.0, 2.0});
        var y = x.pow(2).add(x.mul(3)).exp();  // exp(x² + 3x)

        // DOT 格式可视化
        String dot = AD.render(y);
        System.out.println("Graphviz DOT:\n" + dot);

        // JSON 导出（HPC 桥接）
        String json = AD.exportGraph(y);

        // 图优化（常量折叠）
        var optimized = AD.optimize(y);
        var stats = AD.graphStats(optimized);
        System.out.println("总节点: " + stats.totalNodes());
        System.out.println("叶子节点: " + stats.leafNodes());
        System.out.println("可融合链: " + stats.fusibleChains());

        // 尝试 HPC 执行
        boolean hpcOk = AD.tryHpcExecute(y);
        System.out.println("HPC 执行: " + (hpcOk ? "成功" : "回退 Java"));
    }
}
```

## 检查点（内存优化）/ Checkpointing

```java
import com.yishape.lab.math.autodiff.AD;

public class CheckpointExample {
    public static void main(String[] args) {
        var x = AD.vector(new double[]{1.0, 2.0, 3.0});

        // 深层网络用检查点减少内存：前向不存中间激活，反向时重计算
        var y = AD.checkpoint(
            z -> z.pow(2).exp().tanh().square(),
            x
        );

        var loss = y.sum();
        loss.backward();
        System.out.println("梯度: " + x.getGradient());
    }
}
```

## 在线学习 + Autodiff / Online Learning with Autodiff

```java
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.Opts;

public class OnlineLearningADExample {
    public static void main(String[] args) {
        // 包装在线优化器，自动求梯度
        var baseOpt = Opts.onlineAdam(0.001);
        var autoOpt = AD.autogradOptimizer(baseOpt,
            (w, target) -> w.sub(target).square().sum()
        );

        // 初始化
        autoOpt.initialize(Linalg.vector(new double[]{0.0, 0.0}));

        // 每个 sample 自动计算梯度
        var target1 = AD.constant(Linalg.vector(new double[]{1.0, 2.0}));
        var target2 = AD.constant(Linalg.vector(new double[]{3.0, 4.0}));
        autoOpt.step(target1);
        autoOpt.step(target2);

        System.out.println("学到的参数: " + autoOpt.getCurrentParams());
    }

    public static void onlineLearnExample() {
        // 一站式训练循环
        var initParams = Linalg.vector(new double[]{0.0, 0.0});
        var data = java.util.List.of(
            Linalg.vector(new double[]{1.0, 2.0}),
            Linalg.vector(new double[]{3.0, 4.0}),
            Linalg.vector(new double[]{5.0, 6.0})
        );

        var result = AD.onlineLearn(initParams, data,
            (w, sample) -> w.sub(AD.constant(sample)).square().sum(),
            Opts.onlineSGD(0.01),
            10  // epochs
        );

        System.out.println("训练结果: " + result);
    }
}
```

## 总结 / Summary

| 模式 | 入口 | 适用场景 |
|------|------|---------|
| **反向模式** | `AD.vector()` → `loss.backward()` → `getGradient()` | 标量损失 → 多参数梯度 |
| **ML 训练** | `AD.optimize(w0, loss, optimizer)` | 一行代码替代手写梯度 |
| **前向模式** | `AD.tangent()` / `AD.jacobian()` | 少输入 → 多输出 Jacobian |
| **混合模式** | `MixedMode.hvp()` / `MixedMode.hessian()` | Hessian-vector product、二阶优化 |
| **VJP** | `AD.vjp()` → `vjpFn.apply(g)` | 可复用向量-Jacobian积 |
| **vmap** | `AD.vmap()` / `AD.vmapSum()` | 自动批处理 |
| **融合** | `AD.fuse()` / `AD.elementwise()` | 逐元素链 JIT 编译 |
| **检查点** | `AD.checkpoint()` | 深层网络节省内存 |
| **Neural ODE** | `AD.odeint()` + `loss.backward()` | 微分方程可微积分 |
| **稀疏/复数** | `AD.sparse()` / `AD.complex()` | 稀疏矩阵 / Wirtinger 微积分 |
| **混合精度** | `AD.diffFloat()` | FP32 前向 + FP64 梯度 |
| **Tensor** | `IDiffTensor` | 多维张量自动微分 |
| **自定义操作** | `CustomOp` / `AD.op()` | 自定义前向/反向核 |
| **图工具** | `AD.render()` / `AD.exportGraph()` / `AD.optimize()` | 可视化、导出、优化 |
| **在线学习** | `AD.autogradOptimizer()` / `AD.onlineLearn()` | 在线优化 + 自动梯度 |

---

**自动微分示例** — 从反向传播到 Neural ODE，全面掌握自动微分！