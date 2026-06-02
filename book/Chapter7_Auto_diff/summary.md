# 第 7 章 总结

> **本章回顾**：从计算图的基础概念，到反向/正向模式 AD，再到高阶导数、自定义算子、算子融合、Neural ODE、vmap、HPC/GPU 加速——自动微分是现代机器学习的基石。


## 学习成果

学完本章，你已经掌握了：

| 技能 | 对应章节 | 能力等级 |
|------|---------|---------|
| 理解自动微分原理 | 7.1 | ⭐⭐⭐ |
| 使用 `backward()` 计算梯度 | 7.2 | ⭐⭐⭐⭐ |
| 使用 `AD.grad()` 计算高阶导数 | 7.4 | ⭐⭐⭐ |
| 实现自定义算子 | 7.5 | ⭐⭐⭐ |
| 使用算子融合优化性能 | 7.6 | ⭐⭐⭐ |
| 使用梯度检查点节省显存 | 7.7 | ⭐⭐⭐ |
| 理解 GPU 加速原理 | 7.8 | ⭐⭐⭐ |


## 核心概念回顾

| 概念 | 一句话说明 | 直觉理解 |
|------|-----------|----------|
| **自动微分** | 基于链式法则的精确求导 | "分而治之"——把复杂函数拆成小块分别求导 |
| **计算图** | 函数分解为基本运算节点的有向无环图 | 函数的"X光片"——看清内部结构 |
| **反向模式** | 从输出反向传播梯度 | "多米诺骨牌"——从后往前推导 |
| **正向模式** | 从输入正向传播切向量 | "水流方向"——沿着管道流动 |
| **tape-of-tape** | 梯度本身也是可微的 | "导数的导数"——支持高阶微分 |
| **算子融合** | 合并逐元素运算链 | "流水线优化"——减少中间步骤 |
| **梯度检查点** | 用时间换空间 | "照片备份"——只存关键节点 |
| **Neural ODE** | 将网络视为连续动力系统 | "连续时间"——从离散到连续 |
| **vmap** | 将单样本函数自动向量化 | "批量处理"——一次处理多个样本 |


## API 速查表

### 变量创建

| API | 用途 | 示例 |
|-----|------|------|
| `AD.vector(double...)` | 创建可微向量 | `AD.vector(1.0, 2.0)` |
| `AD.matrix(double[][])` | 创建可微矩阵 | `AD.matrix(new double[][]{{1,2},{3,4}})` |
| `AD.zeros(n)` / `AD.ones(n)` | 快捷工厂 | `AD.zeros(5)` |
| `AD.diffFloat(float[])` | 混合精度 | `AD.diffFloat(new float[]{1f, 2f})` |

### 梯度计算

| API | 用途 | 示例 |
|-----|------|------|
| `.backward()` | 反向传播 | `loss.backward()` |
| `.backward(upstream)` | 自定义上游梯度 | `loss.backward(grad)` |
| `.getGradient()` | 获取梯度 | `x.getGradient()` |
| `.zeroGradient()` | 清除梯度 | `x.zeroGradient()` |
| `AD.grad(y, x)` | 符号梯度（高阶微分） | `AD.grad(loss, params)` |

### 高阶微分

| API | 用途 | 复杂度 |
|-----|------|--------|
| `MixedMode.hvp(fn, x, v)` | Hessian-向量积 | O(前向 + 反向) |
| `MixedMode.hessian(fn, x)` | 完整 Hessian | O(n × (前向 + 反向)) |
| `MixedMode.jvp(fn, x, v)` | Jacobian-向量积 | O(前向) |
| `MixedMode.vjp(fn, x, g)` | Vector-Jacobian 积 | O(反向) |

### 算子融合

| API | 用途 | 示例 |
|-----|------|------|
| `AD.fuse(x).op().compute()` | 手动融合 | `AD.fuse(x).add(1).sigmoid().compute()` |
| `AD.elementwise(x, fn)` | 自动融合 | `AD.elementwise(x, v -> v.add(1).sigmoid())` |
| `AD.optimize(x)` | 常量折叠 | `AD.optimize(y)` |

### 高级功能

| API | 用途 | 示例 |
|-----|------|------|
| `AD.checkpoint(fn, x)` | 梯度检查点 | `AD.checkpoint(net, x)` |
| `AD.odeint(dynamics, z0, t0, t1, dt)` | Neural ODE | `AD.odeint(f, z0, 0, 10, 0.1)` |
| `AD.vmap(fn, xs)` | 自动批处理 | `AD.vmap(lossFn, batch)` |
| `AD.checkGradient(fn, x, tol)` | 梯度校验 | `AD.checkGradient(fn, x, 1e-5)` |


## 章节练习答案提示

### 7.1 基础概念
1. 计算图：`x → [+] → a, y → [×] → a, a → [²] → f`；前向：$f=(x+y)^2$；反向：$\frac{\partial f}{\partial x}=2(x+y)$, $\frac{\partial f}{\partial y}=2(x+y)$
2. 中心差分：$f'(\pi/4) = (e^{\sin(\pi/4+h)} - e^{\sin(\pi/4-h)})/(2h)$；自动微分：$f' = \cos(x) \cdot e^{\sin(x)}$；差异在 $10^{-6}$ 量级
3. 前向模式更高效：3 次正向 vs 1 次反向，但前向模式每次只算一个输入方向

### 7.2 反向模式
1. $y = \sin(x) \cdot x^2$，$\frac{\partial y}{\partial x} = \cos(x) \cdot x^2 + \sin(x) \cdot 2x$
2. $\frac{\partial L}{\partial W} = 2(XW - Y)^T X$
3. 两条路径的梯度累加：$4x = 8$

### 7.3 正向模式
1. JVP = $\cos(1)\cos(1) + 0 = \cos^2(1) \approx 0.2919$
2. HVP = $2I \cdot [1,0,0]^T = [2,0,0]^T$
3. HVP 只需 $O(1)$ 次反向传播，而完整 Hessian 需要 $O(n)$ 次

### 7.4 高阶导数
1. $f'''(\pi/4) = -\cos(\pi/4) = -\sqrt{2}/2 \approx -0.7071$
2. $H = 2A$，$Hv = 2Av$
3. $n$ 次 HVP：每次 HVP 计算 $H$ 的一列，$n$ 列拼成完整矩阵

### 7.5 自定义算子
1. Swish：$f(x) = x \cdot \sigma(x)$，$f'(x) = \sigma(x) + x \cdot \sigma(x)(1-\sigma(x))$
2. Huber Loss：在 $|x| < \delta$ 区域是二次函数，之外是线性
3. `forward()` 接收原始数组是因为它不参与计算图——只有 `apply()` 返回的节点才在图中

### 7.6 算子融合
1. 融合前 5 个节点，融合后 2 个节点
2. 计算图：`x → [sin] → a, x → [²] → b → [cos] → c, a + c → y`
3. `sum()` 改变了向量的形状（从 n 维到 1 维），无法与逐元素运算融合

### 7.7 高级功能
1. 检查点：每 √n 层设一个，显存从 O(n) 降为 O(√n)
2. `vmapSum` 与手动循环结果一致
3. $z(T) = z(0) \cdot e^{-T} = 1 \cdot e^{-5} \approx 0.0067$

### 7.8 HPC/GPU
1. JSON 包含节点 id、shape、op、data、inputs
2. GPU 不可用时返回 `false`，自动回退 CPU
3. `squareSum` 等需要原子操作，GPU 线程竞争导致性能下降


## 学习路线图

### 初学者路线（1-2周）

```
Week 1:
├── Day 1-2: 7.1 基础概念（理解计算图、链式法则）
├── Day 3-5: 7.2 反向模式AD（掌握backward()、梯度累积）
└── Day 6-7: 练习 + 复习

Week 2:
├── Day 1-3: 7.4 高阶导数（理解tape-of-tape）
├── Day 4-5: 7.5 自定义算子（实现自定义Loss）
└── Day 6-7: 项目实践
```

### 进阶路线（2-3周）

```
Week 3:
├── Day 1-2: 7.3 正向模式AD（理解JVP、HVP）
├── Day 3-4: 7.6 算子融合（优化性能）
└── Day 5-7: 7.7 高级功能（Neural ODE、vmap）

Week 4:
├── Day 1-2: 7.8 HPC与GPU加速
├── Day 3-5: 完整项目实践
└── Day 6-7: 总结 + 进阶学习
```

### 专家路线（持续学习）

```
深入研究:
├── 自动微分的数学理论
├── 高性能计算（SIMD、GPU）
├── 特定领域应用（物理模拟、生成模型）
└── 贡献开源项目
```


## 推荐资源

### 论文
1. **"Automatic Differentiation in Machine Learning: a Survey"** - Baydin et al., 2018
2. **"Neural Ordinary Differential Equations"** - Chen et al., 2018
3. **"Operator Fusing for Automatic Differentiation"** - Various

### 博客
1. **"A Primer on Automatic Differentiation"** - 基础概念入门
2. **"Understanding Backpropagation"** - 反向传播详解
3. **"Gradient Checkpointing Explained"** - 梯度检查点原理

### 视频
1. **Stanford CS231n** - 深度学习中的自动微分
2. **3Blue1Brown** - 神经网络可视化
3. **Yannic Kilcher** - 论文解读

### 开源项目
1. **PyTorch autograd** - 参考实现
2. **JAX** - 函数式AD
3. **Functorch** - PyTorch的函数式变换


## 下一步学习

- **第 5 章机器学习**：用 AD 训练神经网络（回归、分类）
- **第 6 章最优化**：用 `Opts` 优化器 + AD 梯度解决实际问题
- **第 8 章时间序列**：用 TSA 预测时间序列（可结合 AD 做自定义损失）

### 实战项目建议

1. **手写数字识别**：用AD实现简单的MLP，识别MNIST
2. **线性回归**：用AD实现带正则化的线性回归
3. **物理模拟**：用Neural ODE模拟弹簧振动
4. **生成模型**：用AD实现简单的VAE或Flow模型


---

**恭喜你完成本章学习！** 现在你已经掌握了自动微分的核心知识。接下来，把这些知识应用到实际项目中吧！

[← HPC 与 GPU 加速](7.8.%20HPC%20and%20GPU.md) ｜ [返回目录](introduction.md) ｜ [FAQ →](FAQ.md)
