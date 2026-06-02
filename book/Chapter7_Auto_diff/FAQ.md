# 常见问题解答（FAQ）

## 入门问题

### Q1：我需要什么数学基础才能学自动微分？

**A**：只需要基础的微积分（求导、链式法则）和线性代数（向量、矩阵）。如果你能理解 $f'(x) = 2x$，就能学自动微分。

### Q2：YiShape-Math 的 AD 和 PyTorch 的 autograd 有什么区别？

**A**：核心原理相同，但 YiShape-Math 更简洁：
- PyTorch：`x = torch.tensor([1.0, 2.0], requires_grad=True)`
- YiShape-Math：`IDiffVector x = AD.vector(1.0, 2.0)`

YiShape-Math 不需要 `requires_grad` 参数——所有 `AD.vector()` 创建的变量都是可微的。

### Q3：为什么不用 Python 的 JAX？

**A**：JAX 很强大，但：
- 学习曲线陡峭（pytree、vmap、grad 的组合）
- 与 Java 项目集成困难
- 调试困难（函数式编程风格）

YiShape-Math 的 AD 更适合 Java 生态，API 更直观。

## 使用问题

### Q4：为什么 `getGradient()` 返回 `null`？

**A**：因为你没有调用 `backward()`。梯度只有在反向传播后才会计算。

```java
IDiffVector x = AD.vector(1.0, 2.0);
IDiffVector y = x.pow(2).sum();
// y.backward();  ← 忘记调用！
System.out.println(x.getGradient());  // null

// 正确做法
y.backward();
System.out.println(x.getGradient());  // [2.0, 4.0]
```

### Q5：为什么我的梯度是累加的？

**A**：这是设计特性，不是bug。如果你的变量通过多条路径影响输出，梯度会自动累加。

```java
IDiffVector x = AD.vector(2.0);
IDiffVector y = x.mul(x).add(x.mul(x));  // y = x² + x²
y.backward();
System.out.println(x.getGradient());  // [8.0] (2x + 2x = 4x = 8)
```

解决方案：每次迭代前调用 `x.zeroGradient()`。

### Q6：如何验证我的梯度计算是否正确？

**A**：使用 `AD.checkGradient()` 进行数值校验：

```java
Function<IDiffVector, IDiffVector> lossFn = x -> x.pow(2).sum();
IDiffVector x = AD.vector(1.0, 2.0, 3.0);
boolean ok = AD.checkGradient(lossFn, x, 1e-5);
System.out.println(ok ? "梯度正确！" : "梯度错误！");
```

### Q7：如何可视化计算图？

**A**：使用 `AD.render()` 生成 Graphviz DOT 格式：

```java
IDiffVector x = AD.vector(1.0, 2.0);
IDiffVector y = x.mul(x).add(x.sum());
String dot = AD.render(y);
// 保存为 .dot 文件，用 Graphviz 渲染
```

## 性能问题

### Q8：自动微分比手写梯度慢吗？

**A**：对于简单函数，手写梯度可能更快。但对于复杂函数：
- 自动微分：**一次编写，处处可用**
- 手写梯度：**每次修改网络结构，都要重推梯度**

对于深度学习，自动微分的便利性远超性能损失。

### Q9：如何让自动微分跑得更快？

**A**：
1. **使用算子融合**：`AD.fuse(x).add(1.0).sigmoid().compute()`
2. **使用 GPU**：`-Dyishape.gpu=true`
3. **使用 HPC**：安装 `yishape-math-hpc`
4. **减少计算图大小**：避免不必要的中间变量

### Q10：`AD.fuse()` 和 `AD.elementwise()` 有什么区别？

**A**：
- `AD.fuse()`：手动指定融合链，性能最优
- `AD.elementwise()`：自动检测可融合的运算，更安全

推荐：先用 `AD.elementwise()`，如果性能不够再用 `AD.fuse()`。

## 进阶问题

### Q11：如何实现自定义的激活函数？

**A**：继承 `CustomOp` 类：

```java
public class MyActivation extends CustomOp {
    @Override
    protected ForwardResult forward(IDoubleVector[] inputs) {
        IDoubleVector x = inputs[0];
        // 前向计算
        return new ForwardResult(result, null);
    }

    @Override
    public IDoubleVector[] backward(IDoubleVector gradOutput, Object context) {
        // 反向计算
        return new IDoubleVector[]{gradInput};
    }
}
```

### Q12：如何用自动微分实现神经网络？

**A**：参考本章7.2节的"实战：线性回归的梯度"，然后扩展到多层网络。

### Q13：Neural ODE 有什么实际应用？

**A**：
- **物理模拟**：求解微分方程
- **生成模型**：Flow-based 模型
- **时间序列**：连续时间动态系统

### Q14：如何调试复杂的计算图？

**A**：
1. **打印计算图**：`AD.render(y)`
2. **数值校验**：`AD.checkGradient(fn, x, 1e-5)`
3. **分段检查**：把大计算图拆成小段，逐段验证
4. **可视化梯度**：打印每层的梯度值

---

**还有问题？** 欢迎在 GitHub Issues 提问！
