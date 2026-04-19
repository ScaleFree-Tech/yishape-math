# YiShape-Math 文档索引 / Documentation Index

本目录包含与 `yishape-math` 库配套的概念说明、API 摘要与可运行示例（`examples/`）。根目录 [`README.md`](../README.md) 提供项目简介、模块概览与安装方式。

This directory contains conceptual notes, API summaries, and runnable examples under `examples/`. The repository root [`README.md`](../README.md) covers the project overview, module map, and installation.

## 建议学习路径（初学者）/ Suggested learning path

1. **总览 / Overview**：阅读仓库根目录 [`README.md`](../README.md) 的「主要功能」与「快速开始」，确认 Java 21 与 Maven 依赖。 / Read **Key Functions** and **Quick Start** in the root [`README.md`](../README.md); confirm Java 21 and Maven dependency.
2. **向量 → 矩阵 / Vectors → matrices**：[`Vector-Operations.md`](Vector-Operations.md) + [`examples/Vector-Examples.md`](examples/Vector-Examples.md)，再读 [`Matrix-Operations.md`](Matrix-Operations.md) + [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md)。线性代数以 `Linalg`、`IMatrix`、`IVector` 为主入口。广播、一维工具与 `.npy` 等入口速查见 [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md) 文首。 / Then [`Matrix-Operations.md`](Matrix-Operations.md) + [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md). Use `Linalg`, `IMatrix`, `IVector` as main linear algebra entry points. Quick reference for broadcasting, 1D helpers, and `.npy`: opening of [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md).
3. **数学工具 / Math utilities**：[`Math-Utilities.md`](Math-Utilities.md) + [`examples/Math-Utilities-Examples.md`](examples/Math-Utilities-Examples.md)（`RereMathUtil`）。 / [`Math-Utilities.md`](Math-Utilities.md) + [`examples/Math-Utilities-Examples.md`](examples/Math-Utilities-Examples.md) (`RereMathUtil`).
4. **按主题深入**（顺序可调整）/ **Topic deep dives** (order flexible)：
   - 统计 / Statistics：[`Statistics-Operations.md`](Statistics-Operations.md) + [`examples/Statistics-Examples.md`](examples/Statistics-Examples.md)（`Stats`）
   - 优化 / Optimization：[`Optimization-Algorithms.md`](Optimization-Algorithms.md) + [`examples/Optimization-Examples.md`](examples/Optimization-Examples.md)
   - 机器学习 / ML：[`Machine-Learning.md`](Machine-Learning.md) + [`examples/Machine-Learning-Examples.md`](examples/Machine-Learning-Examples.md)（`ML`, `ml`）
   - 数据框 / DataFrame：[`DataFrame-Operations.md`](DataFrame-Operations.md) + [`examples/DataFrame-Examples.md`](examples/DataFrame-Examples.md)（包 **`com.yishape.lab.math.data`**）
   - 可视化 / Visualization：[`Visualization-Plotting.md`](Visualization-Plotting.md) + [`examples/Visualization-Plotting-Examples.md`](examples/Visualization-Plotting-Examples.md)（`Plots`, `com.yishape.lab.math.viz`）
   - 时间序列 / Time series：[`Time-Series-Analysis.md`](Time-Series-Analysis.md) + [`examples/Time-Series-Examples.md`](examples/Time-Series-Examples.md)
   - 信号处理 / Signal processing：[`Signal-Processing.md`](Signal-Processing.md) + [`examples/Signal-Processing-Examples.md`](examples/Signal-Processing-Examples.md)
   - 音频 / Audio：[`Audio-Operations.md`](Audio-Operations.md) + [`examples/Audio-Examples.md`](examples/Audio-Examples.md)
5. **查阅接口细节 / API details**：[`API-Reference.md`](API-Reference.md)（与源码对照使用）。 / [`API-Reference.md`](API-Reference.md) alongside source code.

## 文档文件一览 / Files

| 文档 / Document | 说明 / Description |
|-----------------|-------------------|
| [API-Reference.md](API-Reference.md) | 公共 API 与模块索引 / Public API and module index |
| [Vector-Operations.md](Vector-Operations.md) | 向量接口与运算；`dot(IMatrix)` 与 `mmul(IMatrix)` 等价说明见文内 NumPy 对照 / Vector API; `dot(IMatrix)` ≡ `mmul(IMatrix)` in NumPy table |
| [Matrix-Operations.md](Matrix-Operations.md) | 矩阵接口与运算；含与 NumPy `np.dot` / `frobeniusInnerProduct` 对照；[`examples/Matrix-Examples.md`](examples/Matrix-Examples.md) 文首含广播与相关 API 速查 / Matrix API; includes NumPy `np.dot` vs Frobenius mapping; [`examples/Matrix-Examples.md`](examples/Matrix-Examples.md) opens with broadcasting quick reference |
| [Math-Utilities.md](Math-Utilities.md) | 数学工具类 / Math utilities |
| [Statistics-Operations.md](Statistics-Operations.md) | 统计与分布 / Statistics and distributions |
| [Machine-Learning.md](Machine-Learning.md) | 机器学习概述 / Machine learning overview |
| [Optimization-Algorithms.md](Optimization-Algorithms.md) | 优化与线性规划 / Optimization and linear programming |
| [DataFrame-Operations.md](DataFrame-Operations.md) | DataFrame |
| [Visualization-Plotting.md](Visualization-Plotting.md) | 绘图 / Plotting |
| [Time-Series-Analysis.md](Time-Series-Analysis.md) | 时间序列 / Time series |
| [Signal-Processing.md](Signal-Processing.md) | 信号处理 / Signal processing |
| [Audio-Operations.md](Audio-Operations.md) | 音频分析 / Audio analysis |
| [examples/](examples/) | 各主题示例代码片段（矩阵示例文首含广播与相关 API 速查）/ Topic example snippets (matrix examples start with broadcasting quick reference) |

## 使用示例代码时的注意 / Notes on example code

- 许多 `examples/*.md` 中的类为**完整可编译示例**（含 `main` 与 `import`）；部分小节为**节选**，需在同一文件靠前位置补齐 `import`，或参考首节完整示例（如 [`Machine-Learning-Examples.md`](examples/Machine-Learning-Examples.md) 文首说明）。 / Many classes in `examples/*.md` are **full compilable examples** (`main` + imports); some sections are **excerpts**—add imports from earlier in the file or follow the first complete example (see [`Machine-Learning-Examples.md`](examples/Machine-Learning-Examples.md)).
- 依赖文件路径的示例（如 `readCsv("employees.csv")`）需在运行目录放置对应文件或改为实际路径。 / File-path examples (e.g. `readCsv("employees.csv")`) need the file in the working directory or an updated path.
- 向量 API 优先使用 **`Linalg.vector` / `Linalg.matrix`**，与文档示例保持一致。 / Prefer **`Linalg.vector` / `Linalg.matrix`** for vectors/matrices, matching the docs.
