# 稠密 float / double 辅助说明 / Dense float / double helpers

本库**不单独维护**与「 ndarray 风格」平行的一套类名体系；相关能力已并入既有入口。**`float` 路径**下广播与一维工具在接口内**转为 `double` 调用实现后再转回 `float`**（直方图分箱边界仍为 `double[]`，与 double 路径一致）。

This library does **not** maintain a parallel ndarray-style class hierarchy; related capabilities are merged into existing entry points. On the **`float`** path, broadcasting and one-dimensional helpers **convert to `double`, call the implementation, then convert back** (histogram bin edges remain `double[]`, same as the double path).

| 能力 / Capability | 使用方式 / Usage |
|-------------------|------------------|
| 二维广播 / 2D broadcasting | **`IMatrix.broadcastShape`** / **`broadcastTo`**；逐元素运算为 **`IMatrix.broadcastElementWise`（`double[][]` / `IMatrix<Double>`）** 与 **`float[][]`**；两 **`IMatrix<Float>`** 时用 **`IFloatMatrix.broadcastElementWise`**（Java 泛型擦除限制） / Element-wise ops: **`IMatrix.broadcastElementWise`** for `double[][]` / `IMatrix<Double>` and `float[][]`; for two **`IMatrix<Float>`** operands use **`IFloatMatrix.broadcastElementWise`** (type erasure). |
| 矩阵乘、转置、行和等 / Matrix multiply, transpose, row sums, etc. | `IMatrix#mmul`、`IMatrix#transpose`、`IMatrix#rowSums` 等（见 `IMatrix`） / See `IMatrix` for `#mmul`, `#transpose`, `#rowSums`, etc. |
| 一维选取、重复、布尔过滤 / 1D fancy index, repeat, boolean filter | `IVector#fancyGet`、`IVector#repeat`、`IVector#booleanGet`（见 `IVector`） / See `IVector` for `#fancyGet`, `#repeat`, `#booleanGet`. |
| 直方图、分箱、`polyfit`、`where` / Histogram, bins, `polyfit`, `where` | 首选 **`IVector`** 的静态方法（`IDoubleVector` / `IFloatVector` 为同名委托）；直方图结果为 **`IVector.HistogramResult`** / Prefer **`IVector`** static methods (`IDoubleVector` / `IFloatVector` delegate); histogram result type **`IVector.HistogramResult`**. |
| `.npy` 文件 / `.npy` files | `com.yishape.lab.math.util.NpyArrayIO`（仅此独立工具类） / Standalone utility **`NpyArrayIO`** only. |
| 实数 FFT / Real FFT | `RereFFT` |

```java
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.util.NpyArrayIO;

IMatrix<Double> c = IMatrix.broadcastElementWise(a, b, Double::sum);
IMatrix<Float> cF = IFloatMatrix.broadcastElementWise(mF1, mF2, Double::sum);
IMatrix<Double> p = m1.mmul(m2);
IVector<Double> coef = (IVector<Double>) IVector.polyfit(xVec, yVec, 1);
IVector<Float> coefF = (IVector<Float>) IVector.polyfit(xVecF, yVecF, 1);
IMatrix<Double> roundTrip = NpyArrayIO.fromByteArray(NpyArrayIO.toByteArray(matrix));
```

单元测试：`src/test/java/com/yishape/lab/math/linalg/DenseDoubleArrayUtilitiesTest.java`。

Unit tests: `src/test/java/com/yishape/lab/math/linalg/DenseDoubleArrayUtilitiesTest.java`.

更多见 [Matrix-Examples.md](Matrix-Examples.md)、[Vector-Examples.md](Vector-Examples.md)。 / See also [Matrix-Examples.md](Matrix-Examples.md) and [Vector-Examples.md](Vector-Examples.md).
