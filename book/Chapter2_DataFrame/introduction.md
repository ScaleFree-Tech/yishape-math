# 第 2 章：结构化数据与 DataFrame

## 开场：数据分析师每天第一件事

数据分析师入职第一天，导师丢给他一个文件夹：

> 「这是上个月的用户行为数据，你先把数据清洗一下，周五给我个报告。」

打开一看：CSV 文件，100 万行，30 列。列名是 `user_id`、`dt`、`city_tier`、`is_pay`、`pay_amt`、`device_type`……各种类型混在一起：用户 ID 是字符串，城市等级是整数，付费金额是浮点，设备类型又是字符串。

**这个文件不是 `IMatrix`——矩阵只接受数字，而这里面有字符串、有浮点、有缺失值、有异常值。**

这就是 `DataFrame` 诞生的理由：让数据分析师用一套统一的工具，处理现实世界里的「脏乱差」数据。

**历史故事：pandas 诞生前，数据分析师是怎么活的？**

2008 年之前，主流数据分析工具是 R、SAS、Excel。

**Wes McKinney** 在美国量化对冲基金工作，每天处理几十 GB 金融数据。他发现：R 跑不动大数据，Excel 10 万行就卡，SAS 贵到离谱。

他的解决方案：写一个 Python 库，移植 R 的 `data.frame` 概念。2008 年，**pandas** 诞生——名字来自计量经济学术语「面板数据」（panel data）。

今天，pandas 是 GitHub 上被 fork 最多的 Python 库之一。但它的致命弱点：**单线程，100 GB 数据直接撑爆内存**。

YiShape-Math DataFrame 诞生的背景：**JVM 生态里，一直没有好用的 DataFrame**。我们不需要重复造 pandas 的轮子，但需要一个 Java/Kotlin/Scala 生态里原生可用的选择。



**没有 DataFrame 之前：一场噩梦**

假设你只有 `String[][]` 和 `double[][]`，没有 DataFrame——会发生什么？

```java
// 你的数据存在 String[][] 里
String[][] data = readCsv("employees.csv");  // "Alice,25,50000"
// 想算平均薪资？你得：
double sum = 0; int count = 0;
for (int i = 1; i < data.length; i++) {     // 跳过表头行
    try {
        sum += Double.parseDouble(data[i][2]); // 第3列是薪资，索引是2！
        count++;
    } catch (NumberFormatException e) { /* 跳过脏数据 */ }
}
double avg = sum / count;  // 终于算出来了
```

**如果 CSV 里有缺失值 `"NA"`、`""`、`.` 混在一起？** 如果薪资列有 `"50000"`、`"NULL"`、空字符串混在一起？手动处理每一种边界情况，能写出 200 行工程代码。

更可怕的是：**某天 CSV 格式变了，列顺序改了**，你的索引 `[2]` 就指向了完全不同的列——但编译器不会报错，你的平均值悄然变成垃圾。

DataFrame 让你彻底摆脱这种噩梦：
```java
DataFrame df = DataFrame.readCsv("employees.csv");
double avg = df.column("salary").numeric().mean();  // 清晰、安全、自动处理缺失值
```



---

## 2.0 为什么需要 DataFrame？

现实数据很少是「一列数字」——通常是表：员工表含姓名（字符串）、工龄（整数）、薪资（浮点）；传感器日志含时间戳（字符串）、温度（浮点）、异常标记（0/1）。

`IMatrix` 可以存这些数据，但有两个根本局限：
- **同质约束**：矩阵所有元素必须同类型（`Double`），无法直接存储姓名和薪资
- **语义缺失**：第 3 列是「薪资」还是「工龄」？从矩阵里看不出来

DataFrame（数据框）正是解决这两个问题的结构：**按列存储、每列独立类型、表头语义**。灵感来自 R 的 `data.frame` 和 Python 的 `pandas.DataFrame`，但用纯 Java 实现，位于 `com.yishape.lab.math.data`。

```java
import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.data.Column;
import com.yishape.lab.math.data.ColumnType;

// 创建含字符串列和数值列的 DataFrame
DataFrame df = new DataFrame();

Column nameCol = new Column();
nameCol.setName("name");
nameCol.setColumnType(ColumnType.String);
nameCol.setData(Lists.of("Alice", "Bob", "Charlie"));  // 需要 java.util.List
df.addColumn(nameCol);

Column ageCol = new Column();
ageCol.setName("age");
ageCol.setColumnType(ColumnType.Numeric);
ageCol.setData(Lists.of(25.0f, 30.0f, 35.0f));
df.addColumn(ageCol);

System.out.println(df);
// 输出:
// name      | age
// ----------+-----
// Alice     | 25.0
// Bob       | 30.0
// Charlie   | 35.0
```

---

## 2.0.1 DataFrame ↔ IMatrix 的角色分工

| 场景 | 推荐类型 | 原因 |
|------|----------|------|
| 异构列（字符串 + 数值混合） | `DataFrame` | 天然支持多类型 |
| 纯数值计算（矩阵乘法、SVD 等） | `IMatrix` | 高效、连续内存 |
| CSV 读取后直接做统计/ML | DataFrame → `toMatrix()` | 取数值列，转矩阵，再进入第 1、4、5 章的算法 |

---

## 2.0.2 本章模块结构

| 节次 | 文件 | 内容提要 |
|------|------|----------|
| **2.1** | [2.1.md](2.1.md) | DataFrame 创建：`readCsv()`、构造函数、addColumn |
| **2.2** | [2.2.md](2.2.md) | 数据访问：shape、列名/类型、`get()`、`getColumnByName()` |
| **2.3** | [2.3.md](2.3.md) | 切片：`sliceColumn()`、`slice(rowExp, colExp)` NumPy 风格 |
| **2.4** | [2.4.md](2.4.md) | 类型转换与 I/O：`toMatrix()`、`toCsv()`、`Column` 转换 |
| **2.5** | [2.5.md](2.5.md) | 完整预处理流程：数据加载 → 清洗 → 特征选择 → 输入 ML |
| **2.6** | [2.6.md](2.6.md) | 本章小结与知识地图 |

---

## 2.0.3 从数据到图（通用流水线）

```
CSV 文件 / 数据库查询
    ↓
DataFrame（加载、解析）
    ↓
数据清洗（空值、异常值） + 特征选择
    ↓
toMatrix()（只取数值列）
    ↓
第 1/4/5 章：统计分析、回归、聚类…
    ↓
Plots.of(...)（可视化）
```

---
[← 第1章：返回上一章](introduction.md) ｜ [下一章：数据分析实战 →](2.1.md)
