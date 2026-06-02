# 数据可视化 (Data Visualization)

## 概述 / Overview

数据可视化包提供了强大的图表绘制功能，支持多种图表类型，包括基础图表、极坐标图表、统计图表和特殊图表。**二维**绘图可在 **JavaFX**、**ECharts** 与 **矢量 SVG** 三种后端之间按场景选用（详见下文「绘图后端」）。该包为数学计算和数据分析提供了直观的可视化支持，能够将复杂的数值数据转换为易于理解的图形表示。

The data visualization package provides powerful charting capabilities, supporting various chart types including basic charts, polar coordinate charts, statistical charts, and special charts. For **2D** plotting, **JavaFX**, **ECharts**, and **vector SVG** backends are available—see *Plot Backends* below. This package provides intuitive visualization support for mathematical calculations and data analysis, converting complex numerical data into easily understandable graphical representations.

## 文档导航 / Reading Guide

以下按阅读顺序整理了本页内容。**完整签名以源码为准**：[IPlot.java](../src/main/java/com/yishape/lab/math/plot/IPlot.java)、[I3dPlot.java](../src/main/java/com/yishape/lab/math/plot/I3dPlot.java)、[Plots.java](../src/main/java/com/yishape/lab/math/plot/Plots.java)。可运行范例见 **[Visualization-Plotting-Examples.md](./examples/Visualization-Plotting-Examples.md)**。

| 小节 | 内容 |
|------|------|
| 快速上手 | 最小示例与 `Plots` 工厂 |
| 核心类 | 后端（JavaFX / ECharts / SVG）、`IPlot` 方法索引、`AxisTicks`、`PlotException` |
| 主要实现类〜三维数据可视化 | 四个实现类 + `I3dPlot` 能力与对比 |
| 基础图表〜完善图表 | 分类型图解与示例 |
| 图表配置〜输出〜样式系统 | 轴、刻度、导出、`PlotStyle`、调色板、主题 |



## 快速上手 / Quick Start

```java
// 创建画布（宽800 × 高600）
IPlot plot = Plots.of(800, 600);

// 线图 / Line chart
IVector<Double> x = Linalg.arange(0.0, 10.0, 0.1);
IVector<Double> y = x.map(v -> Math.sin(v));
Plots.line(x, y).show();  // 显示正弦波

// 散点图 / Scatter chart
IVector<Double> xs = Linalg.randn(100);
IVector<Double> ys = xs.map(v -> v * 2.5 + Math.random() * 0.5);
Plots.scatter(xs, ys).show();

// 柱状图 / Bar chart
IVector<Double> sales = Linalg.vector(new double[]{120, 75, 230, 180, 90});
Plots.bar(sales, List.of("Mon", "Tue", "Wed", "Thu", "Fri")).show();

// 直方图 / Histogram
IVector<Double> data = Linalg.randn(1000);
Plots.hist(data, true).show();  // true = 显示拟合曲线

// 饼图 / Pie chart
IVector<Double> shares = Linalg.vector(new double[]{35, 25, 20, 12, 8});
Plots.pie(shares, List.of("A", "B", "C", "D", "E")).show();

// 箱线图 / Box plot
IVector<Double> group1 = Linalg.randn(50).map(v -> v * 10 + 70);
Plots.boxplot(group1, List.of("Control")).show();

// 保存为 HTML（交互式，可缩放）
Plots.line(x, y).saveAsHtml("sine_wave.html");
```

#### 图表展示 / Chart Gallery

以下展示了YiShape-Math支持的各种图表类型，点击图片可查看详细的使用示例：

The following showcases various chart types supported by YiShape-Math. Click on the images to view detailed usage examples:

##### 基础图表 / Basic Charts

| 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description | 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description |
|---------------------|------------------|-------------------|---------------------|------------------|-------------------|
| **线图 / Line Chart** | [![单向量线图示例](images/line.png)](./Visualization-Plotting.md#单向量线图--single-vector-line-chart) | 展示数据随时间的变化趋势 / Display data trends over time | **多组线图 / Multi-group Line Chart** | [![多组线图示例](images/line_multi.png)](./Visualization-Plotting.md#多组线图--multi-group-line-chart) | 比较不同组别的数据趋势 / Compare data trends across different groups |
| **散点图 / Scatter Chart** | [![散点图示例](images/scatter.png)](./Visualization-Plotting.md#基础散点图--basic-scatter-chart) | 展示两个变量之间的关系 / Display relationships between two variables | **多组散点图 / Multi-group Scatter Chart** | [![多组散点图示例](images/scatter_multi.png)](./Visualization-Plotting.md#多组散点图--multi-group-scatter-chart) | 比较不同组别的数据分布 / Compare data distributions across different groups |
| **饼图 / Pie Chart** | [![饼图示例](images/pie.png)](./Visualization-Plotting.md#饼图--pie-charts) | 展示各部分占整体的比例 / Display proportion of each part to the whole | **柱状图 / Bar Chart** | [![柱状图示例](images/bar.png)](./Visualization-Plotting.md#基础柱状图--basic-bar-chart) | 比较不同类别的数值大小 / Compare numerical values across different categories |
| **分组柱状图 / Grouped Bar Chart** | [![分组柱状图示例](images/grouped_bar.png)](./Visualization-Plotting.md#分组柱状图--grouped-bar-chart) | 多维度比较分析 / Multi-dimensional comparative analysis | **直方图 / Histogram** | [![直方图示例](images/hist.png)](./Visualization-Plotting.md#直方图--histograms) | 展示数据的分布情况 / Display data distribution |

##### 极坐标图表 / Polar Coordinate Charts

| 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description | 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description |
|---------------------|------------------|-------------------|---------------------|------------------|-------------------|
| **极坐标柱状图 / Polar Bar Chart** | [![极坐标柱状图示例](images/polar_bar.png)](./Visualization-Plotting.md#极坐标柱状图--polar-bar-chart) | 在极坐标系中展示柱状图 / Display bar charts in polar coordinate system | **极坐标线图 / Polar Line Chart** | [![极坐标线图示例](images/polar_line.png)](./Visualization-Plotting.md#极坐标线图--polar-line-chart) | 在极坐标系中展示线图 / Display line charts in polar coordinate system |
| **极坐标散点图 / Polar Scatter Chart** | [![极坐标散点图示例](images/polar_scatter.png)](./Visualization-Plotting.md#极坐标散点图--polar-scatter-chart) | 在极坐标系中展示散点图 / Display scatter charts in polar coordinate system | | | |

##### 统计图表 / Statistical Charts

| 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description | 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description |
|---------------------|------------------|-------------------|---------------------|------------------|-------------------|
| **单组箱线图 / Single Group Box Plot** | [![单组箱线图示例](images/boxplot.png)](./Visualization-Plotting.md#单组箱线图--single-group-box-plot) | 展示单一数据集的分布特征和异常值 / Display distribution characteristics and outliers of a single dataset | **多组箱线图 / Multi-group Box Plot** | [![多组箱线图示例](images/boxplot_multi.png)](./Visualization-Plotting.md#多组箱线图--multi-group-box-plot) | 比较多个数据组的分布特征和异常值 / Compare distribution characteristics and outliers across multiple data groups |
| **单组小提琴图 / Single Group Violin Plot** | [![单组小提琴图示例](images/violinplot.png)](./Visualization-Plotting.md#单组小提琴图--single-group-violin-plot) | 展示单一数据集的分布形状和统计特征 / Display distribution shape and statistical features of a single dataset | **多组小提琴图 / Multi-group Violin Plot** | [![多组小提琴图示例](images/violinplot_multi.png)](./Visualization-Plotting.md#多组小提琴图--multi-group-violin-plot) | 比较多个数据组的分布形状和统计特征 / Compare distribution shapes and statistical features across multiple data groups |
| **K线图 / Candlestick Chart** | [![K线图示例](images/candlestick.png)](./Visualization-Plotting.md#k线图蜡烛图--candlestick-chart) | 展示金融数据的开盘价、收盘价等 / Display financial data including opening, closing prices | | | |

##### 特殊图表 / Special Charts

| 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description | 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description |
|---------------------|------------------|-------------------|---------------------|------------------|-------------------|
| **漏斗图 / Funnel Chart** | [![漏斗图示例](images/funnel.png)](./Visualization-Plotting.md#漏斗图--funnel-chart) | 展示流程中各个阶段的转化情况 / Display conversion rates at each stage of a process | **桑基图 / Sankey Diagram** | [![桑基图示例](images/sankey.png)](./Visualization-Plotting.md#桑基图--sankey-diagram) | 展示数据在多个节点之间的流动 / Display data flow between multiple nodes |
| **旭日图 / Sunburst Chart** | [![旭日图示例](images/sunburst.png)](./Visualization-Plotting.md#旭日图--sunburst-chart) | 展示层次数据的比例关系 / Display proportional relationships in hierarchical data | **主题河流图 / Theme River Chart** | [![主题河流图示例](images/theme_river.png)](./Visualization-Plotting.md#主题河流图--theme-river-chart) | 展示时间序列数据中不同主题的变化 / Display trends of different themes in time series data |
| **树图 / Tree Chart** | [![树图示例](images/tree.png)](./Visualization-Plotting.md#树图--tree-chart) | 展示层次结构数据 / Display hierarchical structure data | **矩形树图 / Treemap Chart** | [![矩形树图示例](images/treemap.png)](./Visualization-Plotting.md#矩形树图--treemap-chart) | 通过矩形面积大小表示数据量 / Represent data volumes through rectangle sizes |
| **关系图 / Graph Chart** | [![关系图示例](images/graph.png)](./Visualization-Plotting.md#关系图--graph-chart) | 展示节点之间的连接关系 / Display connections between nodes | **平行坐标图 / Parallel Coordinates Chart** | [![平行坐标图示例](images/parallel.png)](./Visualization-Plotting.md#平行坐标图--parallel-coordinates-chart) | 展示多维数据的分布和关系 / Display distribution and relationships of multi-dimensional data |

##### 扩展图表 / Enhanced Charts

| 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description | 图表类型 / Chart Type | 示例图片 / Example | 描述 / Description |
|---------------------|------------------|-------------------|---------------------|------------------|-------------------|
| **热力图 / Heatmap** | [![热力图示例](images/heatmap.png)](./Visualization-Plotting.md#热力图--heatmap) | 通过颜色深浅表示二维数据大小 / Represent two-dimensional data magnitude through color intensity | **雷达图 / Radar Chart** | [![雷达图示例](images/radar.png)](./Visualization-Plotting.md#雷达图--radar-chart) | 展示多维数据的分布情况 / Display distribution of multi-dimensional data |
| **仪表盘 / Gauge Chart** | [![仪表盘示例](images/gauge.png)](./Visualization-Plotting.md#仪表盘--gauge-chart) | 展示单一指标的当前值 / Display current values of single indicators | | | |

##### 扩展二维 / Extended 2D（`IPlot` 扩展）

| API | 含义 |
|-----|------|
| `area` | 面积图 |
| `step` | 阶梯图 |
| `barh` | 水平柱状图 |
| `barStacked` | 堆叠柱状图 |
| `errorbar` | 误差棒 |
| `scatter(..., sizes)` | 气泡散点 |
| `regplot` | 回归线（可选置信带） |
| `kdeplot` | 一维 KDE |
| `pairplot` | 配对图 |
| `jointplot` | 联合分布 + 边际 |
| `qqplot` | Q–Q 图 |
| `lineWithSecondaryY` | 双 Y 轴曲线 |
| `subplots` / `subplot` | 子图网格与选定子图 |



## 核心类 / Core Classes

### 绘图后端 / Plot Backends

本库**二维**绘图支持三种后端，彼此独立选用（后两者通过全局 Provider 切换；SVG 为专用工厂，不经 Provider）：

| 后端 | 入口 / 类型 | 适合的场景 |
|------|----------------|------------|
| **JavaFX** | `Plots.of(...)` 在 `PlotProvider.JavaFx` 下（默认），或 `Plots.ofJavaFx(...)` | 桌面应用、`show()` 弹窗预览、与 JavaFX / Swing 画布集成；默认能力最全。导出 `saveAsSvg` 时常见为**画布快照封装**（与纯矢量 SVG 后端不同）。 |
| **ECharts** | `Plots.setProvider(PlotProvider.Echarts)` 后 `Plots.of(...)`或 `Plots.ofEcharts(...)` | 浏览器内**交互**（缩放、tooltip、图例联动）、`saveAsHtml` / `toJson` 直接对接前端或内嵌报表；适合 Web 形态交付。 |
| **SVG** | `Plots.setProvider(PlotProvider.Svg)` 后 `Plots.of(...)`或 `Plots.ofSvg(...)` | **无 JavaFX 图形环境**（如部分 CI、纯批处理）、需要**真矢量** `.svg`（放大不糊、可编辑）、文档/论文插图、对依赖与启动成本敏感的服务端出图。图表子集与 JavaFX 语义对齐，复杂交互弱于 JavaFX / ECharts。 |

English: For **2D**, three backends are available: **JavaFX** (default `Plots.of`, desktop & richest features), **ECharts** (`Plots.ofEcharts)`, web-oriented HTML/JSON and interaction), and **vector SVG** (`Plots.ofSvg`, headless-friendly true SVG output). Pick by deployment target (desktop vs browser vs vector export) and whether you need full interactivity.

- **二维**：`Plots.setProvider(PlotProvider.JavaFx | Echarts)`，默认 JavaFx；**矢量 SVG** 使用 `Plots.ofSvg` / `Plots.svg`，不修改全局 Provider。
- **三维**：`Plots.setProvider3d(PlotProvider3d.JavaFx | EchartsGL)`，默认 EChartsGL（交互 WebGL）；JavaFx 为桌面 3D。（三维暂无独立「纯 SVG 几何」后端，仍以 JavaFX / ECharts GL 为主。）

### 1. Plots - 绘图工厂类 / Plotting Factory Class

`Plots` 类提供了创建各种图表类型的静态工厂方法，是绘图功能的主要入口点。该类提供了简洁的API来快速创建和配置图表。

The `Plots` class provides static factory methods for creating various chart types, serving as the main entry point for plotting functionality. This class provides a concise API for quickly creating and configuring charts.

#### 主要工厂方法 / Main Factory Methods

```java
// 全局后端（可随时切换）
Plots.setProvider(Plots.PlotProvider.JavaFx);    // 或 Plots.PlotProvider.Echarts
Plots.setProvider3d(Plots.PlotProvider3d.EchartsGL); // 或 PlotProvider3d.JavaFx

IPlot plot2d = Plots.of();
IPlot sized = Plots.of(800, 600);
IPlot themed = Plots.of(800, 600, "dark");

I3dPlot plot3d = Plots.of3d();
I3dPlot sized3 = Plots.of3d(800, 600);
I3dPlot themed3 = Plots.of3d(800, 600, "dark");

// 另有大量静态方法如 Plots.line(x,y)，等价于链式调用
```

#### 使用示例 / Usage Examples

```java
// 基础使用 / Basic usage
IPlot plot = Plots.of(800, 600);
plot.setTitle("我的图表");
plot.line(x, y);
plot.show();

// 链式调用 / Chained calls
Plots.of(800, 600)
    .title("线图演示") // Line Chart Demo
    .xlabel("X轴") // X-axis
    .ylabel("Y轴") // Y-axis
    .line(x, y)
    .show();

// 直接创建特定图表 / Direct creation of specific charts
Plots.line(x, y).saveAsHtml("line_chart.html"); // 线图 / Line chart
Plots.scatter(x, y).saveAsHtml("scatter_chart.html"); // 散点图 / Scatter chart
Plots.pie(data).saveAsHtml("pie_chart.html"); // 饼图 / Pie chart
Plots.boxplot(data).saveAsHtml("boxplot_chart.html"); // 箱线图 / Box plot
Plots.violinplot(data).saveAsHtml("violin_chart.html"); // 小提琴图 / Violin plot
```

### 2. IPlot 接口 / IPlot Interface

`IPlot` 是**二维**绘图的契约；常见实现为 **`JavaFxPlot`**（默认）、**`EChartsPlot`**（`Plots.ofEcharts(...)`），以及 **`SvgPlot`**（`Plots.ofSvg`）。数据多为 `com.yishape.lab.math.linalg.IVector` / `IMatrix`。同一几何语义通常存在无样式、`PlotStyle` 与样式字符串多套重载。

`IPlot` is the 2D plotting contract; typical implementations are **`JavaFxPlot`** (default), **`EChartsPlot`**(`Plots.ofEcharts`), and **`SvgPlot`** (`Plots.ofSvg`).

下表汇总**代表性方法分组**——完整列表见 [**IPlot.java**](../src/main/java/com/yishape/lab/math/plot/IPlot.java)。

| 分类 | 方法（节选） |
|------|----------------|
| 基础 | `line`、`scatter`、`pie`、`bar`、`hist` 及 hue / 样式重载 |
| 极坐标 | `polarBar`、`polarLine`、`polarScatter` |
| 统计 | `boxplot`、`violinplot`、`candlestick` |
| 关系数据 | `funnel`、`sankey`、`sunburst`、`themeRiver`、`tree`、`treemap`、`graph`、`parallel` |
| 矩阵 / KPI | `heatmap`、`radar`、`gauge` |
| 扩展二维 | `area`、`step`、`barh`、`barStacked`、`errorbar`、`scatter(x,y,sizes)`、`regplot`、`kdeplot`、`pairplot`、`jointplot`、`qqplot`、`lineWithSecondaryY`、`subplots`、`subplot` |
| 坐标轴 | `xscale`、`yscale`、`y2label` |
| 样式主题 | `setDefaultStyle`、`setPalette`、`enableStyleSystem`、`enableThemeSystem`、`applyTheme`、`registerTheme`、`createGradientTheme`、`createMonochromeTheme` |
| 流式视图 | `title`、`xlabel`、`ylabel`、`size`、`theme`、`show` |
| 导出 | `saveAsHtml`、`saveAsSvg`、`saveAsPng`、`toBase64Svg`、`toBase64Png`、`toHtml`、`toJson` |
| 命令式视图 | `setTitle`、`setXlabel`、`setYlabel`、`setXticks`、`setYticks`、`getWidth`、`getHeight`、`getTheme` |

最小链式示例：

```java
import com.yishape.lab.math.plot.AxisTicks;
// ...
Plots.of(800, 600)
    .line(x, y, "C0-")
    .title("演示")
    .saveAsHtml("demo.html");
```


### AxisTicks 类 / AxisTicks Class

`AxisTicks` 类用于配置坐标轴的刻度和标签，提供灵活的坐标轴定制功能。

The `AxisTicks` class is used to configure axis ticks and labels, providing flexible axis customization functionality.

```java
public class AxisTicks {
    // 构造函数 / Constructors
    public AxisTicks();
    public AxisTicks(IVector<Double> tickValues);
    public AxisTicks(IVector<Double> tickValues, List<String> tickLabels);
    
    // 属性访问 / Property access
    public IVector<Double> getTickValues();
    public void setTickValues(IVector<Double> tickValues);
    public List<String> getTickLabels();
    public void setTickLabels(List<String> tickLabels);
    
    // 工具方法 / Utility methods
    public void addTickLabel(String label);
    public boolean hasTickValues();
    public boolean hasTickLabels();
}
```

### PlotException 类 / PlotException Class

`PlotException` 是绘图过程中使用的专用异常类，用于处理各种绘图错误和异常情况。

`PlotException` is a specialized exception class used during the plotting process to handle various plotting errors and exceptional situations.

```java
public class PlotException extends RuntimeException {
    // 构造函数 / Constructors
    public PlotException(String message);
    public PlotException(String message, Throwable cause);
    public PlotException(Throwable cause);
}
```

#### 异常类型 / Exception Types

- **数据验证异常** / **Data Validation Exceptions**: 当输入数据不符合要求时抛出 / Thrown when input data does not meet requirements
- **渲染异常** / **Rendering Exceptions**: 当图表渲染过程中出现错误时抛出 / Thrown when errors occur during chart rendering
- **配置异常** / **Configuration Exceptions**: 当图表配置参数无效时抛出 / Thrown when chart configuration parameters are invalid
- **文件操作异常** / **File Operation Exceptions**: 当保存或读取文件时出现错误时抛出 / Thrown when errors occur during file save or read operations

#### 使用示例 / Usage Example

```java
try {
    IPlot plot = Plots.of(800, 600);
    plot.line(x, y);
    plot.saveAsHtml("chart.html");
} catch (PlotException e) {
    System.err.println("绘图错误: " + e.getMessage()); // Plotting error
    // 处理绘图异常 / Handle plotting exceptions
} catch (Exception e) {
    System.err.println("未知错误: " + e.getMessage()); // Unknown error
    // 处理其他异常 / Handle other exceptions
}
```

---

## 主要实现类 / Main Implementation Classes

| 维度 | 类 | 说明 |
|------|-----|------|
| 二维 | `JavaFxPlot` | JavaFX 画布；桌面 `show()` |
| 二维 | `EchartsPlot` | ECharts HTML / JSON |
| 二维 | `SvgPlot` | 纯矢量 SVG 元素绘制；`Plots.ofSvg`；适合无 JavaFX 与真 SVG 导出 |
| 三维 | `JavaFx3dPlot` | JavaFX 3D；`saveAsHtml` 为含 **PNG** 快照的页面 |
| 三维 | `Echarts3dPlot` | 可漫游 WebGL 场景（CDN 载入 ECharts/ECharts GL） |

旧文档出现的 **`RerePlot` 已不存在**——类型请写 **`IPlot`** / **`I3dPlot`**。


## 3D数据可视化 / 3D Data Visualization

3D数据可视化包提供了丰富的三维图表绘制功能，支持两种实现后端，满足不同应用场景的需求。

The 3D data visualization package provides rich three-dimensional chart drawing capabilities, supporting two implementation backends to meet different application scenario requirements.

### 3D后端选择 / 3D Backend Selection

```java
// 默认后端：ECharts GL（推荐用于Web/交互式）
// Default backend: ECharts GL (recommended for web/interactive)
I3dPlot plot = Plots.of3d(800, 600);

// 切换到JavaFX 3D后端（桌面应用）
// Switch to JavaFX 3D backend (desktop applications)
Plots.setProvider3d(Plots.PlotProvider3d.JavaFx);
I3dPlot plot2 = Plots.of3d(800, 600);

// 切换回ECharts GL后端
// Switch back to ECharts GL backend
Plots.setProvider3d(Plots.PlotProvider3d.EchartsGL);
```

### 3D核心接口 / 3D Core Interface

#### I3dPlot 接口 / I3dPlot Interface

`I3dPlot` 契约与 **完整重载**（含 `PlotStyle` / 样式字符串、`route3d`、`grouped bar3d`、`radar3d` 系列名等）见源码 [**I3dPlot.java**](../src/main/java/com/yishape/lab/math/plot/I3dPlot.java)。

| 分类 | API（节选） |
|------|-------------|
| 关系与分布 | `scatter3d`、`scatterBubble3d`、`line3d`、`route3d`、`density3d` |
| 统计 | `bar3d`（含 `BarExtrusion3D`、`hue` 分组）、`pie3d`、`hist3d`、`boxplot3d` |
| 曲面与网格 | `surface3d`、`contour3d`、`wireframe3d`、`heatmap3d`、`waterfall3d`、`terrain3d`、`areaFill3d` |
| 矢量 | `vectorField3d`、`streamlines3d` |
| 关系 / 多维 | `graph3d`、`radar3d` |
| 样式与主题 | 与二维类似的 `setDefaultStyle`、`applyTheme`、`setPalette`、`enableThemeSystem` … |
| 视图与导出 | `title`、`xlabel`、`ylabel`、`zlabel`、`size`、`theme`、`show`、`saveAsHtml`、`toHtml`、`toJson` |

> **后端差异**：`Echarts3dPlot` 的 `saveAsHtml` / `toHtml` 产出可交互 GL 页面（需网络加载 CDN）；`JavaFx3dPlot` 的 `saveAsHtml` 实为 **快照 PNG** 嵌入 HTML，`show()` 为桌面 Stage；大数据散点可考虑先降采样或使用 `Plots.setProvider3d(EchartsGL)`。

### 3D柱状图柱体形状 / 3D Bar Extrusion Shapes

```java
public enum BarExtrusion3D {
    BOX,        // 长方体柱（默认）
    CYLINDER,   // 圆柱体
    CONE        // 圆锥体
}
```

### 3D图表使用示例 / 3D Chart Usage Examples

#### 基础3D散点图 / Basic 3D Scatter Plot

```java
// 生成3D数据
IVector<Double> x = Linalg.randn(100);
IVector<Double> y = Linalg.randn(100);
IVector<Double> z = Linalg.randn(100);

// 创建3D散点图（ECharts GL后端，可交互）
Plots.of3d(800, 600)
    .scatter3d(x, y, z)
    .title("3D Gaussian Cloud")
    .xlabel("X")
    .ylabel("Y")
    .zlabel("Z")
    .saveAsHtml("scatter3d.html");  // 生成可交互的Web图表
```

#### 带分组的3D散点图 / 3D Scatter with Hue

```java
IVector<Double> x = Linalg.vector(IntStream.range(0, 60).mapToDouble(i -> i).toArray());
IVector<Double> y = Linalg.vector(IntStream.range(0, 60).mapToDouble(i -> Math.sin(i * 0.1)).toArray());
IVector<Double> z = Linalg.vector(IntStream.range(0, 60).mapToDouble(i -> i * 0.5).toArray());
List<String> hue = IntStream.range(0, 60).mapToObj(i -> "Group" + (i % 3)).toList();

Plots.of3d(800, 600, "dark")
    .scatter3d(x, y, z, hue)
    .title("3D Spiral with Groups")
    .saveAsHtml("scatter3d_hue.html");
```

#### 3D曲面图 / 3D Surface Plot

```java
// 创建网格数据
int nx = 30, ny = 25;
IVector<Double> x = Linalg.vector(IntStream.range(0, nx).mapToDouble(i -> i * 0.2).toArray());
IVector<Double> y = Linalg.vector(IntStream.range(0, ny).mapToDouble(j -> j * 0.2).toArray());

// 计算Z值
double[][] zd = new double[nx][ny];
for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
        zd[i][j] = Math.sin(x.get(i) * 2) * Math.cos(y.get(j) * 2);
    }
}
IMatrix<Double> z = Linalg.matrix(zd);

Plots.of3d(900, 700, "dark")
    .surface3d(x, y, z)
    .title("3D Surface: z = sin(2x) * cos(2y)")
    .saveAsHtml("surface3d.html");
```

#### 3D柱状图 / 3D Bar Chart

```java
List<String> categories = List.of("Product A", "Product B", "Product C", "Product D");
IVector<Double> values = Linalg.vector(new double[]{120, 200, 150, 80});

// BOX柱体（默认）
Plots.of3d(800, 600)
    .bar3d(categories, values, I3dPlot.BarExtrusion3D.BOX)
    .title("3D Bar Chart - Box Style")
    .saveAsHtml("bar3d_box.html");

// CYLINDER柱体
Plots.of3d(800, 600)
    .bar3d(categories, values, I3dPlot.BarExtrusion3D.CYLINDER)
    .title("3D Bar Chart - Cylinder Style")
    .saveAsHtml("bar3d_cylinder.html");

// CONE柱体
Plots.of3d(800, 600)
    .bar3d(categories, values, I3dPlot.BarExtrusion3D.CONE)
    .title("3D Bar Chart - Cone Style")
    .saveAsHtml("bar3d_cone.html");
```

#### 3D向量场 / 3D Vector Field

```java
// 创建网格采样点
int g = 5;
int n = g * g * g;
double[] x = new double[n];
double[] y = new double[n];
double[] z = new double[n];
double[] u = new double[n];
double[] v = new double[n];
double[] w = new double[n];

int idx = 0;
for (int i = 0; i < g; i++) {
    for (int j = 0; j < g; j++) {
        for (int k = 0; k < g; k++) {
            x[idx] = i * 0.6;
            y[idx] = j * 0.6;
            z[idx] = k * 0.5;
            u[idx] = Math.sin(y[idx]);
            v[idx] = -Math.cos(x[idx]);
            w[idx] = 0.15;
            idx++;
        }
    }
}

Plots.of3d(800, 600, "dark")
    .vectorField3d(
        Linalg.vector(x), Linalg.vector(y), Linalg.vector(z),
        Linalg.vector(u), Linalg.vector(v), Linalg.vector(w)
    )
    .title("3D Vector Field")
    .saveAsHtml("vectorfield3d.html");
```

### 3D性能优化 / 3D Performance Optimization

```java
// 大数据量自动降采样
// 当数据点超过2000时，自动启用降采样
IVector<Double> x = Linalg.randn(10000);
IVector<Double> y = Linalg.randn(10000);
IVector<Double> z = Linalg.randn(10000);

Plots.of3d(800, 600)
    .scatter3d(x, y, z)  // 自动降采样到约5000点
    .title("Large Dataset (Auto-sampled)")
    .saveAsHtml("large_scatter3d.html");
```

### 3D主题系统 / 3D Theme System

```java
// 使用预设主题
Plots.of3d(800, 600, "dark")     // 深色主题
    .scatter3d(x, y, z)
    .saveAsHtml("dark_theme.html");

Plots.of3d(800, 600, "futuristic")  // 未来风格主题
    .scatter3d(x, y, z)
    .saveAsHtml("futuristic_theme.html");

// JavaFX 3D后端支持自定义材质和光照配置
// JavaFX 3D backend supports custom material and lighting configuration
Plots.setProvider3d(Plots.PlotProvider3d.JavaFx);
JavaFx3dPlot plot = (JavaFx3dPlot) Plots.of3d(800, 600);
plot.scatter3d(x, y, z).show();
```

### 3D图表特性对比 / 3D Chart Features Comparison

| 特性 / Feature | JavaFX 3D | ECharts GL |
|--------------|-----------|------------|
| **交互方式** / Interaction | 桌面鼠标拖拽 / Desktop mouse drag | Web鼠标+触摸 / Web mouse+touch |
| **导出格式** / Export Format | PNG快照 / PNG snapshot | 交互式HTML / Interactive HTML |
| **性能** / Performance | 大数据量需降采样 / Needs sampling for large data | WebGL硬件加速 / WebGL hardware acceleration |
| **部署方式** / Deployment | 桌面应用 / Desktop app | Web浏览器 / Web browser |
| **旋转缩放** / Rotate/Zoom | 支持 / Supported | 支持 / Supported |
| **图例显示** / Legend | 2D叠加层 / 2D overlay | 原生支持 / Native support |
| **工具提示** / Tooltip | 支持 / Supported | 原生支持 / Native support |


## 基础图表 / Basic Charts

### 1. 线图 / Line Charts

线图用于展示数据随时间或其他连续变量的变化趋势，是数据可视化中最常用的图表类型之一。

Line charts are used to display data trends over time or other continuous variables, and are one of the most commonly used chart types in data visualization.

#### 单向量线图 / Single Vector Line Chart

使用索引作为X轴，适用于展示单一数据序列的变化趋势。

Using index as X-axis, suitable for displaying trends of a single data series.

![单向量线图示例](images/line.png)

```java
// 创建线图（使用索引作为X轴）/ Create line chart (using index as X-axis)
IPlot plot = Plots.of(800, 600);
IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
plot.line(y);
plot.setTitle("单向量线图");
plot.saveAsHtml("single_line_chart.html");
```

#### 双向量线图 / Two Vector Line Chart

指定X和Y轴数据，适用于展示两个变量之间的关系。

Specify X and Y axis data, suitable for displaying relationships between two variables.

```java
// 创建线图（指定X和Y轴数据）/ Create line chart (specify X and Y axis data)
IPlot plot = Plots.of(800, 600);
IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
plot.line(x, y);
plot.setTitle("双向量线图");
plot.setXlabel("时间");
plot.setYlabel("数值");
plot.saveAsHtml("chart.html");
```

#### 多组线图 / Multi-group Line Chart

使用hue参数分组显示多条线，适用于比较不同组别的数据趋势。

Use hue parameter to group multiple lines, suitable for comparing data trends across different groups.

![多组线图示例](images/line_multi.png)

```java
// 创建多组线图 / Create multi-group line chart
IPlot plot = Plots.of(800, 600);
IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
List<String> hue = Arrays.asList("组A", "组A", "组A", "组A", "组A", 
                                "组B", "组B", "组B", "组B", "组B");
plot.line(x, y, hue);
plot.setTitle("多组线图");
plot.saveAsHtml("chart.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建线图 / Create line chart using fluent API
IPlot plot = Plots.of(800, 600, "dark")
    .title("销售趋势图", "2024年各月销售数据") // Sales Trend Chart, Monthly Sales Data for 2024
    .xlabel("月份") // Month
    .ylabel("销售额（万元）") // Sales Amount (10k RMB)
    .line(x, y)
    .show();
```

### 2. 散点图 / Scatter Charts

散点图用于展示两个连续变量之间的关系，能够直观地显示数据的分布模式和相关性。

Scatter charts are used to display relationships between two continuous variables, providing intuitive visualization of data distribution patterns and correlations.

#### 基础散点图 / Basic Scatter Chart

展示两个变量之间的基本关系，适用于相关性分析。

Display basic relationships between two variables, suitable for correlation analysis.

![散点图示例](images/scatter.png)

```java
// 创建散点图 / Create scatter chart
IPlot plot = Plots.of(800, 600);
IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
plot.scatter(x, y);
plot.setTitle("散点图"); // Scatter Chart
plot.saveAsHtml("chart.html"); // 散点图 / Scatter chart
```

#### 多组散点图 / Multi-group Scatter Chart

使用不同颜色或形状区分不同组别的数据点，适用于比较分析。

Use different colors or shapes to distinguish data points from different groups, suitable for comparative analysis.

![多组散点图示例](images/scatter_multi.png)

```java
// 创建多组散点图 / Create multi-group scatter chart
IPlot plot = Plots.of(800, 600);
IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
List<String> hue = Arrays.asList("类别A", "类别A", "类别A", "类别A", "类别A", // Category A
                                "类别B", "类别B", "类别B", "类别B", "类别B"); // Category B
plot.scatter(x, y, hue);
plot.setTitle("多组散点图"); // Multi-group Scatter Chart
plot.saveAsHtml("chart.html"); // 多组散点图 / Multi-group scatter chart
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建散点图 / Create scatter chart using fluent API
IPlot plot = Plots.of(800, 600)
    .title("身高体重关系图", "不同年龄段的身高体重分布") // Height-Weight Relationship, Height-Weight Distribution by Age Group
    .xlabel("身高（cm）") // Height (cm)
    .ylabel("体重（kg）") // Weight (kg)
    .scatter(height, weight)
    .show();
```

### 3. 饼图 / Pie Charts

饼图用于展示各部分占整体的比例，适用于分类数据的可视化。

Pie charts are used to display the proportion of each part to the whole, suitable for visualizing categorical data.

![饼图示例](images/pie.png)

```java
// 创建饼图 / Create pie chart
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{30, 25, 20, 15, 10});
plot.pie(data);
plot.setTitle("饼图"); // Pie Chart
plot.saveAsHtml("chart.html"); // 饼图 / Pie chart
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建饼图 / Create pie chart using fluent API
IPlot plot = Plots.of(600, 600)
    .title("市场份额分布", "2024年各产品线市场份额") // Market Share Distribution, Market Share by Product Line in 2024
    .pie(marketShare)
    .show();
```

### 4. 柱状图 / Bar Charts

柱状图用于比较不同类别的数值大小，是数据可视化中最常用的图表类型之一。

Bar charts are used to compare numerical values across different categories, and are one of the most commonly used chart types in data visualization.

#### 基础柱状图 / Basic Bar Chart

展示单一维度的数据比较，适用于类别数据的可视化。

Display single-dimensional data comparison, suitable for visualizing categorical data.

![柱状图示例](images/bar.png)

```java
// 创建柱状图 / Create bar chart
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
plot.bar(data);
plot.setTitle("柱状图"); // Bar Chart
plot.saveAsHtml("chart.html"); // 柱状图 / Bar chart
```

#### 分组柱状图 / Grouped Bar Chart

使用hue参数分组显示多个系列的数据，适用于多维度比较分析。

Use hue parameter to group multiple series of data, suitable for multi-dimensional comparative analysis.

![分组柱状图示例](images/grouped_bar.png)

```java
// 创建分组柱状图 / Create grouped bar chart
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
List<String> hue = Arrays.asList("组A", "组B", "组A", "组B", "组A");
plot.bar(data, hue);
plot.setTitle("分组柱状图");
plot.saveAsHtml("chart.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建柱状图 / Create bar chart using fluent API
IPlot plot = Plots.of(800, 600)
    .title("销售业绩对比", "2024年各季度销售数据") // Sales Performance Comparison, Quarterly Sales Data for 2024
    .xlabel("季度") // Quarter
    .ylabel("销售额（万元）") // Sales Amount (10k RMB)
    .bar(salesData)
    .show();
```

### 5. 直方图 / Histograms

直方图用于展示数据的分布情况，能够直观地显示数据的集中趋势和离散程度。

Histograms are used to display data distribution, providing intuitive visualization of data central tendency and dispersion.

![直方图示例](images/hist.png)

```java
// 创建直方图（带拟合线）/ Create histogram (with fitting line)
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{1.2, 2.3, 1.8, 3.1, 2.7, 1.5, 2.9, 3.2, 2.1, 2.8});
plot.hist(data, true); // true表示显示正态分布拟合线 / true means show normal distribution fitting line
plot.setTitle("直方图"); // Histogram
plot.setXlabel("数值区间"); // Value range
plot.setYlabel("频次"); // Frequency
plot.saveAsHtml("chart.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建直方图 / Create histogram using fluent API
IPlot plot = Plots.of(800, 600)
    .title("数据分布直方图", "样本数据的正态分布拟合") // Data Distribution Histogram, Normal Distribution Fitting of Sample Data
    .xlabel("数值区间") // Value Range
    .ylabel("频次") // Frequency
    .hist(data, true)
    .show();
```

## 极坐标图表 / Polar Coordinate Charts

极坐标图表使用极坐标系展示数据，适用于周期性数据或需要从中心向外展示数据的场景。极坐标图表通过角度和半径来表示数据，特别适合展示具有周期性特征的数据。

Polar coordinate charts use polar coordinate system to display data, suitable for periodic data or scenarios where data needs to be displayed from center outward. Polar coordinate charts represent data through angles and radii, particularly suitable for displaying data with periodic characteristics.

### 极坐标图表特点 / Polar Chart Features

- **角度轴** / **Angle Axis**: 表示数据的分类或时间维度 / Represents data categories or time dimensions
- **半径轴** / **Radius Axis**: 表示数据的大小或数值 / Represents data magnitude or values
- **周期性展示** / **Periodic Display**: 适合展示具有周期性特征的数据 / Suitable for displaying data with periodic characteristics
- **视觉冲击力** / **Visual Impact**: 提供独特的视觉效果，增强数据表现力 / Provides unique visual effects to enhance data expression

### 1. 极坐标柱状图 / Polar Bar Chart

在极坐标系中展示柱状图，适用于周期性数据的可视化。

Display bar charts in polar coordinate system, suitable for visualizing periodic data.

![极坐标柱状图示例](images/polar_bar.png)

```java
// 创建极坐标柱状图 / Create polar bar chart
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E"); // Category A, B, C, D, E
plot.polarBar(data, categories);
plot.setTitle("极坐标柱状图"); // Polar Bar Chart
plot.saveAsHtml("chart.html"); // 极坐标柱状图 / Polar bar chart
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建极坐标柱状图 / Create polar bar chart using fluent API
IPlot plot = Plots.of(600, 600)
    .title("风向玫瑰图", "24小时风向分布") // Wind Rose Chart, 24-hour Wind Direction Distribution
    .polarBar(windData, directions)
    .show();
```

### 2. 极坐标线图 / Polar Line Chart

在极坐标系中展示线图，适用于周期性趋势数据的可视化。

Display line charts in polar coordinate system, suitable for visualizing periodic trend data.

![极坐标线图示例](images/polar_line.png)

```java
// 创建极坐标线图 / Create polar line chart
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E");
plot.polarLine(data, categories);
plot.setTitle("极坐标线图");
plot.saveAsHtml("chart.html");
```

### 3. 极坐标散点图 / Polar Scatter Chart

在极坐标系中展示散点图，适用于极坐标数据的分布可视化。

Display scatter charts in polar coordinate system, suitable for visualizing polar coordinate data distribution.

![极坐标散点图示例](images/polar_scatter.png)

```java
// 创建极坐标散点图 / Create polar scatter chart
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E"); // Category A, B, C, D, E
plot.polarScatter(data, categories);
plot.setTitle("极坐标散点图"); // Polar Scatter Chart
plot.saveAsHtml("chart.html"); // 极坐标散点图 / Polar scatter chart
```

## 统计图表 / Statistical Charts

统计图表用于展示数据的统计特征，包括分布、异常值、趋势等统计信息。这些图表帮助分析数据的统计性质，识别数据模式和异常情况。

Statistical charts are used to display statistical characteristics of data, including distribution, outliers, trends, and other statistical information. These charts help analyze statistical properties of data and identify data patterns and anomalies.

### 统计图表特点 / Statistical Chart Features

- **数据分布** / **Data Distribution**: 展示数据的分布特征和集中趋势 / Display data distribution characteristics and central tendency
- **异常值检测** / **Outlier Detection**: 识别数据中的异常值和离群点 / Identify outliers and anomalous values in data
- **统计指标** / **Statistical Indicators**: 显示均值、中位数、四分位数等统计指标 / Display statistical indicators such as mean, median, quartiles
- **比较分析** / **Comparative Analysis**: 支持多组数据的统计比较 / Support statistical comparison of multiple data groups

### 1. 箱线图 / Box Plot

箱线图用于展示数据的分布特征，包括中位数、四分位数、异常值等统计信息。

Box plots are used to display data distribution characteristics, including median, quartiles, outliers, and other statistical information.

#### 单组箱线图 / Single Group Box Plot

展示单一数据集的分布情况，包含中位数、四分位数、异常值等关键统计信息。

Display distribution of a single dataset, including median, quartiles, outliers, and other key statistical information.

![单组箱线图示例](images/boxplot.png)

```java
// 创建单组箱线图 / Create single group box plot
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
plot.boxplot(data);
plot.setTitle("单组箱线图", "数据分布统计特征分析"); // Single Group Box Plot, Statistical Feature Analysis of Data Distribution
plot.setXlabel("数据集"); // Dataset
plot.setYlabel("数值"); // Value
plot.saveAsHtml("single_boxplot.html"); // 单组箱线图 / Single group box plot
```

#### 多组箱线图 / Multi-group Box Plot

比较多个数据组的分布情况，适用于对比分析不同组别的统计特征。

Compare distributions of multiple data groups, suitable for comparative analysis of statistical characteristics across different groups.

![多组箱线图示例](images/boxplot_multi.png)

```java
// 创建多组箱线图 / Create multi-group box plot
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                                     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
List<String> labels = Arrays.asList("组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", // Group A
                                   "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B"); // Group B
plot.boxplot(data, labels);
plot.setTitle("多组箱线图对比", "不同组别的数据分布对比分析"); // Multi-group Box Plot Comparison, Comparative Analysis of Data Distribution by Groups
plot.setXlabel("组别"); // Group
plot.setYlabel("数值"); // Value
plot.saveAsHtml("multi_boxplot.html"); // 多组箱线图 / Multi-group box plot
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建单组箱线图 / Create single group box plot using fluent API
IPlot plot = Plots.of(800, 600)
    .title("数据分布箱线图", "各指标的数据分布情况") // Data Distribution Box Plot, Data Distribution of Various Indicators
    .xlabel("指标") // Indicator
    .ylabel("数值") // Value
    .boxplot(data)
    .show();

// 使用流式API创建多组箱线图 / Create multi-group box plot using fluent API
IPlot multiPlot = Plots.of(800, 600)
    .title("多组数据分布对比", "不同组别的统计特征对比") // Multi-group Data Distribution Comparison, Statistical Feature Comparison by Groups
    .xlabel("组别") // Group
    .ylabel("数值") // Value
    .boxplot(data, labels)
    .show();
```

### 2. 小提琴图 / Violin Plot

小提琴图结合了箱线图和密度图的特点，能够同时展示数据的分布形状和统计特征，适用于数据分布的可视化分析。

Violin plots combine the characteristics of box plots and density plots, displaying both data distribution shape and statistical features, suitable for data distribution visualization analysis.


#### 单组小提琴图 / Single Group Violin Plot

展示单一数据集的分布情况，包含密度曲线和箱线图信息。

Display distribution of a single dataset, including density curves and box plot information.

![单组小提琴图示例](images/violinplot.png)

```java
// 创建小提琴图 / Create violin plot
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
plot.violinplot(data);
plot.setTitle("数据分布小提琴图", "展示数据的分布形状和统计特征");
plot.setXlabel("数值");
plot.setYlabel("密度");
plot.saveAsHtml("violin_plot.html");
```

#### 多组小提琴图 / Multi-group Violin Plot

比较多个数据组的分布情况，适用于对比分析。

Compare distributions of multiple data groups, suitable for comparative analysis.

![多组小提琴图示例](images/violinplot_multi.png)

```java
// 创建多组小提琴图 / Create multi-group violin plot
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                                     2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
List<String> labels = Arrays.asList("组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A",
                                   "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B");
plot.violinplot(data, labels);
plot.setTitle("多组数据分布对比", "不同组别的数据分布对比分析");
plot.setXlabel("组别");
plot.setYlabel("密度");
plot.saveAsHtml("multi_violin_plot.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建小提琴图 / Create violin plot using fluent API
IPlot plot = Plots.of(800, 600)
    .title("数据分布分析", "小提琴图展示数据分布特征")
    .xlabel("数值")
    .ylabel("密度")
    .violinplot(data)
    .show();
```

### 3. K线图（蜡烛图）/ Candlestick Chart

K线图用于展示金融数据的开盘价、收盘价、最高价、最低价，是金融分析中常用的图表类型。

Candlestick charts are used to display financial data including opening, closing, highest, and lowest prices, commonly used in financial analysis.

![K线图示例](images/candlestick.png)

```java
// 创建K线图 / Create candlestick chart
IPlot plot = Plots.of(800, 600);
// 数据格式：[开盘价, 收盘价, 最低价, 最高价] / Data format: [open, close, low, high]
double[][] candlestickArray = {
    {100, 110, 95, 115},
    {110, 120, 105, 125},
    {120, 115, 110, 130},
    {115, 125, 110, 135},
    {125, 130, 120, 140}
};
IMatrix<Double> candlestickData = Linalg.matrix(candlestickArray);
List<String> dates = Arrays.asList("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05");
plot.candlestick(candlestickData, dates);
plot.setTitle("K线图");
plot.saveAsHtml("chart.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建K线图 / Create candlestick chart using fluent API
IPlot plot = Plots.of(1000, 600)
    .title("股票价格K线图", "2024年1月股价走势")
    .xlabel("日期")
    .ylabel("价格（元）")
    .candlestick(priceData, dates)
    .show();
```

## 特殊图表 / Special Charts

特殊图表用于展示复杂的数据关系和层次结构，包括流程、网络、层次等特殊数据可视化需求。这些图表能够处理复杂的数据结构，提供直观的可视化效果。

Special charts are used to display complex data relationships and hierarchical structures, including processes, networks, hierarchies, and other special data visualization needs. These charts can handle complex data structures and provide intuitive visualization effects.

### 特殊图表特点 / Special Chart Features

- **复杂关系** / **Complex Relationships**: 展示数据之间的复杂关联关系 / Display complex relationships between data
- **层次结构** / **Hierarchical Structure**: 支持多层次的树形和嵌套结构 / Support multi-level tree and nested structures
- **流程展示** / **Process Display**: 清晰展示数据流转和转换过程 / Clearly display data flow and transformation processes
- **交互性强** / **High Interactivity**: 提供丰富的交互功能和动态效果 / Provide rich interactive features and dynamic effects

### 1. 漏斗图 / Funnel Chart

漏斗图用于展示流程中各个阶段的转化情况，适用于分析用户行为路径。

Funnel charts are used to display conversion rates at each stage of a process, suitable for analyzing user behavior paths.

![漏斗图示例](images/funnel.png)

```java
// 创建漏斗图 / Create funnel chart
IPlot plot = Plots.of(800, 600);
IVector<Double> data = Linalg.vector(new double[]{100, 80, 60, 40, 20});
List<String> labels = Arrays.asList("访问", "注册", "购买", "支付", "完成");
plot.funnel(data, labels);
plot.setTitle("漏斗图");
plot.saveAsHtml("chart.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建漏斗图 / Create funnel chart using fluent API
IPlot plot = Plots.of(800, 600)
    .title("用户转化漏斗", "从访问到购买的转化流程")
    .funnel(conversionData, stages)
    .show();
```

### 2. 桑基图 / Sankey Diagram

桑基图用于展示数据在多个节点之间的流动情况，适用于分析数据流向和转换关系。

Sankey diagrams are used to display data flow between multiple nodes, suitable for analyzing data flow and transformation relationships.

![桑基图示例](images/sankey.png)

```java
// 创建桑基图 / Create Sankey diagram
IPlot plot = Plots.of(800, 600);

// 创建节点数据 / Create node data
List<Map<String, Object>> nodes = new ArrayList<>();
nodes.add(createNode("source1", "源1"));
nodes.add(createNode("source2", "源2"));
nodes.add(createNode("target1", "目标1"));
nodes.add(createNode("target2", "目标2"));

// 创建连接数据 / Create link data
List<Map<String, Object>> links = new ArrayList<>();
links.add(createLink("source1", "target1", 10));
links.add(createLink("source1", "target2", 20));
links.add(createLink("source2", "target1", 15));
links.add(createLink("source2", "target2", 25));

plot.sankey(nodes, links);
plot.setTitle("桑基图");
plot.saveAsHtml("chart.html");

// 辅助方法 / Helper methods
private static Map<String, Object> createNode(String id, String name) {
    Map<String, Object> node = new HashMap<>();
    node.put("id", id);
    node.put("name", name);
    return node;
}

private static Map<String, Object> createLink(String source, String target, int value) {
    Map<String, Object> link = new HashMap<>();
    link.put("source", source);
    link.put("target", target);
    link.put("value", value);
    return link;
}
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建桑基图 / Create Sankey diagram using fluent API
IPlot plot = Plots.of(1000, 600)
    .title("数据流向图", "各系统间的数据流转情况")
    .sankey(nodes, links)
    .show();
```

### 3. 旭日图 / Sunburst Chart

旭日图用于展示层次数据的比例关系，适用于展示树形结构数据的分布情况。

Sunburst charts are used to display proportional relationships in hierarchical data, suitable for showing distribution of tree-structured data.

![旭日图示例](images/sunburst.png)

```java
// 创建旭日图 / Create sunburst chart
IPlot plot = Plots.of(800, 600);

List<Map<String, Object>> sunburstData = new ArrayList<>();
sunburstData.add(createSunburstNode("root", "根节点", 100));
sunburstData.add(createSunburstNode("child1", "子节点1", 60, "root"));
sunburstData.add(createSunburstNode("child2", "子节点2", 40, "root"));
sunburstData.add(createSunburstNode("grandchild1", "孙节点1", 30, "child1"));
sunburstData.add(createSunburstNode("grandchild2", "孙节点2", 30, "child1"));

plot.sunburst(sunburstData);
plot.setTitle("旭日图");
plot.saveAsHtml("chart.html");

// 辅助方法 / Helper method
private static Map<String, Object> createSunburstNode(String id, String name, int value, String parent) {
    Map<String, Object> node = new HashMap<>();
    node.put("id", id);
    node.put("name", name);
    node.put("value", value);
    node.put("parent", parent);
    return node;
}
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建旭日图 / Create sunburst chart using fluent API
IPlot plot = Plots.of(800, 800)
    .title("组织架构图", "公司各部门人员分布")
    .sunburst(orgData)
    .show();
```

### 4. 主题河流图 / Theme River Chart

主题河流图用于展示时间序列数据中不同主题的变化趋势，适用于分析多维度时间数据。

Theme river charts are used to display trends of different themes in time series data, suitable for analyzing multi-dimensional time data.

![主题河流图示例](images/theme_river.png)

```java
// 创建主题河流图 / Create theme river chart
IPlot plot = Plots.of(800, 600);

List<Map<String, Object>> themeRiverData = new ArrayList<>();
themeRiverData.add(createThemeRiverNode("2024-01-01", "类别A", 10));
themeRiverData.add(createThemeRiverNode("2024-01-01", "类别B", 20));
themeRiverData.add(createThemeRiverNode("2024-01-02", "类别A", 15));
themeRiverData.add(createThemeRiverNode("2024-01-02", "类别B", 25));

List<String> categories = Arrays.asList("类别A", "类别B");
plot.themeRiver(themeRiverData, categories);
plot.setTitle("主题河流图");
plot.saveAsHtml("chart.html");

// 辅助方法 / Helper method
private static Map<String, Object> createThemeRiverNode(String time, String category, int value) {
    Map<String, Object> node = new HashMap<>();
    node.put("time", time);
    node.put("category", category);
    node.put("value", value);
    return node;
}
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建主题河流图 / Create theme river chart using fluent API
IPlot plot = Plots.of(1200, 600)
    .title("新闻热度趋势", "各主题新闻的热度变化")
    .themeRiver(newsData, topics)
    .show();
```

### 5. 树图 / Tree Chart

树图用于展示层次结构数据，适用于展示组织结构、分类体系等树形数据。

Tree charts are used to display hierarchical structure data, suitable for showing organizational structures, classification systems, and other tree-structured data.

![树图示例](images/tree.png)

```java
// 创建树图 / Create tree chart
IPlot plot = Plots.of(800, 600);

List<Map<String, Object>> treeData = new ArrayList<>();
treeData.add(createTreeNode("root", "根节点", 100));
treeData.add(createTreeNode("child1", "子节点1", 60, "root"));
treeData.add(createTreeNode("child2", "子节点2", 40, "root"));
treeData.add(createTreeNode("grandchild1", "孙节点1", 30, "child1"));

plot.tree(treeData);
plot.setTitle("树图"); // Tree Chart
plot.saveAsHtml("chart.html"); // 树图 / Tree chart

// 辅助方法 / Helper method
private static Map<String, Object> createTreeNode(String id, String name, int value, String parent) {
    Map<String, Object> node = new HashMap<>();
    node.put("id", id);
    node.put("name", name);
    node.put("value", value);
    node.put("parent", parent);
    return node;
}
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建树图 / Create tree chart using fluent API
IPlot plot = Plots.of(1000, 800)
    .title("决策树", "机器学习决策树模型") // Decision Tree, Machine Learning Decision Tree Model
    .tree(decisionTreeData)
    .show();
```

### 6. 矩形树图 / Treemap Chart

矩形树图用于展示层次数据的比例关系，通过矩形面积大小表示数据量，适用于展示分类数据的分布。

Treemap charts are used to display proportional relationships in hierarchical data, with rectangle sizes representing data volumes, suitable for showing distribution of categorical data.

![矩形树图示例](images/treemap.png)

```java
// 创建矩形树图 / Create treemap chart
IPlot plot = Plots.of(800, 600);

List<Map<String, Object>> treemapData = new ArrayList<>();
treemapData.add(createTreemapNode("root", "根节点", 100)); // Root Node
treemapData.add(createTreemapNode("child1", "子节点1", 60, "root")); // Child Node 1
treemapData.add(createTreemapNode("child2", "子节点2", 40, "root")); // Child Node 2

plot.treemap(treemapData);
plot.setTitle("矩形树图"); // Treemap Chart
plot.saveAsHtml("chart.html"); // 矩形树图 / Treemap chart

// 辅助方法 / Helper method
private static Map<String, Object> createTreemapNode(String id, String name, int value, String parent) {
    Map<String, Object> node = new HashMap<>();
    node.put("id", id);
    node.put("name", name);
    node.put("value", value);
    node.put("parent", parent);
    return node;
}
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建矩形树图 / Create treemap chart using fluent API
IPlot plot = Plots.of(800, 600)
    .title("市场份额分布", "各产品线市场份额对比") // Market Share Distribution, Market Share Comparison by Product Line
    .treemap(marketShareData)
    .show();
```

### 7. 关系图 / Graph Chart

关系图用于展示节点之间的连接关系，适用于网络分析、社交网络等场景。

Graph charts are used to display connections between nodes, suitable for network analysis, social networks, and other scenarios.

![关系图示例](images/graph.png)

```java
// 创建关系图 / Create graph chart
IPlot plot = Plots.of(800, 600);

// 创建节点数据 / Create node data
List<Map<String, Object>> nodes = new ArrayList<>();
nodes.add(createGraphNode("node1", "节点1")); // Node 1
nodes.add(createGraphNode("node2", "节点2")); // Node 2
nodes.add(createGraphNode("node3", "节点3")); // Node 3

// 创建连接数据 / Create link data
List<Map<String, Object>> links = new ArrayList<>();
links.add(createGraphLink("node1", "node2", 10));
links.add(createGraphLink("node2", "node3", 15));
links.add(createGraphLink("node1", "node3", 20));

plot.graph(nodes, links);
plot.setTitle("关系图"); // Relationship Chart
plot.saveAsHtml("chart.html");

// 辅助方法 / Helper methods
private static Map<String, Object> createGraphNode(String id, String name) {
    Map<String, Object> node = new HashMap<>();
    node.put("id", id);
    node.put("name", name);
    return node;
}

private static Map<String, Object> createGraphLink(String source, String target, int value) {
    Map<String, Object> link = new HashMap<>();
    link.put("source", source);
    link.put("target", target);
    link.put("value", value);
    return link;
}
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建关系图 / Create graph chart using fluent API
IPlot plot = Plots.of(1000, 800)
    .title("社交网络图", "用户关系网络分析")
    .graph(users, relationships)
    .show();
```

### 8. 平行坐标图 / Parallel Coordinates Chart

平行坐标图用于展示多维数据的分布和关系，适用于高维数据的可视化分析。

Parallel coordinates charts are used to display distribution and relationships of multi-dimensional data, suitable for visualization analysis of high-dimensional data.

![平行坐标图示例](images/parallel.png)

```java
// 创建平行坐标图 / Create parallel coordinates chart
IPlot plot = Plots.of(800, 600);

// 创建数据矩阵 / Create data matrix
double[][] dataArray = {
    {1, 2, 3, 4},
    {2, 3, 4, 5},
    {3, 4, 5, 6},
    {4, 5, 6, 7}
};
IMatrix<Double> data = Linalg.matrix(dataArray);
List<String> dimensions = Arrays.asList("维度1", "维度2", "维度3", "维度4");

plot.parallel(data, dimensions);
plot.setTitle("平行坐标图"); // Parallel Coordinates Chart
plot.saveAsHtml("chart.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建平行坐标图 / Create parallel coordinates chart using fluent API
IPlot plot = Plots.of(1200, 600)
    .title("多维数据分布", "各维度数据的分布情况")
    .parallel(multiDimData, dimensionNames)
    .show();
```

### 9. 树图 / Tree Chart

树图用于展示层次结构数据，适用于展示组织结构、分类体系等树形数据。

Tree charts are used to display hierarchical structure data, suitable for showing organizational structures, classification systems, and other tree-structured data.

![树图示例](images/tree.png)

```java
// 创建树图 / Create tree chart
IPlot plot = Plots.of(800, 600);

List<Map<String, Object>> treeData = new ArrayList<>();
treeData.add(createTreeNode("root", "根节点", 100));
treeData.add(createTreeNode("child1", "子节点1", 60, "root"));
treeData.add(createTreeNode("child2", "子节点2", 40, "root"));
treeData.add(createTreeNode("grandchild1", "孙节点1", 30, "child1"));

plot.tree(treeData);
plot.setTitle("树图"); // Tree Chart
plot.saveAsHtml("chart.html");

// 辅助方法 / Helper method
private static Map<String, Object> createTreeNode(String id, String name, int value, String parent) {
    Map<String, Object> node = new HashMap<>();
    node.put("id", id);
    node.put("name", name);
    node.put("value", value);
    node.put("parent", parent);
    return node;
}
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建树图 / Create tree chart using fluent API
IPlot plot = Plots.of(1000, 800)
    .title("决策树", "机器学习决策树模型") // Decision Tree, Machine Learning Decision Tree Model
    .tree(decisionTreeData)
    .show();
```

### 10. 矩形树图 / Treemap Chart

矩形树图用于展示层次数据的比例关系，通过矩形面积大小表示数据量，适用于展示分类数据的分布。

Treemap charts are used to display proportional relationships in hierarchical data, with rectangle sizes representing data volumes, suitable for showing distribution of categorical data.

![矩形树图示例](images/treemap.png)

```java
// 创建矩形树图 / Create treemap chart
IPlot plot = Plots.of(800, 600);

List<Map<String, Object>> treemapData = new ArrayList<>();
treemapData.add(createTreemapNode("root", "根节点", 100)); // Root Node
treemapData.add(createTreemapNode("child1", "子节点1", 60, "root")); // Child Node 1
treemapData.add(createTreemapNode("child2", "子节点2", 40, "root")); // Child Node 2

plot.treemap(treemapData);
plot.setTitle("矩形树图"); // Treemap Chart
plot.saveAsHtml("chart.html");

// 辅助方法 / Helper method
private static Map<String, Object> createTreemapNode(String id, String name, int value, String parent) {
    Map<String, Object> node = new HashMap<>();
    node.put("id", id);
    node.put("name", name);
    node.put("value", value);
    node.put("parent", parent);
    return node;
}
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建矩形树图 / Create treemap chart using fluent API
IPlot plot = Plots.of(800, 600)
    .title("市场份额分布", "各产品线市场份额对比") // Market Share Distribution, Market Share Comparison by Product Line
    .treemap(marketShareData)
    .show();
```

## 完善图表 / Enhanced Charts

完善图表提供了更高级的可视化功能，包括热力图、雷达图、仪表盘等专业图表类型。这些图表适用于专业的数据分析和商业智能场景。

Enhanced charts provide more advanced visualization capabilities, including heatmaps, radar charts, gauges, and other professional chart types. These charts are suitable for professional data analysis and business intelligence scenarios.

### 完善图表特点 / Enhanced Chart Features

- **专业分析** / **Professional Analysis**: 提供专业级的数据分析功能 / Provides professional-level data analysis capabilities
- **多维展示** / **Multi-dimensional Display**: 支持多维数据的可视化展示 / Supports visualization of multi-dimensional data
- **实时监控** / **Real-time Monitoring**: 支持实时数据监控和KPI展示 / Supports real-time data monitoring and KPI display
- **交互丰富** / **Rich Interaction**: 提供丰富的交互功能和自定义选项 / Provides rich interaction features and customization options

### 1. 热力图 / Heatmap

热力图用于展示二维数据的分布情况，通过颜色深浅表示数据大小，适用于相关性分析。

Heatmaps are used to display distribution of two-dimensional data, with color intensity representing data magnitude, suitable for correlation analysis.

![热力图示例](images/heatmap.png)

```java
// 创建热力图 / Create heatmap
IPlot plot = Plots.of(800, 600);

// 创建二维数据矩阵 / Create 2D data matrix
double[][] heatmapArray = {
    {1, 2, 3, 4},
    {2, 3, 4, 5},
    {3, 4, 5, 6},
    {4, 5, 6, 7}
};
IMatrix<Double> data = Linalg.matrix(heatmapArray);
List<String> xLabels = Arrays.asList("X1", "X2", "X3", "X4");
List<String> yLabels = Arrays.asList("Y1", "Y2", "Y3", "Y4");

plot.heatmap(data, xLabels, yLabels);
plot.setTitle("热力图"); // Heatmap
plot.saveAsHtml("chart.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建热力图 / Create heatmap using fluent API
IPlot plot = Plots.of(800, 600)
    .title("相关性热力图", "各指标间的相关性分析")
    .heatmap(correlationMatrix, xLabels, yLabels)
    .show();
```

### 2. 雷达图 / Radar Chart

雷达图用于展示多维数据的分布情况，适用于多指标对比分析。

Radar charts are used to display distribution of multi-dimensional data, suitable for multi-indicator comparative analysis.

![雷达图示例](images/radar.png)

```java
// 创建雷达图 / Create radar chart
IPlot plot = Plots.of(800, 600);

IVector<Double> data = Linalg.vector(new double[]{80, 90, 70, 85, 95, 75});
List<String> indicators = Arrays.asList("指标1", "指标2", "指标3", "指标4", "指标5", "指标6");

plot.radar(data, indicators);
plot.setTitle("雷达图"); // Radar Chart
plot.saveAsHtml("chart.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建雷达图 / Create radar chart using fluent API
IPlot plot = Plots.of(600, 600)
    .title("能力雷达图", "各项技能能力评估")
    .radar(skillScores, skillNames)
    .show();
```

### 3. 仪表盘 / Gauge Chart

仪表盘用于展示单一指标的当前值，适用于监控面板和KPI展示。

Gauge charts are used to display current values of single indicators, suitable for monitoring dashboards and KPI display.

![仪表盘示例](images/gauge.png)

```java
// 创建仪表盘 / Create gauge chart
IPlot plot = Plots.of(800, 600);

double value = 75.5f;
double max = 100.0f;
double min = 0.0f;

plot.gauge(value, max, min);
plot.setTitle("仪表盘"); // Gauge Chart
plot.saveAsHtml("chart.html");
```

#### 流式API示例 / Fluent API Example

```java
// 使用流式API创建仪表盘 / Create gauge chart using fluent API
IPlot plot = Plots.of(400, 400)
    .title("系统性能监控", "CPU使用率实时监控")
    .gauge(cpuUsage, 100.0f, 0.0f)
    .show();
```

## 图表配置 / Chart Configuration

### 1. 基本配置 / Basic Configuration

```java
IPlot plot = Plots.of(800, 600, "dark"); // 设置尺寸和主题 / Set size and theme

// 设置标题 / Set title
plot.setTitle("图表标题"); // Chart title
plot.setTitle("主标题", "副标题"); // Main title, subtitle

// 设置坐标轴标签 / Set axis labels
plot.setXlabel("X轴标签"); // X-axis label
plot.setYlabel("Y轴标签"); // Y-axis label

// 设置坐标轴刻度 / Set axis ticks
AxisTicks xTicks = new AxisTicks();
xTicks.setTickValues(Linalg.vector(new double[]{0, 1, 2, 3, 4, 5}));
xTicks.setTickLabels(Arrays.asList("0", "1", "2", "3", "4", "5"));
plot.setXticks(xTicks);

AxisTicks yTicks = new AxisTicks();
yTicks.setTickValues(Linalg.vector(new double[]{0, 10, 20, 30, 40, 50}));
plot.setYticks(yTicks);
```

### 2. 坐标轴刻度配置 / Axis Ticks Configuration

`AxisTicks` 类提供了灵活的坐标轴刻度配置功能： / The `AxisTicks` class provides flexible axis tick configuration functionality:

```java
// 创建刻度配置 / Create tick configuration
AxisTicks ticks = new AxisTicks();

// 设置刻度值 / Set tick values
ticks.setTickValues(Linalg.vector(new double[]{0, 25, 50, 75, 100}));

// 设置刻度标签 / Set tick labels
ticks.setTickLabels(Arrays.asList("0%", "25%", "50%", "75%", "100%"));

// 添加单个标签 / Add single label
ticks.addTickLabel("自定义标签"); // Custom label

// 检查配置 / Check configuration
if (ticks.hasTickValues()) {
    System.out.println("已设置刻度值"); // Tick values have been set
}
if (ticks.hasTickLabels()) {
    System.out.println("已设置刻度标签"); // Tick labels have been set
}

// 应用到图表 / Apply to chart
plot.setXticks(ticks);
plot.setYticks(ticks);
```

### 3. 异常处理配置 / Exception Handling Configuration

```java
try {
    IPlot plot = Plots.of(800, 600);
    
    // 数据验证 / Data validation
    if (x == null || y == null) {
        throw new PlotException("数据不能为null"); // Data cannot be null
    }
    if (x.length() != y.length()) {
        throw new PlotException("X和Y向量长度必须相等"); // X and Y vector lengths must be equal
    }
    
    // 创建图表 / Create chart
    plot.line(x, y)
        .title("数据图表") // Data chart
        .saveAsHtml("chart.html");
        
} catch (PlotException e) {
    System.err.println("绘图异常: " + e.getMessage()); // Plotting exception
    // 处理绘图相关异常 / Handle plotting-related exceptions
} catch (Exception e) {
    System.err.println("未知异常: " + e.getMessage()); // Unknown exception
    // 处理其他异常 / Handle other exceptions
}
```

### 4. 高级配置 / Advanced Configuration

`IPlot`** 不包含** `getTitle()` / `Legend` / ECharts Java 模型的直接访问——请优先使用 **`title(...)`、`applyTheme`、`toJson`** 等与实现解耦的 API。若在 **EchartsPlot** 中需要深入到 `Option`，应在该实现类的扩展封装内完成（本页不展开）。

## 输出和保存 / Output and Save

### 1. 显示图表 / Display Chart

```java
IPlot plot = Plots.of(800, 600);
// ... 配置图表数据 ... / ... configure chart data ...
plot.show(); // 在浏览器中打开图表 / Open chart in browser
```

### 2. 保存为HTML / Save as HTML

```java
IPlot plot = Plots.of(800, 600);
// ... 配置图表数据 ... / ... configure chart data ...
plot.saveAsHtml("chart.html"); // 保存为HTML文件 / Save as HTML file
```

### 3. 获取HTML内容 / Get HTML Content

```java
IPlot plot = Plots.of(800, 600);
// ... 配置图表数据 ... / ... configure chart data ...
String html = plot.toHtml(); // 获取HTML字符串 / Get HTML string
System.out.println(html);
```

### 4. 获取JSON配置 / Get JSON Configuration

```java
IPlot plot = Plots.of(800, 600);
// ... 配置图表数据 ... / ... configure chart data ...
String json = plot.toJson(); // 获取JSON配置 / Get JSON configuration
System.out.println(json);
```

---

## 样式系统 / Style System

YiShape-Math 实现了完整的统一样式系统，为所有图表类型提供类似 matplotlib 的样式控制功能，并支持 seaborn 风格的数据分组。该系统通过统一的内部实现彻底消除了代码重复，提供了强大而灵活的绘图能力。

YiShape-Math implements a complete unified style system that provides matplotlib-like style control for all chart types and supports seaborn-style data grouping. This system completely eliminates code duplication through unified internal implementation, providing powerful and flexible plotting capabilities.

### 核心特性 / Core Features

- **matplotlib 风格的样式表达式**（如 "r-", "b--o", "g:^"）/ **matplotlib-style style expressions** (e.g., "r-", "b--o", "g:^")
- **完整的样式对象系统**（PlotStyle）/ **Complete style object system** (PlotStyle)
- **丰富的调色板管理**（matplotlib, seaborn, echarts 等）/ **Rich palette management** (matplotlib, seaborn, echarts, etc.)
- **seaborn 风格的分组显示**（hue, style, size 映射）/ **seaborn-style grouping display** (hue, style, size mapping)
- **多维数据分组**（同时按颜色、线型、标记分组）/ **Multi-dimensional data grouping** (grouping by color, line style, and markers simultaneously)
- **智能主题管理**（自动推荐、样式融合）/ **Intelligent theme management** (auto-recommendation, style fusion)

### 快速开始 / Quick Start

#### 1. 基础样式表达式 / Basic Style Expressions

```java
// 传统方式（仍然可用）/ Traditional way (still available)
plot.line(x, y);

// 新样式方式 - 红色实线 / New style way - red solid line
plot.line(x, y, "r-");

// 单向量绘图（使用索引作 X 轴）/ Single vector plotting (using index as X-axis)
plot.line(y, "b--");        // 蓝色虚线 / blue dashed line

// 蓝色虚线带圆圈标记 / Blue dashed line with circle markers
plot.line(x, y, "b--o");

// 绿色点线带三角标记 / Green dotted line with triangle markers
plot.line(x, y, "g:^");

// 黑色圆圈散点 / Black circle scatter
plot.scatter(x, y, "ko");

// 十六进制颜色 / Hexadecimal color
plot.line(x, y, "#FF5733-s");
```

#### 2. PlotStyle 对象 / PlotStyle Objects

```java
// 创建自定义样式 / Create custom style
PlotStyle style = new PlotStyle()
    .color("#FF6B6B")
    .lineStyle("dashed")
    .lineWidth(3.0)
    .marker("s")
    .markerSize(8.0)
    .alpha(0.8)
    .label("我的数据");

// 应用样式 / Apply style
plot.line(x, y, style);
plot.scatter(x, y, style);

// 工厂方法 / Factory methods
PlotStyle lineStyle = PlotStyle.line("#E74C3C", "solid", 2.5);
PlotStyle scatterStyle = PlotStyle.scatter("#3498DB", "o", 6.0);
```

#### 3. 调色板支持 / Palette Support

```java
// 设置调色板 / Set palette
plot.setPalette("matplotlib");

// 使用 C0-C9 颜色 / Use C0-C9 colors
plot.line(x, y, "C0-");  // matplotlib 第0个颜色 / matplotlib 0th color
plot.line(x, y, "C1--"); // matplotlib 第1个颜色 / matplotlib 1st color

// 切换到其他调色板 / Switch to other palettes
plot.setPalette("seaborn");
plot.setPalette("echarts");
plot.setPalette("colorblind");
```

#### 4. Seaborn 风格分组显示 / Seaborn-style Grouping Display

```java
// 按颜色分组（自动分配颜色）/ Group by color (automatically assign colors)
List<String> categories = Arrays.asList("A", "B", "A", "C", "B");
plot.line(x, y, categories);        // 每个类别自动分配不同颜色 / Each category automatically assigned different colors
plot.scatter(x, y, categories);     // 散点图也支持相同分组 / Scatter plot also supports same grouping

// 多维分组（颜色 + 线条样式）/ Multi-dimensional grouping (color + line style)
List<String> hue = Arrays.asList("Group1", "Group1", "Group2", "Group2");
List<String> lineStyle = Arrays.asList("solid", "dashed", "solid", "dashed");
plot.line(x, y, hue, lineStyle);   // 同时按颜色和线型分组 / Group by color and line style simultaneously
```

### 样式表达式语法 / Style Expression Syntax

#### 颜色规范 / Color Specifications

| 符号 / Symbol | 颜色 / Color | 示例 / Example |
|---------------|--------------|----------------|
| `r` | 红色 / Red | `"r-"` |
| `g` | 绿色 / Green | `"g:"` |
| `b` | 蓝色 / Blue | `"b--"` |
| `k` | 黑色 / Black | `"ko"` |
| `w` | 白色 / White | `"w^"` |
| `y` | 黄色 / Yellow | `"y-"` |
| `c` | 青色 / Cyan | `"c--"` |
| `m` | 品红 / Magenta | `"ms"` |
| `C0-C9` | matplotlib颜色 / matplotlib colors | `"C0-"`, `"C1:"` |
| `#RRGGBB` | 十六进制 / Hexadecimal | `"#FF5733-"` |

#### 线条样式 / Line Styles

| 符号 / Symbol | 样式 / Style | 描述 / Description |
|---------------|--------------|-------------------|
| `-` | solid | 实线 / Solid line |
| `--` | dashed | 虚线 / Dashed line |
| `:` | dotted | 点线 / Dotted line |
| `-.` | dashdot | 点划线 / Dash-dot line |

#### 标记样式 / Marker Styles

| 符号 / Symbol | 标记 / Marker | 描述 / Description |
|---------------|---------------|-------------------|
| `o` | circle | 圆形 / Circle |
| `s` | square | 方形 / Square |
| `^` | triangle | 三角形 / Triangle |
| `v` | triangle | 下三角 / Downward triangle |
| `<` | triangle | 左三角 / Left triangle |
| `>` | triangle | 右三角 / Right triangle |
| `d` | diamond | 菱形 / Diamond |
| `*` | star | 星形 / Star |
| `+` | plus | 加号 / Plus |
| `x` | cross | 叉号 / Cross |
| `.` | point | 小点 / Point |

### 调色板系统 / Palette System

[ColorPalette](../src/main/java/com/yishape/lab/math/plot/ColorPalette.java) 提供丰富的颜色管理功能，支持多种调色板、颜色操作和无障碍访问优化。

#### 内置调色板 / Built-in Palettes

| 名称 / Name | 描述 / Description | 颜色数量 / Color Count | 适用场景 / Use Cases |
|-------------|-------------------|----------------------|-------------------|
| `matplotlib` | matplotlib 默认调色板，科学计算标准 | 10 | 科学计算、数据分析 |
| `echarts` | ECharts 默认调色板，现代Web风格 | 10 | Web应用、现代界面 |
| `seaborn` | Seaborn deep 调色板，统计可视化 | 10 | 统计图表、数据探索 |
| `muted` | Seaborn muted 调色板，柔和色调 | 10 | 学术报告、专业展示 |
| `pastel` | Seaborn pastel 调色板，柔和淡雅 | 10 | 清新设计、温和展示 |
| `colorblind` | 色盲友好调色板，无障碍访问 | 9 | 无障碍设计、通用展示 |
| `blues` | 蓝色单色调色板，专业商务 | 9 | 商务报告、专业展示 |
| `reds` | 红色单色调色板，警示醒目 | 9 | 警告数据、重要指标 |
| `greens` | 绿色单色调色板，自然环保 | 9 | 环保数据、健康指标 |

#### 调色板使用示例 / Palette Usage Examples

```java
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.ColorPalette;
import com.yishape.lab.math.plot.PlotStyle;
import java.util.Arrays;

// 1. 基础调色板使用
IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
IVector<Double> y1 = Linalg.vector(new double[]{10, 20, 15, 30, 25});
IVector<Double> y2 = Linalg.vector(new double[]{5, 15, 10, 25, 20});

Plots.of(800, 600)
    .setPalette("matplotlib")
    .line(x, y1, "C0-")
    .line(x, y2, "C1--")
    .title("matplotlib 调色板")
    .show();

String color = ColorPalette.getColor("echarts", 2);
String[] gradient = ColorPalette.generateAdvancedGradient(
        "#FF6B6B", "#4ECDC4", 5, "ease-in-out");
Plots.of(400, 300)
    .line(x, y1, new PlotStyle().color(gradient[2]).marker("o"))
    .title("渐变取色示例")
    .show();
```

### 主题系统 / Theme System

二维：`Plots.of(..., "dark")`、`IPlot.theme(String)`、`IPlot.applyTheme(String)`，`registerTheme` / `createGradientTheme` / `createMonochromeTheme` 的参数类型为 **`EchartsThemeManager.CustomTheme`**。

内置/ECharts 侧主题常量见源码 [`EchartsThemeManager`](../src/main/java/com/yishape/lab/math/plot/echarts/EchartsThemeManager.java)；JavaFX 侧见 [`JavaFxThemeManager`](../src/main/java/com/yishape/lab/math/plot/javafx/JavaFxThemeManager.java)。

以下表格为常用 **`theme(String)` 名称**，具体效果以后端实现为准。

#### 内置主题列表 / Built-in Themes

| 主题名称 / Theme Name | 风格特色 / Style Features | 适用场景 / Use Cases | 背景色 / Background | 主色调 / Primary Colors |
|---------------------|-------------------------|-------------------|-------------------|----------------------|
| **default** | 经典默认风格，平衡美观与可读性 | 通用场景，适合大多数图表 | 白色 (#ffffff) | ECharts经典蓝绿色系 |
| **light** | 明亮清新，适合日间使用 | 报告、演示、日常分析 | 纯白 (#ffffff) | 柔和蓝绿色调 |
| **dark** | 深色护眼，现代科技感 | 夜间使用、科技展示 | 深灰 (#1e1e1e) | 明亮蓝绿色系 |
| **blue** | 专业蓝色主题，商务风格 | 商务报告、金融分析 | 淡蓝 (#f0f8ff) | 蓝色渐变系列 |
| **green** | 自然绿色主题，环保清新 | 环保、健康、自然数据 | 淡绿 (#f0fdf4) | 绿色渐变系列 |
| **red** | 热情红色主题，警示醒目 | 警告、紧急、重要数据 | 淡红 (#fef2f2) | 红色渐变系列 |
| **purple** | 优雅紫色主题，高端典雅 | 高端产品、艺术展示 | 淡紫 (#faf5ff) | 紫色渐变系列 |
| **orange** | 活力橙色主题，温暖活泼 | 活力数据、运动统计 | 淡橙 (#fff7ed) | 橙色渐变系列 |
| **academic** | 学术风格，严谨专业 | 学术论文、研究报告 | 浅灰 (#fafafa) | 学术蓝绿色系 |
| **business** | 商务风格，现代专业 | 商业报告、企业展示 | 浅灰 (#f8fafc) | 商务蓝绿色系 |
| **minimal** | 极简风格，简洁清晰 | 简约设计、数据展示 | 纯白 (#ffffff) | 灰色系渐变 |
| **rainbow** | 彩虹风格，生动多彩 | 创意展示、艺术图表 | 近白 (#fefefe) | 彩虹色彩系列 |
| **vintage** | 复古风格，温暖经典 | 怀旧主题、经典展示 | 米色 (#f4f1de) | 复古暖色调 |
| **futuristic** | 未来风格，科技感强 | 科技展示、未来主题 | 深蓝黑 (#0f0f23) | 霓虹色彩系 |

#### 主题使用示例 / Theme Usage Examples

```java
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.PlotStyle;

IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
IVector<Double> sales = Linalg.vector(new double[]{100, 120, 110, 140, 130, 160, 150, 180, 170, 200});
IVector<Double> profit = Linalg.vector(new double[]{20, 25, 22, 30, 28, 35, 32, 40, 38, 45});

// 深色主题
Plots.of(800, 600, "dark").line(x, sales).title("深色主题").xlabel("月份").ylabel("销售额").show();
Plots.of(800, 600, "academic").scatter(x, profit).title("学术风格利润分析").show();

Plots.of()
        .createGradientTheme("sunset", "#FF6B6B", "#4ECDC4", "#F8F9FA")
        .theme("sunset")
        .line(x, sales)
        .title("渐变注册主题 sunset")
        .show();

Plots.of(800, 600, "vintage")
        .line(x, sales, new PlotStyle().color("#FF6B6B").marker("o").lineWidth(2.5))
        .title("vintage + PlotStyle")
        .show();

for (String t : new String[] {"light", "dark", "academic", "business", "vintage", "futuristic"}) {
        Plots.of(400, 300, t).line(x, sales).title("主题: " + t).show();
}
```




## 性能特性 / Performance Features

### 数据处理能力 / Data Processing Capabilities
- **大规模数据支持** / **Large-scale Data Support**: 支持处理大量数据点 / Supports processing large amounts of data points
- **内存优化** / **Memory Optimization**: 高效的数据结构设计 / Efficient data structure design
- **渲染性能** / **Rendering Performance**：JavaFX 与 ECharts 路径不同；三维 ECharts GL 走 WebGL

### 交互功能 / Interactive Features
- **缩放和平移** / **Zoom and Pan**: 支持图表的缩放和平移操作 / Supports zoom and pan operations on charts
- **数据点悬停** / **Data Point Hover**: 鼠标悬停显示详细信息 / Mouse hover displays detailed information
- **图例交互** / **Legend Interaction**: 点击图例显示/隐藏数据系列 / Click legend to show/hide data series
- **工具提示** / **Tooltip**: 丰富的工具提示信息 / Rich tooltip information

### 主题和样式 / Themes and Styles
- **内置主题** / **Built-in Themes**: 支持多种内置主题 / Supports multiple built-in themes
- **自定义样式** / **Custom Styles**: 支持自定义颜色、字体等样式 / Supports custom colors, fonts and other styles
- **响应式设计** / **Responsive Design**: 支持不同屏幕尺寸的适配 / Supports adaptation to different screen sizes

## 最佳实践 / Best Practices

### 1. 数据准备 / Data Preparation

```java
// 确保数据质量 / Ensure data quality
IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5});
if (data.length() == 0) {
    throw new IllegalArgumentException("数据不能为空"); // Data cannot be empty
}

// 处理缺失值 / Handle missing values
for (int i = 0; i < data.length(); i++) {
    if (Float.isNaN(data.get(i))) {
        data.set(i, 0.0f); // 用0替换NaN值 / Replace NaN values with 0
    }
}
```

### 2. 图表选择 / Chart Selection

```java
// 根据数据类型选择合适的图表 / Choose appropriate chart based on data type
if (isTimeSeriesData(data)) {
    plot.line(x, y); // 时间序列数据使用线图 / Use line chart for time series data
} else if (isCategoricalData(data)) {
    plot.bar(data); // 分类数据使用柱状图 / Use bar chart for categorical data
} else if (isDistributionData(data)) {
    plot.hist(data, true); // 分布数据使用直方图 / Use histogram for distribution data
}
```

### 3. 性能优化 / Performance Optimization

```java
// 对于大数据集，考虑数据采样 / For large datasets, consider data sampling
if (data.length() > 10000) {
    data = data.sample(1000); // 随机采样1000个数据点 / Randomly sample 1000 data points
}

// 使用合适的数据结构 / Use appropriate data structures
IVector<Double> optimizedData = data.copy(); // 避免不必要的数据复制 / Avoid unnecessary data copying
```

### 4. 错误处理 / Error Handling

```java
try {
    IPlot plot = Plots.of(800, 600);
    plot.line(x, y);
    plot.saveAsHtml("chart.html");
} catch (Exception e) {
    System.err.println("创建图表时出错: " + e.getMessage()); // Error creating chart
    // 处理错误情况 / Handle error cases
}
```

## 注意事项 / Notes

1. **ECharts依赖** / **ECharts Dependency**: 确保项目中包含ECharts-Java依赖 / Ensure the project includes ECharts-Java dependency
2. **浏览器兼容性** / **Browser Compatibility**: 生成的HTML文件需要现代浏览器支持 / Generated HTML files require modern browser support
3. **数据格式** / **Data Format**: 确保输入数据格式正确，避免空值或无效值 / Ensure input data format is correct, avoid null or invalid values
4. **内存使用** / **Memory Usage**: 对于大数据集，注意内存使用情况 / For large datasets, pay attention to memory usage
5. **文件权限** / **File Permissions**: 确保有足够的文件写入权限 / Ensure sufficient file write permissions

## 扩展性 / Extensibility

可视化包设计支持扩展： / The visualization package is designed to support extensions:

- **新图表类型** / **New Chart Types**: 可以轻松添加新的图表类型 / New chart types can be easily added
- **自定义主题** / **Custom Themes**: 支持自定义主题和样式 / Supports custom themes and styles
- **插件系统** / **Plugin System**: 支持插件扩展功能 / Supports plugin extension functionality
- **导出格式** / **Export Formats**: 支持更多导出格式（PNG、SVG等） / Supports more export formats (PNG, SVG, etc.)


## 应用场景 / Application Scenarios

### 数据分析 / Data Analysis
- 探索性数据分析 / Exploratory data analysis
- 统计图表生成 / Statistical chart generation
- 数据分布可视化 / Data distribution visualization
- 小提琴图分析 / Violin plot analysis
- 箱线图统计 / Box plot statistics

### 科学计算 / Scientific Computing
- 函数图像绘制 / Function plotting
- 实验结果可视化 / Experimental result visualization
- 数值计算展示 / Numerical computation display

### 商业智能 / Business Intelligence
- 仪表板创建 / Dashboard creation
- 业务指标监控 / Business metrics monitoring
- 报告图表生成 / Report chart generation

### 机器学习 / Machine Learning
- 模型性能可视化 / Model performance visualization
- 特征重要性展示 / Feature importance display
- 训练过程监控 / Training process monitoring
- 数据分布分析 / Data distribution analysis
- 模型对比评估 / Model comparison evaluation

---

**数据可视化** - 让数据说话，让洞察更清晰！ / **Data Visualization** - Let data speak, make insights clearer!

