# 数据可视化使用示例 (Data Visualization Examples)

## 概述 / Overview

本文档按照从简单到复杂的顺序，系统性地编排了数据可视化包的详细使用示例。每个级别都包含相应的理论背景、实践示例和进阶指导。

This document systematically organizes detailed usage examples for the data visualization package in order from simple to complex. Each level includes corresponding theoretical background, practical examples, and advanced guidance.

## 文档目录 / Table of Contents

| 章节 | 内容 |
|------|------|
| 第一〜三部分 | 入门、基础、中级 |
| 第四〜五部分 | 极坐标/H 图小提琴等；漏斗、桑基、关系图等专业图 |
| 第六〜七部分 | 样式系统、高级应用场景 |
| **第八部分** | **`IPlot` 扩展二维** |
| **第九部分** | **`I3dPlot` 三维与双后端** |
| 总结与学习路径 | 回顾 |

> 附录性的 **扩展二维 / 三维** 小节曾误标为第四、五部分，已改名为第八、第九部分以免与正文重复。



---

## 第一部分：入门基础 (Part 1 - Getting Started)

### 1.1 环境准备和基本概念 / Environment Setup and Basic Concepts

#### 创建第一个图表 / Create Your First Chart

```java
public class FirstChartExample {
    public static void main(String[] args) {
        // 生成简单的数据
        double[] data = {1, 2, 3, 4, 5};
        IVector<Double> y = Linalg.vector(data);
        
        // 使用流式API创建图表
        Plots.of(800, 600)
            .line(y)
            .title("我的第一个图表")
            .show();
    }
}
```

### 1.2 单向量线图 / Single Vector Line Chart

```java
public class SingleVectorLineExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25, 35, 40, 45, 50, 55});
        
        // 使用流式API创建线图
        Plots.of(800, 600)
            .line(y)
            .title("销售趋势图")
            .xlabel("时间（月）")
            .ylabel("销售额（万元）")
            .show();
    }
}
```

### 1.3 基础散点图 / Basic Scatter Chart

```java
public class BasicScatterExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25, 35, 40, 45, 50, 55});
        
        // 使用流式API创建散点图
        Plots.of(800, 600)
            .scatter(x, y)
            .title("身高体重关系图")
            .xlabel("身高（cm）")
            .ylabel("体重（kg）")
            .show();
    }
}
```

### 1.4 样式系统示例 / Style System Examples

#### 基础样式表达式 / Basic Style Expressions

```java
public class StyleExpressionExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        // 使用样式字符串创建图表
        Plots.of(800, 600)
            .line(x, y, "r-")  // 红色实线
            .title("样式表达式示例")
            .show();
        
        // 散点图样式
        Plots.of(800, 600)
            .scatter(x, y, "ko")  // 黑色圆圈
            .title("散点图样式示例")
            .show();
    }
}
```

#### PlotStyle 对象示例 / PlotStyle Object Examples

```java
public class PlotStyleExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        // 创建自定义样式
        PlotStyle style = new PlotStyle()
            .color("#FF6B6B")
            .lineStyle("dashed")
            .lineWidth(3.0)
            .marker("s")
            .markerSize(8.0)
            .alpha(0.8)
            .label("我的数据");
        
        // 应用样式
        Plots.of(800, 600)
            .line(x, y, style)
            .title("PlotStyle 对象示例")
            .show();
    }
}
```

#### 调色板示例 / Palette Examples

```java
public class PaletteExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y1 = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        IVector<Double> y2 = Linalg.vector(new double[]{15, 25, 20, 35, 30});
        
        // 设置调色板
        IPlot plot = Plots.of(800, 600);
        plot.setPalette("matplotlib");
        
        // 使用 C0-C9 颜色
        plot.line(x, y1, "C0-");  // matplotlib 第0个颜色
        plot.line(x, y2, "C1--"); // matplotlib 第1个颜色
        
        plot.title("调色板示例")
            .show();
    }
}
```

#### 分组显示示例 / Grouping Display Examples

```java
public class GroupingExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
        List<String> categories = Arrays.asList("A", "A", "A", "A", "A", "B", "B", "B", "B", "B");
        
        // 按颜色分组（自动分配颜色）
        Plots.of(800, 600)
            .line(x, y, categories)
            .title("分组显示示例")
            .show();
        
        // 多维分组（颜色 + 线条样式）
        List<String> hue = Arrays.asList("Group1", "Group1", "Group2", "Group2", "Group1", "Group1", "Group2", "Group2", "Group1", "Group1");
        List<String> lineStyle = Arrays.asList("solid", "solid", "dashed", "dashed", "solid", "solid", "dashed", "dashed", "solid", "solid");
        
        Plots.of(800, 600)
            .line(x, y, hue, lineStyle)
            .title("多维分组示例")
            .show();
    }
}
```

#### 样式系统示例 / Style System Examples

```java
public class StyleSystemExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        // 高级样式对象创建
        PlotStyle advancedStyle = new PlotStyle()
            .color("#3498DB")
            .lineStyle("solid")
            .lineWidth(2.5)
            .marker("o")
            .markerSize(6.0)
            .opacity(0.9)
            .emphasis(new PlotStyle()  // hover状态
                .color("#E74C3C")
                .lineWidth(4.0)
                .markerSize(8.0))
            .blur(new PlotStyle()      // 失焦状态
                .opacity(0.3))
            .select(new PlotStyle()    // 选中状态
                .color("#F39C12")
                .lineWidth(3.5));
        
        // 应用高级样式
        Plots.of(800, 600)
            .line(x, y, advancedStyle)
            .title("样式系统示例", "支持交互状态和高级颜色操作")
            .xlabel("X轴")
            .ylabel("Y轴")
            .show();
    }
}
```

#### 智能颜色操作示例 / Intelligent Color Operations Example

```java
public class IntelligentColorExample {
    public static void main(String[] args) {
        // 生成多系列数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        List<IVector<Double>> dataSeries = Arrays.asList(
            Linalg.vector(new double[]{10, 20, 15, 30, 25}),
            Linalg.vector(new double[]{15, 25, 20, 35, 30}),
            Linalg.vector(new double[]{8, 18, 12, 28, 22}),
            Linalg.vector(new double[]{12, 22, 17, 32, 27})
        );
        
        // 基于HSL的智能颜色生成
        String baseColor = "#3498DB";
        List<String> colorSeries = new ArrayList<>();
        
        for (int i = 0; i < dataSeries.size(); i++) {
            // 基于亮度调整生成色彩系列
            String adjustedColor = StyleConverter.adjustBrightness(
                baseColor, -0.1 + (i * 0.2)
            );
            colorSeries.add(adjustedColor);
        }
        
        // 创建多系列图表
        IPlot plot = Plots.of(800, 600);
        for (int i = 0; i < dataSeries.size(); i++) {
            PlotStyle seriesStyle = new PlotStyle()
                .color(colorSeries.get(i))
                .lineWidth(2.0)
                .marker("o")
                .markerSize(5.0)
                .label("系列" + (i + 1));
                
            plot.line(x, dataSeries.get(i), seriesStyle);
        }
        
        plot.title("颜色操作示例", "HSL色彩空间自动调整")
            .xlabel("时间")
            .ylabel("数值")
            .show();
    }
}
```

#### 主题应用示例 / Theme Application Example

```java
public class ThemeApplicationExample {
    public static void main(String[] args) {
        // 生成金融数据示例
        double[][] ohlcArray = {
            {100, 110, 95, 115},   // [open, close, low, high]
            {110, 120, 105, 125},
            {120, 115, 110, 130},
            {115, 125, 110, 135},
            {125, 130, 120, 140}
        };
        IMatrix<Double> ohlcData = Linalg.matrix(ohlcArray);
        List<String> dates = Arrays.asList("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05");
        
        // 主题推荐
        String recommendedTheme = ThemeManager.recommendTheme(
            "financial",    // 数据类型
            "candlestick",  // 图表类型
            "professional"  // 用户偏好
        );
        
        // 创建主题化K线图
        Plots.of(1000, 600, recommendedTheme)
            .candlestick(ohlcData, dates)
            .title("主题K线图", "推荐专业金融主题")
            .xlabel("日期")
            .ylabel("价格（元）")
            .show();
        
        // 自定义样式与主题融合
        PlotStyle customStyle = new PlotStyle()
            .color("#FF6B6B")
            .lineWidth(2.0);
            
        PlotStyle themedStyle = ThemeManager.applyThemeToStyle(
            customStyle, recommendedTheme
        );
        
        // 应用融合后的样式
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        Plots.of(800, 600, recommendedTheme)
            .line(x, y, themedStyle)
            .title("主题样式融合示例")
            .show();
    }
}
```

---

## 第二部分：基础应用 (Part 2 - Basic Applications)

### 2.1 双向量线图 / Two Vector Line Chart

```java
public class TwoVectorLineExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25, 35, 40, 45, 50, 55});
        
        // 使用流式API创建双向量线图
        Plots.of(800, 600)
            .line(x, y)
            .title("销售趋势图", "2024年各月销售数据")
            .xlabel("月份")
            .ylabel("销售额（万元）")
            .show();
    }
}
```

### 2.2 多组散点图 / Multi-group Scatter Chart

```java
public class MultiGroupScatterExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
        List<String> hue = Arrays.asList("产品A", "产品A", "产品A", "产品A", "产品A",
                                        "产品B", "产品B", "产品B", "产品B", "产品B");
        
        // 使用流式API创建多组散点图
        Plots.of(800, 600)
            .scatter(x, y, hue)
            .title("产品对比分析", "2024年各月产品销售对比")
            .xlabel("月份")
            .ylabel("销售额（万元）")
            .show();
    }
}
```

### 2.3 饼图 / Pie Chart

```java
public class PieChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{30, 25, 20, 15, 10});
        
        // 使用流式API创建饼图
        Plots.of(600, 600)
            .pie(data)
            .title("市场份额分布", "2024年各产品线市场份额")
            .show();
    }
}
```

### 2.4 柱状图 / Bar Chart

```java
public class BarChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        // 使用流式API创建柱状图
        Plots.of(800, 600)
            .bar(data)
            .title("销售业绩对比", "2024年各季度销售数据")
            .xlabel("季度")
            .ylabel("销售额（万元）")
            .show();
    }
}
```

---

## 第三部分：中级应用 (Part 3 - Intermediate Applications)

### 3.1 分组柱状图 / Grouped Bar Chart

```java
public class GroupedBarChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        List<String> hue = Arrays.asList("组A", "组B", "组A", "组B", "组A");
        
        // 使用流式API创建分组柱状图
        Plots.of(800, 600)
            .bar(data, hue)
            .title("分组柱状图", "对比不同组别的数据")
            .xlabel("类别")
            .ylabel("数值")
            .show();
    }
}
```

### 3.2 多组线图 / Multi-group Line Chart

```java
public class MultiGroupLineChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
        List<String> hue = Arrays.asList("产品A", "产品A", "产品A", "产品A", "产品A",
                                        "产品B", "产品B", "产品B", "产品B", "产品B");
        
        // 使用流式API创建多组线图
        Plots.of(800, 600)
            .line(x, y, hue)
            .title("产品对比分析", "2024年各月产品销售对比")
            .xlabel("月份")
            .ylabel("销售额（万元）")
            .show();
    }
}
```

### 3.3 直方图 / Histogram

```java
public class HistogramExample {
    public static void main(String[] args) {
        // 生成示例数据
        double[] histData = new double[100];
        for (int i = 0; i < 100; i++) {
            histData[i] = (double) (Math.random() * 10 + 5); // 均值5，标准差约2.9
        }
        IVector<Double> histVector = IVector.of(histData);
        
        // 使用流式API创建直方图（带拟合线）
        Plots.of(800, 600)
            .hist(histVector, true)
            .title("数据分布直方图", "样本数据的正态分布拟合")
            .xlabel("数值区间")
            .ylabel("频次")
            .show();
    }
}
```

### 3.4 箱线图 / Box Plot

```java
public class BoxPlotExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        List<String> labels = Arrays.asList("数据集");
        
        // 使用流式API创建箱线图
        Plots.of(800, 600)
            .boxplot(data, labels)
            .title("数据分布箱线图", "各指标的数据分布情况")
            .xlabel("指标")
            .ylabel("数值")
            .show();
    }
}
```

### 3.5 小提琴图 / Violin Plot

```java
public class ViolinPlotExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        
        // 使用流式API创建小提琴图
        Plots.of(800, 600)
            .violinplot(data)
            .title("数据分布小提琴图", "展示数据的分布形状和统计特征")
            .xlabel("数值")
            .ylabel("密度")
            .show();
    }
}
```

### 3.6 多组小提琴图 / Multi-group Violin Plot

```java
public class MultiGroupViolinPlotExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                                             2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        List<String> labels = Arrays.asList("组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A", "组A",
                                           "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B", "组B");
        
        // 使用流式API创建多组小提琴图
        Plots.of(800, 600)
            .violinplot(data, labels)
            .title("多组数据分布对比", "不同组别的数据分布对比分析")
            .xlabel("组别")
            .ylabel("密度")
            .show();
    }
}
```

---

## 第四部分：高级应用 (Part 4 - Advanced Applications)

### 4.1 极坐标柱状图 / Polar Bar Chart

```java
public class PolarBarChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E");
        
        // 使用流式API创建极坐标柱状图
        Plots.of(600, 600)
            .polarBar(data, categories)
            .title("极坐标柱状图")
            .show();
    }
}
```

### 4.2 极坐标线图 / Polar Line Chart

```java
public class PolarLineChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E");
        
        // 使用流式API创建极坐标线图
        Plots.of(600, 600)
            .polarLine(data, categories)
            .title("极坐标线图")
            .show();
    }
}
```

### 4.3 极坐标散点图 / Polar Scatter Chart

```java
public class PolarScatterChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E");
        
        // 使用流式API创建极坐标散点图
        Plots.of(600, 600)
            .polarScatter(data, categories)
            .title("极坐标散点图")
            .show();
    }
}
```

### 4.4 热力图 / Heatmap

```java
public class HeatmapExample {
    public static void main(String[] args) {
        // 创建二维数据矩阵
        double[][] heatmapArray = {
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6},
            {4, 5, 6, 7}
        };
        IMatrix<Double> data = Linalg.matrix(heatmapArray);
        List<String> xLabels = Arrays.asList("X1", "X2", "X3", "X4");
        List<String> yLabels = Arrays.asList("Y1", "Y2", "Y3", "Y4");
        
        // 使用流式API创建热力图
        Plots.of(800, 600)
            .heatmap(data, xLabels, yLabels)
            .title("相关性热力图", "各指标间的相关性分析")
            .show();
    }
}
```

### 4.5 雷达图 / Radar Chart

```java
public class RadarChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{80, 90, 70, 85, 95, 75});
        List<String> indicators = Arrays.asList("指标1", "指标2", "指标3", "指标4", "指标5", "指标6");
        
        // 使用流式API创建雷达图
        Plots.of(600, 600)
            .radar(data, indicators)
            .title("能力雷达图", "各项技能能力评估")
            .show();
    }
}
```

### 4.6 小提琴图流式API / Violin Plot Fluent API

```java
public class ViolinPlotFluentAPIExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        
        // 使用流式API创建小提琴图
        Plots.of(800, 600)
            .violinplot(data)
            .title("数据分布分析", "小提琴图展示数据分布特征")
            .xlabel("数值")
            .ylabel("密度")
            .show();
    }
}
```

### 4.7 K线图 / Candlestick Chart

```java
public class CandlestickChartExample {
    public static void main(String[] args) {
        // 生成示例数据（开盘价, 收盘价, 最低价, 最高价）
        double[][] candlestickArray = {
            {100, 110, 95, 115},
            {110, 120, 105, 125},
            {120, 115, 110, 130},
            {115, 125, 110, 135},
            {125, 130, 120, 140}
        };
        IMatrix<Double> candlestickData = Linalg.matrix(candlestickArray);
        List<String> dates = Arrays.asList("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05");
        
        // 使用流式API创建K线图
        Plots.of(1000, 600)
            .candlestick(candlestickData, dates)
            .title("股票价格K线图", "2024年1月股价走势")
            .xlabel("日期")
            .ylabel("价格（元）")
            .show();
    }
}
```

---

## 第五部分：专业应用 (Part 5 - Professional Applications)

### 5.1 漏斗图 / Funnel Chart

```java
public class FunnelChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> data = Linalg.vector(new double[]{100, 80, 60, 40, 20});
        List<String> labels = Arrays.asList("访问", "注册", "购买", "支付", "完成");
        
        // 使用流式API创建漏斗图
        Plots.of(800, 600)
            .funnel(data, labels)
            .title("用户转化漏斗", "从访问到购买的转化流程")
            .show();
    }
}
```

### 5.2 桑基图 / Sankey Chart

```java
public class SankeyChartExample {
    public static void main(String[] args) {
        // 创建节点数据
        List<Map<String, Object>> nodes = new ArrayList<>();
        nodes.add(createNode("source1", "源1"));
        nodes.add(createNode("source2", "源2"));
        nodes.add(createNode("target1", "目标1"));
        nodes.add(createNode("target2", "目标2"));
        
        // 创建连接数据
        List<Map<String, Object>> links = new ArrayList<>();
        links.add(createLink("source1", "target1", 10));
        links.add(createLink("source1", "target2", 20));
        links.add(createLink("source2", "target1", 15));
        links.add(createLink("source2", "target2", 25));
        
        // 使用流式API创建桑基图
        Plots.of(1000, 600)
            .sankey(nodes, links)
            .title("数据流向图", "各系统间的数据流转情况")
            .show();
    }
    
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
}
```

### 5.3 旭日图 / Sunburst Chart

```java
public class SunburstChartExample {
    public static void main(String[] args) {
        // 创建层次数据
        List<Map<String, Object>> sunburstData = new ArrayList<>();
        sunburstData.add(createSunburstNode("root", "根节点", 100));
        sunburstData.add(createSunburstNode("child1", "子节点1", 60, "root"));
        sunburstData.add(createSunburstNode("child2", "子节点2", 40, "root"));
        sunburstData.add(createSunburstNode("grandchild1", "孙节点1", 30, "child1"));
        sunburstData.add(createSunburstNode("grandchild2", "孙节点2", 30, "child1"));
        
        // 使用流式API创建旭日图
        Plots.of(800, 800)
            .sunburst(sunburstData)
            .title("组织架构图", "公司各部门人员分布")
            .show();
    }
    
    private static Map<String, Object> createSunburstNode(String id, String name, int value) {
        Map<String, Object> node = new HashMap<>();
        node.put("id", id);
        node.put("name", name);
        node.put("value", value);
        return node;
    }
    
    private static Map<String, Object> createSunburstNode(String id, String name, int value, String parent) {
        Map<String, Object> node = createSunburstNode(id, name, value);
        node.put("parent", parent);
        return node;
    }
}
```

### 5.4 主题河流图 / Theme River Chart

```java
public class ThemeRiverChartExample {
    public static void main(String[] args) {
        // 创建时间序列数据
        List<Map<String, Object>> themeRiverData = new ArrayList<>();
        themeRiverData.add(createThemeRiverNode("2024-01-01", "类别A", 10));
        themeRiverData.add(createThemeRiverNode("2024-01-01", "类别B", 20));
        themeRiverData.add(createThemeRiverNode("2024-01-02", "类别A", 15));
        themeRiverData.add(createThemeRiverNode("2024-01-02", "类别B", 25));
        
        List<String> categories = Arrays.asList("类别A", "类别B");
        
        // 使用流式API创建主题河流图
        Plots.of(1200, 600)
            .themeRiver(themeRiverData, categories)
            .title("新闻热度趋势", "各主题新闻的热度变化")
            .show();
    }
    
    private static Map<String, Object> createThemeRiverNode(String time, String category, int value) {
        Map<String, Object> node = new HashMap<>();
        node.put("time", time);
        node.put("category", category);
        node.put("value", value);
        return node;
    }
}
```

### 5.5 关系图 / Graph Chart

```java
public class GraphChartExample {
    public static void main(String[] args) {
        // 创建节点数据
        List<Map<String, Object>> nodes = new ArrayList<>();
        nodes.add(createGraphNode("node1", "节点1"));
        nodes.add(createGraphNode("node2", "节点2"));
        nodes.add(createGraphNode("node3", "节点3"));
        
        // 创建连接数据
        List<Map<String, Object>> links = new ArrayList<>();
        links.add(createGraphLink("node1", "node2", 10));
        links.add(createGraphLink("node2", "node3", 15));
        links.add(createGraphLink("node1", "node3", 20));
        
        // 使用流式API创建关系图
        Plots.of(1000, 800)
            .graph(nodes, links)
            .title("社交网络图", "用户关系网络分析")
            .show();
    }
    
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
}
```

### 5.6 平行坐标图 / Parallel Coordinates Chart

```java
public class ParallelCoordinatesChartExample {
    public static void main(String[] args) {
        // 创建数据矩阵
        double[][] dataArray = {
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6},
            {4, 5, 6, 7}
        };
        IMatrix<Double> data = Linalg.matrix(dataArray);
        List<String> dimensions = Arrays.asList("维度1", "维度2", "维度3", "维度4");
        
        // 使用流式API创建平行坐标图
        Plots.of(1200, 600)
            .parallel(data, dimensions)
            .title("多维数据分布", "各维度数据的分布情况")
            .show();
    }
}
```

### 5.7 仪表盘 / Gauge Chart

```java
public class GaugeChartExample {
    public static void main(String[] args) {
        // 设置参数
        double value = 75.5f;
        double max = 100.0f;
        double min = 0.0f;
        
        // 使用流式API创建仪表盘
        Plots.of(400, 400)
            .gauge(value, max, min)
            .title("系统性能监控", "CPU使用率实时监控")
            .show();
    }
}
```

### 5.8 综合小提琴图分析 / Comprehensive Violin Plot Analysis

```java
public class ComprehensiveViolinAnalysisExample {
    public static void main(String[] args) {
        // 生成多组对比数据
        double[] groupA = new double[50];
        double[] groupB = new double[50];
        double[] groupC = new double[50];
        
        // 生成不同分布的数据
        for (int i = 0; i < 50; i++) {
            groupA[i] = (double) (Math.random() * 10 + 5); // 正态分布
            groupB[i] = (double) (Math.random() * 15 + 10); // 右偏分布
            groupC[i] = (double) (Math.random() * 8 + 8); // 左偏分布
        }
        
        // 合并数据
        double[] allData = new double[150];
        String[] allLabels = new String[150];
        
        System.arraycopy(groupA, 0, allData, 0, 50);
        System.arraycopy(groupB, 0, allData, 50, 50);
        System.arraycopy(groupC, 0, allData, 100, 50);
        
        Arrays.fill(allLabels, 0, 50, "组A");
        Arrays.fill(allLabels, 50, 100, "组B");
        Arrays.fill(allLabels, 100, 150, "组C");
        
        IVector<Double> data = IVector.of(allData);
        List<String> labels = Arrays.asList(allLabels);
        
        // 使用流式API创建多组小提琴图
        Plots.of(1000, 600)
            .violinplot(data, labels)
            .title("多组数据分布对比分析", "不同组别的数据分布特征对比")
            .xlabel("组别")
            .ylabel("数值密度")
            .show();
    }
}
```

### 5.9 树图示例 / Tree Chart Example

```java
public class TreeChartExample {
    public static void main(String[] args) {
        // 创建树形数据
        List<Map<String, Object>> treeData = new ArrayList<>();
        treeData.add(createTreeNode("root", "根节点", 100));
        treeData.add(createTreeNode("child1", "子节点1", 60, "root"));
        treeData.add(createTreeNode("child2", "子节点2", 40, "root"));
        treeData.add(createTreeNode("grandchild1", "孙节点1", 30, "child1"));
        treeData.add(createTreeNode("grandchild2", "孙节点2", 30, "child1"));
        treeData.add(createTreeNode("grandchild3", "孙节点3", 20, "child2"));
        treeData.add(createTreeNode("grandchild4", "孙节点4", 20, "child2"));
        
        // 使用流式API创建树图
        Plots.of(1000, 800)
            .tree(treeData)
            .title("组织架构图", "公司各部门人员分布")
            .show();
    }
    
    private static Map<String, Object> createTreeNode(String id, String name, int value) {
        Map<String, Object> node = new HashMap<>();
        node.put("id", id);
        node.put("name", name);
        node.put("value", value);
        return node;
    }
    
    private static Map<String, Object> createTreeNode(String id, String name, int value, String parent) {
        Map<String, Object> node = createTreeNode(id, name, value);
        node.put("parent", parent);
        return node;
    }
}
```

### 5.10 矩形树图示例 / Treemap Chart Example

```java
public class TreemapChartExample {
    public static void main(String[] args) {
        // 创建层次数据
        List<Map<String, Object>> treemapData = new ArrayList<>();
        treemapData.add(createTreemapNode("root", "总销售额", 1000));
        treemapData.add(createTreemapNode("region1", "华北区", 400, "root"));
        treemapData.add(createTreemapNode("region2", "华东区", 350, "root"));
        treemapData.add(createTreemapNode("region3", "华南区", 250, "root"));
        treemapData.add(createTreemapNode("product1", "产品A", 200, "region1"));
        treemapData.add(createTreemapNode("product2", "产品B", 200, "region1"));
        treemapData.add(createTreemapNode("product3", "产品C", 180, "region2"));
        treemapData.add(createTreemapNode("product4", "产品D", 170, "region2"));
        treemapData.add(createTreemapNode("product5", "产品E", 120, "region3"));
        treemapData.add(createTreemapNode("product6", "产品F", 130, "region3"));
        
        // 使用流式API创建矩形树图
        Plots.of(800, 600)
            .treemap(treemapData)
            .title("销售数据矩形树图", "各区域产品销售分布")
            .show();
    }
    
    private static Map<String, Object> createTreemapNode(String id, String name, int value) {
        Map<String, Object> node = new HashMap<>();
        node.put("id", id);
        node.put("name", name);
        node.put("value", value);
        return node;
    }
    
    private static Map<String, Object> createTreemapNode(String id, String name, int value, String parent) {
        Map<String, Object> node = createTreemapNode(id, name, value);
        node.put("parent", parent);
        return node;
    }
}
```

---

## 高级配置和自定义 / Advanced Configuration and Customization

### 自定义主题 / Custom Theme

```java
public class CustomThemeExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        // 使用流式API创建带自定义主题的图表
        Plots.of(800, 600, "dark")
            .line(x, y)
            .title("自定义主题图表", "使用深色主题")
            .xlabel("X轴")
            .ylabel("Y轴")
            .show();
    }
}
```

### 自定义坐标轴 / Custom Axis

```java
public class CustomAxisExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        // 自定义坐标轴刻度
        AxisTicks xTicks = new AxisTicks();
        xTicks.setTickValues(Linalg.vector(new double[]{1, 2, 3, 4, 5}));
        xTicks.setTickLabels(Arrays.asList("一月", "二月", "三月", "四月", "五月"));
        
        AxisTicks yTicks = new AxisTicks();
        yTicks.setTickValues(Linalg.vector(new double[]{0, 10, 20, 30, 40}));
        
        // 使用流式API创建自定义坐标轴图表
        Plots.of(800, 600)
            .line(x, y)
            .title("自定义坐标轴图表")
            .xlabel("月份")
            .ylabel("销售额（万元）")
            .setXticks(xTicks)
            .setYticks(yTicks)
            .show();
    }
}
```

---

## 错误处理和性能优化 / Error Handling and Performance Optimization

### 数据验证 / Data Validation

```java
public class DataValidationExample {
    public static void main(String[] args) {
        try {
            // 检查数据是否为空
            IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5});
            if (data.length() == 0) {
                throw new IllegalArgumentException("数据不能为空");
            }
            
            // 使用流式API绘制图表
            Plots.of(800, 600)
                .line(data)
                .show();
            
        } catch (IllegalArgumentException e) {
            System.err.println("数据验证错误: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("创建图表时出错: " + e.getMessage());
        }
    }
}
```

### 大数据集处理 / Large Dataset Processing

```java
public class LargeDatasetExample {
    public static void main(String[] args) {
        // 生成大数据集
        double[] dataArray = new double[10000];
        for (int i = 0; i < 10000; i++) {
            dataArray[i] = (double) Math.sin(i * 0.01) * 100 + (double) Math.random() * 10;
        }
        IVector<Double> data = IVector.of(dataArray);
        
        // 对于大数据集，考虑数据采样
        if (data.length() > 1000) {
            System.out.println("大数据集，建议进行数据采样");
        }
        
        // 使用流式API绘制图表
        Plots.of(1200, 800)
            .line(data)
            .title("大数据集示例", "包含10000个数据点")
            .xlabel("索引")
            .ylabel("数值")
            .show();
    }
}
```

---

---

## 第六部分：样式系统 (Part 6 - Style System)

### 6.1 StyleExpression 样式表达式示例 / Style Expression Examples

```java
public class StyleExpressionExample {
    public static void main(String[] args) {
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        // 基础样式表达式
        Plots.of(800, 600)
            .line(x, y, "r-")  // 红色实线
            .title("红色实线")
            .show();
            
        Plots.of(800, 600)
            .line(x, y, "b--o")  // 蓝色虚线带圆圈
            .title("蓝色虚线带圆圈")
            .show();
            
        Plots.of(800, 600)
            .scatter(x, y, "ko")  // 黑色圆圈散点
            .title("黑色圆圈散点")
            .show();
            
        // 十六进制颜色
        Plots.of(800, 600)
            .line(x, y, "#FF5733-s")  // 十六进制颜色方形标记
            .title("十六进制颜色方形标记")
            .show();
            
        // C0-C9 颜色（matplotlib风格）
        Plots.of(800, 600)
            .line(x, y, "C0-")  // matplotlib第0个颜色
            .title("matplotlib C0 颜色")
            .show();
    }
}
```

### 6.2 StyleConverter 高级颜色操作示例 / Style Converter Advanced Color Operations

```java
public class StyleConverterExample {
    public static void main(String[] args) {
        String baseColor = "#3498DB";
        
        // HSL色彩空间操作
        String brighter = StyleConverter.adjustBrightness(baseColor, 0.3);
        String moreSaturated = StyleConverter.adjustSaturation(baseColor, 0.2);
        String hueShifted = StyleConverter.shiftHue(baseColor, 60);
        
        // 智能渐变生成
        List<String> gradient = StyleConverter.createLinearGradient(
            "#FF6B6B", "#4ECDC4", 5
        );
        
        // 径向渐变
        String radialGradient = StyleConverter.createRadialGradient(
            "#FFD93D", "#FF6B6B", 0.5, 0.5, 0.8
        );
        
        // 颜色和谐度分析
        boolean isHarmonious = StyleConverter.isColorHarmonious("#FF6B6B", "#4ECDC4");
        System.out.println("颜色和谐度: " + isHarmonious);
        
        // 应用到多系列数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IPlot plot = Plots.of(800, 600);
        
        for (int i = 0; i < gradient.size(); i++) {
            IVector<Double> y = Linalg.vector(new double[]{
                10 + i * 5, 20 + i * 3, 15 + i * 4, 30 + i * 2, 25 + i * 6
            });
            PlotStyle style = new PlotStyle().color(gradient.get(i)).lineWidth(2.0);
            plot.line(x, y, style);
        }
        
        plot.title("HSL颜色渐变示例").show();
    }
}
```

### 6.3 SeabornStyleMapper 分组映射示例 / Seaborn Style Mapper Grouping Examples

```java
public class SeabornStyleMapperExample {
    public static void main(String[] args) {
        // 创建映射器
        SeabornStyleMapper mapper = new SeabornStyleMapper()
            .setHuePalette("viridis")           // 设置色调调色板
            .setStyleSequence(new String[]{"solid", "dashed", "dotted"})  // 设置线条样式序列
            .setMarkerSequence(new String[]{"o", "s", "^", "v"});         // 设置标记序列
        
        // 生成分组数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
        List<String> groups = Arrays.asList("GroupA", "GroupA", "GroupA", "GroupA", "GroupA",
                                           "GroupB", "GroupB", "GroupB", "GroupB", "GroupB");
        
        // 获取分组样式
        PlotStyle styleA = mapper.getStyleForGroup("GroupA", "hue");
        PlotStyle styleB = mapper.getStyleForGroup("GroupB", "hue");
        
        // 创建分组图表
        IPlot plot = Plots.of(800, 600);
        
        // 按组分别绘制
        IVector<Double> xA = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> yA = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        IVector<Double> xB = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> yB = Linalg.vector(new double[]{15, 25, 20, 35, 30});
        
        plot.line(xA, yA, styleA);
        plot.line(xB, yB, styleB);
        
        plot.title("Seaborn风格分组示例").show();
    }
}
```

### 6.3 UniversalStyleApplier 通用样式应用示例 / Universal Style Applier Examples

```java
public class UniversalStyleApplierExample {
    public static void main(String[] args) {
        // 创建通用样式
        PlotStyle universalStyle = new PlotStyle()
            .color("#3498DB")
            .lineWidth(2.5)
            .opacity(0.8)
            .emphasis(new PlotStyle().color("#E74C3C").lineWidth(4.0))
            .blur(new PlotStyle().opacity(0.3));
        
        // 应用到不同图表类型
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        // 自动适配到线图、散点图、柱状图等
        Plots.of(800, 600).line(x, y, universalStyle).title("通用样式线图").show();
        Plots.of(800, 600).scatter(x, y, universalStyle).title("通用样式散点图").show();
        Plots.of(800, 600).bar(y, universalStyle).title("通用样式柱状图").show();
    }
}
```

### 6.4 ThemeManager 主题管理示例 / Theme Manager Examples

```java
public class ThemeManagerExample {
    public static void main(String[] args) {
        // 智能主题推荐
        String recommendedTheme = ThemeManager.recommendTheme(
            "financial",    // 数据类型
            "candlestick",  // 图表类型
            "professional"  // 用户偏好
        );
        
        System.out.println("推荐主题: " + recommendedTheme);
        
        // 生成金融数据示例
        double[][] ohlcArray = {
            {100, 110, 95, 115},   // [open, close, low, high]
            {110, 120, 105, 125},
            {120, 115, 110, 130},
            {115, 125, 110, 135},
            {125, 130, 120, 140}
        };
        IMatrix<Double> ohlcData = Linalg.matrix(ohlcArray);
        List<String> dates = Arrays.asList("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05");
        
        // 创建主题化K线图
        Plots.of(1000, 600, recommendedTheme)
            .candlestick(ohlcData, dates)
            .title("主题K线图", "推荐专业金融主题")
            .xlabel("日期")
            .ylabel("价格（元）")
            .show();
        
        // 自定义样式与主题融合
        PlotStyle customStyle = new PlotStyle()
            .color("#FF6B6B")
            .lineWidth(2.0);
            
        PlotStyle themedStyle = ThemeManager.applyThemeToStyle(
            customStyle, recommendedTheme
        );
        
        // 应用融合后的样式
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        Plots.of(800, 600, recommendedTheme)
            .line(x, y, themedStyle)
            .title("主题样式融合示例")
            .show();
    }
}

---

## 第七部分：高级应用场景 (Part 7 - Advanced Application Scenarios)

### 7.1 金融数据分析 / Financial Data Analysis

```java
public class FinancialDataAnalysisExample {
    public static void main(String[] args) {
        // 生成股票价格数据
        double[][] ohlcArray = {
            {100, 110, 95, 115},   // [open, close, low, high]
            {110, 120, 105, 125},
            {120, 115, 110, 130},
            {115, 125, 110, 135},
            {125, 130, 120, 140},
            {130, 140, 125, 145},
            {140, 135, 130, 150},
            {135, 145, 130, 155}
        };
        IMatrix<Double> ohlcData = Linalg.matrix(ohlcArray);
        List<String> dates = Arrays.asList("2024-01-01", "2024-01-02", "2024-01-03", 
                                          "2024-01-04", "2024-01-05", "2024-01-08", 
                                          "2024-01-09", "2024-01-10");
        
        // 创建K线图
        Plots.of(1200, 600, "dark")
            .candlestick(ohlcData, dates)
            .title("股票价格K线图", "2024年1月股价走势分析")
            .xlabel("日期")
            .ylabel("价格（元）")
            .show();
        
        // 创建成交量柱状图
        IVector<Double> volume = Linalg.vector(new double[]{1000, 1200, 800, 1500, 1100, 1300, 900, 1400});
        Plots.of(1200, 300, "dark")
            .bar(volume)
            .title("成交量分析", "对应日期的交易量")
            .xlabel("日期")
            .ylabel("成交量（手）")
            .show();
    }
}
```

### 7.2 科学实验数据可视化 / Scientific Experiment Data Visualization

```java
public class ScientificDataVisualizationExample {
    public static void main(String[] args) {
        // 生成实验数据
        double[] time = new double[100];
        double[] temperature = new double[100];
        double[] pressure = new double[100];
        double[] concentration = new double[100];
        
        for (int i = 0; i < 100; i++) {
            time[i] = i * 0.1;
            temperature[i] = 20 + 10 * Math.sin(i * 0.1) + (Math.random() - 0.5) * 2;
            pressure[i] = 1.0 + 0.5 * Math.cos(i * 0.1) + (Math.random() - 0.5) * 0.1;
            concentration[i] = 0.5 + 0.3 * Math.sin(i * 0.2) + (Math.random() - 0.5) * 0.05;
        }
        
        IVector<Double> timeVec = Linalg.vector(time);
        IVector<Double> tempVec = Linalg.vector(temperature);
        IVector<Double> pressVec = Linalg.vector(pressure);
        IVector<Double> concVec = Linalg.vector(concentration);
        
        // 创建多子图布局
        IPlot plot = Plots.of(1200, 800);
        
        // 温度曲线
        plot.line(timeVec, tempVec, "r-")
            .title("实验数据监控", "温度、压力、浓度随时间变化")
            .xlabel("时间（小时）")
            .ylabel("温度（°C）");
        
        // 压力曲线
        plot.line(timeVec, pressVec, "b--")
            .ylabel("压力（atm）");
        
        // 浓度散点图
        plot.scatter(timeVec, concVec, "go")
            .ylabel("浓度（mol/L）");
        
        plot.show();
    }
}
```

### 7.3 商业智能仪表板 / Business Intelligence Dashboard

```java
public class BusinessIntelligenceDashboardExample {
    public static void main(String[] args) {
        // 销售数据
        IVector<Double> salesData = Linalg.vector(new double[]{120, 135, 148, 162, 175, 189, 203, 218, 234, 251, 268, 285});
        List<String> months = Arrays.asList("1月", "2月", "3月", "4月", "5月", "6月", 
                                          "7月", "8月", "9月", "10月", "11月", "12月");
        
        // 创建销售趋势图
        Plots.of(1000, 400)
            .line(salesData, "b-o")
            .title("年度销售趋势", "2024年各月销售数据")
            .xlabel("月份")
            .ylabel("销售额（万元）")
            .show();
        
        // 产品分布饼图
        IVector<Double> productData = Linalg.vector(new double[]{35, 25, 20, 15, 5});
        Plots.of(600, 400)
            .pie(productData)
            .title("产品市场份额", "各产品线销售占比")
            .show();
        
        // 区域对比柱状图
        IVector<Double> regionData = Linalg.vector(new double[]{180, 220, 195, 250, 210});
        List<String> regions = Arrays.asList("华北", "华东", "华南", "西南", "东北");
        Plots.of(800, 400)
            .bar(regionData)
            .title("区域销售对比", "各区域销售业绩")
            .xlabel("区域")
            .ylabel("销售额（万元）")
            .show();
        
        // 性能指标仪表盘
        Plots.of(400, 400)
            .gauge(85.5, 100.0, 0.0)
            .title("客户满意度", "当前满意度指标")
            .show();
    }
}
```

### 7.4 机器学习模型可视化 / Machine Learning Model Visualization

```java
public class MachineLearningVisualizationExample {
    public static void main(String[] args) {
        // 生成分类数据
        double[] x1 = new double[200];
        double[] x2 = new double[200];
        String[] labels = new String[200];
        
        for (int i = 0; i < 200; i++) {
            if (i < 100) {
                x1[i] = Math.random() * 2 + 1;
                x2[i] = Math.random() * 2 + 1;
                labels[i] = "类别A";
            } else {
                x1[i] = Math.random() * 2 + 3;
                x2[i] = Math.random() * 2 + 3;
                labels[i] = "类别B";
            }
        }
        
        IVector<Double> x1Vec = Linalg.vector(x1);
        IVector<Double> x2Vec = Linalg.vector(x2);
        List<String> labelList = Arrays.asList(labels);
        
        // 创建分类散点图
        Plots.of(800, 600)
            .scatter(x1Vec, x2Vec, labelList)
            .title("分类数据分布", "二分类数据集可视化")
            .xlabel("特征1")
            .ylabel("特征2")
            .show();
        
        // 生成模型性能数据
        double[] epochs = new double[50];
        double[] trainLoss = new double[50];
        double[] valLoss = new double[50];
        double[] accuracy = new double[50];
        
        for (int i = 0; i < 50; i++) {
            epochs[i] = i + 1;
            trainLoss[i] = 1.0 * Math.exp(-i * 0.1) + 0.1 + Math.random() * 0.05;
            valLoss[i] = 1.2 * Math.exp(-i * 0.08) + 0.15 + Math.random() * 0.05;
            accuracy[i] = 1.0 - 0.8 * Math.exp(-i * 0.1) + Math.random() * 0.02;
        }
        
        IVector<Double> epochsVec = Linalg.vector(epochs);
        IVector<Double> trainLossVec = Linalg.vector(trainLoss);
        IVector<Double> valLossVec = Linalg.vector(valLoss);
        IVector<Double> accuracyVec = Linalg.vector(accuracy);
        
        // 创建训练过程图
        IPlot plot = Plots.of(1000, 600);
        plot.line(epochsVec, trainLossVec, "r-")
            .title("模型训练过程", "损失函数和准确率变化")
            .xlabel("训练轮次")
            .ylabel("损失值");
        
        plot.line(epochsVec, valLossVec, "b--")
            .ylabel("损失值");
        
        plot.line(epochsVec, accuracyVec, "g:")
            .ylabel("准确率");
        
        plot.show();
    }
}
```

---

## 第八部分：扩展二维图表 (Part 8 - Extended 2D Charts)

本节 API 均来自 `IPlot`；小节标题中的 **4.x** 为历史编号，可忽略，仅表示本部分内部顺序。

### 4.1 面积图与阶梯图 / Area & Step Charts

#### 面积图 / Area Chart

```java
public class AreaChartExample {
    public static void main(String[] args) {
        // 生成时间序列数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        IVector<Double> y = Linalg.vector(new double[]{10, 25, 18, 35, 28, 42, 38, 55, 48, 62});
        
        // 面积图 - 折线下方填充
        Plots.of(800, 600)
            .area(x, y)
            .title("Cumulative Sales Trend")
            .xlabel("Month")
            .ylabel("Sales")
            .saveAsHtml("area_chart.html");
    }
}
```

#### 阶梯图 / Step Chart

```java
public class StepChartExample {
    public static void main(String[] args) {
        // 阶梯数据（如价格变动、状态切换）
        IVector<Double> x = Linalg.vector(new double[]{0, 1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 10, 15, 15, 12, 12});
        
        // 阶梯图 - 先水平后垂直变化
        Plots.of(800, 600)
            .step(x, y)
            .title("Step Changes (Post-step)")
            .saveAsHtml("step_chart.html");
    }
}
```

### 4.2 高级柱状图 / Advanced Bar Charts

#### 水平柱状图 / Horizontal Bar Chart

```java
public class HorizontalBarExample {
    public static void main(String[] args) {
        List<String> categories = List.of("Product A", "Product B", "Product C", "Product D", "Product E");
        IVector<Double> values = Linalg.vector(new double[]{450, 380, 320, 290, 210});
        
        // 水平柱状图 - 类别在Y轴
        Plots.of(800, 600)
            .barh(categories, values)
            .title("Sales by Product (Horizontal)")
            .saveAsHtml("barh_chart.html");
    }
}
```

#### 堆叠柱状图 / Stacked Bar Chart

```java
public class StackedBarExample {
    public static void main(String[] args) {
        List<String> quarters = List.of("Q1", "Q2", "Q3", "Q4");
        
        // 每个季度的多层数据
        double[][] data = {
            {120, 150, 180, 200},  // 线上销售
            {80, 90, 100, 110}     // 线下销售
        };
        IMatrix<Double> values = Linalg.matrix(data);
        List<String> layers = List.of("Online", "Offline");
        
        Plots.of(800, 600)
            .barStacked(quarters, values, layers)
            .title("Quarterly Sales by Channel (Stacked)")
            .saveAsHtml("stacked_bar.html");
    }
}
```

### 4.3 高级散点图 / Advanced Scatter Plots

#### 气泡散点图 / Bubble Scatter Chart

```java
public class BubbleScatterExample {
    public static void main(String[] args) {
        IVector<Double> x = Linalg.vector(new double[]{10, 20, 30, 40, 50});
        IVector<Double> y = Linalg.vector(new double[]{15, 25, 35, 45, 55});
        
        // 第三维度：气泡大小
        IVector<Double> sizes = Linalg.vector(new double[]{100, 200, 300, 400, 500});
        
        Plots.of(800, 600)
            .scatter(x, y, sizes)
            .title("Bubble Chart (Size = Market Share)")
            .saveAsHtml("bubble_scatter.html");
    }
}
```

#### 带回归线的散点图 / Regression Plot

```java
public class RegressionPlotExample {
    public static void main(String[] args) {
        // 生成相关数据
        java.util.Random rand = new java.util.Random(42);
        int n = 50;
        double[] xData = new double[n];
        double[] yData = new double[n];
        
        for (int i = 0; i < n; i++) {
            xData[i] = i;
            yData[i] = 2 * i + 10 + rand.nextGaussian() * 5;
        }
        
        // 散点 + 回归线（无置信带）
        Plots.of(800, 600)
            .regplot(Linalg.vector(xData), Linalg.vector(yData))
            .title("Linear Regression (y = 2x + 10)")
            .saveAsHtml("regplot_basic.html");
            
        // 带置信带的回归图
        Plots.of(800, 600)
            .regplot(Linalg.vector(xData), Linalg.vector(yData), true)  // true = 显示置信带
            .title("Linear Regression with Confidence Band")
            .saveAsHtml("regplot_confidence.html");
    }
}
```

#### 误差棒图 / Error Bar Chart

```java
public class ErrorBarExample {
    public static void main(String[] args) {
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 15, 13, 18, 16});
        
        // 误差值
        IVector<Double> errors = Linalg.vector(new double[]{1.5, 2.0, 1.8, 2.2, 1.6});
        
        Plots.of(800, 600)
            .errorbar(x, y, errors)
            .title("Measurements with Error Bars")
            .saveAsHtml("errorbar_chart.html");
    }
}
```

### 4.4 统计分布图表 / Statistical Distribution Charts

#### 核密度估计图 / KDE Plot

```java
public class KdePlotExample {
    public static void main(String[] args) {
        // 生成双峰分布数据
        java.util.Random rand = new java.util.Random(42);
        double[] data = new double[1000];
        for (int i = 0; i < 1000; i++) {
            data[i] = rand.nextBoolean() ? rand.nextGaussian() * 2 + 5 : rand.nextGaussian() * 2 + 10;
        }
        
        Plots.of(800, 600)
            .kdeplot(Linalg.vector(data))  // 默认256网格点，自动带宽
            .title("KDE: Bimodal Distribution")
            .saveAsHtml("kde_chart.html");
            
        // 自定义参数
        Plots.of(800, 600)
            .kdeplot(Linalg.vector(data), 512, 0.5)  // 512网格点，带宽0.5
            .title("KDE: Custom Parameters")
            .saveAsHtml("kde_custom.html");
    }
}
```

#### QQ图 / QQ Plot

```java
public class QqPlotExample {
    public static void main(String[] args) {
        // 生成正态分布数据
        IVector<Double> normalData = Linalg.randn(200);
        
        // QQ图 - 验证数据是否符合正态分布
        Plots.of(800, 600)
            .qqplot(normalData)
            .title("QQ Plot: Normal Distribution Check")
            .saveAsHtml("qqplot_normal.html");
    }
}
```

### 4.5 多变量分析图表 / Multi-variable Analysis

#### 成对关系图 / Pair Plot

```java
public class PairPlotExample {
    public static void main(String[] args) {
        // 生成多变量数据（模拟鸢尾花数据集风格）
        java.util.Random rand = new java.util.Random(42);
        int n = 150;
        double[][] data = new double[n][4];
        
        for (int i = 0; i < n; i++) {
            data[i][0] = 5 + rand.nextGaussian();   // Sepal Length
            data[i][1] = 3 + rand.nextGaussian() * 0.5;   // Sepal Width
            data[i][2] = 4 + rand.nextGaussian() * 0.7;   // Petal Length
            data[i][3] = 1.5 + rand.nextGaussian() * 0.3;   // Petal Width
        }
        IMatrix<Double> matrix = Linalg.matrix(data);
        List<String> columns = List.of("Sepal L", "Sepal W", "Petal L", "Petal W");
        
        // 成对关系图 - 对角线使用KDE
        Plots.of(1000, 1000)
            .pairplot(matrix, columns, PairplotDiagonal.KDE)
            .title("Pair Plot: Iris-like Data")
            .saveAsHtml("pairplot_kde.html");
            
        // 对角线使用直方图
        Plots.of(1000, 1000)
            .pairplot(matrix, columns, PairplotDiagonal.HIST)
            .title("Pair Plot: Diagonal Histograms")
            .saveAsHtml("pairplot_hist.html");
    }
}
```

#### 联合分布图 / Joint Plot

```java
public class JointPlotExample {
    public static void main(String[] args) {
        // 生成相关数据
        java.util.Random rand = new java.util.Random(42);
        int n = 300;
        double[] x = new double[n];
        double[] y = new double[n];
        
        for (int i = 0; i < n; i++) {
            x[i] = rand.nextGaussian();
            y[i] = x[i] * 0.8 + rand.nextGaussian() * 0.5;
        }
        
        // 联合分布图 - 边缘使用KDE
        Plots.of(800, 800)
            .jointplot(Linalg.vector(x), Linalg.vector(y), JointplotMarginal.KDE)
            .title("Joint Plot: X vs Y (KDE Marginals)")
            .saveAsHtml("jointplot_kde.html");
            
        // 边缘使用直方图
        Plots.of(800, 800)
            .jointplot(Linalg.vector(x), Linalg.vector(y), JointplotMarginal.HIST)
            .title("Joint Plot: X vs Y (Histogram Marginals)")
            .saveAsHtml("jointplot_hist.html");
    }
}
```

### 4.6 组合图表 / Combined Charts

#### 双Y轴图 / Dual Y-Axis Chart

```java
public class DualYAxisExample {
    public static void main(String[] args) {
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        
        // 左Y轴数据（温度）
        IVector<Double> temperature = Linalg.vector(new double[]{22, 24, 26, 28, 30, 32, 31, 29, 27, 25});
        
        // 右Y轴数据（湿度）- 不同数量级
        IVector<Double> humidity = Linalg.vector(new double[]{45, 50, 55, 60, 65, 70, 68, 62, 58, 52});
        
        Plots.of(900, 600)
            .lineWithSecondaryY(x, temperature, humidity)
            .ylabel("Temperature (°C)")      // 左Y轴标签
            .y2label("Humidity (%)")          // 右Y轴标签
            .title("Temperature vs Humidity (Dual Y-Axis)")
            .saveAsHtml("dual_y_axis.html");
    }
}
```

#### 子图网格 / Subplots

```java
public class SubplotsExample {
    public static void main(String[] args) {
        IVector<Double> x = Linalg.arange(0, 10, 0.1);
        IVector<Double> y1 = x.map(v -> Math.sin(v));
        IVector<Double> y2 = x.map(v -> Math.cos(v));
        IVector<Double> y3 = x.map(v -> Math.sin(v) * Math.cos(v));
        IVector<Double> y4 = x.map(v -> Math.sin(v) + Math.cos(v));
        
        // 创建2x2子图网格
        Plots.subplots(2, 2)
            .title("2x2 Subplot Grid")
            // 第一个子图 (0,0)
            .subplot(0, 0)
            .line(x, y1)
            .title("sin(x)")
            // 第二个子图 (0,1)
            .subplot(0, 1)
            .line(x, y2)
            .title("cos(x)")
            // 第三个子图 (1,0)
            .subplot(1, 0)
            .line(x, y3)
            .title("sin(x)*cos(x)")
            // 第四个子图 (1,1)
            .subplot(1, 1)
            .line(x, y4)
            .title("sin(x)+cos(x)")
            .saveAsHtml("subplots_2x2.html");
    }
}
```

---

## 第九部分：三维数据可视化 (Part 9 - 3D Data Visualization)

本节 API 均来自 `I3dPlot`。小节标题中的 **5.x** 为历史编号。

### 4.1 3D图表基础入门 / 3D Chart Basics

#### 3D后端选择与切换 / 3D Backend Selection

```java
public class BackendSelectionExample {
    public static void main(String[] args) {
        // 默认使用ECharts GL后端（推荐用于Web导出）
        // Default uses ECharts GL backend (recommended for web export)
        
        // 切换到JavaFX 3D后端（桌面应用）
        // Switch to JavaFX 3D backend (desktop applications)
        Plots.setProvider3d(Plots.PlotProvider3d.JavaFx);
        I3dPlot desktopPlot = Plots.of3d(800, 600);
        
        // 切换回ECharts GL后端
        // Switch back to ECharts GL backend
        Plots.setProvider3d(Plots.PlotProvider3d.EchartsGL);
        I3dPlot webPlot = Plots.of3d(800, 600);
    }
}
```

#### 基础3D散点图 / Basic 3D Scatter Plot

```java
public class Basic3DScatterExample {
    public static void main(String[] args) {
        // 生成3D高斯分布数据
        // Generate 3D Gaussian distribution data
        IVector<Double> x = Linalg.randn(200);
        IVector<Double> y = Linalg.randn(200);
        IVector<Double> z = Linalg.randn(200);
        
        // ECharts GL后端：生成可交互的Web图表
        // ECharts GL backend: generate interactive web chart
        Plots.of3d(800, 600, "dark")
            .scatter3d(x, y, z)
            .title("3D Gaussian Cloud")
            .xlabel("X Axis")
            .ylabel("Y Axis")
            .zlabel("Z Axis")
            .saveAsHtml("3d_scatter.html");
            
        System.out.println("已生成可交互3D图表: 3d_scatter.html");
        System.out.println("Generated interactive 3D chart: 3d_scatter.html");
    }
}
```

#### 带分组的3D散点图 / 3D Scatter with Groups

```java
public class Grouped3DScatterExample {
    public static void main(String[] args) {
        // 生成螺旋数据
        // Generate spiral data
        int n = 150;
        IVector<Double> t = Linalg.vector(IntStream.range(0, n)
            .mapToDouble(i -> i * 0.1).toArray());
        IVector<Double> x = t.map(v -> v * Math.cos(v));
        IVector<Double> y = t.map(v -> v * Math.sin(v));
        IVector<Double> z = t.map(v -> v);
        
        // 创建分组标签
        // Create group labels
        List<String> hue = IntStream.range(0, n)
            .mapToObj(i -> "Section" + (i / 50))
            .toList();
        
        Plots.of3d(900, 700)
            .scatter3d(x, y, z, hue)
            .title("3D Spiral with Groups")
            .subtitle("Colored by section")
            .saveAsHtml("3d_spiral.html");
    }
}
```

#### 3D气泡图 / 3D Bubble Chart

```java
public class Bubble3DExample {
    public static void main(String[] args) {
        int n = 80;
        IVector<Double> x = Linalg.vector(IntStream.range(0, n)
            .mapToDouble(i -> i * 0.2).toArray());
        IVector<Double> y = Linalg.vector(IntStream.range(0, n)
            .mapToDouble(i -> Math.sin(i * 0.2)).toArray());
        IVector<Double> z = Linalg.vector(IntStream.range(0, n)
            .mapToDouble(i -> Math.cos(i * 0.18)).toArray());
        
        // 第四维度：气泡大小
        // Fourth dimension: bubble size
        IVector<Double> sizes = Linalg.vector(IntStream.range(0, n)
            .mapToDouble(i -> 20 + i * 2).toArray());
        
        Plots.of3d(800, 600)
            .scatterBubble3d(x, y, z, sizes)
            .title("3D Bubble Chart")
            .saveAsHtml("3d_bubble.html");
    }
}
```

### 4.2 3D曲面与地形 / 3D Surfaces & Terrain

#### 3D曲面图 / 3D Surface Plot

```java
public class Surface3DExample {
    public static void main(String[] args) {
        // 创建网格数据
        // Create grid data
        int nx = 30, ny = 25;
        IVector<Double> x = Linalg.vector(IntStream.range(0, nx)
            .mapToDouble(i -> i * 0.2).toArray());
        IVector<Double> y = Linalg.vector(IntStream.range(0, ny)
            .mapToDouble(j -> j * 0.2).toArray());
        
        // 计算Z值：z = sin(2x) * cos(2y)
        double[][] zd = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                zd[i][j] = Math.sin(x.get(i) * 2) * Math.cos(y.get(j) * 2);
            }
        }
        IMatrix<Double> z = Linalg.matrix(zd);
        
        // 基础曲面图
        // Basic surface plot
        Plots.of3d(900, 700, "dark")
            .surface3d(x, y, z)
            .title("3D Surface: z = sin(2x) * cos(2y)")
            .saveAsHtml("surface_basic.html");
            
        // 带底部等高线的曲面图
        // Surface with bottom contour projection
        Plots.of3d(900, 700, "dark")
            .surface3d(x, y, z, true)  // true = 显示底部等高线
            .title("3D Surface with Contour")
            .saveAsHtml("surface_contour.html");
    }
}
```

#### 3D等高线图 / 3D Contour Plot

```java
public class Contour3DExample {
    public static void main(String[] args) {
        int nx = 25, ny = 20;
        IVector<Double> x = Linalg.vector(IntStream.range(0, nx)
            .mapToDouble(i -> i * 0.3).toArray());
        IVector<Double> y = Linalg.vector(IntStream.range(0, ny)
            .mapToDouble(j -> j * 0.3).toArray());
        
        // 创建山峰数据
        // Create peak data
        double[][] zd = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double dx = x.get(i) - 3;
                double dy = y.get(j) - 3;
                zd[i][j] = 10 * Math.exp(-(dx*dx + dy*dy) / 4);
            }
        }
        IMatrix<Double> z = Linalg.matrix(zd);
        
        Plots.of3d(800, 600)
            .contour3d(x, y, z)
            .title("3D Contour Lines")
            .saveAsHtml("contour3d.html");
    }
}
```

#### 3D地形图 / 3D Terrain Plot

```java
public class Terrain3DExample {
    public static void main(String[] args) {
        // 创建DEM-like地形数据
        // Create DEM-like terrain data
        int nx = 40, ny = 35;
        IVector<Double> x = Linalg.vector(IntStream.range(0, nx)
            .mapToDouble(i -> i * 0.5).toArray());
        IVector<Double> y = Linalg.vector(IntStream.range(0, ny)
            .mapToDouble(j -> j * 0.5).toArray());
        
        double[][] elevation = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                // 复合噪声模拟地形
                // Composite noise to simulate terrain
                double e = Math.sin(i * 0.3) * Math.cos(j * 0.2) * 5
                    + Math.sin(i * 0.1 + j * 0.1) * 10
                    + Math.random() * 2;
                elevation[i][j] = e;
            }
        }
        IMatrix<Double> z = Linalg.matrix(elevation);
        
        Plots.of3d(1000, 800, "dark")
            .terrain3d(x, y, z)
            .title("3D Terrain / DEM Visualization")
            .saveAsHtml("terrain3d.html");
    }
}
```

### 4.3 3D柱状图与统计图表 / 3D Bar & Statistical Charts

#### 3D柱状图（多种柱体形状） / 3D Bar Chart with Various Shapes

```java
public class Bar3DShapesExample {
    public static void main(String[] args) {
        List<String> categories = List.of("Q1", "Q2", "Q3", "Q4");
        IVector<Double> values = Linalg.vector(new double[]{120, 200, 150, 180});
        
        // BOX柱体（长方体，默认）
        // BOX extrusion (rectangular prism, default)
        Plots.of3d(800, 600)
            .bar3d(categories, values, I3dPlot.BarExtrusion3D.BOX)
            .title("3D Bar Chart - BOX Style")
            .saveAsHtml("bar3d_box.html");
            
        // CYLINDER柱体（圆柱体）
        // CYLINDER extrusion
        Plots.of3d(800, 600)
            .bar3d(categories, values, I3dPlot.BarExtrusion3D.CYLINDER)
            .title("3D Bar Chart - CYLINDER Style")
            .saveAsHtml("bar3d_cylinder.html");
            
        // CONE柱体（圆锥体）
        // CONE extrusion
        Plots.of3d(800, 600)
            .bar3d(categories, values, I3dPlot.BarExtrusion3D.CONE)
            .title("3D Bar Chart - CONE Style")
            .saveAsHtml("bar3d_cone.html");
    }
}
```

#### 分组3D柱状图 / Grouped 3D Bar Chart

```java
public class GroupedBar3DExample {
    public static void main(String[] args) {
        // 创建分组数据
        // Create grouped data
        List<String> quarters = List.of("Q1", "Q1", "Q2", "Q2", "Q3", "Q3", "Q4", "Q4");
        IVector<Double> sales = Linalg.vector(new double[]{
            100, 120,  // Q1: East, West
            110, 140,  // Q2: East, West
            130, 150,  // Q3: East, West
            160, 180   // Q4: East, West
        });
        List<String> regions = List.of("East", "West", "East", "West", 
                                        "East", "West", "East", "West");
        
        Plots.of3d(900, 700)
            .bar3d(quarters, sales, regions, I3dPlot.BarExtrusion3D.BOX)
            .title("Quarterly Sales by Region (3D)")
            .saveAsHtml("bar3d_grouped.html");
    }
}
```

#### 3D直方图 / 3D Histogram

```java
public class Histogram3DExample {
    public static void main(String[] args) {
        // 生成二维相关数据
        // Generate 2D correlated data
        java.util.Random rand = new java.util.Random(42);
        int n = 2000;
        double[] xData = new double[n];
        double[] yData = new double[n];
        
        for (int i = 0; i < n; i++) {
            double x = rand.nextGaussian();
            xData[i] = x;
            yData[i] = 0.6 * x + 0.4 * rand.nextGaussian();  // 相关性
        }
        
        IVector<Double> x = Linalg.vector(xData);
        IVector<Double> y = Linalg.vector(yData);
        
        Plots.of3d(800, 600)
            .hist3d(x, y, 15, 12)  // 15x12 bins
            .title("3D Joint Histogram")
            .saveAsHtml("hist3d.html");
    }
}
```

### 4.4 3D场数据可视化 / 3D Field Data Visualization

#### 3D向量场 / 3D Vector Field

```java
public class VectorField3DExample {
    public static void main(String[] args) {
        // 创建网格采样点
        // Create grid sampling points
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
                    // 定义向量场：v = [sin(y), -cos(x), 0.15]
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
    }
}
```

#### 3D流线图 / 3D Streamlines

```java
public class Streamlines3DExample {
    public static void main(String[] args) {
        // 使用与向量场相同的网格
        int g = 4;
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
                    x[idx] = i;
                    y[idx] = j;
                    z[idx] = k;
                    // 旋转向量场
                    u[idx] = y[idx] - 1.5;
                    v[idx] = -(x[idx] - 1.5);
                    w[idx] = 0.1;
                    idx++;
                }
            }
        }
        
        Plots.of3d(900, 700, "dark")
            .streamlines3d(
                Linalg.vector(x), Linalg.vector(y), Linalg.vector(z),
                Linalg.vector(u), Linalg.vector(v), Linalg.vector(w)
            )
            .title("3D Streamlines")
            .saveAsHtml("streamlines3d.html");
    }
}
```

### 4.5 3D高级图表 / Advanced 3D Charts

#### 3D关系图（网络图） / 3D Graph

```java
public class Graph3DExample {
    public static void main(String[] args) {
        // 定义节点
        // Define nodes
        List<Map<String, Object>> nodes = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            Map<String, Object> node = new HashMap<>();
            node.put("x", (i % 4) * 2.0);  // X坐标
            node.put("y", (i / 4) * 2.0);  // Y坐标
            node.put("z", (i % 2) * 1.0);  // Z坐标
            node.put("name", "Node" + i);
            nodes.add(node);
        }
        
        // 定义连接
        // Define links
        List<Map<String, Object>> links = new ArrayList<>();
        links.add(Map.of("source", 0, "target", 1));
        links.add(Map.of("source", 1, "target", 2));
        links.add(Map.of("source", 2, "target", 3));
        links.add(Map.of("source", 0, "target", 4));
        links.add(Map.of("source", 4, "target", 5));
        links.add(Map.of("source", 5, "target", 6));
        links.add(Map.of("source", 6, "target", 7));
        links.add(Map.of("source", 3, "target", 7));
        
        Plots.of3d(800, 600)
            .graph3d(nodes, links)
            .title("3D Network Graph")
            .saveAsHtml("graph3d.html");
    }
}
```

#### 3D雷达图 / 3D Radar Chart

```java
public class Radar3DExample {
    public static void main(String[] args) {
        // 两组产品的多维指标对比
        // Multi-dimensional indicator comparison for two products
        double[][] data = {
            {0.8, 0.6, 0.9, 0.7},  // Product A
            {0.5, 0.8, 0.4, 0.85}  // Product B
        };
        IMatrix<Double> matrix = Linalg.matrix(data);
        List<String> indicators = List.of("Performance", "Price", "Quality", "UX");
        List<String> products = List.of("Product A", "Product B");
        
        Plots.of3d(800, 600)
            .radar3d(matrix, indicators, products)
            .title("Product Comparison (3D Radar)")
            .saveAsHtml("radar3d.html");
    }
}
```

### 4.6 3D性能优化与大数据 / 3D Performance & Large Data

#### 大数据量自动降采样 / Automatic Sampling for Large Datasets

```java
public class LargeData3DExample {
    public static void main(String[] args) {
        // 生成10000个数据点
        // Generate 10,000 data points
        IVector<Double> x = Linalg.randn(10000);
        IVector<Double> y = Linalg.randn(10000);
        IVector<Double> z = Linalg.randn(10000);
        
        // 当数据量超过2000时，自动启用降采样
        // When data exceeds 2000 points, auto-sampling is enabled
        Plots.of3d(800, 600)
            .scatter3d(x, y, z)
            .title("Large Dataset: 10,000 points (Auto-sampled to ~5,000)")
            .saveAsHtml("large_data_3d.html");
            
        // 查看降采样提示
        // Check the subtitle for sampling information
    }
}
```

#### 使用LOD级别控制细节 / Controlling Detail with LOD Levels

```java
public class LOD3DExample {
    public static void main(String[] args) {
        // 准备数据
        double[][] data = new double[5000][3];
        for (int i = 0; i < 5000; i++) {
            data[i] = new double[]{
                Math.random() * 100,
                Math.random() * 100,
                Math.random() * 100
            };
        }
        
        // 手动应用LOD采样
        // Apply LOD sampling manually
        var sampled = DataSamplingUtils.lodSample(data, DataSamplingUtils.LodLevel.MEDIUM);
        System.out.println("Original: " + data.length + " points");
        System.out.println("Sampled: " + sampled.length + " points");
        
        // 在图表中使用采样后的数据
        // Use sampled data in chart (implementation-specific)
    }
}
```

### 4.7 3D图表主题与样式 / 3D Chart Themes & Styles

#### 3D主题应用 / Applying 3D Themes

```java
public class Theme3DExample {
    public static void main(String[] args) {
        IVector<Double> x = Linalg.randn(100);
        IVector<Double> y = Linalg.randn(100);
        IVector<Double> z = Linalg.randn(100);
        
        // 深色主题
        // Dark theme
        Plots.of3d(800, 600, "dark")
            .scatter3d(x, y, z)
            .title("Dark Theme 3D")
            .saveAsHtml("3d_dark.html");
            
        // 未来风格主题
        // Futuristic theme
        Plots.of3d(800, 600, "futuristic")
            .scatter3d(x, y, z)
            .title("Futuristic Theme 3D")
            .saveAsHtml("3d_futuristic.html");
            
        // 学术风格主题
        // Academic theme
        Plots.of3d(800, 600, "academic")
            .scatter3d(x, y, z)
            .title("Academic Theme 3D")
            .saveAsHtml("3d_academic.html");
    }
}
```

---

**数据可视化示例** - 让数据可视化更简单！

**Data Visualization Examples** - Make data visualization simpler!

## 总结 / Summary

本文档按照从简单到复杂的顺序，系统性地介绍了数据可视化包的各种功能。通过循序渐进的学习，您可以：

- **掌握基础**：从最简单的图表开始，逐步建立可视化基础
- **应用实践**：通过实际案例学习不同图表的使用场景
- **进阶提升**：掌握高级功能和专业应用，包括小提琴图等统计图表
- **灵活运用**：根据实际需求选择合适的可视化方案
- **数据分析**：掌握数据分布分析和统计可视化方法
- **样式定制**：学会使用完整的样式系统进行图表美化
- **主题管理**：掌握智能主题推荐和样式融合技术

This document systematically introduces various functions of the data visualization package in order from simple to complex. Through progressive learning, you can:

- **Master the basics**: Start with the simplest charts and gradually build visualization foundations
- **Apply in practice**: Learn usage scenarios of different charts through real cases
- **Advance and improve**: Master advanced features and professional applications
- **Use flexibly**: Choose appropriate visualization solutions based on actual needs
- **Data analysis**: Master data distribution analysis and statistical visualization methods
- **Style customization**: Learn to use the complete style system for chart beautification
- **Theme management**: Master intelligent theme recommendation and style fusion techniques



---

## 学习路径建议 / Learning Path Recommendations

### 初学者路径 / Beginner Path
1. 从第一部分开始，掌握基本的图表创建
2. 理解数据结构和基本配置
3. 练习简单的数据可视化

### 中级用户路径 / Intermediate Path
1. 掌握多组数据对比和分组图表
2. 学习数据分布分析方法
3. 理解不同图表类型的适用场景
4. 掌握箱线图和小提琴图的使用
5. 学习统计图表的应用

### 高级用户路径 / Advanced Path
1. 掌握复杂图表和特殊可视化
2. 学习自定义配置和主题
3. 理解性能优化和错误处理

### 专业用户路径 / Professional Path
1. 掌握所有图表类型和高级功能
2. 能够根据业务需求选择合适的可视化方案
3. 能够处理复杂的数据分析和可视化任务

---

