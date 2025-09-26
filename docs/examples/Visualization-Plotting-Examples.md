# 数据可视化使用示例 (Data Visualization Examples)

## 概述 / Overview

本文档按照从简单到复杂的顺序，系统性地编排了数据可视化包的详细使用示例。每个级别都包含相应的理论背景、实践示例和进阶指导。

This document systematically organizes detailed usage examples for the data visualization package in order from simple to complex. Each level includes corresponding theoretical background, practical examples, and advanced guidance.

---

## 第一部分：入门基础 (Part 1 - Getting Started)

### 1.1 环境准备和基本概念 / Environment Setup and Basic Concepts

#### 导入必要的类 / Import Required Classes

```java
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.IPlot;
import com.reremouse.lab.math.viz.PlotStyle;
import java.util.Arrays;
import java.util.List;
```

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
        RerePlot plot = Plots.of(800, 600);
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
        RerePlot plot = Plots.of(800, 600);
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
        RerePlot plot = Plots.of(800, 600);
        
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
        RerePlot plot = Plots.of(800, 600);
        
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

### 6.4 UniversalStyleApplier 通用样式应用示例 / Universal Style Applier Examples

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

### 6.5 ThemeManager 主题管理示例 / Theme Manager Examples

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

### 6.2 HSL颜色空间操作示例 / HSL Color Space Operations Example

```java
public class HSLColorOperationsExample {
    public static void main(String[] args) {
        String baseColor = "#3498DB";
        
        // HSL色彩调整
        String brighter = StyleConverter.adjustBrightness(baseColor, 0.3);
        String moreSaturated = StyleConverter.adjustSaturation(baseColor, 0.2);
        String hueShifted = StyleConverter.shiftHue(baseColor, 60);
        
        // 创建渐变色彩系列
        List<String> gradient = StyleConverter.createLinearGradient(
            "#FF6B6B", "#4ECDC4", 5
        );
        
        // 应用到多系列数据
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        RerePlot plot = Plots.of(800, 600);
        
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

### 6.3 交互状态样式示例 / Interactive State Styling Example

```java
public class InteractiveStateExample {
    public static void main(String[] args) {
        IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> y = Linalg.vector(new double[]{10, 20, 15, 30, 25});
        
        // 创建带交互状态的样式
        PlotStyle interactiveStyle = new PlotStyle()
            .color("#3498DB")
            .emphasis(new PlotStyle().color("#E74C3C").lineWidth(4.0))  // hover
            .blur(new PlotStyle().opacity(0.3))                         // blur
            .select(new PlotStyle().color("#F39C12").lineWidth(3.0));   // select
        
        Plots.of(800, 600)
            .line(x, y, interactiveStyle)
            .title("交互状态样式示例")
            .show();
    }
}
```


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
        RerePlot plot = Plots.of(1200, 800);
        
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
        RerePlot plot = Plots.of(1000, 600);
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

---

**数据可视化示例** - 让数据可视化更简单！

**Data Visualization Examples** - Make data visualization simpler!