# 数据可视化使用示例 - 系统性编排版 (Data Visualization Examples - Systematic Organization)

## 概述 / Overview

本文档按照从简单到复杂的顺序，系统性地编排了数据可视化包的详细使用示例。每个级别都包含相应的理论背景、实践示例和进阶指导。

This document systematically organizes detailed usage examples for the data visualization package in order from simple to complex. Each level includes corresponding theoretical background, practical examples, and advanced guidance.

---

## 第一部分：入门基础 (Level 1 - 基础入门)

### 1.1 环境准备和基本概念 / Environment Setup and Basic Concepts

#### 导入必要的类 / Import Required Classes

```java
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.IPlot;
import java.util.Arrays;
import java.util.List;
```

#### 创建第一个图表 / Create Your First Chart

```java
public class FirstChartExample {
    public static void main(String[] args) {
        // 创建图表对象
        IPlot plot = Plots.of(800, 600);
        
        // 生成简单的数据
        float[] data = {1, 2, 3, 4, 5};
        IVector y = IVector.of(data);
        
        // 绘制最简单的线图
        plot.line(y);
        plot.setTitle("我的第一个图表");
        plot.saveAsHtml("first_chart.html");
        
        System.out.println("图表已保存为 first_chart.html");
    }
}
```

### 1.2 单向量线图 / Single Vector Line Chart

```java
public class SingleVectorLineExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 35, 40, 45, 50, 55});
        
        // 绘制线图
        plot.line(y);
        plot.setTitle("销售趋势图");
        plot.setXlabel("时间（月）");
        plot.setYlabel("销售额（万元）");
        plot.saveAsHtml("single_line_chart.html");
    }
}
```

### 1.3 基础散点图 / Basic Scatter Chart

```java
public class BasicScatterExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 35, 40, 45, 50, 55});
        
        // 绘制散点图
        plot.scatter(x, y);
        plot.setTitle("身高体重关系图");
        plot.setXlabel("身高（cm）");
        plot.setYlabel("体重（kg）");
        plot.saveAsHtml("basic_scatter_chart.html");
    }
}
```

---

## 第二部分：基础应用 (Level 2 - 基础应用)

### 2.1 双向量线图 / Two Vector Line Chart

```java
public class TwoVectorLineExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 35, 40, 45, 50, 55});
        
        // 绘制双向量线图
        plot.line(x, y);
        plot.setTitle("销售趋势图", "2024年各月销售数据");
        plot.setXlabel("月份");
        plot.setYlabel("销售额（万元）");
        plot.saveAsHtml("two_vector_line_chart.html");
    }
}
```

### 2.2 多组散点图 / Multi-group Scatter Chart

```java
public class MultiGroupScatterExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
        List<String> hue = Arrays.asList("产品A", "产品A", "产品A", "产品A", "产品A",
                                        "产品B", "产品B", "产品B", "产品B", "产品B");
        
        // 绘制多组散点图
        plot.scatter(x, y, hue);
        plot.setTitle("产品对比分析", "2024年各月产品销售对比");
        plot.setXlabel("月份");
        plot.setYlabel("销售额（万元）");
        plot.saveAsHtml("multi_group_scatter_chart.html");
    }
}
```

### 2.3 饼图 / Pie Chart

```java
public class PieChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(600, 600);
        
        // 生成示例数据
        IVector data = IVector.of(new float[]{30, 25, 20, 15, 10});
        
        // 绘制饼图
        plot.pie(data);
        plot.setTitle("市场份额分布", "2024年各产品线市场份额");
        plot.saveAsHtml("pie_chart.html");
    }
}
```

### 2.4 柱状图 / Bar Chart

```java
public class BarChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
        
        // 绘制柱状图
        plot.bar(data);
        plot.setTitle("销售业绩对比", "2024年各季度销售数据");
        plot.setXlabel("季度");
        plot.setYlabel("销售额（万元）");
        plot.saveAsHtml("bar_chart.html");
    }
}
```

---

## 第三部分：中级应用 (Level 3 - 中级应用)

### 3.1 分组柱状图 / Grouped Bar Chart

```java
public class GroupedBarChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
        List<String> hue = Arrays.asList("组A", "组B", "组A", "组B", "组A");
        
        // 绘制分组柱状图
        plot.bar(data, hue);
        plot.setTitle("分组柱状图", "对比不同组别的数据");
        plot.setXlabel("类别");
        plot.setYlabel("数值");
        plot.saveAsHtml("grouped_bar_chart.html");
    }
}
```

### 3.2 多组线图 / Multi-group Line Chart

```java
public class MultiGroupLineChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
        List<String> hue = Arrays.asList("产品A", "产品A", "产品A", "产品A", "产品A",
                                        "产品B", "产品B", "产品B", "产品B", "产品B");
        
        // 绘制多组线图
        plot.line(x, y, hue);
        plot.setTitle("产品对比分析", "2024年各月产品销售对比");
        plot.setXlabel("月份");
        plot.setYlabel("销售额（万元）");
        plot.saveAsHtml("multi_group_line_chart.html");
    }
}
```

### 3.3 直方图 / Histogram

```java
public class HistogramExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        float[] histData = new float[100];
        for (int i = 0; i < 100; i++) {
            histData[i] = (float) (Math.random() * 10 + 5); // 均值5，标准差约2.9
        }
        IVector histVector = IVector.of(histData);
        
        // 绘制直方图（带拟合线）
        plot.hist(histVector, true);
        plot.setTitle("数据分布直方图", "样本数据的正态分布拟合");
        plot.setXlabel("数值区间");
        plot.setYlabel("频次");
        plot.saveAsHtml("histogram_chart.html");
    }
}
```

### 3.4 箱线图 / Box Plot

```java
public class BoxPlotExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector data = IVector.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        List<String> labels = Arrays.asList("数据集");
        
        // 绘制箱线图
        plot.boxplot(data, labels);
        plot.setTitle("数据分布箱线图", "各指标的数据分布情况");
        plot.setXlabel("指标");
        plot.setYlabel("数值");
        plot.saveAsHtml("boxplot_chart.html");
    }
}
```

---

## 第四部分：高级应用 (Level 4 - 高级应用)

### 4.1 极坐标柱状图 / Polar Bar Chart

```java
public class PolarBarChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(600, 600);
        
        // 生成示例数据
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
        List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E");
        
        // 绘制极坐标柱状图
        plot.polarBar(data, categories);
        plot.setTitle("极坐标柱状图");
        plot.saveAsHtml("polar_bar_chart.html");
    }
}
```

### 4.2 极坐标线图 / Polar Line Chart

```java
public class PolarLineChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(600, 600);
        
        // 生成示例数据
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
        List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E");
        
        // 绘制极坐标线图
        plot.polarLine(data, categories);
        plot.setTitle("极坐标线图");
        plot.saveAsHtml("polar_line_chart.html");
    }
}
```

### 4.3 极坐标散点图 / Polar Scatter Chart

```java
public class PolarScatterChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(600, 600);
        
        // 生成示例数据
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
        List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E");
        
        // 绘制极坐标散点图
        plot.polarScatter(data, categories);
        plot.setTitle("极坐标散点图");
        plot.saveAsHtml("polar_scatter_chart.html");
    }
}
```

### 4.4 热力图 / Heatmap

```java
public class HeatmapExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 创建二维数据矩阵
        float[][] heatmapArray = {
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6},
            {4, 5, 6, 7}
        };
        IMatrix data = IMatrix.of(heatmapArray);
        List<String> xLabels = Arrays.asList("X1", "X2", "X3", "X4");
        List<String> yLabels = Arrays.asList("Y1", "Y2", "Y3", "Y4");
        
        // 绘制热力图
        plot.heatmap(data, xLabels, yLabels);
        plot.setTitle("相关性热力图", "各指标间的相关性分析");
        plot.saveAsHtml("heatmap_chart.html");
    }
}
```

### 4.5 雷达图 / Radar Chart

```java
public class RadarChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(600, 600);
        
        // 生成示例数据
        IVector data = IVector.of(new float[]{80, 90, 70, 85, 95, 75});
        List<String> indicators = Arrays.asList("指标1", "指标2", "指标3", "指标4", "指标5", "指标6");
        
        // 绘制雷达图
        plot.radar(data, indicators);
        plot.setTitle("能力雷达图", "各项技能能力评估");
        plot.saveAsHtml("radar_chart.html");
    }
}
```

### 4.6 K线图 / Candlestick Chart

```java
public class CandlestickChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(1000, 600);
        
        // 生成示例数据（开盘价, 收盘价, 最低价, 最高价）
        float[][] candlestickArray = {
            {100, 110, 95, 115},
            {110, 120, 105, 125},
            {120, 115, 110, 130},
            {115, 125, 110, 135},
            {125, 130, 120, 140}
        };
        IMatrix candlestickData = IMatrix.of(candlestickArray);
        List<String> dates = Arrays.asList("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05");
        
        // 绘制K线图
        plot.candlestick(candlestickData, dates);
        plot.setTitle("股票价格K线图", "2024年1月股价走势");
        plot.setXlabel("日期");
        plot.setYlabel("价格（元）");
        plot.saveAsHtml("candlestick_chart.html");
    }
}
```

---

## 第五部分：专业应用 (Level 5 - 专业应用)

### 5.1 漏斗图 / Funnel Chart

```java
public class FunnelChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector data = IVector.of(new float[]{100, 80, 60, 40, 20});
        List<String> labels = Arrays.asList("访问", "注册", "购买", "支付", "完成");
        
        // 绘制漏斗图
        plot.funnel(data, labels);
        plot.setTitle("用户转化漏斗", "从访问到购买的转化流程");
        plot.saveAsHtml("funnel_chart.html");
    }
}
```

### 5.2 桑基图 / Sankey Chart

```java
public class SankeyChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(1000, 600);
        
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
        
        // 绘制桑基图
        plot.sankey(nodes, links);
        plot.setTitle("数据流向图", "各系统间的数据流转情况");
        plot.saveAsHtml("sankey_chart.html");
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
        // 创建图表
        IPlot plot = Plots.of(800, 800);
        
        // 创建层次数据
        List<Map<String, Object>> sunburstData = new ArrayList<>();
        sunburstData.add(createSunburstNode("root", "根节点", 100));
        sunburstData.add(createSunburstNode("child1", "子节点1", 60, "root"));
        sunburstData.add(createSunburstNode("child2", "子节点2", 40, "root"));
        sunburstData.add(createSunburstNode("grandchild1", "孙节点1", 30, "child1"));
        sunburstData.add(createSunburstNode("grandchild2", "孙节点2", 30, "child1"));
        
        // 绘制旭日图
        plot.sunburst(sunburstData);
        plot.setTitle("组织架构图", "公司各部门人员分布");
        plot.saveAsHtml("sunburst_chart.html");
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
        // 创建图表
        IPlot plot = Plots.of(1200, 600);
        
        // 创建时间序列数据
        List<Map<String, Object>> themeRiverData = new ArrayList<>();
        themeRiverData.add(createThemeRiverNode("2024-01-01", "类别A", 10));
        themeRiverData.add(createThemeRiverNode("2024-01-01", "类别B", 20));
        themeRiverData.add(createThemeRiverNode("2024-01-02", "类别A", 15));
        themeRiverData.add(createThemeRiverNode("2024-01-02", "类别B", 25));
        
        List<String> categories = Arrays.asList("类别A", "类别B");
        
        // 绘制主题河流图
        plot.themeRiver(themeRiverData, categories);
        plot.setTitle("新闻热度趋势", "各主题新闻的热度变化");
        plot.saveAsHtml("theme_river_chart.html");
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
        // 创建图表
        IPlot plot = Plots.of(1000, 800);
        
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
        
        // 绘制关系图
        plot.graph(nodes, links);
        plot.setTitle("社交网络图", "用户关系网络分析");
        plot.saveAsHtml("graph_chart.html");
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
        // 创建图表
        IPlot plot = Plots.of(1200, 600);
        
        // 创建数据矩阵
        float[][] dataArray = {
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6},
            {4, 5, 6, 7}
        };
        IMatrix data = IMatrix.of(dataArray);
        List<String> dimensions = Arrays.asList("维度1", "维度2", "维度3", "维度4");
        
        // 绘制平行坐标图
        plot.parallel(data, dimensions);
        plot.setTitle("多维数据分布", "各维度数据的分布情况");
        plot.saveAsHtml("parallel_coordinates_chart.html");
    }
}
```

### 5.7 仪表盘 / Gauge Chart

```java
public class GaugeChartExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(400, 400);
        
        // 设置参数
        float value = 75.5f;
        float max = 100.0f;
        float min = 0.0f;
        
        // 绘制仪表盘
        plot.gauge(value, max, min);
        plot.setTitle("系统性能监控", "CPU使用率实时监控");
        plot.saveAsHtml("gauge_chart.html");
    }
}
```

---

## 高级配置和自定义 / Advanced Configuration and Customization

### 自定义主题 / Custom Theme

```java
public class CustomThemeExample {
    public static void main(String[] args) {
        // 创建带自定义主题的图表
        IPlot plot = Plots.of(800, 600, "dark");
        
        // 生成示例数据
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25});
        
        // 绘制图表
        plot.line(x, y);
        plot.setTitle("自定义主题图表", "使用深色主题");
        plot.setXlabel("X轴");
        plot.setYlabel("Y轴");
        plot.saveAsHtml("custom_theme_chart.html");
    }
}
```

### 自定义坐标轴 / Custom Axis

```java
public class CustomAxisExample {
    public static void main(String[] args) {
        // 创建图表
        IPlot plot = Plots.of(800, 600);
        
        // 生成示例数据
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25});
        
        // 自定义坐标轴刻度
        AxisTicks xTicks = new AxisTicks();
        xTicks.setTickValues(IVector.of(new float[]{1, 2, 3, 4, 5}));
        xTicks.setTickLabels(Arrays.asList("一月", "二月", "三月", "四月", "五月"));
        plot.setXticks(xTicks);
        
        AxisTicks yTicks = new AxisTicks();
        yTicks.setTickValues(IVector.of(new float[]{0, 10, 20, 30, 40}));
        plot.setYticks(yTicks);
        
        // 绘制图表
        plot.line(x, y);
        plot.setTitle("自定义坐标轴图表");
        plot.setXlabel("月份");
        plot.setYlabel("销售额（万元）");
        plot.saveAsHtml("custom_axis_chart.html");
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
            // 创建图表
            IPlot plot = Plots.of(800, 600);
            
            // 检查数据是否为空
            IVector data = IVector.of(new float[]{1, 2, 3, 4, 5});
            if (data.length() == 0) {
                throw new IllegalArgumentException("数据不能为空");
            }
            
            // 绘制图表
            plot.line(data);
            plot.saveAsHtml("chart.html");
            
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
        // 创建图表
        IPlot plot = Plots.of(1200, 800);
        
        // 生成大数据集
        float[] dataArray = new float[10000];
        for (int i = 0; i < 10000; i++) {
            dataArray[i] = (float) Math.sin(i * 0.01) * 100 + (float) Math.random() * 10;
        }
        IVector data = IVector.of(dataArray);
        
        // 对于大数据集，考虑数据采样
        if (data.length() > 1000) {
            System.out.println("大数据集，建议进行数据采样");
        }
        
        // 绘制图表
        plot.line(data);
        plot.setTitle("大数据集示例", "包含10000个数据点");
        plot.setXlabel("索引");
        plot.setYlabel("数值");
        plot.saveAsHtml("large_dataset_chart.html");
    }
}
```

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

### 高级用户路径 / Advanced Path
1. 掌握复杂图表和特殊可视化
2. 学习自定义配置和主题
3. 理解性能优化和错误处理

### 专业用户路径 / Professional Path
1. 掌握所有图表类型和高级功能
2. 能够根据业务需求选择合适的可视化方案
3. 能够处理复杂的数据分析和可视化任务

---

## 总结 / Summary

本文档按照从简单到复杂的顺序，系统性地介绍了数据可视化包的各种功能。通过循序渐进的学习，您可以：

- **掌握基础**：从最简单的图表开始，逐步建立可视化基础
- **应用实践**：通过实际案例学习不同图表的使用场景
- **进阶提升**：掌握高级功能和专业应用
- **灵活运用**：根据实际需求选择合适的可视化方案

This document systematically introduces various functions of the data visualization package in order from simple to complex. Through progressive learning, you can:

- **Master the basics**: Start with the simplest charts and gradually build visualization foundations
- **Apply in practice**: Learn usage scenarios of different charts through real cases
- **Advance and improve**: Master advanced features and professional applications
- **Use flexibly**: Choose appropriate visualization solutions based on actual needs

---

**数据可视化示例** - 让数据可视化更简单！

**Data Visualization Examples** - Make data visualization simpler!
