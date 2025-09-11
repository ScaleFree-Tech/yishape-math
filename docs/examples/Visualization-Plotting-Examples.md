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
        // 生成简单的数据
        float[] data = {1, 2, 3, 4, 5};
        IVector y = IVector.of(data);
        
        // 使用流式API创建图表
        Plots.of(800, 600)
            .line(y)
            .title("我的第一个图表")
            .show();
        
        System.out.println("图表已保存为 first_chart.html");
    }
}
```

### 1.2 单向量线图 / Single Vector Line Chart

```java
public class SingleVectorLineExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 35, 40, 45, 50, 55});
        
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
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 35, 40, 45, 50, 55});
        
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

---

## 第二部分：基础应用 (Level 2 - 基础应用)

### 2.1 双向量线图 / Two Vector Line Chart

```java
public class TwoVectorLineExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 35, 40, 45, 50, 55});
        
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
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
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
        IVector data = IVector.of(new float[]{30, 25, 20, 15, 10});
        
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
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
        
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

## 第三部分：中级应用 (Level 3 - 中级应用)

### 3.1 分组柱状图 / Grouped Bar Chart

```java
public class GroupedBarChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
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
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25, 15, 25, 20, 35, 30});
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
        float[] histData = new float[100];
        for (int i = 0; i < 100; i++) {
            histData[i] = (float) (Math.random() * 10 + 5); // 均值5，标准差约2.9
        }
        IVector histVector = IVector.of(histData);
        
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
        IVector data = IVector.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
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
        IVector data = IVector.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        
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
        IVector data = IVector.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
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

## 第四部分：高级应用 (Level 4 - 高级应用)

### 4.1 极坐标柱状图 / Polar Bar Chart

```java
public class PolarBarChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
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
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
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
        IVector data = IVector.of(new float[]{10, 20, 15, 30, 25});
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
        float[][] heatmapArray = {
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6},
            {4, 5, 6, 7}
        };
        IMatrix data = IMatrix.of(heatmapArray);
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
        IVector data = IVector.of(new float[]{80, 90, 70, 85, 95, 75});
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
        IVector data = IVector.of(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        
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
        float[][] candlestickArray = {
            {100, 110, 95, 115},
            {110, 120, 105, 125},
            {120, 115, 110, 130},
            {115, 125, 110, 135},
            {125, 130, 120, 140}
        };
        IMatrix candlestickData = IMatrix.of(candlestickArray);
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

## 第五部分：专业应用 (Level 5 - 专业应用)

### 5.1 漏斗图 / Funnel Chart

```java
public class FunnelChartExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector data = IVector.of(new float[]{100, 80, 60, 40, 20});
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
        float[][] dataArray = {
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6},
            {4, 5, 6, 7}
        };
        IMatrix data = IMatrix.of(dataArray);
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
        float value = 75.5f;
        float max = 100.0f;
        float min = 0.0f;
        
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
        float[] groupA = new float[50];
        float[] groupB = new float[50];
        float[] groupC = new float[50];
        
        // 生成不同分布的数据
        for (int i = 0; i < 50; i++) {
            groupA[i] = (float) (Math.random() * 10 + 5); // 正态分布
            groupB[i] = (float) (Math.random() * 15 + 10); // 右偏分布
            groupC[i] = (float) (Math.random() * 8 + 8); // 左偏分布
        }
        
        // 合并数据
        float[] allData = new float[150];
        String[] allLabels = new String[150];
        
        System.arraycopy(groupA, 0, allData, 0, 50);
        System.arraycopy(groupB, 0, allData, 50, 50);
        System.arraycopy(groupC, 0, allData, 100, 50);
        
        Arrays.fill(allLabels, 0, 50, "组A");
        Arrays.fill(allLabels, 50, 100, "组B");
        Arrays.fill(allLabels, 100, 150, "组C");
        
        IVector data = IVector.of(allData);
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

---

## 高级配置和自定义 / Advanced Configuration and Customization

### 自定义主题 / Custom Theme

```java
public class CustomThemeExample {
    public static void main(String[] args) {
        // 生成示例数据
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25});
        
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
        IVector x = IVector.of(new float[]{1, 2, 3, 4, 5});
        IVector y = IVector.of(new float[]{10, 20, 15, 30, 25});
        
        // 自定义坐标轴刻度
        AxisTicks xTicks = new AxisTicks();
        xTicks.setTickValues(IVector.of(new float[]{1, 2, 3, 4, 5}));
        xTicks.setTickLabels(Arrays.asList("一月", "二月", "三月", "四月", "五月"));
        
        AxisTicks yTicks = new AxisTicks();
        yTicks.setTickValues(IVector.of(new float[]{0, 10, 20, 30, 40}));
        
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
            IVector data = IVector.of(new float[]{1, 2, 3, 4, 5});
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
        float[] dataArray = new float[10000];
        for (int i = 0; i < 10000; i++) {
            dataArray[i] = (float) Math.sin(i * 0.01) * 100 + (float) Math.random() * 10;
        }
        IVector data = IVector.of(dataArray);
        
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

## 总结 / Summary

本文档按照从简单到复杂的顺序，系统性地介绍了数据可视化包的各种功能。通过循序渐进的学习，您可以：

- **掌握基础**：从最简单的图表开始，逐步建立可视化基础
- **应用实践**：通过实际案例学习不同图表的使用场景
- **进阶提升**：掌握高级功能和专业应用，包括小提琴图等统计图表
- **灵活运用**：根据实际需求选择合适的可视化方案
- **数据分析**：掌握数据分布分析和统计可视化方法

This document systematically introduces various functions of the data visualization package in order from simple to complex. Through progressive learning, you can:

- **Master the basics**: Start with the simplest charts and gradually build visualization foundations
- **Apply in practice**: Learn usage scenarios of different charts through real cases
- **Advance and improve**: Master advanced features and professional applications
- **Use flexibly**: Choose appropriate visualization solutions based on actual needs

---

**数据可视化示例** - 让数据可视化更简单！

**Data Visualization Examples** - Make data visualization simpler!
