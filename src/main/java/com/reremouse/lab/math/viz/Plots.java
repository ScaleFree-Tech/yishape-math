package com.reremouse.lab.math.viz;

import java.util.List;
import java.util.Map;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 * Plotting static factory class that provides static methods for creating various chart types
 * 绘图静态工厂类，提供创建各种图表类型的静态方法
 * @author lteb2
 */
public final class Plots {
    
//    private Plots() {
//        // 工具类，防止实例化
//        // Utility class, prevent instantiation
//    }
    
    // ========== Basic Factory Methods 基础工厂方法 ==========
    
    /**
     * Create a default plot object
     * 创建默认绘图对象
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot of() {
        return new RerePlot();
    }

    /**
     * Create a plot object with specified dimensions
     * 创建指定尺寸的绘图对象
     * @param width Chart width 图表宽度
     * @param height Chart height 图表高度
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot of(int width, int height) {
        return new RerePlot(width, height);
    }

    /**
     * Create a plot object with specified dimensions and theme
     * 创建指定尺寸和主题的绘图对象
     * @param width Chart width 图表宽度
     * @param height Chart height 图表高度
     * @param theme Theme name 主题名称
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot of(int width, int height, String theme) {
        return new RerePlot(width, height, theme);
    }
    
    // ========== Chart Type Specific Factory Methods 图表类型专用工厂方法 ==========
    
    // ========== Basic Chart Factory Methods 基础图表工厂方法 ==========
    
    /**
     * Create a line chart
     * 创建线图
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot line(IVector<?> x, IVector<?> y) {
        return new RerePlot().line(x, y);
    }
    
    /**
     * Create a line chart (with style string support)
     * 创建线图（支持样式字符串）
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param styleString Style string 样式字符串
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot line(IVector<?> x, IVector<?> y, String styleString) {
        return new RerePlot().line(x, y, styleString);
    }
    
    /**
     * Create a single vector line chart
     * 创建单向量线图
     * @param y Y-axis data Y轴数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot line(IVector<?> y) {
        return new RerePlot().line(y);
    }
    
    /**
     * Create a grouped line chart
     * 创建分组线图
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param hue Color grouping 颜色分组
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot line(IVector<?> x, IVector<?> y, List<String> hue) {
        return new RerePlot().line(x, y, hue);
    }
    
    /**
     * Create a scatter plot
     * 创建散点图
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot scatter(IVector<?> x, IVector<?> y) {
        return new RerePlot().scatter(x, y);
    }
    
    /**
     * Create a scatter plot (with style string support)
     * 创建散点图（支持样式字符串）
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param styleString Style string 样式字符串
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot scatter(IVector<?> x, IVector<?> y, String styleString) {
        return new RerePlot().scatter(x, y, styleString);
    }
    
    /**
     * Create a grouped scatter plot
     * 创建分组散点图
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param hue Color grouping 颜色分组
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot scatter(IVector<?> x, IVector<?> y, List<String> hue) {
        return new RerePlot().scatter(x, y, hue);
    }
    
    /**
     * Create a pie chart
     * 创建饼图
     * @param data Data 数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot pie(IVector<?> data) {
        return new RerePlot().pie(data);
    }
    
    /**
     * Create a pie chart (with label support)
     * 创建饼图（支持标签）
     * @param data Data 数据
     * @param labels Labels 标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot pie(IVector<?> data, List<String> labels) {
        return new RerePlot().pie(data, labels, (PlotStyle) null);
    }
    
    /**
     * Create a bar chart
     * 创建柱状图
     * @param data Data 数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot bar(IVector<?> data) {
        return new RerePlot().bar(data);
    }
    
    /**
     * Create a grouped bar chart
     * 创建分组柱状图
     * @param data Data 数据
     * @param hue Grouping labels 分组标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot bar(IVector<?> data, List<String> hue) {
        return new RerePlot().bar(data, hue);
    }
    
    /**
     * Create a histogram
     * 创建直方图
     * @param data Data 数据
     * @param fittingLine Whether to display the fitting line 是否显示拟合线
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot hist(IVector<?> data, boolean fittingLine) {
        return new RerePlot().hist(data, fittingLine);
    }
    
    // ========== Polar Coordinate Chart Factory Methods 极坐标图表工厂方法 ==========
    
    /**
     * Create a polar bar chart
     * 创建极坐标柱状图
     * @param data Data 数据
     * @param categories Category labels 类别标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot polarBar(IVector<?> data, List<String> categories) {
        return new RerePlot().polarBar(data, categories);
    }
    
    /**
     * Create a polar line chart
     * 创建极坐标线图
     * @param data Data 数据
     * @param categories Category labels 类别标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot polarLine(IVector<?> data, List<String> categories) {
        return new RerePlot().polarLine(data, categories);
    }
    
    /**
     * Create a polar scatter plot
     * 创建极坐标散点图
     * @param data Data 数据
     * @param categories Category labels 类别标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot polarScatter(IVector<?> data, List<String> categories) {
        return new RerePlot().polarScatter(data, categories);
    }
    
    // ========== Statistical Chart Factory Methods 统计图表工厂方法 ==========
    
    /**
     * Create a box plot
     * 创建箱线图
     * @param data Data 数据
     * @param labels Labels 标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot boxplot(IVector<?> data, List<String> labels) {
        return new RerePlot().boxplot(data, labels);
    }
    
    /**
     * Create a violin plot
     * 创建小提琴图
     * @param data Data 数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot violinplot(IVector<?> data) {
        RerePlot plot = new RerePlot();
        plot.violinplot(data);
        return plot;
    }
    
    /**
     * Create a violin plot (with label support)
     * 创建小提琴图（支持标签）
     * @param data Data 数据
     * @param labels Labels 标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot violinplot(IVector<?> data, List<String> labels) {
        RerePlot plot = new RerePlot();
        plot.violinplot(data, labels);
        return plot;
    }
    
    /**
     * Create a candlestick chart
     * 创建K线图
     * @param data K-line data matrix (each row contains [open price, close price, lowest price, highest price])
     *             K线数据矩阵（每行包含[开盘价, 收盘价, 最低价, 最高价]）
     * @param dates Date labels 日期标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot candlestick(IMatrix<?> data, List<String> dates) {
        return new RerePlot().candlestick(data, dates);
    }
    
    // ========== Special Chart Factory Methods 特殊图表工厂方法 ==========
    
    /**
     * Create a funnel chart
     * 创建漏斗图
     * @param data Data 数据
     * @param labels Labels 标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot funnel(IVector<?> data, List<String> labels) {
        return new RerePlot().funnel(data, labels);
    }
    
    /**
     * Create a Sankey diagram
     * 创建桑基图
     * @param nodes Node data 节点数据
     * @param links Connection data 连接数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return new RerePlot().sankey(nodes, links);
    }
    
    /**
     * Create a sunburst chart
     * 创建旭日图
     * @param data Hierarchical data 层次数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot sunburst(List<Map<String, Object>> data) {
        return new RerePlot().sunburst(data);
    }
    
    /**
     * Create a theme river chart
     * 创建主题河流图
     * @param data Time series data 时间序列数据
     * @param categories Category labels 类别标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot themeRiver(List<Map<String, Object>> data, List<String> categories) {
        return new RerePlot().themeRiver(data, categories);
    }
    
    /**
     * Create a tree diagram
     * 创建树图
     * @param data Tree data 树形数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot tree(List<Map<String, Object>> data) {
        return new RerePlot().tree(data);
    }
    
    /**
     * Create a treemap
     * 创建矩形树图
     * @param data Hierarchical data 层次数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot treemap(List<Map<String, Object>> data) {
        return new RerePlot().treemap(data);
    }
    
    /**
     * Create a relationship graph
     * 创建关系图
     * @param nodes Node data 节点数据
     * @param links Connection data 连接数据
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return new RerePlot().graph(nodes, links);
    }
    
    /**
     * Create a parallel coordinates plot
     * 创建平行坐标图
     * @param data Multidimensional data matrix 多维数据矩阵
     * @param dimensions Dimension labels 维度标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot parallel(IMatrix<?> data, List<String> dimensions) {
        return new RerePlot().parallel(data, dimensions);
    }
    
    // ========== Extended Chart Factory Methods 扩展图表工厂方法 ==========
    
    /**
     * Create a heatmap
     * 创建热力图
     * @param data 2D data matrix 二维数据矩阵
     * @param xLabels X-axis labels X轴标签
     * @param yLabels Y-axis labels Y轴标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot heatmap(IMatrix<?> data, List<String> xLabels, List<String> yLabels) {
        return new RerePlot().heatmap(data, xLabels, yLabels);
    }
    
    /**
     * Create a heatmap
     * 创建热力图
     * @param data 2D data matrix 二维数据矩阵
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot heatmap(IMatrix<?> data) {
        return new RerePlot().heatmap(data);
    }
    
    /**
     * Create a radar chart
     * 创建雷达图
     * @param data Data 数据
     * @param indicators Indicator labels 指标标签
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot radar(IVector<?> data, List<String> indicators) {
        return new RerePlot().radar(data, indicators);
    }
    
    /**
     * Create a gauge chart
     * 创建仪表盘
     * @param value Current value 当前值
     * @param max Maximum value 最大值
     * @param min Minimum value 最小值
     * @return RerePlot instance RerePlot实例
     */
    public static RerePlot gauge(double value, double max, double min) {
        return new RerePlot().gauge(value, max, min);
    }
    
}