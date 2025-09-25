package com.reremouse.lab.math.viz;

import java.util.List;
import java.util.Map;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import java.util.ArrayList;

/**
 * Plotting interface that defines basic methods for plot instances
 * 绘图接口，定义绘图实例的基本方法
 *
 * @author lteb2
 */
public interface IPlot {

    // ========== Basic Chart Methods 基础图表方法 ==========
    /**
     * Draw a line chart
     * 绘制线图
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot line(IVector x, IVector y);

    /**
     * Draw a single vector line chart
     * 绘制单向量线图
     *
     * @param x Data vector 数据向量
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot line(IVector x);

    /**
     * Draw a multi-line chart
     * 绘制多线图
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param hue Grouping labels 分组标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot line(IVector x, IVector y, List<String> hue);

    /**
     * Draw a scatter plot
     * 绘制散点图
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot scatter(IVector x, IVector y);

    /**
     * Draw a multi-group scatter plot
     * 绘制多组散点图
     *
     * @param x X-axis data X轴数据
     * @param y Y-axis data Y轴数据
     * @param hue Grouping labels 分组标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot scatter(IVector x, IVector y, List<String> hue);

    /**
     * Draw a pie chart
     * 绘制饼图
     *
     * @param x Data vector 数据向量
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot pie(IVector x);

    /**
     * Draw a bar chart
     * 绘制柱状图
     *
     * @param x Data vector 数据向量
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot bar(IVector x);

    /**
     * Draw a grouped bar chart
     * 绘制分组柱状图
     *
     * @param xticks Group labels 分组标签
     * @param y Data vector 数据向量
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot bar(List<String> xticks, IVector y);

    /**
     * Draw a grouped bar chart with hue grouping
     * 使用hue分组绘制分组柱状图
     *
     * @param y Data vector 数据向量
     * @param hue Grouping labels 分组标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    default IPlot bar(IVector y, List<String> hue) {
        var is = Linalg.range(y.length()).toIntArray();
        List ts = new ArrayList();
        for (int i : is) {
            ts.add("类别" + (i + 1));
        }
        return bar(ts, y, hue);
    }

    /**
     * Draw a grouped bar chart with custom xticks and hue grouping
     * 使用自定义xticks和hue分组绘制分组柱状图
     *
     * @param xticks Group labels 分组标签
     * @param y Data vector 数据向量
     * @param hue Grouping labels 分组标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot bar(List<String> xticks, IVector y, List<String> hue);

    /**
     * Draw a histogram
     * 绘制直方图
     *
     * @param x Data vector 数据向量
     * @param fittingLine Whether to display the fitting line 是否显示拟合线
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot hist(IVector x, boolean fittingLine);

    // ========== Polar Coordinate Chart Methods 极坐标图表方法 ==========
    /**
     * Draw a polar bar chart
     * 绘制极坐标柱状图
     *
     * @param data Data vector 数据向量
     * @param categories Category labels 类别标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot polarBar(IVector data, List<String> categories);

    /**
     * Draw a polar line chart
     * 绘制极坐标线图
     *
     * @param data Data vector 数据向量
     * @param categories Category labels 类别标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot polarLine(IVector data, List<String> categories);

    /**
     * Draw a polar scatter plot
     * 绘制极坐标散点图
     *
     * @param data Data vector 数据向量
     * @param categories Category labels 类别标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot polarScatter(IVector data, List<String> categories);

    // ========== Statistical Chart Methods 统计图表方法 ==========
    /**
     * Draw a box plot
     * 绘制箱线图
     *
     * @param data Data vector 数据向量
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot boxplot(IVector data);

    /**
     * Draw a box plot with labels
     * 绘制带标签的箱线图
     *
     * @param data Data vector 数据向量
     * @param labels Labels 标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot boxplot(IVector data, List<String> labels);

    /**
     * Draw a violin plot
     * 绘制小提琴图
     *
     * @param data Data vector 数据向量
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot violinplot(IVector data);

    /**
     * Draw a violin plot with labels
     * 绘制带标签的小提琴图
     *
     * @param data Data vector 数据向量
     * @param labels Labels 标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot violinplot(IVector data, List<String> labels);

    /**
     * Draw a candlestick chart
     * 绘制K线图
     *
     * @param data Data matrix, each row contains [open price, close price, lowest price, highest price] 
     *             数据矩阵，每行包含[开盘价, 收盘价, 最低价, 最高价]
     * @param dates Date labels 日期标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot candlestick(IMatrix data, List<String> dates);

    // ========== Special Chart Methods 特殊图表方法 ==========
    /**
     * Draw a funnel chart
     * 绘制漏斗图
     *
     * @param data Data vector 数据向量
     * @param labels Labels 标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot funnel(IVector data, List<String> labels);

    /**
     * Draw a Sankey diagram
     * 绘制桑基图
     *
     * @param nodes Node data 节点数据
     * @param links Connection data 连接数据
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links);

    /**
     * Draw a sunburst chart
     * 绘制旭日图
     *
     * @param data Hierarchical data 层次数据
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot sunburst(List<Map<String, Object>> data);

    /**
     * Draw a theme river chart
     * 绘制主题河流图
     *
     * @param data Time series data 时间序列数据
     * @param categories Categories 类别
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot themeRiver(List<Map<String, Object>> data, List<String> categories);

    /**
     * Draw a tree diagram
     * 绘制树图
     *
     * @param data Tree data 树形数据
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot tree(List<Map<String, Object>> data);

    /**
     * Draw a treemap
     * 绘制矩形树图
     *
     * @param data Hierarchical data 层次数据
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot treemap(List<Map<String, Object>> data);

    /**
     * Draw a relationship graph
     * 绘制关系图
     *
     * @param nodes Node data 节点数据
     * @param links Connection data 连接数据
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links);

    /**
     * Draw a parallel coordinates plot
     * 绘制平行坐标图
     *
     * @param data Data matrix 数据矩阵
     * @param dimensions Dimension names 维度名称
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot parallel(IMatrix data, List<String> dimensions);

    // ========== Advanced Chart Methods 完善图表方法 ==========
    /**
     * Draw a heatmap
     * 绘制热力图
     *
     * @param data 2D data matrix 二维数据矩阵
     * @return RerePlot instance RerePlot实例
     */
    public RerePlot heatmap(IMatrix data);

    /**
     * Draw a heatmap with custom labels
     * 绘制带自定义标签的热力图
     *
     * @param data 2D data matrix 二维数据矩阵
     * @param xLabels X-axis labels X轴标签
     * @param yLabels Y-axis labels Y轴标签
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels);

    /**
     * Draw a radar chart
     * 绘制雷达图
     *
     * @param data Data vector 数据向量
     * @param indicators Indicator names 指标名称
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot radar(IVector data, List<String> indicators);

    /**
     * Draw a gauge chart
     * 绘制仪表盘
     *
     * @param value Value 数值
     * @param max Maximum value 最大值
     * @param min Minimum value 最小值
     * @return IPlot instance for method chaining 返回当前实例，支持链式调用
     */
    IPlot gauge(double value, double max, double min);

    // ========== Fluent API Methods 流式API方法 ==========
    /**
     * Set chart title (Fluent API)
     * 设置图表标题（流式API）
     *
     * @param titleText Title text 标题文本
     * @return Current instance for method chaining 当前实例，支持链式调用
     */
    IPlot title(String titleText);

    /**
     * Set chart title and subtitle (Fluent API)
     * 设置图表标题和副标题（流式API）
     *
     * @param titleText Title text 标题文本
     * @param subtitleText Subtitle text 副标题文本
     * @return Current instance for method chaining 当前实例，支持链式调用
     */
    IPlot title(String titleText, String subtitleText);

    /**
     * Set X-axis label (Fluent API)
     * 设置X轴标签（流式API）
     *
     * @param name X-axis label name X轴标签名称
     * @return Current instance for method chaining 当前实例，支持链式调用
     */
    IPlot xlabel(String name);

    /**
     * Set Y-axis label (Fluent API)
     * 设置Y轴标签（流式API）
     *
     * @param name Y-axis label name Y轴标签名称
     * @return Current instance for method chaining 当前实例，支持链式调用
     */
    IPlot ylabel(String name);

    /**
     * Set chart size (Fluent API)
     * 设置图表尺寸（流式API）
     *
     * @param width Chart width 图表宽度
     * @param height Chart height 图表高度
     * @return Current instance for method chaining 当前实例，支持链式调用
     */
    IPlot size(int width, int height);

    /**
     * Set chart theme (Fluent API)
     * 设置图表主题（流式API）
     *
     * @param theme Theme name 主题名称
     * @return Current instance for method chaining 当前实例，支持链式调用
     */
    IPlot theme(String theme);

    /**
     * Display the chart (Fluent API)
     * 显示图表（流式API）
     *
     * @return Current instance for method chaining 当前实例，支持链式调用
     */
    IPlot show();

    /**
     * Save chart as HTML file (Fluent API)
     * 保存图表为HTML文件（流式API）
     *
     * @param filename File name 文件名
     * @return Current instance for method chaining 当前实例，支持链式调用
     */
    IPlot saveAsHtml(String filename);

    // ========== Utility Methods 工具方法 ==========
    /**
     * Get the HTML content of the chart
     * 获取图表的HTML内容
     *
     * @return HTML string HTML字符串
     */
    String toHtml();

    /**
     * Get the JSON configuration of the chart
     * 获取图表的JSON配置
     *
     * @return JSON string JSON字符串
     */
    String toJson();

    // ========== Configuration Methods 配置方法 ==========
    /**
     * Set chart title
     * 设置图表标题
     *
     * @param titleText Title text 标题文本
     */
    void setTitle(String titleText);

    /**
     * Set chart title and subtitle
     * 设置图表标题和副标题
     *
     * @param titleText Title text 标题文本
     * @param subtitleText Subtitle text 副标题文本
     */
    void setTitle(String titleText, String subtitleText);

    /**
     * Set X-axis label
     * 设置X轴标签
     *
     * @param name X-axis label name X轴标签名称
     */
    void setXlabel(String name);

    /**
     * Set Y-axis label
     * 设置Y轴标签
     *
     * @param name Y-axis label name Y轴标签名称
     */
    void setYlabel(String name);

    /**
     * Set X-axis ticks
     * 设置X轴刻度
     *
     * @param xticks X-axis ticks configuration X轴刻度配置
     */
    void setXticks(com.reremouse.lab.math.viz.AxisTicks xticks);

    /**
     * Set Y-axis ticks
     * 设置Y轴刻度
     *
     * @param yticks Y-axis ticks configuration Y轴刻度配置
     */
    void setYticks(com.reremouse.lab.math.viz.AxisTicks yticks);

    /**
     * Get chart width
     * 获取图表宽度
     *
     * @return Chart width 图表宽度
     */
    int getWidth();

    /**
     * Get chart height
     * 获取图表高度
     *
     * @return Chart height 图表高度
     */
    int getHeight();

    /**
     * Get chart theme
     * 获取图表主题
     *
     * @return Theme name 主题名称
     */
    String getTheme();
}