package com.yishape.lab.math.plot;

import com.yishape.lab.math.plot.echarts.EchartsThemeManager;
import java.util.List;
import java.util.Map;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import java.io.Serializable;

import java.util.ArrayList;

/**
 * Plotting interface that defines basic methods for plot instances
 * 绘图接口，定义绘图实例的基本方法
 *
 * @author lteb2
 */
public interface IPlot extends Serializable {

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
        List<String> ts = new ArrayList<>();
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

    // ========== PlotStyle / style-string overloads (both providers) ==========

    /**
     * Line chart with style string; same data semantics as {@link #line(IVector, IVector)}.
     * 线图，样式由字符串指定。
     *
     * @param x X-axis data
     * @param y Y-axis data
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot line(IVector x, IVector y, String styleString);

    /**
     * Line chart with {@link PlotStyle}; same data semantics as {@link #line(IVector, IVector)}.
     * 线图，结构化样式。
     *
     * @param x X-axis data
     * @param y Y-axis data
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot line(IVector x, IVector y, PlotStyle style);

    /**
     * Single-series line (index vs {@code y}) with style string; same as {@link #line(IVector)} plus style.
     *
     * @param y ordinate values
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot line(IVector y, String styleString);

    /**
     * Single-series line with {@link PlotStyle}; same as {@link #line(IVector)} plus style.
     *
     * @param y ordinate values
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot line(IVector y, PlotStyle style);

    /**
     * Multi-line chart with hue and per-point or per-series style grouping.
     * 多线 + 分组 + 样式分组。
     *
     * @param x X-axis data
     * @param y Y-axis data
     * @param hue group label per observation
     * @param styleGroup style bucket labels (backend-defined mapping)
     * @return this instance for chaining
     */
    IPlot line(IVector x, IVector y, List<String> hue, List<String> styleGroup);

    /**
     * Scatter with style string; same as {@link #scatter(IVector, IVector)} plus style.
     *
     * @param x X-axis data
     * @param y Y-axis data
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot scatter(IVector x, IVector y, String styleString);

    /**
     * Scatter with {@link PlotStyle}; same as {@link #scatter(IVector, IVector)} plus style.
     *
     * @param x X-axis data
     * @param y Y-axis data
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot scatter(IVector x, IVector y, PlotStyle style);

    /**
     * Pie chart with style string; same as {@link #pie(IVector)} plus style.
     *
     * @param x slice values
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot pie(IVector x, String styleString);

    /**
     * Pie chart with {@link PlotStyle}; same as {@link #pie(IVector)} plus style.
     *
     * @param x slice values
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot pie(IVector x, PlotStyle style);

    /**
     * Pie with slice labels and {@link PlotStyle}; same as label overload without style plus styling.
     *
     * @param x slice values
     * @param labels slice labels
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot pie(IVector x, List<String> labels, PlotStyle style);

    /**
     * Pie with slice labels and style string.
     *
     * @param x slice values
     * @param labels slice labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot pie(IVector x, List<String> labels, String styleString);

    /**
     * Bar chart (category index vs {@code y}) with style string; same as {@link #bar(IVector)} plus style.
     *
     * @param y bar heights
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot bar(IVector y, String styleString);

    /**
     * Bar chart with {@link PlotStyle}; same as {@link #bar(IVector)} plus style.
     *
     * @param y bar heights
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot bar(IVector y, PlotStyle style);

    /**
     * Grouped bar with {@link PlotStyle}; same as {@link #bar(IVector, List)} default-xticks overload plus style.
     *
     * @param y bar heights
     * @param hue group per observation
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot bar(IVector y, List<String> hue, PlotStyle style);

    /**
     * Grouped bar with style string; same as {@link #bar(IVector, List)} plus style.
     *
     * @param y bar heights
     * @param hue group per observation
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot bar(IVector y, List<String> hue, String styleString);

    /**
     * Histogram with style string; same as {@link #hist(IVector, boolean)} plus style.
     *
     * @param x sample data
     * @param fittingLine draw density/normal fit overlay if supported
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot hist(IVector x, boolean fittingLine, String styleString);

    /**
     * Histogram with {@link PlotStyle}; same as {@link #hist(IVector, boolean)} plus style.
     *
     * @param x sample data
     * @param fittingLine draw overlay if supported
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot hist(IVector x, boolean fittingLine, PlotStyle style);

    /**
     * Histogram with optional explicit bin count; {@code bins} may be {@code null} for backend default.
     *
     * @param x sample data
     * @param fittingLine draw overlay if supported
     * @param style structured style
     * @param bins number of bins, or {@code null} for automatic
     * @return this instance for chaining
     */
    IPlot hist(IVector x, boolean fittingLine, PlotStyle style, Integer bins);

    /**
     * Polar bar with style string; same as {@link #polarBar(IVector, List)} plus style.
     *
     * @param data radii or values per category
     * @param categories angular category labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot polarBar(IVector data, List<String> categories, String styleString);

    /**
     * Polar bar with {@link PlotStyle}; same as {@link #polarBar(IVector, List)} plus style.
     *
     * @param data radii or values per category
     * @param categories angular category labels
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot polarBar(IVector data, List<String> categories, PlotStyle style);

    /**
     * Polar line with style string; same as {@link #polarLine(IVector, List)} plus style.
     *
     * @param data values
     * @param categories angular labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot polarLine(IVector data, List<String> categories, String styleString);

    /**
     * Polar line with {@link PlotStyle}; same as {@link #polarLine(IVector, List)} plus style.
     *
     * @param data values
     * @param categories angular labels
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot polarLine(IVector data, List<String> categories, PlotStyle style);

    /**
     * Polar scatter with style string; same as {@link #polarScatter(IVector, List)} plus style.
     *
     * @param data values
     * @param categories angular labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot polarScatter(IVector data, List<String> categories, String styleString);

    /**
     * Polar scatter with {@link PlotStyle}; same as {@link #polarScatter(IVector, List)} plus style.
     *
     * @param data values
     * @param categories angular labels
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot polarScatter(IVector data, List<String> categories, PlotStyle style);

    /**
     * Box plot with style string; same as {@link #boxplot(IVector, List)} plus style.
     *
     * @param data sample
     * @param labels group labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot boxplot(IVector data, List<String> labels, String styleString);

    /**
     * Box plot with {@link PlotStyle}; same as {@link #boxplot(IVector, List)} plus style.
     *
     * @param data sample
     * @param labels group labels
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot boxplot(IVector data, List<String> labels, PlotStyle style);

    /**
     * Violin plot with style string; same as {@link #violinplot(IVector)} plus style.
     *
     * @param data sample
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot violinplot(IVector data, String styleString);

    /**
     * Violin plot with {@link PlotStyle}; same as {@link #violinplot(IVector)} plus style.
     *
     * @param data sample
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot violinplot(IVector data, PlotStyle style);

    /**
     * Violin plot with labels and style string; same as {@link #violinplot(IVector, List)} plus style.
     *
     * @param data sample
     * @param labels group labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot violinplot(IVector data, List<String> labels, String styleString);

    /**
     * Violin plot with labels and {@link PlotStyle}; same as {@link #violinplot(IVector, List)} plus style.
     *
     * @param data sample
     * @param labels group labels
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot violinplot(IVector data, List<String> labels, PlotStyle style);

    /**
     * Candlestick with style string; same as {@link #candlestick(IMatrix, List)} plus style.
     *
     * @param data OHLC rows
     * @param dates date labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot candlestick(IMatrix data, List<String> dates, String styleString);

    /**
     * Candlestick with {@link PlotStyle}; same as {@link #candlestick(IMatrix, List)} plus style.
     *
     * @param data OHLC rows
     * @param dates date labels
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot candlestick(IMatrix data, List<String> dates, PlotStyle style);

    /**
     * Funnel chart with style string; same as {@link #funnel(IVector, List)} plus style.
     *
     * @param data stage magnitudes
     * @param labels stage names
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot funnel(IVector data, List<String> labels, String styleString);

    /**
     * Funnel chart with {@link PlotStyle}; same as {@link #funnel(IVector, List)} plus style.
     *
     * @param data stage magnitudes
     * @param labels stage names
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot funnel(IVector data, List<String> labels, PlotStyle style);

    /**
     * Sankey diagram with style string; same as {@link #sankey(List, List)} plus style.
     *
     * @param nodes graph nodes as property maps
     * @param links graph links as property maps
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString);

    /**
     * Sankey diagram with {@link PlotStyle}.
     *
     * @param nodes graph nodes as property maps
     * @param links graph links as property maps
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style);

    /**
     * Sunburst with style string; same as {@link #sunburst(List)} plus style.
     *
     * @param data hierarchical records
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot sunburst(List<Map<String, Object>> data, String styleString);

    /**
     * Sunburst with {@link PlotStyle}.
     *
     * @param data hierarchical records
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot sunburst(List<Map<String, Object>> data, PlotStyle style);

    /**
     * Theme river with style string; same as {@link #themeRiver(List, List)} plus style.
     *
     * @param data multi-series stream records
     * @param categories series/category keys
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot themeRiver(List<Map<String, Object>> data, List<String> categories, String styleString);

    /**
     * Theme river with {@link PlotStyle}.
     *
     * @param data multi-series stream records
     * @param categories series/category keys
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot themeRiver(List<Map<String, Object>> data, List<String> categories, PlotStyle style);

    /**
     * Tree diagram with style string; same as {@link #tree(List)} plus style.
     *
     * @param data hierarchical tree payload
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot tree(List<Map<String, Object>> data, String styleString);

    /**
     * Tree diagram with {@link PlotStyle}.
     *
     * @param data hierarchical tree payload
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot tree(List<Map<String, Object>> data, PlotStyle style);

    /**
     * Treemap with style string; same as {@link #treemap(List)} plus style.
     *
     * @param data hierarchical records
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot treemap(List<Map<String, Object>> data, String styleString);

    /**
     * Treemap with {@link PlotStyle}.
     *
     * @param data hierarchical records
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot treemap(List<Map<String, Object>> data, PlotStyle style);

    /**
     * Relationship graph with style string; same as {@link #graph(List, List)} plus style.
     *
     * @param nodes node property maps
     * @param links link property maps
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString);

    /**
     * Relationship graph with {@link PlotStyle}.
     *
     * @param nodes node property maps
     * @param links link property maps
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style);

    /**
     * Parallel coordinates with style string; same as {@link #parallel(IMatrix, List)} plus style.
     *
     * @param data observation matrix
     * @param dimensions axis names
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot parallel(IMatrix data, List<String> dimensions, String styleString);

    /**
     * Parallel coordinates with {@link PlotStyle}.
     *
     * @param data observation matrix
     * @param dimensions axis names
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot parallel(IMatrix data, List<String> dimensions, PlotStyle style);

    /**
     * Heatmap with style string; same as {@link #heatmap(IMatrix, List, List)} plus style.
     *
     * @param data 2D matrix
     * @param xLabels column labels
     * @param yLabels row labels
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels, String styleString);

    /**
     * Heatmap with {@link PlotStyle}.
     *
     * @param data 2D matrix
     * @param xLabels column labels
     * @param yLabels row labels
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels, PlotStyle style);

    /**
     * Radar chart with style string; same as {@link #radar(IVector, List)} plus style.
     *
     * @param data axis values
     * @param indicators axis names
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot radar(IVector data, List<String> indicators, String styleString);

    /**
     * Radar chart with {@link PlotStyle}.
     *
     * @param data axis values
     * @param indicators axis names
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot radar(IVector data, List<String> indicators, PlotStyle style);

    /**
     * Gauge with style string; same as {@link #gauge(double, double, double)} plus style.
     *
     * @param value needle value
     * @param max scale maximum
     * @param min scale minimum
     * @param styleString backend-specific style tokens
     * @return this instance for chaining
     */
    IPlot gauge(double value, double max, double min, String styleString);

    /**
     * Gauge with {@link PlotStyle}.
     *
     * @param value needle value
     * @param max scale maximum
     * @param min scale minimum
     * @param style structured style
     * @return this instance for chaining
     */
    IPlot gauge(double value, double max, double min, PlotStyle style);

    /**
     * Default series style used when the style system is on and a layer omits explicit styling.
     * 默认序列样式（启用样式系统时作为未显式指定时的基底）。
     *
     * @param style structured default
     * @return this instance for chaining
     */
    IPlot setDefaultStyle(PlotStyle style);

    /**
     * Sets the active palette name (must be registered on {@link ColorPalette}).
     * 设置色板名称（须为 {@link ColorPalette} 已注册的名称）。
     *
     * @param paletteName registered palette key
     * @return this instance for chaining
     */
    IPlot setPalette(String paletteName);

    /**
     * Enables or disables the structured style system for subsequent layers.
     * 启用或关闭样式系统。
     *
     * @param enabled {@code true} to apply {@link PlotStyle} / palette resolution
     * @return this instance for chaining
     */
    IPlot enableStyleSystem(boolean enabled);

    /**
     * Enables or disables the theme system (built-in and registered themes).
     * 启用或关闭主题系统。
     *
     * @param enabled {@code true} to honor {@link #applyTheme(String)} and related hooks
     * @return this instance for chaining
     */
    IPlot enableThemeSystem(boolean enabled);

    /**
     * Applies a registered or built-in theme when the theme system is enabled.
     * 在启用主题系统时应用已注册/内置主题名。
     *
     * @param themeName theme key
     * @return this instance for chaining
     */
    IPlot applyTheme(String themeName);

    /**
     * Registers a custom ECharts-side theme object under {@code themeName}.
     *
     * @param themeName key for later {@link #applyTheme(String)}
     * @param theme custom theme definition
     * @return this instance for chaining
     */
    IPlot registerTheme(String themeName, EchartsThemeManager.CustomTheme theme);

    /**
     * Creates and registers a two-color gradient theme.
     *
     * @param themeName registration key
     * @param startColor start color (CSS/hex as accepted by backend)
     * @param endColor end color
     * @param backgroundColor plot background
     * @return this instance for chaining
     */
    IPlot createGradientTheme(String themeName, String startColor, String endColor, String backgroundColor);

    /**
     * Creates and registers a monochrome theme from a base color.
     *
     * @param themeName registration key
     * @param baseColor base accent color
     * @param backgroundColor plot background
     * @return this instance for chaining
     */
    IPlot createMonochromeTheme(String themeName, String baseColor, String backgroundColor);

    // ========== Seaborn / Matplotlib-style Cartesian extensions ==========
    /**
     * Set X-axis scale for subsequent Cartesian charts (line, scatter, area, step, regplot, etc.).
     * 设置后续笛卡尔图表的 X 轴比例（线性或对数）。
     */
    IPlot xscale(PlotAxisScale scale);

    /**
     * Set Y-axis scale for subsequent Cartesian charts.
     * 设置后续笛卡尔图表的 Y 轴比例。
     */
    IPlot yscale(PlotAxisScale scale);

    /**
     * Label for the secondary (right) Y axis when using {@link #lineWithSecondaryY}.
     * 双 Y 轴时右侧轴标题。
     */
    IPlot y2label(String label);

    /**
     * Filled area under a line (to y = baseline, default 0 mapped in renderer).
     * 面积图：折线与基线之间的填充。
     */
    IPlot area(IVector x, IVector y);

    /**
     * Step plot (post step: horizontal then vertical, like Matplotlib where="post").
     * 阶梯图（先水平后垂直）。
     */
    IPlot step(IVector x, IVector y);

    /**
     * Horizontal bar chart: categories on Y, values on X.
     * 条形图：类别在纵轴，数值在横轴。
     */
    IPlot barh(List<String> categories, IVector values);

    /**
     * Stacked bar chart: {@code values} has one row per stack layer and one column per category.
     * 堆叠柱：矩阵每行一层，每列一类别。
     */
    IPlot barStacked(List<String> categories, IMatrix values, List<String> layerNames);

    /**
     * Symmetric Y error bars at each (x, y).
     * 对称误差棒：每个点的竖线 y ± err。
     */
    IPlot errorbar(IVector x, IVector y, IVector yerr);

    /**
     * Bubble scatter: {@code sizes} are positive areas/radii scale factors (implementation maps to pixel radius).
     * 气泡图：sizes 映射为标记大小。
     */
    IPlot scatter(IVector x, IVector y, IVector sizes);

    /**
     * Scatter + ordinary least squares regression line (like seaborn regplot without CI band).
     * 散点 + 最小二乘回归线（不含置信带）。
     */
    IPlot regplot(IVector x, IVector y);

    /**
     * Like {@link #regplot(IVector, IVector)}; when {@code confidenceBand} is true, draw an approximate
     * 95% confidence band for the mean response (t on residual df, same x order as input).
     * 是否在回归线周围绘制均值响应的约 95% 置信带。
     */
    IPlot regplot(IVector x, IVector y, boolean confidenceBand);

    /**
     * Facet grid: divide subsequent Cartesian layers into {@code rows}×{@code cols} subplots.
     * Use {@link #subplot(int, int)} to choose the active cell (default is 0,0).
     * 子图网格：将后续笛卡尔图层分到多格；用 subplot 选择当前格。
     */
    IPlot subplots(int rows, int cols);

    /**
     * Select active facet cell (0-based). Must call {@link #subplots(int, int)} first.
     * 选择当前子图格子。
     */
    IPlot subplot(int row, int col);

    /**
     * Univariate kernel density estimate as a density curve.
     * 一维 KDE 密度曲线。
     */
    default IPlot kdeplot(IVector data) {
        return kdeplot(data, 256, 0.0);
    }

    /**
     * KDE with grid size and bandwidth; {@code bandwidth == 0} uses Scott's rule.
     *
     * @param data univariate sample
     * @param gridPoints evaluation grid resolution
     * @param bandwidth kernel bandwidth (0 ⇒ automatic)
     * @return this instance for chaining
     */
    IPlot kdeplot(IVector data, int gridPoints, double bandwidth);

    /**
     * Pair grid: column {@code j} vs row {@code i} (seaborn-style axes).
     * The diagonal uses {@code diagonal} style; off-diagonal are scatter plots.
     */
    default IPlot pairplot(IMatrix data) {
        return pairplot(data, null, PairplotDiagonal.KDE);
    }

    /**
     * Pair plot grid: off-diagonal cells are scatter plots; diagonal uses {@code diagonal}.
     * 配对图网格：非对角为散点，对角线样式由 {@code diagonal} 决定。
     *
     * @param data matrix whose columns are variables
     * @param columnNames optional names per column ({@code null} for defaults)
     * @param diagonal marginal visualization on diagonal cells
     * @return this instance for chaining
     */
    IPlot pairplot(IMatrix data, List<String> columnNames, PairplotDiagonal diagonal);

    /**
     * Scatter of {@code x} vs {@code y} with marginal distributions (KDE or histogram).
     *
     * @param x horizontal sample
     * @param y vertical sample
     * @param marginal marginal plot type on top/right
     * @return this instance for chaining
     */
    IPlot jointplot(IVector x, IVector y, JointplotMarginal marginal);

    /**
     * Q–Q plot against standard normal (ordered sample vs theoretical quantiles).
     * 正态 Q-Q 图。
     *
     * @param data univariate sample
     * @return this instance for chaining
     */
    IPlot qqplot(IVector data);

    /**
     * Two Y axes: left series {@code yLeft}, right series {@code yRight} vs common {@code x}.
     * 双 Y 轴：共享 X，左右各一条序列。
     */
    IPlot lineWithSecondaryY(IVector x, IVector yLeft, IVector yRight);

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
     * @return IPlot instance 当前绘图实例（JavaFX 与 ECharts 实现均返回可链式调用的 this）
     */
    IPlot heatmap(IMatrix data);

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

    /**
     * Persists the current figure as SVG (backend-dependent; may rasterize layered charts).
     * 保存为 SVG（依赖后端实现；复杂图层可能被栅格化）。
     *
     * @param filename output path
     * @return this instance for chaining
     */
    IPlot saveAsSvg(String filename);

    /**
     * Persists the current figure as PNG.
     * 保存为 PNG 位图。
     *
     * @param filename output path
     * @return this instance for chaining
     */
    IPlot saveAsPng(String filename);

    /**
     * Returns a Base64-encoded SVG payload (typically a data-URL-ready string without the {@code data:} prefix;
     * exact format depends on the provider).
     * Base64 编码的 SVG 字符串。
     *
     * @return Base64 SVG
     */
    String toBase64Svg();

    /**
     * Returns a Base64-encoded PNG payload.
     * Base64 编码的 PNG 字符串。
     *
     * @return Base64 PNG
     */
    String toBase64Png();

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
    void setXticks(AxisTicks xticks);

    /**
     * Set Y-axis ticks
     * 设置Y轴刻度
     *
     * @param yticks Y-axis ticks configuration Y轴刻度配置
     */
    void setYticks(AxisTicks yticks);

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