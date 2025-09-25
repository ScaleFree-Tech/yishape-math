package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import java.awt.Desktop;
import java.io.Serializable;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.icepear.echarts.Bar;
import org.icepear.echarts.Boxplot;
import org.icepear.echarts.Candlestick;
import org.icepear.echarts.Funnel;
import org.icepear.echarts.Gauge;
import org.icepear.echarts.Graph;
import org.icepear.echarts.Heatmap;
import org.icepear.echarts.Line;
import org.icepear.echarts.Option;
import org.icepear.echarts.Parallel;
import org.icepear.echarts.Pie;
import org.icepear.echarts.PolarBar;
import org.icepear.echarts.PolarLine;
import org.icepear.echarts.PolarScatter;
import org.icepear.echarts.Radar;
import org.icepear.echarts.Sankey;
import org.icepear.echarts.Scatter;
import org.icepear.echarts.Sunburst;
import org.icepear.echarts.ThemeRiver;
import org.icepear.echarts.Tree;
import org.icepear.echarts.Treemap;
import org.icepear.echarts.charts.bar.BarSeries;
import org.icepear.echarts.charts.line.LineSeries;
import org.icepear.echarts.charts.scatter.ScatterSeries;
import org.icepear.echarts.components.coord.cartesian.CategoryAxis;
import org.icepear.echarts.components.coord.cartesian.ValueAxis;
import org.icepear.echarts.components.legend.Legend;
import org.icepear.echarts.components.title.Title;
import org.icepear.echarts.components.tooltip.Tooltip;
import org.icepear.echarts.render.Engine;

/**
 * 结合ECharts-Java实现基本的画图功能 ECharts-Java的所有类文件都在org.icepear包下面
 * ECharts-Java文档地址：https://echarts.icepear.org/#/quick-start
 *
 * @author lteb2
 */
public class RerePlot implements Serializable, IPlot {

    private Title title;
    private Legend legend;
    private Tooltip tooltip;
    private AxisTicks xticks;
    private AxisTicks yticks;
    private String xlabel;
    private String ylabel;
    private Engine engine;
    private Option option;
    private int width;
    private int height;
    private String theme;

    // ========== 样式系统 ==========
    private PlotStyle defaultStyle;        // 默认样式
    private String currentPalette;         // 当前调色板
    private boolean useStyleSystem;        // 是否启用新样式系统
    private SeabornStyleMapper styleMapper; // seaborn风格样式映射器

    // ========== 主题系统 ==========
    private String currentTheme;           // 当前主题
    private boolean useThemeSystem;        // 是否启用主题系统

    /**
     * 默认构造函数
     */
    public RerePlot() {
        this.title = new Title();
        this.legend = new Legend();
        this.tooltip = new Tooltip();
        this.xticks = new AxisTicks();
        this.yticks = new AxisTicks();
        this.xlabel = "";
        this.ylabel = "";
        try {
            this.engine = new Engine();
        } catch (Exception e) {
            this.engine = null;
            System.err.println("警告：无法初始化 ECharts Engine，图表渲染功能将不可用");
        }
        this.option = new Option();
        this.width = 800;
        this.height = 600;
        this.theme = "default";

        // 初始化样式系统
        this.defaultStyle = PlotStyle.defaultStyle();
        this.currentPalette = "echarts";
        this.useStyleSystem = true;
        this.styleMapper = new SeabornStyleMapper();

        // 初始化主题系统
        this.currentTheme = ThemeManager.THEME_DEFAULT;
        this.useThemeSystem = true;
    }

    /**
     * 带尺寸的构造函数
     *
     * @param width 图表宽度
     * @param height 图表高度
     */
    public RerePlot(int width, int height) {
        this();
        this.width = width;
        this.height = height;
    }

    /**
     * 带主题的构造函数
     *
     * @param width 图表宽度
     * @param height 图表高度
     * @param theme 主题名称
     */
    public RerePlot(int width, int height, String theme) {
        this(width, height);
        this.theme = theme;
    }

    // ========== 统一的线图方法 ==========
    /**
     * 创建线图（基础版本）
     *
     * @param x X轴数据
     * @param y Y轴数据
     * @return 当前实例
     */
    public RerePlot line(IVector x, IVector y) {
        return lineInternal(x, y, null, null, null);
    }

    /**
     * 创建线图（支持样式字符串）
     *
     * @param x X轴数据
     * @param y Y轴数据
     * @param styleString 样式字符串（如 "r-", "b--o", "g:^"）
     * @return 当前实例
     */
    public RerePlot line(IVector x, IVector y, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return lineInternal(x, y, style, null, null);
    }

    /**
     * 创建线图（支持PlotStyle样式）
     *
     * @param x X轴数据
     * @param y Y轴数据
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot line(IVector x, IVector y, PlotStyle style) {
        return lineInternal(x, y, style, null, null);
    }

    /**
     * 创建单向量线图（使用索引作为X轴）
     *
     * @param y Y轴数据
     * @return 当前实例
     */
    public RerePlot line(IVector y) {
        return lineInternal(null, y, null, null, null);
    }

    /**
     * 创建单向量线图（支持样式）
     *
     * @param y Y轴数据
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot line(IVector y, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return lineInternal(null, y, style, null, null);
    }

    /**
     * 创建单向量线图（支持PlotStyle）
     *
     * @param y Y轴数据
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot line(IVector y, PlotStyle style) {
        return lineInternal(null, y, style, null, null);
    }

    /**
     * 创建分组线图（seaborn风格）
     *
     * @param x X轴数据
     * @param y Y轴数据
     * @param hue 颜色分组
     * @return 当前实例
     */
    public RerePlot line(IVector x, IVector y, List<String> hue) {
        return lineInternal(x, y, null, hue, null);
    }

    /**
     * 创建分组线图（seaborn风格，支持样式和标记分组）
     *
     * @param x X轴数据
     * @param y Y轴数据
     * @param hue 颜色分组
     * @param style 线条样式分组
     * @return 当前实例
     */
    public RerePlot line(IVector x, IVector y, List<String> hue, List<String> style) {
        return lineInternal(x, y, null, hue, style);
    }

    /**
     * 统一的线图内部实现
     *
     * @param x X轴数据（如果为null，使用索引）
     * @param y Y轴数据
     * @param style 个体样式（优先级最高）
     * @param hue 颜色分组
     * @param styleGroup 样式分组
     * @return 当前实例
     */
    private RerePlot lineInternal(IVector x, IVector y, PlotStyle style,
            List<String> hue, List<String> styleGroup) {
        try {
            Line lineChart = new Line();

            // 处理单向量情况（使用索引作X轴）
            IVector xData = x;
            if (x == null) {
                double[] indices = new double[y.length()];
                for (int i = 0; i < y.length(); i++) {
                    indices[i] = i;
                }
                xData = Linalg.vector(indices);
            }

            // 处理分组情况
            if (hue != null || styleGroup != null) {
                return createGroupedLineChart(lineChart, xData, y, style, hue, styleGroup);
            } else {
                return createSingleLineChart(lineChart, xData, y, style);
            }

        } catch (Exception e) {
            throw new PlotException("创建线图失败: " + e.getMessage(), e);
        }
    }

    /**
     * 创建单条线图
     */
    private RerePlot createSingleLineChart(Line lineChart, IVector x, IVector y, PlotStyle style) {
        Object[] data = vectorToNumber(x, y);
        LineSeries lineSeries = new LineSeries();
        lineSeries.setData(data);

        // 应用样式
        PlotStyle effectiveStyle = style != null ? style
                : (useStyleSystem ? defaultStyle : null);

        if (effectiveStyle != null && useStyleSystem) {
            applyStyleToLineSeries(lineSeries, effectiveStyle);
        } else {
            lineSeries.setName("数据线");
        }

        lineChart.addSeries(lineSeries);

        // 配置图表
        setupChartOptions(lineChart);
        return this;
    }

    /**
     * 创建分组线图（seaborn风格）
     */
    private RerePlot createGroupedLineChart(Line lineChart, IVector x, IVector y,
            PlotStyle baseStyle, List<String> hue,
            List<String> styleGroup) {
        if (x.length() != y.length()
                || (hue != null && x.length() != hue.size())
                || (styleGroup != null && x.length() != styleGroup.size())) {
            throw new PlotException("数据和分组标签长度不一致");
        }

        // 创建样式映射
        SeabornStyleMapper.GroupStyleMapping mapping = styleMapper.createMapping(
                hue, styleGroup, null, null);

        // 分组数据
        Map<String, SeabornStyleMapper.GroupedData> groups
                = styleMapper.groupData(x, y, hue, mapping);

        // 为每个组创建系列
        for (SeabornStyleMapper.GroupedData group : groups.values()) {
            LineSeries lineSeries = new LineSeries();
            lineSeries.setData(group.getData());

            // 应用组样式
            PlotStyle groupStyle = group.getStyle();
            if (baseStyle != null) {
                // 如果有基础样式，合并样式
                groupStyle = mergeStyles(baseStyle, groupStyle);
            }

            if (useStyleSystem) {
                applyStyleToLineSeries(lineSeries, groupStyle);
            } else {
                lineSeries.setName(group.getGroupName());
            }

            lineChart.addSeries(lineSeries);
        }

        setupChartOptions(lineChart);
        return this;
    }

    // ========== 辅助方法 ==========
    /**
     * 设置图表通用选项
     */
    private void setupChartOptions(Line lineChart) {
        ValueAxis xAxis = new ValueAxis();
        ValueAxis yAxis = new ValueAxis();
        configureAxes(xAxis, yAxis);

        this.option = lineChart.getOption();
        setCommonOptions(this.option);
        this.option.setXAxis(xAxis);
        this.option.setYAxis(yAxis);
    }

    /**
     * 合并两个样式对象
     *
     * @param base 基础样式
     * @param overlay 覆盖样式
     * @return 合并后的样式
     */
    private PlotStyle mergeStyles(PlotStyle base, PlotStyle overlay) {
        PlotStyle merged = new PlotStyle(base);

        // 覆盖非空属性
        if (overlay.getColor() != null && !overlay.getColor().equals(base.getColor())) {
            merged.setColor(overlay.getColor());
        }
        if (overlay.getLineStyle() != null && !overlay.getLineStyle().equals(base.getLineStyle())) {
            merged.setLineStyle(overlay.getLineStyle());
        }
        if (overlay.getMarker() != null && !overlay.getMarker().equals(base.getMarker())) {
            merged.setMarker(overlay.getMarker());
        }
        if (overlay.getLabel() != null && !overlay.getLabel().isEmpty()) {
            merged.setLabel(overlay.getLabel());
        }

        return merged;
    }

    /**
     * 创建散点图（支持样式字符串）
     *
     * @param x X轴数据
     * @param y Y轴数据
     * @param styleString 样式字符串
     * @return 当前实例，支持链式调用
     */
    public RerePlot scatter(IVector x, IVector y, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return scatterWithStyle(x, y, style);
    }

    /**
     * 创建散点图（支持PlotStyle样式）
     *
     * @param x X轴数据
     * @param y Y轴数据
     * @param style 绘图样式
     * @return 当前实例，支持链式调用
     */
    public RerePlot scatter(IVector x, IVector y, PlotStyle style) {
        return scatterWithStyle(x, y, style);
    }

    /**
     * 创建散点图的内部实现
     */
    private RerePlot scatterWithStyle(IVector x, IVector y, PlotStyle style) {
        try {
            // 创建散点图
            Scatter scatterChart = new Scatter();

            // 设置数据
            Object[] data = vectorToNumber(x, y);
            ScatterSeries scatterSeries = new ScatterSeries();
            scatterSeries.setData(data);

            // 应用样式
            if (style != null && useStyleSystem) {
                applyStyleToScatterSeries(scatterSeries, style);
            } else {
                scatterSeries.setName("散点数据");
            }

            scatterChart.addSeries(scatterSeries);

            // 配置坐标轴
            ValueAxis xAxis = new ValueAxis();
            ValueAxis yAxis = new ValueAxis();
            configureAxes(xAxis, yAxis);

            this.option = scatterChart.getOption();
            setCommonOptions(this.option);
            this.option.setXAxis(xAxis);
            this.option.setYAxis(yAxis);

        } catch (Exception e) {
            throw new PlotException("创建散点图失败: " + e.getMessage(), e);
        }
        return this;
    }

    /**
     * 创建散点图（原有方法，保持向后兼容）
     */
    public RerePlot scatter(IVector x, IVector y) {
        try {
            // 创建散点图
            Scatter scatterChart = new Scatter();

            // 设置数据
            Object[] data = vectorToNumber(x, y);
            ScatterSeries scatterSeries = new ScatterSeries();
            scatterSeries.setData(data);
            scatterSeries.setName("散点数据");
            scatterChart.addSeries(scatterSeries);

            // 配置坐标轴
            ValueAxis xAxis = new ValueAxis();
            ValueAxis yAxis = new ValueAxis();
            configureAxes(xAxis, yAxis);

            this.option = scatterChart.getOption();
            setCommonOptions(this.option);
            this.option.setXAxis(xAxis);
            this.option.setYAxis(yAxis);

        } catch (Exception e) {
            throw new PlotException("创建散点图失败: " + e.getMessage(), e);
        }
        return this;
    }

    public RerePlot scatter(IVector x, IVector y, List<String> hue) {
        try {
            // 创建多组散点图
            Scatter scatterChart = new Scatter();

            if (x.length() != y.length() || x.length() != hue.size()) {
                throw new PlotException("X、Y向量和hue列表长度必须相等");
            }

            // 按hue分组数据
            Map<String, List<Object[]>> groupedData = new HashMap<>();
            for (int i = 0; i < x.length(); i++) {
                String groupName = hue.get(i);
                if (!groupedData.containsKey(groupName)) {
                    groupedData.put(groupName, new ArrayList<>());
                }
                groupedData.get(groupName).add(new Number[]{x.get(i), y.get(i)});
            }

            // 为每个组创建散点系列
            for (Map.Entry<String, List<Object[]>> entry : groupedData.entrySet()) {
                ScatterSeries scatterSeries = new ScatterSeries();
                scatterSeries.setData(entry.getValue().toArray(new Object[0]));
                scatterSeries.setName(entry.getKey());
                scatterChart.addSeries(scatterSeries);
            }

            // 配置坐标轴
            ValueAxis xAxis = new ValueAxis();
            ValueAxis yAxis = new ValueAxis();
            configureAxes(xAxis, yAxis);

            this.option = scatterChart.getOption();
            setCommonOptions(this.option);
            this.option.setXAxis(xAxis);
            this.option.setYAxis(yAxis);

        } catch (Exception e) {
            throw new PlotException("创建多组散点图失败: " + e.getMessage(), e);
        }
        return this;
    }

    @Override
    public RerePlot pie(IVector x) {
        return pieInternal(x, null, null);
    }

    /**
     * 创建饼图（支持样式字符串）
     *
     * @param x 数据向量
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot pie(IVector x, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return pieInternal(x, style, null);
    }

    /**
     * 创建饼图（支持PlotStyle样式）
     *
     * @param x 数据向量
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot pie(IVector x, PlotStyle style) {
        return pieInternal(x, style, null);
    }

    /**
     * 创建饼图（支持标签）
     *
     * @param x 数据向量
     * @param labels 标签列表
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot pie(IVector x, List<String> labels, PlotStyle style) {
        return pieInternal(x, style, labels);
    }

    /**
     * 创建饼图（支持标签和样式字符串）
     *
     * @param x 数据向量
     * @param labels 标签列表
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot pie(IVector x, List<String> labels, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return pieInternal(x, style, labels);
    }

    /**
     * 统一的饼图内部实现
     *
     * @param x 数据向量
     * @param style 绘图样式
     * @param labels 标签列表
     * @return 当前实例
     */
    private RerePlot pieInternal(IVector x, PlotStyle style, List<String> labels) {
        try {
            // 创建饼图
            Pie pieChart = new Pie();

            // 转换数据格式 - 饼图需要对象数组格式 {name, value}
            org.icepear.echarts.charts.pie.PieDataItem[] pieData = new org.icepear.echarts.charts.pie.PieDataItem[x.length()];
            for (int i = 0; i < x.length(); i++) {
                String name = (labels != null && i < labels.size()) ? labels.get(i) : "数据" + (i + 1);
                pieData[i] = new org.icepear.echarts.charts.pie.PieDataItem()
                        .setName(name)
                        .setValue(x.get(i));
            }

            // 创建PieSeries并正确设置data
            org.icepear.echarts.charts.pie.PieSeries series = new org.icepear.echarts.charts.pie.PieSeries()
                    .setType("pie")
                    .setData(pieData);

            // 应用样式
            PlotStyle effectiveStyle = style != null ? style
                    : (useStyleSystem ? defaultStyle : null);

            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToPieSeries(series, effectiveStyle);
            } else {
                series.setName("饼图数据");
            }

            this.option = pieChart.getOption();
            setCommonOptions(this.option);
            this.option.setSeries(new org.icepear.echarts.charts.pie.PieSeries[]{series});

        } catch (Exception e) {
            throw new PlotException("创建饼图失败: " + e.getMessage(), e);
        }
        return this;
    }

    @Override
    public RerePlot bar(IVector y) {
        return barInternal(y, null, null, null);
    }

    @Override
    public IPlot bar(List<String> xticks, IVector y) {
        return barInternalWithLabels(y, null, xticks, null);
    }

    @Override
    public IPlot bar(List<String> xticks, IVector y, List<String> hue) {
        return barInternalWithLabels(y, null, xticks, hue);
    }

    
    /**
     * 创建柱状图（支持样式字符串）
     *
     * @param x 数据向量
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot bar(IVector x, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return barInternal(x, style, null, null);
    }

    /**
     * 创建柱状图（支持PlotStyle样式）
     *
     * @param x 数据向量
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot bar(IVector x, PlotStyle style) {
        return barInternal(x, style, null, null);
    }

    /**
     * 创建分组柱状图（支持样式）
     *
     * @param x 数据向量
     * @param hue 分组标签
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot bar(IVector x, List<String> hue, PlotStyle style) {
        return barInternal(x, style, hue, null);
    }

    /**
     * 创建分组柱状图（支持样式字符串）
     *
     * @param x 数据向量
     * @param hue 分组标签
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot bar(IVector x, List<String> hue, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return barInternal(x, style, hue, null);
    }

    /**
     * 统一的带标签柱状图内部实现
     *
     * @param y 数据向量
     * @param style 个体样式
     * @param xticks X轴标签
     * @param hue 颜色分组
     * @return 当前实例
     */
    private RerePlot barInternalWithLabels(IVector y, PlotStyle style, List<String> xticks, List<String> hue) {
        try {
            // 创建柱状图
            Bar barChart = new Bar();

            // 处理分组情况
            if (hue != null) {
                return createGroupedBarChartWithLabels(barChart, y, style, xticks, hue, null);
            } else {
                return createSingleBarChartWithLabels(barChart, y, style, xticks);
            }

        } catch (Exception e) {
            throw new PlotException("创建柱状图失败: " + e.getMessage(), e);
        }
    }

    /**
     * 统一的柱状图内部实现
     *
     * @param x 数据向量
     * @param style 个体样式
     * @param hue 颜色分组
     * @param styleGroup 样式分组
     * @return 当前实例
     */
    private RerePlot barInternal(IVector x, PlotStyle style, List<String> hue, List<String> styleGroup) {
        try {
            // 创建柱状图
            Bar barChart = new Bar();

            // 处理分组情况
            if (hue != null) {
                return createGroupedBarChart(barChart, x, style, hue, styleGroup);
            } else {
                return createSingleBarChart(barChart, x, style);
            }

        } catch (Exception e) {
            throw new PlotException("创建柱状图失败: " + e.getMessage(), e);
        }
    }

    /**
     * 创建单组柱状图
     */
    private RerePlot createSingleBarChart(Bar barChart, IVector x, PlotStyle style) {
        // 设置数据 - 只包含Y值
        Object[] data = new Object[x.length()];
        for (int i = 0; i < x.length(); i++) {
            data[i] = x.get(i);
        }

        BarSeries barSeries = new BarSeries();
        barSeries.setData(data);

        // 应用样式
        PlotStyle effectiveStyle = style != null ? style
                : (useStyleSystem ? defaultStyle : null);

        if (effectiveStyle != null && useStyleSystem) {
            UniversalStyleApplier.applyToBarSeries(barSeries, effectiveStyle);
        } else {
            barSeries.setName("柱状图数据");
        }

        barChart.addSeries(barSeries);

        // 配置坐标轴
        CategoryAxis xAxis = new CategoryAxis();
        String[] categories = new String[x.length()];
        for (int i = 0; i < x.length(); i++) {
            categories[i] = (xticks.hasTickLabels() && i < xticks.getTickLabels().size()) ? 
                xticks.getTickLabels().get(i) : "类别" + (i + 1);
        }
        xAxis.setData(categories);

        ValueAxis yAxis = new ValueAxis();
        configureCategoryAxes(xAxis, yAxis);

        this.option = barChart.getOption();
        setCommonOptions(this.option);
        this.option.setXAxis(xAxis);
        this.option.setYAxis(yAxis);

        return this;
    }

    /**
     * 创建分组柱状图
     */
    private RerePlot createGroupedBarChart(Bar barChart, IVector x, PlotStyle baseStyle,
            List<String> hue, List<String> styleGroup) {
        if (x.length() != hue.size()) {
            throw new PlotException("X向量和hue列表长度必须相等");
        }

        // 创建样式映射
        SeabornStyleMapper.GroupStyleMapping mapping = styleMapper.createMapping(
                hue, styleGroup, null, null);

        // 分组数据
        Map<String, SeabornStyleMapper.GroupedData> groups
                = styleMapper.groupData(x, null, hue, mapping);

        // 为每个组创建系列
        for (SeabornStyleMapper.GroupedData group : groups.values()) {
            BarSeries barSeries = new BarSeries();

            // 转换数据格式 - 只包含Y值
            Object[] data = new Object[group.getData().length];
            for (int i = 0; i < group.getData().length; i++) {
                data[i] = group.getData()[i];
            }
            barSeries.setData(data);

            // 应用组样式
            PlotStyle groupStyle = group.getStyle();
            if (baseStyle != null) {
                // 如果有基础样式，合并样式
                groupStyle = mergeStyles(baseStyle, groupStyle);
            }

            if (useStyleSystem) {
                UniversalStyleApplier.applyToBarSeries(barSeries, groupStyle);
            } else {
                barSeries.setName(group.getGroupName());
            }

            barChart.addSeries(barSeries);
        }

        // 配置坐标轴
        CategoryAxis xAxis = new CategoryAxis();
        String[] categories = new String[x.length()];
        for (int i = 0; i < x.length(); i++) {
            categories[i] = (xticks.hasTickLabels() && i < xticks.getTickLabels().size()) ? 
                xticks.getTickLabels().get(i) : "类别" + (i + 1);
        }
        xAxis.setData(categories);

        ValueAxis yAxis = new ValueAxis();
        configureCategoryAxes(xAxis, yAxis);

        this.option = barChart.getOption();
        setCommonOptions(this.option);
        this.option.setXAxis(xAxis);
        this.option.setYAxis(yAxis);

        return this;
    }

    /**
     * 创建带标签的单组柱状图
     */
    private RerePlot createSingleBarChartWithLabels(Bar barChart, IVector x, PlotStyle style, List<String> xticks) {
        // 设置数据 - 只包含Y值
        Object[] data = new Object[x.length()];
        for (int i = 0; i < x.length(); i++) {
            data[i] = x.get(i);
        }

        BarSeries barSeries = new BarSeries();
        barSeries.setData(data);

        // 应用样式
        PlotStyle effectiveStyle = style != null ? style
                : (useStyleSystem ? defaultStyle : null);

        if (effectiveStyle != null && useStyleSystem) {
            UniversalStyleApplier.applyToBarSeries(barSeries, effectiveStyle);
        } else {
            barSeries.setName("柱状图数据");
        }

        barChart.addSeries(barSeries);

        // 配置坐标轴
        CategoryAxis xAxis = new CategoryAxis();
        String[] categories = new String[x.length()];
        for (int i = 0; i < x.length(); i++) {
            categories[i] = (xticks != null && i < xticks.size()) ? xticks.get(i) : "类别" + (i + 1);
        }
        xAxis.setData(categories);

        ValueAxis yAxis = new ValueAxis();
        configureCategoryAxes(xAxis, yAxis);

        this.option = barChart.getOption();
        setCommonOptions(this.option);
        this.option.setXAxis(xAxis);
        this.option.setYAxis(yAxis);

        return this;
    }

    /**
     * 创建带标签的分组柱状图
     */
    private RerePlot createGroupedBarChartWithLabels(Bar barChart, IVector y, PlotStyle baseStyle,
            List<String> xticks, List<String> hue, List<String> styleGroup) {
        if (y.length() != hue.size()) {
            throw new PlotException("X向量和hue列表长度必须相等");
        }

        // Group the data by hue values
        Map<String, List<Double>> groupedData = new LinkedHashMap<>();
        Map<String, List<Integer>> groupedIndices = new LinkedHashMap<>();
        
        // Initialize groups
        Set<String> uniqueHues = new LinkedHashSet<>(hue);
        for (String group : uniqueHues) {
            groupedData.put(group, new ArrayList<>());
            groupedIndices.put(group, new ArrayList<>());
        }
        
        // Group the data and keep track of indices
        for (int i = 0; i < y.length(); i++) {
            String group = hue.get(i);
            groupedData.get(group).add((double)y.get(i));
            groupedIndices.get(group).add(i);
        }
        
        // For bar charts, we need to create series data aligned with the x-axis labels
        // Each series needs to have the same length as the number of x-axis labels
        List<String> allLabels = xticks;
        
        // Create series for each group
        for (Map.Entry<String, List<Double>> entry : groupedData.entrySet()) {
            String groupName = entry.getKey();
            List<Double> values = entry.getValue();
            List<Integer> indices = groupedIndices.get(groupName);
            
            BarSeries barSeries = new BarSeries();
            barSeries.setName(groupName);
            
            // Create data array with nulls for missing values
            Object[] seriesData = new Object[allLabels.size()];
            for (int i = 0; i < allLabels.size(); i++) {
                seriesData[i] = null; // Default to null
            }
            
            // Fill in the actual values
            for (int i = 0; i < indices.size(); i++) {
                int index = indices.get(i);
                if (index < seriesData.length) {
                    seriesData[index] = values.get(i);
                }
            }
            
            barSeries.setData(seriesData);

            // Apply style
            if (baseStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToBarSeries(barSeries, baseStyle);
            }

            barChart.addSeries(barSeries);
        }

        // 配置坐标轴
        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setData(allLabels.toArray(new String[0]));

        ValueAxis yAxis = new ValueAxis();
        configureCategoryAxes(xAxis, yAxis);

        this.option = barChart.getOption();
        setCommonOptions(this.option);
        this.option.setXAxis(xAxis);
        this.option.setYAxis(yAxis);

        return this;
    }

    public RerePlot bar(IVector x, List<String> hue) {
        return barInternal(x, null, hue, null);
    }

    public RerePlot hist(IVector x, boolean fittingLine) {
        return histInternal(x, fittingLine, null, null);
    }

    /**
     * 创建直方图（支持样式字符串）
     *
     * @param x 数据向量
     * @param fittingLine 是否显示拟合线
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot hist(IVector x, boolean fittingLine, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return histInternal(x, fittingLine, style, null);
    }

    /**
     * 创建直方图（支持PlotStyle样式）
     *
     * @param x 数据向量
     * @param fittingLine 是否显示拟合线
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot hist(IVector x, boolean fittingLine, PlotStyle style) {
        return histInternal(x, fittingLine, style, null);
    }

    /**
     * 创建直方图（支持样式和bins参数）
     *
     * @param x 数据向量
     * @param fittingLine 是否显示拟合线
     * @param style 绘图样式
     * @param bins bin数量
     * @return 当前实例
     */
    public RerePlot hist(IVector x, boolean fittingLine, PlotStyle style, Integer bins) {
        return histInternal(x, fittingLine, style, bins);
    }

    /**
     * 统一的直方图内部实现
     *
     * @param x 数据向量
     * @param fittingLine 是否显示拟合线
     * @param style 绘图样式
     * @param bins bin数量
     * @return 当前实例
     */
    private RerePlot histInternal(IVector x, boolean fittingLine, PlotStyle style, Integer bins) {
        try {
            // 计算直方图数据
            int binCount = (bins != null) ? bins : Math.min(20, (int) Math.sqrt(x.length()));
            double min = (double) x.min();
            double max = (double) x.max();

            // 确保bin宽度不为0
            if (max == min) {
                max = min + 1.0f;
            }

            double binWidth = (max - min) / binCount;

            // 创建直方图数据
            int[] counts = new int[binCount];
            double[] binCenters = new double[binCount];

            for (int i = 0; i < binCount; i++) {
                binCenters[i] = min + (i + 0.5) * binWidth;
            }

            // 统计每个bin的数据点数量
            for (int i = 0; i < x.length(); i++) {
                double value = (double) x.get(i);
                int binIndex = Math.min((int) ((value - min) / binWidth), binCount - 1);
                if (binIndex >= 0 && binIndex < binCount) {
                    counts[binIndex]++;
                }
            }

            // 创建柱状图
            Bar barChart = new Bar();

            // 设置数据
            Object[] data = new Object[binCount];
            for (int i = 0; i < binCount; i++) {
                data[i] = new Object[]{String.format("%.2f", binCenters[i]), counts[i]};
            }

            BarSeries barSeries = new BarSeries();
            barSeries.setData(data);

            // 应用样式
            PlotStyle effectiveStyle = style != null ? style
                    : (useStyleSystem ? defaultStyle : null);

            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToBarSeries(barSeries, effectiveStyle);
            } else {
                barSeries.setName("直方图");
                // 设置默认颜色
                barSeries.setColor("#91cc75");  // 清新的绿色
            }

            barChart.addSeries(barSeries);

            // 配置坐标轴 - 使用ValueAxis而不是CategoryAxis以支持拟合线
            ValueAxis xAxis = new ValueAxis();
            xAxis.setName(xlabel.isEmpty() ? "数值区间" : xlabel);
            xAxis.setType("value");

            ValueAxis yAxis = new ValueAxis();
            yAxis.setName(ylabel.isEmpty() ? "频次" : ylabel);
            yAxis.setType("value");

            this.option = barChart.getOption();
            setCommonOptions(this.option);
            this.option.setXAxis(xAxis);
            this.option.setYAxis(yAxis);

            // 如果需要拟合线，添加核密度估计曲线
            if (fittingLine) {
                addKernelDensityFit(x, min, max, binCount, binCenters, counts);
            }

        } catch (Exception e) {
            throw new PlotException("创建直方图失败: " + e.getMessage(), e);
        }
        return this;
    }

    /**
     * 添加核密度估计拟合线
     *
     * @param x 原始数据
     * @param min 最小值
     * @param max 最大值
     * @param bins bin数量
     * @param binCenters bin中心点
     * @param counts bin计数
     */
    private void addKernelDensityFit(IVector x, double min, double max, int bins, double[] binCenters, int[] counts) {
        try {
            // 使用核密度估计计算密度曲线
            List<double[]> kdeData = kernelDensityEstimation(x, 2.5);

            // 计算最大频次，用于缩放密度曲线
            int maxCount = 0;
            for (int count : counts) {
                maxCount = Math.max(maxCount, count);
            }

            // 计算密度曲线的最大密度值
            double maxDensity = 0;
            for (double[] point : kdeData) {
                maxDensity = Math.max(maxDensity, point[1]);
            }

            // 缩放因子：将密度值转换为频次
            double scaleFactor = maxCount / maxDensity;

            // 创建拟合线数据
            Object[] fitData = new Object[kdeData.size()];
            for (int i = 0; i < kdeData.size(); i++) {
                double[] point = kdeData.get(i);
                double xVal = point[0];
                double density = point[1] * scaleFactor;  // 缩放到频次范围
                fitData[i] = new Number[]{xVal, density};
            }

            LineSeries fitSeries = new LineSeries();
            fitSeries.setData(fitData);
            fitSeries.setName("核密度拟合");
            fitSeries.setType("line");
            fitSeries.setYAxisIndex(0);
            fitSeries.setXAxisIndex(0);
            fitSeries.setShowSymbol(false);  // 不显示圆圈点

            // 添加透明填充效果，与条形图形成协调的同色系配色
            fitSeries.setLineStyle(new org.icepear.echarts.components.series.LineStyle()
                    .setColor("#3ba272") // 深绿色线条，与条形图形成同色系渐变
                    .setWidth(3) // 稍粗的线条更突出
                    .setType("solid"));  // 实线
            fitSeries.setAreaStyle(new org.icepear.echarts.charts.line.LineAreaStyle()
                    .setColor("#73c0de") // 浅蓝色填充，形成清新的对比
                    .setOpacity(0.3));  // 适中的透明度

            // 获取现有的系列数组并添加拟合线
            Object existingSeriesObj = this.option.getSeries();
            if (existingSeriesObj instanceof org.icepear.echarts.origin.util.SeriesOption[]) {
                org.icepear.echarts.origin.util.SeriesOption[] existingSeries = (org.icepear.echarts.origin.util.SeriesOption[]) existingSeriesObj;
                org.icepear.echarts.origin.util.SeriesOption[] newSeries = new org.icepear.echarts.origin.util.SeriesOption[existingSeries.length + 1];

                // 复制现有系列
                System.arraycopy(existingSeries, 0, newSeries, 0, existingSeries.length);
                // 添加拟合线系列
                newSeries[existingSeries.length] = fitSeries;

                // 设置新的系列数组
                this.option.setSeries(newSeries);
            } else {
                // 如果没有现有系列，直接设置拟合线系列
                this.option.setSeries(new org.icepear.echarts.origin.util.SeriesOption[]{fitSeries});
            }

        } catch (Exception e) {
            throw new PlotException("添加核密度拟合失败: " + e.getMessage(), e);
        }
    }

    public Title getTitle() {
        return title;
    }

    public void setTitle(Title title) {
        this.title = title;
    }

    public Legend getLegend() {
        return legend;
    }

    public void setLegend(Legend legend) {
        this.legend = legend;
    }

    public Tooltip getTooltip() {
        return tooltip;
    }

    public void setTooltip(Tooltip tooltip) {
        this.tooltip = tooltip;
    }

    public AxisTicks getXticks() {
        return xticks;
    }

    public void setXticks(AxisTicks xticks) {
        this.xticks = xticks;
    }

    public AxisTicks getYticks() {
        return yticks;
    }

    public void setYticks(AxisTicks yticks) {
        this.yticks = yticks;
    }

    public String getXlabel() {
        return xlabel;
    }

    public void setXlabel(String xlabel) {
        this.xlabel = xlabel;
    }

    public String getYlabel() {
        return ylabel;
    }

    public void setYlabel(String ylabel) {
        this.ylabel = ylabel;
    }

    /**
     * 获取图表宽度
     *
     * @return 图表宽度
     */
    public int getWidth() {
        return width;
    }

    /**
     * 设置图表宽度
     *
     * @param width 图表宽度
     */
    public void setWidth(int width) {
        this.width = width;
    }

    /**
     * 获取图表高度
     *
     * @return 图表高度
     */
    public int getHeight() {
        return height;
    }

    /**
     * 设置图表高度
     *
     * @param height 图表高度
     */
    public void setHeight(int height) {
        this.height = height;
    }

    /**
     * 获取主题
     *
     * @return 主题名称
     */
    public String getTheme() {
        return theme;
    }

    /**
     * 设置图表标题
     *
     * @param titleText 标题文本
     */
    public void setTitle(String titleText) {
        this.title.setText(titleText);
    }

    /**
     * 设置图表标题和副标题
     *
     * @param titleText 标题文本
     * @param subtitleText 副标题文本
     */
    public void setTitle(String titleText, String subtitleText) {
        this.title.setText(titleText);
        this.title.setSubtext(subtitleText);
    }

    // ========== 公共工具方法 ==========
    /**
     * 配置坐标轴
     *
     * @param xAxis X轴
     * @param yAxis Y轴
     */
    private void configureAxes(ValueAxis xAxis, ValueAxis yAxis) {
        xAxis.setName(xlabel.isEmpty() ? "X轴" : xlabel);
        if (xticks.hasTickValues()) {
            xAxis.setData(vectorToStringArray(xticks.getTickValues()));
        }

        yAxis.setName(ylabel.isEmpty() ? "Y轴" : ylabel);
        if (yticks.hasTickValues()) {
            yAxis.setData(vectorToStringArray(yticks.getTickValues()));
        }
    }

    /**
     * 配置分类坐标轴
     *
     * @param xAxis X轴
     * @param yAxis Y轴
     */
    private void configureCategoryAxes(CategoryAxis xAxis, ValueAxis yAxis) {
        xAxis.setName(xlabel.isEmpty() ? "类别" : xlabel);
        yAxis.setName(ylabel.isEmpty() ? "数值" : ylabel);
        if (yticks.hasTickValues()) {
            yAxis.setData(vectorToStringArray(yticks.getTickValues()));
        }
    }

    /**
     * 设置通用选项
     *
     * @param option 图表选项
     */
    private void setCommonOptions(Option option) {
        option.setTitle(title);
        option.setLegend(legend);
        option.setTooltip(tooltip);
    }

    /**
     * 将向量数据转换为Number数组（单向量）
     *
     * @param y Y轴数据
     * @return Number数组
     */
    private Object[] vectorToNumber(IVector y) {
        if (y == null) {
            throw new PlotException("向量数据不能为null");
        }
        Object[] data = new Object[y.length()];
        for (int i = 0; i < y.length(); i++) {
            data[i] = y.get(i);
        }
        return data;
    }

    /**
     * 将两个向量数据转换为Number数组（双向量）
     *
     * @param x X轴数据
     * @param y Y轴数据
     * @return Number数组
     */
    private Object[] vectorToNumber(IVector x, IVector y) {
        if (x == null || y == null) {
            throw new PlotException("向量数据不能为null");
        }
        if (x.length() != y.length()) {
            throw new PlotException("X和Y向量长度必须相等");
        }
        Object[] data = new Object[x.length()];
        for (int i = 0; i < x.length(); i++) {
            data[i] = new Number[]{x.get(i), y.get(i)};
        }
        return data;
    }

    /**
     * 将向量数据转换为字符串数组
     *
     * @param vector 向量数据
     * @return 字符串数组
     */
    private String[] vectorToStringArray(IVector vector) {
        if (vector == null) {
            throw new PlotException("向量数据不能为null");
        }
        String[] result = new String[vector.length()];
        for (int i = 0; i < vector.length(); i++) {
            result[i] = String.valueOf(vector.get(i));
        }
        return result;
    }

    // ========== 流式API方法实现 ==========
    /**
     * 设置图表标题（流式API）
     *
     * @param titleText 标题文本
     * @return 当前实例，支持链式调用
     */
    public RerePlot title(String titleText) {
        this.title.setText(titleText);
        return this;
    }

    /**
     * 设置图表标题和副标题（流式API）
     *
     * @param titleText 标题文本
     * @param subtitleText 副标题文本
     * @return 当前实例，支持链式调用
     */
    public RerePlot title(String titleText, String subtitleText) {
        this.title.setText(titleText);
        this.title.setSubtext(subtitleText);
        return this;
    }

    /**
     * 设置X轴标签（流式API）
     *
     * @param name X轴标签名称
     * @return 当前实例，支持链式调用
     */
    public RerePlot xlabel(String name) {
        this.xlabel = name;
        return this;
    }

    /**
     * 设置Y轴标签（流式API）
     *
     * @param name Y轴标签名称
     * @return 当前实例，支持链式调用
     */
    public RerePlot ylabel(String name) {
        this.ylabel = name;
        return this;
    }

    /**
     * 设置图表尺寸（流式API）
     *
     * @param width 图表宽度
     * @param height 图表高度
     * @return 当前实例，支持链式调用
     */
    public RerePlot size(int width, int height) {
        this.width = width;
        this.height = height;
        return this;
    }

    /**
     * 设置图表主题（流式API，已废弃）
     *
     * @param theme 主题名称
     * @return 当前实例，支持链式调用
     * @deprecated 请使用新的主题系统 theme(String) 方法
     */
    @Deprecated
    public RerePlot setTheme(String theme) {
        this.theme = theme;
        return this;
    }

    private void openBrowseWith(URI uri) {
        try {
            // 1. 检查当前平台是否支持Desktop
            if (!Desktop.isDesktopSupported()) {
                System.out.println("当前环境不支持 Desktop.");
                return;
            }

            Desktop desktop = Desktop.getDesktop();

            // 2. 检查是否支持BROWSE动作
            if (!desktop.isSupported(Desktop.Action.BROWSE)) {
                System.out.println("当前环境不支持打开浏览器.");
                return;
            }

            // 4. 调用默认浏览器打开
            desktop.browse(uri);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 显示图表（流式API）
     *
     * @return 当前实例，支持链式调用
     */
    public RerePlot show() {
        try {
            if (engine == null) {
                throw new PlotException("无法显示图表：ECharts Engine 未初始化");
            }
            String html = engine.renderHtml(option, height + "px", width + "px");
//            String html = engine.renderHtml(option);
            // 保存为临时文件并在浏览器中打开
            String tempFile = "temp_chart_" + System.currentTimeMillis() + ".html";
            var path = java.nio.file.Paths.get(tempFile);
            java.nio.file.Files.write(path, html.getBytes());
            this.openBrowseWith(path.toUri());
//            engine.render(tempFile, option, width + "px", height + "px", true);
            System.out.println("图表已生成并在浏览器中打开: " + tempFile);
        } catch (Exception e) {
            throw new PlotException("显示图表失败: " + e.getMessage(), e);
        }
        return this;
    }

    /**
     * 保存图表为HTML文件（流式API）
     *
     * @param filename 文件名
     * @return 当前实例，支持链式调用
     */
    public RerePlot saveAsHtml(String filename) {
        try {
            if (engine == null) {
                throw new PlotException("无法保存图表：ECharts Engine 未初始化");
            }
            String html = engine.renderHtml(option, height + "px", width + "px");
//            String html = engine.renderHtml(option);
            java.nio.file.Files.write(java.nio.file.Paths.get(filename), html.getBytes());
            System.out.println("图表已保存到: " + filename);
        } catch (Exception e) {
            throw new PlotException("保存图表失败: " + e.getMessage(), e);
        }
        return this;
    }

    /**
     * 获取图表的HTML内容
     *
     * @return HTML字符串
     */
    public String toHtml() {
        try {
            if (engine == null) {
                throw new PlotException("ECharts Engine 未初始化，无法生成HTML");
            }
            return engine.renderHtml(option, width + "px", height + "px");
        } catch (Exception e) {
            throw new PlotException("生成HTML失败: " + e.getMessage(), e);
        }
    }

    /**
     * 获取图表的JSON配置
     *
     * @return JSON字符串
     */
    public String toJson() {
        try {
            if (engine == null) {
                throw new PlotException("ECharts Engine 未初始化，无法生成JSON");
            }
            return engine.renderJsonOption(option);
        } catch (Exception e) {
            throw new PlotException("生成JSON失败: " + e.getMessage(), e);
        }
    }

    /**
     * 设置图表尺寸
     *
     * @param width 宽度
     * @param height 高度
     */
    public void setSize(int width, int height) {
        this.width = width;
        this.height = height;
    }

    /**
     * 热力图
     *
     * @param data 二维数据矩阵
     */
    public RerePlot heatmap(IMatrix data) {
        return heatmapInternal(data, List.of(), List.of(), null);
    }
    
    /**
     * 热力图
     *
     * @param data 二维数据矩阵
     * @param xLabels X轴标签
     * @param yLabels Y轴标签
     */
    public RerePlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels) {
        return heatmapInternal(data, xLabels, yLabels, null);
    }

    /**
     * 创建热力图（支持样式字符串）
     *
     * @param data 数据矩阵
     * @param xLabels X轴标签
     * @param yLabels Y轴标签
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return heatmapInternal(data, xLabels, yLabels, style);
    }

    /**
     * 创建热力图（支持PlotStyle样式）
     *
     * @param data 数据矩阵
     * @param xLabels X轴标签
     * @param yLabels Y轴标签
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels, PlotStyle style) {
        return heatmapInternal(data, xLabels, yLabels, style);
    }

    /**
     * 统一的热力图内部实现
     *
     * @param data 数据矩阵
     * @param xLabels X轴标签
     * @param yLabels Y轴标签
     * @param style 绘图样式
     * @return 当前实例
     */
    private RerePlot heatmapInternal(IMatrix data, List<String> xLabels, List<String> yLabels, PlotStyle style) {
        try {
            Heatmap heatmapChart = new Heatmap();

            // 转换数据格式 - 热力图需要一维数组格式
            Object[] heatmapData = new Object[data.getRowNum() * data.getColNum()];
            int index = 0;
            for (int i = 0; i < data.getRowNum(); i++) {
                for (int j = 0; j < data.getColNum(); j++) {
                    heatmapData[index++] = new Object[]{j, i, data.get(i, j)};
                }
            }

            // 创建热力图系列
            org.icepear.echarts.charts.heatmap.HeatmapSeries series = new org.icepear.echarts.charts.heatmap.HeatmapSeries()
                    .setType("heatmap")
                    .setData(heatmapData);

            // 应用样式
            PlotStyle effectiveStyle = style != null ? style
                    : (useStyleSystem ? defaultStyle : null);

            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToHeatmapSeries(series, effectiveStyle);
            } else {
                series.setName("热力图");
            }

            heatmapChart.addSeries(series);

            // 配置坐标轴
            CategoryAxis xAxis = new CategoryAxis();
            xAxis.setName(xlabel.isEmpty() ? "X轴" : xlabel);
            xAxis.setData(xLabels.toArray(new String[0]));

            CategoryAxis yAxis = new CategoryAxis();
            yAxis.setName(ylabel.isEmpty() ? "Y轴" : ylabel);
            yAxis.setData(yLabels.toArray(new String[0]));

            this.option = heatmapChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);
            this.option.setXAxis(xAxis);
            this.option.setYAxis(yAxis);

            // 添加视觉映射配置
            org.icepear.echarts.components.visualMap.ContinousVisualMap visualMap
                    = new org.icepear.echarts.components.visualMap.ContinousVisualMap();
            visualMap.setMin(data.min());
            visualMap.setMax(data.max());
            visualMap.setCalculable(true);
            this.option.setVisualMap(visualMap);

        } catch (Exception e) {
            System.err.println("创建热力图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 雷达图
     *
     * @param data 数据向量
     * @param indicators 指标名称
     */
    public RerePlot radar(IVector data, List<String> indicators) {
        return radarInternal(data, indicators, null);
    }

    /**
     * 创建雷达图（支持样式字符串）
     *
     * @param data 数据向量
     * @param indicators 指标名称
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot radar(IVector data, List<String> indicators, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return radarInternal(data, indicators, style);
    }

    /**
     * 创建雷达图（支持PlotStyle样式）
     *
     * @param data 数据向量
     * @param indicators 指标名称
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot radar(IVector data, List<String> indicators, PlotStyle style) {
        return radarInternal(data, indicators, style);
    }

    /**
     * 统一的雷达图内部实现
     *
     * @param data 数据向量
     * @param indicators 指标名称
     * @param style 绘图样式
     * @return 当前实例
     */
    private RerePlot radarInternal(IVector data, List<String> indicators, PlotStyle style) {
        try {
            Radar radarChart = new Radar();

            // 转换数据格式 - 雷达图需要对象格式
            Object[] radarData = new Object[1];
            Object[] values = new Object[data.length()];
            for (int i = 0; i < data.length(); i++) {
                values[i] = data.get(i);
            }
            radarData[0] = values;

            // 创建雷达图系列
            org.icepear.echarts.charts.radar.RadarSeries series = new org.icepear.echarts.charts.radar.RadarSeries()
                    .setType("radar")
                    .setData(radarData);

            // 应用样式
            PlotStyle effectiveStyle = style != null ? style
                    : (useStyleSystem ? defaultStyle : null);

            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToRadarSeries(series, effectiveStyle);
            } else {
                series.setName("雷达图");
            }

            radarChart.addSeries(series);

            // 设置雷达图指标配置
            org.icepear.echarts.components.coord.radar.RadarIndicator[] indicatorsArray
                    = new org.icepear.echarts.components.coord.radar.RadarIndicator[indicators.size()];
            for (int i = 0; i < indicators.size(); i++) {
                indicatorsArray[i] = new org.icepear.echarts.components.coord.radar.RadarIndicator()
                        .setName(indicators.get(i))
                        .setMax(100);
            }
            radarChart.setRadarAxis(indicatorsArray);

            this.option = radarChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);

        } catch (Exception e) {
            System.err.println("创建雷达图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 仪表盘
     *
     * @param value 数值
     * @param max 最大值
     * @param min 最小值
     */
    public RerePlot gauge(double value, double max, double min) {
        return gaugeInternal(value, max, min, null);
    }

    /**
     * 创建仪表盘（支持样式字符串）
     *
     * @param value 数值
     * @param max 最大值
     * @param min 最小值
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot gauge(double value, double max, double min, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return gaugeInternal(value, max, min, style);
    }

    /**
     * 创建仪表盘（支持PlotStyle样式）
     *
     * @param value 数值
     * @param max 最大值
     * @param min 最小值
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot gauge(double value, double max, double min, PlotStyle style) {
        return gaugeInternal(value, max, min, style);
    }

    /**
     * 统一的仪表盘内部实现
     *
     * @param value 数值
     * @param max 最大值
     * @param min 最小值
     * @param style 绘图样式
     * @return 当前实例
     */
    private RerePlot gaugeInternal(double value, double max, double min, PlotStyle style) {
        try {
            Gauge gaugeChart = new Gauge();

            // 创建仪表盘系列
            org.icepear.echarts.charts.gauge.GaugeSeries series = new org.icepear.echarts.charts.gauge.GaugeSeries()
                    .setType("gauge")
                    .setData(new Object[]{value});

            // 应用样式
            PlotStyle effectiveStyle = style != null ? style
                    : (useStyleSystem ? defaultStyle : null);

            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToGaugeSeries(series, effectiveStyle);
            } else {
                series.setName("仪表盘");
            }

            gaugeChart.addSeries(series);

            this.option = gaugeChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);

        } catch (Exception e) {
            System.err.println("创建仪表盘时出错: " + e.getMessage());
        }
        return this;
    }

    // ========== 极坐标图表 ==========
    /**
     * 极坐标柱状图
     *
     * @param data 数据向量
     * @param categories 类别标签
     */
    public RerePlot polarBar(IVector data, List<String> categories) {
        return polarBarInternal(data, categories, null);
    }

    public RerePlot polarBar(IVector data, List<String> categories, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return polarBarInternal(data, categories, style);
    }

    public RerePlot polarBar(IVector data, List<String> categories, PlotStyle style) {
        return polarBarInternal(data, categories, style);
    }

    private RerePlot polarBarInternal(IVector data, List<String> categories, PlotStyle style) {
        try {
            PolarBar polarBarChart = new PolarBar();

            // 设置极坐标配置
            polarBarChart.setPolarAxis(new String[]{"30%", "80%"});
            polarBarChart.setAngleAxis(categories.toArray(new String[0]));
            polarBarChart.setRadiusAxis();

            // 设置数据 - 极坐标柱状图使用简单的数值数组
            Object[] dataArray = new Object[data.length()];
            for (int i = 0; i < data.length(); i++) {
                dataArray[i] = data.get(i);
            }

            // 直接创建series并设置数据
            org.icepear.echarts.charts.bar.BarSeries series = new org.icepear.echarts.charts.bar.BarSeries()
                    .setType("bar")
                    .setName("极坐标柱状图")
                    .setData(dataArray)
                    .setCoordinateSystem("polar");

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToBarSeries(series, effectiveStyle);
            }

            this.option = polarBarChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);
            this.option.setSeries(new org.icepear.echarts.charts.bar.BarSeries[]{series});

        } catch (Exception e) {
            System.err.println("创建极坐标柱状图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 极坐标线图
     *
     * @param data 数据向量
     * @param categories 类别标签
     */
    public RerePlot polarLine(IVector data, List<String> categories) {
        return polarLineInternal(data, categories, null);
    }

    public RerePlot polarLine(IVector data, List<String> categories, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return polarLineInternal(data, categories, style);
    }

    public RerePlot polarLine(IVector data, List<String> categories, PlotStyle style) {
        return polarLineInternal(data, categories, style);
    }

    private RerePlot polarLineInternal(IVector data, List<String> categories, PlotStyle style) {
        try {
            PolarLine polarLineChart = new PolarLine();

            // 设置极坐标配置
            polarLineChart.setPolarAxis(new String[]{"30%", "80%"});
            polarLineChart.setAngleAxis(categories.toArray(new String[0]));
            polarLineChart.setRadiusAxis();

            // 设置数据
            Object[] dataArray = new Object[data.length()];
            for (int i = 0; i < data.length(); i++) {
                dataArray[i] = data.get(i);
            }

            // 创建线图系列
            org.icepear.echarts.charts.line.LineSeries series = new org.icepear.echarts.charts.line.LineSeries()
                    .setType("line")
                    .setName("极坐标线图")
                    .setData(dataArray)
                    .setCoordinateSystem("polar");

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToLineSeries(series, effectiveStyle);
            }

            polarLineChart.addSeries("极坐标线图", dataArray);

            this.option = polarLineChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);

        } catch (Exception e) {
            System.err.println("创建极坐标线图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 极坐标散点图
     *
     * @param data 数据向量
     * @param categories 类别标签
     */
    public RerePlot polarScatter(IVector data, List<String> categories) {
        return polarScatterInternal(data, categories, null);
    }

    public RerePlot polarScatter(IVector data, List<String> categories, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return polarScatterInternal(data, categories, style);
    }

    public RerePlot polarScatter(IVector data, List<String> categories, PlotStyle style) {
        return polarScatterInternal(data, categories, style);
    }

    private RerePlot polarScatterInternal(IVector data, List<String> categories, PlotStyle style) {
        try {
            PolarScatter polarScatterChart = new PolarScatter();

            // 设置极坐标配置
            polarScatterChart.setPolarAxis(new String[]{"30%", "80%"});
            polarScatterChart.setAngleAxis(categories.toArray(new String[0]));
            polarScatterChart.setRadiusAxis();

            // 设置数据
            Object[] dataArray = new Object[data.length()];
            for (int i = 0; i < data.length(); i++) {
                dataArray[i] = data.get(i);
            }

            // 创建散点图系列
            org.icepear.echarts.charts.scatter.ScatterSeries series = new org.icepear.echarts.charts.scatter.ScatterSeries()
                    .setType("scatter")
                    .setName("极坐标散点图")
                    .setData(dataArray)
                    .setCoordinateSystem("polar");

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToScatterSeries(series, effectiveStyle);
            }

            polarScatterChart.addSeries("极坐标散点图", dataArray);

            this.option = polarScatterChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);

        } catch (Exception e) {
            System.err.println("创建极坐标散点图时出错: " + e.getMessage());
        }
        return this;
    }

    // ========== 统计图表 ==========    
    // ========== 统计图表 ==========    
    // ========== 统计图表 ==========    
    // ========== 统计图表 ==========    
    // ========== 统计图表 ==========    
    // ========== 统计图表 ==========    
    // ========== 统计图表 ==========    
    // ========== 统计图表 ==========
    /**
     * 箱线图（使用ECharts原生Boxplot）
     *
     * @param data 数据向量
     * @param labels 标签
     */
    public RerePlot boxplot(IVector data, List<String> labels) {
        return boxplotInternal(data, labels, null);
    }

    /**
     * 创建箱线图（支持样式字符串）
     *
     * @param data 数据向量
     * @param labels 标签列表
     * @param styleString 样式字符串
     * @return 当前实例
     */
    public RerePlot boxplot(IVector data, List<String> labels, String styleString) {
        PlotStyle style = (styleString != null && useStyleSystem)
                ? StyleExpression.parse(styleString)
                : null;
        return boxplotInternal(data, labels, style);
    }

    /**
     * 创建箱线图（支持PlotStyle样式）
     *
     * @param data 数据向量
     * @param labels 标签列表
     * @param style 绘图样式
     * @return 当前实例
     */
    public RerePlot boxplot(IVector data, List<String> labels, PlotStyle style) {
        return boxplotInternal(data, labels, style);
    }

    /**
     * 统一的箱线图内部实现
     *
     * @param data 数据向量
     * @param labels 标签列表
     * @param style 绘图样式
     * @return 当前实例
     */
    private RerePlot boxplotInternal(IVector data, List<String> labels, PlotStyle style) {
        try {
            if (data.length() != labels.size()) {
                throw new PlotException("数据向量和标签列表长度必须相等");
            }

            Boxplot boxplotChart = new Boxplot();

            // 按标签分组数据
            Map<String, List<Double>> groupedData = new HashMap<>();
            for (int i = 0; i < data.length(); i++) {
                String label = labels.get(i);
                if (!groupedData.containsKey(label)) {
                    groupedData.put(label, new ArrayList<>());
                }
                groupedData.get(label).add((double) data.get(i));
            }

            // 获取所有唯一的标签，保持顺序
            List<String> uniqueLabels = new ArrayList<>();
            for (String label : labels) {
                if (!uniqueLabels.contains(label)) {
                    uniqueLabels.add(label);
                }
            }

            // 为每个组计算箱线图统计量
            Object[][] boxData = new Object[uniqueLabels.size()][];
            for (int i = 0; i < uniqueLabels.size(); i++) {
                String label = uniqueLabels.get(i);
                List<Double> groupData = groupedData.get(label);

                // 转换为数组并排序
                double[] groupArray = new double[groupData.size()];
                for (int j = 0; j < groupData.size(); j++) {
                    groupArray[j] = groupData.get(j);
                }
                java.util.Arrays.sort(groupArray);

                // 计算统计量 - 使用更准确的分位数计算
                int n = groupArray.length;
                if (n == 0) {
                    // 空数据组，使用默认值
                    boxData[i] = new Object[]{0, 0, 0, 0, 0};
                    continue;
                }

                double min = groupArray[0];
                double max = groupArray[n - 1];

                // 计算Q1 (25%分位数)
                int q1Index = (int) Math.ceil(n * 0.25) - 1;
                q1Index = Math.max(0, Math.min(q1Index, n - 1));
                double q1 = groupArray[q1Index];

                // 计算中位数 (50%分位数)
                int q2Index = (int) Math.ceil(n * 0.5) - 1;
                q2Index = Math.max(0, Math.min(q2Index, n - 1));
                double q2 = groupArray[q2Index];

                // 计算Q3 (75%分位数)
                int q3Index = (int) Math.ceil(n * 0.75) - 1;
                q3Index = Math.max(0, Math.min(q3Index, n - 1));
                double q3 = groupArray[q3Index];

                // 创建箱线图数据 [min, Q1, median, Q3, max]
                boxData[i] = new Object[]{min, q1, q2, q3, max};
            }

            // 创建箱线图系列
            org.icepear.echarts.charts.boxplot.BoxplotSeries series = new org.icepear.echarts.charts.boxplot.BoxplotSeries()
                    .setType("boxplot")
                    .setData(boxData);

            // 应用样式
            PlotStyle effectiveStyle = style != null ? style
                    : (useStyleSystem ? defaultStyle : null);

            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToBoxplotSeries(series, effectiveStyle);
            } else {
                series.setName("箱线图");
            }

            boxplotChart.addSeries(series);

            // 配置坐标轴
            CategoryAxis xAxis = new CategoryAxis();
            xAxis.setName(xlabel.isEmpty() ? "类别" : xlabel);
            xAxis.setData(uniqueLabels.toArray(new String[0]));

            ValueAxis yAxis = new ValueAxis();
            yAxis.setName(ylabel.isEmpty() ? "数值" : ylabel);

            this.option = boxplotChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);
            this.option.setXAxis(xAxis);
            this.option.setYAxis(yAxis);

        } catch (Exception e) {
            throw new PlotException("创建箱线图失败: " + e.getMessage(), e);
        }
        return this;
    }

    @Override
    public IPlot boxplot(IVector data) {
        try {
            Boxplot boxplotChart = new Boxplot();

            // 计算箱线图统计量
            double[] sortedData = new double[data.length()];
            for (int i = 0; i < data.length(); i++) {
                sortedData[i] = (double) data.get(i);
            }
            java.util.Arrays.sort(sortedData);

            int n = sortedData.length;
            if (n == 0) {
                throw new PlotException("数据向量不能为空");
            }

            double min = sortedData[0];
            double max = sortedData[n - 1];

            // 计算Q1 (25%分位数)
            int q1Index = (int) Math.ceil(n * 0.25) - 1;
            q1Index = Math.max(0, Math.min(q1Index, n - 1));
            double q1 = sortedData[q1Index];

            // 计算中位数 (50%分位数)
            int q2Index = (int) Math.ceil(n * 0.5) - 1;
            q2Index = Math.max(0, Math.min(q2Index, n - 1));
            double q2 = sortedData[q2Index];

            // 计算Q3 (75%分位数)
            int q3Index = (int) Math.ceil(n * 0.75) - 1;
            q3Index = Math.max(0, Math.min(q3Index, n - 1));
            double q3 = sortedData[q3Index];

            // 创建箱线图数据 [min, Q1, median, Q3, max]
            Object[] boxData = new Object[1];
            boxData[0] = new Object[]{min, q1, q2, q3, max};

            boxplotChart.addSeries("箱线图", boxData);

            // 配置坐标轴
            CategoryAxis xAxis = new CategoryAxis();
            xAxis.setName(xlabel.isEmpty() ? "数据" : xlabel);
            xAxis.setData(new String[]{"数据集"});

            ValueAxis yAxis = new ValueAxis();
            yAxis.setName(ylabel.isEmpty() ? "数值" : ylabel);

            this.option = boxplotChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);
            this.option.setXAxis(xAxis);
            this.option.setYAxis(yAxis);

        } catch (Exception e) {
            throw new PlotException("创建箱线图失败: " + e.getMessage(), e);
        }
        return this;
    }

    @Override
    public IPlot violinplot(IVector data) {
        return violinplotInternal(data, null, null);
    }

    public IPlot violinplot(IVector data, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return violinplotInternal(data, style, null);
    }

    public IPlot violinplot(IVector data, PlotStyle style) {
        return violinplotInternal(data, style, null);
    }

    @Override
    public IPlot violinplot(IVector data, List<String> labels) {
        return violinplotInternal(data, null, labels);
    }

    public IPlot violinplot(IVector data, List<String> labels, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return violinplotInternal(data, style, labels);
    }

    public IPlot violinplot(IVector data, List<String> labels, PlotStyle style) {
        return violinplotInternal(data, style, labels);
    }

    private IPlot violinplotInternal(IVector data, PlotStyle style, List<String> labels) {
        try {
            // Handle single group case (no labels provided)
            if (labels == null || labels.isEmpty()) {
                // Create a single group with all data
                labels = new ArrayList<>();
                for (int i = 0; i < data.length(); i++) {
                    labels.add("数据集");
                }
            }
            
            if (data.length() != labels.size()) {
                throw new PlotException("数据向量和标签列表长度必须相等");
            }

            // Use Line chart to simulate violin plot
            Line lineChart = new Line();

            // Group data by labels
            Map<String, List<Double>> groupedData = new HashMap<>();
            for (int i = 0; i < data.length(); i++) {
                String label = labels.get(i);
                if (!groupedData.containsKey(label)) {
                    groupedData.put(label, new ArrayList<>());
                }
                groupedData.get(label).add((double) data.get(i));
            }

            // Create density curves for each group
            String[] colors = {"#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc"};
            int colorIndex = 0;

            for (Map.Entry<String, List<Double>> entry : groupedData.entrySet()) {
                String groupName = entry.getKey();
                List<Double> groupData = entry.getValue();

                // Convert to IVector format for calculation
                double[] groupArray = new double[groupData.size()];
                for (int i = 0; i < groupData.size(); i++) {
                    groupArray[i] = groupData.get(i);
                }
                IVector groupVector = new com.reremouse.lab.math.linalg.RereDoubleVector(groupArray);

                // Calculate kernel density estimation
                List<double[]> kdeData = kernelDensityEstimation(groupVector, 2.5);

                // Add violin density curves (left and right symmetric)
                String color = colors[colorIndex % colors.length];
                addViolinDensitySeries(lineChart, kdeData, groupName, color);

                // Add boxplot data (multi-group mode)
                addBoxplotToViolin(lineChart, groupVector, groupName + "_箱线", color, true);
                colorIndex++;
            }

            // Configure axes - show axes as borders
            ValueAxis xAxis = new ValueAxis();
            xAxis.setName(xlabel.isEmpty() ? "数值" : xlabel);
            xAxis.setType("value");
            xAxis.setAxisLine(new org.icepear.echarts.components.coord.AxisLine().setShow(true));
            xAxis.setAxisTick(new org.icepear.echarts.components.coord.CategoryAxisTick().setShow(true));
            xAxis.setSplitLine(new org.icepear.echarts.components.coord.SplitLine().setShow(false));

            ValueAxis yAxis = new ValueAxis();
            yAxis.setName(ylabel.isEmpty() ? "密度" : ylabel);
            yAxis.setType("value");
            yAxis.setAxisLine(new org.icepear.echarts.components.coord.AxisLine().setShow(true));
            yAxis.setAxisTick(new org.icepear.echarts.components.coord.CategoryAxisTick().setShow(true));
            yAxis.setSplitLine(new org.icepear.echarts.components.coord.SplitLine().setShow(false));

            this.option = lineChart.getOption();
            this.option.setTitle(title);
            // Set legend to only show main series, filter out detailed boxplot parts
            Legend filteredLegend = new Legend();
            // Only show main series names for each group
            Set<String> groupNames = new HashSet<>();
            for (Map.Entry<String, List<Double>> entry : groupedData.entrySet()) {
                groupNames.add(entry.getKey());
            }
            filteredLegend.setData(groupNames.toArray(new String[0]));
            this.option.setLegend(filteredLegend);
            this.option.setTooltip(tooltip);
            this.option.setXAxis(xAxis);
            this.option.setYAxis(yAxis);

        } catch (Exception e) {
            throw new PlotException("创建小提琴图失败: " + e.getMessage(), e);
        }
        return this;
    }

    /**
     * K线图（蜡烛图）
     *
     * @param data 数据矩阵，每行包含[开盘价, 收盘价, 最低价, 最高价]
     * @param dates 日期标签
     */
    public RerePlot candlestick(IMatrix data, List<String> dates) {
        return candlestickInternal(data, dates, null);
    }

    public RerePlot candlestick(IMatrix data, List<String> dates, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return candlestickInternal(data, dates, style);
    }

    public RerePlot candlestick(IMatrix data, List<String> dates, PlotStyle style) {
        return candlestickInternal(data, dates, style);
    }

    private RerePlot candlestickInternal(IMatrix data, List<String> dates, PlotStyle style) {
        try {
            Candlestick candlestickChart = new Candlestick();

            // 转换数据格式
            Object[] candlestickData = new Object[data.getRowNum()];
            for (int i = 0; i < data.getRowNum(); i++) {
                if (data.getColNum() >= 4) {
                    candlestickData[i] = new Object[]{
                        data.get(i, 0), // 开盘价
                        data.get(i, 1), // 收盘价
                        data.get(i, 2), // 最低价
                        data.get(i, 3) // 最高价
                    };
                }
            }

            // 创建K线图系列
            org.icepear.echarts.charts.candlestick.CandlestickSeries series = new org.icepear.echarts.charts.candlestick.CandlestickSeries()
                    .setType("candlestick")
                    .setName("K线图")
                    .setData(candlestickData);

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToCandlestickSeries(series, effectiveStyle);
            }

            candlestickChart.addSeries("K线图", candlestickData);

            // 配置坐标轴
            CategoryAxis xAxis = new CategoryAxis();
            xAxis.setName(xlabel.isEmpty() ? "日期" : xlabel);
            xAxis.setData(dates.toArray(new String[0]));

            ValueAxis yAxis = new ValueAxis();
            yAxis.setName(ylabel.isEmpty() ? "价格" : ylabel);

            this.option = candlestickChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);
            this.option.setXAxis(xAxis);
            this.option.setYAxis(yAxis);

        } catch (Exception e) {
            System.err.println("创建K线图时出错: " + e.getMessage());
        }
        return this;
    }

    // ========== 特殊图表 ==========
    /**
     * 漏斗图
     *
     * @param data 数据向量
     * @param labels 标签
     */
    public RerePlot funnel(IVector data, List<String> labels) {
        return funnelInternal(data, labels, null);
    }

    public RerePlot funnel(IVector data, List<String> labels, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return funnelInternal(data, labels, style);
    }

    public RerePlot funnel(IVector data, List<String> labels, PlotStyle style) {
        return funnelInternal(data, labels, style);
    }

    private RerePlot funnelInternal(IVector data, List<String> labels, PlotStyle style) {
        try {
            Funnel funnelChart = new Funnel();

            // 设置数据 - 漏斗图需要对象格式
            Object[] funnelData = new Object[data.length()];
            for (int i = 0; i < data.length(); i++) {
                Map<String, Object> item = new HashMap<>();
                item.put("name", labels.get(i));
                item.put("value", data.get(i));
                funnelData[i] = item;
            }

            // 创建漏斗图系列
            org.icepear.echarts.charts.funnel.FunnelSeries series = new org.icepear.echarts.charts.funnel.FunnelSeries()
                    .setType("funnel")
                    .setName("漏斗图")
                    .setData(funnelData);

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToFunnelSeries(series, effectiveStyle);
            }

            funnelChart.addSeries("漏斗图", funnelData);

            this.option = funnelChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);

        } catch (Exception e) {
            System.err.println("创建漏斗图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 桑基图
     *
     * @param nodes 节点数据
     * @param links 连接数据
     */
    public RerePlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return sankeyInternal(nodes, links, null);
    }

    public RerePlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return sankeyInternal(nodes, links, style);
    }

    public RerePlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        return sankeyInternal(nodes, links, style);
    }

    private RerePlot sankeyInternal(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        try {
            Sankey sankeyChart = new Sankey();

            // 创建SankeyEdgeItem数组
            org.icepear.echarts.charts.sankey.SankeyEdgeItem[] sankeyLinks = new org.icepear.echarts.charts.sankey.SankeyEdgeItem[links.size()];
            for (int i = 0; i < links.size(); i++) {
                Map<String, Object> link = links.get(i);
                org.icepear.echarts.charts.sankey.SankeyEdgeItem edgeItem = new org.icepear.echarts.charts.sankey.SankeyEdgeItem();

                // 设置source和target，支持String和Number类型
                Object source = link.get("source");
                if (source instanceof String) {
                    edgeItem.setSource((String) source);
                } else if (source instanceof Number) {
                    edgeItem.setSource((Number) source);
                }

                Object target = link.get("target");
                if (target instanceof String) {
                    edgeItem.setTarget((String) target);
                } else if (target instanceof Number) {
                    edgeItem.setTarget((Number) target);
                }

                // 设置value
                Object value = link.get("value");
                if (value instanceof Number) {
                    edgeItem.setValue((Number) value);
                } else if (value instanceof String) {
                    edgeItem.setValue((String) value);
                }

                sankeyLinks[i] = edgeItem;
            }

            // 创建SankeySeries并正确设置data和links
            org.icepear.echarts.charts.sankey.SankeySeries series = new org.icepear.echarts.charts.sankey.SankeySeries()
                    .setType("sankey")
                    .setName("桑基图")
                    .setData(nodes.toArray(new Object[0]))
                    .setLinks(sankeyLinks);

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToSankeySeries(series, effectiveStyle);
            }

            this.option = sankeyChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);
            this.option.setSeries(new org.icepear.echarts.charts.sankey.SankeySeries[]{series});

        } catch (Exception e) {
            System.err.println("创建桑基图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 旭日图
     *
     * @param data 层次数据，每个Map应包含id、name、value字段，可选parent字段
     */
    public RerePlot sunburst(List<Map<String, Object>> data) {
        return sunburstInternal(data, null);
    }

    public RerePlot sunburst(List<Map<String, Object>> data, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return sunburstInternal(data, style);
    }

    public RerePlot sunburst(List<Map<String, Object>> data, PlotStyle style) {
        return sunburstInternal(data, style);
    }

    private RerePlot sunburstInternal(List<Map<String, Object>> data, PlotStyle style) {
        try {
            Sunburst sunburstChart = new Sunburst();

            // 将平级数据转换为层次结构
            Object[] sunburstData = buildSunburstHierarchy(data);

            // 创建旭日图系列
            org.icepear.echarts.charts.sunburst.SunburstSeries series = new org.icepear.echarts.charts.sunburst.SunburstSeries()
                    .setType("sunburst")
                    .setName("旭日图")
                    .setData(sunburstData);

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToSunburstSeries(series, effectiveStyle);
            }

            // 设置数据
            sunburstChart.addSeries(sunburstData);

            this.option = sunburstChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);

        } catch (Exception e) {
            System.err.println("创建旭日图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 构建旭日图的层次结构
     *
     * @param data 平级数据列表
     * @return 层次结构数据
     */
    private Object[] buildSunburstHierarchy(List<Map<String, Object>> data) {
        if (data == null || data.isEmpty()) {
            return new Object[0];
        }

        // 检查是否已经包含parent字段
        boolean hasParentField = data.stream().anyMatch(item -> item.containsKey("parent"));

        if (hasParentField) {
            // 使用parent字段构建层次结构
            return buildHierarchyFromParent(data);
        } else {
            // 如果没有parent字段，检查是否有children字段
            boolean hasChildrenField = data.stream().anyMatch(item -> item.containsKey("children"));
            if (hasChildrenField) {
                // 直接使用现有的层次结构
                return data.toArray(new Object[0]);
            } else {
                // 创建默认的根节点包含所有子节点
                Map<String, Object> rootNode = new HashMap<>();
                rootNode.put("name", "根节点");
                rootNode.put("value", 0); // 根节点值设为0，让子节点自动计算
                rootNode.put("children", data);
                return new Object[]{rootNode};
            }
        }
    }

    /**
     * 根据parent字段构建层次结构
     *
     * @param data 包含parent字段的数据
     * @return 层次结构数据
     */
    private Object[] buildHierarchyFromParent(List<Map<String, Object>> data) {
        Map<String, Map<String, Object>> nodeMap = new HashMap<>();
        Map<String, Object> rootNode = null;

        // 首先创建所有节点
        for (Map<String, Object> item : data) {
            String id = (String) item.get("id");
            String name = (String) item.get("name");
            if (id != null && name != null) {
                Map<String, Object> node = new HashMap<>();
                node.put("name", name);
                if (item.containsKey("value")) {
                    node.put("value", item.get("value"));
                }
                nodeMap.put(id, node);
            }
        }

        // 然后建立父子关系
        for (Map<String, Object> item : data) {
            String id = (String) item.get("id");
            String parent = (String) item.get("parent");

            if (id != null && nodeMap.containsKey(id)) {
                Map<String, Object> node = nodeMap.get(id);

                if (parent == null || parent.isEmpty()) {
                    // 这是根节点
                    rootNode = node;
                } else if (nodeMap.containsKey(parent)) {
                    // 这是子节点，添加到父节点的children中
                    Map<String, Object> parentNode = nodeMap.get(parent);
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> children = (List<Map<String, Object>>) parentNode.get("children");
                    if (children == null) {
                        children = new ArrayList<>();
                        parentNode.put("children", children);
                    }
                    children.add(node);
                }
            }
        }

        if (rootNode != null) {
            return new Object[]{rootNode};
        } else {
            // 如果没有找到根节点，创建一个默认的根节点
            Map<String, Object> defaultRoot = new HashMap<>();
            defaultRoot.put("name", "根节点");
            List<Map<String, Object>> children = new ArrayList<>();
            for (Map<String, Object> node : nodeMap.values()) {
                children.add(node);
            }
            defaultRoot.put("children", children);
            return new Object[]{defaultRoot};
        }
    }

    /**
     * 主题河流图
     *
     * @param data 时间序列数据，格式为 [时间, 数值, 类别] 的数组
     * @param categories 类别
     */
    public RerePlot themeRiver(List<Map<String, Object>> data, List<String> categories) {
        return themeRiverInternal(data, categories, null);
    }

    public RerePlot themeRiver(List<Map<String, Object>> data, List<String> categories, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return themeRiverInternal(data, categories, style);
    }

    public RerePlot themeRiver(List<Map<String, Object>> data, List<String> categories, PlotStyle style) {
        return themeRiverInternal(data, categories, style);
    }

    private RerePlot themeRiverInternal(List<Map<String, Object>> data, List<String> categories, PlotStyle style) {
        try {
            ThemeRiver themeRiverChart = new ThemeRiver();

            // 转换数据格式 - 主题河流图需要 [时间, 数值, 类别] 格式
            Object[][] themeRiverData = new Object[data.size()][];
            for (int i = 0; i < data.size(); i++) {
                Map<String, Object> item = data.get(i);
                // 确保时间字段不为null，使用当前时间作为默认值
                Object time = item.get("time");
                if (time == null) {
                    time = "2023-01-0" + (i % 9 + 1); // 生成默认时间
                }
                themeRiverData[i] = new Object[]{
                    time,
                    item.get("value"),
                    item.get("category")
                };
            }

            // 创建ThemeRiverSeries并正确设置data
            org.icepear.echarts.charts.themeRiver.ThemeRiverSeries series = new org.icepear.echarts.charts.themeRiver.ThemeRiverSeries()
                    .setType("themeRiver")
                    .setName("主题河流图")
                    .setData(themeRiverData);

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToThemeRiverSeries(series, effectiveStyle);
            }

            this.option = themeRiverChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);
            this.option.setSeries(new org.icepear.echarts.charts.themeRiver.ThemeRiverSeries[]{series});

            // 添加TimeSingleAxis配置 - 主题河流图必需
            org.icepear.echarts.components.coord.single.TimeSingleAxis singleAxis = new org.icepear.echarts.components.coord.single.TimeSingleAxis()
                    .setType("time");
            themeRiverChart.setSingleAxis(singleAxis);

        } catch (Exception e) {
            System.err.println("创建主题河流图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 树图
     *
     * @param data 树形数据，应该包含name和children字段
     */
    public RerePlot tree(List<Map<String, Object>> data) {
        return treeInternal(data, null);
    }

    public RerePlot tree(List<Map<String, Object>> data, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return treeInternal(data, style);
    }

    public RerePlot tree(List<Map<String, Object>> data, PlotStyle style) {
        return treeInternal(data, style);
    }

    private RerePlot treeInternal(List<Map<String, Object>> data, PlotStyle style) {
        try {
            Tree treeChart = new Tree();

            // 转换数据格式 - 树图需要嵌套的children结构
            // 如果输入数据是平级的，创建一个根节点包含所有子节点
            Object[] treeData;
            if (data.size() == 1 && data.get(0).containsKey("children")) {
                // 如果已经是树结构，直接使用
                treeData = new Object[data.size()];
                for (int i = 0; i < data.size(); i++) {
                    treeData[i] = convertToTreeStructure(data.get(i));
                }
            } else {
                // 检查是否包含parent字段，如果有则构建正确的树结构
                boolean hasParentField = data.stream().anyMatch(item -> item.containsKey("parent"));

                if (hasParentField) {
                    // 根据parent字段构建树结构
                    Map<String, Map<String, Object>> nodeMap = new HashMap<>();
                    Map<String, Object> rootNode = null;

                    // 首先创建所有节点
                    for (Map<String, Object> item : data) {
                        String id = (String) item.get("id");
                        String name = (String) item.get("name");
                        if (id != null && name != null) {
                            Map<String, Object> node = new HashMap<>();
                            node.put("name", name);
                            if (item.containsKey("value")) {
                                node.put("value", item.get("value"));
                            }
                            nodeMap.put(id, node);
                        }
                    }

                    // 然后建立父子关系
                    for (Map<String, Object> item : data) {
                        String id = (String) item.get("id");
                        String parent = (String) item.get("parent");

                        if (id != null && nodeMap.containsKey(id)) {
                            Map<String, Object> node = nodeMap.get(id);

                            if (parent == null || parent.isEmpty()) {
                                // 这是根节点
                                rootNode = node;
                            } else if (nodeMap.containsKey(parent)) {
                                // 这是子节点，添加到父节点的children中
                                Map<String, Object> parentNode = nodeMap.get(parent);
                                @SuppressWarnings("unchecked")
                                List<Map<String, Object>> children = (List<Map<String, Object>>) parentNode.get("children");
                                if (children == null) {
                                    children = new ArrayList<>();
                                    parentNode.put("children", children);
                                }
                                children.add(node);
                            }
                        }
                    }

                    if (rootNode != null) {
                        treeData = new Object[]{convertToTreeStructure(rootNode)};
                    } else {
                        // 如果没有找到根节点，创建一个默认的根节点
                        Map<String, Object> defaultRoot = new HashMap<>();
                        defaultRoot.put("name", "根节点");
                        List<Map<String, Object>> children = new ArrayList<>();
                        for (Map<String, Object> node : nodeMap.values()) {
                            children.add(node);
                        }
                        defaultRoot.put("children", children);
                        treeData = new Object[]{convertToTreeStructure(defaultRoot)};
                    }
                } else {
                    // 如果输入是平级数据且没有parent字段，创建根节点包含所有子节点
                    Map<String, Object> rootNode = new HashMap<>();
                    rootNode.put("name", "根节点");
                    List<Map<String, Object>> children = new ArrayList<>();
                    Set<String> usedNames = new HashSet<>();

                    for (Map<String, Object> item : data) {
                        // 确保子节点名称唯一，避免重复
                        Map<String, Object> childNode = new HashMap<>(item);
                        String originalName = (String) childNode.get("name");
                        String uniqueName = originalName;

                        // 如果名称已存在，添加数字后缀
                        int counter = 1;
                        while (usedNames.contains(uniqueName)) {
                            uniqueName = originalName + "_" + counter;
                            counter++;
                        }

                        childNode.put("name", uniqueName);
                        usedNames.add(uniqueName);
                        children.add(childNode);
                    }

                    rootNode.put("children", children);
                    treeData = new Object[]{convertToTreeStructure(rootNode)};
                }
            }

            // 创建TreeSeries并正确设置data
            org.icepear.echarts.charts.tree.TreeSeries series = new org.icepear.echarts.charts.tree.TreeSeries()
                    .setType("tree")
                    .setName("树图")
                    .setData(treeData);

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToTreeSeries(series, effectiveStyle);
            }

            this.option = treeChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);
            this.option.setSeries(new org.icepear.echarts.charts.tree.TreeSeries[]{series});

        } catch (Exception e) {
            System.err.println("创建树图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 将Map数据转换为树结构
     */
    private Object convertToTreeStructure(Map<String, Object> data) {
        Map<String, Object> node = new HashMap<>();
        node.put("name", data.get("name"));
        if (data.containsKey("value")) {
            node.put("value", data.get("value"));
        }
        if (data.containsKey("children")) {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> children = (List<Map<String, Object>>) data.get("children");
            if (children != null && !children.isEmpty()) {
                Object[] childrenArray = new Object[children.size()];
                for (int i = 0; i < children.size(); i++) {
                    childrenArray[i] = convertToTreeStructure(children.get(i));
                }
                node.put("children", childrenArray);
            }
        }
        return node;
    }

    /**
     * 矩形树图
     *
     * @param data 层次数据，每个Map应包含id、name、value字段，可选parent字段
     */
    public RerePlot treemap(List<Map<String, Object>> data) {
        return treemapInternal(data, null);
    }

    public RerePlot treemap(List<Map<String, Object>> data, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return treemapInternal(data, style);
    }

    public RerePlot treemap(List<Map<String, Object>> data, PlotStyle style) {
        return treemapInternal(data, style);
    }

    private RerePlot treemapInternal(List<Map<String, Object>> data, PlotStyle style) {
        try {
            Treemap treemapChart = new Treemap();

            // 将平级数据转换为层次结构
            Object[] treemapData = buildTreemapHierarchy(data);

            // 创建矩形树图系列
            org.icepear.echarts.charts.treemap.TreemapSeries series = new org.icepear.echarts.charts.treemap.TreemapSeries()
                    .setType("treemap")
                    .setName("矩形树图")
                    .setData(treemapData);

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToTreemapSeries(series, effectiveStyle);
            }

            // 设置数据
            treemapChart.addSeries(treemapData);

            this.option = treemapChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);

        } catch (Exception e) {
            System.err.println("创建矩形树图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 构建矩形树图的层次结构
     *
     * @param data 平级数据列表
     * @return 层次结构数据
     */
    private Object[] buildTreemapHierarchy(List<Map<String, Object>> data) {
        if (data == null || data.isEmpty()) {
            return new Object[0];
        }

        // 检查是否已经包含parent字段
        boolean hasParentField = data.stream().anyMatch(item -> item.containsKey("parent"));

        if (hasParentField) {
            // 使用parent字段构建层次结构
            return buildHierarchyFromParent(data);
        } else {
            // 如果没有parent字段，检查是否有children字段
            boolean hasChildrenField = data.stream().anyMatch(item -> item.containsKey("children"));
            if (hasChildrenField) {
                // 直接使用现有的层次结构
                return data.toArray(new Object[0]);
            } else {
                // 创建默认的根节点包含所有子节点
                Map<String, Object> rootNode = new HashMap<>();
                rootNode.put("name", "根节点");
                rootNode.put("value", 0); // 根节点值设为0，让子节点自动计算
                rootNode.put("children", data);
                return new Object[]{rootNode};
            }
        }
    }

    /**
     * 关系图
     *
     * @param nodes 节点数据
     * @param links 连接数据
     */
    public RerePlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return graphInternal(nodes, links, null);
    }

    public RerePlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return graphInternal(nodes, links, style);
    }

    public RerePlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        return graphInternal(nodes, links, style);
    }

    private RerePlot graphInternal(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        try {
            Graph graphChart = new Graph();

            // 创建GraphNodeItem数组，添加坐标信息
            org.icepear.echarts.charts.graph.GraphNodeItem[] graphNodes = new org.icepear.echarts.charts.graph.GraphNodeItem[nodes.size()];
            for (int i = 0; i < nodes.size(); i++) {
                Map<String, Object> node = nodes.get(i);
                org.icepear.echarts.charts.graph.GraphNodeItem nodeItem = new org.icepear.echarts.charts.graph.GraphNodeItem();

                // 设置基本属性
                nodeItem.setName((String) node.get("name"));
                if (node.containsKey("id")) {
                    Object id = node.get("id");
                    if (id instanceof String) {
                        nodeItem.setId((String) id);
                    } else if (id instanceof Number) {
                        nodeItem.setId(((Number) id).toString());
                    }
                }

                // 添加坐标信息，避免节点重叠
                double angle = 2 * Math.PI * i / nodes.size();
                double radius = 200;
                nodeItem.setX(400 + radius * Math.cos(angle));
                nodeItem.setY(300 + radius * Math.sin(angle));

                graphNodes[i] = nodeItem;
            }

            // 创建GraphEdgeItem数组
            org.icepear.echarts.charts.graph.GraphEdgeItem[] graphLinks = new org.icepear.echarts.charts.graph.GraphEdgeItem[links.size()];
            for (int i = 0; i < links.size(); i++) {
                Map<String, Object> link = links.get(i);
                org.icepear.echarts.charts.graph.GraphEdgeItem edgeItem = new org.icepear.echarts.charts.graph.GraphEdgeItem();

                // 设置source和target，支持String和Number类型
                Object source = link.get("source");
                if (source instanceof String) {
                    edgeItem.setSource((String) source);
                } else if (source instanceof Number) {
                    edgeItem.setSource((Number) source);
                }

                Object target = link.get("target");
                if (target instanceof String) {
                    edgeItem.setTarget((String) target);
                } else if (target instanceof Number) {
                    edgeItem.setTarget((Number) target);
                }

                // 设置value
                Object value = link.get("value");
                if (value instanceof Number) {
                    edgeItem.setValue((Number) value);
                } else if (value instanceof String) {
                    edgeItem.setValue((String) value);
                }

                graphLinks[i] = edgeItem;
            }

            // 创建GraphSeries并正确设置data和links
            // 关系图应该使用data属性而不是nodes属性
            org.icepear.echarts.charts.graph.GraphSeries series = new org.icepear.echarts.charts.graph.GraphSeries()
                    .setType("graph")
                    .setName("关系图")
                    .setData(graphNodes)
                    .setLinks(graphLinks);

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToGraphSeries(series, effectiveStyle);
            }

            this.option = graphChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);
            this.option.setSeries(new org.icepear.echarts.charts.graph.GraphSeries[]{series});

        } catch (Exception e) {
            System.err.println("创建关系图时出错: " + e.getMessage());
        }
        return this;
    }

    /**
     * 平行坐标图
     *
     * @param data 数据矩阵
     * @param dimensions 维度名称
     */
    public RerePlot parallel(IMatrix data, List<String> dimensions) {
        return parallelInternal(data, dimensions, null);
    }

    public RerePlot parallel(IMatrix data, List<String> dimensions, String styleString) {
        PlotStyle style = StyleExpression.parse(styleString);
        return parallelInternal(data, dimensions, style);
    }

    public RerePlot parallel(IMatrix data, List<String> dimensions, PlotStyle style) {
        return parallelInternal(data, dimensions, style);
    }

    private RerePlot parallelInternal(IMatrix data, List<String> dimensions, PlotStyle style) {
        try {
            Parallel parallelChart = new Parallel();

            // 转换数据格式
            Object[] parallelData = new Object[data.getRowNum()];
            for (int i = 0; i < data.getRowNum(); i++) {
                Object[] rowData = new Object[data.getColNum()];
                for (int j = 0; j < data.getColNum(); j++) {
                    rowData[j] = data.get(i, j);
                }
                parallelData[i] = rowData;
            }

            // 添加平行坐标轴 - 这是必需的
            for (int i = 0; i < dimensions.size(); i++) {
                parallelChart.addParallelAxis(dimensions.get(i), i);
            }

            // 创建平行坐标图系列
            org.icepear.echarts.charts.parallel.ParallelSeries series = new org.icepear.echarts.charts.parallel.ParallelSeries()
                    .setType("parallel")
                    .setName("平行坐标图")
                    .setData(parallelData);

            // 应用样式
            PlotStyle effectiveStyle = (style != null) ? style : defaultStyle;
            if (effectiveStyle != null && useStyleSystem) {
                UniversalStyleApplier.applyToParallelSeries(series, effectiveStyle);
            }

            parallelChart.addSeries("平行坐标图", parallelData);

            this.option = parallelChart.getOption();
            this.option.setTitle(title);
            this.option.setLegend(legend);
            this.option.setTooltip(tooltip);

        } catch (Exception e) {
            System.err.println("创建平行坐标图时出错: " + e.getMessage());
        }
        return this;
    }

    // ========== 小提琴图辅助方法 ==========
    /**
     * 核密度估计
     *
     * @param data 数据向量
     * @param bandwidth 带宽参数
     * @return 密度估计点列表
     */
    private List<double[]> kernelDensityEstimation(IVector data, double bandwidth) {
        List<double[]> points = new ArrayList<>();
        double min = (double) data.min();
        double max = (double) data.max();
        double range = max - min;
        
        // Handle edge case where all values are the same
        if (range == 0) {
            points.add(new double[]{min, 1.0});
            return points;
        }
        
        double step = range / 100.0;

        for (double x = min - range * 0.2; x <= max + range * 0.2; x += step) {
            double density = 0;
            for (int i = 0; i < data.length(); i++) {
                double u = (x - (double) data.get(i)) / bandwidth;
                density += Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
            }
            density /= (data.length() * bandwidth);
            points.add(new double[]{x, density});
        }

        return points;
    }

    /**
     * 添加小提琴密度曲线系列（左右对称）
     *
     * @param chart 图表对象
     * @param kdeData 核密度估计数据
     * @param name 系列名称
     * @param color 颜色
     */
    private void addViolinDensitySeries(Line chart, List<double[]> kdeData, String name, String color) {
        // 创建小提琴形状的数据：先左半边（从右到左），再右半边（从左到右）
        List<Number[]> violinData = new ArrayList<>();

        // 添加左半边数据（从右到左，负密度值）
        for (int i = kdeData.size() - 1; i >= 0; i--) {
            double[] point = kdeData.get(i);
            double x = point[0];
            double density = point[1] * 100; // 放大密度值以便更好显示
            violinData.add(new Number[]{x, -density});
        }

        // 添加右半边数据（从左到右，正密度值）
        for (double[] point : kdeData) {
            double x = point[0];
            double density = point[1] * 100; // 放大密度值以便更好显示
            violinData.add(new Number[]{x, density});
        }

        // 创建小提琴系列，使用面积填充
        org.icepear.echarts.charts.line.LineSeries violinSeries = new org.icepear.echarts.charts.line.LineSeries()
                .setName(name)
                .setData(violinData.toArray(new Number[0][]))
                .setShowSymbol(false)
                .setLineStyle(new org.icepear.echarts.components.series.LineStyle()
                        .setColor(color)
                        .setWidth(2))
                .setAreaStyle(new org.icepear.echarts.charts.line.LineAreaStyle()
                        .setColor(color)
                        .setOpacity(0.3));

        chart.addSeries(violinSeries);
    }

    /**
     * 添加箱线图到小提琴图中
     *
     * @param chart 图表对象
     * @param data 数据向量
     * @param name 系列名称
     * @param color 箱线图颜色
     */
    private void addBoxplotToViolin(Line chart, IVector data, String name, String color) {
        addBoxplotToViolin(chart, data, name, color, false);
    }

    /**
     * 添加箱线图到小提琴图中（支持多组模式）
     *
     * @param chart 图表对象
     * @param data 数据向量
     * @param name 系列名称
     * @param color 箱线图颜色
     * @param isMultiGroup 是否为多组模式
     */
    private void addBoxplotToViolin(Line chart, IVector data, String name, String color, boolean isMultiGroup) {
        // 计算箱线图统计量
        double[] sortedData = new double[data.length()];
        for (int i = 0; i < data.length(); i++) {
            sortedData[i] = (double) data.get(i);
        }
        java.util.Arrays.sort(sortedData);

        int n = sortedData.length;
        double q1 = sortedData[n / 4];
        double q2 = sortedData[n / 2];
        double q3 = sortedData[3 * n / 4];
        double min = (double) data.min();
        double max = (double) data.max();

        // 计算密度曲线的实际高度范围，用于调整箱线图高度
        List<double[]> kdeData = kernelDensityEstimation(data, 2.5);
        double maxDensity = 0;
        for (double[] point : kdeData) {
            maxDensity = Math.max(maxDensity, point[1]);
        }
        // 将密度值放大100倍（与密度曲线保持一致）
        double densityHeight = (float) (maxDensity * 100);

        // 根据密度曲线高度自动调整箱线图高度
        double boxHeight;

        if (isMultiGroup) {
            // 多组模式：概率密度大的组应该有更高的箱线图
            if (densityHeight > 30) {
                // 高密度：使用较高比例，箱线图更高
                boxHeight = Math.max(2.0f, densityHeight * 0.20f);
            } else if (densityHeight > 15) {
                // 中高密度：使用中等比例
                boxHeight = Math.max(1.5f, densityHeight * 0.15f);
            } else if (densityHeight > 8) {
                // 中等密度：使用标准比例
                boxHeight = Math.max(1.2f, densityHeight * 0.12f);
            } else if (densityHeight > 3) {
                // 中低密度：使用较低比例
                boxHeight = Math.max(1.0f, densityHeight * 0.10f);
            } else {
                // 低密度：使用最低比例，箱线图最低
                boxHeight = Math.max(0.8f, densityHeight * 0.08f);
            }
            // 多组模式：允许更大的高度范围以突出差异
            boxHeight = Math.min(boxHeight, 4.0f);
        } else {
            // 单组模式：使用更保守的比例计算
            if (densityHeight > 20) {
                boxHeight = Math.max(0.8f, densityHeight * 0.08f);
            } else if (densityHeight > 10) {
                boxHeight = Math.max(0.6f, densityHeight * 0.06f);
            } else if (densityHeight > 5) {
                boxHeight = Math.max(0.5f, densityHeight * 0.05f);
            } else {
                boxHeight = Math.max(0.4f, densityHeight * 0.04f);
            }
            // 单组模式：严格限制高度
            boxHeight = Math.min(boxHeight, 1.2f);
        }

        // 1. 左须线
        List<Number[]> leftWhisker = new ArrayList<>();
        leftWhisker.add(new Number[]{min, 0});
        leftWhisker.add(new Number[]{q1, 0});

        // 2. 右须线
        List<Number[]> rightWhisker = new ArrayList<>();
        rightWhisker.add(new Number[]{q3, 0});
        rightWhisker.add(new Number[]{max, 0});

        // 3. 箱体（矩形）- 分别绘制四条边，确保每条线都清晰可见
        // 底部线
        List<Number[]> bottomLine = new ArrayList<>();
        bottomLine.add(new Number[]{q1, -boxHeight});
        bottomLine.add(new Number[]{q3, -boxHeight});

        // 顶部线
        List<Number[]> topLine = new ArrayList<>();
        topLine.add(new Number[]{q1, boxHeight});
        topLine.add(new Number[]{q3, boxHeight});

        // 左侧竖线
        List<Number[]> leftLine = new ArrayList<>();
        leftLine.add(new Number[]{q1, -boxHeight});
        leftLine.add(new Number[]{q1, boxHeight});

        // 右侧竖线
        List<Number[]> rightLine = new ArrayList<>();
        rightLine.add(new Number[]{q3, -boxHeight});
        rightLine.add(new Number[]{q3, boxHeight});

        // 4. 中位数线
        List<Number[]> median = new ArrayList<>();
        median.add(new Number[]{q2, -boxHeight});
        median.add(new Number[]{q2, boxHeight});

        // 添加各个部分到图表，隐藏详细部分避免legend过多
        // 左须线
        chart.addSeries(new org.icepear.echarts.charts.line.LineSeries()
                .setName(name + "_左须")
                .setData(leftWhisker.toArray(new Number[0][]))
                .setShowSymbol(false)
                .setLegendHoverLink(false)
                .setLineStyle(new org.icepear.echarts.components.series.LineStyle()
                        .setColor(color)
                        .setWidth(2)));

        // 右须线
        chart.addSeries(new org.icepear.echarts.charts.line.LineSeries()
                .setName(name + "_右须")
                .setData(rightWhisker.toArray(new Number[0][]))
                .setShowSymbol(false)
                .setLegendHoverLink(false)
                .setLineStyle(new org.icepear.echarts.components.series.LineStyle()
                        .setColor(color)
                        .setWidth(2)));

        // 箱体底部线
        chart.addSeries(new org.icepear.echarts.charts.line.LineSeries()
                .setName(name + "_箱底")
                .setData(bottomLine.toArray(new Number[0][]))
                .setShowSymbol(false)
                .setLegendHoverLink(false)
                .setLineStyle(new org.icepear.echarts.components.series.LineStyle()
                        .setColor(color)
                        .setWidth(2)));

        // 箱体顶部线
        chart.addSeries(new org.icepear.echarts.charts.line.LineSeries()
                .setName(name + "_箱顶")
                .setData(topLine.toArray(new Number[0][]))
                .setShowSymbol(false)
                .setLegendHoverLink(false)
                .setLineStyle(new org.icepear.echarts.components.series.LineStyle()
                        .setColor(color)
                        .setWidth(2)));

        // 箱体左侧竖线
        chart.addSeries(new org.icepear.echarts.charts.line.LineSeries()
                .setName(name + "_箱左")
                .setData(leftLine.toArray(new Number[0][]))
                .setShowSymbol(false)
                .setLegendHoverLink(false)
                .setLineStyle(new org.icepear.echarts.components.series.LineStyle()
                        .setColor(color)
                        .setWidth(2)));

        // 箱体右侧竖线
        chart.addSeries(new org.icepear.echarts.charts.line.LineSeries()
                .setName(name + "_箱右")
                .setData(rightLine.toArray(new Number[0][]))
                .setShowSymbol(false)
                .setLegendHoverLink(false)
                .setLineStyle(new org.icepear.echarts.components.series.LineStyle()
                        .setColor(color)
                        .setWidth(2)));

        // 中位数线
        chart.addSeries(new org.icepear.echarts.charts.line.LineSeries()
                .setName(name + "_中位数")
                .setData(median.toArray(new Number[0][]))
                .setShowSymbol(false)
                .setLegendHoverLink(false)
                .setLineStyle(new org.icepear.echarts.components.series.LineStyle()
                        .setColor(color)
                        .setWidth(3)));
    }

    // ========== 样式系统辅助方法 ==========
    /**
     * 应用样式到LineSeries
     *
     * @param lineSeries 线条系列
     * @param style 样式对象
     */
    private void applyStyleToLineSeries(LineSeries lineSeries, PlotStyle style) {
        if (style == null) {
            lineSeries.setName("数据线");
            return;
        }

        // 设置名称/标签
        String label = style.getLabel();
        lineSeries.setName(label.isEmpty() ? "数据线" : label);

        // 设置线条样式
        org.icepear.echarts.components.series.LineStyle lineStyle = StyleConverter.toEChartsLineStyle(style);
        lineSeries.setLineStyle(lineStyle);

        // 设置标记符号
        if (style.getMarker() != null && !style.getMarker().isEmpty()) {
            String symbol = StyleConverter.convertMarkerToSymbol(style.getMarker());
            lineSeries.setSymbol(symbol);
            lineSeries.setSymbolSize((int) style.getMarkerSize());
            lineSeries.setShowSymbol(true);
        }

        // 设置透明度和填充
        if (style.getAlpha() < 1.0f) {
            org.icepear.echarts.charts.line.LineAreaStyle areaStyle = StyleConverter.toEChartsAreaStyle(style);
            lineSeries.setAreaStyle(areaStyle);
        }
    }

    /**
     * 应用样式到ScatterSeries
     *
     * @param scatterSeries 散点系列
     * @param style 样式对象
     */
    private void applyStyleToScatterSeries(ScatterSeries scatterSeries, PlotStyle style) {
        if (style == null) {
            scatterSeries.setName("散点数据");
            return;
        }

        // 设置名称/标签
        String label = style.getLabel();
        scatterSeries.setName(label.isEmpty() ? "散点数据" : label);

        // 设置标记符号和颜色
        if (style.getMarker() != null && !style.getMarker().isEmpty()) {
            String symbol = StyleConverter.convertMarkerToSymbol(style.getMarker());
            scatterSeries.setSymbol(symbol);
        }

        scatterSeries.setSymbolSize((int) style.getMarkerSize());

        // 设置颜色（散点图主要通过itemStyle设置颜色）
        if (style.getColor() != null) {
            // 这里需要设置itemStyle，但ECharts-Java可能需要特殊处理
            // 暂时通过系列级别设置
        }
    }

    /**
     * 设置默认样式
     *
     * @param style 默认样式
     * @return 当前实例，支持链式调用
     */
    public RerePlot setDefaultStyle(PlotStyle style) {
        this.defaultStyle = style != null ? style : PlotStyle.defaultStyle();
        return this;
    }

    /**
     * 获取默认样式
     *
     * @return 默认样式
     */
    public PlotStyle getDefaultStyle() {
        return this.defaultStyle;
    }

    /**
     * 设置调色板
     *
     * @param paletteName 调色板名称
     * @return 当前实例，支持链式调用
     */
    public RerePlot setPalette(String paletteName) {
        if (ColorPalette.hasPalette(paletteName)) {
            this.currentPalette = paletteName;
        }
        return this;
    }

    /**
     * 获取当前调色板名称
     *
     * @return 调色板名称
     */
    public String getPalette() {
        return this.currentPalette;
    }

    /**
     * 启用或禁用样式系统
     *
     * @param enabled 是否启用
     * @return 当前实例，支持链式调用
     */
    public RerePlot enableStyleSystem(boolean enabled) {
        this.useStyleSystem = enabled;
        return this;
    }

    /**
     * 检查样式系统是否启用
     *
     * @return 是否启用
     */
    public boolean isStyleSystemEnabled() {
        return this.useStyleSystem;
    }

    /**
     * 应用样式字符串到图表元素
     *
     * @param styleString 样式字符串
     * @return 解析后的样式对象
     */
    public PlotStyle parseStyle(String styleString) {
        return StyleExpression.parse(styleString);
    }

    /**
     * 创建自定义样式
     *
     * @param color 颜色
     * @param lineStyle 线条样式
     * @param marker 标记
     * @return 样式对象
     */
    public static PlotStyle createStyle(String color, String lineStyle, String marker) {
        return new PlotStyle()
                .color(color)
                .lineStyle(lineStyle)
                .marker(marker);
    }

    // ========== 主题系统方法 ==========
    /**
     * 设置当前主题
     *
     * @param themeName 主题名称
     * @return 当前实例，支持链式调用
     */
    public RerePlot theme(String themeName) {
        this.currentTheme = themeName;
        if (this.option != null && this.useThemeSystem) {
            ThemeManager.applyTheme(this.option, themeName);
        }
        return this;
    }

    /**
     * 获取当前主题
     *
     * @return 当前主题名称
     */
    public String getCurrentTheme() {
        return this.currentTheme;
    }

    /**
     * 启用或禁用主题系统
     *
     * @param enabled 是否启用
     * @return 当前实例，支持链式调用
     */
    public RerePlot enableThemeSystem(boolean enabled) {
        this.useThemeSystem = enabled;
        return this;
    }

    /**
     * 检查主题系统是否启用
     *
     * @return 是否启用
     */
    public boolean isThemeSystemEnabled() {
        return this.useThemeSystem;
    }

    /**
     * 应用主题到当前图表
     *
     * @param themeName 主题名称
     * @return 当前实例，支持链式调用
     */
    public RerePlot applyTheme(String themeName) {
        if (this.option != null && this.useThemeSystem) {
            ThemeManager.applyTheme(this.option, themeName);
            this.currentTheme = themeName;
        }
        return this;
    }

    /**
     * 注册自定义主题
     *
     * @param themeName 主题名称
     * @param theme 自定义主题对象
     * @return 当前实例，支持链式调用
     */
    public RerePlot registerTheme(String themeName, ThemeManager.CustomTheme theme) {
        ThemeManager.registerCustomTheme(themeName, theme);
        return this;
    }

    /**
     * 创建渐变主题
     *
     * @param themeName 主题名称
     * @param startColor 起始颜色
     * @param endColor 结束颜色
     * @param backgroundColor 背景颜色
     * @return 当前实例，支持链式调用
     */
    public RerePlot createGradientTheme(String themeName, String startColor, String endColor, String backgroundColor) {
        ThemeManager.CustomTheme theme = ThemeManager.createGradientTheme(themeName, startColor, endColor, backgroundColor);
        ThemeManager.registerCustomTheme(themeName, theme);
        return this;
    }

    /**
     * 创建单色主题
     *
     * @param themeName 主题名称
     * @param baseColor 基础颜色
     * @param backgroundColor 背景颜色
     * @return 当前实例，支持链式调用
     */
    public RerePlot createMonochromeTheme(String themeName, String baseColor, String backgroundColor) {
        ThemeManager.CustomTheme theme = ThemeManager.createMonochromeTheme(themeName, baseColor, backgroundColor);
        ThemeManager.registerCustomTheme(themeName, theme);
        return this;
    }

    /**
     * 获取所有可用的主题名称
     *
     * @return 主题名称列表
     */
    public static List<String> getAvailableThemes() {
        return ThemeManager.getRegisteredThemeNames();
    }

    /**
     * 获取内置主题名称
     *
     * @return 内置主题名称列表
     */
    public static List<String> getBuiltinThemes() {
        return ThemeManager.getBuiltinThemeNames();
    }

}
