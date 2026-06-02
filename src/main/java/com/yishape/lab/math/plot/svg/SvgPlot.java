package com.yishape.lab.math.plot.svg;

import com.yishape.lab.math.plot.svg.renderers.SvgPolarRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgPieRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgCartesianRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgGaugeRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgHeatmapRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgQqplotRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgSunburstRenderer;
import com.yishape.lab.math.plot.svg.renderers.AbstractSvgChartRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgHistogramRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgBarRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgViolinRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgSubplotsRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgJointplotRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgBoxplotRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgRadarRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgPairplotRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgTreemapRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgFunnelRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgSankeyRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgParallelRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgTreeRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgRegplotRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgThemeRiverRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgGraphRenderer;
import com.yishape.lab.math.plot.svg.renderers.SvgCandlestickRenderer;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.PlotAxisScale;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.PlotException;
import com.yishape.lab.math.plot.PlotHostBridge;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxPlot.ChartType;
import com.yishape.lab.math.plot.javafx.SvgPlotFigureWindow;
import com.yishape.lab.math.plot.PlotStyle;
import com.yishape.lab.math.plot.PlotKde;
import com.yishape.lab.math.plot.PairplotDiagonal;
import com.yishape.lab.math.plot.JointplotMarginal;
import com.yishape.lab.math.plot.LegendPositions;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.Base64;

/**
 * 真正的矢量SVG图表实现。
 * 支持箱线图、直方图等，SVG元素（&lt;rect&gt;、&lt;line&gt;、&lt;text&gt;）绘制，放大无模糊。
 *
 * <p>实现 IPlot 接口，使用方式：
 * <pre>{@code
 * IPlot plot = Plots.ofSvg(800, 600);
 * plot.boxplot(Linalg.vector(data)).title("箱线图");
 * plot.saveAsSvg("out.svg");
 * }</pre>
 */
public class SvgPlot implements IPlot {

    private ChartType currentChartType = ChartType.LINE;
    private final List<SeriesData> seriesList = new ArrayList<>();
    private ChartConfig chartConfig;
    /** Legend position string (see {@link LegendPositions}). */
    private String legendPosition = LegendPositions.TOP_RIGHT;
    private String theme = "default";
    private Integer pendingBins = null;

    /** IPlot subplots / subplot：按格子暂存序列，在 {@link #toSvgString()} 时合成 subplots 图层 */
    private boolean facetMode;
    private int facetRows = 1, facetCols = 1;
    private int facetCursor;
    private SeriesData[] facetCells;

    /**
     * 在 {@link #finalizeFacetSubplotsIfNeeded()} 写入 subplots 持有者的 extraData：
     * {@code facetLayout}=pairplot 时附 {@code columnNames}；与 JavaFx 的 facet 网格约定一致。
     */
    private String pendingFacetLayout;
    private List<String> pendingFacetColumnNames;

    private final Map<String, AbstractSvgChartRenderer> renderers = Map.ofEntries(
        Map.entry("line", new SvgCartesianRenderer()),
        Map.entry("scatter", new SvgCartesianRenderer()),
        Map.entry("area", new SvgCartesianRenderer()),
        Map.entry("step", new SvgCartesianRenderer()),
        Map.entry("bubble", new SvgCartesianRenderer()),
        Map.entry("errorbar", new SvgCartesianRenderer()),
        Map.entry("ci_band", new SvgCartesianRenderer()),
        Map.entry("bar", new SvgBarRenderer()),
        Map.entry("pie", new SvgPieRenderer()),
        Map.entry("histogram", new SvgHistogramRenderer()),
        Map.entry("polar_bar", new SvgPolarRenderer()),
        Map.entry("polar_line", new SvgPolarRenderer()),
        Map.entry("polar_scatter", new SvgPolarRenderer()),
        Map.entry("radar", new SvgRadarRenderer()),
        Map.entry("heatmap", new SvgHeatmapRenderer()),
        Map.entry("boxplot", new SvgBoxplotRenderer()),
        Map.entry("violin", new SvgViolinRenderer()),
        Map.entry("candlestick", new SvgCandlestickRenderer()),
        Map.entry("funnel", new SvgFunnelRenderer()),
        Map.entry("sankey", new SvgSankeyRenderer()),
        Map.entry("sunburst", new SvgSunburstRenderer()),
        Map.entry("treemap", new SvgTreemapRenderer()),
        Map.entry("tree", new SvgTreeRenderer()),
        Map.entry("graph", new SvgGraphRenderer()),
        Map.entry("parallel", new SvgParallelRenderer()),
        Map.entry("theme_river", new SvgThemeRiverRenderer()),
        Map.entry("gauge", new SvgGaugeRenderer()),
        Map.entry("pairplot", new SvgPairplotRenderer()),
        Map.entry("jointplot", new SvgJointplotRenderer()),
        Map.entry("subplots", new SvgSubplotsRenderer()),
        Map.entry("qqplot", new SvgQqplotRenderer()),
        Map.entry("regplot", new SvgRegplotRenderer())
    );

    public SvgPlot() { this(800, 600); }
    public SvgPlot(int width, int height) { this(width, height, "default"); }
    public SvgPlot(int width, int height, String theme) {
        this.chartConfig = new ChartConfig(width, height);
        this.chartConfig.paddingLeft = 80;
        this.chartConfig.paddingRight = 50;
        this.chartConfig.paddingTop = 80;
        this.chartConfig.paddingBottom = 80;
        this.theme = theme;
    }

    public int getWidth() { return chartConfig.width; }
    public int getHeight() { return chartConfig.height; }
    public ChartConfig getChartConfig() { return chartConfig; }

    /**
     * 供 {@link SvgPlotFigureWindow} 等就地编辑序列展示名；与内部 {@link #seriesList} 为同一列表。
     */
    public List<SeriesData> mutableSeriesListForFigureUi() {
        finalizeFacetSubplotsIfNeeded();
        return seriesList;
    }

    private void abandonFacet() {
        facetMode = false;
        facetCells = null;
        pendingFacetLayout = null;
        pendingFacetColumnNames = null;
    }

    /** 单序列：子图模式写入当前格，否则替换 {@link #seriesList} */
    private void assignPrimaryOrFacet(SeriesData s, ChartType chartType) {
        currentChartType = chartType;
        if (facetMode && facetCells != null && facetCursor >= 0 && facetCursor < facetCells.length) {
            facetCells[facetCursor] = s;
            return;
        }
        seriesList.clear();
        seriesList.add(s);
    }

    /** 将子图格子合并为 type=subplots 的单一图层，供 {@link SvgSubplotsRenderer} 渲染 */
    private void finalizeFacetSubplotsIfNeeded() {
        if (!facetMode || facetCells == null) return;
        boolean any = false;
        for (SeriesData s : facetCells) {
            if (s != null) {
                any = true;
                break;
            }
        }
        if (!any) return;
        List<SeriesData> cellList = new ArrayList<>(Arrays.asList(facetCells));
        SeriesData holder = new SeriesData("Subplots", null, null, new PlotStyle(), "subplots");
        holder.extraData.put("subplotSeries", cellList);
        holder.extraData.put("grid", new int[]{facetRows, facetCols});
        if (pendingFacetLayout != null) {
            holder.extraData.put("facetLayout", pendingFacetLayout);
            if (pendingFacetColumnNames != null) {
                holder.extraData.put("columnNames", new ArrayList<>(pendingFacetColumnNames));
            }
        }
        pendingFacetLayout = null;
        pendingFacetColumnNames = null;
        seriesList.clear();
        seriesList.add(holder);
        facetMode = false;
        facetCells = null;
    }

    private static IVector<Double> matrixColumnForPairplot(IMatrix<?> m, int col) {
        int r = m.getRowNum();
        double[] buf = new double[r];
        for (int i = 0; i < r; i++) {
            buf[i] = m.get(i, col);
        }
        return Linalg.vector(buf);
    }

    private IVector<Double>[] kdeForPairplotCell(IVector<Double> col, int gridPoints, double bandwidthIn) {
        double bw = bandwidthIn <= 0 ? PlotKde.scottBandwidth(col) : bandwidthIn;
        return PlotKde.toVectors(PlotKde.evaluate(col, bw, gridPoints));
    }

    // ──────────────────────────────────────────────────────
    // 笛卡尔系：line / scatter / area / step / bubble / errorBar
    // ──────────────────────────────────────────────────────

    public SvgPlot line(IVector x, IVector y) {
        SeriesData s = new SeriesData("Line", x, (IVector<Double>) y, new PlotStyle(), "line");
        assignPrimaryOrFacet(s, ChartType.LINE);
        return this;
    }
    public SvgPlot line(IVector x) {
        SeriesData s = new SeriesData("Line", x, (IVector<Double>) x, new PlotStyle(), "line");
        assignPrimaryOrFacet(s, ChartType.LINE);
        return this;
    }
    public SvgPlot scatter(IVector x, IVector y) {
        SeriesData s = new SeriesData("Scatter", x, (IVector<Double>) y, new PlotStyle(), "scatter");
        assignPrimaryOrFacet(s, ChartType.SCATTER);
        return this;
    }
    public SvgPlot scatter(IVector x, IVector y, IVector sizes) {
        return bubble(x, y, sizes);
    }
    public SvgPlot area(IVector x, IVector y) {
        SeriesData s = new SeriesData("Area", x, (IVector<Double>) y, new PlotStyle(), "area");
        s.extraData.put("area", true);
        assignPrimaryOrFacet(s, ChartType.LINE);
        return this;
    }
    public SvgPlot step(IVector x, IVector y) {
        SeriesData s = new SeriesData("Step", x, (IVector<Double>) y, new PlotStyle(), "step");
        assignPrimaryOrFacet(s, ChartType.LINE);
        return this;
    }
    public SvgPlot bubble(IVector x, IVector y, IVector sizes) {
        SeriesData s = new SeriesData("Bubble", x, (IVector<Double>) y, new PlotStyle(), "bubble");
        s.extraData.put("sizes", sizes);
        assignPrimaryOrFacet(s, ChartType.SCATTER);
        return this;
    }
    public SvgPlot errorBar(IVector x, IVector y, IVector yErr) {
        SeriesData s = new SeriesData("ErrorBar", x, (IVector<Double>) y, new PlotStyle(), "errorbar");
        s.extraData.put("yerr", yErr);
        assignPrimaryOrFacet(s, ChartType.SCATTER);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 柱状图
    // ──────────────────────────────────────────────────────

    public SvgPlot bar(IVector y) {
        SeriesData s = new SeriesData("Bar", null, (IVector<Double>) y, new PlotStyle(), "bar");
        assignPrimaryOrFacet(s, ChartType.BAR);
        return this;
    }

    public SvgPlot groupedBar(IVector[] groups) {
        abandonFacet();
        currentChartType = ChartType.BAR;
        seriesList.clear();
        for (int i = 0; i < groups.length; i++) {
            SeriesData s = new SeriesData("Group" + i, null, (IVector<Double>) groups[i], new PlotStyle(), "bar");
            seriesList.add(s);
        }
        return this;
    }

    public SvgPlot stackedBar(IVector[] stacks) {
        abandonFacet();
        currentChartType = ChartType.BAR;
        seriesList.clear();
        for (int i = 0; i < stacks.length; i++) {
            SeriesData s = new SeriesData("Stack" + i, null, (IVector<Double>) stacks[i], new PlotStyle(), "bar");
            s.extraData.put("stacked", true);
            seriesList.add(s);
        }
        return this;
    }

    public SvgPlot horizontalBar(IVector y) {
        SeriesData s = new SeriesData("HBar", null, (IVector<Double>) y, new PlotStyle(), "bar");
        s.extraData.put("horizontal", true);
        assignPrimaryOrFacet(s, ChartType.BAR);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 置信带
    // ──────────────────────────────────────────────────────

    public SvgPlot ciBand(IVector x, IVector y, IVector yLow, IVector yHigh) {
        SeriesData s = new SeriesData("CiBand", x, (IVector<Double>) y, new PlotStyle(), "ci_band");
        s.extraData.put("yLow", yLow);
        s.extraData.put("yHigh", yHigh);
        assignPrimaryOrFacet(s, ChartType.LINE);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 极坐标变体
    // ──────────────────────────────────────────────────────

    public SvgPlot polarBar(IVector theta, IVector r) {
        SeriesData s = new SeriesData("PolarBar", theta, (IVector<Double>) r, new PlotStyle(), "polar_bar");
        assignPrimaryOrFacet(s, ChartType.POLAR_BAR);
        return this;
    }

    public SvgPlot polarScatter(IVector theta, IVector r) {
        SeriesData s = new SeriesData("PolarScatter", theta, (IVector<Double>) r, new PlotStyle(), "polar_scatter");
        assignPrimaryOrFacet(s, ChartType.POLAR_SCATTER);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 饼图
    // ──────────────────────────────────────────────────────

    public SvgPlot pie(IVector data) {
        SeriesData s = new SeriesData("Pie", null, (IVector<Double>) data, new PlotStyle(), "pie");
        assignPrimaryOrFacet(s, ChartType.PIE);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 极坐标
    // ──────────────────────────────────────────────────────

    public SvgPlot polar(IVector theta, IVector r) {
        SeriesData s = new SeriesData("Polar", theta, (IVector<Double>) r, new PlotStyle(), "polar_line");
        assignPrimaryOrFacet(s, ChartType.POLAR_LINE);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 雷达图
    // ──────────────────────────────────────────────────────

    public SvgPlot radar(IVector[] values, List<String> categories) {
        abandonFacet();
        currentChartType = ChartType.RADAR;
        seriesList.clear();
        for (IVector v : values) {
            SeriesData s = new SeriesData("Radar", null, (IVector<Double>) v, new PlotStyle(), "radar");
            s.labels = new ArrayList<>(categories);
            seriesList.add(s);
        }
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 热力图
    // ──────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public SvgPlot heatmap(IMatrix matrix) {
        SeriesData s = new SeriesData("Heatmap", null, null, new PlotStyle(), "heatmap");
        s.extraData.put("matrixData", matrix);
        assignPrimaryOrFacet(s, ChartType.HEATMAP);
        return this;
    }
    @SuppressWarnings("unchecked")
    public SvgPlot heatmap(IMatrix matrix, List<String> xLabels, List<String> yLabels) {
        SeriesData s = new SeriesData("Heatmap", null, null, new PlotStyle(), "heatmap");
        s.extraData.put("matrixData", matrix);
        s.extraData.put("xLabels", xLabels);
        s.extraData.put("yLabels", yLabels);
        assignPrimaryOrFacet(s, ChartType.HEATMAP);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 箱线图
    // ──────────────────────────────────────────────────────

    public SvgPlot boxplot(IVector data) { return boxplot(data, null); }
    public SvgPlot boxplot(IVector data, List<String> labels) {
        SeriesData series = new SeriesData("Boxplot", null, (IVector<Double>) data,
            new PlotStyle(), "boxplot");
        series.labels = labels != null ? labels : new ArrayList<>();
        assignPrimaryOrFacet(series, ChartType.BOXPLOT);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 直方图
    // ──────────────────────────────────────────────────────

    public SvgPlot histogram(IVector data) {
        SeriesData series = new SeriesData("Histogram", null, (IVector<Double>) data,
            new PlotStyle(), "histogram");
        if (pendingBins != null) {
            series.extraData.put("bins", pendingBins);
            pendingBins = null;
        }
        assignPrimaryOrFacet(series, ChartType.HIST);
        return this;
    }
    public SvgPlot histogram(IVector data, int bins) {
        pendingBins = bins;
        return histogram(data);
    }

    // ──────────────────────────────────────────────────────
    // 小提琴图
    // ──────────────────────────────────────────────────────

    public SvgPlot violinplot(IVector data) { return violinplot(data, (List<String>) null); }
    public SvgPlot violinplot(IVector data, List<String> labels) {
        SeriesData series = new SeriesData("Violin", null, (IVector<Double>) data,
            new PlotStyle(), "violin");
        series.labels = labels != null ? labels : new ArrayList<>();
        assignPrimaryOrFacet(series, ChartType.VIOLINPLOT);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // K线图
    // ──────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public SvgPlot candlestick(IMatrix data) { return candlestick(data, null); }
    @SuppressWarnings("unchecked")
    public SvgPlot candlestick(IMatrix data, List<String> dates) {
        SeriesData s = new SeriesData("Candlestick", null, null, new PlotStyle(), "candlestick");
        s.extraData.put("matrixData", data);
        if (dates != null) s.labels = new ArrayList<>(dates);
        assignPrimaryOrFacet(s, ChartType.CANDLESTICK);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 漏斗图
    // ──────────────────────────────────────────────────────

    public SvgPlot funnel(IVector y) {
        SeriesData s = new SeriesData("Funnel", null, (IVector<Double>) y, new PlotStyle(), "funnel");
        assignPrimaryOrFacet(s, ChartType.FUNNEL);
        return this;
    }
    public SvgPlot funnel(IVector y, List<String> labels) {
        SeriesData s = new SeriesData("Funnel", null, (IVector<Double>) y, new PlotStyle(), "funnel");
        s.labels = labels != null ? labels : new ArrayList<>();
        assignPrimaryOrFacet(s, ChartType.FUNNEL);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 桑基图
    // ──────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public SvgPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        SeriesData s = new SeriesData("Sankey", null, null, new PlotStyle(), "sankey");
        s.extraData.put("nodes", nodes);
        s.extraData.put("links", links);
        assignPrimaryOrFacet(s, ChartType.SANKEY);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 旭日图
    // ──────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public SvgPlot sunburst(Map<String, Object> tree) {
        SeriesData s = new SeriesData("Sunburst", null, null, new PlotStyle(), "sunburst");
        s.extraData.put("treeData", tree);
        assignPrimaryOrFacet(s, ChartType.SUNBURST);
        return this;
    }

    @Override public IPlot sunburst(List<Map<String, Object>> data) {
        if (!data.isEmpty()) sunburst(data.get(0));
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 矩形树图
    // ──────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public SvgPlot treemap(List<Map<String, Object>> data) {
        SeriesData s = new SeriesData("Treemap", null, null, new PlotStyle(), "treemap");
        s.extraData.put("treeData", data);
        assignPrimaryOrFacet(s, ChartType.TREEMAP);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 树图
    // ──────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public SvgPlot tree(Map<String, Object> treeData, String rootLabel) {
        SeriesData s = new SeriesData("Tree", null, null, new PlotStyle(), "tree");
        s.extraData.put("treeData", treeData);
        assignPrimaryOrFacet(s, ChartType.TREE);
        return this;
    }

    @Override public IPlot tree(List<Map<String, Object>> data) {
        if (data == null || data.isEmpty()) return this;
        Map<String, Object> wrap = new LinkedHashMap<>();
        wrap.put("name", "");
        wrap.put("children", new ArrayList<>(data));
        tree(wrap, "");
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 关系图
    // ──────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public SvgPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        SeriesData s = new SeriesData("Graph", null, null, new PlotStyle(), "graph");
        s.extraData.put("nodes", nodes);
        s.extraData.put("links", links);
        assignPrimaryOrFacet(s, ChartType.GRAPH);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 平行坐标图
    // ──────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public SvgPlot parallel(IMatrix data, List<String> dimensions) {
        SeriesData s = new SeriesData("Parallel", null, null, new PlotStyle(), "parallel");
        s.extraData.put("matrixData", data);
        s.extraData.put("dimensions", dimensions);
        assignPrimaryOrFacet(s, ChartType.PARALLEL);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 主题河流图
    // ──────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    public SvgPlot themeRiver(List<Map<String, Object>> data) {
        SeriesData s = new SeriesData("ThemeRiver", null, null, new PlotStyle(), "theme_river");
        s.extraData.put("riverData", data);
        assignPrimaryOrFacet(s, ChartType.THEME_RIVER);
        return this;
    }

    @Override public IPlot themeRiver(List<Map<String, Object>> data, List<String> categories) {
        themeRiver(data); return this;
    }

    // ──────────────────────────────────────────────────────
    // 仪表盘
    // ──────────────────────────────────────────────────────

    public SvgPlot gaugeWithLabel(double value, double min, double max, String label) {
        SeriesData s = new SeriesData("Gauge", null, null, new PlotStyle(), "gauge");
        s.extraData.put("value", value);
        s.extraData.put("min", min);
        s.extraData.put("max", max);
        s.extraData.put("label", label);
        assignPrimaryOrFacet(s, ChartType.GAUGE);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 链式配置方法
    // ──────────────────────────────────────────────────────

    public SvgPlot title(String titleText) { chartConfig.title = titleText; return this; }
    public SvgPlot title(String titleText, String subtitleText) {
        chartConfig.title = titleText; chartConfig.subtitle = subtitleText; return this;
    }
    public SvgPlot xlabel(String name) { chartConfig.xlabel = name; return this; }
    public SvgPlot ylabel(String name) { chartConfig.ylabel = name; return this; }
    public SvgPlot size(int width, int height) { chartConfig.width = width; chartConfig.height = height; return this; }

    // ========== Legend fluent API ==========

    @Override
    public IPlot legend() {
        return legend(LegendPositions.BEST);
    }

    @Override
    public IPlot legend(String position) {
        this.legendPosition = LegendPositions.sanitize(position);
        return this;
    }

    @Override
    public IPlot legend(boolean show) {
        chartConfig.showLegend = show;
        return this;
    }

    public SvgPlot theme(String themeName) { this.theme = themeName; return this; }
    public String getTheme() { return theme; }
    @Override
    public SvgPlot show() {
        if (PlotHostBridge.trySendToIde(this)) {
            return this;
        }
        SvgPlotFigureWindow.open(this);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // 导出方法
    // ──────────────────────────────────────────────────────

    /** 生成真正的矢量SVG文件 */
    @Override
    public SvgPlot saveAsSvg(String filename) {
        finalizeFacetSubplotsIfNeeded();
        if (seriesList.isEmpty()) return this;
        try {
            Files.writeString(Path.of(filename), toSvgString(), StandardCharsets.UTF_8);
        } catch (Exception e) {
            throw new RuntimeException("SVG保存失败: " + e.getMessage(), e);
        }
        return this;
    }

    public String toSvgString() {
        finalizeFacetSubplotsIfNeeded();
        chartConfig.legendPosition = this.legendPosition;
        if (seriesList.isEmpty()) return placeholderSvg();
        SeriesData series = seriesList.get(0);
        AbstractSvgChartRenderer renderer = renderers.get(series.type);
        if (renderer != null) {
            return renderer.renderMulti(seriesList, chartConfig, theme);
        }
        return placeholderSvg();
    }

    private String placeholderSvg() {
        return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" +
               "<svg xmlns=\"http://www.w3.org/2000/svg\" " +
               "width=\"" + chartConfig.width + "\" height=\"" + chartConfig.height + "\">\n" +
               "<rect width=\"100%\" height=\"100%\" fill=\"#f9fafb\"/>\n" +
               "<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dy=\".3em\" " +
               "font-family=\"Arial\" font-size=\"14\" fill=\"#6b7280\">" +
               "SVG暂不支持此图表类型: " + currentChartType + "</text>\n</svg>\n";
    }

    // ──────────────────────────────────────────────────────
    // IPlot: Style-string and PlotStyle overloads
    // ──────────────────────────────────────────────────────

    @Override public IPlot line(IVector x, IVector y, String styleString) {
        line(x, y); return this;
    }
    @Override public IPlot line(IVector x, IVector y, PlotStyle style) {
        line(x, y); return this;
    }
    @Override public IPlot line(IVector y, String styleString) {
        line(y); return this;
    }
    @Override public IPlot line(IVector y, PlotStyle style) {
        line(y); return this;
    }
    @Override public IPlot line(IVector x, IVector y, List<String> hue) {
        return line(x, y);
    }
    @Override public IPlot line(IVector x, IVector y, List<String> hue, List<String> styleGroup) {
        return line(x, y);
    }

    @Override public IPlot scatter(IVector x, IVector y, String styleString) {
        scatter(x, y); return this;
    }
    @Override public IPlot scatter(IVector x, IVector y, PlotStyle style) {
        scatter(x, y); return this;
    }
    @Override public IPlot scatter(IVector x, IVector y, List<String> hue) {
        return scatter(x, y);
    }

    @Override public IPlot pie(IVector x, String styleString) {
        pie(x); return this;
    }
    @Override public IPlot pie(IVector x, PlotStyle style) {
        pie(x); return this;
    }
    @Override public IPlot pie(IVector x, List<String> labels, PlotStyle style) {
        return pie(x);
    }
    @Override public IPlot pie(IVector x, List<String> labels, String styleString) {
        return pie(x);
    }

    @Override public IPlot bar(IVector y, String styleString) {
        bar(y); return this;
    }
    @Override public IPlot bar(IVector y, PlotStyle style) {
        bar(y); return this;
    }
    @Override public IPlot bar(IVector y, List<String> hue, PlotStyle style) {
        return bar(y);
    }
    @Override public IPlot bar(IVector y, List<String> hue, String styleString) {
        return bar(y);
    }
    @Override public IPlot bar(List<String> xticks, IVector y) {
        return bar(y);
    }
    @Override public IPlot bar(List<String> xticks, IVector y, List<String> hue) {
        return bar(y);
    }

    @Override public IPlot hist(IVector x, boolean fittingLine) {
        return histogram(x);
    }
    @Override public IPlot hist(IVector x, boolean fittingLine, String styleString) {
        return histogram(x);
    }
    @Override public IPlot hist(IVector x, boolean fittingLine, PlotStyle style) {
        return histogram(x);
    }
    @Override public IPlot hist(IVector x, boolean fittingLine, PlotStyle style, Integer bins) {
        return histogram(x, bins != null ? bins : 10);
    }

    @Override public IPlot polarBar(IVector data, List<String> categories, String styleString) {
        return polarBar(data, categories);
    }
    @Override public IPlot polarBar(IVector data, List<String> categories, PlotStyle style) {
        return polarBar(data, categories);
    }

    @Override public IPlot polarLine(IVector data, List<String> categories, String styleString) {
        return polarLine(data, categories);
    }
    @Override public IPlot polarLine(IVector data, List<String> categories, PlotStyle style) {
        return polarLine(data, categories);
    }

    @Override public IPlot polarScatter(IVector data, List<String> categories, String styleString) {
        return polarScatter(data, categories);
    }
    @Override public IPlot polarScatter(IVector data, List<String> categories, PlotStyle style) {
        return polarScatter(data, categories);
    }

    @Override public IPlot boxplot(IVector data, List<String> labels, String styleString) {
        boxplot(data, labels); return this;
    }
    @Override public IPlot boxplot(IVector data, List<String> labels, PlotStyle style) {
        boxplot(data, labels); return this;
    }

    @Override public IPlot violinplot(IVector data, String styleString) {
        violinplot(data, (List<String>) null); return this;
    }
    @Override public IPlot violinplot(IVector data, PlotStyle style) {
        violinplot(data, (List<String>) null); return this;
    }
    @Override public IPlot violinplot(IVector data, List<String> labels, String styleString) {
        violinplot(data, labels); return this;
    }
    @Override public IPlot violinplot(IVector data, List<String> labels, PlotStyle style) {
        violinplot(data, labels); return this;
    }

    @Override public IPlot candlestick(IMatrix data, List<String> dates, String styleString) {
        candlestick(data, dates); return this;
    }
    @Override public IPlot candlestick(IMatrix data, List<String> dates, PlotStyle style) {
        candlestick(data, dates); return this;
    }

    @Override public IPlot funnel(IVector data, List<String> labels, String styleString) {
        funnel(data, labels); return this;
    }
    @Override public IPlot funnel(IVector data, List<String> labels, PlotStyle style) {
        funnel(data, labels); return this;
    }

    @Override public IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString) {
        sankey(nodes, links); return this;
    }
    @Override public IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        sankey(nodes, links); return this;
    }

    @Override public IPlot sunburst(List<Map<String, Object>> data, String styleString) {
        sunburst(data); return this;
    }
    @Override public IPlot sunburst(List<Map<String, Object>> data, PlotStyle style) {
        sunburst(data); return this;
    }

    @Override public IPlot themeRiver(List<Map<String, Object>> data, List<String> categories, String styleString) {
        themeRiver(data); return this;
    }
    @Override public IPlot themeRiver(List<Map<String, Object>> data, List<String> categories, PlotStyle style) {
        themeRiver(data); return this;
    }

    @Override public IPlot tree(List<Map<String, Object>> data, String styleString) {
        return tree(data);
    }
    @Override public IPlot tree(List<Map<String, Object>> data, PlotStyle style) {
        return tree(data);
    }

    @Override public IPlot treemap(List<Map<String, Object>> data, String styleString) {
        treemap(data); return this;
    }
    @Override public IPlot treemap(List<Map<String, Object>> data, PlotStyle style) {
        treemap(data); return this;
    }

    @Override public IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString) {
        graph(nodes, links); return this;
    }
    @Override public IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        graph(nodes, links); return this;
    }

    @Override public IPlot parallel(IMatrix data, List<String> dimensions, String styleString) {
        parallel(data, dimensions); return this;
    }
    @Override public IPlot parallel(IMatrix data, List<String> dimensions, PlotStyle style) {
        parallel(data, dimensions); return this;
    }

    @Override public IPlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels, String styleString) {
        heatmap(data, xLabels, yLabels); return this;
    }
    @Override public IPlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels, PlotStyle style) {
        heatmap(data, xLabels, yLabels); return this;
    }

    @Override public IPlot radar(IVector data, List<String> indicators, String styleString) {
        return radar(data, indicators);
    }
    @Override public IPlot radar(IVector data, List<String> indicators, PlotStyle style) {
        return radar(data, indicators);
    }

    // gauge methods - IPlot gauge(value,max,min) not supported (existing gauge(value,min,max,label) exists)

    // ──────────────────────────────────────────────────────
    // IPlot: Style/Theme system methods
    // ──────────────────────────────────────────────────────

    @Override public IPlot setDefaultStyle(PlotStyle style) { return this; }
    @Override public IPlot setPalette(String paletteName) { return this; }
    @Override public IPlot enableStyleSystem(boolean enabled) { return this; }
    @Override public IPlot enableThemeSystem(boolean enabled) { return this; }
    @Override public IPlot applyTheme(String themeName) { this.theme = themeName; return this; }
    @Override public IPlot registerTheme(String themeName, com.yishape.lab.math.plot.echarts.EchartsThemeManager.CustomTheme theme) { return this; }
    @Override public IPlot createGradientTheme(String themeName, String startColor, String endColor, String backgroundColor) { return this; }
    @Override public IPlot createMonochromeTheme(String themeName, String baseColor, String backgroundColor) { return this; }

    // ──────────────────────────────────────────────────────
    // IPlot: Seaborn/Matplotlib-style Cartesian extensions
    // ──────────────────────────────────────────────────────

    @Override public IPlot xscale(PlotAxisScale scale) { return this; }
    @Override public IPlot yscale(PlotAxisScale scale) { return this; }
    @Override public IPlot y2label(String label) {
        chartConfig.y2AxisLabel = label != null ? label : "";
        return this;
    }

    @Override public IPlot barh(List<String> categories, IVector values) {
        SeriesData series = new SeriesData("BarH", null, (IVector<Double>) values, new PlotStyle(), "bar");
        series.labels = categories != null ? new ArrayList<>(categories) : new ArrayList<>();
        series.extraData.put("horizontal", true);
        assignPrimaryOrFacet(series, ChartType.BAR);
        return this;
    }

    @Override public IPlot barStacked(List<String> categories, IMatrix values, List<String> layerNames) {
        abandonFacet();
        if (values == null || categories == null) {
            throw new PlotException("categories 与 values 不能为 null");
        }
        int rows = values.getRowNum();
        int cols = values.getColNum();
        if (cols != categories.size()) {
            throw new PlotException("类别数量必须与矩阵列数一致");
        }
        String[] palette = new JavaFxThemeManager(theme).getColorPalette();
        currentChartType = ChartType.BAR;
        seriesList.clear();
        for (int r = 0; r < rows; r++) {
            double[] colVals = new double[cols];
            for (int c = 0; c < cols; c++) {
                colVals[c] = values.get(r, c);
            }
            String name = layerNames != null && r < layerNames.size()
                ? layerNames.get(r) : ("L" + (r + 1));
            PlotStyle style = PlotStyle.defaultStyle();
            style.setColor(palette[r % palette.length]);
            SeriesData series = new SeriesData(name, null, Linalg.vector(colVals), style, "bar");
            series.labels = new ArrayList<>(categories);
            series.extraData.put("stacked", true);
            seriesList.add(series);
        }
        return this;
    }

    @Override public IPlot errorbar(IVector x, IVector y, IVector yerr) {
        errorBar(x, y, yerr); return this;
    }

    @Override public IPlot regplot(IVector x, IVector y) { return regplot(x, y, false); }
    @Override public IPlot regplot(IVector x, IVector y, boolean confidenceBand) {
        abandonFacet();
        int n = x.length();
        double xm = (x.meanValue());
        double ym = (y.meanValue());
        double sx = (x.stdValue());
        double cov = 0;
        for (int i = 0; i < n; i++) {
            cov += ((x.get(i)) - xm) * ((y.get(i)) - ym);
        }
        cov /= (n - 1);
        double slope = cov / (sx * sx + 1e-10);
        double intercept = ym - slope * xm;

        double xMin = (x.minValue());
        double xMax = (x.maxValue());
        IVector<Double> xLine = Linalg.linspace(xMin, xMax, 50);
        int nLine = xLine.length();
        double[] yLineArr = new double[nLine];
        for (int i = 0; i < nLine; i++) {
            yLineArr[i] = slope * (xLine.get(i)) + intercept;
        }

        SeriesData s = new SeriesData("Regplot", null, null, new PlotStyle(), "regplot");
        double[] xArr = new double[n];
        double[] yArr = new double[n];
        for (int i = 0; i < n; i++) {
            xArr[i] = (x.get(i));
            yArr[i] = (y.get(i));
        }
        s.extraData.put("xData", xArr);
        s.extraData.put("yData", yArr);
        s.extraData.put("xLine", xLine.toDoubleArray());
        s.extraData.put("yLine", yLineArr);

        if (confidenceBand && n > 2) {
            double rss = 0;
            for (int i = 0; i < n; i++) {
                double xi = (x.get(i));
                double yi = (y.get(i));
                double yhat = slope * xi + intercept;
                rss += (yi - yhat) * (yi - yhat);
            }
            double residualSe = Math.sqrt(rss / (n - 2));
            double ssX = 0;
            for (int i = 0; i < n; i++) {
                double dx = (x.get(i)) - xm;
                ssX += dx * dx;
            }
            double sxx = ssX + 1e-15;
            double[] yErrLow = new double[nLine];
            double[] yErrHigh = new double[nLine];
            for (int i = 0; i < nLine; i++) {
                double xi = (xLine.get(i));
                double seMean = residualSe * Math.sqrt(1.0 / n + (xi - xm) * (xi - xm) / sxx);
                double half = 1.96 * seMean;
                yErrLow[i] = yLineArr[i] - half;
                yErrHigh[i] = yLineArr[i] + half;
            }
            s.extraData.put("yErrLow", yErrLow);
            s.extraData.put("yErrHigh", yErrHigh);
        } else if (confidenceBand) {
            double yGlobalMin = Double.MAX_VALUE, yGlobalMax = -Double.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                double yv = (y.get(i));
                yGlobalMin = Math.min(yGlobalMin, yv);
                yGlobalMax = Math.max(yGlobalMax, yv);
            }
            double yRange = yGlobalMax - yGlobalMin;
            if (yRange == 0) yRange = 1;
            double[] yErrLow = new double[nLine];
            double[] yErrHigh = new double[nLine];
            for (int i = 0; i < nLine; i++) {
                yErrLow[i] = yLineArr[i] - yRange * 0.12;
                yErrHigh[i] = yLineArr[i] + yRange * 0.12;
            }
            s.extraData.put("yErrLow", yErrLow);
            s.extraData.put("yErrHigh", yErrHigh);
        }
        currentChartType = ChartType.REGPLOT;
        seriesList.clear();
        seriesList.add(s);
        title("回归图");
        return this;
    }

    @Override public IPlot qqplot(IVector data) {
        int n = data.length();
        double[] sorted = new double[n];
        for (int i = 0; i < n; i++) sorted[i] = (data.get(i));
        java.util.Arrays.sort(sorted);
        SeriesData s = new SeriesData("Qqplot", null, null, new PlotStyle(), "qqplot");
        s.extraData.put("sortedData", sorted);
        assignPrimaryOrFacet(s, ChartType.QQPLOT);
        title("Q-Q Plot");
        return this;
    }

    @Override public IPlot kdeplot(IVector data, int gridPoints, double bandwidth) {
        int n = data.length();
        double min = (data.minValue());
        double max = (data.maxValue());
        double bw = bandwidth > 0 ? bandwidth : 1.06 * (data.stdValue()) * Math.pow(n, -0.2);
        int gp = gridPoints > 0 ? gridPoints : 256;
        IVector<Double> xGrid = Linalg.linspace(min - 3 * bw, max + 3 * bw, gp);
        double[] density = new double[gp];
        for (int j = 0; j < gp; j++) {
            double xj = (xGrid.get(j));
            double sum = 0;
            for (int i = 0; i < n; i++) {
                double xi = (data.get(i));
                sum += Math.exp(-0.5 * Math.pow((xj - xi) / bw, 2));
            }
            density[j] = sum / (n * bw * Math.sqrt(2 * Math.PI));
        }
        IVector<Double> yDense = IVector.of(density);
        area(xGrid, yDense);
        return this;
    }

    @Override public IPlot subplots(int rows, int cols) {
        if (rows < 1 || cols < 1) {
            throw new PlotException("subplots: 行列数须至少为 1");
        }
        pendingFacetLayout = null;
        pendingFacetColumnNames = null;
        facetMode = true;
        facetRows = rows;
        facetCols = cols;
        facetCells = new SeriesData[rows * cols];
        facetCursor = 0;
        seriesList.clear();
        return this;
    }

    @Override public IPlot subplot(int row, int col) {
        if (!facetMode || facetCells == null) {
            throw new PlotException("请先调用 subplots(rows, cols)");
        }
        if (row < 0 || col < 0 || row >= facetRows || col >= facetCols) {
            throw new PlotException("subplot(row,col) 越界");
        }
        facetCursor = row * facetCols + col;
        return this;
    }

    public SvgPlot addSubplotSeries(SeriesData s) {
        if (facetMode && facetCells != null && facetCursor >= 0 && facetCursor < facetCells.length) {
            facetCells[facetCursor] = s;
        }
        return this;
    }

    @Override public IPlot pairplot(IMatrix data, List<String> columnNames, PairplotDiagonal diagonal) {
        abandonFacet();
        if (data == null || data.getColNum() < 1 || data.getRowNum() < 1) {
            throw new PlotException("pairplot: 数据矩阵不能为空");
        }
        int p = data.getColNum();
        List<String> names = new ArrayList<>();
        for (int j = 0; j < p; j++) {
            names.add(columnNames != null && j < columnNames.size()
                ? columnNames.get(j) : ("x" + j));
        }
        String[] palette = new JavaFxThemeManager(theme).getColorPalette();
        facetMode = true;
        facetRows = p;
        facetCols = p;
        facetCells = new SeriesData[p * p];
        PairplotDiagonal d = diagonal != null ? diagonal : PairplotDiagonal.KDE;
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                int idx = i * p + j;
                if (i == j) {
                    IVector<Double> col = matrixColumnForPairplot(data, i);
                    if (d == PairplotDiagonal.NONE) {
                        facetCells[idx] = new SeriesData(names.get(i), null, null, new PlotStyle(), "empty");
                    } else if (d == PairplotDiagonal.KDE) {
                        IVector<Double>[] xy = kdeForPairplotCell(col, 128, 0.0);
                        PlotStyle st = PlotStyle.defaultStyle();
                        st.setColor(palette[i % palette.length]);
                        facetCells[idx] = new SeriesData(names.get(i), xy[0], xy[1], st, "line");
                    } else {
                        PlotStyle st = PlotStyle.defaultStyle();
                        SeriesData s = new SeriesData(names.get(i), null, col, st, "histogram");
                        s.extraData.put("fittingLine", false);
                        facetCells[idx] = s;
                    }
                } else {
                    IVector<Double> xcol = matrixColumnForPairplot(data, j);
                    IVector<Double> ycol = matrixColumnForPairplot(data, i);
                    PlotStyle st = PlotStyle.defaultStyle().marker("o").markerSize(4);
                    st.setColor(palette[0]);
                    String nm = names.get(j) + " vs " + names.get(i);
                    facetCells[idx] = new SeriesData(nm, xcol, ycol, st, "scatter");
                }
            }
        }
        pendingFacetLayout = "pairplot";
        pendingFacetColumnNames = names;
        finalizeFacetSubplotsIfNeeded();
        currentChartType = ChartType.PAIRPLOT;
        chartConfig.showLegend = false;
        return this;
    }

    @Override public IPlot jointplot(IVector x, IVector y, JointplotMarginal marginal) {
        abandonFacet();
        if (x == null || y == null) {
            throw new PlotException("jointplot: x 与 y 不能为 null");
        }
        if (x.length() != y.length()) {
            throw new PlotException("jointplot: x 与 y 长度须相同");
        }
        double[] xArr = new double[x.length()];
        double[] yArr = new double[y.length()];
        for (int i = 0; i < x.length(); i++) {
            xArr[i] = (x.get(i));
            yArr[i] = (y.get(i));
        }
        SeriesData holder = new SeriesData("Jointplot", null, null, new PlotStyle(), "subplots");
        holder.extraData.put("facetLayout", "jointplot");
        holder.extraData.put("xData", xArr);
        holder.extraData.put("yData", yArr);
        holder.extraData.put("marginal", marginal != null ? marginal.name().toLowerCase() : "hist");
        seriesList.clear();
        seriesList.add(holder);
        currentChartType = ChartType.JOINTPLOT;
        chartConfig.showLegend = false;
        return this;
    }

    @Override public IPlot lineWithSecondaryY(IVector x, IVector yLeft, IVector yRight) {
        abandonFacet();
        currentChartType = ChartType.LINE;
        seriesList.clear();
        SeriesData left = new SeriesData("Y1", (IVector<Double>) x, (IVector<Double>) yLeft, new PlotStyle(), "line");
        left.yAxisIndex = 0;
        seriesList.add(left);
        SeriesData right = new SeriesData("Y2", (IVector<Double>) x, (IVector<Double>) yRight, new PlotStyle(), "line");
        right.yAxisIndex = 1;
        seriesList.add(right);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // IPlot: Advanced chart methods (IPlot signatures)
    // ──────────────────────────────────────────────────────

    @Override public IPlot radar(IVector data, List<String> indicators) {
        IVector[] arr = new IVector[] { data };
        radar(arr, indicators);
        return this;
    }

    @Override public IPlot gauge(double value, double max, double min) {
        return (IPlot) gaugeWithLabel(value, max, min, null);
    }

    @Override public IPlot gauge(double value, double max, double min, String styleString) {
        return gauge(value, max, min);
    }
    @Override public IPlot gauge(double value, double max, double min, PlotStyle style) {
        return gauge(value, max, min);
    }

    @Override public IPlot polarBar(IVector data, List<String> categories) {
        int n = data.length();
        if (n <= 0) return this;
        IVector theta = Linalg.linspace(0.0, 2 * Math.PI * (n - 1) / n, n);
        polarBar(theta, data);
        return this;
    }
    @Override public IPlot polarLine(IVector data, List<String> categories) {
        int n = data.length();
        IVector theta = Linalg.linspace(0.0, 2 * Math.PI * (n - 1) / n, n);
        polar(theta, data);
        return this;
    }
    @Override public IPlot polarScatter(IVector data, List<String> categories) {
        int n = data.length();
        IVector theta = Linalg.linspace(0.0, 2 * Math.PI * (n - 1) / n, n);
        polarScatter(theta, data);
        return this;
    }

    // ──────────────────────────────────────────────────────
    // IPlot: Export methods
    // ──────────────────────────────────────────────────────

    @Override public IPlot saveAsHtml(String filename) {
        throw new UnsupportedOperationException("SVG backend does not support HTML export");
    }
    @Override public IPlot saveAsPng(String filename) {
        throw new UnsupportedOperationException("SVG backend does not support PNG export");
    }

    @Override
    public IPlot saveAsPdf(String filename) {
        throw new PlotException(
            "SvgPlot 未实现编程式 saveAsPdf；请使用 show() 窗口中「导出 → PDF（视图快照）」，或使用 saveAsSvg() 获得矢量图后再转换。");
    }

    @Override public String toBase64Svg() {
        return Base64.getEncoder().encodeToString(toSvgString().getBytes(StandardCharsets.UTF_8));
    }
    @Override public String toBase64Png() {
        throw new UnsupportedOperationException("SVG backend does not support PNG encoding");
    }
    @Override public String toHtml() {
        throw new UnsupportedOperationException("SVG backend does not support HTML export");
    }
    @Override public String toJson() {
        return "{}";
    }

    // ──────────────────────────────────────────────────────
    // IPlot: Configuration methods
    // ──────────────────────────────────────────────────────

    @Override public void setTitle(String titleText) { chartConfig.title = titleText; }
    @Override public void setTitle(String titleText, String subtitleText) {
        chartConfig.title = titleText; chartConfig.subtitle = subtitleText;
    }
    @Override public void setXlabel(String name) { chartConfig.xlabel = name; }
    @Override public void setYlabel(String name) { chartConfig.ylabel = name; }
    @Override public void setXticks(com.yishape.lab.math.plot.AxisTicks xticks) { }
    @Override public void setYticks(com.yishape.lab.math.plot.AxisTicks yticks) { }
}
