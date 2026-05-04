package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.AxisTicks;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.PlotAxisScale;
import com.yishape.lab.math.plot.ColorPalette;
import com.yishape.lab.math.plot.echarts.EchartsThemeManager;
import com.yishape.lab.math.plot.SeabornStyleMapper;
import com.yishape.lab.math.plot.StyleExpression;
import com.yishape.lab.math.plot.JointplotMarginal;
import com.yishape.lab.math.plot.PairplotDiagonal;
import com.yishape.lab.math.plot.PlotException;
import com.yishape.lab.math.plot.PlotKde;
import com.yishape.lab.math.plot.PlotStats;
import com.yishape.lab.math.plot.PlotStyle;
import com.yishape.lab.math.plot.javafx.renderers.*;

import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Stage;

import javax.imageio.ImageIO;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 基于JavaFX Canvas的图表功能实现（重构版）
 * 采用组件化架构，支持多种图表类型、主题、动画和交互
 * 
 * @author lteb2
 */
public class JavaFxPlot implements IPlot {
    
    private static final Logger log = LoggerFactory.getLogger(JavaFxPlot.class);
    
    // ========== 核心组件 ==========
    private JavaFxThemeManager themeManager;
    private JavaFxChartRenderer.ChartConfig chartConfig;
    private JavaFxInteractionHandler interactionHandler;
    private JavaFxChartRenderer currentRenderer;

    /** 与 {@link #RENDERERS} 的 key 一致；切换图表类型时清空 {@link #seriesList} */
    private String activeRendererKey;
    
    // ========== 数据存储 ==========
    private List<JavaFxChartRenderer.SeriesData> seriesList;
    private ChartType currentChartType;
    private Map<String, Object> chartData;

    /**
     * 与 ECharts 实现一致的样式/主题开关与默认序列样式。
     */
    private PlotStyle defaultSeriesStyle = PlotStyle.defaultStyle();
    private boolean useSeriesStyleSystem = true;
    private boolean usePlotThemeSystem = true;
    private final SeabornStyleMapper seabornStyleMapper = new SeabornStyleMapper();

    private static final class FacetPane {
        String rendererKey = "cartesian";
        final List<JavaFxChartRenderer.SeriesData> data = new ArrayList<>();
    }

    private boolean facetMode;
    private int facetRows = 1;
    private int facetCols = 1;
    private int facetCursor;
    private List<FacetPane> facetPanes;

    private boolean jointplotMode;
    private FacetPane jointTop;
    private FacetPane jointMain;
    private FacetPane jointRight;
    
    // ========== JavaFX组件 ==========
    private Canvas canvas;
    private GraphicsContext gc;
    private Stage stage;
    private Scene scene;
    private BorderPane rootPane;
    
    // ========== 图表类型枚举 ==========
    public enum ChartType {
        LINE, SCATTER, PIE, BAR, HIST,
        POLAR_BAR, POLAR_LINE, POLAR_SCATTER,
        BOXPLOT, VIOLINPLOT, CANDLESTICK,
        FUNNEL, SANKEY, SUNBURST, THEME_RIVER,
        TREE, TREEMAP, GRAPH, PARALLEL,
        HEATMAP, RADAR, GAUGE, PAIRPLOT, JOINTPLOT, SUBPLOTS, QQPLOT, REGPLOT, NONE
    }
    
    // ========== 渲染器注册表 ==========
    private static final Map<String, JavaFxChartRenderer> RENDERERS = new HashMap<>();
    static {
        RENDERERS.put("cartesian", new CartesianComboRenderer());
        RENDERERS.put("line", RENDERERS.get("cartesian"));
        RENDERERS.put("scatter", RENDERERS.get("cartesian"));
        RENDERERS.put("bar", new BarChartRenderer());
        RENDERERS.put("pie", new PieChartRenderer());
        RENDERERS.put("histogram", new HistogramRenderer());
        RENDERERS.put("polar_bar", new PolarChartRenderer("bar"));
        RENDERERS.put("polar_line", new PolarChartRenderer("line"));
        RENDERERS.put("polar_scatter", new PolarChartRenderer("scatter"));
        RENDERERS.put("radar", new RadarChartRenderer());
        RENDERERS.put("heatmap", new HeatmapRenderer());
        RENDERERS.put("boxplot", new StatisticalChartRenderer("boxplot"));
        RENDERERS.put("violin", new StatisticalChartRenderer("violin"));
        RENDERERS.put("candlestick", new StatisticalChartRenderer("candlestick"));
        RENDERERS.put("funnel", new ComplexChartRenderer("funnel"));
        RENDERERS.put("sankey", new ComplexChartRenderer("sankey"));
        RENDERERS.put("sunburst", new ComplexChartRenderer("sunburst"));
        RENDERERS.put("treemap", new ComplexChartRenderer("treemap"));
        RENDERERS.put("tree", new ComplexChartRenderer("tree"));
        RENDERERS.put("graph", new ComplexChartRenderer("graph"));
        RENDERERS.put("parallel", new ComplexChartRenderer("parallel"));
        RENDERERS.put("theme_river", new ComplexChartRenderer("themeRiver"));
        RENDERERS.put("gauge", new GaugeRenderer());
    }
    
    // ========== 构造函数 ==========
    public JavaFxPlot() {
        this(800, 600);
    }
    
    public JavaFxPlot(int width, int height) {
        // 与 kde/regplot/pairplot 等扩展图默认观感一致：seaborn-muted 调色 + 浅底软网格
        this(width, height, JavaFxThemeManager.THEME_SEABORN);
    }
    
    public JavaFxPlot(int width, int height, String theme) {
        this.themeManager = new JavaFxThemeManager(theme);
        this.chartConfig = new JavaFxChartRenderer.ChartConfig(width, height);
        this.chartConfig.theme = theme; // 设置主题名称
        this.seriesList = new ArrayList<>();
        this.currentChartType = ChartType.NONE;
        this.chartData = new HashMap<>();
        
        initializeCanvas();
        syncSeabornPaletteFromTheme();
        syncDefaultSeriesStyleFromTheme();
    }

    /**
     * 默认序列样式与当前主题调色板对齐（线色/标记色、略加粗的线宽与圆角端点），避免仍用全局 {@link PlotStyle#defaultStyle()} 的 ECharts 蓝。
     */
    private void syncDefaultSeriesStyleFromTheme() {
        String[] pal = themeManager.getColorPalette();
        PlotStyle s = new PlotStyle(PlotStyle.defaultStyle());
        if (pal != null && pal.length > 0) {
            String c0 = pal[0];
            s.setColor(c0);
            s.setFaceColor(c0);
            s.setMarkerColor(c0);
            String rim = pal[Math.min(7, pal.length - 1)];
            s.setMarkerEdgeColor(rim);
            s.setMarkerEdgeWidth(0.85);
        }
        s.setLineWidth(2.35);
        s.setLineCap("round");
        s.setLineJoin("round");
        s.setMarkerSize(6.5);
        defaultSeriesStyle = s;
    }

    private void initializeCanvas() {
        this.canvas = new Canvas(chartConfig.width, chartConfig.height);
        this.gc = canvas.getGraphicsContext2D();
        gc.setImageSmoothing(true);
        if (interactionHandler != null) {
            interactionHandler.updateCanvas(canvas, chartConfig);
        }
    }

    private void syncSeabornPaletteFromTheme() {
        seabornStyleMapper.setHuePalette(themeManager.getColorPalette());
    }

    private PlotStyle styleFromStringParam(String styleString) {
        if (!useSeriesStyleSystem || styleString == null) {
            return null;
        }
        String t = styleString.trim();
        if (t.isEmpty()) {
            return null;
        }
        return StyleExpression.parse(t);
    }

    private PlotStyle baseSeriesStyle() {
        return useSeriesStyleSystem ? new PlotStyle(defaultSeriesStyle) : PlotStyle.defaultStyle();
    }

    private PlotStyle effectiveCartesianStyle(PlotStyle explicit, String styleString, boolean scatterMarkerDefaults) {
        PlotStyle fromStr = styleFromStringParam(styleString);
        PlotStyle chosen = explicit != null ? new PlotStyle(explicit) : fromStr != null ? fromStr : baseSeriesStyle();
        if (scatterMarkerDefaults && explicit == null && fromStr == null) {
            chosen = new PlotStyle(chosen).marker("o").markerSize(6);
        }
        return chosen;
    }

    private PlotStyle effectiveGeneralStyle(PlotStyle explicit, String styleString) {
        PlotStyle fromStr = styleFromStringParam(styleString);
        if (explicit != null) {
            return new PlotStyle(explicit);
        }
        if (fromStr != null) {
            return fromStr;
        }
        return baseSeriesStyle();
    }

    private static PlotStyle mergeStyles(PlotStyle base, PlotStyle overlay) {
        PlotStyle merged = new PlotStyle(base);
        if (overlay.getColor() != null) {
            merged.setColor(overlay.getColor());
        }
        if (overlay.getLineStyle() != null) {
            merged.setLineStyle(overlay.getLineStyle());
        }
        if (overlay.getMarker() != null) {
            merged.setMarker(overlay.getMarker());
        }
        if (overlay.getMarkerSize() > 0) {
            merged.setMarkerSize(overlay.getMarkerSize());
        }
        if (overlay.getLineWidth() > 0) {
            merged.setLineWidth(overlay.getLineWidth());
        }
        if (overlay.getLabel() != null && !overlay.getLabel().isEmpty()) {
            merged.setLabel(overlay.getLabel());
        }
        return merged;
    }

    private static List<String> autoCategoryLabels(int n) {
        List<String> ts = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            ts.add("类别" + (i + 1));
        }
        return ts;
    }

    private void ensureRendererKind(String rendererKey) {
        if (jointplotMode) {
            throw new PlotException("jointplot 布局已完成，不能切换或追加图表类型");
        }
        if (facetMode) {
            if (!"cartesian".equals(rendererKey)) {
                throw new PlotException("子图模式下仅支持笛卡尔图层，不能切换为: " + rendererKey);
            }
            if (activeRendererKey != null && !activeRendererKey.equals(rendererKey)) {
                facetPanes.get(facetCursor).data.clear();
            }
            activeRendererKey = rendererKey;
            return;
        }
        if (activeRendererKey != null && !activeRendererKey.equals(rendererKey)) {
            seriesList.clear();
        }
        activeRendererKey = rendererKey;
    }

    private void addSeries(JavaFxChartRenderer.SeriesData series) {
        if (jointplotMode) {
            throw new PlotException("jointplot 布局下不能再添加序列");
        }
        if (facetMode) {
            if (facetPanes == null || facetCursor < 0 || facetCursor >= facetPanes.size()) {
                throw new PlotException("子图状态无效，请先调用 subplots / subplot");
            }
            facetPanes.get(facetCursor).data.add(series);
            return;
        }
        appendToMainSeriesList(series);
    }

    private void appendToMainSeriesList(JavaFxChartRenderer.SeriesData series) {
        seriesList.add(series);
    }

    @SuppressWarnings("unchecked")
    private static IVector<Double> matrixColumn(IMatrix<?> m, int col) {
        int r = m.getRowNum();
        double[] buf = new double[r];
        for (int i = 0; i < r; i++) {
            buf[i] = m.get(i, col).doubleValue();
        }
        return Linalg.vector(buf);
    }

    private IVector<Double>[] kdeColumn(IVector<Double> col, int gridPoints, double bandwidthIn) {
        double bw = bandwidthIn <= 0 ? PlotKde.scottBandwidth(col) : bandwidthIn;
        return PlotKde.toVectors(PlotKde.evaluate(col, bw, gridPoints));
    }

    private static void ensureFxToolkit() {
        try {
            Platform.startup(() -> { });
        } catch (IllegalStateException ignored) {
            // JavaFX 已初始化
        }
    }

    private static void runOnFxThreadSync(Runnable action) {
        ensureFxToolkit();
        if (Platform.isFxApplicationThread()) {
            action.run();
            return;
        }
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Throwable> err = new AtomicReference<>();
        Platform.runLater(() -> {
            try {
                action.run();
            } catch (Throwable t) {
                err.set(t);
            } finally {
                done.countDown();
            }
        });
        try {
            done.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new PlotException("JavaFX 操作被中断", e);
        }
        if (err.get() != null) {
            if (err.get() instanceof RuntimeException) {
                throw (RuntimeException) err.get();
            }
            throw new PlotException("JavaFX 渲染失败: " + err.get().getMessage(), err.get());
        }
    }

    private static String jsonEscape(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t");
    }

    private void drawHoverRing(GraphicsContext g, JavaFxInteractionHandler.DataPoint p) {
        g.save();
        g.setStroke(Color.YELLOW);
        g.setLineWidth(3);
        g.strokeOval(p.x - 8, p.y - 8, 16, 16);
        g.restore();
    }
    
    // ========== IPlot 接口实现 ==========
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot line(IVector x, IVector y) {
        ensureRendererKind("cartesian");
        currentChartType = ChartType.LINE;
        currentRenderer = RENDERERS.get("cartesian");
        
        PlotStyle style = effectiveCartesianStyle(null, null, false);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Line", (IVector<Double>) x, (IVector<Double>) y, style, "line");
        addSeries(series);
        
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot line(IVector y) {
        IVector<Double> yd = (IVector<Double>) y;
        IVector<Double> x = Linalg.range(0, yd.length());
        return line((IVector<Double>) x, (IVector<Double>) y);
    }
    
    @Override
    public IPlot line(IVector x, IVector y, List<String> hue) {
        return createGroupedSeries(x, y, hue, ChartType.LINE, "line");
    }

    @Override
    public IPlot line(IVector x, IVector y, String styleString) {
        PlotStyle st = effectiveCartesianStyle(null, styleString, false);
        return lineWithResolvedStyle(x, y, st, null, null);
    }

    @Override
    public IPlot line(IVector x, IVector y, PlotStyle style) {
        PlotStyle st = style != null ? new PlotStyle(style) : baseSeriesStyle();
        return lineWithResolvedStyle(x, y, st, null, null);
    }

    @Override
    public IPlot line(IVector y, String styleString) {
        IVector<Double> yd = (IVector<Double>) y;
        IVector<Double> x = Linalg.range(0, yd.length());
        return line(x, y, styleString);
    }

    @Override
    public IPlot line(IVector y, PlotStyle style) {
        IVector<Double> yd = (IVector<Double>) y;
        IVector<Double> x = Linalg.range(0, yd.length());
        return line(x, y, style);
    }

    @Override
    public IPlot line(IVector x, IVector y, List<String> hue, List<String> styleGroup) {
        return createGroupedLineWithStyleGroup(x, y, hue, styleGroup, null);
    }

    @SuppressWarnings("unchecked")
    private IPlot lineWithResolvedStyle(IVector x, IVector y, PlotStyle resolvedStyle,
            List<String> hue, List<String> styleGroup) {
        if (hue != null || styleGroup != null) {
            return createGroupedLineWithStyleGroup(x, y, hue, styleGroup, resolvedStyle);
        }
        ensureRendererKind("cartesian");
        currentChartType = ChartType.LINE;
        currentRenderer = RENDERERS.get("cartesian");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Line", (IVector<Double>) x, (IVector<Double>) y, resolvedStyle, "line");
        addSeries(series);
        return this;
    }

    @SuppressWarnings("unchecked")
    private IPlot createGroupedLineWithStyleGroup(IVector x, IVector y, List<String> hue,
            List<String> styleGroup, PlotStyle baseOverlay) {
        ensureRendererKind("cartesian");
        currentChartType = ChartType.LINE;
        currentRenderer = RENDERERS.get("cartesian");
        if (x.length() != y.length()
                || (hue != null && x.length() != hue.size())
                || (styleGroup != null && x.length() != styleGroup.size())) {
            throw new PlotException("数据和分组标签长度不一致");
        }
        SeabornStyleMapper.GroupStyleMapping mapping = seabornStyleMapper.createMapping(hue, styleGroup, null, null);
        Map<String, SeabornStyleMapper.GroupedData> groups = seabornStyleMapper.groupData(x, y, hue, mapping);
        for (SeabornStyleMapper.GroupedData group : groups.values()) {
            PlotStyle groupStyle = new PlotStyle(group.getStyle());
            if (baseOverlay != null) {
                groupStyle = mergeStyles(groupStyle, baseOverlay);
            }
            Object[] pts = group.getData();
            double[] gx = new double[pts.length];
            double[] gy = new double[pts.length];
            for (int i = 0; i < pts.length; i++) {
                Number[] pair = (Number[]) pts[i];
                gx[i] = pair[0].doubleValue();
                gy[i] = pair[1].doubleValue();
            }
            JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
                    group.getGroupName(),
                    Linalg.vector(gx),
                    Linalg.vector(gy),
                    groupStyle,
                    "line");
            addSeries(series);
        }
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot scatter(IVector x, IVector y) {
        ensureRendererKind("cartesian");
        currentChartType = ChartType.SCATTER;
        currentRenderer = RENDERERS.get("cartesian");
        
        PlotStyle style = effectiveCartesianStyle(null, null, true);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Scatter", (IVector<Double>) x, (IVector<Double>) y, style, "scatter");
        addSeries(series);
        
        return this;
    }
    
    @Override
    public IPlot scatter(IVector x, IVector y, List<String> hue) {
        return createGroupedSeries(x, y, hue, ChartType.SCATTER, "scatter");
    }

    @Override
    public IPlot scatter(IVector x, IVector y, String styleString) {
        PlotStyle st = effectiveCartesianStyle(null, styleString, true);
        ensureRendererKind("cartesian");
        currentChartType = ChartType.SCATTER;
        currentRenderer = RENDERERS.get("cartesian");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Scatter", (IVector<Double>) x, (IVector<Double>) y, st, "scatter");
        addSeries(series);
        return this;
    }

    @Override
    public IPlot scatter(IVector x, IVector y, PlotStyle style) {
        PlotStyle st = style != null ? new PlotStyle(style) : effectiveCartesianStyle(null, null, true);
        ensureRendererKind("cartesian");
        currentChartType = ChartType.SCATTER;
        currentRenderer = RENDERERS.get("cartesian");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Scatter", (IVector<Double>) x, (IVector<Double>) y, st, "scatter");
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot pie(IVector x) {
        ensureRendererKind("pie");
        currentChartType = ChartType.PIE;
        currentRenderer = RENDERERS.get("pie");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Pie", null, (IVector<Double>) x, style, "pie");
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot pie(IVector x, String styleString) {
        return pieWithLabels(x, null, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot pie(IVector x, PlotStyle style) {
        return pieWithLabels(x, null, effectiveGeneralStyle(style, null));
    }

    @Override
    public IPlot pie(IVector x, List<String> labels, PlotStyle style) {
        return pieWithLabels(x, labels, effectiveGeneralStyle(style, null));
    }

    @Override
    public IPlot pie(IVector x, List<String> labels, String styleString) {
        return pieWithLabels(x, labels, effectiveGeneralStyle(null, styleString));
    }

    @SuppressWarnings("unchecked")
    private IPlot pieWithLabels(IVector x, List<String> labels, PlotStyle style) {
        ensureRendererKind("pie");
        currentChartType = ChartType.PIE;
        currentRenderer = RENDERERS.get("pie");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Pie", null, (IVector<Double>) x, style, "pie");
        if (labels != null) {
            series.labels = new ArrayList<>(labels);
        }
        addSeries(series);
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot bar(IVector x) {
        ensureRendererKind("bar");
        currentChartType = ChartType.BAR;
        currentRenderer = RENDERERS.get("bar");
        
        List<String> labels = new ArrayList<>();
        for (int i = 0; i < x.length(); i++) {
            labels.add("Item " + (i + 1));
        }
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Bar", null, (IVector<Double>) x, style, "bar");
        series.labels = labels;
        addSeries(series);
        
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot bar(List<String> xticks, IVector y) {
        ensureRendererKind("bar");
        currentChartType = ChartType.BAR;
        currentRenderer = RENDERERS.get("bar");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Bar", null, (IVector<Double>) y, style, "bar");
        series.labels = xticks != null ? xticks : new ArrayList<>();
        addSeries(series);
        
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot bar(List<String> xticks, IVector y, List<String> hue) {
        return barGroupedWithLabels(xticks, y, hue, null, null);
    }

    @Override
    public IPlot bar(IVector y, String styleString) {
        return barVectorSingleton(y, null, styleString);
    }

    @Override
    public IPlot bar(IVector y, PlotStyle style) {
        return barVectorSingleton(y, style, null);
    }

    @Override
    public IPlot bar(IVector y, List<String> hue, PlotStyle style) {
        return barGroupedWithLabels(autoCategoryLabels(y.length()), y, hue, style, null);
    }

    @Override
    public IPlot bar(IVector y, List<String> hue, String styleString) {
        return barGroupedWithLabels(autoCategoryLabels(y.length()), y, hue, null, styleString);
    }

    @SuppressWarnings("unchecked")
    private IPlot barVectorSingleton(IVector y, PlotStyle explicit, String styleString) {
        ensureRendererKind("bar");
        currentChartType = ChartType.BAR;
        currentRenderer = RENDERERS.get("bar");
        List<String> labels = new ArrayList<>();
        for (int i = 0; i < y.length(); i++) {
            labels.add("Item " + (i + 1));
        }
        PlotStyle style = effectiveGeneralStyle(explicit, styleString);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Bar", null, (IVector<Double>) y, style, "bar");
        series.labels = labels;
        addSeries(series);
        return this;
    }

    @SuppressWarnings("unchecked")
    private IPlot barGroupedWithLabels(List<String> xticks, IVector y, List<String> hue,
            PlotStyle styleOverlay, String styleString) {
        ensureRendererKind("bar");
        currentChartType = ChartType.BAR;
        currentRenderer = RENDERERS.get("bar");
        PlotStyle overlay = styleOverlay != null ? new PlotStyle(styleOverlay) : styleFromStringParam(styleString);

        if (hue != null && !hue.isEmpty() && y.length() == hue.size()) {
            java.util.LinkedHashSet<String> uniqueGroups = new java.util.LinkedHashSet<>(hue);
            java.util.List<String> groupList = new java.util.ArrayList<>(uniqueGroups);
            java.util.LinkedHashSet<String> uniqueLabels = new java.util.LinkedHashSet<>(xticks);
            java.util.List<String> xLabelList = new java.util.ArrayList<>(uniqueLabels);
            String[] palette = themeManager.getColorPalette();
            for (int g = 0; g < groupList.size(); g++) {
                String groupName = groupList.get(g);
                PlotStyle style = PlotStyle.defaultStyle();
                style.setColor(palette[g % palette.length]);
                if (overlay != null) {
                    style = mergeStyles(style, overlay);
                }
                double[] groupValues = new double[xLabelList.size()];
                for (int x = 0; x < xLabelList.size(); x++) {
                    String xLabel = xLabelList.get(x);
                    for (int i = 0; i < xticks.size(); i++) {
                        if (xticks.get(i).equals(xLabel) && hue.get(i).equals(groupName)) {
                            groupValues[x] = ((IVector<Double>) y).get(i);
                            break;
                        }
                    }
                }
                IVector<Double> groupVector = new com.yishape.lab.math.linalg.RereDoubleVector(groupValues);
                JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
                    groupName, null, groupVector, style, "bar");
                series.labels = xLabelList;
                series.extraData.put("groupIndex", g);
                series.extraData.put("isGrouped", true);
                addSeries(series);
            }
        } else {
            PlotStyle style = overlay != null ? mergeStyles(effectiveGeneralStyle(null, null), overlay)
                    : effectiveGeneralStyle(null, null);
            JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
                "Bar", null, (IVector<Double>) y, style, "bar");
            series.labels = xticks != null ? xticks : new ArrayList<>();
            addSeries(series);
        }
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot hist(IVector x, boolean fittingLine) {
        ensureRendererKind("histogram");
        currentChartType = ChartType.HIST;
        currentRenderer = RENDERERS.get("histogram");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Histogram", null, (IVector<Double>) x, style, "histogram");
        series.extraData.put("fittingLine", fittingLine);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot hist(IVector x, boolean fittingLine, String styleString) {
        return histFull(x, fittingLine, null, styleString, null);
    }

    @Override
    public IPlot hist(IVector x, boolean fittingLine, PlotStyle style) {
        return histFull(x, fittingLine, style, null, null);
    }

    @Override
    public IPlot hist(IVector x, boolean fittingLine, PlotStyle style, Integer bins) {
        return histFull(x, fittingLine, style, null, bins);
    }

    @SuppressWarnings("unchecked")
    private IPlot histFull(IVector x, boolean fittingLine, PlotStyle explicit,
            String styleString, Integer bins) {
        ensureRendererKind("histogram");
        currentChartType = ChartType.HIST;
        currentRenderer = RENDERERS.get("histogram");
        PlotStyle style;
        if (explicit != null) {
            style = new PlotStyle(explicit);
        } else {
            style = effectiveGeneralStyle(null, styleString);
        }
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Histogram", null, (IVector<Double>) x, style, "histogram");
        series.extraData.put("fittingLine", fittingLine);
        if (bins != null && bins > 0) {
            series.extraData.put("bins", bins);
        }
        addSeries(series);
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot polarBar(IVector data, List<String> categories) {
        ensureRendererKind("polar_bar");
        currentChartType = ChartType.POLAR_BAR;
        currentRenderer = RENDERERS.get("polar_bar");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Polar Bar", null, (IVector<Double>) data, style, "polar_bar");
        series.labels = categories != null ? categories : new ArrayList<>();
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot polarBar(IVector data, List<String> categories, String styleString) {
        return polarWithStyle(data, categories, effectiveGeneralStyle(null, styleString), "polar_bar", ChartType.POLAR_BAR);
    }

    @Override
    public IPlot polarBar(IVector data, List<String> categories, PlotStyle style) {
        return polarWithStyle(data, categories, effectiveGeneralStyle(style, null), "polar_bar", ChartType.POLAR_BAR);
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot polarLine(IVector data, List<String> categories) {
        ensureRendererKind("polar_line");
        currentChartType = ChartType.POLAR_LINE;
        currentRenderer = RENDERERS.get("polar_line");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Polar Line", null, (IVector<Double>) data, style, "polar_line");
        series.labels = categories != null ? categories : new ArrayList<>();
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot polarLine(IVector data, List<String> categories, String styleString) {
        return polarWithStyle(data, categories, effectiveGeneralStyle(null, styleString), "polar_line", ChartType.POLAR_LINE);
    }

    @Override
    public IPlot polarLine(IVector data, List<String> categories, PlotStyle style) {
        return polarWithStyle(data, categories, effectiveGeneralStyle(style, null), "polar_line", ChartType.POLAR_LINE);
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot polarScatter(IVector data, List<String> categories) {
        ensureRendererKind("polar_scatter");
        currentChartType = ChartType.POLAR_SCATTER;
        currentRenderer = RENDERERS.get("polar_scatter");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Polar Scatter", null, (IVector<Double>) data, style, "polar_scatter");
        series.labels = categories != null ? categories : new ArrayList<>();
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot polarScatter(IVector data, List<String> categories, String styleString) {
        return polarWithStyle(data, categories, effectiveGeneralStyle(null, styleString), "polar_scatter", ChartType.POLAR_SCATTER);
    }

    @Override
    public IPlot polarScatter(IVector data, List<String> categories, PlotStyle style) {
        return polarWithStyle(data, categories, effectiveGeneralStyle(style, null), "polar_scatter", ChartType.POLAR_SCATTER);
    }

    @SuppressWarnings("unchecked")
    private IPlot polarWithStyle(IVector data, List<String> categories, PlotStyle style,
            String rendererKey, ChartType chartType) {
        ensureRendererKind(rendererKey);
        currentChartType = chartType;
        currentRenderer = RENDERERS.get(rendererKey);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Polar", null, (IVector<Double>) data, style, rendererKey);
        series.labels = categories != null ? categories : new ArrayList<>();
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot boxplot(IVector data) {
        return boxplot(data, null);
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot boxplot(IVector data, List<String> labels) {
        ensureRendererKind("boxplot");
        currentChartType = ChartType.BOXPLOT;
        currentRenderer = RENDERERS.get("boxplot");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Boxplot", null, (IVector<Double>) data, style, "boxplot");
        series.labels = labels != null ? labels : new ArrayList<>();
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot boxplot(IVector data, List<String> labels, String styleString) {
        return boxplotStyled(data, labels, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot boxplot(IVector data, List<String> labels, PlotStyle style) {
        return boxplotStyled(data, labels, effectiveGeneralStyle(style, null));
    }

    @SuppressWarnings("unchecked")
    private IPlot boxplotStyled(IVector data, List<String> labels, PlotStyle style) {
        ensureRendererKind("boxplot");
        currentChartType = ChartType.BOXPLOT;
        currentRenderer = RENDERERS.get("boxplot");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Boxplot", null, (IVector<Double>) data, style, "boxplot");
        series.labels = labels != null ? labels : new ArrayList<>();
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot violinplot(IVector data) {
        return violinplotInternal(data, null, null, null);
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot violinplot(IVector data, List<String> labels) {
        return violinplotInternal(data, labels, null, null);
    }

    @Override
    public IPlot violinplot(IVector data, String styleString) {
        return violinplotInternal(data, null, effectiveGeneralStyle(null, styleString), null);
    }

    @Override
    public IPlot violinplot(IVector data, PlotStyle style) {
        return violinplotInternal(data, null, effectiveGeneralStyle(style, null), null);
    }

    @Override
    public IPlot violinplot(IVector data, List<String> labels, String styleString) {
        return violinplotInternal(data, labels, null, styleString);
    }

    @Override
    public IPlot violinplot(IVector data, List<String> labels, PlotStyle style) {
        return violinplotInternal(data, labels, effectiveGeneralStyle(style, null), null);
    }

    @SuppressWarnings("unchecked")
    private IPlot violinplotInternal(IVector data, List<String> labels,
            PlotStyle explicitOverlay, String styleString) {
        PlotStyle overlay = explicitOverlay != null ? explicitOverlay : styleFromStringParam(styleString);
        ensureRendererKind("violin");
        currentChartType = ChartType.VIOLINPLOT;
        currentRenderer = RENDERERS.get("violin");

        if (labels != null && !labels.isEmpty() && data.length() == labels.size()) {
            java.util.LinkedHashSet<String> uniqueGroups = new java.util.LinkedHashSet<>(labels);
            java.util.List<String> groupList = new java.util.ArrayList<>(uniqueGroups);
            String[] palette = themeManager.getColorPalette();
            int nGroups = groupList.size();
            for (int g = 0; g < nGroups; g++) {
                String groupName = groupList.get(g);
                java.util.List<Double> groupValues = new java.util.ArrayList<>();
                for (int i = 0; i < labels.size(); i++) {
                    if (labels.get(i).equals(groupName)) {
                        groupValues.add(((IVector<Double>) data).get(i));
                    }
                }
                double[] valuesArray = new double[groupValues.size()];
                for (int i = 0; i < groupValues.size(); i++) {
                    valuesArray[i] = groupValues.get(i);
                }
                IVector<Double> groupVector = new com.yishape.lab.math.linalg.RereDoubleVector(valuesArray);
                PlotStyle style = PlotStyle.defaultStyle();
                // 在调色板内均匀取色，避免相邻索引（如青/蓝绿）在半透明填充下过近、肉眼不可分
                int colorIdx = nGroups <= 1
                        ? (g % palette.length)
                        : (g * palette.length / nGroups) % palette.length;
                style.setColor(palette[colorIdx]);
                if (overlay != null) {
                    style = mergeStyles(style, overlay);
                }
                JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
                    groupName, null, groupVector, style, "violin");
                series.extraData.put("groupIndex", g);
                addSeries(series);
            }
        } else {
            PlotStyle style = overlay != null ? new PlotStyle(overlay) : effectiveGeneralStyle(null, null);
            JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
                "Violin", null, (IVector<Double>) data, style, "violin");
            series.labels = labels != null ? labels : new ArrayList<>();
            addSeries(series);
        }
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot candlestick(IMatrix data, List<String> dates) {
        ensureRendererKind("candlestick");
        currentChartType = ChartType.CANDLESTICK;
        currentRenderer = RENDERERS.get("candlestick");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Candlestick", null, null, style, "candlestick");
        series.extraData.put("matrixData", (IMatrix<Double>) data);
        series.labels = dates != null ? dates : new ArrayList<>();
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot candlestick(IMatrix data, List<String> dates, String styleString) {
        return candlestickStyled(data, dates, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot candlestick(IMatrix data, List<String> dates, PlotStyle style) {
        return candlestickStyled(data, dates, effectiveGeneralStyle(style, null));
    }

    @SuppressWarnings("unchecked")
    private IPlot candlestickStyled(IMatrix data, List<String> dates, PlotStyle style) {
        ensureRendererKind("candlestick");
        currentChartType = ChartType.CANDLESTICK;
        currentRenderer = RENDERERS.get("candlestick");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Candlestick", null, null, style, "candlestick");
        series.extraData.put("matrixData", (IMatrix<Double>) data);
        series.labels = dates != null ? dates : new ArrayList<>();
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot funnel(IVector data, List<String> labels) {
        ensureRendererKind("funnel");
        currentChartType = ChartType.FUNNEL;
        currentRenderer = RENDERERS.get("funnel");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Funnel", null, (IVector<Double>) data, style, "funnel");
        series.labels = labels != null ? labels : new ArrayList<>();
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot funnel(IVector data, List<String> labels, String styleString) {
        return funnelStyled(data, labels, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot funnel(IVector data, List<String> labels, PlotStyle style) {
        return funnelStyled(data, labels, effectiveGeneralStyle(style, null));
    }

    @SuppressWarnings("unchecked")
    private IPlot funnelStyled(IVector data, List<String> labels, PlotStyle style) {
        ensureRendererKind("funnel");
        currentChartType = ChartType.FUNNEL;
        currentRenderer = RENDERERS.get("funnel");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Funnel", null, (IVector<Double>) data, style, "funnel");
        series.labels = labels != null ? labels : new ArrayList<>();
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        ensureRendererKind("sankey");
        currentChartType = ChartType.SANKEY;
        currentRenderer = RENDERERS.get("sankey");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Sankey", null, null, style, "sankey");
        series.extraData.put("nodes", nodes);
        series.extraData.put("links", links);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString) {
        return sankeyStyled(nodes, links, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot sankey(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        return sankeyStyled(nodes, links, effectiveGeneralStyle(style, null));
    }

    private IPlot sankeyStyled(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        ensureRendererKind("sankey");
        currentChartType = ChartType.SANKEY;
        currentRenderer = RENDERERS.get("sankey");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Sankey", null, null, style, "sankey");
        series.extraData.put("nodes", nodes);
        series.extraData.put("links", links);
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot sunburst(List<Map<String, Object>> data) {
        ensureRendererKind("sunburst");
        currentChartType = ChartType.SUNBURST;
        currentRenderer = RENDERERS.get("sunburst");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Sunburst", null, null, style, "sunburst");
        series.extraData.put("hierarchicalData", data);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot sunburst(List<Map<String, Object>> data, String styleString) {
        return sunburstStyled(data, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot sunburst(List<Map<String, Object>> data, PlotStyle style) {
        return sunburstStyled(data, effectiveGeneralStyle(style, null));
    }

    private IPlot sunburstStyled(List<Map<String, Object>> data, PlotStyle style) {
        ensureRendererKind("sunburst");
        currentChartType = ChartType.SUNBURST;
        currentRenderer = RENDERERS.get("sunburst");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Sunburst", null, null, style, "sunburst");
        series.extraData.put("hierarchicalData", data);
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot themeRiver(List<Map<String, Object>> data, List<String> categories) {
        ensureRendererKind("theme_river");
        currentChartType = ChartType.THEME_RIVER;
        currentRenderer = RENDERERS.get("theme_river");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "ThemeRiver", null, null, style, "themeRiver");
        series.extraData.put("riverData", data);
        series.extraData.put("categories", categories);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot themeRiver(List<Map<String, Object>> data, List<String> categories, String styleString) {
        return themeRiverStyled(data, categories, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot themeRiver(List<Map<String, Object>> data, List<String> categories, PlotStyle style) {
        return themeRiverStyled(data, categories, effectiveGeneralStyle(style, null));
    }

    private IPlot themeRiverStyled(List<Map<String, Object>> data, List<String> categories, PlotStyle style) {
        ensureRendererKind("theme_river");
        currentChartType = ChartType.THEME_RIVER;
        currentRenderer = RENDERERS.get("theme_river");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "ThemeRiver", null, null, style, "themeRiver");
        series.extraData.put("riverData", data);
        series.extraData.put("categories", categories);
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot tree(List<Map<String, Object>> data) {
        ensureRendererKind("tree");
        currentChartType = ChartType.TREE;
        currentRenderer = RENDERERS.get("tree");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Tree", null, null, style, "tree");
        series.extraData.put("treeData", data);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot tree(List<Map<String, Object>> data, String styleString) {
        return treeStyled(data, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot tree(List<Map<String, Object>> data, PlotStyle style) {
        return treeStyled(data, effectiveGeneralStyle(style, null));
    }

    private IPlot treeStyled(List<Map<String, Object>> data, PlotStyle style) {
        ensureRendererKind("tree");
        currentChartType = ChartType.TREE;
        currentRenderer = RENDERERS.get("tree");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Tree", null, null, style, "tree");
        series.extraData.put("treeData", data);
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot treemap(List<Map<String, Object>> data) {
        ensureRendererKind("treemap");
        currentChartType = ChartType.TREEMAP;
        currentRenderer = RENDERERS.get("treemap");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Treemap", null, null, style, "treemap");
        series.extraData.put("treemapData", data);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot treemap(List<Map<String, Object>> data, String styleString) {
        return treemapStyled(data, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot treemap(List<Map<String, Object>> data, PlotStyle style) {
        return treemapStyled(data, effectiveGeneralStyle(style, null));
    }

    private IPlot treemapStyled(List<Map<String, Object>> data, PlotStyle style) {
        ensureRendererKind("treemap");
        currentChartType = ChartType.TREEMAP;
        currentRenderer = RENDERERS.get("treemap");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Treemap", null, null, style, "treemap");
        series.extraData.put("treemapData", data);
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        ensureRendererKind("graph");
        currentChartType = ChartType.GRAPH;
        currentRenderer = RENDERERS.get("graph");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Graph", null, null, style, "graph");
        series.extraData.put("graphNodes", nodes);
        series.extraData.put("graphLinks", links);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString) {
        return graphStyled(nodes, links, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot graph(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        return graphStyled(nodes, links, effectiveGeneralStyle(style, null));
    }

    private IPlot graphStyled(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        ensureRendererKind("graph");
        currentChartType = ChartType.GRAPH;
        currentRenderer = RENDERERS.get("graph");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Graph", null, null, style, "graph");
        series.extraData.put("graphNodes", nodes);
        series.extraData.put("graphLinks", links);
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot parallel(IMatrix data, List<String> dimensions) {
        ensureRendererKind("parallel");
        currentChartType = ChartType.PARALLEL;
        currentRenderer = RENDERERS.get("parallel");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Parallel", null, null, style, "parallel");
        series.extraData.put("matrixData", (IMatrix<Double>) data);
        series.extraData.put("dimensions", dimensions);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot parallel(IMatrix data, List<String> dimensions, String styleString) {
        return parallelStyled(data, dimensions, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot parallel(IMatrix data, List<String> dimensions, PlotStyle style) {
        return parallelStyled(data, dimensions, effectiveGeneralStyle(style, null));
    }

    @SuppressWarnings("unchecked")
    private IPlot parallelStyled(IMatrix data, List<String> dimensions, PlotStyle style) {
        ensureRendererKind("parallel");
        currentChartType = ChartType.PARALLEL;
        currentRenderer = RENDERERS.get("parallel");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Parallel", null, null, style, "parallel");
        series.extraData.put("matrixData", (IMatrix<Double>) data);
        series.extraData.put("dimensions", dimensions);
        addSeries(series);
        return this;
    }
    
    @Override
    public IPlot heatmap(IMatrix data) {
        int rows = data.getRowNum();
        int cols = data.getColNum();
        List<String> xLabs = new ArrayList<>();
        List<String> yLabs = new ArrayList<>();
        for (int j = 0; j < cols; j++) {
            xLabs.add(String.valueOf(j));
        }
        for (int i = 0; i < rows; i++) {
            yLabs.add(String.valueOf(i));
        }
        return heatmap(data, xLabs, yLabs);
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels) {
        ensureRendererKind("heatmap");
        currentChartType = ChartType.HEATMAP;
        currentRenderer = RENDERERS.get("heatmap");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Heatmap", null, null, style, "heatmap");
        series.extraData.put("matrixData", (IMatrix<Double>) data);
        series.extraData.put("xLabels", xLabels);
        series.extraData.put("yLabels", yLabels);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels, String styleString) {
        return heatmapStyled(data, xLabels, yLabels, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot heatmap(IMatrix data, List<String> xLabels, List<String> yLabels, PlotStyle style) {
        return heatmapStyled(data, xLabels, yLabels, effectiveGeneralStyle(style, null));
    }

    @SuppressWarnings("unchecked")
    private IPlot heatmapStyled(IMatrix data, List<String> xLabels, List<String> yLabels, PlotStyle style) {
        ensureRendererKind("heatmap");
        currentChartType = ChartType.HEATMAP;
        currentRenderer = RENDERERS.get("heatmap");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Heatmap", null, null, style, "heatmap");
        series.extraData.put("matrixData", (IMatrix<Double>) data);
        series.extraData.put("xLabels", xLabels);
        series.extraData.put("yLabels", yLabels);
        addSeries(series);
        return this;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public IPlot radar(IVector data, List<String> indicators) {
        ensureRendererKind("radar");
        currentChartType = ChartType.RADAR;
        currentRenderer = RENDERERS.get("radar");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Radar", null, (IVector<Double>) data, style, "radar");
        series.labels = indicators != null ? indicators : new ArrayList<>();
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot radar(IVector data, List<String> indicators, String styleString) {
        return radarStyled(data, indicators, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot radar(IVector data, List<String> indicators, PlotStyle style) {
        return radarStyled(data, indicators, effectiveGeneralStyle(style, null));
    }

    @SuppressWarnings("unchecked")
    private IPlot radarStyled(IVector data, List<String> indicators, PlotStyle style) {
        ensureRendererKind("radar");
        currentChartType = ChartType.RADAR;
        currentRenderer = RENDERERS.get("radar");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Radar", null, (IVector<Double>) data, style, "radar");
        series.labels = indicators != null ? indicators : new ArrayList<>();
        addSeries(series);
        return this;
    }
    
    @Override
    public IPlot gauge(double value, double max, double min) {
        ensureRendererKind("gauge");
        currentChartType = ChartType.GAUGE;
        currentRenderer = RENDERERS.get("gauge");
        
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Gauge", null, null, style, "gauge");
        series.extraData.put("value", value);
        series.extraData.put("max", max);
        series.extraData.put("min", min);
        addSeries(series);
        
        return this;
    }

    @Override
    public IPlot gauge(double value, double max, double min, String styleString) {
        return gaugeStyled(value, max, min, effectiveGeneralStyle(null, styleString));
    }

    @Override
    public IPlot gauge(double value, double max, double min, PlotStyle style) {
        return gaugeStyled(value, max, min, effectiveGeneralStyle(style, null));
    }

    private IPlot gaugeStyled(double value, double max, double min, PlotStyle style) {
        ensureRendererKind("gauge");
        currentChartType = ChartType.GAUGE;
        currentRenderer = RENDERERS.get("gauge");
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Gauge", null, null, style, "gauge");
        series.extraData.put("value", value);
        series.extraData.put("max", max);
        series.extraData.put("min", min);
        addSeries(series);
        return this;
    }

    @Override
    public IPlot setDefaultStyle(PlotStyle style) {
        defaultSeriesStyle = style != null ? new PlotStyle(style) : PlotStyle.defaultStyle();
        return this;
    }

    @Override
    public IPlot setPalette(String paletteName) {
        if (paletteName != null && ColorPalette.hasPalette(paletteName)) {
            seabornStyleMapper.setHuePalette(paletteName);
        }
        return this;
    }

    @Override
    public IPlot enableStyleSystem(boolean enabled) {
        useSeriesStyleSystem = enabled;
        return this;
    }

    @Override
    public IPlot enableThemeSystem(boolean enabled) {
        usePlotThemeSystem = enabled;
        return this;
    }

    @Override
    public IPlot applyTheme(String themeName) {
        if (usePlotThemeSystem && themeName != null) {
            theme(themeName);
        }
        return this;
    }

    @Override
    public IPlot registerTheme(String themeName, EchartsThemeManager.CustomTheme theme) {
        if (themeName != null && theme != null) {
            EchartsThemeManager.registerCustomTheme(themeName, theme);
        }
        return this;
    }

    @Override
    public IPlot createGradientTheme(String themeName, String startColor, String endColor, String backgroundColor) {
        if (themeName == null) {
            return this;
        }
        EchartsThemeManager.CustomTheme ec = EchartsThemeManager.createGradientTheme(
                themeName, startColor, endColor, backgroundColor);
        EchartsThemeManager.registerCustomTheme(themeName, ec);
        JavaFxThemeManager.Theme jfx = JavaFxThemeManager.createGradientTheme(
                themeName, startColor, endColor, backgroundColor);
        JavaFxThemeManager.registerCustomTheme(themeName, jfx);
        return this;
    }

    @Override
    public IPlot createMonochromeTheme(String themeName, String baseColor, String backgroundColor) {
        if (themeName == null) {
            return this;
        }
        EchartsThemeManager.CustomTheme ec = EchartsThemeManager.createMonochromeTheme(
                themeName, baseColor, backgroundColor);
        EchartsThemeManager.registerCustomTheme(themeName, ec);
        String[] colors = new String[10];
        for (int i = 0; i < colors.length; i++) {
            colors[i] = baseColor;
        }
        Map<String, Object> cfg = new HashMap<>();
        cfg.put("backgroundColor", backgroundColor);
        cfg.put("textColor", ColorPalette.getContrastColor(backgroundColor));
        cfg.put("color", colors);
        JavaFxThemeManager.registerCustomTheme(themeName, new JavaFxThemeManager.Theme(themeName, cfg));
        return this;
    }

    @Override
    public IPlot xscale(PlotAxisScale scale) {
        chartConfig.xAxisScale = scale != null ? scale : PlotAxisScale.LINEAR;
        return this;
    }

    @Override
    public IPlot yscale(PlotAxisScale scale) {
        chartConfig.yAxisScale = scale != null ? scale : PlotAxisScale.LINEAR;
        return this;
    }

    @Override
    public IPlot y2label(String label) {
        chartConfig.y2AxisLabel = label != null ? label : "";
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot area(IVector x, IVector y) {
        ensureRendererKind("cartesian");
        currentChartType = ChartType.LINE;
        currentRenderer = RENDERERS.get("cartesian");
        PlotStyle style = effectiveCartesianStyle(null, null, false);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Area", (IVector<Double>) x, (IVector<Double>) y, style, "area");
        addSeries(series);
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot step(IVector x, IVector y) {
        ensureRendererKind("cartesian");
        currentChartType = ChartType.LINE;
        currentRenderer = RENDERERS.get("cartesian");
        PlotStyle style = effectiveCartesianStyle(null, null, false);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Step", (IVector<Double>) x, (IVector<Double>) y, style, "step");
        addSeries(series);
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot barh(List<String> categories, IVector values) {
        ensureRendererKind("bar");
        currentChartType = ChartType.BAR;
        currentRenderer = RENDERERS.get("bar");
        PlotStyle style = effectiveGeneralStyle(null, null);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "BarH", null, (IVector<Double>) values, style, "bar");
        series.labels = categories != null ? categories : new ArrayList<>();
        series.extraData.put("horizontal", true);
        addSeries(series);
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot barStacked(List<String> categories, IMatrix values, List<String> layerNames) {
        ensureRendererKind("bar");
        currentChartType = ChartType.BAR;
        currentRenderer = RENDERERS.get("bar");
        if (values == null || categories == null) {
            throw new PlotException("categories 与 values 不能为 null");
        }
        int rows = values.getRowNum();
        int cols = values.getColNum();
        if (cols != categories.size()) {
            throw new PlotException("类别数量必须与矩阵列数一致");
        }
        String[] palette = themeManager.getColorPalette();
        for (int r = 0; r < rows; r++) {
            double[] colVals = new double[cols];
            for (int c = 0; c < cols; c++) {
                colVals[c] = values.get(r, c).doubleValue();
            }
            String name = layerNames != null && r < layerNames.size()
                ? layerNames.get(r) : ("L" + (r + 1));
            PlotStyle style = PlotStyle.defaultStyle();
            style.setColor(palette[r % palette.length]);
            JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
                name, null, Linalg.vector(colVals), style, "bar");
            series.labels = new ArrayList<>(categories);
            series.extraData.put("stacked", true);
            addSeries(series);
        }
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot errorbar(IVector x, IVector y, IVector yerr) {
        ensureRendererKind("cartesian");
        currentChartType = ChartType.SCATTER;
        currentRenderer = RENDERERS.get("cartesian");
        PlotStyle style = PlotStyle.defaultStyle();
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Error", (IVector<Double>) x, (IVector<Double>) y, style, "errorbar");
        series.extraData.put("yerr", (IVector<Double>) yerr);
        addSeries(series);
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot scatter(IVector x, IVector y, IVector sizes) {
        ensureRendererKind("cartesian");
        currentChartType = ChartType.SCATTER;
        currentRenderer = RENDERERS.get("cartesian");
        PlotStyle style = PlotStyle.defaultStyle().marker("o").markerSize(6);
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "Bubble", (IVector<Double>) x, (IVector<Double>) y, style, "scatter");
        series.extraData.put("sizes", (IVector<Double>) sizes);
        addSeries(series);
        return this;
    }

    @Override
    public IPlot regplot(IVector x, IVector y) {
        return regplot(x, y, false);
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot regplot(IVector x, IVector y, boolean confidenceBand) {
        if (x.length() != y.length()) {
            throw new PlotException("regplot: x 与 y 长度须相同");
        }
        ensureRendererKind("cartesian");
        currentChartType = ChartType.SCATTER;
        currentRenderer = RENDERERS.get("cartesian");
        IVector<Double> xd = (IVector<Double>) x;
        IVector<Double> yd = (IVector<Double>) y;
        if (confidenceBand && xd.length() >= 3) {
            PlotStats.OlsMeanBand band = PlotStats.olsMeanResponseBand95(xd, yd);
            PlotStyle bandStyle = PlotStyle.defaultStyle();
            String[] pal = themeManager.getColorPalette();
            bandStyle.setColor(pal.length > 1 ? pal[1] : pal[0]);
            JavaFxChartRenderer.SeriesData ci = new JavaFxChartRenderer.SeriesData(
                "95% CI (mean)", band.x, band.yHat, bandStyle, "ci_band");
            ci.extraData.put("yLow", band.yLow);
            ci.extraData.put("yHigh", band.yHigh);
            addSeries(ci);
        }
        PlotStyle ptStyle = PlotStyle.defaultStyle().marker("o").markerSize(6);
        JavaFxChartRenderer.SeriesData pts = new JavaFxChartRenderer.SeriesData(
            "Data", xd, yd, ptStyle, "scatter");
        addSeries(pts);
        double[] coef = PlotStats.ols(xd, yd);
        double xmin = Double.POSITIVE_INFINITY;
        double xmax = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < xd.length(); i++) {
            double xv = xd.get(i);
            xmin = Math.min(xmin, xv);
            xmax = Math.max(xmax, xv);
        }
        if (xmax - xmin < 1e-15) {
            double pad = Math.abs(xmin) > 1e-6 ? Math.abs(xmin) * 0.02 : 0.02;
            xmin -= pad;
            xmax += pad;
        }
        double yLo = coef[0] * xmin + coef[1];
        double yHi = coef[0] * xmax + coef[1];
        IVector<Double> xFit = Linalg.vector(new double[] {xmin, xmax});
        IVector<Double> yFit = Linalg.vector(new double[] {yLo, yHi});
        PlotStyle lineStyle = PlotStyle.defaultStyle();
        JavaFxChartRenderer.SeriesData fit = new JavaFxChartRenderer.SeriesData(
            "OLS fit", xFit, yFit, lineStyle, "line");
        addSeries(fit);
        return this;
    }

    @Override
    public IPlot subplots(int rows, int cols) {
        if (rows < 1 || cols < 1) {
            throw new PlotException("subplots: 行列数须至少为 1");
        }
        jointplotMode = false;
        jointTop = null;
        jointMain = null;
        jointRight = null;
        facetMode = true;
        facetRows = rows;
        facetCols = cols;
        facetPanes = new ArrayList<>();
        for (int i = 0; i < rows * cols; i++) {
            facetPanes.add(new FacetPane());
        }
        facetCursor = 0;
        seriesList.clear();
        activeRendererKey = null;
        currentRenderer = RENDERERS.get("cartesian");
        currentChartType = ChartType.LINE;
        return this;
    }

    @Override
    public IPlot subplot(int row, int col) {
        if (!facetMode || facetPanes == null) {
            throw new PlotException("请先调用 subplots(rows, cols)");
        }
        if (row < 0 || col < 0 || row >= facetRows || col >= facetCols) {
            throw new PlotException("subplot(row,col) 越界");
        }
        facetCursor = row * facetCols + col;
        activeRendererKey = null;
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot kdeplot(IVector data, int gridPoints, double bandwidth) {
        ensureRendererKind("cartesian");
        currentChartType = ChartType.LINE;
        currentRenderer = RENDERERS.get("cartesian");
        IVector<Double> d = (IVector<Double>) data;
        IVector<Double>[] xy = kdeColumn(d, gridPoints, bandwidth);
        PlotStyle style = PlotStyle.defaultStyle();
        JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
            "KDE", xy[0], xy[1], style, "line");
        addSeries(series);
        return this;
    }

    @Override
    public IPlot pairplot(IMatrix data, List<String> columnNames, PairplotDiagonal diagonal) {
        jointplotMode = false;
        jointTop = null;
        jointMain = null;
        jointRight = null;
        if (data == null || data.getColNum() < 1 || data.getRowNum() < 1) {
            throw new PlotException("pairplot: 数据矩阵不能为空");
        }
        int p = data.getColNum();
        List<String> names = new ArrayList<>();
        for (int j = 0; j < p; j++) {
            names.add(columnNames != null && j < columnNames.size()
                ? columnNames.get(j) : ("x" + j));
        }
        facetRows = p;
        facetCols = p;
        facetMode = true;
        facetPanes = new ArrayList<>();
        String[] palette = themeManager.getColorPalette();
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                FacetPane pane = new FacetPane();
                if (i == j) {
                    IVector<Double> col = matrixColumn(data, i);
                    if (diagonal == PairplotDiagonal.NONE) {
                        pane.rendererKey = "cartesian";
                    } else if (diagonal == PairplotDiagonal.KDE) {
                        pane.rendererKey = "cartesian";
                        IVector<Double>[] xy = kdeColumn(col, 128, 0.0);
                        PlotStyle st = PlotStyle.defaultStyle();
                        st.setColor(palette[i % palette.length]);
                        JavaFxChartRenderer.SeriesData s = new JavaFxChartRenderer.SeriesData(
                            names.get(i), xy[0], xy[1], st, "line");
                        pane.data.add(s);
                    } else {
                        pane.rendererKey = "histogram";
                        PlotStyle st = PlotStyle.defaultStyle();
                        JavaFxChartRenderer.SeriesData s = new JavaFxChartRenderer.SeriesData(
                            names.get(i), null, col, st, "histogram");
                        s.extraData.put("fittingLine", false);
                        pane.data.add(s);
                    }
                } else {
                    pane.rendererKey = "cartesian";
                    IVector<Double> xcol = matrixColumn(data, j);
                    IVector<Double> ycol = matrixColumn(data, i);
                    PlotStyle st = PlotStyle.defaultStyle().marker("o").markerSize(4);
                    st.setColor(palette[0]);
                    String nm = names.get(j) + " vs " + names.get(i);
                    JavaFxChartRenderer.SeriesData s = new JavaFxChartRenderer.SeriesData(
                        nm, xcol, ycol, st, "scatter");
                    pane.data.add(s);
                }
                facetPanes.add(pane);
            }
        }
        seriesList.clear();
        activeRendererKey = null;
        currentRenderer = null;
        chartConfig.showLegend = false;
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot jointplot(IVector x, IVector y, JointplotMarginal marginal) {
        if (x.length() != y.length()) {
            throw new PlotException("jointplot: x 与 y 长度须相同");
        }
        facetMode = false;
        facetPanes = null;
        jointplotMode = true;
        seriesList.clear();
        activeRendererKey = null;
        IVector<Double> xd = (IVector<Double>) x;
        IVector<Double> yd = (IVector<Double>) y;
        jointMain = new FacetPane();
        jointMain.rendererKey = "cartesian";
        PlotStyle sc = PlotStyle.defaultStyle().marker("o").markerSize(5);
        jointMain.data.add(new JavaFxChartRenderer.SeriesData("joint", xd, yd, sc, "scatter"));

        jointTop = new FacetPane();
        jointRight = new FacetPane();
        double[] lockX = JavaFxChartUtils.calculateRange(jointMain.data, true);
        double[] lockY = JavaFxChartUtils.calculateRange(jointMain.data, false);
        if (marginal == JointplotMarginal.KDE) {
            jointTop.rendererKey = "cartesian";
            IVector<Double>[] kx = kdeColumn(xd, 144, 0.0);
            jointTop.data.add(new JavaFxChartRenderer.SeriesData("p(x)", kx[0], kx[1],
                PlotStyle.defaultStyle(), "line"));
            jointRight.rendererKey = "cartesian";
            IVector<Double>[] ky = kdeColumn(yd, 144, 0.0);
            JavaFxChartRenderer.SeriesData py = new JavaFxChartRenderer.SeriesData("p(y)", ky[1], ky[0],
                PlotStyle.defaultStyle(), "line");
            py.sortLineByY = true;
            jointRight.data.add(py);
        } else {
            jointTop.rendererKey = "histogram";
            JavaFxChartRenderer.SeriesData hx = new JavaFxChartRenderer.SeriesData("x", null, xd,
                PlotStyle.defaultStyle(), "histogram");
            hx.extraData.put("fittingLine", false);
            jointTop.data.add(hx);

            jointRight.rendererKey = "bar";
            int bins = Math.min(24, Math.max(6, (int) Math.sqrt(yd.length())));
            double ymin = lockY[0];
            double ymax = lockY[1];
            double bw = ymax - ymin;
            if (bw <= 0) {
                bw = 1.0;
            }
            double binW = bw / bins;
            int[] cnt = new int[bins];
            for (int i = 0; i < yd.length(); i++) {
                double v = yd.get(i);
                int bi = (int) Math.floor((v - ymin) / binW);
                if (bi < 0) {
                    bi = 0;
                }
                if (bi >= bins) {
                    bi = bins - 1;
                }
                cnt[bi]++;
            }
            List<String> labs = new ArrayList<>();
            double[] vals = new double[bins];
            for (int b = 0; b < bins; b++) {
                labs.add(String.format(Locale.ROOT, "%.4g", ymin + (b + 0.5) * binW));
                vals[b] = cnt[b];
            }
            JavaFxChartRenderer.SeriesData br = new JavaFxChartRenderer.SeriesData("y hist", null,
                Linalg.vector(vals), PlotStyle.defaultStyle(), "bar");
            br.labels = labs;
            br.extraData.put("horizontal", true);
            // 与上方直方图一致：单色柱（避免横向 bar 按 bin 循环调色板显得花哨）
            br.extraData.put("uniformBarColor", true);
            jointRight.data.add(br);
        }
        currentRenderer = null;
        chartConfig.showLegend = false;
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot qqplot(IVector data) {
        ensureRendererKind("cartesian");
        currentChartType = ChartType.SCATTER;
        currentRenderer = RENDERERS.get("cartesian");
        IVector<Double> d = (IVector<Double>) data;
        int n = d.length();
        if (n < 2) {
            throw new PlotException("qqplot 至少需要 2 个样本点");
        }
        IVector<Double> sorted = d.sort();
        double[] theo = new double[n];
        for (int i = 0; i < n; i++) {
            double p = (i + 0.5) / n;
            theo[i] = PlotStats.normalPpf(p);
        }
        IVector<Double> xq = Linalg.vector(theo);
        PlotStyle ptStyle = PlotStyle.defaultStyle().marker("o").markerSize(5);
        JavaFxChartRenderer.SeriesData pts = new JavaFxChartRenderer.SeriesData(
            "Sample", xq, sorted, ptStyle, "scatter");
        addSeries(pts);
        double tMin = Math.min(theo[0], theo[n - 1]);
        double tMax = Math.max(theo[0], theo[n - 1]);
        double lo = Math.min(tMin, sorted.get(0));
        double hi = Math.max(tMax, sorted.get(n - 1));
        double[] diagX = new double[] { lo, hi };
        double[] diagY = new double[] { lo, hi };
        PlotStyle refStyle = PlotStyle.defaultStyle();
        refStyle.setColor("#94a3b8");
        refStyle.setLineStyle("dotted");
        JavaFxChartRenderer.SeriesData refLine = new JavaFxChartRenderer.SeriesData(
            "y = x", Linalg.vector(diagX), Linalg.vector(diagY), refStyle, "line");
        addSeries(refLine);
        if (chartConfig.xlabel.isEmpty()) {
            chartConfig.xlabel = "理论分位数";
        }
        if (chartConfig.ylabel.isEmpty()) {
            chartConfig.ylabel = "有序样本";
        }
        return this;
    }

    @SuppressWarnings("unchecked")
    @Override
    public IPlot lineWithSecondaryY(IVector x, IVector yLeft, IVector yRight) {
        if (x.length() != yLeft.length() || x.length() != yRight.length()) {
            throw new PlotException("lineWithSecondaryY: 向量长度须一致");
        }
        ensureRendererKind("cartesian");
        currentChartType = ChartType.LINE;
        currentRenderer = RENDERERS.get("cartesian");
        chartConfig.paddingRight = Math.max(chartConfig.paddingRight, 78);
        if (chartConfig.y2AxisLabel == null || chartConfig.y2AxisLabel.isEmpty()) {
            chartConfig.y2AxisLabel = "右轴";
        }
        IVector<Double> xv = (IVector<Double>) x;
        PlotStyle s0 = PlotStyle.defaultStyle();
        JavaFxChartRenderer.SeriesData primary = new JavaFxChartRenderer.SeriesData(
            "Y1", xv, (IVector<Double>) yLeft, s0, "line");
        primary.yAxisIndex = 0;
        addSeries(primary);
        PlotStyle s1 = PlotStyle.defaultStyle();
        String[] palette = themeManager.getColorPalette();
        if (palette.length > 1) {
            s1.setColor(palette[1]);
        }
        JavaFxChartRenderer.SeriesData secondary = new JavaFxChartRenderer.SeriesData(
            "Y2", xv, (IVector<Double>) yRight, s1, "line");
        secondary.yAxisIndex = 1;
        addSeries(secondary);
        return this;
    }
    
    // ========== 配置方法 ==========
    
    @Override
    public IPlot title(String titleText) {
        chartConfig.title = titleText;
        return this;
    }
    
    @Override
    public IPlot title(String titleText, String subtitleText) {
        chartConfig.title = titleText;
        chartConfig.subtitle = subtitleText;
        return this;
    }
    
    @Override
    public IPlot xlabel(String name) {
        chartConfig.xlabel = name;
        return this;
    }
    
    @Override
    public IPlot ylabel(String name) {
        chartConfig.ylabel = name;
        return this;
    }
    
    @Override
    public IPlot size(int width, int height) {
        chartConfig.width = width;
        chartConfig.height = height;
        initializeCanvas();
        return this;
    }
    
    @Override
    public IPlot theme(String theme) {
        themeManager.setTheme(theme);
        chartConfig.theme = theme;
        syncSeabornPaletteFromTheme();
        syncDefaultSeriesStyleFromTheme();
        return this;
    }
    
    @Override
    public IPlot show() {
        ensureFxToolkit();
        Platform.runLater(() -> {
            render();
            if (stage == null) {
                stage = new Stage();
            }
            stage.setTitle(chartConfig.title.isEmpty() ? "JavaFX Chart" : chartConfig.title);

            rootPane = new BorderPane();

            if (!chartConfig.title.isEmpty()) {
                VBox titleBox = createTitleBox();
                rootPane.setTop(titleBox);
            }

            rootPane.setCenter(canvas);

            scene = new Scene(rootPane, chartConfig.width,
                chartConfig.height + (chartConfig.title.isEmpty() ? 0 : 80));

            setupInteractionHandler();

            stage.setScene(scene);
            stage.show();
        });
        return this;
    }

    @Override
    public IPlot saveAsHtml(String filename) {
        runOnFxThreadSync(() -> {
            try {
                render();
                WritableImage image = new WritableImage(chartConfig.width, chartConfig.height);
                canvas.snapshot(new SnapshotParameters(), image);
                java.awt.image.BufferedImage buffered = SwingFXUtils.fromFXImage(image, null);
                String pngFilename = filename.replace(".html", ".png").replace(".HTML", ".png");
                ImageIO.write(buffered, "png", new File(pngFilename));
                log.info("图表已保存为PNG: {}", pngFilename);

                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                ImageIO.write(buffered, "png", baos);
                String base64 = Base64.getEncoder().encodeToString(baos.toByteArray());
                String html = "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head><meta charset=\"UTF-8\"><title>"
                    + jsonEscape(chartConfig.title.isEmpty() ? "Chart" : chartConfig.title)
                    + "</title></head>\n<body>\n<img alt=\"chart\" src=\"data:image/png;base64," + base64 + "\"/>\n</body>\n</html>\n";
                Files.writeString(Path.of(filename), html, StandardCharsets.UTF_8);
                log.info("图表已保存为HTML(内嵌PNG): {}", filename);
            } catch (IOException e) {
                throw new PlotException("保存图表失败: " + e.getMessage(), e);
            }
        });
        return this;
    }
    
    @Override
    public String toHtml() {
        return "<html><body><p>JavaFX图表不支持HTML输出，请使用show()方法显示图表或使用saveAsHtml()保存为PNG图像。</p></body></html>";
    }
    
    @Override
    public String toJson() {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"chartType\": \"").append(currentChartType.name()).append("\",\n");
        json.append("  \"title\": \"").append(jsonEscape(chartConfig.title)).append("\",\n");
        json.append("  \"width\": ").append(chartConfig.width).append(",\n");
        json.append("  \"height\": ").append(chartConfig.height).append(",\n");
        json.append("  \"seriesCount\": ").append(seriesList.size()).append("\n");
        json.append("}");
        return json.toString();
    }
    
    @Override
    public void setTitle(String titleText) {
        chartConfig.title = titleText;
    }
    
    @Override
    public void setTitle(String titleText, String subtitleText) {
        chartConfig.title = titleText;
        chartConfig.subtitle = subtitleText;
    }
    
    @Override
    public void setXlabel(String name) {
        chartConfig.xlabel = name;
    }
    
    @Override
    public void setYlabel(String name) {
        chartConfig.ylabel = name;
    }
    
    @Override
    public void setXticks(AxisTicks xticks) {
        chartConfig.xAxisTicks = xticks;
    }
    
    @Override
    public void setYticks(AxisTicks yticks) {
        chartConfig.yAxisTicks = yticks;
    }
    
    @Override
    public int getWidth() {
        return chartConfig.width;
    }
    
    @Override
    public int getHeight() {
        return chartConfig.height;
    }
    
    @Override
    public String getTheme() {
        return themeManager.getCurrentTheme();
    }
    
    // ========== 私有辅助方法 ==========
    
    private VBox createTitleBox() {
        VBox titleBox = new VBox(5);
        titleBox.setAlignment(Pos.CENTER);
        titleBox.setPadding(new Insets(15, 0, 10, 0));
        
        Label titleLabel = new Label(chartConfig.title);
        titleLabel.setFont(Font.font("Arial", FontWeight.BOLD, 20));
        titleLabel.setStyle("-fx-text-fill: #333;");
        titleBox.getChildren().add(titleLabel);
        
        if (!chartConfig.subtitle.isEmpty()) {
            Label subtitleLabel = new Label(chartConfig.subtitle);
            subtitleLabel.setFont(Font.font("Arial", 12));
            subtitleLabel.setStyle("-fx-text-fill: #666;");
            titleBox.getChildren().add(subtitleLabel);
        }
        
        return titleBox;
    }
    
    private IPlot createGroupedSeries(IVector x, IVector y, List<String> hue,
                                     ChartType type, String seriesType) {
        if ("line".equals(seriesType)) {
            return createGroupedLineWithStyleGroup(x, y, hue, null, null);
        }
        ensureRendererKind("cartesian");
        currentChartType = type;
        currentRenderer = RENDERERS.get("cartesian");
        if (x.length() != y.length() || hue == null || x.length() != hue.size()) {
            throw new PlotException("分组散点：x、y、hue 长度须一致且 hue 非 null");
        }
        SeabornStyleMapper.GroupStyleMapping mapping = seabornStyleMapper.createMapping(hue, null, null, null);
        Map<String, SeabornStyleMapper.GroupedData> groups = seabornStyleMapper.groupData(x, y, hue, mapping);
        for (SeabornStyleMapper.GroupedData group : groups.values()) {
            PlotStyle groupStyle = new PlotStyle(group.getStyle());
            if (groupStyle.getMarker() == null || groupStyle.getMarker().isEmpty()) {
                groupStyle = new PlotStyle(groupStyle).marker("o").markerSize(6);
            }
            Object[] pts = group.getData();
            double[] gx = new double[pts.length];
            double[] gy = new double[pts.length];
            for (int i = 0; i < pts.length; i++) {
                Number[] pair = (Number[]) pts[i];
                gx[i] = pair[0].doubleValue();
                gy[i] = pair[1].doubleValue();
            }
            JavaFxChartRenderer.SeriesData series = new JavaFxChartRenderer.SeriesData(
                    group.getGroupName(),
                    Linalg.vector(gx),
                    Linalg.vector(gy),
                    groupStyle,
                    "scatter");
            addSeries(series);
        }
        return this;
    }
    
    /**
     * 设置交互处理器
     */
    private void setupInteractionHandler() {
        if (interactionHandler == null) {
            interactionHandler = new JavaFxInteractionHandler(canvas, chartConfig);
            interactionHandler.setRepaintCallback(this::render);
        } else {
            interactionHandler.updateCanvas(canvas, chartConfig);
            interactionHandler.updateConfig(chartConfig);
            interactionHandler.setRepaintCallback(this::render);
        }
        if (currentRenderer != null && currentRenderer.supportsAnimation()) {
            interactionHandler.startAnimation(() -> { });
        }
    }    
    /**
     * 获取当前图表类型（用于测试）
     * @return 当前图表类型
     */
    public ChartType getCurrentChartType() {
        return currentChartType;
    }
    
    /**
     * 获取Canvas（用于测试和图片生成）
     * @return Canvas对象
     */
    public Canvas getCanvas() {
        return canvas;
    }
    
    private void renderMiniPane(GraphicsContext g, double ox, double oy, double w, double h, FacetPane pane) {
        renderMiniPane(g, ox, oy, w, h, pane, null, null, 0, 0);
    }

    private void renderMiniPane(GraphicsContext g, double ox, double oy, double w, double h, FacetPane pane,
                                double[] axisLockX, double[] axisLockY, int maxAxisTicks, int extraPaddingRight) {
        if (pane == null) {
            return;
        }
        int cw = Math.max(10, (int) Math.round(w));
        int ch = Math.max(10, (int) Math.round(h));
        JavaFxChartRenderer.ChartConfig cellCfg = new JavaFxChartRenderer.ChartConfig(cw, ch);
        cellCfg.theme = chartConfig.theme;
        cellCfg.showLegend = false;
        cellCfg.title = "";
        cellCfg.subtitle = "";
        cellCfg.xlabel = "";
        cellCfg.ylabel = "";
        cellCfg.paddingLeft = Math.min(44, cw / 4);
        cellCfg.paddingRight = Math.min(14, cw / 8) + extraPaddingRight;
        cellCfg.paddingTop = Math.min(28, ch / 5);
        cellCfg.paddingBottom = Math.min(32, ch / 4);
        cellCfg.axisLockX = axisLockX;
        cellCfg.axisLockY = axisLockY;
        if (maxAxisTicks > 0) {
            cellCfg.maxAxisTicks = maxAxisTicks;
        }
        cellCfg.xAxisScale = chartConfig.xAxisScale;
        cellCfg.yAxisScale = chartConfig.yAxisScale;
        cellCfg.showGrid = chartConfig.showGrid;
        cellCfg.y2AxisLabel = chartConfig.y2AxisLabel;
        g.save();
        g.translate(ox, oy);
        g.beginPath();
        g.rect(0, 0, w, h);
        g.clip();
        JavaFxChartRenderer r = RENDERERS.get(pane.rendererKey);
        r.render(g, pane.data, cellCfg);
        g.restore();
    }

    private void renderFacetGrid(GraphicsContext g) {
        g.setFill(themeManager.getBackgroundColor());
        g.fillRect(0, 0, chartConfig.width, chartConfig.height);
        int titleOffset = 0;
        if (!chartConfig.title.isEmpty() || !chartConfig.subtitle.isEmpty()) {
            JavaFxChartUtils.drawTitle(g, chartConfig, themeManager);
            titleOffset = 52;
        }
        int gh = chartConfig.height - chartConfig.paddingBottom - chartConfig.paddingTop - titleOffset;
        int gw = chartConfig.width - chartConfig.paddingLeft - chartConfig.paddingRight;
        if (gh < 20 || gw < 20 || facetPanes == null) {
            return;
        }
        double cellW = (double) gw / facetCols;
        double cellH = (double) gh / facetRows;
        double baseLeft = chartConfig.paddingLeft;
        double baseTop = chartConfig.paddingTop + titleOffset;
        for (int idx = 0; idx < facetPanes.size(); idx++) {
            int fr = idx / facetCols;
            int fc = idx % facetCols;
            double ox = baseLeft + fc * cellW;
            double oy = baseTop + fr * cellH;
            renderMiniPane(g, ox, oy, cellW, cellH, facetPanes.get(idx));
        }
    }

    private void renderJointplotComposite(GraphicsContext g) {
        g.setFill(themeManager.getBackgroundColor());
        g.fillRect(0, 0, chartConfig.width, chartConfig.height);
        int titleOffset = 0;
        if (!chartConfig.title.isEmpty() || !chartConfig.subtitle.isEmpty()) {
            JavaFxChartUtils.drawTitle(g, chartConfig, themeManager);
            titleOffset = 52;
        }
        double W = chartConfig.width - chartConfig.paddingLeft - chartConfig.paddingRight;
        double H = chartConfig.height - chartConfig.paddingTop - chartConfig.paddingBottom - titleOffset;
        double left = chartConfig.paddingLeft;
        double top = chartConfig.paddingTop + titleOffset;
        double gap = 6;
        double fracTop = 0.22;
        double fracRight = 0.22;
        double mainW = W * (1 - fracRight) - gap;
        double mainH = H * (1 - fracTop) - gap;
        double topW = mainW;
        double topH = H * fracTop;
        double rightW = W * fracRight;
        double[] lockX = JavaFxChartUtils.calculateRange(jointMain.data, true);
        double[] lockY = JavaFxChartUtils.calculateRange(jointMain.data, false);
        int miniTicks = 3;
        boolean rightIsBar = jointRight != null && "bar".equals(jointRight.rendererKey);
        int rightExtraPad = rightIsBar ? 40 : 14;
        renderMiniPane(g, left, top, topW, topH, jointTop, lockX, null, miniTicks, 0);
        renderMiniPane(g, left, top + topH + gap, mainW, mainH, jointMain, lockX, lockY, 0, 0);
        renderMiniPane(g, left + mainW + gap, top + topH + gap, rightW, mainH, jointRight,
            null, lockY, miniTicks, rightExtraPad);
    }

    /**
     * 执行渲染（用于测试和图片生成）
     */
    public void render() {
        chartConfig.backgroundColor = themeManager.getBackgroundColor();

        if (interactionHandler != null) {
            interactionHandler.clearDataPoints();
            chartConfig.hitTestHandler = interactionHandler;
        }

        gc.save();
        try {
            if (interactionHandler != null) {
                interactionHandler.applyTransform(gc);
            }
            if (jointplotMode) {
                renderJointplotComposite(gc);
            } else if (facetMode && facetPanes != null) {
                renderFacetGrid(gc);
            } else {
                if (currentRenderer == null) {
                    log.warn("没有可用的渲染器");
                    return;
                }
                currentRenderer.render(gc, seriesList, chartConfig);
            }
            if (interactionHandler != null) {
                interactionHandler.syncHoverAfterRebuild();
                JavaFxInteractionHandler.DataPoint hp = interactionHandler.getHoveredPoint();
                if (hp != null) {
                    drawHoverRing(gc, hp);
                }
            }
        } finally {
            gc.restore();
            chartConfig.hitTestHandler = null;
        }
    }

    private byte[] captureCanvasPngBytesOnFxThread() throws IOException {
        render();
        WritableImage image = new WritableImage(chartConfig.width, chartConfig.height);
        canvas.snapshot(new SnapshotParameters(), image);
        java.awt.image.BufferedImage buffered = SwingFXUtils.fromFXImage(image, null);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(buffered, "png", baos);
        return baos.toByteArray();
    }

    /**
     * 将光栅图封装为可独立打开的 SVG（内嵌 PNG）。
     */
    private static String wrapRasterPngAsSvgDocument(int w, int h, byte[] pngBytes) {
        String b64 = Base64.getEncoder().encodeToString(pngBytes);
        return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            + "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" "
            + "width=\"" + w + "\" height=\"" + h + "\" viewBox=\"0 0 " + w + " " + h + "\">\n"
            + "<image width=\"" + w + "\" height=\"" + h + "\" xlink:href=\"data:image/png;base64," + b64 + "\"/>\n"
            + "</svg>\n";
    }

    @Override
    public IPlot saveAsSvg(String filename) {
        if (filename == null) {
            return this;
        }
        // 扩展名为 .png 时写入标准 PNG 字节流（避免误把「SVG 包 PNG」当 PNG 保存导致看图软件无法打开）
        String t = filename.trim();
        if (t.length() >= 4 && t.regionMatches(true, t.length() - 4, ".png", 0, 4)) {
            return saveAsPng(filename);
        }
        runOnFxThreadSync(() -> {
            try {
                byte[] png = captureCanvasPngBytesOnFxThread();
                String svg = wrapRasterPngAsSvgDocument(chartConfig.width, chartConfig.height, png);
                Files.writeString(Path.of(filename), svg, StandardCharsets.UTF_8);
                log.info("图表已保存为 SVG（内嵌 PNG）: {}", filename);
            } catch (IOException e) {
                throw new PlotException("保存 SVG 失败: " + e.getMessage(), e);
            }
        });
        return this;
    }

    @Override
    public IPlot saveAsPng(String filename) {
        if (filename == null) {
            return this;
        }
        runOnFxThreadSync(() -> {
            try {
                byte[] png = captureCanvasPngBytesOnFxThread();
                Files.write(Path.of(filename), png);
                log.info("图表已保存为 PNG: {}", filename);
            } catch (IOException e) {
                throw new PlotException("保存 PNG 失败: " + e.getMessage(), e);
            }
        });
        return this;
    }

    @Override
    public String toBase64Svg() {
        AtomicReference<String> ref = new AtomicReference<>();
        runOnFxThreadSync(() -> {
            try {
                byte[] png = captureCanvasPngBytesOnFxThread();
                String svg = wrapRasterPngAsSvgDocument(chartConfig.width, chartConfig.height, png);
                ref.set(Base64.getEncoder().encodeToString(svg.getBytes(StandardCharsets.UTF_8)));
            } catch (IOException e) {
                throw new PlotException("生成 Base64 SVG 失败: " + e.getMessage(), e);
            }
        });
        return ref.get() != null ? ref.get() : "";
    }

    @Override
    public String toBase64Png() {
        AtomicReference<String> ref = new AtomicReference<>();
        runOnFxThreadSync(() -> {
            try {
                ref.set(Base64.getEncoder().encodeToString(captureCanvasPngBytesOnFxThread()));
            } catch (IOException e) {
                throw new PlotException("生成 Base64 PNG 失败: " + e.getMessage(), e);
            }
        });
        return ref.get() != null ? ref.get() : "";
    }
}
