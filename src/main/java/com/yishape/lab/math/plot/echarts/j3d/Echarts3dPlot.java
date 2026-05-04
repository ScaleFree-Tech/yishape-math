package com.yishape.lab.math.plot.echarts.j3d;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.AxisTicks;
import com.yishape.lab.math.plot.I3dPlot;
import com.yishape.lab.math.plot.PlotException;
import com.yishape.lab.math.plot.PlotStyle;
import com.yishape.lab.math.plot.echarts.EchartsThemeManager;

import java.awt.Desktop;
import java.io.File;
import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * 基于 ECharts GL 的 3D 图表实现。
 * <p>职责划分（与 JavaFX 3D 分包思路一致）：</p>
 * <ul>
 *   <li>{@link Echarts3dSeriesConfig} — 系列快照</li>
 *   <li>{@link Echarts3dSceneDefaults} — 默认 viewControl / 光照 / 后处理</li>
 *   <li>{@link Echarts3dPlotVectors} — 向量/矩阵与 IDW 等小计算</li>
 *   <li>{@link Echarts3dPalette} — 调色板展开</li>
 *   <li>{@link Echarts3dOptionJson} — option 字面量</li>
 *   <li>{@link Echarts3dHtmlTemplate} — HTML 壳与启动脚本</li>
 * </ul>
 *
 * @author lteb2
 */
public class Echarts3dPlot implements I3dPlot, Serializable {

    @Serial
    private static final long serialVersionUID = 2L;

    // 配置存储（可序列化）
    private final List<Echarts3dSeriesConfig> seriesList = new ArrayList<>();
    private final Map<String, Object> sceneConfig = new HashMap<>();
    private final Map<String, String> labels = new HashMap<>();
    private final Map<String, double[]> bounds = new HashMap<>();

    // 画布尺寸
    private int width = 800;
    private int height = 600;

    // 主题
    private String themeName = EchartsThemeManager.THEME_DEFAULT;
    private String paletteName = "echarts";
    private boolean useStyleSystem = true;
    private boolean useThemeSystem = true;

    // 默认样式
    private PlotStyle defaultStyle;

    // 坐标轴配置
    private AxisTicks xticks;
    private AxisTicks yticks;
    private AxisTicks zticks;

    public Echarts3dPlot() {
        this.defaultStyle = PlotStyle.defaultStyle();
        initSceneConfig();
    }

    public Echarts3dPlot(int width, int height) {
        this();
        this.width = width;
        this.height = height;
    }

    public Echarts3dPlot(int width, int height, String theme) {
        this(width, height);
        this.themeName = theme != null ? theme : EchartsThemeManager.THEME_DEFAULT;
    }

    private void initSceneConfig() {
        Echarts3dSceneDefaults.initialize(sceneConfig);
    }

    // ==================== 3D图表方法实现 ====================

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z) {
        return scatter3d(x, y, z, null, (PlotStyle) null);
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, List<String> hue) {
        return scatter3d(x, y, z, hue, (PlotStyle) null);
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, String styleString) {
        return scatter3d(x, y, z, null, parseStyle(styleString));
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, PlotStyle style) {
        return scatter3d(x, y, z, null, style);
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, List<String> hue, String styleString) {
        return scatter3d(x, y, z, hue, parseStyle(styleString));
    }

    @Override
    public I3dPlot scatter3d(IVector x, IVector y, IVector z, List<String> hue, PlotStyle style) {
        int n = Math.min(x.length(), Math.min(y.length(), z.length()));

        if (hue != null && !hue.isEmpty()) {
            // 分组散点
            Map<String, List<double[]>> groups = new LinkedHashMap<>();
            for (int i = 0; i < n; i++) {
                String h = i < hue.size() ? hue.get(i) : "Default";
                groups.computeIfAbsent(h, k -> new ArrayList<>())
                        .add(new double[]{Echarts3dPlotVectors.toDouble(x.get(i)), Echarts3dPlotVectors.toDouble(y.get(i)), Echarts3dPlotVectors.toDouble(z.get(i))});
            }

            String[] colors = Echarts3dPalette.colors(paletteName,groups.size());
            int idx = 0;
            for (Map.Entry<String, List<double[]>> entry : groups.entrySet()) {
                Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("scatter3D", entry.getKey());
                series.data = entry.getValue().toArray(new double[0][]);
                series.color = colors[idx++ % colors.length];
                series.symbolSize = style != null ? style.getMarkerSize() : 8;
                seriesList.add(series);
            }
        } else {
            // 单组散点
            Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("scatter3D", "Data");
            series.data = Echarts3dPlotVectors.extractXYZ(x, y, z, n);
            series.color = style != null ? style.getColor() : Echarts3dPalette.colors(paletteName,1)[0];
            series.symbolSize = style != null ? style.getMarkerSize() : 8;
            seriesList.add(series);
        }

        updateBounds(x, y, z);
        return this;
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes) {
        return scatterBubble3d(x, y, z, sizes, null, (PlotStyle) null);
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, List<String> hue) {
        return scatterBubble3d(x, y, z, sizes, hue, (PlotStyle) null);
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, String styleString) {
        return scatterBubble3d(x, y, z, sizes, null, parseStyle(styleString));
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, PlotStyle style) {
        return scatterBubble3d(x, y, z, sizes, null, style);
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, List<String> hue, String styleString) {
        return scatterBubble3d(x, y, z, sizes, hue, parseStyle(styleString));
    }

    @Override
    public I3dPlot scatterBubble3d(IVector x, IVector y, IVector z, IVector sizes, List<String> hue, PlotStyle style) {
        int n = Math.min(x.length(), Math.min(y.length(), Math.min(z.length(), sizes.length())));

        // 计算大小范围
        double minSize = Double.POSITIVE_INFINITY, maxSize = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < sizes.length(); i++) {
            double s = Echarts3dPlotVectors.toDouble(sizes.get(i));
            minSize = Math.min(minSize, s);
            maxSize = Math.max(maxSize, s);
        }
        if (!Double.isFinite(minSize) || maxSize <= minSize) {
            minSize = 0; maxSize = 1;
        }

        if (hue != null && !hue.isEmpty()) {
            Map<String, List<double[]>> groups = new LinkedHashMap<>();
            for (int i = 0; i < n; i++) {
                String h = i < hue.size() ? hue.get(i) : "Default";
                double s = Echarts3dPlotVectors.toDouble(sizes.get(i));
                double normalizedSize = 5 + 25 * (s - minSize) / (maxSize - minSize);
                groups.computeIfAbsent(h, k -> new ArrayList<>())
                        .add(new double[]{Echarts3dPlotVectors.toDouble(x.get(i)), Echarts3dPlotVectors.toDouble(y.get(i)), Echarts3dPlotVectors.toDouble(z.get(i)), normalizedSize});
            }

            String[] colors = Echarts3dPalette.colors(paletteName,groups.size());
            int idx = 0;
            for (Map.Entry<String, List<double[]>> entry : groups.entrySet()) {
                Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("scatter3D", entry.getKey());
                series.data = entry.getValue().toArray(new double[0][]);
                series.color = colors[idx++ % colors.length];
                series.symbolSize = -1; // 使用数据中的第四维作为大小
                seriesList.add(series);
            }
        } else {
            double[][] data = new double[n][4];
            for (int i = 0; i < n; i++) {
                double s = Echarts3dPlotVectors.toDouble(sizes.get(i));
                data[i][0] = Echarts3dPlotVectors.toDouble(x.get(i));
                data[i][1] = Echarts3dPlotVectors.toDouble(y.get(i));
                data[i][2] = Echarts3dPlotVectors.toDouble(z.get(i));
                data[i][3] = 5 + 25 * (s - minSize) / (maxSize - minSize);
            }
            Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("scatter3D", "Data");
            series.data = data;
            series.color = style != null ? style.getColor() : Echarts3dPalette.colors(paletteName,1)[0];
            series.symbolSize = -1;
            seriesList.add(series);
        }

        updateBounds(x, y, z);
        return this;
    }

    @Override
    public I3dPlot line3d(IVector x, IVector y, IVector z) {
        return line3d(x, y, z, (PlotStyle) null);
    }

    @Override
    public I3dPlot line3d(IVector x, IVector y, IVector z, String styleString) {
        return line3d(x, y, z, parseStyle(styleString));
    }

    @Override
    public I3dPlot line3d(IVector x, IVector y, IVector z, PlotStyle style) {
        int n = Math.min(x.length(), Math.min(y.length(), z.length()));

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("line3D", "Line");
        series.data = Echarts3dPlotVectors.extractXYZ(x, y, z, n);
        series.color = style != null ? style.getColor() : Echarts3dPalette.colors(paletteName,1)[0];
        series.symbolSize = 0; // 不显示点
        series.extra.put("lineStyle", Map.of(
                "width", style != null ? style.getLineWidth() : 2,
                "opacity", 0.9
        ));
        seriesList.add(series);

        updateBounds(x, y, z);
        return this;
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z) {
        return density3d(x, y, z, 24, (PlotStyle) null);
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, int resolution) {
        return density3d(x, y, z, resolution, (PlotStyle) null);
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, String styleString) {
        return density3d(x, y, z, 24, parseStyle(styleString));
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, PlotStyle style) {
        return density3d(x, y, z, 24, style);
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, int resolution, String styleString) {
        return density3d(x, y, z, resolution, parseStyle(styleString));
    }

    @Override
    public I3dPlot density3d(IVector x, IVector y, IVector z, int resolution, PlotStyle style) {
        int n = Math.min(x.length(), Math.min(y.length(), z.length()));
        int g = Math.max(8, Math.min(40, resolution <= 0 ? 16 : resolution));

        double[] xRange = Echarts3dPlotVectors.range(x);
        double[] yRange = Echarts3dPlotVectors.range(y);
        double[] zRange = Echarts3dPlotVectors.range(z);

        // 3D直方图/密度体素
        double[][][] grid = new double[g][g][g];
        double maxCount = 0;

        for (int i = 0; i < n; i++) {
            int xi = (int) ((Echarts3dPlotVectors.toDouble(x.get(i)) - xRange[0]) / (xRange[1] - xRange[0]) * g);
            int yi = (int) ((Echarts3dPlotVectors.toDouble(y.get(i)) - yRange[0]) / (yRange[1] - yRange[0]) * g);
            int zi = (int) ((Echarts3dPlotVectors.toDouble(z.get(i)) - zRange[0]) / (zRange[1] - zRange[0]) * g);
            xi = Echarts3dPlotVectors.clamp(xi, 0, g - 1);
            yi = Echarts3dPlotVectors.clamp(yi, 0, g - 1);
            zi = Echarts3dPlotVectors.clamp(zi, 0, g - 1);
            grid[xi][yi][zi]++;
            maxCount = Math.max(maxCount, grid[xi][yi][zi]);
        }

        // 转换为散点（体素中心）
        List<double[]> points = new ArrayList<>();
        for (int i = 0; i < g; i++) {
            for (int j = 0; j < g; j++) {
                for (int k = 0; k < g; k++) {
                    if (grid[i][j][k] > 0) {
                        double cx = xRange[0] + (i + 0.5) * (xRange[1] - xRange[0]) / g;
                        double cy = yRange[0] + (j + 0.5) * (yRange[1] - yRange[0]) / g;
                        double cz = zRange[0] + (k + 0.5) * (zRange[1] - zRange[0]) / g;
                        double intensity = grid[i][j][k] / maxCount;
                        points.add(new double[]{cx, cy, cz, intensity});
                    }
                }
            }
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("scatter3D", "Density");
        series.data = points.toArray(new double[0][]);
        series.color = style != null ? style.getColor() : "#5470c6";
        series.symbolSize = -1; // 使用第四维
        series.opacity = 0.6;
        seriesList.add(series);

        updateBounds(x, y, z);
        return this;
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values) {
        return bar3d(categories, values, BarExtrusion3D.BOX, (PlotStyle) null);
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, BarExtrusion3D extrusion) {
        return bar3d(categories, values, extrusion, (PlotStyle) null);
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue) {
        return bar3d(xticks, y, hue, BarExtrusion3D.BOX, (PlotStyle) null);
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion) {
        return bar3d(xticks, y, hue, extrusion, (PlotStyle) null);
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, String styleString) {
        return bar3d(categories, values, BarExtrusion3D.BOX, parseStyle(styleString));
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, PlotStyle style) {
        return bar3d(categories, values, BarExtrusion3D.BOX, style);
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, BarExtrusion3D extrusion, String styleString) {
        return bar3d(categories, values, extrusion, parseStyle(styleString));
    }

    @Override
    public I3dPlot bar3d(List<String> categories, IVector values, BarExtrusion3D extrusion, PlotStyle style) {
        int n = Math.min(categories.size(), values.length());
        String[] colors = Echarts3dPalette.colors(paletteName,n);

        List<double[]> data = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            data.add(new double[]{i, 0, Echarts3dPlotVectors.toDouble(values.get(i))});
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("bar3D", "Bars");
        series.data = data.toArray(new double[0][]);
        series.extra.put("categories", categories.toArray(new String[0]));
        series.extra.put("barSize", new double[]{0.6, 0.6});
        series.color = style != null ? style.getColor() : colors[0];
        seriesList.add(series);

        bounds.put("x", new double[]{0, n});
        bounds.put("y", new double[]{0, 1});
        bounds.put("z", new double[]{0, Echarts3dPlotVectors.maxValue(values)});
        return this;
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, String styleString) {
        return bar3d(xticks, y, hue, BarExtrusion3D.BOX, parseStyle(styleString));
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, PlotStyle style) {
        return bar3d(xticks, y, hue, BarExtrusion3D.BOX, style);
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion, String styleString) {
        return bar3d(xticks, y, hue, extrusion, parseStyle(styleString));
    }

    @Override
    public I3dPlot bar3d(List<String> xticks, IVector y, List<String> hue, BarExtrusion3D extrusion, PlotStyle style) {
        // 分组柱状图实现
        Set<String> uniqueX = new LinkedHashSet<>(xticks);
        Set<String> uniqueHue = new LinkedHashSet<>(hue);
        List<String> xList = new ArrayList<>(uniqueX);
        List<String> hList = new ArrayList<>(uniqueHue);

        String[] colors = Echarts3dPalette.colors(paletteName,hList.size());
        List<double[]> data = new ArrayList<>();

        for (int i = 0; i < xticks.size() && i < y.length(); i++) {
            int xi = xList.indexOf(xticks.get(i));
            int hi = hList.indexOf(hue.get(i));
            data.add(new double[]{xi, hi, Echarts3dPlotVectors.toDouble(y.get(i))});
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("bar3D", "Grouped");
        series.data = data.toArray(new double[0][]);
        series.extra.put("xCategories", xList.toArray(new String[0]));
        series.extra.put("yCategories", hList.toArray(new String[0]));
        series.extra.put("barSize", new double[]{0.5, 0.5});
        seriesList.add(series);

        return this;
    }

    @Override
    public I3dPlot pie3d(IVector data) {
        return pie3d(data, null, (PlotStyle) null);
    }

    @Override
    public I3dPlot pie3d(IVector data, List<String> labels) {
        return pie3d(data, labels, (PlotStyle) null);
    }

    @Override
    public I3dPlot pie3d(IVector data, String styleString) {
        return pie3d(data, null, parseStyle(styleString));
    }

    @Override
    public I3dPlot pie3d(IVector data, PlotStyle style) {
        return pie3d(data, null, style);
    }

    @Override
    public I3dPlot pie3d(IVector data, List<String> labels, String styleString) {
        return pie3d(data, labels, parseStyle(styleString));
    }

    @Override
    public I3dPlot pie3d(IVector data, List<String> labels, PlotStyle style) {
        // 3D饼图使用bar3D堆叠成环形
        int n = data.length();
        double sum = 0;
        for (int i = 0; i < n; i++) sum += Math.max(0, Echarts3dPlotVectors.toDouble(data.get(i)));
        if (sum <= 0) sum = 1;

        String[] colors = Echarts3dPalette.colors(paletteName,n);
        List<double[]> points = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            double frac = Math.max(0, Echarts3dPlotVectors.toDouble(data.get(i))) / sum;
            double angle = i * 2 * Math.PI / n;
            double r = 3;
            double h = frac * 2;
            points.add(new double[]{Math.cos(angle) * r, Math.sin(angle) * r, h});
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("scatter3D", "Pie3D");
        series.data = points.toArray(new double[0][]);
        series.symbolSize = 30;
        seriesList.add(series);

        return this;
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y) {
        return hist3d(x, y, 12, 12, (PlotStyle) null);
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, int xBins, int yBins) {
        return hist3d(x, y, xBins, yBins, (PlotStyle) null);
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, String styleString) {
        return hist3d(x, y, 12, 12, parseStyle(styleString));
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, PlotStyle style) {
        return hist3d(x, y, 12, 12, style);
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, int xBins, int yBins, String styleString) {
        return hist3d(x, y, xBins, yBins, parseStyle(styleString));
    }

    @Override
    public I3dPlot hist3d(IVector x, IVector y, int xBins, int yBins, PlotStyle style) {
        int n = Math.min(x.length(), y.length());
        int bx = Math.max(2, xBins);
        int by = Math.max(2, yBins);

        double[] xRange = Echarts3dPlotVectors.range(x);
        double[] yRange = Echarts3dPlotVectors.range(y);
        double dx = (xRange[1] - xRange[0]) / bx;
        double dy = (yRange[1] - yRange[0]) / by;

        double[][] counts = new double[bx][by];
        double maxCount = 0;
        for (int i = 0; i < n; i++) {
            int xi = (int) ((Echarts3dPlotVectors.toDouble(x.get(i)) - xRange[0]) / dx);
            int yi = (int) ((Echarts3dPlotVectors.toDouble(y.get(i)) - yRange[0]) / dy);
            xi = Echarts3dPlotVectors.clamp(xi, 0, bx - 1);
            yi = Echarts3dPlotVectors.clamp(yi, 0, by - 1);
            counts[xi][yi]++;
            maxCount = Math.max(maxCount, counts[xi][yi]);
        }

        List<double[]> data = new ArrayList<>();
        for (int i = 0; i < bx; i++) {
            for (int j = 0; j < by; j++) {
                if (counts[i][j] > 0) {
                    double cx = xRange[0] + (i + 0.5) * dx;
                    double cy = yRange[0] + (j + 0.5) * dy;
                    data.add(new double[]{cx, cy, counts[i][j]});
                }
            }
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("bar3D", "Histogram");
        series.data = data.toArray(new double[0][]);
        series.color = style != null ? style.getColor() : Echarts3dPalette.colors(paletteName,1)[0];
        seriesList.add(series);

        return this;
    }

    @Override
    public I3dPlot boxplot3d(IVector data, List<String> labels) {
        return boxplot3d(data, labels, (PlotStyle) null);
    }

    @Override
    public I3dPlot boxplot3d(IVector data, List<String> labels, String styleString) {
        return boxplot3d(data, labels, parseStyle(styleString));
    }

    @Override
    public I3dPlot boxplot3d(IVector data, List<String> labels, PlotStyle style) {
        int n = data.length();
        double[] sorted = new double[n];
        for (int i = 0; i < n; i++) sorted[i] = Echarts3dPlotVectors.toDouble(data.get(i));
        Arrays.sort(sorted);

        double q1 = sorted[n / 4];
        double q2 = sorted[n / 2];
        double q3 = sorted[(3 * n) / 4];
        double iqr = q3 - q1;
        double low = Math.max(sorted[0], q1 - 1.5 * iqr);
        double high = Math.min(sorted[n - 1], q3 + 1.5 * iqr);

        // 简化为3D散点表示箱线图关键位置
        double[][] points = {
                {0, 0, low}, {0, 0, q1}, {0, 0, q2}, {0, 0, q3}, {0, 0, high}
        };

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("scatter3D", "Boxplot");
        series.data = points;
        series.symbolSize = 15;
        series.color = style != null ? style.getColor() : Echarts3dPalette.colors(paletteName,1)[0];
        seriesList.add(series);

        return this;
    }

    @Override
    public I3dPlot boxplot3d(IMatrix data, List<String> labels) {
        return boxplot3d(data, labels, (PlotStyle) null);
    }

    @Override
    public I3dPlot boxplot3d(IMatrix data, List<String> labels, String styleString) {
        return boxplot3d(data, labels, parseStyle(styleString));
    }

    @Override
    public I3dPlot boxplot3d(IMatrix data, List<String> labels, PlotStyle style) {
        int cols = data.cols();
        for (int c = 0; c < cols; c++) {
            int rows = data.rows();
            double[] colData = new double[rows];
            for (int r = 0; r < rows; r++) colData[r] = Echarts3dPlotVectors.toDouble(data.get(r, c));
            Arrays.sort(colData);

            double q2 = colData[rows / 2];
            Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("scatter3D", labels != null && c < labels.size() ? labels.get(c) : "Group" + c);
            series.data = new double[][]{{c * 1.5, 0, q2}};
            series.symbolSize = 20;
            seriesList.add(series);
        }
        return this;
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z) {
        return surface3d(x, y, z, false, (PlotStyle) null);
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, boolean bottomContourProjection) {
        return surface3d(x, y, z, bottomContourProjection, (PlotStyle) null);
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, String styleString) {
        return surface3d(x, y, z, false, parseStyle(styleString));
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, PlotStyle style) {
        return surface3d(x, y, z, false, style);
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, boolean bottomContourProjection, String styleString) {
        return surface3d(x, y, z, bottomContourProjection, parseStyle(styleString));
    }

    @Override
    public I3dPlot surface3d(IVector x, IVector y, IMatrix z, boolean bottomContourProjection, PlotStyle style) {
        int nx = x.length();
        int ny = y.length();

        // surface3D需要二维数组形式的数据
        double[][] data = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                data[i][j] = Echarts3dPlotVectors.toDouble(z.get(i, j));
            }
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("surface", "Surface");
        series.data = Echarts3dPlotVectors.convertToPointGrid(x, y, data);
        series.shading = true;
        series.color = style != null ? style.getFaceColor() : Echarts3dPalette.colors(paletteName,1)[0];
        series.extra.put("wireframe", Map.of("show", false));
        seriesList.add(series);

        if (bottomContourProjection) {
            // 添加底部等高线
            Echarts3dSeriesConfig contour = new Echarts3dSeriesConfig("line3D", "Contour");
            // 简化实现：只添加边界框线
            contour.data = Echarts3dPlotVectors.generateContourLinesSimple(x, y, data);
            contour.wireframe = true;
            seriesList.add(contour);
        }

        updateBounds(x, y, z);
        return this;
    }

    @Override
    public I3dPlot contour3d(IVector x, IVector y, IMatrix z) {
        return contour3d(x, y, z, (PlotStyle) null);
    }

    @Override
    public I3dPlot contour3d(IVector x, IVector y, IMatrix z, String styleString) {
        return contour3d(x, y, z, parseStyle(styleString));
    }

    @Override
    public I3dPlot contour3d(IVector x, IVector y, IMatrix z, PlotStyle style) {
        int nx = x.length();
        int ny = y.length();

        double[][] data = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                data[i][j] = Echarts3dPlotVectors.toDouble(z.get(i, j));
            }
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("line3D", "Contour");
        series.data = Echarts3dPlotVectors.generateContourLinesSimple(x, y, data);
        series.color = style != null ? style.getColor() : Echarts3dPalette.colors(paletteName,1)[0];
        series.wireframe = true;
        seriesList.add(series);

        return this;
    }

    @Override
    public I3dPlot wireframe3d(IVector x, IVector y, IMatrix z) {
        return wireframe3d(x, y, z, (PlotStyle) null);
    }

    @Override
    public I3dPlot wireframe3d(IVector x, IVector y, IMatrix z, String styleString) {
        return wireframe3d(x, y, z, parseStyle(styleString));
    }

    @Override
    public I3dPlot wireframe3d(IVector x, IVector y, IMatrix z, PlotStyle style) {
        int nx = x.length();
        int ny = y.length();

        double[][] data = new double[nx][ny];
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                data[i][j] = Echarts3dPlotVectors.toDouble(z.get(i, j));
            }
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("surface", "Wireframe");
        series.data = Echarts3dPlotVectors.convertToPointGrid(x, y, data);
        series.wireframe = true;
        series.shading = false;
        series.color = style != null ? style.getColor() : "#333333";
        seriesList.add(series);

        return this;
    }

    @Override
    public I3dPlot heatmap3d(IMatrix z, List<String> xLabels, List<String> yLabels) {
        return heatmap3d(z, xLabels, yLabels, (PlotStyle) null);
    }

    @Override
    public I3dPlot heatmap3d(IMatrix z, List<String> xLabels, List<String> yLabels, String styleString) {
        return heatmap3d(z, xLabels, yLabels, parseStyle(styleString));
    }

    @Override
    public I3dPlot heatmap3d(IMatrix z, List<String> xLabels, List<String> yLabels, PlotStyle style) {
        int nx = z.rows();
        int ny = z.cols();

        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double v = Echarts3dPlotVectors.toDouble(z.get(i, j));
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }

        List<double[]> data = new ArrayList<>();
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double v = Echarts3dPlotVectors.toDouble(z.get(i, j));
                double h = Math.max(0.1, (v - min) / (max - min + 1e-9) * 5);
                data.add(new double[]{i, j, h, v}); // x, y, height, value
            }
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("bar3D", "Heatmap");
        series.data = data.toArray(new double[0][]);
        series.symbolSize = -1; // 使用value
        series.extra.put("colorBy", "data");
        seriesList.add(series);

        return this;
    }

    @Override
    public I3dPlot waterfall3d(IVector x, IMatrix layerHeights) {
        return waterfall3d(x, layerHeights, (PlotStyle) null);
    }

    @Override
    public I3dPlot waterfall3d(IVector x, IMatrix layerHeights, String styleString) {
        return waterfall3d(x, layerHeights, parseStyle(styleString));
    }

    @Override
    public I3dPlot waterfall3d(IVector x, IMatrix layerHeights, PlotStyle style) {
        int nr = layerHeights.rows();
        int nc = layerHeights.cols();

        String[] colors = Echarts3dPalette.colors(paletteName,nr);

        for (int r = 0; r < nr; r++) {
            List<double[]> lineData = new ArrayList<>();
            for (int c = 0; c < nc && c < x.length(); c++) {
                lineData.add(new double[]{
                        Echarts3dPlotVectors.toDouble(x.get(c)),
                        r * 0.5, // 层偏移
                        Echarts3dPlotVectors.toDouble(layerHeights.get(r, c))
                });
            }

            Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("line3D", "Layer" + r);
            series.data = lineData.toArray(new double[0][]);
            series.color = colors[r % colors.length];
            series.extra.put("lineStyle", Map.of("width", 3));
            seriesList.add(series);
        }

        return this;
    }

    @Override
    public I3dPlot vectorField3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w) {
        return vectorField3d(x, y, z, u, v, w, (PlotStyle) null);
    }

    @Override
    public I3dPlot vectorField3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, String styleString) {
        return vectorField3d(x, y, z, u, v, w, parseStyle(styleString));
    }

    @Override
    public I3dPlot vectorField3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, PlotStyle style) {
        int n = Math.min(x.length(), Math.min(y.length(), Math.min(z.length(),
                Math.min(u.length(), Math.min(v.length(), w.length())))));

        // 计算最大长度用于归一化
        double maxLen = 0;
        for (int i = 0; i < n; i++) {
            double len = Math.sqrt(
                    Echarts3dPlotVectors.toDouble(u.get(i)) * Echarts3dPlotVectors.toDouble(u.get(i)) +
                            Echarts3dPlotVectors.toDouble(v.get(i)) * Echarts3dPlotVectors.toDouble(v.get(i)) +
                            Echarts3dPlotVectors.toDouble(w.get(i)) * Echarts3dPlotVectors.toDouble(w.get(i))
            );
            maxLen = Math.max(maxLen, len);
        }
        if (maxLen == 0) maxLen = 1;

        List<double[]> points = new ArrayList<>();
        double scale = Math.max(0.2, (Echarts3dPlotVectors.maxValue(x) - Echarts3dPlotVectors.minValue(x)) / Math.max(12, n / 2));

        for (int i = 0; i < n; i++) {
            double xi = Echarts3dPlotVectors.toDouble(x.get(i));
            double yi = Echarts3dPlotVectors.toDouble(y.get(i));
            double zi = Echarts3dPlotVectors.toDouble(z.get(i));
            double ui = Echarts3dPlotVectors.toDouble(u.get(i));
            double vi = Echarts3dPlotVectors.toDouble(v.get(i));
            double wi = Echarts3dPlotVectors.toDouble(w.get(i));

            double len = Math.sqrt(ui * ui + vi * vi + wi * wi);
            if (len < 1e-12) continue;

            // 箭头终点
            double arrowLen = scale * len / maxLen;
            points.add(new double[]{xi, yi, zi, 0}); // 起点
            points.add(new double[]{xi + ui / len * arrowLen, yi + vi / len * arrowLen, zi + wi / len * arrowLen, 1}); // 终点
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("line3D", "VectorField");
        series.data = points.toArray(new double[0][]);
        series.color = style != null ? style.getColor() : Echarts3dPalette.colors(paletteName,1)[0];
        series.extra.put("lineStyle", Map.of("width", 2));
        seriesList.add(series);

        return this;
    }

    @Override
    public I3dPlot streamlines3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w) {
        return streamlines3d(x, y, z, u, v, w, (PlotStyle) null);
    }

    @Override
    public I3dPlot streamlines3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, String styleString) {
        return streamlines3d(x, y, z, u, v, w, parseStyle(styleString));
    }

    @Override
    public I3dPlot streamlines3d(IVector x, IVector y, IVector z, IVector u, IVector v, IVector w, PlotStyle style) {
        int n = Math.min(x.length(), Math.min(y.length(), Math.min(z.length(),
                Math.min(u.length(), Math.min(v.length(), w.length())))));

        int seeds = Math.min(24, n);
        double step = Math.max(1e-3, (Echarts3dPlotVectors.maxValue(x) - Echarts3dPlotVectors.minValue(x) + Echarts3dPlotVectors.maxValue(y) - Echarts3dPlotVectors.minValue(y) + Echarts3dPlotVectors.maxValue(z) - Echarts3dPlotVectors.minValue(z)) / 80);

        List<double[]> allPoints = new ArrayList<>();
        String[] colors = Echarts3dPalette.colors(paletteName,seeds);

        for (int s = 0; s < seeds; s++) {
            int si = s * Math.max(1, n / Math.max(1, seeds));
            double px = Echarts3dPlotVectors.toDouble(x.get(si));
            double py = Echarts3dPlotVectors.toDouble(y.get(si));
            double pz = Echarts3dPlotVectors.toDouble(z.get(si));

            List<double[]> streamline = new ArrayList<>();
            streamline.add(new double[]{px, py, pz});

            for (int stepi = 0; stepi < 40; stepi++) {
                // IDW插值计算速度
                double[] vel = Echarts3dPlotVectors.idwVelocity(px, py, pz, x, y, z, u, v, w);
                double nrm = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
                if (nrm < 1e-9) break;

                double vx = vel[0] / nrm * step;
                double vy = vel[1] / nrm * step;
                double vz = vel[2] / nrm * step;

                px += vx;
                py += vy;
                pz += vz;
                streamline.add(new double[]{px, py, pz});
            }

            Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("line3D", "Streamline" + s);
            series.data = streamline.toArray(new double[0][]);
            series.color = colors[s % colors.length];
            series.extra.put("lineStyle", Map.of("width", 1.5));
            seriesList.add(series);
        }

        return this;
    }

    @Override
    public I3dPlot terrain3d(IVector x, IVector y, IMatrix elevation) {
        return surface3d(x, y, elevation, false, (PlotStyle) null);
    }

    @Override
    public I3dPlot terrain3d(IVector x, IVector y, IMatrix elevation, String styleString) {
        return surface3d(x, y, elevation, false, parseStyle(styleString));
    }

    @Override
    public I3dPlot terrain3d(IVector x, IVector y, IMatrix elevation, PlotStyle style) {
        return surface3d(x, y, elevation, false, style);
    }

    @Override
    public I3dPlot graph3d(List<Map<String, Object>> nodes, List<Map<String, Object>> links) {
        return graph3d(nodes, links, (PlotStyle) null);
    }

    @Override
    public I3dPlot graph3d(List<Map<String, Object>> nodes, List<Map<String, Object>> links, String styleString) {
        return graph3d(nodes, links, parseStyle(styleString));
    }

    @Override
    public I3dPlot graph3d(List<Map<String, Object>> nodes, List<Map<String, Object>> links, PlotStyle style) {
        if (nodes == null || nodes.isEmpty()) return this;

        int n = nodes.size();
        double[][] positions = new double[n][3];

        // 提取或生成节点位置
        for (int i = 0; i < n; i++) {
            Map<String, Object> node = nodes.get(i);
            positions[i][0] = node.get("x") instanceof Number ? ((Number) node.get("x")).doubleValue() : Math.random() * 10;
            positions[i][1] = node.get("y") instanceof Number ? ((Number) node.get("y")).doubleValue() : Math.random() * 10;
            positions[i][2] = node.get("z") instanceof Number ? ((Number) node.get("z")).doubleValue() : Math.random() * 5;
        }

        // 节点
        Echarts3dSeriesConfig nodeSeries = new Echarts3dSeriesConfig("scatter3D", "Nodes");
        nodeSeries.data = positions;
        nodeSeries.symbolSize = 15;
        nodeSeries.color = style != null ? style.getColor() : Echarts3dPalette.colors(paletteName,1)[0];
        seriesList.add(nodeSeries);

        // 边
        if (links != null && !links.isEmpty()) {
            List<double[]> edgeData = new ArrayList<>();
            for (Map<String, Object> link : links) {
                int source = Echarts3dPlotVectors.asInt(link.get("source"), 0);
                int target = Echarts3dPlotVectors.asInt(link.get("target"), 0);
                if (source >= 0 && source < n && target >= 0 && target < n) {
                    edgeData.add(positions[source]);
                    edgeData.add(positions[target]);
                }
            }

            Echarts3dSeriesConfig edgeSeries = new Echarts3dSeriesConfig("line3D", "Edges");
            edgeSeries.data = edgeData.toArray(new double[0][]);
            edgeSeries.color = "#999999";
            edgeSeries.extra.put("lineStyle", Map.of("width", 1));
            seriesList.add(edgeSeries);
        }

        return this;
    }

    @Override
    public I3dPlot areaFill3d(IVector x, IVector y, IVector z) {
        return areaFill3d(x, y, z, (PlotStyle) null);
    }

    @Override
    public I3dPlot areaFill3d(IVector x, IVector y, IVector z, String styleString) {
        return areaFill3d(x, y, z, parseStyle(styleString));
    }

    @Override
    public I3dPlot areaFill3d(IVector x, IVector y, IVector z, PlotStyle style) {
        // 简化为填充曲面
        int n = Math.min(x.length(), Math.min(y.length(), z.length()));

        double[][] data = new double[n][3];
        for (int i = 0; i < n; i++) {
            data[i][0] = Echarts3dPlotVectors.toDouble(x.get(i));
            data[i][1] = Echarts3dPlotVectors.toDouble(y.get(i));
            data[i][2] = Echarts3dPlotVectors.toDouble(z.get(i));
        }

        Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("surface", "AreaFill");
        series.data = data;
        series.shading = true;
        series.color = style != null ? style.getFaceColor() : Echarts3dPalette.colors(paletteName,1)[0];
        series.opacity = 0.5;
        seriesList.add(series);

        return this;
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators) {
        return radar3d(data, indicators, null, (PlotStyle) null);
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, List<String> seriesNames) {
        return radar3d(data, indicators, seriesNames, (PlotStyle) null);
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, String styleString) {
        return radar3d(data, indicators, null, parseStyle(styleString));
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, PlotStyle style) {
        return radar3d(data, indicators, null, style);
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, List<String> seriesNames, String styleString) {
        return radar3d(data, indicators, seriesNames, parseStyle(styleString));
    }

    @Override
    public I3dPlot radar3d(IMatrix data, List<String> indicators, List<String> seriesNames, PlotStyle style) {
        int axes = Math.min(indicators != null ? indicators.size() : 0, data.cols());
        int rows = data.rows();

        double maxV = 1;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < axes; j++) {
                maxV = Math.max(maxV, Math.abs(Echarts3dPlotVectors.toDouble(data.get(i, j))));
            }
        }
        if (maxV <= 0) maxV = 1;

        String[] colors = Echarts3dPalette.colors(paletteName,rows);

        for (int r = 0; r < rows; r++) {
            List<double[]> points = new ArrayList<>();
            double yLayer = r * 1.2;

            for (int t = 0; t < axes; t++) {
                double ang = t * 2 * Math.PI / axes;
                double vr = Echarts3dPlotVectors.toDouble(data.get(r, t)) / maxV * 3;
                points.add(new double[]{vr * Math.cos(ang), yLayer, vr * Math.sin(ang)});
            }
            // 闭合
            if (!points.isEmpty()) points.add(points.get(0));

            Echarts3dSeriesConfig series = new Echarts3dSeriesConfig("line3D",
                    seriesNames != null && r < seriesNames.size() ? seriesNames.get(r) : "Series" + r);
            series.data = points.toArray(new double[0][]);
            series.color = colors[r % colors.length];
            series.extra.put("lineStyle", Map.of("width", 2));
            seriesList.add(series);
        }

        return this;
    }

    // ==================== 样式与主题 ====================

    @Override
    public I3dPlot setDefaultStyle(PlotStyle style) {
        this.defaultStyle = style != null ? new PlotStyle(style) : PlotStyle.defaultStyle();
        return this;
    }

    @Override
    public I3dPlot setPalette(String paletteName) {
        this.paletteName = paletteName;
        return this;
    }

    @Override
    public I3dPlot enableStyleSystem(boolean enabled) {
        this.useStyleSystem = enabled;
        return this;
    }

    @Override
    public I3dPlot enableThemeSystem(boolean enabled) {
        this.useThemeSystem = enabled;
        return this;
    }

    @Override
    public I3dPlot applyTheme(String themeName) {
        if (themeName != null && useThemeSystem) {
            this.themeName = themeName;
        }
        return this;
    }

    @Override
    public I3dPlot registerTheme(String themeName, EchartsThemeManager.CustomTheme theme) {
        if (themeName != null && theme != null) {
            EchartsThemeManager.registerCustomTheme(themeName, theme);
        }
        return this;
    }

    @Override
    public I3dPlot createGradientTheme(String themeName, String startColor, String endColor, String backgroundColor) {
        if (themeName == null) return this;
        EchartsThemeManager.CustomTheme theme = EchartsThemeManager.createGradientTheme(
                themeName, startColor, endColor, backgroundColor);
        EchartsThemeManager.registerCustomTheme(themeName, theme);
        return this;
    }

    @Override
    public I3dPlot createMonochromeTheme(String themeName, String baseColor, String backgroundColor) {
        if (themeName == null) return this;
        EchartsThemeManager.CustomTheme theme = EchartsThemeManager.createMonochromeTheme(
                themeName, baseColor, backgroundColor);
        EchartsThemeManager.registerCustomTheme(themeName, theme);
        return this;
    }

    // ==================== 流式API ====================

    @Override
    public I3dPlot title(String titleText) {
        labels.put("title", titleText != null ? titleText : "");
        return this;
    }

    @Override
    public I3dPlot title(String titleText, String subtitleText) {
        labels.put("title", titleText != null ? titleText : "");
        labels.put("subtitle", subtitleText != null ? subtitleText : "");
        return this;
    }

    @Override
    public I3dPlot xlabel(String name) {
        labels.put("x", name != null ? name : "");
        return this;
    }

    @Override
    public I3dPlot ylabel(String name) {
        labels.put("y", name != null ? name : "");
        return this;
    }

    @Override
    public I3dPlot zlabel(String name) {
        labels.put("z", name != null ? name : "");
        return this;
    }

    @Override
    public I3dPlot size(int width, int height) {
        this.width = width;
        this.height = height;
        return this;
    }

    @Override
    public I3dPlot theme(String theme) {
        return applyTheme(theme);
    }

    @Override
    public I3dPlot show() {
        // 生成临时HTML文件并打开浏览器
        try {
            File tempFile = File.createTempFile("echarts3d_", ".html");
            saveAsHtml(tempFile.getAbsolutePath());
            Desktop.getDesktop().browse(tempFile.toURI());
        } catch (Exception e) {
            throw new PlotException("无法打开3D图表: " + e.getMessage(), e);
        }
        return this;
    }

    @Override
    public I3dPlot saveAsHtml(String filename) {
        String html = generateFullHtml();
        try {
            Files.writeString(Path.of(filename), html, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new PlotException("保存HTML失败: " + e.getMessage(), e);
        }
        return this;
    }

    @Override
    public String toHtml() {
        return generateFullHtml();
    }

    @Override
    public String toJson() {
        return Echarts3dOptionJson.build(seriesList, sceneConfig, labels);
    }

    // ==================== 配置方法 ====================

    @Override
    public void setTitle(String titleText) {
        labels.put("title", titleText != null ? titleText : "");
    }

    @Override
    public void setTitle(String titleText, String subtitleText) {
        labels.put("title", titleText != null ? titleText : "");
        labels.put("subtitle", subtitleText != null ? subtitleText : "");
    }

    @Override
    public void setXlabel(String name) {
        labels.put("x", name != null ? name : "");
    }

    @Override
    public void setYlabel(String name) {
        labels.put("y", name != null ? name : "");
    }

    @Override
    public void setZlabel(String name) {
        labels.put("z", name != null ? name : "");
    }

    @Override
    public void setXticks(AxisTicks xticks) {
        this.xticks = xticks;
    }

    @Override
    public void setYticks(AxisTicks yticks) {
        this.yticks = yticks;
    }

    @Override
    public void setZticks(AxisTicks zticks) {
        this.zticks = zticks;
    }

    @Override
    public int getWidth() {
        return width;
    }

    @Override
    public int getHeight() {
        return height;
    }

    @Override
    public String getTheme() {
        return themeName;
    }

    // ==================== 私有辅助方法 ====================

    private void updateBounds(IVector x, IVector y, IVector z) {
        bounds.put("x", Echarts3dPlotVectors.range(x));
        bounds.put("y", Echarts3dPlotVectors.range(y));
        bounds.put("z", Echarts3dPlotVectors.range(z));
    }

    private void updateBounds(IVector x, IVector y, IMatrix z) {
        bounds.put("x", Echarts3dPlotVectors.range(x));
        bounds.put("y", Echarts3dPlotVectors.range(y));
        bounds.put("z", Echarts3dPlotVectors.matrixZBounds(z));
    }

    private PlotStyle parseStyle(String styleString) {
        if (styleString == null || !useStyleSystem) return defaultStyle;
        return defaultStyle;
    }

    private String generateFullHtml() {
        return Echarts3dHtmlTemplate.fullPage(seriesList, sceneConfig, labels, themeName);
    }

    // 特殊writeObject/readObject方法处理序列化
    @Serial
    private void writeObject(java.io.ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
    }

    @Serial
    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        // 恢复不可序列化字段的默认值
        Echarts3dSceneDefaults.ensureDefaults(sceneConfig);
        if (defaultStyle == null) defaultStyle = PlotStyle.defaultStyle();
    }
}
