package com.yishape.lab.math.plot.javafx.renderers;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxStyleApplier;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.scene.shape.StrokeLineJoin;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Unified Cartesian renderer for line, scatter, area, step, bubble, error bar overlays,
 * log-scaled axes and optional secondary Y (seaborn/matplotlib parity on desktop).
 * <p>线、面积、阶梯序列在绘制路径时按 x 升序连点，避免未排序表格行序导致折线折返与虚线视觉异常。</p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class CartesianComboRenderer implements JavaFxChartRenderer {

    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) {
            return;
        }
        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);
        String[] palette = themeManager.getColorPalette();
        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        double[] xRange = (config.axisLockX != null && config.axisLockX.length == 2)
            ? new double[] { config.axisLockX[0], config.axisLockX[1] } : rangeX(data);
        double[] yRange = (config.axisLockY != null && config.axisLockY.length == 2)
            ? new double[] { config.axisLockY[0], config.axisLockY[1] } : rangeY(data, 0, config);
        double[] y2Range = hasSecondaryY(data) ? rangeY(data, 1, config) : null;

        config.xRange = xRange;
        config.yRange = yRange;

        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        Double y2Min = y2Range != null ? y2Range[0] : null;
        Double y2Max = y2Range != null ? y2Range[1] : null;
        JavaFxChartUtils.drawAxesCartesian(gc, config, xRange[0], xRange[1], yRange[0], yRange[1],
            y2Min, y2Max, themeManager);

        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            order.add(i);
        }
        order.sort(Comparator.comparingInt(idx -> layerOrder(data.get(idx))));

        for (int si : order) {
            gc.save();
            try {
                SeriesData series = data.get(si);
                String t = series.type != null ? series.type : "line";
                PlotAxisPair scales = scalesForSeries(config, series.yAxisIndex);
                double yMin = series.yAxisIndex == 0 ? yRange[0] : y2Range[0];
                double yMax = series.yAxisIndex == 0 ? yRange[1] : y2Range[1];

                if ("ci_band".equals(t)) {
                    if (series.x != null && ciLow(series) != null && ciHigh(series) != null) {
                        drawCiBand(gc, series, config, xRange[0], xRange[1], yMin, yMax, scales, themeManager, palette, si);
                    }
                    continue;
                }
                if (series.x == null || series.y == null) {
                    continue;
                }
                if ("errorbar".equals(t)) {
                    drawErrorBars(gc, series, config, xRange[0], xRange[1], yMin, yMax, scales.yScale,
                        themeManager, palette, si);
                } else if ("scatter".equals(t)) {
                    drawScatter(gc, series, config, xRange[0], xRange[1], yMin, yMax, scales,
                        themeManager, palette, si);
                } else if ("area".equals(t)) {
                    drawArea(gc, series, config, xRange[0], xRange[1], yMin, yMax, scales, themeManager, palette, si);
                } else if ("step".equals(t)) {
                    drawStep(gc, series, config, xRange[0], xRange[1], yMin, yMax, scales, themeManager, palette, si);
                } else {
                    drawLine(gc, series, config, xRange[0], xRange[1], yMin, yMax, scales, themeManager, palette, si);
                }
            } finally {
                gc.restore();
            }
        }
        if (config.showLegend) {
            JavaFxChartUtils.drawLegend(gc, data, config, themeManager);
        }
    }

    private static int layerOrder(SeriesData s) {
        String t = s.type != null ? s.type : "line";
        if ("ci_band".equals(t)) {
            return 0;
        }
        if ("area".equals(t)) {
            return 1;
        }
        if ("step".equals(t)) {
            return 2;
        }
        if ("line".equals(t)) {
            return 3;
        }
        if ("errorbar".equals(t)) {
            return 4;
        }
        if ("scatter".equals(t)) {
            return 5;
        }
        return 3;
    }

    private record PlotAxisPair(com.yishape.lab.math.plot.PlotAxisScale xScale,
                                com.yishape.lab.math.plot.PlotAxisScale yScale) {
    }

    private static PlotAxisPair scalesForSeries(ChartConfig config, int yAxisIndex) {
        if (yAxisIndex == 1) {
            return new PlotAxisPair(config.xAxisScale, com.yishape.lab.math.plot.PlotAxisScale.LINEAR);
        }
        return new PlotAxisPair(config.xAxisScale, config.yAxisScale);
    }

    private static boolean hasSecondaryY(List<SeriesData> data) {
        for (SeriesData s : data) {
            if (s.yAxisIndex == 1) {
                return true;
            }
        }
        return false;
    }

    private static double[] rangeX(List<SeriesData> data) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (SeriesData s : data) {
            if (s.x == null) {
                continue;
            }
            for (int i = 0; i < s.x.length(); i++) {
                double v = s.x.get(i);
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }
        return padRange(min, max);
    }

    @SuppressWarnings("unchecked")
    private static IVector<Double> ciLow(SeriesData s) {
        Object o = s.extraData != null ? s.extraData.get("yLow") : null;
        return o instanceof IVector ? (IVector<Double>) o : null;
    }

    @SuppressWarnings("unchecked")
    private static IVector<Double> ciHigh(SeriesData s) {
        Object o = s.extraData != null ? s.extraData.get("yHigh") : null;
        return o instanceof IVector ? (IVector<Double>) o : null;
    }

    private static double[] rangeY(List<SeriesData> data, int yAxisIndex, ChartConfig config) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (SeriesData s : data) {
            if (s.yAxisIndex != yAxisIndex) {
                continue;
            }
            if ("ci_band".equals(s.type)) {
                IVector<Double> lo = ciLow(s);
                IVector<Double> hi = ciHigh(s);
                if (lo != null && hi != null) {
                    for (int i = 0; i < lo.length(); i++) {
                        min = Math.min(min, lo.get(i));
                        max = Math.max(max, hi.get(i));
                    }
                }
                if (s.y != null) {
                    for (int i = 0; i < s.y.length(); i++) {
                        double v = s.y.get(i);
                        min = Math.min(min, v);
                        max = Math.max(max, v);
                    }
                }
                continue;
            }
            if (s.y == null) {
                continue;
            }
            IVector<Double> err = errorVector(s);
            for (int i = 0; i < s.y.length(); i++) {
                double v = s.y.get(i);
                min = Math.min(min, v);
                max = Math.max(max, v);
                if (err != null && i < err.length()) {
                    double e = err.get(i);
                    min = Math.min(min, v - e);
                    max = Math.max(max, v + e);
                }
            }
        }
        if (config.yAxisScale == com.yishape.lab.math.plot.PlotAxisScale.LOG && yAxisIndex == 0) {
            double eps = 1e-12;
            if (min <= 0) {
                min = Math.max(eps, max * 1e-6);
            }
            if (max <= min) {
                max = min * 10;
            }
        }
        return padRange(min, max);
    }

    @SuppressWarnings("unchecked")
    private static IVector<Double> errorVector(SeriesData s) {
        Object oly = s.extraData != null ? s.extraData.get("yerr") : null;
        return oly instanceof IVector ? (IVector<Double>) oly : null;
    }

    private static double[] padRange(double min, double max) {
        if (min == Double.POSITIVE_INFINITY) {
            return new double[] { 0, 1 };
        }
        double margin = (max - min) * 0.1;
        if (margin == 0) {
            margin = Math.abs(min) * 0.1;
        }
        if (margin == 0) {
            margin = 1;
        }
        return new double[] { min - margin, max + margin };
    }

    /** 按 x 升序（同 x 按索引）排列点序，用于折线/面积/阶梯的路径，避免行序乱导致折返。 */
    private static List<Integer> sortedIndicesByX(SeriesData series) {
        int n = series.x.length();
        List<Integer> ord = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            ord.add(i);
        }
        ord.sort(Comparator.<Integer>comparingDouble(i -> series.x.get(i)).thenComparingInt(i -> i));
        return ord;
    }

    private void drawCiBand(GraphicsContext gc, SeriesData series, ChartConfig config,
                           double xMin, double xMax, double yMin, double yMax,
                           PlotAxisPair scales, JavaFxThemeManager themeManager, String[] palette, int seriesIndex) {
        IVector<Double> lo = ciLow(series);
        IVector<Double> hi = ciHigh(series);
        if (lo == null || hi == null || series.x == null || lo.length() != series.x.length()
            || hi.length() != series.x.length()) {
            return;
        }
        int n = series.x.length();
        // 置信带必须按 x 升序连成简单多边形；原始数据常为表格行序（x 非单调），否则路径自交导致填充断裂/假空洞
        List<Integer> ord = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            ord.add(i);
        }
        ord.sort(Comparator.<Integer>comparingDouble(i -> series.x.get(i)).thenComparingInt(i -> i));

        Color stroke = series.style != null && series.style.getColor() != null
            ? JavaFxStyleApplier.parseColor(series.style.getColor())
            : Color.web(palette[seriesIndex % palette.length]);
        Color fill = Color.color(stroke.getRed(), stroke.getGreen(), stroke.getBlue(), 0.22);
        gc.beginPath();
        for (int k = 0; k < n; k++) {
            int i = ord.get(k);
            double px = JavaFxChartUtils.mapXToPixel(config, series.x.get(i), xMin, xMax);
            double py = JavaFxChartUtils.mapYToPixel(config, lo.get(i), yMin, yMax, scales.yScale);
            if (k == 0) {
                gc.moveTo(px, py);
            } else {
                gc.lineTo(px, py);
            }
        }
        for (int k = n - 1; k >= 0; k--) {
            int i = ord.get(k);
            double px = JavaFxChartUtils.mapXToPixel(config, series.x.get(i), xMin, xMax);
            double py = JavaFxChartUtils.mapYToPixel(config, hi.get(i), yMin, yMax, scales.yScale);
            gc.lineTo(px, py);
        }
        gc.closePath();
        gc.setFill(fill);
        gc.fill();
    }

    private void drawLine(GraphicsContext gc, SeriesData series, ChartConfig config,
                         double xMin, double xMax, double yMin, double yMax,
                         PlotAxisPair scales, JavaFxThemeManager themeManager, String[] palette, int seriesIndex) {
        if (series.style != null) {
            JavaFxStyleApplier.applyStyle(gc, series.style);
        } else {
            gc.setStroke(Color.web(palette[seriesIndex % palette.length]));
            gc.setLineWidth(2);
        }
        gc.setLineCap(StrokeLineCap.ROUND);
        gc.setLineJoin(StrokeLineJoin.ROUND);
        List<Integer> ord = series.sortLineByY ? sortedIndicesByY(series) : sortedIndicesByX(series);
        gc.beginPath();
        for (int k = 0; k < ord.size(); k++) {
            int i = ord.get(k);
            double px = JavaFxChartUtils.mapXToPixel(config, series.x.get(i), xMin, xMax);
            double py = JavaFxChartUtils.mapYToPixel(config, series.y.get(i), yMin, yMax, scales.yScale);
            if (config.hitTestHandler != null) {
                JavaFxChartUtils.registerHit(config, px, py, series.x.get(i), series.y.get(i), series.name, seriesIndex, i);
            }
            if (k == 0) {
                gc.moveTo(px, py);
            } else {
                gc.lineTo(px, py);
            }
        }
        gc.stroke();
        maybeMarkers(gc, series, config, xMin, xMax, yMin, yMax, scales.yScale);
    }

    /** 边际密度 p(y)：数据为 x=密度、y=样本维取值 时应按 y 排序（按 x=密度排序会乱序绕圈，视觉上像“填充”）。 */
    private static List<Integer> sortedIndicesByY(SeriesData series) {
        int n = series.y.length();
        List<Integer> ord = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            ord.add(i);
        }
        ord.sort(Comparator.<Integer>comparingDouble(i -> series.y.get(i)).thenComparingInt(i -> i));
        return ord;
    }

    private void drawArea(GraphicsContext gc, SeriesData series, ChartConfig config,
                          double xMin, double xMax, double yMin, double yMax,
                          PlotAxisPair scales, JavaFxThemeManager themeManager, String[] palette, int seriesIndex) {
        Color stroke = series.style != null && series.style.getColor() != null
            ? JavaFxStyleApplier.parseColor(series.style.getColor())
            : Color.web(palette[seriesIndex % palette.length]);
        Color fill = Color.color(stroke.getRed(), stroke.getGreen(), stroke.getBlue(), 0.35);
        List<Integer> ord = sortedIndicesByX(series);
        int n = ord.size();
        gc.beginPath();
        for (int k = 0; k < n; k++) {
            int i = ord.get(k);
            double px = JavaFxChartUtils.mapXToPixel(config, series.x.get(i), xMin, xMax);
            double py = JavaFxChartUtils.mapYToPixel(config, series.y.get(i), yMin, yMax, scales.yScale);
            if (k == 0) {
                gc.moveTo(px, py);
            } else {
                gc.lineTo(px, py);
            }
        }
        double baselineY = JavaFxChartUtils.mapYToPixel(config, Math.max(0, yMin), yMin, yMax, scales.yScale);
        int iLast = ord.get(n - 1);
        int iFirst = ord.get(0);
        gc.lineTo(JavaFxChartUtils.mapXToPixel(config, series.x.get(iLast), xMin, xMax), baselineY);
        gc.lineTo(JavaFxChartUtils.mapXToPixel(config, series.x.get(iFirst), xMin, xMax), baselineY);
        gc.closePath();
        gc.setFill(fill);
        gc.fill();
        gc.setStroke(stroke);
        gc.setLineWidth(2);
        gc.stroke();
    }

    private void drawStep(GraphicsContext gc, SeriesData series, ChartConfig config,
                         double xMin, double xMax, double yMin, double yMax,
                         PlotAxisPair scales, JavaFxThemeManager themeManager, String[] palette, int seriesIndex) {
        if (series.style != null) {
            JavaFxStyleApplier.applyStyle(gc, series.style);
        } else {
            gc.setStroke(Color.web(palette[seriesIndex % palette.length]));
            gc.setLineWidth(2);
        }
        gc.beginPath();
        List<Integer> ord = sortedIndicesByX(series);
        int n = ord.size();
        for (int k = 0; k < n; k++) {
            int i = ord.get(k);
            double px = JavaFxChartUtils.mapXToPixel(config, series.x.get(i), xMin, xMax);
            double py = JavaFxChartUtils.mapYToPixel(config, series.y.get(i), yMin, yMax, scales.yScale);
            if (k == 0) {
                gc.moveTo(px, py);
            } else {
                int ip = ord.get(k - 1);
                gc.lineTo(px, JavaFxChartUtils.mapYToPixel(config, series.y.get(ip), yMin, yMax, scales.yScale));
                gc.lineTo(px, py);
            }
            if (config.hitTestHandler != null) {
                JavaFxChartUtils.registerHit(config, px, py, series.x.get(i), series.y.get(i), series.name, seriesIndex, i);
            }
        }
        gc.stroke();
    }

    private void drawScatter(GraphicsContext gc, SeriesData series, ChartConfig config,
                            double xMin, double xMax, double yMin, double yMax,
                            PlotAxisPair scales, JavaFxThemeManager themeManager, String[] palette, int seriesIndex) {
        IVector<Double> sizes = bubbleSizes(series);
        double maxS = 1;
        if (sizes != null) {
            for (int i = 0; i < sizes.length(); i++) {
                maxS = Math.max(maxS, sizes.get(i));
            }
        }
        for (int i = 0; i < series.x.length(); i++) {
            double px = JavaFxChartUtils.mapXToPixel(config, series.x.get(i), xMin, xMax);
            double py = JavaFxChartUtils.mapYToPixel(config, series.y.get(i), yMin, yMax, scales.yScale);
            double base = series.style != null ? series.style.getMarkerSize() : 6;
            double r = base;
            if (sizes != null && i < sizes.length()) {
                r = base * (0.35 + 0.65 * Math.sqrt(Math.max(0, sizes.get(i)) / Math.sqrt(maxS)));
            }
            Color c = series.style != null && series.style.getColor() != null
                ? JavaFxStyleApplier.parseColor(series.style.getColor())
                : Color.web(palette[seriesIndex % palette.length]);
            gc.setFill(c);
            JavaFxChartUtils.registerHit(config, px, py, series.x.get(i), series.y.get(i), series.name, seriesIndex, i);
            JavaFxStyleApplier.drawMarker(gc, px, py, JavaFxStyleApplier.MarkerType.CIRCLE, Math.max(2, r));
        }
    }

    @SuppressWarnings("unchecked")
    private static IVector<Double> bubbleSizes(SeriesData s) {
        Object o = s.extraData != null ? s.extraData.get("sizes") : null;
        return o instanceof IVector ? (IVector<Double>) o : null;
    }

    private void drawErrorBars(GraphicsContext gc, SeriesData series, ChartConfig config,
                              double xMin, double xMax, double yMin, double yMax,
                              com.yishape.lab.math.plot.PlotAxisScale yScale,
                              JavaFxThemeManager themeManager, String[] palette, int seriesIndex) {
        IVector<Double> err = errorVector(series);
        if (err == null) {
            return;
        }
        gc.setStroke(themeManager.getTextColor());
        gc.setLineWidth(1.25);
        double cap = 5;
        for (int i = 0; i < series.x.length() && i < err.length(); i++) {
            double x = series.x.get(i);
            double y = series.y.get(i);
            double e = err.get(i);
            double px = JavaFxChartUtils.mapXToPixel(config, x, xMin, xMax);
            double pyTop = JavaFxChartUtils.mapYToPixel(config, y + e, yMin, yMax, yScale);
            double pyBot = JavaFxChartUtils.mapYToPixel(config, y - e, yMin, yMax, yScale);
            gc.strokeLine(px, pyTop, px, pyBot);
            gc.strokeLine(px - cap, pyTop, px + cap, pyTop);
            gc.strokeLine(px - cap, pyBot, px + cap, pyBot);
            JavaFxChartUtils.registerHit(config, px, (pyTop + pyBot) / 2, x, y, series.name, seriesIndex, i);
        }
    }

    private void maybeMarkers(GraphicsContext gc, SeriesData series, ChartConfig config,
                             double xMin, double xMax, double yMin, double yMax,
                             com.yishape.lab.math.plot.PlotAxisScale yScale) {
        if (series.style == null || series.style.getMarker() == null || series.style.getMarker().isEmpty()) {
            return;
        }
        JavaFxStyleApplier.applyMarkerStyle(gc, series.style);
        var markerType = JavaFxStyleApplier.getMarkerType(series.style.getMarker());
        double markerSize = series.style.getMarkerSize();
        for (int i = 0; i < series.x.length(); i++) {
            double px = JavaFxChartUtils.mapXToPixel(config, series.x.get(i), xMin, xMax);
            double py = JavaFxChartUtils.mapYToPixel(config, series.y.get(i), yMin, yMax, yScale);
            JavaFxStyleApplier.drawMarker(gc, px, py, markerType, markerSize);
        }
    }

    @Override
    public String getChartType() {
        return "cartesian";
    }

    @Override
    public boolean supportsAnimation() {
        return true;
    }

    @Override
    public int getAnimationDuration() {
        return 900;
    }
}
