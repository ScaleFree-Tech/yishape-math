package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.PlotAxisScale;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * 笛卡尔图表SVG渲染器：折线、散点、面积图、阶梯图、气泡图、误差棒、置信带。
 * 支持对数坐标轴、双Y轴、副Y轴标签。
 */
public class SvgCartesianRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        // 多序列委托给 renderMultiSeries，最终 dispatch 到各子方法
        List<SeriesData> allSeries = new ArrayList<>();
        allSeries.add(series);
        renderMultiSeries(sb, allSeries, config, themeManager);
    }

    @Override
    public String renderMulti(List<SeriesData> seriesList, ChartConfig config, String theme) {
        if (seriesList.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        buildHeader(sb, config);
        themeManager = new JavaFxThemeManager(theme);
        applyTheme(themeManager);
        renderBackground(sb, config);
        renderTitle(sb, config);
        renderMultiSeries(sb, seriesList, config, themeManager);
        buildFooter(sb);
        return sb.toString();
    }

    /** 完整渲染（支持多序列叠加） */
    public void renderMultiSeries(StringBuilder sb, List<SeriesData> dataList,
                                  ChartConfig config, JavaFxThemeManager themeManager) {
        if (dataList.isEmpty()) return;
        String[] palette = themeManager.getColorPalette();

        double[] xRange = rangeX(dataList);
        double[] yRange = rangeY(dataList, 0);
        double[] y2Range = hasSecondaryY(dataList) ? rangeY(dataList, 1) : null;

        drawBoxOnly(sb, config, xRange[0], xRange[1], yRange[0], yRange[1],
                 config.xlabel, config.ylabel);
        if (y2Range != null) {
            drawRightYAxis(sb, config, y2Range[0], y2Range[1]);
        }

        // 按层级顺序绘制
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < dataList.size(); i++) order.add(i);
        order.sort(Comparator.comparingInt(idx -> layerOrder(dataList.get(idx))));
        for (int si : order) {
            SeriesData s = dataList.get(si);
            String t = s.type != null ? s.type : "line";
            double yMin = s.yAxisIndex == 0 ? yRange[0] : y2Range[0];
            double yMax = s.yAxisIndex == 0 ? yRange[1] : y2Range[1];
            PlotAxisScale yScale = s.yAxisIndex == 1 ? PlotAxisScale.LINEAR : config.yAxisScale;
            String color = palette[si % palette.length];

            if ("ci_band".equals(t)) {
                drawCiBand(sb, s, config, xRange[0], xRange[1], yMin, yMax, yScale, color);
            } else if (isErrorBarType(t)) {
                drawErrorBars(sb, s, config, xRange[0], xRange[1], yMin, yMax, yScale, color);
            } else if ("scatter".equals(t)) {
                drawScatter(sb, s, config, xRange[0], xRange[1], yMin, yMax, yScale, color);
            } else if ("bubble".equals(t)) {
                drawBubble(sb, s, config, xRange[0], xRange[1], yMin, yMax, yScale, color);
            } else if ("area".equals(t)) {
                drawArea(sb, s, config, xRange[0], xRange[1], yMin, yMax, yScale, color);
            } else if ("step".equals(t)) {
                drawStep(sb, s, config, xRange[0], xRange[1], yMin, yMax, yScale, color);
            } else {
                drawLine(sb, s, config, xRange[0], xRange[1], yMin, yMax, yScale, color);
            }
        }
    }

    private static boolean isErrorBarType(String t) {
        return t != null && "errorbar".equalsIgnoreCase(t);
    }

    private int layerOrder(SeriesData s) {
        String t = s.type != null ? s.type : "line";
        if ("ci_band".equals(t)) return 0;
        if ("area".equals(t)) return 1;
        if ("step".equals(t)) return 2;
        if ("line".equals(t)) return 3;
        if (isErrorBarType(t)) return 4;
        if ("scatter".equals(t)) return 5;
        return 3;
    }

    /** 右侧 Y 轴刻度与标签（副轴） */
    private void drawRightYAxis(StringBuilder sb, ChartConfig config, double yMin, double yMax) {
        double cRight = config.width - config.paddingRight;
        double cTop = config.paddingTop;
        double cBottom = config.height - config.paddingBottom;
        double cHeight = cBottom - cTop;
        int yTicks = Math.min(axisTickCount(config.height), 8);
        sb.append("<line x1=\"").append((int) cRight).append("\" y1=\"").append((int) cTop)
          .append("\" x2=\"").append((int) cRight).append("\" y2=\"").append((int) cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");
        for (int i = 0; i <= yTicks; i++) {
            double y = cBottom - i * cHeight / yTicks;
            double val = yMin + i * (yMax - yMin) / yTicks;
            sb.append("<text x=\"").append((int) (cRight + 8)).append("\" y=\"").append((int) (y + 4))
              .append("\" text-anchor=\"start\" class=\"tick-label\">")
              .append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int) cRight).append("\" y1=\"").append((int) y)
              .append("\" x2=\"").append((int) (cRight + 4)).append("\" y2=\"").append((int) y)
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }
        if (config.y2AxisLabel != null && !config.y2AxisLabel.isEmpty()) {
            int cy = (int) (cTop + cHeight / 2);
            int rx = config.width - 10;
            sb.append("<text x=\"").append(rx).append("\" y=\"").append(cy)
              .append("\" text-anchor=\"middle\" class=\"axis-label\" ")
              .append("transform=\"rotate(90, ").append(rx).append(", ").append(cy).append(")\">")
              .append(escXml(config.y2AxisLabel)).append("</text>\n");
        }
    }

    private boolean hasSecondaryY(List<SeriesData> data) {
        for (SeriesData s : data) if (s.yAxisIndex == 1) return true;
        return false;
    }

    private double[] rangeX(List<SeriesData> data) {
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (SeriesData s : data) {
            if (s.x == null) continue;
            for (int i = 0; i < s.x.length(); i++) { min = Math.min(min, s.x.get(i)); max = Math.max(max, s.x.get(i)); }
        }
        return padRange(min, max);
    }

    @SuppressWarnings("unchecked")
    private double[] rangeY(List<SeriesData> data, int yAxisIndex) {
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (SeriesData s : data) {
            if (s.yAxisIndex != yAxisIndex) continue;
            if (s.y != null) {
                for (int i = 0; i < s.y.length(); i++) {
                    min = Math.min(min, s.y.get(i));
                    max = Math.max(max, s.y.get(i));
                }
            }
            // 误差棒：Y 轴须包含 y±err，否则端帽会画出绘图区
            if (isErrorBarType(s.type) && s.y != null) {
                IVector<Double> err = errorVector(s);
                if (err != null) {
                    for (int i = 0; i < s.y.length() && i < err.length(); i++) {
                        double yi = s.y.get(i);
                        double e = err.get(i);
                        min = Math.min(min, yi - e);
                        max = Math.max(max, yi + e);
                    }
                }
            }
            // For ci_band, also consider yLow/yHigh extra data
            if ("ci_band".equals(s.type) && s.extraData != null) {
                IVector<Double> lo = ciLow(s);
                IVector<Double> hi = ciHigh(s);
                if (lo != null) for (int i = 0; i < lo.length(); i++) { min = Math.min(min, lo.get(i)); }
                if (hi != null) for (int i = 0; i < hi.length(); i++) { max = Math.max(max, hi.get(i)); }
            }
        }
        return padRange(min, max);
    }

    private double[] padRange(double min, double max) {
        if (min == Double.POSITIVE_INFINITY) return new double[]{0, 1};
        double margin = (max - min) * 0.1;
        if (margin == 0) margin = Math.abs(min) * 0.1;
        if (margin == 0) margin = 1;
        return new double[]{min - margin, max + margin};
    }

    private List<Integer> sortedByX(SeriesData s) {
        List<Integer> ord = new ArrayList<>();
        for (int i = 0; i < s.x.length(); i++) ord.add(i);
        ord.sort(Comparator.<Integer>comparingDouble(i -> s.x.get(i)).thenComparingInt(i -> i));
        return ord;
    }

    private void drawLine(StringBuilder sb, SeriesData s, ChartConfig config,
                          double xMin, double xMax, double yMin, double yMax,
                          PlotAxisScale yScale, String color) {
        List<Integer> ord = sortedByX(s);
        sb.append("<path d=\"");
        for (int k = 0; k < ord.size(); k++) {
            int i = ord.get(k);
            double px = dataToX(config, s.x.get(i), xMin, xMax);
            double py = dataToY(config, s.y.get(i), yMin, yMax);
            sb.append(k == 0 ? "M" : "L")
              .append(String.format("%.1f %.1f", px, py));
        }
        sb.append("\" stroke=\"").append(color).append("\" stroke-width=\"2\" fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n");
    }

    private void drawScatter(StringBuilder sb, SeriesData s, ChartConfig config,
                             double xMin, double xMax, double yMin, double yMax,
                             PlotAxisScale yScale, String color) {
        sb.append("<g fill=\"").append(color).append("\">\n");
        for (int i = 0; i < s.x.length(); i++) {
            double px = dataToX(config, s.x.get(i), xMin, xMax);
            double py = dataToY(config, s.y.get(i), yMin, yMax);
            sb.append("  <circle cx=\"").append((int)px).append("\" cy=\"").append((int)py).append("\" r=\"4\"/>\n");
        }
        sb.append("</g>\n");
    }

    @SuppressWarnings("unchecked")
    private void drawBubble(StringBuilder sb, SeriesData s, ChartConfig config,
                           double xMin, double xMax, double yMin, double yMax,
                           PlotAxisScale yScale, String color) {
        IVector<Double> sizes = (IVector<Double>) s.extraData.get("sizes");
        double maxSize = 1;
        if (sizes != null) {
            for (int i = 0; i < sizes.length(); i++) maxSize = Math.max(maxSize, sizes.get(i));
        }
        sb.append("<g fill=\"").append(color).append("\" opacity=\"0.7\">\n");
        for (int i = 0; i < s.x.length(); i++) {
            double px = dataToX(config, s.x.get(i), xMin, xMax);
            double py = dataToY(config, s.y.get(i), yMin, yMax);
            double r = sizes != null && i < sizes.length()
                ? Math.max(4, (sizes.get(i) / maxSize) * 30) : 8;
            sb.append("  <circle cx=\"").append((int)px).append("\" cy=\"").append((int)py)
              .append("\" r=\"").append((int)r).append("\"/>\n");
        }
        sb.append("</g>\n");
    }

    private void drawArea(StringBuilder sb, SeriesData s, ChartConfig config,
                           double xMin, double xMax, double yMin, double yMax,
                           PlotAxisScale yScale, String color) {
        List<Integer> ord = sortedByX(s);
        double baselineY = dataToY(config, Math.max(0, yMin), yMin, yMax);
        StringBuilder path = new StringBuilder();
        for (int k = 0; k < ord.size(); k++) {
            int i = ord.get(k);
            double px = dataToX(config, s.x.get(i), xMin, xMax);
            double py = dataToY(config, s.y.get(i), yMin, yMax);
            path.append(k == 0 ? "M" : "L").append(String.format("%.1f %.1f", px, py));
        }
        int last = ord.get(ord.size() - 1);
        int first = ord.get(0);
        path.append("L").append(String.format("%.1f %.1f", dataToX(config, s.x.get(last), xMin, xMax), baselineY));
        path.append("L").append(String.format("%.1f %.1f", dataToX(config, s.x.get(first), xMin, xMax), baselineY)).append(" Z");
        sb.append("<path d=\"").append(path).append("\" fill=\"").append(color).append("\" opacity=\"0.35\"/>\n");
        sb.append("<path d=\"").append(path).append("\" fill=\"none\" stroke=\"").append(color).append("\" stroke-width=\"2\"/>\n");
    }

    private void drawStep(StringBuilder sb, SeriesData s, ChartConfig config,
                           double xMin, double xMax, double yMin, double yMax,
                           PlotAxisScale yScale, String color) {
        List<Integer> ord = sortedByX(s);
        sb.append("<g stroke=\"").append(color).append("\" stroke-width=\"2\" fill=\"none\" stroke-linecap=\"round\">\n");
        for (int k = 0; k < ord.size(); k++) {
            int i = ord.get(k);
            double px = dataToX(config, s.x.get(i), xMin, xMax);
            double py = dataToY(config, s.y.get(i), yMin, yMax);
            if (k == 0) {
                sb.append("  <path d=\"M").append(String.format("%.1f %.1f", px, py)).append("\"/>\n");
            } else {
                int ip = ord.get(k - 1);
                double pxPrev = dataToX(config, s.x.get(ip), xMin, xMax);
                double pyPrev = dataToY(config, s.y.get(ip), yMin, yMax);
                sb.append("  <path d=\"M").append(String.format("%.1f %.1fH%.1fV%.1f", pxPrev, pyPrev, px, py)).append("\"/>\n");
            }
        }
        sb.append("</g>\n");
    }

    private void drawErrorBars(StringBuilder sb, SeriesData s, ChartConfig config,
                                double xMin, double xMax, double yMin, double yMax,
                                PlotAxisScale yScale, String color) {
        IVector<Double> err = errorVector(s);
        if (err == null) return;
        double cap = 5;
        for (int i = 0; i < s.x.length() && i < err.length(); i++) {
            double px = dataToX(config, s.x.get(i), xMin, xMax);
            double pyTop = dataToY(config, s.y.get(i) + err.get(i), yMin, yMax);
            double pyBot = dataToY(config, s.y.get(i) - err.get(i), yMin, yMax);
            sb.append("<g stroke=\"").append(color).append("\" stroke-width=\"1.25\">\n");
            sb.append("  <line x1=\"").append((int)px).append("\" y1=\"").append((int)pyTop)
              .append("\" x2=\"").append((int)px).append("\" y2=\"").append((int)pyBot).append("\"/>\n");
            sb.append("  <line x1=\"").append((int)(px-cap)).append("\" y1=\"").append((int)pyTop)
              .append("\" x2=\"").append((int)(px+cap)).append("\" y2=\"").append((int)pyTop).append("\"/>\n");
            sb.append("  <line x1=\"").append((int)(px-cap)).append("\" y1=\"").append((int)pyBot)
              .append("\" x2=\"").append((int)(px+cap)).append("\" y2=\"").append((int)pyBot).append("\"/>\n");
            sb.append("</g>\n");
        }
    }

    @SuppressWarnings("unchecked")
    private IVector<Double> errorVector(SeriesData s) {
        if (s.extraData == null) return null;
        Object o = s.extraData.get("yerr");
        if (o == null) o = s.extraData.get("yErr");
        return o instanceof IVector ? (IVector<Double>) o : null;
    }

    @SuppressWarnings("unchecked")
    private IVector<Double> ciLow(SeriesData s) {
        Object o = s.extraData != null ? s.extraData.get("yLow") : null;
        return o instanceof IVector ? (IVector<Double>) o : null;
    }

    @SuppressWarnings("unchecked")
    private IVector<Double> ciHigh(SeriesData s) {
        Object o = s.extraData != null ? s.extraData.get("yHigh") : null;
        return o instanceof IVector ? (IVector<Double>) o : null;
    }

    private void drawCiBand(StringBuilder sb, SeriesData s, ChartConfig config,
                             double xMin, double xMax, double yMin, double yMax,
                             PlotAxisScale yScale, String color) {
        IVector<Double> lo = ciLow(s), hi = ciHigh(s);
        if (lo == null || hi == null || s.x == null) return;
        List<Integer> ord = sortedByX(s);
        StringBuilder path = new StringBuilder();
        for (int k = 0; k < ord.size(); k++) {
            int i = ord.get(k);
            double px = dataToX(config, s.x.get(i), xMin, xMax);
            double pyHi = dataToY(config, hi.get(i), yMin, yMax);
            path.append(k == 0 ? "M" : "L").append(String.format("%.1f %.1f", px, pyHi));
        }
        for (int k = ord.size() - 1; k >= 0; k--) {
            int i = ord.get(k);
            double px = dataToX(config, s.x.get(i), xMin, xMax);
            double pyLo = dataToY(config, lo.get(i), yMin, yMax);
            path.append("L").append(String.format("%.1f %.1f", px, pyLo));
        }
        path.append(" Z");
        sb.append("<path d=\"").append(path).append("\" fill=\"").append(color).append("\" opacity=\"0.22\"/>\n");
    }
}
