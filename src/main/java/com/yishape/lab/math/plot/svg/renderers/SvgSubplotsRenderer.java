package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.PlotStyle;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.List;

/**
 * 子图网格 SVG：普通笛卡尔格子、pairplot（p×p 与 JavaFx facet 同构）、jointplot（委托 {@link SvgJointplotRenderer}）。
 */
public class SvgSubplotsRenderer extends AbstractSvgChartRenderer {

    private static final int PAIR_LEFT_TICK_BAND = 34;
    private static final int PAIR_BOTTOM_TICK_BAND = 14;
    private static final int PAIR_TOP_PAD = 4;
    private static final int PAIR_RIGHT_PAD = 4;

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        String facetLayout = series.extraData != null
            ? (String) series.extraData.get("facetLayout") : null;
        if ("jointplot".equals(facetLayout)) {
            renderJointplotDelegate(sb, series, config, themeManager);
            return;
        }
        if ("pairplot".equals(facetLayout)) {
            renderPairplotFacetGrid(sb, series, config, themeManager);
            return;
        }
        renderCartesianFacetGrid(sb, series, config, themeManager);
    }

    private void renderJointplotDelegate(StringBuilder sb, SeriesData series, ChartConfig config,
                                        JavaFxThemeManager themeManager) {
        SeriesData syn = new SeriesData("_joint", null, null, new PlotStyle(), "jointplot");
        syn.extraData.put("xData", series.extraData.get("xData"));
        syn.extraData.put("yData", series.extraData.get("yData"));
        syn.extraData.put("marginal", series.extraData.getOrDefault("marginal", "hist"));
        SvgJointplotRenderer jr = new SvgJointplotRenderer();
        jr.applyTheme(themeManager);
        jr.renderSvgContent(sb, syn, config, themeManager);
    }

    @SuppressWarnings("unchecked")
    private void renderPairplotFacetGrid(StringBuilder sb, SeriesData series, ChartConfig config,
                                         JavaFxThemeManager themeManager) {
        List<SeriesData> subplotSeries = series.extraData != null
            ? (List<SeriesData>) series.extraData.get("subplotSeries") : null;
        List<String> columnNames = series.extraData != null
            ? (List<String>) series.extraData.get("columnNames") : null;
        int[] grid = series.extraData != null
            ? (int[]) series.extraData.get("grid") : new int[]{1, 1};

        int rows = grid[0], cols = grid[1];
        if (rows <= 0) rows = 1;
        if (cols <= 0) cols = 1;
        if (subplotSeries == null || subplotSeries.isEmpty()) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("pairplot: 无格子数据").append("</text>\n");
            return;
        }

        double marginTop = 72;
        double marginRight = 32;
        double marginBottom = 44;
        double marginLeft = 72;
        double labelArea = 50;
        double cellGap = 8;

        double availW = config.width - marginLeft - marginRight - labelArea;
        double availH = config.height - marginTop - marginBottom - labelArea;
        double cellW = (availW - cellGap * (cols - 1)) / cols;
        double cellH = (availH - cellGap * (rows - 1)) / rows;

        for (int j = 0; j < cols; j++) {
            double x = marginLeft + labelArea + j * cellW + cellW / 2;
            int labelY = (int) (marginTop - 6);
            String name = (columnNames != null && j < columnNames.size()) ? columnNames.get(j) : "Var" + j;
            sb.append("<text x=\"").append((int) x).append("\" y=\"").append(labelY)
              .append("\" text-anchor=\"middle\" class=\"tick-label\" transform=\"rotate(-30,")
              .append((int) x).append(",").append(labelY).append(")\">")
              .append(escXml(name)).append("</text>\n");
        }
        for (int i = 0; i < rows; i++) {
            double y = marginTop + labelArea + i * cellH + cellH / 2;
            int labelX = (int) (marginLeft - 6);
            String name = (columnNames != null && i < columnNames.size()) ? columnNames.get(i) : "Var" + i;
            sb.append("<text x=\"").append(labelX).append("\" y=\"").append((int) y)
              .append("\" text-anchor=\"end\" class=\"tick-label\" transform=\"rotate(-90,")
              .append(labelX).append(",").append((int) y).append(")\">")
              .append(escXml(name)).append("</text>\n");
        }

        int count = Math.min(subplotSeries.size(), rows * cols);
        for (int idx = 0; idx < count; idx++) {
            int row = idx / cols;
            int col = idx % cols;
            double x = marginLeft + labelArea + col * (cellW + cellGap);
            double y = marginTop + labelArea + row * (cellH + cellGap);
            double innerW = cellW - 4;
            double innerH = cellH - 4;
            SeriesData sub = subplotSeries.get(idx);
            String color = cellColor(sub, idx);
            if (sub == null || "empty".equals(sub.type)) {
                drawEmptyFacetCell(sb, x, y, innerW, innerH);
                continue;
            }
            String t = sub.type != null ? sub.type : "scatter";
            if ("histogram".equals(t)) {
                drawMiniHistogramFromY(sb, sub, x, y, innerW, innerH, color);
            } else if ("line".equals(t)) {
                drawMiniLine(sb, sub, x, y, innerW, innerH, color);
            } else if ("scatter".equals(t)) {
                drawMiniScatter(sb, sub, x, y, innerW, innerH, color);
            } else {
                drawEmptyFacetCell(sb, x, y, innerW, innerH);
            }
        }
    }

    private String cellColor(SeriesData sub, int idx) {
        if (sub.style != null && sub.style.getColor() != null && !sub.style.getColor().isEmpty()) {
            return sub.style.getColor();
        }
        return colorPalette[idx % colorPalette.length];
    }

    private void drawEmptyFacetCell(StringBuilder sb, double x, double y, double w, double h) {
        sb.append("<rect x=\"").append((int) x).append("\" y=\"").append((int) y)
          .append("\" width=\"").append((int) w).append("\" height=\"").append((int) h)
          .append("\" fill=\"none\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\" stroke-dasharray=\"3,2\"/>\n");
    }

    private void drawMiniHistogramFromY(StringBuilder sb, SeriesData sub,
                                        double x, double y, double w, double h, String color) {
        if (sub.y == null || sub.y.length() == 0) {
            drawEmptyFacetCell(sb, x, y, w, h);
            return;
        }
        double plotLeft = x + PAIR_LEFT_TICK_BAND;
        double plotRight = x + w - PAIR_RIGHT_PAD;
        double plotTop = y + PAIR_TOP_PAD;
        double plotBottom = y + h - PAIR_BOTTOM_TICK_BAND;
        double pw = Math.max(1, plotRight - plotLeft);
        double ph = Math.max(1, plotBottom - plotTop);

        int n = sub.y.length();
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = sub.y.get(i);
        }
        java.util.Arrays.sort(data);
        double dataMin = data[0];
        double dataMax = data[data.length - 1];
        double dataRange = dataMax - dataMin;
        if (dataRange <= 0) {
            dataRange = 1;
        }
        int bins = Math.min(20, Math.max(2, n / 3));
        double binW = pw / bins;
        double[] counts = new double[bins];
        for (double v : data) {
            int bin = (int) ((v - dataMin) / dataRange * bins);
            if (bin >= bins) {
                bin = bins - 1;
            }
            if (bin < 0) {
                bin = 0;
            }
            counts[bin]++;
        }
        double maxCount = 0;
        for (double c : counts) {
            maxCount = Math.max(maxCount, c);
        }
        if (maxCount <= 0) {
            maxCount = 1;
        }
        double barH = ph - 4;
        double padX = 2;
        for (int b = 0; b < bins; b++) {
            double barHeight = (maxCount > 0) ? counts[b] / maxCount * barH : 0;
            double bx = plotLeft + padX + b * binW;
            double by = plotBottom - barHeight;
            sb.append("<rect x=\"").append((int) bx).append("\" y=\"").append((int) by)
              .append("\" width=\"").append((int) (binW - 1)).append("\" height=\"").append((int) barHeight)
              .append("\" fill=\"").append(color).append("\" opacity=\"0.6\"/>\n");
        }
        sb.append("<line x1=\"").append((int) plotLeft).append("\" y1=\"").append((int) plotBottom)
          .append("\" x2=\"").append((int) plotRight).append("\" y2=\"").append((int) plotBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        sb.append("<line x1=\"").append((int) plotLeft).append("\" y1=\"").append((int) plotTop)
          .append("\" x2=\"").append((int) plotLeft).append("\" y2=\"").append((int) plotBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        drawInsetBottomXTicksInCell(sb, plotLeft, plotRight, plotBottom, y + h - 2, dataMin, dataMax, 3);
        drawLeftYTicksOnly(sb, plotLeft, plotTop, plotBottom, 0, maxCount, 2);
    }

    private void drawMiniLine(StringBuilder sb, SeriesData sub,
                              double x, double y, double w, double h, String color) {
        if (sub.x == null || sub.y == null) {
            drawEmptyFacetCell(sb, x, y, w, h);
            return;
        }
        int n = Math.min(sub.x.length(), sub.y.length());
        if (n < 2) {
            drawEmptyFacetCell(sb, x, y, w, h);
            return;
        }
        double xMin = Double.MAX_VALUE, xMax = -Double.MAX_VALUE;
        double yMin = Double.MAX_VALUE, yMax = -Double.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            double xv = sub.x.get(i);
            double yv = sub.y.get(i);
            xMin = Math.min(xMin, xv);
            xMax = Math.max(xMax, xv);
            yMin = Math.min(yMin, yv);
            yMax = Math.max(yMax, yv);
        }
        double xRange = xMax - xMin;
        double yRange = yMax - yMin;
        if (xRange == 0) {
            xRange = 1;
        }
        if (yRange == 0) {
            yRange = 1;
        }
        double plotLeft = x + PAIR_LEFT_TICK_BAND;
        double plotRight = x + w - PAIR_RIGHT_PAD;
        double plotTop = y + PAIR_TOP_PAD;
        double plotBottom = y + h - PAIR_BOTTOM_TICK_BAND;
        double pad = 4;
        double iw = Math.max(1e-9, plotRight - plotLeft - 2 * pad);
        double ih = Math.max(1e-9, plotBottom - plotTop - 2 * pad);
        StringBuilder path = new StringBuilder();
        for (int i = 0; i < n; i++) {
            double px = plotLeft + pad + (sub.x.get(i) - xMin) / xRange * iw;
            double py = plotBottom - pad - (sub.y.get(i) - yMin) / yRange * ih;
            path.append(i == 0 ? "M" : "L").append((int) px).append(",").append((int) py);
        }
        sb.append("<line x1=\"").append((int) plotLeft).append("\" y1=\"").append((int) plotBottom)
          .append("\" x2=\"").append((int) plotRight).append("\" y2=\"").append((int) plotBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        sb.append("<line x1=\"").append((int) plotLeft).append("\" y1=\"").append((int) plotTop)
          .append("\" x2=\"").append((int) plotLeft).append("\" y2=\"").append((int) plotBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        drawInsetBottomXTicksInCell(sb, plotLeft, plotRight, plotBottom, y + h - 2, xMin, xMax, 3);
        drawLeftYTicksOnly(sb, plotLeft, plotTop, plotBottom, yMin, yMax, 3);
        sb.append("<path d=\"").append(path).append("\" fill=\"none\" stroke=\"")
          .append(color).append("\" stroke-width=\"1.5\"/>\n");
    }

    private void drawMiniScatter(StringBuilder sb, SeriesData sub,
                                 double x, double y, double w, double h, String color) {
        if (sub.x == null || sub.y == null) {
            drawEmptyFacetCell(sb, x, y, w, h);
            return;
        }
        int n = Math.min(sub.x.length(), sub.y.length());
        if (n == 0) {
            drawEmptyFacetCell(sb, x, y, w, h);
            return;
        }
        double xMin = Double.MAX_VALUE, xMax = -Double.MAX_VALUE;
        double yMin = Double.MAX_VALUE, yMax = -Double.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            double xv = sub.x.get(i);
            double yv = sub.y.get(i);
            xMin = Math.min(xMin, xv);
            xMax = Math.max(xMax, xv);
            yMin = Math.min(yMin, yv);
            yMax = Math.max(yMax, yv);
        }
        double xRange = xMax - xMin;
        double yRange = yMax - yMin;
        if (xRange == 0) {
            xRange = 1;
        }
        if (yRange == 0) {
            yRange = 1;
        }
        double plotLeft = x + PAIR_LEFT_TICK_BAND;
        double plotRight = x + w - PAIR_RIGHT_PAD;
        double plotTop = y + PAIR_TOP_PAD;
        double plotBottom = y + h - PAIR_BOTTOM_TICK_BAND;
        double pad = 4;
        double iw = Math.max(1e-9, plotRight - plotLeft - 2 * pad);
        double ih = Math.max(1e-9, plotBottom - plotTop - 2 * pad);
        sb.append("<line x1=\"").append((int) plotLeft).append("\" y1=\"").append((int) plotBottom)
          .append("\" x2=\"").append((int) plotRight).append("\" y2=\"").append((int) plotBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        sb.append("<line x1=\"").append((int) plotLeft).append("\" y1=\"").append((int) plotTop)
          .append("\" x2=\"").append((int) plotLeft).append("\" y2=\"").append((int) plotBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        drawInsetBottomXTicksInCell(sb, plotLeft, plotRight, plotBottom, y + h - 2, xMin, xMax, 3);
        drawLeftYTicksOnly(sb, plotLeft, plotTop, plotBottom, yMin, yMax, 3);
        for (int i = 0; i < n; i++) {
            double px = plotLeft + pad + (sub.x.get(i) - xMin) / xRange * iw;
            double py = plotBottom - pad - (sub.y.get(i) - yMin) / yRange * ih;
            sb.append("<circle cx=\"").append((int) px).append("\" cy=\"").append((int) py)
              .append("\" r=\"2\" fill=\"").append(color).append("\" opacity=\"0.55\"/>\n");
        }
    }

    @SuppressWarnings("unchecked")
    private void renderCartesianFacetGrid(StringBuilder sb, SeriesData series, ChartConfig config,
                                          JavaFxThemeManager themeManager) {
        List<SeriesData> subplotSeries = series.extraData != null
            ? (List<SeriesData>) series.extraData.get("subplotSeries") : null;
        int[] grid = series.extraData != null
            ? (int[]) series.extraData.get("grid") : new int[]{1, 1};

        int rows = grid[0], cols = grid[1];
        if (rows <= 0) rows = 1;
        if (cols <= 0) cols = 1;

        if (subplotSeries == null || subplotSeries.isEmpty()) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("Subplots requires subplotSeries data").append("</text>\n");
            return;
        }

        double marginTop = 30;
        double marginRight = 20;
        double marginBottom = 40;
        double marginLeft = 50;
        double gap = 10;

        double totalW = config.width - marginLeft - marginRight;
        double totalH = config.height - marginTop - marginBottom;
        double cellW = (totalW - gap * (cols - 1)) / cols;
        double cellH = (totalH - gap * (rows - 1)) / rows;

        int count = Math.min(subplotSeries.size(), rows * cols);
        for (int idx = 0; idx < count; idx++) {
            SeriesData sub = subplotSeries.get(idx);
            int row = idx / cols;
            int col = idx % cols;
            double x = marginLeft + col * (cellW + gap);
            double y = marginTop + row * (cellH + gap);

            IVector<Double> xs = sub.x;
            IVector<Double> ys = sub.y;
            if (xs == null || ys == null) continue;

            int n = Math.min(xs.length(), ys.length());
            double xMin = Double.MAX_VALUE, xMax = -Double.MAX_VALUE;
            double yMin = Double.MAX_VALUE, yMax = -Double.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                double xv = xs.get(i);
                double yv = ys.get(i);
                xMin = Math.min(xMin, xv);
                xMax = Math.max(xMax, xv);
                yMin = Math.min(yMin, yv);
                yMax = Math.max(yMax, yv);
            }
            double xRange = xMax - xMin;
            double yRange = yMax - yMin;
            if (xRange == 0) xRange = 1;
            if (yRange == 0) yRange = 1;

            double pad = 5;
            String type = sub.type != null ? sub.type : "scatter";
            String color = colorPalette[idx % colorPalette.length];

            if ("line".equals(type) || "area".equals(type)) {
                StringBuilder path = new StringBuilder();
                for (int i = 0; i < n; i++) {
                    double px = x + pad + (xs.get(i) - xMin) / xRange * (cellW - 2 * pad);
                    double py = y + cellH - pad - (ys.get(i) - yMin) / yRange * (cellH - 2 * pad);
                    if (i == 0) path.append("M").append((int) px).append(",").append((int) py);
                    else path.append("L").append((int) px).append(",").append((int) py);
                }
                sb.append("<path d=\"").append(path).append("\" fill=\"none\" stroke=\"")
                  .append(color).append("\" stroke-width=\"1.5\"/>\n");
            } else {
                for (int i = 0; i < n; i++) {
                    double px = x + pad + (xs.get(i) - xMin) / xRange * (cellW - 2 * pad);
                    double py = y + cellH - pad - (ys.get(i) - yMin) / yRange * (cellH - 2 * pad);
                    sb.append("<circle cx=\"").append((int) px).append("\" cy=\"").append((int) py)
                      .append("\" r=\"2\" fill=\"").append(color).append("\" opacity=\"0.5\"/>\n");
                }
            }

            sb.append("<rect x=\"").append((int) x).append("\" y=\"").append((int) y)
              .append("\" width=\"").append((int) cellW).append("\" height=\"").append((int) cellH)
              .append("\" fill=\"none\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        }
    }
}
