package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.*;

/**
 * Pairplot SVG：p×p 格子与 {@link com.yishape.lab.math.plot.javafx.JavaFxPlot#renderMiniPane} 一致思路
 *（左/下内边距放刻度、无格子描边；JavaFx 仅 clip 不画矩形框）。
 */
public class SvgPairplotRenderer extends AbstractSvgChartRenderer {

    /** 左侧留给 Y 刻度（与 JavaFx mini pane 的 paddingLeft=min(44,cw/4) 同量级） */
    private static final int PAIR_LEFT_TICK_BAND = 34;
    /** 底部留给 X 刻度数字，避免与下方格子重叠 */
    private static final int PAIR_BOTTOM_TICK_BAND = 14;
    private static final int PAIR_TOP_PAD = 4;
    private static final int PAIR_RIGHT_PAD = 4;

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        IMatrix matrix = series.extraData != null
            ? (IMatrix) series.extraData.get("matrixData") : null;
        List<String> columnNames = series.extraData != null
            ? (List<String>) series.extraData.get("columnNames") : null;
        String diagonal = series.extraData != null
            ? (String) series.extraData.getOrDefault("diagonal", "kde") : "kde";

        if (matrix == null) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("Pairplot requires matrix data").append("</text>\n");
            return;
        }

        int cols = matrix.cols();

        // Use compact layout with margins for labels
        double marginTop = 60;
        double marginRight = 30;
        double marginBottom = 40;
        double marginLeft = 60;
        double labelArea = 50;
        double cellGap = 8; // spacing between cells

        double availW = config.width - marginLeft - marginRight - labelArea;
        double availH = config.height - marginTop - marginBottom - labelArea;
        double cellW = (availW - cellGap * (cols - 1)) / cols;
        double cellH = (availH - cellGap * (cols - 1)) / cols;

        // Draw column labels on top
        for (int j = 0; j < cols; j++) {
            double x = marginLeft + labelArea + j * cellW + cellW / 2;
            String name = (columnNames != null && j < columnNames.size()) ? columnNames.get(j) : "Var" + j;
            sb.append("<text x=\"").append((int)x).append("\" y=\"").append((int)(marginTop - 10))
              .append("\" text-anchor=\"middle\" class=\"tick-label\" transform=\"rotate(-30,")
              .append((int)x).append(",").append((int)(marginTop - 10)).append(")\">")
              .append(escXml(name)).append("</text>\n");
        }

        // Draw row labels on left
        for (int i = 0; i < cols; i++) {
            double y = marginTop + labelArea + i * cellH + cellH / 2;
            String name = (columnNames != null && i < columnNames.size()) ? columnNames.get(i) : "Var" + i;
            sb.append("<text x=\"").append((int)(marginLeft - 10)).append("\" y=\"").append((int)y)
              .append("\" text-anchor=\"middle\" class=\"tick-label\" transform=\"rotate(-90,")
              .append((int)(marginLeft - 10)).append(",").append((int)y).append(")\">")
              .append(escXml(name)).append("</text>\n");
        }

        // Draw each cell
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < cols; j++) {
                double cellX = marginLeft + labelArea + j * (cellW + cellGap);
                double cellY = marginTop + labelArea + i * (cellH + cellGap);
                double innerW = cellW - 4; // slight inner margin for content
                double innerH = cellH - 4;

                if (i == j) {
                    // Diagonal: histogram or KDE
                    drawDiagonalCell(sb, matrix, i, cellX, cellY, innerW, innerH, diagonal);
                } else {
                    // Off-diagonal: scatter plot
                    drawScatterCell(sb, matrix, i, j, cellX, cellY, innerW, innerH);
                }
            }
        }
    }

    private void drawDiagonalCell(StringBuilder sb, IMatrix matrix, int col,
                                   double x, double y, double w, double h,
                                   String type) {
        double plotLeft = x + PAIR_LEFT_TICK_BAND;
        double plotRight = x + w - PAIR_RIGHT_PAD;
        double plotTop = y + PAIR_TOP_PAD;
        double plotBottom = y + h - PAIR_BOTTOM_TICK_BAND;
        double pw = Math.max(1, plotRight - plotLeft);
        double ph = Math.max(1, plotBottom - plotTop);

        // Get column data
        double[] data = new double[matrix.rows()];
        for (int i = 0; i < matrix.rows(); i++) {
            Object val = matrix.get(i, col);
            data[i] = ((Number) val).doubleValue();
        }
        Arrays.sort(data);

        double dataMin = data[0];
        double dataMax = data[data.length - 1];
        double dataRange = dataMax - dataMin;
        if (dataRange <= 0) {
            dataRange = 1;
        }

        double yAxisMax = 1;
        if ("hist".equals(type)) {
            int bins = Math.max(2, Math.min(20, data.length / 3));
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
            yAxisMax = maxCount;

            double barH = ph - 4;
            double padX = 2;
            for (int b = 0; b < bins; b++) {
                double barHeight = (maxCount > 0) ? counts[b] / maxCount * barH : 0;
                double bx = plotLeft + padX + b * binW;
                double by = plotBottom - barHeight;
                sb.append("<rect x=\"").append((int) bx).append("\" y=\"").append((int) by)
                  .append("\" width=\"").append((int) (binW - 1)).append("\" height=\"").append((int) barHeight)
                  .append("\" fill=\"").append(colorPalette[0]).append("\" opacity=\"0.6\"/>\n");
            }
        } else {
            // KDE approximation
            int gp = 30;
            double bw = 1.06 * std(data) * Math.pow(data.length, -0.2);
            if (Double.isNaN(bw) || bw <= 0) {
                bw = (dataMax - dataMin) / 30;
            }
            double[] xGrid = new double[gp];
            double[] density = new double[gp];
            for (int k = 0; k < gp; k++) {
                xGrid[k] = dataMin - bw + (dataMax - dataMin + 2 * bw) * k / (gp - 1);
            }
            for (int k = 0; k < gp; k++) {
                double sum = 0;
                for (double v : data) {
                    sum += Math.exp(-0.5 * Math.pow((xGrid[k] - v) / bw, 2));
                }
                density[k] = sum / (data.length * bw * Math.sqrt(2 * Math.PI));
            }
            double maxD = 0;
            for (double d : density) {
                maxD = Math.max(maxD, d);
            }
            if (maxD <= 0) {
                maxD = 1;
            }
            yAxisMax = maxD;

            StringBuilder path = new StringBuilder();
            for (int k = 0; k < gp; k++) {
                double px = plotLeft + 2 + (xGrid[k] - (dataMin - bw)) / (dataMax - dataMin + 2 * bw) * (pw - 4);
                double py = plotBottom - (maxD > 0 ? density[k] / maxD * (ph - 4) : 0);
                if (k == 0) {
                    path.append("M").append((int) px).append(",").append((int) py);
                } else {
                    path.append("L").append((int) px).append(",").append((int) py);
                }
            }
            sb.append("<path d=\"").append(path).append("\" fill=\"none\" stroke=\"")
              .append(colorPalette[0]).append("\" stroke-width=\"1.5\"/>\n");
        }

        sb.append("<line x1=\"").append((int) plotLeft).append("\" y1=\"").append((int) plotBottom)
          .append("\" x2=\"").append((int) plotRight).append("\" y2=\"").append((int) plotBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        sb.append("<line x1=\"").append((int) plotLeft).append("\" y1=\"").append((int) plotTop)
          .append("\" x2=\"").append((int) plotLeft).append("\" y2=\"").append((int) plotBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        drawInsetBottomXTicksInCell(sb, plotLeft, plotRight, plotBottom, y + h - 2, dataMin, dataMax, 3);
        drawLeftYTicksOnly(sb, plotLeft, plotTop, plotBottom, 0, yAxisMax, 2);
    }

    private void drawScatterCell(StringBuilder sb, IMatrix matrix, int row, int col,
                                  double x, double y, double w, double h) {
        double[] xData = new double[matrix.rows()];
        double[] yData = new double[matrix.rows()];
        for (int i = 0; i < matrix.rows(); i++) {
            xData[i] = ((Number) matrix.get(i, col)).doubleValue();
            yData[i] = ((Number) matrix.get(i, row)).doubleValue();
        }

        double xMin = min(xData), xMax = max(xData);
        double yMin = min(yData), yMax = max(yData);
        double xRange = xMax - xMin, yRange = yMax - yMin;
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

        int numTicks = 3;
        drawInsetBottomXTicksInCell(sb, plotLeft, plotRight, plotBottom, y + h - 2, xMin, xMax, numTicks);
        drawLeftYTicksOnly(sb, plotLeft, plotTop, plotBottom, yMin, yMax, numTicks);

        for (int i = 0; i < xData.length; i++) {
            double px = plotLeft + pad + (xData[i] - xMin) / xRange * iw;
            double py = plotBottom - pad - (yData[i] - yMin) / yRange * ih;
            sb.append("<circle cx=\"").append((int) px).append("\" cy=\"").append((int) py)
              .append("\" r=\"2\" fill=\"").append(colorPalette[0]).append("\" opacity=\"0.5\"/>\n");
        }
    }

    private double min(double[] arr) {
        double m = arr[0];
        for (double v : arr) m = Math.min(m, v);
        return m;
    }

    private double max(double[] arr) {
        double m = arr[0];
        for (double v : arr) m = Math.max(m, v);
        return m;
    }

    private double std(double[] arr) {
        double m = 0;
        for (double v : arr) m += v;
        m /= arr.length;
        double s = 0;
        for (double v : arr) s += (v - m) * (v - m);
        return Math.sqrt(s / (arr.length - 1));
    }
}
