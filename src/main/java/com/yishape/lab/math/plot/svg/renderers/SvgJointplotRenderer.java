package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

/**
 * Jointplot SVG renderer - scatter plot with marginal histograms.
 */
public class SvgJointplotRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        double[] xData = series.extraData != null
            ? (double[]) series.extraData.get("xData") : null;
        double[] yData = series.extraData != null
            ? (double[]) series.extraData.get("yData") : null;
        if (xData == null || yData == null) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("Jointplot requires x and y data").append("</text>\n");
            return;
        }

        double marginTop = config.paddingTop;
        double marginBottom = config.paddingBottom;
        double marginLeft = config.paddingLeft;
        double marginRight = config.paddingRight;
        double marginalSize = Math.min(config.width, config.height) / 5;
        marginalSize = Math.max(marginalSize, 60);

        double scatterW = config.width - marginLeft - marginRight - marginalSize;
        double scatterH = config.height - marginTop - marginBottom - marginalSize;
        double scatterX = marginLeft;
        double scatterY = marginTop;

        double xMin = min(xData), xMax = max(xData);
        double yMin = min(yData), yMax = max(yData);
        double xRange = xMax - xMin, yRange = yMax - yMin;
        if (xRange == 0) xRange = 1;
        if (yRange == 0) yRange = 1;

        double[] xp = padRange(xMin, xMax);
        double[] yp = padRange(yMin, yMax);
        double xMinP = xp[0], xMaxP = xp[1], yMinP = yp[0], yMaxP = yp[1];
        double xRangeP = xMaxP - xMinP;
        double yRangeP = yMaxP - yMinP;
        if (xRangeP == 0) xRangeP = 1;
        if (yRangeP == 0) yRangeP = 1;

        double pad = 5;

        // Draw scatter points（与轴域一致）
        for (int i = 0; i < xData.length; i++) {
            double px = scatterX + pad + (xData[i] - xMinP) / xRangeP * (scatterW - 2 * pad);
            double py = scatterY + scatterH - pad - (yData[i] - yMinP) / yRangeP * (scatterH - 2 * pad);
            sb.append("<circle cx=\"").append((int)px).append("\" cy=\"").append((int)py)
              .append("\" r=\"3\" fill=\"").append(colorPalette[0]).append("\" opacity=\"0.5\"/>\n");
        }

        drawInnerBox(sb, scatterX, scatterY, scatterX + scatterW, scatterY + scatterH,
            xMinP, xMaxP, yMinP, yMaxP, 4, 4, true);

        // Draw marginal on bottom (histogram - bars grow UPWARD from baseline)
        int bins = Math.min(20, xData.length / 3);
        bins = Math.max(bins, 2);
        double binW = scatterW / bins;
        double[] xCounts = new double[bins];
        for (double v : xData) {
            double binIdx = Math.floor((v - xMinP) / xRangeP * bins);
            int bin = (int) binIdx;
            if (bin < 0) bin = 0;
            if (bin >= bins) bin = bins - 1;
            xCounts[bin]++;
        }
        double xMaxCount = 0;
        for (double c : xCounts) xMaxCount = Math.max(xMaxCount, c);

        double margBarH = marginalSize - 10;
        double histBaselineY = scatterY + scatterH + 5;
        for (int b = 0; b < bins; b++) {
            double barHeight = (xMaxCount > 0) ? xCounts[b] / xMaxCount * margBarH : 0;
            double bx = scatterX + b * binW;
            double by = histBaselineY;
            sb.append("<rect x=\"").append((int)bx).append("\" y=\"").append((int)by)
              .append("\" width=\"").append((int)(binW - 1)).append("\" height=\"").append((int)barHeight)
              .append("\" fill=\"").append(colorPalette[0]).append("\" opacity=\"0.6\"/>\n");
        }

        // Draw marginal on right (histogram - bars grow RIGHTWARD from baseline)
        int binsY = Math.min(20, yData.length / 3);
        binsY = Math.max(binsY, 2);
        double binH = scatterH / binsY;
        double[] yCounts = new double[binsY];
        for (double v : yData) {
            double binIdx = Math.floor((v - yMinP) / yRangeP * binsY);
            int bin = (int) binIdx;
            if (bin < 0) bin = 0;
            if (bin >= binsY) bin = binsY - 1;
            yCounts[bin]++;
        }
        double yMaxCount = 0;
        for (double c : yCounts) yMaxCount = Math.max(yMaxCount, c);

        double margBarW = marginalSize - 10;
        double histBaselineX = scatterX + scatterW + 5;
        for (int b = 0; b < binsY; b++) {
            double barWidth = (yMaxCount > 0) ? yCounts[b] / yMaxCount * margBarW : 0;
            double bx = histBaselineX;
            double by = scatterY + scatterH - (b + 1) * binH;
            sb.append("<rect x=\"").append((int)bx).append("\" y=\"").append((int)by)
              .append("\" width=\"").append((int)barWidth).append("\" height=\"")
              .append((int)(binH - 1)).append("\" fill=\"").append(colorPalette[0]).append("\" opacity=\"0.6\"/>\n");
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

    private static double[] padRange(double min, double max) {
        if (!Double.isFinite(min) || !Double.isFinite(max)) {
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
}
