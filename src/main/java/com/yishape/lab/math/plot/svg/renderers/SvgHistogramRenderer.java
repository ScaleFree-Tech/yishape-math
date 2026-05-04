package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.Locale;

/**
 * 直方图SVG渲染器（支持KDE拟合曲线）。
 */
public class SvgHistogramRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                    ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null) return;
        IVector<Double> yData = series.y;

        // ── 数据范围 & bins ─────────────────────────────────
        int bins = Math.min(20, (int) Math.sqrt(yData.length()));
        Integer binsOverride = series.extraData != null
            ? (Integer) series.extraData.get("bins") : null;
        if (binsOverride != null && binsOverride > 0) bins = Math.min(binsOverride, 500);
        double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
        for (int i = 0; i < yData.length(); i++) {
            double v = yData.get(i);
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double binWidth = (max - min) / bins;

        int[] counts = new int[bins];
        for (int i = 0; i < yData.length(); i++) {
            double value = yData.get(i);
            int binIndex = (int) Math.min((value - min) / binWidth, bins - 1);
            counts[binIndex]++;
        }
        double maxCount = 0;
        for (int c : counts) if (c > maxCount) maxCount = c;

        // ── 布局参数 ────────────────────────────────────────
        double cLeft = config.paddingLeft;
        double cRight = config.width - config.paddingRight;
        double cTop = config.paddingTop;
        double cBottom = config.height - config.paddingBottom;
        double chartWidth = cRight - cLeft;
        double chartHeight = cBottom - cTop;
        double barWidth = chartWidth / bins * 0.95;

        // ── 坐标轴 ─────────────────────────────────────────
        drawAxes(sb, config, 0, 1, 0, maxCount * 1.05, config.xlabel, config.ylabel);

        // ── 颜色 ───────────────────────────────────────────
        String primary = colorPalette != null && colorPalette.length > 0 ? colorPalette[0] : "#5470c6";

        // ── 柱状图 ──────────────────────────────────────────
        sb.append("<g fill=\"").append(primary).append("\" stroke=\"")
          .append(textColor).append("\" stroke-width=\"0.8\">\n");
        for (int i = 0; i < bins; i++) {
            double barHeight = (counts[i] / maxCount) * chartHeight * 0.9;
            double x = cLeft + i * (chartWidth / bins);
            double y = cBottom - barHeight;
            sb.append("  <rect x=\"").append((int) x).append("\" y=\"").append((int) y)
              .append("\" width=\"").append((int) barWidth).append("\" height=\"").append((int) barHeight).append("\"/>\n");
        }
        sb.append("</g>\n");

        // ── KDE拟合曲线 ─────────────────────────────────────
        Boolean fittingLine = series.extraData != null
            ? (Boolean) series.extraData.get("fittingLine") : null;
        if (fittingLine != null && fittingLine) {
            drawKdeCurve(sb, yData, min, max, config, chartHeight, maxCount);
        }
    }

    private void drawKdeCurve(StringBuilder sb, IVector<Double> data,
                              double min, double max,
                              ChartConfig config, double chartHeight, double maxCount) {
        int numPoints = 200;
        double std = calculateStd(data);
        double iqr = calculateIQR(data);
        double bandwidth = 0.9 * Math.min(std, iqr / 1.34) * Math.pow(data.length(), -0.2);
        if (bandwidth <= 0 || Double.isNaN(bandwidth)) bandwidth = (max - min) / 20;

        double[] kdeValues = new double[numPoints];
        double maxKde = 0;
        for (int i = 0; i < numPoints; i++) {
            double x = min + (max - min) * i / (numPoints - 1);
            double kde = 0;
            for (int j = 0; j < data.length(); j++) {
                double u = (x - data.get(j)) / bandwidth;
                kde += Math.exp(-0.5 * u * u);
            }
            kde /= (data.length() * bandwidth * Math.sqrt(2 * Math.PI));
            kdeValues[i] = kde;
            if (kde > maxKde) maxKde = kde;
        }

        double baselineY = config.height - config.paddingBottom;
        double targetHeight = chartHeight * 0.54;
        double scaleFactor = maxKde > 0 ? targetHeight / maxKde : 0;
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;

        // 面积填充
        String accent = colorPalette != null && colorPalette.length > 1
            ? colorPalette[1] : "#dd8452";
        sb.append("<polygon fill=\"").append(accent).append("\" opacity=\"0.24\" points=\"");
        for (int i = 0; i < numPoints; i++) {
            double x = min + (max - min) * i / (numPoints - 1);
            double px = config.paddingLeft + (x - min) / (max - min) * chartWidth;
            double py = baselineY - kdeValues[i] * scaleFactor;
            if (i == 0) sb.append(String.format(Locale.US, "%.1f,%.1f", px, baselineY));
            sb.append(String.format(Locale.US, " %.1f,%.1f", px, py));
        }
        sb.append(String.format(Locale.US, " %.1f,%.1f",
            config.paddingLeft + chartWidth, baselineY));
        sb.append("\"/>\n");

        // 描边线
        sb.append("<polyline fill=\"none\" stroke=\"").append(accent)
          .append("\" stroke-width=\"2.2\" points=\"");
        for (int i = 0; i < numPoints; i++) {
            double x = min + (max - min) * i / (numPoints - 1);
            double px = config.paddingLeft + (x - min) / (max - min) * chartWidth;
            double py = baselineY - kdeValues[i] * scaleFactor;
            if (i > 0) sb.append(" ");
            sb.append(String.format(Locale.US, "%.1f,%.1f", px, py));
        }
        sb.append("\"/>\n");
    }

    private double calculateStd(IVector<Double> data) {
        double mean = 0;
        for (int i = 0; i < data.length(); i++) mean += data.get(i);
        mean /= data.length();
        double variance = 0;
        for (int i = 0; i < data.length(); i++) {
            double d = data.get(i) - mean;
            variance += d * d;
        }
        variance /= data.length();
        return Math.sqrt(variance);
    }

    private double calculateIQR(IVector<Double> data) {
        double[] sorted = new double[data.length()];
        for (int i = 0; i < data.length(); i++) sorted[i] = data.get(i);
        java.util.Arrays.sort(sorted);
        return quantileLinear(sorted, 0.75) - quantileLinear(sorted, 0.25);
    }
}
