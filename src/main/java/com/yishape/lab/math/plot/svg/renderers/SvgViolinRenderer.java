package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.List;

/**
 * 小提琴图SVG渲染器（seaborn风格：KDE曲线+内部箱线图）。
 */
public class SvgViolinRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null) return;
        IVector<Double> yData = series.y;

        double[] sorted = new double[yData.length()];
        for (int i = 0; i < yData.length(); i++) sorted[i] = yData.get(i);
        java.util.Arrays.sort(sorted);

        double min = sorted[0], max = sorted[sorted.length - 1];
        double q1 = quantileLinear(sorted, 0.25), median = quantileLinear(sorted, 0.50);
        double q3 = quantileLinear(sorted, 0.75), iqr = q3 - q1;

        double std = calculateStd(sorted);
        double bandwidth = 1.06 * Math.min(std, iqr / 1.34) * Math.pow(series.y.length(), -0.2);
        if (bandwidth <= 0 || Double.isNaN(bandwidth)) bandwidth = (max - min) / 30;

        double kdeMin = min - 3 * bandwidth, kdeMax = max + 3 * bandwidth;
        int numPoints = 200;

        double[] kdeValues = new double[numPoints], py = new double[numPoints];
        double maxKde = 0;
        double normFactor = 1.0 / (Math.sqrt(2 * Math.PI) * sorted.length * bandwidth);
        for (int i = 0; i < numPoints; i++) {
            double y = kdeMin + (kdeMax - kdeMin) * i / (numPoints - 1);
            double kde = 0;
            for (double v : sorted) {
                double u = (y - v) / bandwidth;
                kde += Math.exp(-0.5 * u * u);
            }
            kde *= normFactor;
            kdeValues[i] = kde;
            maxKde = Math.max(maxKde, kde);
            py[i] = dataToY(config, y, kdeMin, kdeMax);
        }

        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double centerX = config.paddingLeft + chartWidth / 2;
        double maxViolinWidth = Math.min(50, chartWidth / 8);
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;

        // Y轴
        drawYAxisCartesianFrame(sb, config, kdeMin, kdeMax);

        // 小提琴填充
        String violinFill = hexAlpha(colorPalette[0], 0.6);
        sb.append("<path d=\"");
        sb.append(String.format("M %.1f %.1f", centerX, py[0]));
        for (int i = 1; i < numPoints; i++) {
            double w = (kdeValues[i] / maxKde) * maxViolinWidth;
            sb.append(String.format(" L %.1f %.1f", centerX + w, py[i]));
        }
        sb.append(String.format(" L %.1f %.1f", centerX, py[numPoints - 1]));
        for (int i = numPoints - 2; i >= 0; i--) {
            double w = (kdeValues[i] / maxKde) * maxViolinWidth;
            sb.append(String.format(" L %.1f %.1f", centerX - w, py[i]));
        }
        sb.append(" Z\" fill=\"").append(violinFill).append("\" stroke=\"")
          .append(colorPalette[0]).append("\" stroke-width=\"1.5\"/>\n");

        // 内部箱线图
        double boxW = maxViolinWidth * 0.15;
        double q1Y = dataToY(config, q1, kdeMin, kdeMax);
        double medY = dataToY(config, median, kdeMin, kdeMax);
        double q3Y = dataToY(config, q3, kdeMin, kdeMax);
        double lwrW = Math.max(min, q1 - 1.5 * iqr);
        double uprW = Math.min(max, q3 + 1.5 * iqr);
        double lwrY = dataToY(config, lwrW, kdeMin, kdeMax);
        double uprY = dataToY(config, uprW, kdeMin, kdeMax);

        sb.append("<rect x=\"").append((int)(centerX - boxW / 2)).append("\" y=\"").append((int)q3Y)
          .append("\" width=\"").append((int)boxW).append("\" height=\"").append((int)(q1Y - q3Y))
          .append("\" fill=\"white\" fill-opacity=\"0.9\" stroke=\"").append(textColor).append("\" stroke-width=\"1.2\"/>\n");
        sb.append("<line x1=\"").append((int)(centerX - boxW / 2 - 2)).append("\" y1=\"").append((int)medY)
          .append("\" x2=\"").append((int)(centerX + boxW / 2 + 2)).append("\" y2=\"").append((int)medY)
          .append("\" stroke=\"").append(textColor).append("\" stroke-width=\"2\"/>\n");

        // 须线
        double whiskerW = boxW * 0.6;
        sb.append("<line x1=\"").append((int)centerX).append("\" y1=\"").append((int)q3Y)
          .append("\" x2=\"").append((int)centerX).append("\" y2=\"").append((int)uprY)
          .append("\" stroke=\"").append(textColor).append("\" stroke-width=\"1\"/>\n");
        sb.append("<line x1=\"").append((int)(centerX - whiskerW / 2)).append("\" y1=\"").append((int)uprY)
          .append("\" x2=\"").append((int)(centerX + whiskerW / 2)).append("\" y2=\"").append((int)uprY)
          .append("\" stroke=\"").append(textColor).append("\" stroke-width=\"1\"/>\n");
        sb.append("<line x1=\"").append((int)centerX).append("\" y1=\"").append((int)q1Y)
          .append("\" x2=\"").append((int)centerX).append("\" y2=\"").append((int)lwrY)
          .append("\" stroke=\"").append(textColor).append("\" stroke-width=\"1\"/>\n");
        sb.append("<line x1=\"").append((int)(centerX - whiskerW / 2)).append("\" y1=\"").append((int)lwrY)
          .append("\" x2=\"").append((int)(centerX + whiskerW / 2)).append("\" y2=\"").append((int)lwrY)
          .append("\" stroke=\"").append(textColor).append("\" stroke-width=\"1\"/>\n");
        appendAxisTitleLabels(sb, config);
    }

    private double calculateStd(double[] data) {
        double mean = 0;
        for (double v : data) mean += v;
        mean /= data.length;
        double variance = 0;
        for (double v : data) { double d = v - mean; variance += d * d; }
        variance /= data.length;
        return Math.sqrt(variance);
    }

    /** 将 hex 色值加上透明度的 SVG 表示（返回 rgba 格式） */
    private String hexAlpha(String hex, double alpha) {
        if (hex == null) hex = "#5470c6";
        int r = Integer.parseInt(hex.substring(1, 3), 16);
        int g = Integer.parseInt(hex.substring(3, 5), 16);
        int b = Integer.parseInt(hex.substring(5, 7), 16);
        return String.format("rgba(%d,%d,%d,%.2f)", r, g, b, alpha);
    }
}
