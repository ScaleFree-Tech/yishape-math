package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.List;

/**
 * 热力图SVG渲染器（冷暖色渐变 + 颜色条）。
 */
public class SvgHeatmapRenderer extends AbstractSvgChartRenderer {

    private static final double COLORBAR_MIN_WIDTH = 16;
    private static final double COLORBAR_MAX_WIDTH = 26;
    private static final double CELL_GAP = 1;

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        IMatrix<Double> matrix = (IMatrix<Double>) series.extraData.get("matrixData");
        if (matrix == null) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("热力图需要 extraData.matrixData").append("</text>\n");
            return;
        }

        List<String> xLabels = (List<String>) series.extraData.get("xLabels");
        List<String> yLabels = (List<String>) series.extraData.get("yLabels");

        int rows = matrix.getRowNum();
        int cols = matrix.getColNum();

        double minValue = Double.MAX_VALUE;
        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v = matrix.get(i, j);
                minValue = Math.min(minValue, v);
                maxValue = Math.max(maxValue, v);
            }
        }

        double colorBarWidth = Math.min(COLORBAR_MAX_WIDTH,
            Math.max(COLORBAR_MIN_WIDTH, (config.width - config.paddingLeft - config.paddingRight) / 25.0));
        double chartWidth = config.width - config.paddingLeft - config.paddingRight - colorBarWidth - 10;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;
        double cellWidth = chartWidth / cols;
        double cellHeight = chartHeight / rows;

        double valueRange = maxValue - minValue;

        // 绘制热力图单元格
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = matrix.get(i, j);
                double normalized = valueRange > 0 ? (value - minValue) / valueRange : 0.5;
                String cellColor = interpolateColor(normalized);

                double x = config.paddingLeft + j * cellWidth;
                double y = config.paddingTop + i * cellHeight;

                sb.append("<rect x=\"").append((int)x).append("\" y=\"").append((int)y)
                  .append("\" width=\"").append((int)(cellWidth - CELL_GAP)).append("\" height=\"")
                  .append((int)(cellHeight - CELL_GAP)).append("\" fill=\"")
                  .append(cellColor).append("\"/>\n");
            }
        }

        // 绘制颜色条
        double cbX = config.paddingLeft + chartWidth + 10;
        double cbHeight = chartHeight;
        for (int i = 0; i < 50; i++) {
            double t = i / 49.0;
            double norm = 1.0 - t;
            String c = interpolateColor(norm);
            double y = config.paddingTop + t * cbHeight;
            double h = cbHeight / 50;
            sb.append("<rect x=\"").append((int)cbX).append("\" y=\"").append((int)y)
              .append("\" width=\"").append((int)colorBarWidth).append("\" height=\"")
              .append((int)h + 1).append("\" fill=\"").append(c).append("\"/>\n");
        }
        // 颜色条边框
        sb.append("<rect x=\"").append((int)cbX).append("\" y=\"").append((int)config.paddingTop)
          .append("\" width=\"").append((int)colorBarWidth).append("\" height=\"")
          .append((int)cbHeight).append("\" fill=\"none\" stroke=\"")
          .append(axisColor).append("\" stroke-width=\"1\"/>\n");

        // 颜色条标签
        sb.append("<text x=\"").append((int)(cbX + colorBarWidth + 4)).append("\" y=\"")
          .append((int)(config.paddingTop + 4)).append("\" class=\"tick-label\">")
          .append(formatTickLabel(maxValue)).append("</text>\n");
        sb.append("<text x=\"").append((int)(cbX + colorBarWidth + 4)).append("\" y=\"")
          .append((int)(config.paddingTop + cbHeight)).append("\" class=\"tick-label\">")
          .append(formatTickLabel(minValue)).append("</text>\n");
    }

    private String interpolateColor(double t) {
        // 冷色(#4d8f72) -> 暖色(#c45b5b) 插值
        double r1 = 0.302, g1 = 0.561, b1 = 0.447; // #4d8f72 cold
        double r2 = 0.769, g2 = 0.357, b2 = 0.357; // #c45b5b warm
        double r = r1 + (r2 - r1) * t;
        double g = g1 + (g2 - g1) * t;
        double b = b1 + (b2 - b1) * t;
        int ri = (int) Math.round(r * 255);
        int gi = (int) Math.round(g * 255);
        int bi = (int) Math.round(b * 255);
        return String.format("#%02x%02x%02x", ri, gi, bi);
    }
}
