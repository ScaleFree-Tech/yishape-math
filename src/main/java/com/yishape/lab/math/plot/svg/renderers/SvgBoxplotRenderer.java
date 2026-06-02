package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

/**
 * 箱线图SVG渲染器：seaborn风格（1.5×IQR须、whisker caps、离群点）。
 */
public class SvgBoxplotRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                    ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null) return;
        IVector<Double> yData = series.y;

        // ── 数据预处理 ──────────────────────────────────────
        double[] sorted = new double[yData.length()];
        for (int i = 0; i < yData.length(); i++) sorted[i] = yData.get(i);
        java.util.Arrays.sort(sorted);

        double dataMin = sorted[0];
        double dataMax = sorted[sorted.length - 1];
        double q1 = quantileLinear(sorted, 0.25);
        double median = quantileLinear(sorted, 0.50);
        double q3 = quantileLinear(sorted, 0.75);
        double iqr = q3 - q1;

        // 1.5×IQR 须边界
        double whiskerLo = Math.max(dataMin, q1 - 1.5 * iqr);
        double whiskerHi = Math.min(dataMax, q3 + 1.5 * iqr);

        // 显示范围（须跨度 × padding）
        double whiskerRange = whiskerHi - whiskerLo;
        double displayMin = whiskerLo - whiskerRange * 0.08;
        double displayMax = whiskerHi + whiskerRange * 0.08;

        // ── 布局参数 ────────────────────────────────────────
        double cLeft = config.paddingLeft;
        double cRight = config.width - config.paddingRight;
        double cBottom = config.height - config.paddingBottom;
        double cTop = config.paddingTop;
        double chartWidth = cRight - cLeft;
        double chartHeight = cBottom - cTop;

        double boxWidth = Math.min(80, chartWidth / 3.5);
        double centerX = cLeft + chartWidth / 2.0;

        double boxTopY = dataToY(config, q3, displayMin, displayMax);
        double boxBotY = dataToY(config, q1, displayMin, displayMax);
        double medianY = dataToY(config, median, displayMin, displayMax);
        double whiskerTopY = dataToY(config, whiskerHi, displayMin, displayMax);
        double whiskerBotY = dataToY(config, whiskerLo, displayMin, displayMax);

        // ── 坐标轴 ─────────────────────────────────────────
        drawYAxisCartesianFrame(sb, config, displayMin, displayMax);

        // ── 颜色 ───────────────────────────────────────────
        String primary = colorPalette != null && colorPalette.length > 0 ? colorPalette[0] : "#5470c6";
        String boxFill = hexWithAlpha(primary, 0.28);
        String boxStroke = primary;

        // ── 绘制须线（垂直线从须端到箱体） ──────────────────
        sb.append("<g stroke=\"").append(boxStroke).append("\" stroke-width=\"1.5\" fill=\"none\">\n");
        // 上须
        sb.append("  <line x1=\"").append((int) centerX).append("\" y1=\"").append((int) whiskerTopY)
          .append("\" x2=\"").append((int) centerX).append("\" y2=\"").append((int) boxTopY).append("\"/>\n");
        // 下须
        sb.append("  <line x1=\"").append((int) centerX).append("\" y1=\"").append((int) boxBotY)
          .append("\" x2=\"").append((int) centerX).append("\" y2=\"").append((int) whiskerBotY).append("\"/>\n");
        sb.append("</g>\n");

        // ── Whisker caps（两端短横线） ──────────────────────
        double capSpan = boxWidth * 0.32;
        sb.append("<g stroke=\"").append(boxStroke).append("\" stroke-width=\"1.5\" fill=\"none\">\n");
        sb.append("  <line x1=\"").append((int) (centerX - capSpan)).append("\" y1=\"").append((int) whiskerTopY)
          .append("\" x2=\"").append((int) (centerX + capSpan)).append("\" y2=\"").append((int) whiskerTopY).append("\"/>\n");
        sb.append("  <line x1=\"").append((int) (centerX - capSpan)).append("\" y1=\"").append((int) whiskerBotY)
          .append("\" x2=\"").append((int) (centerX + capSpan)).append("\" y2=\"").append((int) whiskerBotY).append("\"/>\n");
        sb.append("</g>\n");

        // ── 箱体 ───────────────────────────────────────────
        double boxHeight = boxBotY - boxTopY;
        sb.append("<rect x=\"").append((int) (centerX - boxWidth / 2)).append("\" y=\"").append((int) boxTopY)
          .append("\" width=\"").append((int) boxWidth).append("\" height=\"").append((int) boxHeight)
          .append("\" fill=\"").append(boxFill).append("\" stroke=\"").append(boxStroke)
          .append("\" stroke-width=\"1.8\"/>\n");

        // ── 中位数线 ───────────────────────────────────────
        sb.append("<line x1=\"").append((int) (centerX - boxWidth / 2)).append("\" y1=\"").append((int) medianY)
          .append("\" x2=\"").append((int) (centerX + boxWidth / 2)).append("\" y2=\"").append((int) medianY)
          .append("\" stroke=\"").append(boxStroke).append("\" stroke-width=\"2\"/>\n");

        // ── 离群点 ─────────────────────────────────────────
        sb.append("<g fill=\"").append(primary).append("\" opacity=\"0.55\">\n");
        for (double v : sorted) {
            if (v < whiskerLo || v > whiskerHi) {
                double py = dataToY(config, v, displayMin, displayMax);
                sb.append("  <circle cx=\"").append((int) centerX).append("\" cy=\"").append((int) py)
                  .append("\" r=\"4\"/>\n");
            }
        }
        sb.append("</g>\n");

        // ── X轴标签 ─────────────────────────────────────────
        if (series.labels != null && !series.labels.isEmpty()) {
            sb.append("<text x=\"").append((int) centerX).append("\" y=\"")
              .append(config.height - config.paddingBottom + 34)
              .append("\" text-anchor=\"middle\" class=\"tick-label\">")
              .append(escXml(series.labels.get(0))).append("</text>\n");
        }
        appendAxisTitleLabels(sb, config);
    }

    private String hexWithAlpha(String hex, double alpha) {
        // SVG不支持透明hex，改用 rgba() 或在渲染时直接用带透明度的颜色
        // 这里直接返回填充色（透明度通过opacity属性实现）
        return hex;
    }
}
