package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

/**
 * 漏斗图SVG渲染器。
 */
public class SvgFunnelRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null) return;
        double cx = config.width / 2.0;
        double chartTop = config.paddingTop + 50;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom - 50;
        double maxWidth = (config.width - config.paddingLeft - config.paddingRight) * 0.8;

        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < series.y.length(); i++) maxValue = Math.max(maxValue, series.y.get(i));

        int n = series.y.length();
        double sectionH = chartHeight / n - 5;

        for (int i = 0; i < n; i++) {
            double val = series.y.get(i);
            double width = (val / maxValue) * maxWidth;
            double y = chartTop + (i / (double) n) * chartHeight;
            double x1 = cx - width / 2, x2 = cx + width / 2;
            double x1n = i + 1 < n ? cx - (series.y.get(i + 1) / maxValue) * maxWidth / 2 : x1;
            double x2n = i + 1 < n ? cx + (series.y.get(i + 1) / maxValue) * maxWidth / 2 : x2;

            String color = colorPalette[i % colorPalette.length];
            sb.append("<polygon points=\"");
            sb.append(String.format("%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f",
                x1, y, x2, y, x2n, y + sectionH, x1n, y + sectionH));
            sb.append("\" fill=\"").append(color).append("\" opacity=\"0.85\" stroke=\"white\" stroke-width=\"1\"/>\n");

            // 标签
            String label = series.labels != null && i < series.labels.size() ? series.labels.get(i) : String.format("%.0f", val);
            sb.append("<text x=\"").append((int)cx).append("\" y=\"").append((int)(y + sectionH / 2 + 4))
              .append("\" text-anchor=\"middle\" class=\"tick-label\" fill=\"white\" font-size=\"11\">")
              .append(escXml(label)).append("</text>\n");
        }
    }
}
