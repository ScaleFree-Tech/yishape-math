package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.List;

/**
 * 饼图SVG渲染器（SVG扇形路径 + 图例）。
 */
public class SvgPieRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null) return;

        int n = series.y.length();
        double total = 0;
        for (int i = 0; i < n; i++) total += series.y.get(i);
        if (total <= 0) total = 1;

        double cx = config.width / 2.0;
        double cy = config.height / 2.0 + 20;
        double radius = Math.min(config.width - 200, config.height - 150) / 2.0;

        double startAngle = 0;
        for (int i = 0; i < n; i++) {
            double value = series.y.get(i);
            double angle = (value / total) * 360;
            double midAngle = Math.toRadians(startAngle + angle / 2);

            // SVG扇形路径
            double x1 = cx + Math.cos(Math.toRadians(startAngle)) * radius;
            double y1 = cy + Math.sin(Math.toRadians(startAngle)) * radius;
            double x2 = cx + Math.cos(Math.toRadians(startAngle + angle)) * radius;
            double y2 = cy + Math.sin(Math.toRadians(startAngle + angle)) * radius;
            int large = angle > 180 ? 1 : 0;

            String color = colorPalette[i % colorPalette.length];
            if (angle >= 360) {
                sb.append("<circle cx=\"").append((int)cx).append("\" cy=\"").append((int)cy)
                  .append("\" r=\"").append((int)radius).append("\" fill=\"").append(color).append("\"/>\n");
            } else {
                sb.append("<path d=\"M ").append((int)cx).append(" ").append((int)cy)
                  .append(" L ").append((int)x1).append(" ").append((int)y1)
                  .append(" A ").append((int)radius).append(" ").append((int)radius)
                  .append(" 0 ").append(large).append(" 1 ")
                  .append((int)x2).append(" ").append((int)y2).append(" Z\" fill=\"")
                  .append(color).append("\" stroke=\"white\" stroke-width=\"2\"/>\n");
            }

            // 百分比标签
            if (angle > 10) {
                double labelR = radius + 30;
                double lx = cx + Math.cos(midAngle) * labelR;
                double ly = cy + Math.sin(midAngle) * labelR;
                sb.append("<text x=\"").append((int)lx).append("\" y=\"").append((int)ly)
                  .append("\" text-anchor=\"middle\" class=\"tick-label\" fill=\"")
                  .append(textColor).append("\">")
                  .append(String.format("%.1f%%", (value / total) * 100))
                  .append("</text>\n");
            }

            startAngle += angle;
        }

        // 图例
        double legendX = cx + radius + 60;
        double legendY = cy - n * 12;
        for (int i = 0; i < n; i++) {
            String label = series.labels != null && i < series.labels.size()
                ? series.labels.get(i) : "Item " + (i + 1);
            String color = colorPalette[i % colorPalette.length];
            sb.append("<rect x=\"").append((int)legendX).append("\" y=\"")
              .append((int)(legendY + i * 25)).append("\" width=\"15\" height=\"15\" fill=\"")
              .append(color).append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");
            sb.append("<text x=\"").append((int)(legendX + 25)).append("\" y=\"")
              .append((int)(legendY + i * 25 + 12)).append("\" class=\"tick-label\" fill=\"")
              .append(textColor).append("\">").append(escXml(label)).append("</text>\n");
        }
    }
}
