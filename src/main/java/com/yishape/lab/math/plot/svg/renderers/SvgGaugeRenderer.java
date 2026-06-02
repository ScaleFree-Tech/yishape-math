package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

/**
 * 仪表盘SVG渲染器。
 */
public class SvgGaugeRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        double value = 0.0, min = 0.0, max = 100.0;
        if (series.extraData != null) {
            value = ((Number) series.extraData.getOrDefault("value", 0.0)).doubleValue();
            min = ((Number) series.extraData.getOrDefault("min", 0.0)).doubleValue();
            max = ((Number) series.extraData.getOrDefault("max", 100.0)).doubleValue();
        }
        double cx = config.width / 2.0;
        double cy = config.height / 2.0 + 50;
        double r = Math.min(config.width - 100, config.height - 150) / 2.0;
        double pct = (value - min) / (max - min);

        // 背景弧（使用sweep-flag=0，逆时针从180°到0°，走上方）
        sb.append("<path d=\"");
        sb.append(String.format("M %.1f %.1f A %.1f %.1f 0 0 1 %.1f %.1f",
            cx - r, cy, r, r, cx + r, cy));
        sb.append("\" fill=\"none\" stroke=\"#e5e7eb\" stroke-width=\"20\" stroke-linecap=\"round\"/>\n");

        // 值弧（也用sweep-flag=0，逆时针，large-arc-flag根据弧长是否超过90°来决定）
        double angleDeg = pct * 180;  // 弧长（度数）
        // 正确计算弧终点的角度：180° - angleDeg（逆时针从180°往0°方向走）
        double endAngle = 180 - angleDeg;
        double endX = cx + Math.cos(Math.toRadians(endAngle)) * r;
        double endY = cy - Math.sin(Math.toRadians(endAngle)) * r;
        // large-arc-flag=1表示取弧长>180°的路径（对于pct<0.5）或弧长<=180°的路径（对于pct>=0.5）
        // 由于我们总是从180°逆时针走，angleDeg<=180时总是走短路径，large-arc=0
        sb.append("<path d=\"");
        sb.append(String.format("M %.1f %.1f A %.1f %.1f 0 0 1 %.1f %.1f",
            cx - r, cy, r, r, endX, endY));
        sb.append("\" fill=\"none\" stroke=\"").append(colorPalette[0]).append("\" stroke-width=\"20\" stroke-linecap=\"round\"/>\n");

        // 刻度
        for (int i = 0; i <= 10; i++) {
            double a = Math.PI * i / 10;
            double tx = cx + Math.cos(Math.PI - a) * (r + 30);
            double ty = cy - Math.sin(a) * (r + 30);
            double tickVal = min + (max - min) * i / 10;
            sb.append("<text x=\"").append((int)tx).append("\" y=\"").append((int)ty)
              .append("\" text-anchor=\"middle\" class=\"tick-label\" dy=\".3em\">")
              .append(String.format("%.0f", tickVal)).append("</text>\n");
        }

        // 中心指针
        double needleAngle = Math.PI - pct * Math.PI;
        double nx = cx + Math.cos(needleAngle) * r * 0.8;
        double ny = cy - Math.sin(needleAngle) * r * 0.8;
        sb.append("<line x1=\"").append((int)cx).append("\" y1=\"").append((int)cy)
          .append("\" x2=\"").append((int)nx).append("\" y2=\"").append((int)ny)
          .append("\" stroke=\"#ef4444\" stroke-width=\"3\" stroke-linecap=\"round\"/>\n");
        sb.append("<circle cx=\"").append((int)cx).append("\" cy=\"").append((int)cy)
          .append("\" r=\"8\" fill=\"").append(colorPalette[0]).append("\"/>\n");

        // 数值显示
        sb.append("<text x=\"").append((int)cx).append("\" y=\"").append((int)(cy + 20))
          .append("\" text-anchor=\"middle\" class=\"title\" font-size=\"18\">")
          .append(String.format("%.1f", value)).append("</text>\n");
    }
}
