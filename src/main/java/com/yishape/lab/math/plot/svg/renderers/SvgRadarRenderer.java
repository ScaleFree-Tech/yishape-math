package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

/**
 * 雷达图SVG渲染器（网格 + 轴标签 + 多序列数据）。
 */
public class SvgRadarRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null) return;
        int n = series.y.length();
        if (n < 3) n = 3;

        double cx = config.width / 2.0;
        double cy = config.height / 2.0 + 20;
        double maxR = Math.min(config.width, config.height) / 2.0 - 120;
        double angleStep = 2 * Math.PI / n;

        // 最大值（用于归一化）
        double maxVal = 0;
        for (int i = 0; i < series.y.length(); i++) maxVal = Math.max(maxVal, series.y.get(i));
        if (maxVal <= 0) maxVal = 1;

        // 同心多边形网格
        for (int level = 1; level <= 5; level++) {
            double r = maxR * level / 5.0;
            sb.append("<polygon points=\"");
            for (int i = 0; i <= n; i++) {
                int idx = i % n;
                double angle = idx * angleStep - Math.PI / 2;
                double px = cx + Math.cos(angle) * r;
                double py = cy + Math.sin(angle) * r;
                if (i > 0) sb.append(" ");
                sb.append(String.format("%.1f,%.1f", px, py));
            }
            sb.append("\" fill=\"none\" stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\"/>\n");
            // 层级的数值标签
            double labelVal = maxVal * level / 5.0;
            sb.append("<text x=\"").append((int)(cx - 8)).append("\" y=\"").append((int)(cy - r + 4))
              .append("\" text-anchor=\"end\" class=\"tick-label\">").append(formatTickLabel(labelVal))
              .append("</text>\n");
        }

        // 指标轴（从中心到外沿的射线）
        for (int i = 0; i < n; i++) {
            double angle = i * angleStep - Math.PI / 2;
            double ex = cx + Math.cos(angle) * maxR;
            double ey = cy + Math.sin(angle) * maxR;
            sb.append("<line x1=\"").append((int)cx).append("\" y1=\"").append((int)cy)
              .append("\" x2=\"").append((int)ex).append("\" y2=\"").append((int)ey)
              .append("\" stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        // 绘制每个序列的数据
        java.util.List<SeriesData> allSeries = new java.util.ArrayList<>();
        allSeries.add(series);

        // 多序列支持（从renderMultiSeries或单序列进入）
        renderRadarSeries(sb, series, cx, cy, maxR, angleStep, n, maxVal, colorPalette[0]);

        // 绘制轴标签（维度名称）
        java.util.List<String> labels = series.labels;
        for (int i = 0; i < n; i++) {
            double angle = i * angleStep - Math.PI / 2;
            double labelR = maxR + 20;
            double lx = cx + Math.cos(angle) * labelR;
            double ly = cy + Math.sin(angle) * labelR;
            String label = (labels != null && i < labels.size()) ? labels.get(i) : ("D" + (i + 1));
            String anchor = "middle";
            if (Math.abs(lx - cx) > 10) {
                anchor = lx > cx ? "start" : "end";
            }
            sb.append("<text x=\"").append((int)lx).append("\" y=\"").append((int)ly)
              .append("\" text-anchor=\"").append(anchor).append("\" class=\"axis-label\">")
              .append(escXml(label)).append("</text>\n");
        }
    }

    private void renderRadarSeries(StringBuilder sb, SeriesData s,
                                   double cx, double cy, double maxR,
                                   double angleStep, int n, double maxVal, String color) {
        if (s.y == null) return;
        int len = Math.min(s.y.length(), n);

        // 数据多边形（闭合）
        sb.append("<polygon points=\"");
        for (int i = 0; i <= len; i++) {
            int idx = i % len;
            double angle = idx * angleStep - Math.PI / 2;
            double val = s.y.get(idx);
            double r = maxVal > 0 ? (val / maxVal) * maxR : 0;
            double px = cx + Math.cos(angle) * r;
            double py = cy + Math.sin(angle) * r;
            if (i > 0) sb.append(" ");
            sb.append(String.format("%.1f,%.1f", px, py));
        }
        sb.append("\" fill=\"").append(color).append("\" opacity=\"0.3\" stroke=\"")
          .append(color).append("\" stroke-width=\"2\"/>\n");

        // 数据点
        for (int i = 0; i < len; i++) {
            double angle = i * angleStep - Math.PI / 2;
            double val = s.y.get(i);
            double r = maxVal > 0 ? (val / maxVal) * maxR : 0;
            double px = cx + Math.cos(angle) * r;
            double py = cy + Math.sin(angle) * r;
            sb.append("<circle cx=\"").append((int)px).append("\" cy=\"").append((int)py)
              .append("\" r=\"3\" fill=\"").append(color).append("\"/>\n");
        }
    }
}
