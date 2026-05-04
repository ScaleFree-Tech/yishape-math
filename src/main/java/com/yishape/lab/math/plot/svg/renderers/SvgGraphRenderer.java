package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.*;

/**
 * 关系图SVG渲染器（Fruchterman-Reingold 力导向布局）。
 * 节点从 extraData("nodes") 获取，链接从 extraData("links") 获取。
 */
public class SvgGraphRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nodes = series.extraData != null
            ? (List<Map<String, Object>>) series.extraData.get("nodes") : null;
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> links = series.extraData != null
            ? (List<Map<String, Object>>) series.extraData.get("links") : null;

        if (nodes == null || links == null || nodes.isEmpty()) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("关系图数据格式：nodes=[{name}], links=[{source,target}]").append("</text>\n");
            return;
        }

        int n = nodes.size();
        double margin = 60;
        double W = config.width - margin * 2;
        double H = config.height - margin * 2;

        // 节点初始位置（均匀分布）
        double[] px = new double[n], py = new double[n];
        for (int i = 0; i < n; i++) {
            px[i] = margin + Math.random() * W;
            py[i] = margin + Math.random() * H;
        }

        // Fruchterman-Reingold 力导向迭代
        double area = W * H;
        double k = Math.sqrt(area / n);
        double t = W * 0.1;
        for (int iter = 0; iter < 80; iter++) {
            double[] fx = new double[n], fy = new double[n];
            // 节点间斥力
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double dx = px[i] - px[j], dy = py[i] - py[j];
                    double dist = Math.sqrt(dx * dx + dy * dy) + 1e-6;
                    double f = k * k / dist;
                    fx[i] += dx / dist * f; fy[i] += dy / dist * f;
                    fx[j] -= dx / dist * f; fy[j] -= dy / dist * f;
                }
            }
            // 链接引力
            for (Map<String, Object> link : links) {
                int s = toIndex(link.get("source"));
                int t2 = toIndex(link.get("target"));
                if (s < 0 || s >= n || t2 < 0 || t2 >= n || s == t2) continue;
                double dx = px[s] - px[t2], dy = py[s] - py[t2];
                double dist = Math.sqrt(dx * dx + dy * dy) + 1e-6;
                double f = dist * dist / k;
                fx[s] -= dx / dist * f; fy[s] -= dy / dist * f;
                fx[t2] += dx / dist * f; fy[t2] += dy / dist * f;
            }
            // 位置更新
            t *= 0.9;
            for (int i = 0; i < n; i++) {
                double len = Math.sqrt(fx[i] * fx[i] + fy[i] * fy[i]) + 1e-6;
                px[i] += fx[i] / len * Math.min(t, len);
                py[i] += fy[i] / len * Math.min(t, len);
                px[i] = Math.max(margin, Math.min(config.width - margin, px[i]));
                py[i] = Math.max(margin, Math.min(config.height - margin, py[i]));
            }
        }

        // 绘制链接
        for (Map<String, Object> link : links) {
            int s = toIndex(link.get("source"));
            int t2 = toIndex(link.get("target"));
            if (s < 0 || s >= n || t2 < 0 || t2 >= n || s == t2) continue;
            String color = colorPalette[Math.abs(s) % colorPalette.length];
            sb.append("<line x1=\"").append((int)px[s]).append("\" y1=\"")
              .append((int)py[s]).append("\" x2=\"").append((int)px[t2])
              .append("\" y2=\"").append((int)py[t2])
              .append("\" stroke=\"").append(color).append("\" stroke-width=\"1.5\" opacity=\"0.5\"/>\n");
        }

        // 绘制节点
        for (int i = 0; i < n; i++) {
            String name = nodes.get(i).getOrDefault("name", "Node " + i).toString();
            String color = colorPalette[i % colorPalette.length];
            double nodeR = 12;
            sb.append("<circle cx=\"").append((int)px[i]).append("\" cy=\"").append((int)py[i])
              .append("\" r=\"").append((int)nodeR).append("\" fill=\"").append(color)
              .append("\" stroke=\"white\" stroke-width=\"2\"/>\n");
            sb.append("<text x=\"").append((int)(px[i] + nodeR + 4)).append("\" y=\"")
              .append((int)(py[i] + 4)).append("\" class=\"tick-label\">")
              .append(escXml(name)).append("</text>\n");
        }
    }

    private int toIndex(Object o) {
        if (o instanceof Number) return ((Number) o).intValue();
        if (o instanceof String) {
            try { return Integer.parseInt((String) o); } catch (NumberFormatException e) { }
        }
        return -1;
    }
}
