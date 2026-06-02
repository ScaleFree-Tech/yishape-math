package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.List;
import java.util.Map;

/**
 * 主题河流图SVG渲染器（堆叠面积图的时间演变）。
 */
public class SvgThemeRiverRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> data = series.extraData != null
            ? (List<Map<String, Object>>) series.extraData.get("riverData") : null;
        if (data == null || data.isEmpty()) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("主题河流图需要 extraData.riverData").append("</text>\n");
            return;
        }

        double margin = 60;
        double chartWidth = config.width - margin * 2;
        double chartHeight = config.height - margin * 2;
        double startX = margin, startY = margin;

        int n = data.size();
        // 假设每个Map的key是主题名，value是数值
        // 找出所有主题名和时间点数
        java.util.Set<String> themes = new java.util.LinkedHashSet<>();
        for (Map<String, Object> row : data) {
            for (String key : row.keySet()) {
                if (!key.equals("time") && !key.equals("date") && !key.equals("x")) {
                    themes.add(key);
                }
            }
        }
        List<String> themeList = new java.util.ArrayList<>(themes);
        int tCount = themeList.size();

        if (n < 2 || tCount < 1) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("数据不足").append("</text>\n");
            return;
        }

        // 收集每个主题的时间序列
        double[] xVals = new double[n];
        double[][] yVals = new double[tCount][n];
        double[] sums = new double[n];
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = data.get(i);
            xVals[i] = i;
            sums[i] = 0;
            for (int t = 0; t < tCount; t++) {
                Object v = row.get(themeList.get(t));
                double dv = v instanceof Number ? ((Number) v).doubleValue() : 0;
                yVals[t][i] = dv;
                sums[i] += dv;
            }
        }

        // 绘制每个主题层（堆叠面积）
        double[] yBase = new double[n]; // 累积基线
        for (int i = 0; i < n; i++) yBase[i] = 0;
        double yMax = 0;
        for (int i = 0; i < n; i++) yMax = Math.max(yMax, sums[i]);

        double chartBottom = startY + chartHeight;
        for (int t = 0; t < tCount; t++) {
            double[] yTop = new double[n];
            for (int i = 0; i < n; i++) {
                yTop[i] = yBase[i] + (sums[i] > 0 ? yVals[t][i] / sums[i] * chartHeight : 0);
            }

            sb.append("<polygon points=\"");
            // 底边从前到后
            for (int i = 0; i < n; i++) {
                double x = startX + xVals[i] / Math.max(1, n - 1) * chartWidth;
                double y = chartBottom - yBase[i] / Math.max(1, yMax) * chartHeight;
                if (i > 0) sb.append(" ");
                sb.append((int)x).append(",").append((int)y);
            }
            // 顶边从后到前
            for (int i = n - 1; i >= 0; i--) {
                double x = startX + xVals[i] / Math.max(1, n - 1) * chartWidth;
                double y = chartBottom - yTop[i] / Math.max(1, yMax) * chartHeight;
                sb.append(" ").append((int)x).append(",").append((int)y);
            }
            sb.append("\" fill=\"").append(colorPalette[t % colorPalette.length])
              .append("\" opacity=\"0.7\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");

            // 更新基线
            for (int i = 0; i < n; i++) yBase[i] = yTop[i];
        }

        // 绘制X轴
        sb.append("<line x1=\"").append((int)startX).append("\" y1=\"")
          .append((int)chartBottom).append("\" x2=\"").append((int)(startX + chartWidth))
          .append("\" y2=\"").append((int)chartBottom).append("\" stroke=\"")
          .append(axisColor).append("\" stroke-width=\"1\"/>\n");

        // 绘制时间标签
        for (int i = 0; i < n; i++) {
            if (n <= 10 || i % (n / 5 + 1) == 0) {
                double x = startX + xVals[i] / Math.max(1, n - 1) * chartWidth;
                sb.append("<text x=\"").append((int)x).append("\" y=\"")
                  .append((int)(chartBottom + 16)).append("\" text-anchor=\"middle\" class=\"tick-label\">")
                  .append("T").append(i + 1).append("</text>\n");
            }
        }

        // 绘制图例 (per-theme, not per-series)
        double legendX = startX + chartWidth - tCount * 80;
        double legendY = startY + 10;
        for (int t = 0; t < tCount; t++) {
            sb.append("<rect x=\"").append((int)legendX + t * 80).append("\" y=\"")
              .append((int)legendY).append("\" width=\"12\" height=\"12\" fill=\"")
              .append(colorPalette[t % colorPalette.length]).append("\"/>\n");
            sb.append("<text x=\"").append((int)legendX + t * 80 + 16).append("\" y=\"")
              .append((int)(legendY + 10)).append("\" class=\"tick-label\">")
              .append(escXml(truncate(themeList.get(t), 8))).append("</text>\n");
        }
    }

    private String truncate(String s, int max) {
        return s.length() <= max ? s : s.substring(0, Math.max(1, max - 1)) + "…";
    }
}
