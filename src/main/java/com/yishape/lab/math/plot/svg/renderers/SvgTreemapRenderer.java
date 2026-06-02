package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.*;

/**
 * 矩形树图SVG渲染器（Bruls Squarify 算法）。
 */
public class SvgTreemapRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> data = series.extraData != null
            ? (List<Map<String, Object>>) series.extraData.get("treeData") : null;
        if (data == null || data.isEmpty()) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("矩形树图需要 extraData.treeData=[{name,value}]").append("</text>\n");
            return;
        }

        double margin = 2;
        double W = config.width - config.paddingLeft - config.paddingRight - margin * 2;
        double H = config.height - config.paddingTop - config.paddingBottom - margin * 2;
        double x = config.paddingLeft + margin, y = config.paddingTop + margin;

        List<Map<String, Object>> sorted = new ArrayList<>(data);
        sorted.sort((a, b) -> {
            double va = ((Number) b.getOrDefault("value", 0)).doubleValue();
            double vb = ((Number) a.getOrDefault("value", 0)).doubleValue();
            return Double.compare(va, vb);
        });

        squarify(sb, sorted, x, y, W, H, 0);
    }

    private void squarify(StringBuilder sb, List<Map<String, Object>> items,
                           double x, double y, double W, double H, int depth) {
        if (items.isEmpty()) return;
        if (items.size() == 1) {
            Map<String, Object> it = items.get(0);
            String name = it.getOrDefault("name", "").toString();
            String color = colorPalette[depth % colorPalette.length];
            sb.append("<rect x=\"").append((int)x).append("\" y=\"").append((int)y)
              .append("\" width=\"").append((int)W).append("\" height=\"").append((int)H)
              .append("\" fill=\"").append(color).append("\" stroke=\"white\" stroke-width=\"1\"/>\n");
            if (W > 30 && H > 16) {
                sb.append("<text x=\"").append((int)(x + W / 2)).append("\" y=\"").append((int)(y + H / 2 + 4))
                  .append("\" text-anchor=\"middle\" fill=\"white\" font-size=\"11\">")
                  .append(escXml(name)).append("</text>\n");
            }
            return;
        }

        double total = 0;
        for (Map<String, Object> it : items) {
            total += ((Number) it.getOrDefault("value", 0)).doubleValue();
        }

        boolean layoutHorizontal = W >= H;
        double used = 0;
        List<Map<String, Object>> row = new ArrayList<>();
        double rowValue = 0;

        for (Map<String, Object> it : items) {
            double v = ((Number) it.getOrDefault("value", 0)).doubleValue();
            double newRowValue = rowValue + v;
            double currentRatio = worstAspectRatio(row, rowValue, layoutHorizontal ? H : W);
            List<Map<String, Object>> testRow = new ArrayList<>(row);
            testRow.add(it);
            double testRatio = worstAspectRatio(testRow, newRowValue, layoutHorizontal ? H : W);

            if (row.isEmpty() || testRatio <= currentRatio) {
                row.add(it);
                rowValue = newRowValue;
            } else {
                // 绘制当前行
                double rowSize = (rowValue / total) * (layoutHorizontal ? W : H);
                drawRow(sb, row, x, y, layoutHorizontal ? rowSize : W, layoutHorizontal ? H : rowSize, depth);
                used += rowSize;
                if (layoutHorizontal) x += rowSize;
                else y += rowSize;
                double remaining = (layoutHorizontal ? W : H) - used;
                row.clear(); rowValue = 0;
                if (remaining <= 0) break;
                row.add(it); rowValue = v;
            }
        }
        if (!row.isEmpty()) {
            double rowSize = (rowValue / total) * (layoutHorizontal ? W : H);
            drawRow(sb, row, x, y, layoutHorizontal ? rowSize : W, layoutHorizontal ? H : rowSize, depth);
        }
    }

    private double worstAspectRatio(List<Map<String, Object>> row, double sum, double side) {
        if (row.isEmpty() || side <= 0 || sum <= 0) return Double.MAX_VALUE;
        double area = sum;
        double rowWidth = area / side;
        double worst = 0;
        for (Map<String, Object> it : row) {
            double v = ((Number) it.getOrDefault("value", 0)).doubleValue();
            double h = area > 0 ? v / rowWidth : 0;
            if (h <= 0) continue;
            double ratio = Math.max(rowWidth / h, h / rowWidth);
            if (ratio > worst) worst = ratio;
        }
        return worst;
    }

    private void drawRow(StringBuilder sb, List<Map<String, Object>> row,
                         double x, double y, double W, double H, int depth) {
        double total = 0;
        for (Map<String, Object> it : row) total += ((Number) it.getOrDefault("value", 0)).doubleValue();
        double offset = 0;
        for (int i = 0; i < row.size(); i++) {
            Map<String, Object> it = row.get(i);
            double v = ((Number) it.getOrDefault("value", 0)).doubleValue();
            double size = total > 0 ? (v / total) * W : 0;
            String name = it.getOrDefault("name", "").toString();
            String color = colorPalette[(depth + i) % colorPalette.length];
            sb.append("<rect x=\"").append((int)x + (int)offset).append("\" y=\"").append((int)y)
              .append("\" width=\"").append((int)size).append("\" height=\"").append((int)H)
              .append("\" fill=\"").append(color).append("\" stroke=\"white\" stroke-width=\"1\"/>\n");
            if (size > 40 && H > 16) {
                sb.append("<text x=\"").append((int)(x + offset + size / 2)).append("\" y=\"")
                  .append((int)(y + H / 2 + 4)).append("\" text-anchor=\"middle\" fill=\"white\" font-size=\"10\">")
                  .append(escXml(name)).append("</text>\n");
            }
            offset += size;
        }
    }
}
