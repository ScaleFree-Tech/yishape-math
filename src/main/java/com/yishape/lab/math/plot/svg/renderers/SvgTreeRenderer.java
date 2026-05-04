package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.*;

/**
 * 树图SVG渲染器（水平层次布局，深度优先定位）。
 */
public class SvgTreeRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        Map<String, Object> tree = series.extraData != null
            ? (Map<String, Object>) series.extraData.get("treeData") : null;
        if (tree == null) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("树图需要 extraData.treeData（{name, children}）").append("</text>\n");
            return;
        }

        double margin = 80;
        double W = config.width - margin * 2;
        double H = config.height - margin * 2;

        // 计算树的深度
        int maxDepth = getTreeDepth(tree, 0);
        double nodeSpacingX = W / (maxDepth + 1);
        double nodeSpacingY = Math.min(80, H / 5);

        // 布局节点（深度优先）
        Map<String, double[]> pos = new HashMap<>();
        double rootY = margin + H / 2;
        layoutNode(tree, 0, rootY, nodeSpacingX, nodeSpacingY, pos);

        // 计算实际边界
        double minX = Double.MAX_VALUE, maxX = -Double.MAX_VALUE;
        double minY = Double.MAX_VALUE, maxY = -Double.MAX_VALUE;
        for (double[] p : pos.values()) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
        }
        maxY += 20;

        // 缩放以适应边界
        double scaleX = (maxX > margin + W) ? (margin + W) / maxX : 1.0;
        double scaleY = (maxY > margin + H) ? (margin + H) / maxY : 1.0;
        double scale = Math.min(scaleX, scaleY);

        // 应用缩放
        for (double[] p : pos.values()) {
            p[0] = margin + p[0] * scale;
            p[1] = margin + (p[1] - minY) * scale + 20;
        }

        // 绘制边
        drawEdges(sb, tree, pos);

        // 绘制节点
        for (Map.Entry<String, double[]> e : pos.entrySet()) {
            double[] p = e.getValue();
            String name = e.getKey();
            String color = colorPalette[0];
            sb.append("<circle cx=\"").append((int)p[0]).append("\" cy=\"").append((int)p[1])
              .append("\" r=\"8\" fill=\"").append(color).append("\"/>\n");
            sb.append("<text x=\"").append((int)(p[0] + 12)).append("\" y=\"").append((int)(p[1] + 4))
              .append("\" class=\"tick-label\">").append(escXml(name)).append("</text>\n");
        }
    }

    @SuppressWarnings("unchecked")
    private int getTreeDepth(Map<String, Object> node, int currentDepth) {
        int depth = currentDepth;
        if (node.containsKey("children")) {
            List<Map<String, Object>> children = (List<Map<String, Object>>) node.get("children");
            for (Map<String, Object> child : children) {
                depth = Math.max(depth, getTreeDepth(child, currentDepth + 1));
            }
        }
        return depth;
    }

    @SuppressWarnings("unchecked")
    private double[] layoutNode(Map<String, Object> node, double x, double y,
                                double dx, double dy, Map<String, double[]> pos) {
        String name = node.getOrDefault("name", "Node").toString();

        List<Map<String, Object>> children = null;
        if (node.containsKey("children")) {
            children = (List<Map<String, Object>>) node.get("children");
        }
        boolean hasChildren = children != null && !children.isEmpty();

        if (!hasChildren) {
            pos.put(name, new double[]{x, y});
            return new double[]{x, y};
        }

        // 递归布局子节点
        double totalChildHeight = dy * (children.size() - 1);
        double startY = y - totalChildHeight / 2;
        double minX = Double.MAX_VALUE, maxX = -Double.MAX_VALUE;

        for (int i = 0; i < children.size(); i++) {
            double childY = startY + i * dy;
            double childX = x + dx;
            double[] result = layoutNode(children.get(i), childX, childY, dx, dy, pos);
            minX = Math.min(minX, result[0]);
            maxX = Math.max(maxX, result[0]);
        }

        // 父节点位于子节点的中间位置
        pos.put(name, new double[]{x, y});
        return new double[]{minX, y};
    }

    @SuppressWarnings("unchecked")
    private void drawEdges(StringBuilder sb, Map<String, Object> node, Map<String, double[]> pos) {
        if (!node.containsKey("children")) return;
        String parentName = node.getOrDefault("name", "").toString();
        double[] pp = pos.get(parentName);
        List<Map<String, Object>> children = (List<Map<String, Object>>) node.get("children");
        for (Map<String, Object> child : children) {
            String childName = child.getOrDefault("name", "").toString();
            double[] cp = pos.get(childName);
            if (pp != null && cp != null) {
                sb.append("<line x1=\"").append((int)pp[0]).append("\" y1=\"").append((int)pp[1])
                  .append("\" x2=\"").append((int)cp[0]).append("\" y2=\"").append((int)cp[1])
                  .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1.5\"/>\n");
            }
            drawEdges(sb, child, pos);
        }
    }
}
