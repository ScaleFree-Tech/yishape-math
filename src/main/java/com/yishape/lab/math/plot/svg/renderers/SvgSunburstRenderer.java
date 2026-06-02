package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * 旭日图 SVG：按子树权重递归划分整圆心角，每层为同心圆环上的扇区（多边形近似，避免弧方向错误）。
 */
public class SvgSunburstRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        Map<String, Object> tree = series.extraData != null
            ? (Map<String, Object>) series.extraData.get("treeData") : null;
        if (tree == null) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("旭日图需要 extraData.treeData（层级数据）").append("</text>\n");
            return;
        }

        double cx = config.width / 2.0;
        double cy = config.height / 2.0;
        double maxR = Math.min(config.width, config.height) / 2.0 - 70;
        if (maxR < 40) {
            maxR = 40;
        }
        double holeR = Math.min(28, maxR * 0.12);

        int dMax = maxDepth(tree);
        if (dMax <= 0) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("旭日图需要含 children 的树").append("</text>\n");
            return;
        }

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> children = (List<Map<String, Object>>) tree.get("children");
        if (children == null || children.isEmpty()) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("旭日图根节点无 children").append("</text>\n");
            return;
        }

        double total = 0;
        for (Map<String, Object> c : children) {
            total += subtreeValue(c);
        }
        if (total <= 0) {
            total = 1;
        }

        double a0 = -Math.PI / 2;
        double sweep = 2 * Math.PI;
        int ci = 0;
        for (Map<String, Object> c : children) {
            double v = subtreeValue(c);
            double span = sweep * (v / total);
            double a1 = a0 + span;
            drawSectorRecursive(sb, cx, cy, holeR, maxR, c, a0, a1, 1, dMax, ci);
            a0 = a1;
            ci++;
        }
    }

    /** 根深度=0；返回根到最深叶子的边数（根下第一层为 1）。 */
    private static int maxDepth(Map<String, Object> node) {
        Object chObj = node.get("children");
        if (!(chObj instanceof List) || ((List<?>) chObj).isEmpty()) {
            return 0;
        }
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> ch = (List<Map<String, Object>>) chObj;
        int m = 0;
        for (Map<String, Object> c : ch) {
            m = Math.max(m, 1 + maxDepth(c));
        }
        return m;
    }

    static double subtreeValue(Map<String, Object> n) {
        Object chObj = n.get("children");
        if (chObj instanceof List && !((List<?>) chObj).isEmpty()) {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> ch = (List<Map<String, Object>>) chObj;
            double s = 0;
            for (Map<String, Object> c : ch) {
                s += subtreeValue(c);
            }
            if (s > 0) {
                return s;
            }
        }
        Object v = n.get("value");
        if (v instanceof Number) {
            return Math.max(((Number) v).doubleValue(), 1e-9);
        }
        return 1;
    }

    private void drawSectorRecursive(StringBuilder sb, double cx, double cy, double holeR, double maxR,
                                     Map<String, Object> node, double ang0, double ang1,
                                     int depth, int dMax, int colorIdx) {
        double inner = holeR + (maxR - holeR) * (depth - 1) / dMax;
        double outer = holeR + (maxR - holeR) * depth / dMax;
        String fill = colorPalette[colorIdx % colorPalette.length];
        fillAnnularSector(sb, cx, cy, inner, outer, ang0, ang1 - ang0, fill);
        appendSectorLabelIfFits(sb, cx, cy, inner, outer, ang0, ang1, node);

        Object chObj = node.get("children");
        if (!(chObj instanceof List) || ((List<?>) chObj).isEmpty()) {
            return;
        }
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> children = (List<Map<String, Object>>) chObj;
        double sum = 0;
        for (Map<String, Object> c : children) {
            sum += subtreeValue(c);
        }
        if (sum <= 0) {
            return;
        }
        double cur = ang0;
        int i = 0;
        for (Map<String, Object> c : children) {
            double v = subtreeValue(c);
            double span = (ang1 - ang0) * (v / sum);
            drawSectorRecursive(sb, cx, cy, holeR, maxR, c, cur, cur + span, depth + 1, dMax, colorIdx + i);
            cur += span;
            i++;
        }
    }

    /** 圆环扇区：数学角从 ang0 逆时针扫 spanRad（弧度），y 轴向下故 y = cy - r*sin(θ)。 */
    private void fillAnnularSector(StringBuilder sb, double cx, double cy,
                                   double rInner, double rOuter, double ang0, double spanRad, String fill) {
        if (spanRad < 1e-7 || rOuter <= rInner + 0.5) {
            return;
        }
        int n = Math.max(10, (int) (Math.abs(spanRad) / (Math.PI / 24)));
        StringBuilder d = new StringBuilder();
        for (int k = 0; k <= n; k++) {
            double t = k / (double) n;
            double ang = ang0 + spanRad * t;
            double x = cx + rOuter * Math.cos(ang);
            double y = cy - rOuter * Math.sin(ang);
            if (k == 0) {
                d.append(String.format(Locale.US, "M %.2f %.2f", x, y));
            } else {
                d.append(String.format(Locale.US, " L %.2f %.2f", x, y));
            }
        }
        for (int k = n; k >= 0; k--) {
            double t = k / (double) n;
            double ang = ang0 + spanRad * t;
            double x = cx + rInner * Math.cos(ang);
            double y = cy - rInner * Math.sin(ang);
            d.append(String.format(Locale.US, " L %.2f %.2f", x, y));
        }
        d.append(" Z");
        sb.append("<path d=\"").append(d).append("\" fill=\"").append(fill)
          .append("\" stroke=\"#ffffff\" stroke-width=\"0.9\" opacity=\"0.95\"/>\n");
    }

    private static String nodeLabel(Map<String, Object> n) {
        if (n == null) {
            return "";
        }
        Object name = n.get("name");
        if (name != null) {
            return name.toString();
        }
        Object label = n.get("label");
        if (label != null) {
            return label.toString();
        }
        return "";
    }

    /** 扇区足够大时绘制名称；小扇区省略以免叠字（与常见旭日图一致）。 */
    private void appendSectorLabelIfFits(StringBuilder sb, double cx, double cy,
                                         double inner, double outer, double ang0, double ang1,
                                         Map<String, Object> node) {
        double span = ang1 - ang0;
        double midR = 0.5 * (inner + outer);
        double arcLen = Math.abs(span) * midR;
        if (arcLen < 24 || Math.abs(span) < 0.12) {
            return;
        }
        String raw = nodeLabel(node);
        if (raw.isEmpty()) {
            return;
        }
        String label = raw.length() > 12 ? raw.substring(0, 12) + "…" : raw;
        double midAng = ang0 + 0.5 * span;
        double tx = cx + midR * Math.cos(midAng);
        double ty = cy - midR * Math.sin(midAng);
        sb.append(String.format(Locale.US,
            "<text x=\"%.2f\" y=\"%.2f\" text-anchor=\"middle\" dominant-baseline=\"central\" "
                + "class=\"tick-label\" font-size=\"9\" font-weight=\"600\" fill=\"#f8fafc\" "
                + "stroke=\"#1f2937\" stroke-width=\"0.35\" paint-order=\"stroke\">%s</text>\n",
            tx, ty, escXml(label)));
    }
}
