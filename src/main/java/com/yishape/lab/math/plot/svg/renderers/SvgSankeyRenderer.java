package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * 桑基图 SVG：按有向边迭代松弛得到分层（左→右），层内按流量堆叠节点，曲线连接层间。
 */
public class SvgSankeyRenderer extends AbstractSvgChartRenderer {

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
              .append("桑基图数据格式：nodes=[{name}], links=[{source,target,value}]").append("</text>\n");
            return;
        }

        int n = nodes.size();
        double marginL = 72;
        double marginR = 72;
        double marginV = 70;
        double W = config.width - marginL - marginR;
        double H = config.height - marginV * 2;

        int[] layer = computeLayers(n, links);
        int maxL = 0;
        for (int L : layer) {
            maxL = Math.max(maxL, L);
        }
        if (maxL < 0) {
            maxL = 0;
        }

        double[] nodeTop = new double[n];
        double[] nodeBot = new double[n];
        double[] nodeX = new double[n];
        double[] inSum = new double[n];
        double[] outSum = new double[n];
        for (Map<String, Object> lk : links) {
            int s = asInt(lk.get("source"));
            int t = asInt(lk.get("target"));
            double v = asDouble(lk.getOrDefault("value", 1));
            if (s >= 0 && s < n) {
                outSum[s] += v;
            }
            if (t >= 0 && t < n) {
                inSum[t] += v;
            }
        }
        double[] thickness = new double[n];
        for (int i = 0; i < n; i++) {
            thickness[i] = Math.max(inSum[i], outSum[i]);
            if (thickness[i] < 1e-9) {
                thickness[i] = 1;
            }
        }

        @SuppressWarnings("unchecked")
        List<Integer>[] byLayer = new List[maxL + 1];
        for (int L = 0; L <= maxL; L++) {
            byLayer[L] = new ArrayList<>();
        }
        for (int i = 0; i < n; i++) {
            int L = Math.max(0, Math.min(maxL, layer[i]));
            byLayer[L].add(i);
        }
        for (int L = 0; L <= maxL; L++) {
            byLayer[L].sort(Comparator.comparingInt(a -> a));
        }

        for (int L = 0; L <= maxL; L++) {
            double x = marginL + (maxL == 0 ? W / 2 : W * L / (double) maxL);
            List<Integer> ids = byLayer[L];
            double sumH = 0;
            for (int id : ids) {
                sumH += thickness[id];
            }
            double gap = ids.size() > 1 ? Math.min(16, H * 0.04) : 0;
            double total = sumH + gap * Math.max(0, ids.size() - 1);
            double y = marginV + (H - total) / 2;
            for (int id : ids) {
                nodeX[id] = x;
                nodeTop[id] = y;
                nodeBot[id] = y + thickness[id];
                y = nodeBot[id] + gap;
            }
        }

        List<double[]> linkGeom = new ArrayList<>();
        for (Map<String, Object> lk : links) {
            int s = asInt(lk.get("source"));
            int t = asInt(lk.get("target"));
            if (s < 0 || s >= n || t < 0 || t >= n) {
                continue;
            }
            double v = asDouble(lk.getOrDefault("value", 1));
            linkGeom.add(new double[] { s, t, v });
        }
        linkGeom.sort(Comparator.comparingDouble(a -> -a[2]));

        Map<String, double[]> outCursor = new HashMap<>();
        Map<String, double[]> inCursor = new HashMap<>();
        for (int i = 0; i < n; i++) {
            outCursor.put("n" + i, new double[] { nodeTop[i] });
            inCursor.put("n" + i, new double[] { nodeTop[i] });
        }

        for (double[] lg : linkGeom) {
            int s = (int) lg[0];
            int t = (int) lg[1];
            double v = lg[2];
            double[] oc = outCursor.get("n" + s);
            double[] ic = inCursor.get("n" + t);
            double y0a = oc[0];
            double y1a = oc[0] + v;
            oc[0] = y1a;
            double y0b = ic[0];
            double y1b = ic[0] + v;
            ic[0] = y1b;

            double xs = nodeX[s];
            double xt = nodeX[t];
            double dx = Math.max(48, (xt - xs) * 0.42);
            String color = colorPalette[s % colorPalette.length];
            sb.append(String.format(Locale.US,
                "<path d=\"M %.2f %.2f C %.2f %.2f %.2f %.2f %.2f %.2f L %.2f %.2f C %.2f %.2f %.2f %.2f %.2f %.2f Z\" "
                    + "fill=\"%s\" fill-opacity=\"0.38\" stroke=\"none\"/>\n",
                xs, y0a, xs + dx, y0a, xt - dx, y0b, xt, y0b,
                xt, y1b, xt - dx, y1b, xs + dx, y1a, xs, y1a, color));
        }

        double barHalf = 5;
        for (int i = 0; i < n; i++) {
            double x = nodeX[i];
            String name = nodes.get(i).getOrDefault("name", "N" + i).toString();
            String color = colorPalette[i % colorPalette.length];
            sb.append(String.format(Locale.US,
                "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" rx=\"2\" fill=\"%s\" "
                    + "stroke=\"#e5e7eb\" stroke-width=\"0.8\"/>\n",
                x - barHalf, nodeTop[i], 2 * barHalf, nodeBot[i] - nodeTop[i], color));
            sb.append("<text x=\"").append((int) (x + barHalf + 6)).append("\" y=\"")
              .append((int) ((nodeTop[i] + nodeBot[i]) / 2 + 4))
              .append("\" class=\"tick-label\" font-size=\"11\">").append(escXml(name)).append("</text>\n");
        }
    }

    private static int asInt(Object o) {
        if (o instanceof Number) {
            return ((Number) o).intValue();
        }
        return Integer.parseInt(String.valueOf(o));
    }

    private static double asDouble(Object o) {
        if (o instanceof Number) {
            return ((Number) o).doubleValue();
        }
        return Double.parseDouble(String.valueOf(o));
    }

    /**
     * 分层：反复松弛 layer[target] = max(layer[source]+1)，直到收敛（DAG 或含环时尽量左到右）。
     */
    private static int[] computeLayers(int n, List<Map<String, Object>> links) {
        int[] layer = new int[n];
        Arrays.fill(layer, 0);
        for (int iter = 0; iter < n + 2; iter++) {
            boolean ch = false;
            for (Map<String, Object> lk : links) {
                int s = asInt(lk.get("source"));
                int t = asInt(lk.get("target"));
                if (s < 0 || s >= n || t < 0 || t >= n) {
                    continue;
                }
                int nl = layer[s] + 1;
                if (nl > layer[t]) {
                    layer[t] = nl;
                    ch = true;
                }
            }
            if (!ch) {
                break;
            }
        }
        int minL = Arrays.stream(layer).min().orElse(0);
        for (int i = 0; i < n; i++) {
            layer[i] -= minL;
        }
        return layer;
    }
}
