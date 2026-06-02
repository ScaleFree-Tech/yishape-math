package com.yishape.lab.math.plot.javafx.renderers;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 复杂图表渲染器：漏斗、桑基、旭日、矩形树、树、关系图、平行坐标、主题河流等。
 * <p>旭日图按层级数据 {@code children} 做面积比例扇环；矩形树图为 Bruls 类 Squarify；
 * 树图为深度优先叶子序 + 父结点水平居中；关系图为 Fruchterman–Reingold 力导向布局（固定迭代）。</p>
 *
 * @author lteb2
 */
public class ComplexChartRenderer implements JavaFxChartRenderer {

    /** 与 {@link #ComplexChartRenderer(String)} 传入值一致。 */
    private String chartType;
    
    public ComplexChartRenderer(String chartType) {
        this.chartType = chartType;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) return;
        
        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);
        SeriesData series = data.get(0);
        
        // 清空画布
        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        
        // 绘制标题
        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        
        switch (chartType) {
            case "funnel":
                renderFunnel(gc, series, config, themeManager);
                break;
            case "sankey":
                renderSankey(gc, series, config, themeManager);
                break;
            case "sunburst":
                renderSunburst(gc, series, config, themeManager);
                break;
            case "treemap":
                renderTreemap(gc, series, config, themeManager);
                break;
            case "tree":
                renderTree(gc, series, config, themeManager);
                break;
            case "graph":
                renderGraph(gc, series, config, themeManager);
                break;
            case "parallel":
                renderParallel(gc, series, config, themeManager);
                break;
            case "themeRiver":
                renderThemeRiver(gc, series, config, themeManager);
                break;
        }
        JavaFxChartUtils.drawCartesianAxisTitles(gc, config, themeManager);
    }
    
    private void renderFunnel(GraphicsContext gc, SeriesData series,
                           ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null) return;
        
        double centerX = config.width / 2.0;
        double chartTop = config.paddingTop + 50;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom - 50;
        double maxWidth = (config.width - config.paddingLeft - config.paddingRight) * 0.8;
        
        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < series.y.length(); i++) {
            maxValue = Math.max(maxValue, series.y.get(i));
        }
        
        int numItems = series.y.length();
        String[] palette = themeManager.getColorPalette();
        
        for (int i = 0; i < numItems; i++) {
            double value = series.y.get(i);
            double y = chartTop + (i / (double) numItems) * chartHeight;
            double sectionHeight = chartHeight / numItems - 5;
            double sectionWidth = (value / maxValue) * maxWidth;
            
            gc.setFill(Color.web(palette[i % palette.length]));
            gc.fillRect(centerX - sectionWidth / 2, y, sectionWidth, sectionHeight);
            
            // 绘制边框
            gc.setStroke(Color.WHITE);
            gc.setLineWidth(2);
            gc.strokeRect(centerX - sectionWidth / 2, y, sectionWidth, sectionHeight);
            
            // 绘制标签
            gc.setFill(Color.WHITE);
            gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 12));
            gc.setTextAlign(TextAlignment.CENTER);
            String label = series.labels != null && i < series.labels.size()
                ? series.labels.get(i) + ": " + String.format("%.1f", value)
                : String.format("%.1f", value);
            gc.fillText(label, centerX, y + sectionHeight / 2 + 5);
        }
    }
    
    private void renderSankey(GraphicsContext gc, SeriesData series,
                             ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nodeList = (List<Map<String, Object>>) series.extraData.get("nodes");
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> linkList = (List<Map<String, Object>>) series.extraData.get("links");

        if (nodeList == null || linkList == null || nodeList.isEmpty()) {
            return;
        }

        int n = nodeList.size();
        String[] palette = themeManager.getColorPalette();
        if (palette == null || palette.length == 0) {
            palette = new String[]{"#4878d0"};
        }

        List<SankeyLink> links = new ArrayList<>();
        for (Map<String, Object> m : linkList) {
            Integer s = sankeyNodeIndex(m.get("source"));
            Integer t = sankeyNodeIndex(m.get("target"));
            if (s == null || t == null || s.equals(t) || s < 0 || t < 0 || s >= n || t >= n) {
                continue;
            }
            double v = sankeyLinkValue(m);
            if (v <= 0) {
                v = 1e-6;
            }
            links.add(new SankeyLink(s, t, v));
        }
        if (links.isEmpty()) {
            return;
        }

        double[] inSum = new double[n];
        double[] outSum = new double[n];
        for (SankeyLink L : links) {
            outSum[L.source] += L.value;
            inSum[L.target] += L.value;
        }
        double[] flow = new double[n];
        double colSumAll = 0;
        for (int i = 0; i < n; i++) {
            flow[i] = Math.max(inSum[i], outSum[i]);
            if (flow[i] < 1e-9) {
                flow[i] = 1e-6;
            }
            colSumAll += flow[i];
        }

        int[] depth = sankeyComputeDepth(n, links);
        int maxDepth = 0;
        for (int d : depth) {
            maxDepth = Math.max(maxDepth, d);
        }
        int numCols = maxDepth + 1;

        Map<Integer, List<Integer>> byCol = new HashMap<>();
        for (int i = 0; i < n; i++) {
            byCol.computeIfAbsent(depth[i], k -> new ArrayList<>()).add(i);
        }
        for (List<Integer> ids : byCol.values()) {
            ids.sort(Comparator.comparingInt(a -> a));
        }

        double innerW = config.width - config.paddingLeft - config.paddingRight;
        double chartTop = config.paddingTop + 42;
        double chartBottom = config.height - config.paddingBottom - 28;
        double innerH = Math.max(80, chartBottom - chartTop);
        final double nodeBarW = 16;
        final double nodeGap = 5;
        // 左右留空：节点文字在首列左侧、末列右侧，避免贴边裁切
        final double colPad = 56;

        double[] colX = new double[numCols];
        double usableW = Math.max(innerW - 2 * colPad, numCols * 40);
        for (int c = 0; c < numCols; c++) {
            colX[c] = config.paddingLeft + colPad + (c + 0.5) * (usableW / Math.max(1, numCols));
        }

        double[] nodeTop = new double[n];
        double[] nodeBottom = new double[n];
        double[] nodeRightX = new double[n];
        double[] nodeLeftX = new double[n];

        for (int c = 0; c < numCols; c++) {
            List<Integer> ids = byCol.get(c);
            if (ids == null || ids.isEmpty()) {
                continue;
            }
            double colFlow = 0;
            for (int id : ids) {
                colFlow += flow[id];
            }
            double stackBudget = innerH * 0.9;
            double gapTotal = nodeGap * Math.max(0, ids.size() - 1);
            double scale = (stackBudget - gapTotal) / Math.max(colFlow, 1e-9);
            double y = chartTop + (innerH - (colFlow * scale + gapTotal)) / 2;
            double xc = colX[c];
            for (int id : ids) {
                double h = flow[id] * scale;
                nodeTop[id] = y;
                nodeBottom[id] = y + h;
                nodeLeftX[id] = xc - nodeBarW / 2;
                nodeRightX[id] = xc + nodeBarW / 2;
                y += h + nodeGap;
            }
        }

        Map<Integer, List<SankeyLink>> outBy = new HashMap<>();
        Map<Integer, List<SankeyLink>> inBy = new HashMap<>();
        for (SankeyLink L : links) {
            outBy.computeIfAbsent(L.source, k -> new ArrayList<>()).add(L);
            inBy.computeIfAbsent(L.target, k -> new ArrayList<>()).add(L);
        }
        for (List<SankeyLink> lst : outBy.values()) {
            lst.sort(Comparator.comparingInt(a -> a.target));
        }
        for (List<SankeyLink> lst : inBy.values()) {
            lst.sort(Comparator.comparingInt(a -> a.source));
        }

        record LinkGeom(SankeyLink link, double y0a, double y0b, double y1a, double y1b) {}
        List<LinkGeom> geoms = new ArrayList<>();

        for (SankeyLink L : links) {
            List<SankeyLink> outs = outBy.get(L.source);
            List<SankeyLink> ins = inBy.get(L.target);
            if (outs == null || ins == null) {
                continue;
            }
            double outBase = nodeTop[L.source];
            double outSpan = nodeBottom[L.source] - nodeTop[L.source];
            double cum = 0;
            for (SankeyLink o : outs) {
                if (o == L) {
                    break;
                }
                cum += o.value;
            }
            double y0a = outBase + (cum / outSum[L.source]) * outSpan;
            double y0b = y0a + (L.value / outSum[L.source]) * outSpan;

            double inBase = nodeTop[L.target];
            double inSpan = nodeBottom[L.target] - nodeTop[L.target];
            cum = 0;
            for (SankeyLink iL : ins) {
                if (iL == L) {
                    break;
                }
                cum += iL.value;
            }
            double y1a = inBase + (cum / inSum[L.target]) * inSpan;
            double y1b = y1a + (L.value / inSum[L.target]) * inSpan;
            geoms.add(new LinkGeom(L, y0a, y0b, y1a, y1b));
        }

        geoms.sort(Comparator.comparingDouble(g -> g.link().value));

        Color axisStroke = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("axisColor"), Color.web("#94a3b8"));

        for (LinkGeom g : geoms) {
            SankeyLink L = g.link();
            double xr = nodeRightX[L.source];
            double xl = nodeLeftX[L.target];
            if (xl <= xr + 2) {
                continue;
            }
            double spanX = xl - xr;
            double dx = spanX * 0.52;
            double cx1 = xr + dx;
            double cx2 = xl - dx;
            // 经典桑基：控制点相对端点做竖直偏移，使带状在中间隆起/外鼓（宽度仍由端点决定）
            double hL = Math.max(1e-6, g.y0b - g.y0a);
            double hR = Math.max(1e-6, g.y1b - g.y1a);
            double minHRib = Math.min(hL, hR);
            double bump = Math.min(innerH * 0.12, spanX * 0.065 + 0.22 * minHRib);
            double floorBump = Math.min(innerH * 0.028, spanX * 0.022);
            bump = Math.max(bump, floorBump);
            double safeHalf = 0.42 * minHRib;
            if (safeHalf > 1e-9) {
                bump = Math.min(bump, safeHalf);
            }
            double cy1Top = g.y0a - bump;
            double cy2Top = g.y1a - bump;
            double cy2Bot = g.y1b + bump;
            double cy1Bot = g.y0b + bump;
            Color base = Color.web(palette[L.source % palette.length]);
            Color fill = new Color(base.getRed(), base.getGreen(), base.getBlue(), 0.42);
            gc.setFill(fill);
            gc.beginPath();
            gc.moveTo(xr, g.y0a);
            gc.bezierCurveTo(cx1, cy1Top, cx2, cy2Top, xl, g.y1a);
            gc.lineTo(xl, g.y1b);
            gc.bezierCurveTo(cx2, cy2Bot, cx1, cy1Bot, xr, g.y0b);
            gc.closePath();
            gc.fill();
            Color strokeC = new Color(base.getRed(), base.getGreen(), base.getBlue(), 0.55);
            gc.setStroke(strokeC);
            gc.setLineWidth(0.85);
            gc.stroke();
        }

        gc.setGlobalAlpha(1.0);
        for (int i = 0; i < n; i++) {
            Color base = Color.web(palette[i % palette.length]);
            gc.setFill(new Color(base.getRed(), base.getGreen(), base.getBlue(), 0.88));
            gc.fillRect(nodeLeftX[i], nodeTop[i], nodeBarW, nodeBottom[i] - nodeTop[i]);
            gc.setStroke(axisStroke);
            gc.setLineWidth(1);
            gc.strokeRect(nodeLeftX[i], nodeTop[i], nodeBarW, nodeBottom[i] - nodeTop[i]);
        }

        gc.setFill(themeManager.getTextColor());
        Font nameFont = Font.font(themeManager.getLabelFont().getFamily(), 11);
        Font subFont = Font.font(themeManager.getLabelFont().getFamily(), 9);
        final double labelGap = 8;
        for (int i = 0; i < n; i++) {
            Map<String, Object> node = nodeList.get(i);
            String name = String.valueOf(node.getOrDefault("name", "Node " + i));
            String sub = colSumAll > 0 ? String.format("%.0f", flow[i]) : "";
            boolean hasSub = !sub.isEmpty() && flow[i] > 0;
            double midY = (nodeTop[i] + nodeBottom[i]) * 0.5;
            int col = depth[i];
            double tx;
            TextAlignment align;
            if (col == 0) {
                tx = nodeLeftX[i] - labelGap;
                align = TextAlignment.RIGHT;
            } else if (col == maxDepth) {
                tx = nodeRightX[i] + labelGap;
                align = TextAlignment.LEFT;
            } else {
                tx = nodeLeftX[i] - labelGap;
                align = TextAlignment.RIGHT;
            }
            gc.setTextAlign(align);
            gc.setFont(nameFont);
            double nameBaseline = hasSub ? midY - 3 : midY + 4;
            gc.fillText(name, tx, nameBaseline);
            if (hasSub) {
                gc.setFill(themeManager.getMutedTextColor());
                gc.setFont(subFont);
                gc.fillText(sub, tx, midY + 12);
                gc.setFill(themeManager.getTextColor());
                gc.setFont(nameFont);
            }
        }
    }

    private static Integer sankeyNodeIndex(Object o) {
        if (o instanceof Number) {
            return ((Number) o).intValue();
        }
        if (o == null) {
            return null;
        }
        try {
            return Integer.parseInt(String.valueOf(o));
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static double sankeyLinkValue(Map<String, Object> m) {
        Number v = (Number) m.get("value");
        return v != null ? v.doubleValue() : 1.0;
    }

    /** 按有向边计算分层 depth（多轮松弛）；保证 source 在 target 左侧。 */
    private static int[] sankeyComputeDepth(int n, List<SankeyLink> links) {
        int[] depth = new int[n];
        for (int iter = 0; iter < n + 2; iter++) {
            boolean ch = false;
            for (SankeyLink L : links) {
                int nd = depth[L.source] + 1;
                if (nd > depth[L.target]) {
                    depth[L.target] = nd;
                    ch = true;
                }
            }
            if (!ch) {
                break;
            }
        }
        return depth;
    }

    private static final class SankeyLink {
        final int source;
        final int target;
        final double value;

        SankeyLink(int source, int target, double value) {
            this.source = source;
            this.target = target;
            this.value = value;
        }
    }
    
    private void renderSunburst(GraphicsContext gc, SeriesData series,
                               ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> raw = (List<Map<String, Object>>) series.extraData.get("hierarchicalData");
        HNode root = buildSunburstRoot(raw);
        if (root == null || root.children.isEmpty()) {
            return;
        }
        String[] palette = themeManager.getColorPalette();
        if (palette == null || palette.length == 0) {
            palette = new String[] {"#4878d0"};
        }
        double availW = config.width - config.paddingLeft - config.paddingRight;
        double availH = config.height - config.paddingTop - config.paddingBottom - 36;
        double cx = config.paddingLeft + availW / 2.0;
        double cy = config.paddingTop + 42 + availH / 2.0;
        double rMax = Math.min(availW, availH) / 2.0 - 6;
        double rHole = Math.max(20, rMax * 0.16);
        if (rMax <= rHole + 8) {
            rHole = rMax * 0.22;
        }
        int maxRing = Math.max(1, maxSunburstRings(root));
        int[] colorCounter = {0};
        drawSunburstLevel(gc, root, cx, cy, -90, 360, 0, maxRing, rHole, rMax,
            palette, themeManager, colorCounter);
    }

    private void renderTreemap(GraphicsContext gc, SeriesData series,
                              ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> raw = (List<Map<String, Object>>) series.extraData.get("treemapData");
        if (raw == null || raw.isEmpty()) {
            return;
        }
        TmItem root = buildTreemapRoot(raw);
        if (root == null) {
            return;
        }
        String[] palette = themeManager.getColorPalette();
        if (palette == null || palette.length == 0) {
            palette = new String[] {"#4878d0"};
        }
        double x = config.paddingLeft + 3;
        double y = config.paddingTop + 42;
        double w = config.width - config.paddingLeft - config.paddingRight - 6;
        double h = config.height - config.paddingTop - config.paddingBottom - 48;
        if (w < 12 || h < 12) {
            return;
        }
        layoutTreemapNode(root, x, y, w, h);
        int[] idx = {0};
        drawTreemapItems(gc, root, palette, idx, themeManager);
    }

    /**
     * 层次树：约定 {@code treeData} 为单层或多层 {@code Map}，支持 {@code children} 为
     * {@code List<Map<String,Object>>}；多根时包在虚拟根下。布局为深度下排、叶子按序占列、父水平居中。
     */
    @SuppressWarnings("unchecked")
    private void renderTree(GraphicsContext gc, SeriesData series,
                           ChartConfig config, JavaFxThemeManager themeManager) {
        List<Map<String, Object>> raw = (List<Map<String, Object>>) series.extraData.get("treeData");
        TreeNode root = buildTreeRoot(raw);
        if (root == null) {
            return;
        }
        String[] palette = themeManager.getColorPalette();
        if (palette == null || palette.length == 0) {
            palette = new String[] {"#4878d0"};
        }
        double[] nextCol = {0};
        assignTreeLeafX(root, nextCol);
        setTreeDepth(root, 0);
        TreeBounds tb = treeBounds(root);
        double leafSpan = tb.maxX - tb.minX + 1;
        double chartW = config.width - config.paddingLeft - config.paddingRight - 40;
        double chartH = config.height - config.paddingTop - config.paddingBottom - 56;
        double xScale = leafSpan > 0 ? chartW / Math.max(leafSpan, 1) : chartW;
        double yStep = Math.max(52, Math.min(76, chartH / Math.max(2, tb.maxDepth + 1)));
        double x0 = config.paddingLeft + 20 - tb.minX * xScale;
        double y0 = config.paddingTop + 52;
        mapTreeToPixels(root, x0, y0, xScale, yStep);
        centerTreeInPlotBounds(root, config);
        Color edgeColor = themeManager.getMutedTextColor();
        gc.setStroke(edgeColor);
        gc.setLineWidth(1.25);
        gc.setLineDashes();
        drawTreeEdges(gc, root);
        drawTreeNodes(gc, root, palette, themeManager);
    }

    /**
     * 力导向关系图：{@code graphNodes}、{@code graphLinks}，边 {@code source}/{@code target} 为结点下标。
     */
    @SuppressWarnings("unchecked")
    private void renderGraph(GraphicsContext gc, SeriesData series,
                            ChartConfig config, JavaFxThemeManager themeManager) {
        List<Map<String, Object>> nodes = (List<Map<String, Object>>) series.extraData.get("graphNodes");
        List<Map<String, Object>> links = (List<Map<String, Object>>) series.extraData.get("graphLinks");
        if (nodes == null || nodes.isEmpty()) {
            return;
        }
        String[] palette = themeManager.getColorPalette();
        if (palette == null || palette.length == 0) {
            palette = new String[] {"#4878d0"};
        }
        int n = nodes.size();
        double plotLeft = config.paddingLeft + 28;
        double plotTop = config.paddingTop + 52;
        double plotW = config.width - config.paddingLeft - config.paddingRight - 56;
        double plotH = config.height - config.paddingTop - config.paddingBottom - 44;
        if (plotW < 80 || plotH < 80) {
            return;
        }
        double cx = plotLeft + plotW / 2.0;
        double cy = plotTop + plotH / 2.0;
        double R = Math.min(plotW, plotH) * 0.38;
        double[] px = new double[n];
        double[] py = new double[n];
        for (int i = 0; i < n; i++) {
            double ang = 2 * Math.PI * i / Math.max(1, n) - Math.PI / 2;
            px[i] = cx + R * Math.cos(ang);
            py[i] = cy + R * Math.sin(ang);
        }
        Color ecol = themeManager.getMutedTextColor();
        double area = Math.max(plotW * plotH, 8000);
        fruchtermanReingold(px, py, links, n, area);
        fitGraphPositionsToRect(px, py, n, plotLeft + 32, plotTop + 28, plotLeft + plotW - 32, plotTop + plotH - 28);
        if (links != null) {
            gc.setStroke(new Color(ecol.getRed(), ecol.getGreen(), ecol.getBlue(), 0.78));
            gc.setLineWidth(2);
            gc.setLineDashes();
            for (Map<String, Object> L : links) {
                Integer s = sankeyNodeIndex(L.get("source"));
                Integer t = sankeyNodeIndex(L.get("target"));
                if (s == null || t == null || s < 0 || t < 0 || s >= n || t >= n || s.equals(t)) {
                    continue;
                }
                gc.strokeLine(px[s], py[s], px[t], py[t]);
            }
        }
        double nodeR = Math.max(18, Math.min(32, Math.min(plotW, plotH) / (3.2 + n * 0.35)));
        for (int i = 0; i < n; i++) {
            Color base = Color.web(palette[i % palette.length]);
            gc.setFill(base);
            gc.fillOval(px[i] - nodeR, py[i] - nodeR, nodeR * 2, nodeR * 2);
            gc.setStroke(ecol);
            gc.setLineWidth(1.35);
            gc.strokeOval(px[i] - nodeR, py[i] - nodeR, nodeR * 2, nodeR * 2);
            gc.setFill(themeManager.getTextColor());
            gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 10));
            gc.setTextAlign(TextAlignment.CENTER);
            String nm = String.valueOf(nodes.get(i).getOrDefault("name", "N" + i));
            gc.fillText(truncateLabel(nm, 12), px[i], py[i] + 4);
        }
    }

    // --- 旭日图：分层扇环 --------------------------------------------------------

    private static final class HNode {
        String name = "";
        double value;
        final List<HNode> children = new ArrayList<>();
    }

    private static HNode buildSunburstRoot(List<Map<String, Object>> raw) {
        if (raw == null || raw.isEmpty()) {
            return null;
        }
        HNode root = new HNode();
        root.name = "";
        for (Map<String, Object> m : raw) {
            root.children.add(mapToHNode(m));
        }
        revalueSunburst(root);
        return root;
    }

    @SuppressWarnings("unchecked")
    private static HNode mapToHNode(Map<String, Object> m) {
        HNode n = new HNode();
        n.name = String.valueOf(m.getOrDefault("name", ""));
        Object ch = m.get("children");
        if (ch instanceof List<?>) {
            for (Object o : (List<?>) ch) {
                if (o instanceof Map) {
                    n.children.add(mapToHNode((Map<String, Object>) o));
                }
            }
        }
        if (n.children.isEmpty()) {
            n.value = numericOr(m.get("value"), 1.0);
        } else {
            n.value = 0;
        }
        return n;
    }

    private static void revalueSunburst(HNode n) {
        if (n.children.isEmpty()) {
            if (n.value <= 0) {
                n.value = 1;
            }
            return;
        }
        double s = 0;
        for (HNode c : n.children) {
            revalueSunburst(c);
            s += Math.max(1e-9, c.value);
        }
        n.value = s;
    }

    private static int maxSunburstRings(HNode n) {
        if (n.children.isEmpty()) {
            return 0;
        }
        int m = 0;
        for (HNode c : n.children) {
            m = Math.max(m, maxSunburstRings(c));
        }
        return 1 + m;
    }

    private void drawSunburstLevel(GraphicsContext gc, HNode node, double cx, double cy,
                                  double startDeg, double extentDeg, int depth, int maxRing,
                                  double rHole, double rMax, String[] palette,
                                  JavaFxThemeManager tm, int[] colorCounter) {
        if (node.children.isEmpty()) {
            return;
        }
        double sum = 0;
        for (HNode c : node.children) {
            sum += Math.max(1e-9, c.value);
        }
        double ringW = (rMax - rHole) / maxRing;
        double r0 = rHole + depth * ringW;
        double r1 = r0 + ringW;
        double acc = startDeg;
        for (HNode c : node.children) {
            double frac = Math.max(1e-9, c.value) / sum;
            double ext = extentDeg * frac;
            Color base = Color.web(palette[colorCounter[0] % palette.length]);
            colorCounter[0]++;
            Color fill = new Color(base.getRed(), base.getGreen(), base.getBlue(), 0.9);
            fillAnnularSector(gc, cx, cy, r0, r1, acc, ext, fill, Color.color(1, 1, 1, 0.42));
            if (ext > 14 && (r1 - r0) > 12) {
                double mid = acc + ext / 2;
                double rm = (r0 + r1) / 2;
                double rmRad = Math.toRadians(mid);
                gc.setFill(tm.getTextColor());
                gc.setFont(Font.font(tm.getLabelFont().getFamily(), 10));
                gc.setTextAlign(TextAlignment.CENTER);
                gc.fillText(truncateLabel(c.name, 14), cx + rm * Math.cos(rmRad), cy + rm * Math.sin(rmRad) + 3);
            }
            drawSunburstLevel(gc, c, cx, cy, acc, ext, depth + 1, maxRing, rHole, rMax, palette, tm, colorCounter);
            acc += ext;
        }
    }

    private static void fillAnnularSector(GraphicsContext gc, double cx, double cy,
                                         double r0, double r1, double startDeg, double extentDeg,
                                         Color fill, Color stroke) {
        if (extentDeg <= 0.05 || r1 <= r0 + 0.5) {
            return;
        }
        double rLo = Math.min(r0, r1);
        double rHi = Math.max(r0, r1);
        int steps = Math.max(16, (int) Math.ceil(Math.abs(extentDeg) / 3));
        double rad0 = Math.toRadians(startDeg);
        double rad1 = Math.toRadians(startDeg + extentDeg);
        gc.beginPath();
        gc.moveTo(cx + rLo * Math.cos(rad0), cy + rLo * Math.sin(rad0));
        gc.lineTo(cx + rHi * Math.cos(rad0), cy + rHi * Math.sin(rad0));
        for (int s = 1; s <= steps; s++) {
            double t = s / (double) steps;
            double a = rad0 + t * (rad1 - rad0);
            gc.lineTo(cx + rHi * Math.cos(a), cy + rHi * Math.sin(a));
        }
        gc.lineTo(cx + rLo * Math.cos(rad1), cy + rLo * Math.sin(rad1));
        for (int s = steps - 1; s >= 0; s--) {
            double t = s / (double) steps;
            double a = rad0 + t * (rad1 - rad0);
            gc.lineTo(cx + rLo * Math.cos(a), cy + rLo * Math.sin(a));
        }
        gc.closePath();
        gc.setFill(fill);
        gc.fill();
        if (stroke != null) {
            gc.setStroke(stroke);
            gc.setLineWidth(1);
            gc.stroke();
        }
    }

    private static String truncateLabel(String s, int maxChars) {
        if (s == null) {
            return "";
        }
        if (s.length() <= maxChars) {
            return s;
        }
        return s.substring(0, Math.max(1, maxChars - 1)) + "…";
    }

    private static double numericOr(Object v, double def) {
        if (v instanceof Number) {
            return ((Number) v).doubleValue();
        }
        return def;
    }

    // --- 矩形树图：Squarify -----------------------------------------------------

    private static final class TmItem {
        String name = "";
        double leafValue;
        final List<TmItem> children = new ArrayList<>();
        double rx, ry, rw, rh;

        double subtreeValue() {
            if (children.isEmpty()) {
                return Math.max(1e-9, leafValue);
            }
            double s = 0;
            for (TmItem c : children) {
                s += c.subtreeValue();
            }
            return Math.max(1e-9, s);
        }
    }

    private static TmItem buildTreemapRoot(List<Map<String, Object>> raw) {
        if (raw == null || raw.isEmpty()) {
            return null;
        }
        TmItem root = new TmItem();
        for (Map<String, Object> m : raw) {
            root.children.add(mapToTmItem(m));
        }
        ensureTreemapValues(root);
        return root;
    }

    @SuppressWarnings("unchecked")
    private static TmItem mapToTmItem(Map<String, Object> m) {
        TmItem n = new TmItem();
        n.name = String.valueOf(m.getOrDefault("name", ""));
        Object ch = m.get("children");
        if (ch instanceof List<?>) {
            for (Object o : (List<?>) ch) {
                if (o instanceof Map) {
                    n.children.add(mapToTmItem((Map<String, Object>) o));
                }
            }
        }
        if (n.children.isEmpty()) {
            n.leafValue = numericOr(m.get("value"), 1.0);
        }
        return n;
    }

    private static void ensureTreemapValues(TmItem n) {
        if (n.children.isEmpty()) {
            if (n.leafValue <= 0) {
                n.leafValue = 1;
            }
            return;
        }
        for (TmItem c : n.children) {
            ensureTreemapValues(c);
        }
    }

    private void layoutTreemapNode(TmItem n, double x, double y, double w, double h) {
        if (w < 2 || h < 2) {
            return;
        }
        if (n.children.isEmpty()) {
            n.rx = x;
            n.ry = y;
            n.rw = w;
            n.rh = h;
            return;
        }
        List<TmItem> cells = new ArrayList<>(n.children);
        cells.sort(Comparator.comparingDouble(TmItem::subtreeValue).reversed());
        double tot = 0;
        for (TmItem c : cells) {
            tot += c.subtreeValue();
        }
        if (tot <= 0) {
            return;
        }
        squarifyRemainder(cells, 0, x, y, w, h, tot);
        final double pad = 1.5;
        for (TmItem c : n.children) {
            if (c.children.isEmpty()) {
                c.rx += pad;
                c.ry += pad;
                c.rw = Math.max(0.5, c.rw - 2 * pad);
                c.rh = Math.max(0.5, c.rh - 2 * pad);
            } else {
                layoutTreemapNode(c, c.rx + pad, c.ry + pad, Math.max(0.5, c.rw - 2 * pad), Math.max(0.5, c.rh - 2 * pad));
            }
        }
    }

    private void squarifyRemainder(List<TmItem> items, int start, double x, double y, double w, double h, double total) {
        if (start >= items.size() || w < 1.5 || h < 1.5 || total <= 0) {
            return;
        }
        List<TmItem> row = new ArrayList<>();
        row.add(items.get(start));
        int idx = start + 1;
        while (idx < items.size()) {
            TmItem next = items.get(idx);
            List<TmItem> tryRow = new ArrayList<>(row);
            tryRow.add(next);
            double s1 = sumValues(row);
            double s2 = s1 + next.subtreeValue();
            if (row.size() >= 1) {
                double w1 = worstRowAspect(row, w, h, s1, total);
                double w2 = worstRowAspect(tryRow, w, h, s2, total);
                if (tryRow.size() > 1 && w2 > w1) {
                    break;
                }
            }
            row.add(next);
            idx++;
        }
        double rowSum = sumValues(row);
        boolean horiz = w >= h;
        double shortLen = horiz ? h : w;
        double longLen = horiz ? w : h;
        double thick = shortLen * (rowSum / total);
        double pos = 0;
        for (TmItem it : row) {
            double along = longLen * (it.subtreeValue() / rowSum);
            if (horiz) {
                it.rx = x + pos;
                it.ry = y;
                it.rw = along;
                it.rh = thick;
            } else {
                it.rx = x;
                it.ry = y + pos;
                it.rw = thick;
                it.rh = along;
            }
            pos += along;
        }
        if (horiz) {
            squarifyRemainder(items, start + row.size(), x, y + thick, w, h - thick,
                total - rowSum);
        } else {
            squarifyRemainder(items, start + row.size(), x + thick, y, w - thick, h,
                total - rowSum);
        }
    }

    private static double sumValues(List<TmItem> row) {
        double s = 0;
        for (TmItem t : row) {
            s += t.subtreeValue();
        }
        return s;
    }

    private static double worstRowAspect(List<TmItem> row, double w, double h, double rowSum, double totalInRect) {
        if (row.isEmpty() || rowSum <= 0 || totalInRect <= 0) {
            return Double.POSITIVE_INFINITY;
        }
        boolean horiz = w >= h;
        double shortLen = horiz ? h : w;
        double longLen = horiz ? w : h;
        double thick = shortLen * (rowSum / totalInRect);
        double worst = 1;
        for (TmItem it : row) {
            double along = longLen * (it.subtreeValue() / rowSum);
            double dw = horiz ? along : thick;
            double dh = horiz ? thick : along;
            double ar = dw / Math.max(1e-9, dh);
            if (ar < 1) {
                ar = 1 / ar;
            }
            worst = Math.max(worst, ar);
        }
        return worst;
    }

    private void drawTreemapItems(GraphicsContext gc, TmItem n, String[] palette, int[] idx, JavaFxThemeManager tm) {
        // 扁平数据会挂 synthetic root 下，根结点不分配 rx/rw，必须先递归子结点再对叶子做尺寸判断
        if (!n.children.isEmpty()) {
            List<TmItem> sorted = new ArrayList<>(n.children);
            sorted.sort(Comparator.comparingDouble(a -> a.ry * 10000 + a.rx));
            for (TmItem c : sorted) {
                drawTreemapItems(gc, c, palette, idx, tm);
            }
            return;
        }
        if (n.rw < 0.5 || n.rh < 0.5) {
            return;
        }
        Color base = Color.web(palette[idx[0] % palette.length]);
        idx[0]++;
        gc.setFill(new Color(base.getRed(), base.getGreen(), base.getBlue(), 0.88));
        gc.fillRect(n.rx, n.ry, n.rw, n.rh);
        gc.setStroke(Color.color(1, 1, 1, 0.7));
        gc.setLineWidth(1.1);
        gc.setLineDashes();
        gc.strokeRect(n.rx, n.ry, n.rw, n.rh);
        if (n.rw > 26 && n.rh > 14) {
            gc.setFill(tm.getTextColor());
            gc.setFont(Font.font(tm.getLabelFont().getFamily(), 10));
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText(truncateLabel(n.name, 20), n.rx + n.rw / 2, n.ry + n.rh / 2 + 4);
        }
    }

    // --- 树图 -------------------------------------------------------------------

    private static final class TreeNode {
        String name = "";
        double px, py;
        double colX;
        int depth;
        final List<TreeNode> children = new ArrayList<>();
    }

    private static final class TreeBounds {
        double minX = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY;
        int maxDepth = 0;

        void merge(TreeNode n) {
            minX = Math.min(minX, n.colX);
            maxX = Math.max(maxX, n.colX);
            maxDepth = Math.max(maxDepth, n.depth);
        }
    }

    private static TreeNode buildTreeRoot(List<Map<String, Object>> raw) {
        if (raw == null || raw.isEmpty()) {
            return null;
        }
        if (raw.size() == 1) {
            return mapToTreeNode(raw.get(0));
        }
        TreeNode virt = new TreeNode();
        virt.name = "";
        for (Map<String, Object> m : raw) {
            virt.children.add(mapToTreeNode(m));
        }
        return virt;
    }

    @SuppressWarnings("unchecked")
    private static TreeNode mapToTreeNode(Map<String, Object> m) {
        TreeNode n = new TreeNode();
        n.name = String.valueOf(m.getOrDefault("name", "?"));
        Object ch = m.get("children");
        if (ch instanceof List<?>) {
            for (Object o : (List<?>) ch) {
                if (o instanceof Map) {
                    n.children.add(mapToTreeNode((Map<String, Object>) o));
                }
            }
        }
        return n;
    }

    private static void assignTreeLeafX(TreeNode n, double[] nextCol) {
        if (n.children.isEmpty()) {
            n.colX = nextCol[0];
            nextCol[0] += 1;
            return;
        }
        for (TreeNode c : n.children) {
            assignTreeLeafX(c, nextCol);
        }
        double sx = 0;
        for (TreeNode c : n.children) {
            sx += c.colX;
        }
        n.colX = sx / Math.max(1, n.children.size());
    }

    private static void setTreeDepth(TreeNode n, int d) {
        n.depth = d;
        for (TreeNode c : n.children) {
            setTreeDepth(c, d + 1);
        }
    }

    private static TreeBounds treeBounds(TreeNode n) {
        TreeBounds b = new TreeBounds();
        collectBounds(n, b);
        return b;
    }

    private static void collectBounds(TreeNode n, TreeBounds b) {
        if (n.children.isEmpty()) {
            b.merge(n);
            return;
        }
        for (TreeNode c : n.children) {
            collectBounds(c, b);
        }
        b.merge(n);
    }

    private static void mapTreeToPixels(TreeNode n, double x0, double y0, double xScale, double yStep) {
        n.px = x0 + n.colX * xScale;
        n.py = y0 + n.depth * yStep;
        for (TreeNode c : n.children) {
            mapTreeToPixels(c, x0, y0, xScale, yStep);
        }
    }

    private void drawTreeEdges(GraphicsContext gc, TreeNode n) {
        for (TreeNode c : n.children) {
            gc.strokeLine(n.px, n.py, c.px, c.py);
            drawTreeEdges(gc, c);
        }
    }

    private void drawTreeNodes(GraphicsContext gc, TreeNode n, String[] palette, JavaFxThemeManager tm) {
        double r = n.children.isEmpty() ? 16 : 20;
        Color base = Color.web(palette[Math.abs(n.name.hashCode()) % palette.length]);
        gc.setFill(new Color(base.getRed(), base.getGreen(), base.getBlue(), 0.9));
        gc.fillOval(n.px - r, n.py - r, r * 2, r * 2);
        gc.setStroke(tm.getTextColor());
        gc.setLineWidth(1.2);
        gc.strokeOval(n.px - r, n.py - r, r * 2, r * 2);
        gc.setFill(tm.getTextColor());
        gc.setFont(Font.font(tm.getLabelFont().getFamily(), 9));
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(truncateLabel(n.name, 10), n.px, n.py + 3);
        for (TreeNode c : n.children) {
            drawTreeNodes(gc, c, palette, tm);
        }
    }

    // --- 力导向图 ---------------------------------------------------------------

    private void fruchtermanReingold(double[] px, double[] py, List<Map<String, Object>> links,
                                     int n, double area) {
        if (n <= 1) {
            return;
        }
        double k = Math.sqrt(Math.max(400, area) / n);
        double[] fx = new double[n];
        double[] fy = new double[n];
        int iterMax = 140 + n * 14;
        for (int iter = 0; iter < iterMax; iter++) {
            Arrays.fill(fx, 0);
            Arrays.fill(fy, 0);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == j) {
                        continue;
                    }
                    double dx = px[j] - px[i];
                    double dy = py[j] - py[i];
                    double dist = Math.sqrt(dx * dx + dy * dy) + 0.08;
                    double rep = k * k / dist;
                    fx[i] -= rep * dx / dist;
                    fy[i] -= rep * dy / dist;
                }
            }
            if (links != null) {
                for (Map<String, Object> L : links) {
                    Integer si = sankeyNodeIndex(L.get("source"));
                    Integer ti = sankeyNodeIndex(L.get("target"));
                    if (si == null || ti == null || si < 0 || ti < 0 || si >= n || ti >= n || si == ti) {
                        continue;
                    }
                    double dx = px[ti] - px[si];
                    double dy = py[ti] - py[si];
                    double dist = Math.sqrt(dx * dx + dy * dy) + 0.08;
                    double att = dist * dist / k;
                    fx[si] += att * dx / dist;
                    fy[si] += att * dy / dist;
                    fx[ti] -= att * dx / dist;
                    fy[ti] -= att * dy / dist;
                }
            }
            double t = iter / (double) iterMax;
            for (int i = 0; i < n; i++) {
                double disp = Math.sqrt(fx[i] * fx[i] + fy[i] * fy[i]) + 1e-9;
                double lim = Math.min(24, k * (0.12 + 0.42 * (1.0 - t * 0.9)));
                lim = Math.max(lim, 0.5);
                double d = Math.min(lim, disp);
                px[i] += fx[i] / disp * d;
                py[i] += fy[i] / disp * d;
            }
        }
    }

    /** 将结点包络矩形等比缩放并平移，铺满绘图区（避免力导向尺度过小）。 */
    private static void fitGraphPositionsToRect(double[] px, double[] py, int n,
                                               double left, double top, double right, double bottom) {
        if (n <= 0) {
            return;
        }
        double minX = px[0], maxX = px[0], minY = py[0], maxY = py[0];
        for (int i = 1; i < n; i++) {
            minX = Math.min(minX, px[i]);
            maxX = Math.max(maxX, px[i]);
            minY = Math.min(minY, py[i]);
            maxY = Math.max(maxY, py[i]);
        }
        double dx = maxX - minX;
        double dy = maxY - minY;
        if (dx < 1e-6) {
            dx = 1;
        }
        if (dy < 1e-6) {
            dy = 1;
        }
        double targetW = right - left;
        double targetH = bottom - top;
        double scale = Math.min(targetW / dx, targetH / dy) * 0.9;
        double mx = (minX + maxX) / 2;
        double my = (minY + maxY) / 2;
        double tcx = (left + right) / 2;
        double tcy = (top + bottom) / 2;
        for (int i = 0; i < n; i++) {
            px[i] = tcx + (px[i] - mx) * scale;
            py[i] = tcy + (py[i] - my) * scale;
        }
    }

    private static void collectTreePixelBounds(TreeNode n, double[] b) {
        b[0] = Math.min(b[0], n.px);
        b[1] = Math.max(b[1], n.px);
        b[2] = Math.min(b[2], n.py);
        b[3] = Math.max(b[3], n.py);
        for (TreeNode c : n.children) {
            collectTreePixelBounds(c, b);
        }
    }

    private static void shiftTreeBy(TreeNode n, double dx, double dy) {
        n.px += dx;
        n.py += dy;
        for (TreeNode c : n.children) {
            shiftTreeBy(c, dx, dy);
        }
    }

    private void centerTreeInPlotBounds(TreeNode root, ChartConfig config) {
        double[] b = {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        collectTreePixelBounds(root, b);
        double plotMidX = config.paddingLeft + (config.width - config.paddingLeft - config.paddingRight) / 2.0;
        double plotMidY = config.paddingTop + 52 + (config.height - config.paddingTop - config.paddingBottom - 56) / 2.0;
        double cx = (b[0] + b[1]) / 2;
        double cy = (b[2] + b[3]) / 2;
        shiftTreeBy(root, plotMidX - cx, plotMidY - cy);
    }

    /**
     * 渲染平行坐标图：每维独立数值域与刻度标签；首列刻度文字在轴左，其余在轴右，减少与折线重叠。
     */
    @SuppressWarnings("unchecked")
    private void renderParallel(GraphicsContext gc, SeriesData series,
                                 ChartConfig config, JavaFxThemeManager themeManager) {
        String[] palette = themeManager.getColorPalette();
        if (palette == null || palette.length == 0) {
            palette = new String[] {"#4878d0"};
        }

        IMatrix<Double> matrix = (IMatrix<Double>) series.extraData.get("matrixData");
        List<String> dimList = (List<String>) series.extraData.get("dimensions");

        int numDimensions = 5;
        String[] dimensions = new String[] {"Dim1", "Dim2", "Dim3", "Dim4", "Dim5"};
        boolean demoData = matrix == null || matrix.getColNum() <= 0 || matrix.getRowNum() <= 0;

        if (!demoData) {
            numDimensions = matrix.getColNum();
            dimensions = new String[numDimensions];
            for (int i = 0; i < numDimensions; i++) {
                if (dimList != null && i < dimList.size() && dimList.get(i) != null) {
                    dimensions[i] = dimList.get(i);
                } else {
                    dimensions[i] = "Dim" + (i + 1);
                }
            }
        }

        double chartLeft = config.paddingLeft;
        double chartRight = config.width - config.paddingRight;
        double chartTop = config.paddingTop + 34;
        final double dimLabelReserve = 54;
        double chartBottom = config.height - config.paddingBottom - dimLabelReserve;
        double plotH = Math.max(40, chartBottom - chartTop);

        Color axisColor = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("axisColor"), Color.web("#64748b"));
        Color gridColor = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("gridColor"), Color.web("#94a3b8"));
        boolean showGrid = config.showGrid;

        double[] dimensionX = new double[numDimensions];
        if (numDimensions == 1) {
            dimensionX[0] = (chartLeft + chartRight) / 2;
        } else {
            for (int i = 0; i < numDimensions; i++) {
                dimensionX[i] = chartLeft + (chartRight - chartLeft) * i / (numDimensions - 1);
            }
        }

        double[] dmin = new double[numDimensions];
        double[] dmax = new double[numDimensions];
        if (!demoData) {
            for (int d = 0; d < numDimensions; d++) {
                double lo = Double.POSITIVE_INFINITY;
                double hi = Double.NEGATIVE_INFINITY;
                for (int r = 0; r < matrix.getRowNum(); r++) {
                    double v = matrix.get(r, d);
                    lo = Math.min(lo, v);
                    hi = Math.max(hi, v);
                }
                if (hi - lo < 1e-15) {
                    double pad = Math.abs(lo) > 1e-6 ? Math.abs(lo) * 0.06 : 0.5;
                    lo -= pad;
                    hi += pad;
                }
                dmin[d] = lo;
                dmax[d] = hi;
            }
        } else {
            for (int d = 0; d < numDimensions; d++) {
                dmin[d] = 0;
                dmax[d] = 1;
            }
        }

        final int numTickSteps = 5;
        if (showGrid) {
            gc.save();
            gc.setStroke(gridColor);
            gc.setLineWidth(0.75);
            gc.setGlobalAlpha(0.4);
            for (int j = 0; j <= numTickSteps; j++) {
                double y = chartBottom - plotH * j / numTickSteps;
                gc.strokeLine(chartLeft, y, chartRight, y);
            }
            gc.restore();
        }

        if (demoData) {
            java.util.concurrent.ThreadLocalRandom rnd = java.util.concurrent.ThreadLocalRandom.current();
            for (int i = 0; i < 20; i++) {
                Color c = Color.web(palette[i % palette.length]);
                gc.setStroke(new Color(c.getRed(), c.getGreen(), c.getBlue(), 0.56));
                gc.setLineWidth(1.2);
                gc.beginPath();
                for (int d = 0; d < numDimensions; d++) {
                    double val = dmin[d] + (dmax[d] - dmin[d]) * rnd.nextDouble();
                    double y = parallelValueToY(val, dmin[d], dmax[d], chartTop, chartBottom);
                    if (d == 0) {
                        gc.moveTo(dimensionX[d], y);
                    } else {
                        gc.lineTo(dimensionX[d], y);
                    }
                }
                gc.stroke();
            }
        } else {
            for (int r = 0; r < matrix.getRowNum(); r++) {
                Color c = Color.web(palette[r % palette.length]);
                gc.setStroke(new Color(c.getRed(), c.getGreen(), c.getBlue(), 0.58));
                gc.setLineWidth(1.25);
                gc.beginPath();
                for (int d = 0; d < numDimensions; d++) {
                    double val = matrix.get(r, d);
                    double y = parallelValueToY(val, dmin[d], dmax[d], chartTop, chartBottom);
                    if (d == 0) {
                        gc.moveTo(dimensionX[d], y);
                    } else {
                        gc.lineTo(dimensionX[d], y);
                    }
                }
                gc.stroke();
            }
        }

        final double tickHalf = 5;
        Font tickFont = Font.font(themeManager.getLabelFont().getFamily(), 9);
        Font dimFont = Font.font(themeManager.getLabelFont().getFamily(), 12);
        Color muted = themeManager.getMutedTextColor();

        gc.setLineWidth(1);
        for (int d = 0; d < numDimensions; d++) {
            gc.setStroke(axisColor);
            gc.strokeLine(dimensionX[d], chartTop, dimensionX[d], chartBottom);
            for (int j = 0; j <= numTickSteps; j++) {
                double y = chartBottom - plotH * j / numTickSteps;
                double v = dmin[d] + (dmax[d] - dmin[d]) * j / numTickSteps;
                gc.setStroke(axisColor);
                gc.strokeLine(dimensionX[d] - tickHalf, y, dimensionX[d] + tickHalf, y);
                gc.setFont(tickFont);
                gc.setFill(muted);
                String lab = JavaFxChartUtils.formatNumber(v);
                if (d == 0) {
                    gc.setTextAlign(TextAlignment.RIGHT);
                    gc.fillText(lab, dimensionX[d] - 11, y + 4);
                } else {
                    gc.setTextAlign(TextAlignment.LEFT);
                    gc.fillText(lab, dimensionX[d] + 11, y + 4);
                }
            }
        }

        gc.setFill(themeManager.getTextColor());
        gc.setFont(dimFont);
        gc.setTextAlign(TextAlignment.CENTER);
        double dimNameY = config.height - config.paddingBottom - 12;
        for (int d = 0; d < numDimensions; d++) {
            gc.fillText(dimensions[d], dimensionX[d], dimNameY);
        }
    }

    /** 数据值映射到画布 Y（自下向上为 min→max）。 */
    private static double parallelValueToY(double v, double lo, double hi,
                                          double chartTop, double chartBottom) {
        if (!(hi > lo)) {
            return (chartTop + chartBottom) / 2;
        }
        double u = (v - lo) / (hi - lo);
        u = Math.min(1, Math.max(0, u));
        return chartBottom - u * (chartBottom - chartTop);
    }

    /**
     * 主题河流图：按时间（列表顺序）堆叠面积，数据行为各类别取值（Map 键为类别名）
     */
    @SuppressWarnings("unchecked")
    private void renderThemeRiver(GraphicsContext gc, SeriesData series,
                                  ChartConfig config, JavaFxThemeManager themeManager) {
        List<Map<String, Object>> riverData =
            (List<Map<String, Object>>) series.extraData.get("riverData");
        List<String> categories = (List<String>) series.extraData.get("categories");
        if (riverData == null || riverData.isEmpty()) {
            return;
        }

        List<String> cats = categories;
        if (cats == null || cats.isEmpty()) {
            cats = new java.util.ArrayList<>(riverData.get(0).keySet());
        }

        int tCount = riverData.size();
        int cCount = cats.size();
        if (cCount == 0) {
            return;
        }

        double[][] values = new double[tCount][cCount];
        for (int t = 0; t < tCount; t++) {
            Map<String, Object> row = riverData.get(t);
            for (int c = 0; c < cCount; c++) {
                Object v = row.get(cats.get(c));
                values[t][c] = v instanceof Number ? ((Number) v).doubleValue() : 0;
            }
        }

        double maxStack = 0;
        for (int t = 0; t < tCount; t++) {
            double sum = 0;
            for (int c = 0; c < cCount; c++) {
                sum += values[t][c];
            }
            maxStack = Math.max(maxStack, sum);
        }
        if (maxStack <= 0) {
            maxStack = 1;
        }

        double chartLeft = config.paddingLeft;
        double chartRight = config.width - config.paddingRight;
        double chartTop = config.paddingTop + 40;
        double chartBottom = config.height - config.paddingBottom;
        double chartW = chartRight - chartLeft;
        double chartH = chartBottom - chartTop;

        String[] palette = themeManager.getColorPalette();

        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        JavaFxChartUtils.drawTitle(gc, config, themeManager);

        double yScale = chartH / maxStack;
        double[] lower = new double[tCount];
        for (int t = 0; t < tCount; t++) {
            lower[t] = chartBottom;
        }

        for (int c = 0; c < cCount; c++) {
            double[] upper = new double[tCount];
            for (int t = 0; t < tCount; t++) {
                upper[t] = lower[t] - values[t][c] * yScale;
            }

            gc.setFill(Color.web(palette[c % palette.length], 0.75));
            gc.beginPath();
            for (int t = 0; t < tCount; t++) {
                double x = chartLeft + chartW * (tCount <= 1 ? 0.5 : t / (double) (tCount - 1));
                if (t == 0) {
                    gc.moveTo(x, upper[t]);
                } else {
                    gc.lineTo(x, upper[t]);
                }
            }
            for (int t = tCount - 1; t >= 0; t--) {
                double x = chartLeft + chartW * (tCount <= 1 ? 0.5 : t / (double) (tCount - 1));
                gc.lineTo(x, lower[t]);
            }
            gc.closePath();
            gc.fill();
            lower = upper;
        }

        gc.setStroke(themeManager.getTextColor());
        gc.setLineWidth(2);
        gc.strokeLine(chartLeft, chartTop, chartLeft, chartBottom);
        gc.strokeLine(chartLeft, chartBottom, chartRight, chartBottom);

        gc.setFill(themeManager.getTextColor());
        gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 10));
        gc.setTextAlign(TextAlignment.CENTER);
        for (int t = 0; t < tCount; t++) {
            double x = chartLeft + chartW * (tCount <= 1 ? 0.5 : t / (double) (tCount - 1));
            gc.fillText(String.valueOf(t), x, chartBottom + 18);
        }

        double legX = chartRight - 120;
        double legY = chartTop + 10;
        for (int c = 0; c < cCount; c++) {
            gc.setFill(Color.web(palette[c % palette.length]));
            gc.fillRect(legX, legY + c * 16, 12, 12);
            gc.setFill(themeManager.getTextColor());
            gc.setTextAlign(TextAlignment.LEFT);
            gc.fillText(cats.get(c), legX + 16, legY + c * 16 + 10);
        }
    }
    
    @Override
    public String getChartType() {
        return chartType;
    }
    
    @Override
    public boolean supportsAnimation() {
        return true;
    }
    
    @Override
    public int getAnimationDuration() {
        return 1000;
    }
}
