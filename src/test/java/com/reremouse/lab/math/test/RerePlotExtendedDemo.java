package com.reremouse.lab.math.test;

import com.reremouse.lab.math.RereVector;
import com.reremouse.lab.math.RereMatrix;
import com.reremouse.lab.math.viz.Plots;
import com.reremouse.lab.math.viz.IPlot;
import java.util.*;

/**
 * IPlot扩展功能演示
 * 展示所有新增的ECharts-Java图表类型
 * @author lteb2
 */
public class RerePlotExtendedDemo {
    
    public static void main(String[] args) {
        System.out.println("=== IPlot扩展功能演示 ===");
        
        // 创建测试数据
        RereVector data1 = new RereVector(new float[]{10, 20, 30, 40, 50});
        RereVector data2 = new RereVector(new float[]{15, 25, 35, 45, 55});
        RereVector data3 = new RereVector(new float[]{5, 15, 25, 35, 45});
        
        List<String> categories = Arrays.asList("类别A", "类别B", "类别C", "类别D", "类别E");
        List<String> labels = Arrays.asList("数据1", "数据2", "数据3", "数据4", "数据5");
        
        IPlot plot = Plots.of(800, 600);
        plot.setTitle("IPlot扩展功能演示");
        
        // 1. 极坐标图表
        System.out.println("1. 测试极坐标柱状图...");
        plot.polarBar(data1, categories);
        plot.saveAsHtml("polar_bar_demo.html");
        
        System.out.println("2. 测试极坐标线图...");
        plot.polarLine(data2, categories);
        plot.saveAsHtml("polar_line_demo.html");
        
        System.out.println("3. 测试极坐标散点图...");
        plot.polarScatter(data3, categories);
        plot.saveAsHtml("polar_scatter_demo.html");
        
        // 2. 统计图表
        System.out.println("4. 测试箱线图...");
        plot.boxplot(data1, labels);
        plot.saveAsHtml("boxplot_demo.html");
        
        // 创建K线图数据
        float[][] candlestickArray = {
            {100, 110, 95, 115},
            {110, 120, 105, 125},
            {120, 115, 110, 130},
            {115, 125, 110, 135},
            {125, 130, 120, 140}
        };
        RereMatrix candlestickData = new RereMatrix(candlestickArray);
        
        List<String> dates = Arrays.asList("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05");
        
        System.out.println("5. 测试K线图...");
        plot.candlestick(candlestickData, dates);
        plot.saveAsHtml("candlestick_demo.html");
        
        // 3. 特殊图表
        System.out.println("6. 测试漏斗图...");
        plot.funnel(data1, categories);
        plot.saveAsHtml("funnel_demo.html");
        
        // 创建桑基图数据
        List<Map<String, Object>> nodes = new ArrayList<>();
        nodes.add(createNode("source1", "源1"));
        nodes.add(createNode("source2", "源2"));
        nodes.add(createNode("target1", "目标1"));
        nodes.add(createNode("target2", "目标2"));
        
        List<Map<String, Object>> links = new ArrayList<>();
        links.add(createLink("source1", "target1", 10));
        links.add(createLink("source1", "target2", 20));
        links.add(createLink("source2", "target1", 15));
        links.add(createLink("source2", "target2", 25));
        
        System.out.println("7. 测试桑基图...");
        plot.sankey(nodes, links);
        plot.saveAsHtml("sankey_demo.html");
        
        // 创建旭日图数据
        List<Map<String, Object>> sunburstData = new ArrayList<>();
        sunburstData.add(createSunburstNode("root", "根节点", 100));
        sunburstData.add(createSunburstNode("child1", "子节点1", 60, "root"));
        sunburstData.add(createSunburstNode("child2", "子节点2", 40, "root"));
        sunburstData.add(createSunburstNode("grandchild1", "孙节点1", 30, "child1"));
        sunburstData.add(createSunburstNode("grandchild2", "孙节点2", 30, "child1"));
        
        System.out.println("8. 测试旭日图...");
        plot.sunburst(sunburstData);
        plot.saveAsHtml("sunburst_demo.html");
        
        // 创建主题河流图数据
        List<Map<String, Object>> themeRiverData = new ArrayList<>();
        themeRiverData.add(createThemeRiverNode("2024-01-01", "类别A", 10));
        themeRiverData.add(createThemeRiverNode("2024-01-01", "类别B", 20));
        themeRiverData.add(createThemeRiverNode("2024-01-02", "类别A", 15));
        themeRiverData.add(createThemeRiverNode("2024-01-02", "类别B", 25));
        
        List<String> riverCategories = Arrays.asList("类别A", "类别B");
        
        System.out.println("9. 测试主题河流图...");
        plot.themeRiver(themeRiverData, riverCategories);
        plot.saveAsHtml("theme_river_demo.html");
        
        // 创建树图数据
        List<Map<String, Object>> treeData = new ArrayList<>();
        treeData.add(createTreeNode("root", "根节点"));
        treeData.add(createTreeNode("child1", "子节点1", "root"));
        treeData.add(createTreeNode("child2", "子节点2", "root"));
        treeData.add(createTreeNode("grandchild1", "孙节点1", "child1"));
        
        System.out.println("10. 测试树图...");
        plot.tree(treeData);
        plot.saveAsHtml("tree_demo.html");
        
        // 创建矩形树图数据
        List<Map<String, Object>> treemapData = new ArrayList<>();
        treemapData.add(createTreemapNode("root", "根节点", 100));
        treemapData.add(createTreemapNode("child1", "子节点1", 60, "root"));
        treemapData.add(createTreemapNode("child2", "子节点2", 40, "root"));
        
        System.out.println("11. 测试矩形树图...");
        plot.treemap(treemapData);
        plot.saveAsHtml("treemap_demo.html");
        
        // 创建关系图数据
        List<Map<String, Object>> graphNodes = new ArrayList<>();
        graphNodes.add(createGraphNode("node1", "节点1"));
        graphNodes.add(createGraphNode("node2", "节点2"));
        graphNodes.add(createGraphNode("node3", "节点3"));
        
        List<Map<String, Object>> graphLinks = new ArrayList<>();
        graphLinks.add(createGraphLink("node1", "node2", 10));
        graphLinks.add(createGraphLink("node2", "node3", 15));
        graphLinks.add(createGraphLink("node1", "node3", 20));
        
        System.out.println("12. 测试关系图...");
        plot.graph(graphNodes, graphLinks);
        plot.saveAsHtml("graph_demo.html");
        
        // 创建平行坐标图数据
        float[][] parallelArray = {
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6}
        };
        RereMatrix parallelData = new RereMatrix(parallelArray);
        
        List<String> dimensions = Arrays.asList("维度1", "维度2", "维度3", "维度4");
        
        System.out.println("13. 测试平行坐标图...");
        plot.parallel(parallelData, dimensions);
        plot.saveAsHtml("parallel_demo.html");
        
        // 4. 现有图表完善
        System.out.println("14. 测试热力图...");
        float[][] heatmapArray = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        RereMatrix heatmapData = new RereMatrix(heatmapArray);
        
        List<String> xLabels = Arrays.asList("X1", "X2", "X3");
        List<String> yLabels = Arrays.asList("Y1", "Y2", "Y3");
        
        plot.heatmap(heatmapData, xLabels, yLabels);
        plot.saveAsHtml("heatmap_demo.html");
        
        System.out.println("15. 测试雷达图...");
        List<String> indicators = Arrays.asList("指标1", "指标2", "指标3", "指标4", "指标5");
        plot.radar(data1, indicators);
        plot.saveAsHtml("radar_demo.html");
        
        System.out.println("16. 测试仪表盘...");
        plot.gauge(75, 100, 0);
        plot.saveAsHtml("gauge_demo.html");
        
        System.out.println("=== 所有图表演示完成 ===");
        System.out.println("生成的HTML文件：");
        System.out.println("- polar_bar_demo.html");
        System.out.println("- polar_line_demo.html");
        System.out.println("- polar_scatter_demo.html");
        System.out.println("- boxplot_demo.html");
        System.out.println("- candlestick_demo.html");
        System.out.println("- funnel_demo.html");
        System.out.println("- sankey_demo.html");
        System.out.println("- sunburst_demo.html");
        System.out.println("- theme_river_demo.html");
        System.out.println("- tree_demo.html");
        System.out.println("- treemap_demo.html");
        System.out.println("- graph_demo.html");
        System.out.println("- parallel_demo.html");
        System.out.println("- heatmap_demo.html");
        System.out.println("- radar_demo.html");
        System.out.println("- gauge_demo.html");
    }
    
    // 辅助方法
    private static Map<String, Object> createNode(String id, String name) {
        Map<String, Object> node = new HashMap<>();
        node.put("id", id);
        node.put("name", name);
        return node;
    }
    
    private static Map<String, Object> createLink(String source, String target, int value) {
        Map<String, Object> link = new HashMap<>();
        link.put("source", source);
        link.put("target", target);
        link.put("value", value);
        return link;
    }
    
    private static Map<String, Object> createSunburstNode(String id, String name, int value) {
        Map<String, Object> node = new HashMap<>();
        node.put("id", id);
        node.put("name", name);
        node.put("value", value);
        return node;
    }
    
    private static Map<String, Object> createSunburstNode(String id, String name, int value, String parent) {
        Map<String, Object> node = createSunburstNode(id, name, value);
        node.put("parent", parent);
        return node;
    }
    
    private static Map<String, Object> createThemeRiverNode(String date, String category, int value) {
        Map<String, Object> node = new HashMap<>();
        node.put("date", date);
        node.put("category", category);
        node.put("value", value);
        return node;
    }
    
    private static Map<String, Object> createTreeNode(String id, String name) {
        Map<String, Object> node = new HashMap<>();
        node.put("id", id);
        node.put("name", name);
        return node;
    }
    
    private static Map<String, Object> createTreeNode(String id, String name, String parent) {
        Map<String, Object> node = createTreeNode(id, name);
        node.put("parent", parent);
        return node;
    }
    
    private static Map<String, Object> createTreemapNode(String id, String name, int value) {
        Map<String, Object> node = new HashMap<>();
        node.put("id", id);
        node.put("name", name);
        node.put("value", value);
        return node;
    }
    
    private static Map<String, Object> createTreemapNode(String id, String name, int value, String parent) {
        Map<String, Object> node = createTreemapNode(id, name, value);
        node.put("parent", parent);
        return node;
    }
    
    private static Map<String, Object> createGraphNode(String id, String name) {
        Map<String, Object> node = new HashMap<>();
        node.put("id", id);
        node.put("name", name);
        return node;
    }
    
    private static Map<String, Object> createGraphLink(String source, String target, int value) {
        Map<String, Object> link = new HashMap<>();
        link.put("source", source);
        link.put("target", target);
        link.put("value", value);
        return link;
    }
}
