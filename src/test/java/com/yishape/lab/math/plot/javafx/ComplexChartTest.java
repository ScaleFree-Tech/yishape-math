package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 复杂图表测试类（漏斗图、桑基图、旭日图、矩形树图、树图、关系图、平行坐标图）
 * 
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ComplexChartTest extends JavaFxChartTestBase {

    private static Map<String, Object> treeNode(String name, List<Map<String, Object>> children) {
        Map<String, Object> m = new HashMap<>();
        m.put("name", name);
        if (children != null && !children.isEmpty()) {
            m.put("children", children);
        }
        return m;
    }

    private static void graphLink(List<Map<String, Object>> links, int s, int t) {
        Map<String, Object> L = new HashMap<>();
        L.put("source", s);
        L.put("target", t);
        L.put("value", 1);
        links.add(L);
    }
    
    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 复杂图表测试开始 ==========");
    }
    
    @Test
    @Order(1)
    @DisplayName("测试漏斗图")
    void testFunnel() {
        System.out.println("\n[测试1] 漏斗图");
        
        double[] data = {100, 80, 60, 40, 20};
        List<String> labels = List.of("访问", "点击", "咨询", "订单", "成交");
        
        IPlot plot = Plots.of(800, 600);
        plot.funnel(Linalg.vector(data), labels)
            .title("漏斗图", "销售转化漏斗");
        
        boolean success = generateImage(plot, "complex_01_funnel.png");
        Assertions.assertTrue(success, "漏斗图图片生成失败");
    }
    
    @Test
    @Order(2)
    @DisplayName("测试桑基图")
    void testSankey() {
        System.out.println("\n[测试2] 桑基图");
        
        List<Map<String, Object>> nodes = new ArrayList<>();
        Map<String, Object> node1 = new HashMap<>();
        node1.put("name", "来源A");
        nodes.add(node1);
        Map<String, Object> node2 = new HashMap<>();
        node2.put("name", "来源B");
        nodes.add(node2);
        Map<String, Object> node3 = new HashMap<>();
        node3.put("name", "来源C");
        nodes.add(node3);
        Map<String, Object> node4 = new HashMap<>();
        node4.put("name", "目标");
        nodes.add(node4);
        
        List<Map<String, Object>> links = new ArrayList<>();
        Map<String, Object> link1 = new HashMap<>();
        link1.put("source", 0);
        link1.put("target", 3);
        link1.put("value", 30);
        links.add(link1);
        Map<String, Object> link2 = new HashMap<>();
        link2.put("source", 1);
        link2.put("target", 3);
        link2.put("value", 20);
        links.add(link2);
        Map<String, Object> link3 = new HashMap<>();
        link3.put("source", 2);
        link3.put("target", 3);
        link3.put("value", 15);
        links.add(link3);
        
        IPlot plot = Plots.of(800, 600);
        plot.sankey(nodes, links)
            .title("桑基图", "流量来源分析");
        
        boolean success = generateImage(plot, "complex_02_sankey.png");
        Assertions.assertTrue(success, "桑基图图片生成失败");
    }
    
    @Test
    @Order(3)
    @DisplayName("测试旭日图")
    void testSunburst() {
        System.out.println("\n[测试3] 旭日图");
        
        List<Map<String, Object>> data = new ArrayList<>();
        Map<String, Object> item1 = new HashMap<>();
        item1.put("name", "A");
        item1.put("value", 40);
        data.add(item1);
        Map<String, Object> item2 = new HashMap<>();
        item2.put("name", "B");
        item2.put("value", 35);
        data.add(item2);
        Map<String, Object> item3 = new HashMap<>();
        item3.put("name", "C");
        item3.put("value", 25);
        data.add(item3);
        
        IPlot plot = Plots.of(800, 600);
        plot.sunburst(data)
            .title("旭日图");
        
        boolean success = generateImage(plot, "complex_03_sunburst.png");
        Assertions.assertTrue(success, "旭日图图片生成失败");
    }
    
    @Test
    @Order(4)
    @DisplayName("测试矩形树图")
    void testTreemap() {
        System.out.println("\n[测试4] 矩形树图");
        
        List<Map<String, Object>> data = new ArrayList<>();
        Map<String, Object> item1 = new HashMap<>();
        item1.put("name", "Category A");
        item1.put("value", 60);
        data.add(item1);
        Map<String, Object> item2 = new HashMap<>();
        item2.put("name", "Category B");
        item2.put("value", 30);
        data.add(item2);
        Map<String, Object> item3 = new HashMap<>();
        item3.put("name", "Category C");
        item3.put("value", 10);
        data.add(item3);
        
        IPlot plot = Plots.of(800, 600);
        plot.treemap(data)
            .title("矩形树图", "分类占比");
        
        boolean success = generateImage(plot, "complex_04_treemap.png");
        Assertions.assertTrue(success, "矩形树图图片生成失败");
    }
    
    @Test
    @Order(5)
    @DisplayName("测试树图")
    void testTree() {
        System.out.println("\n[测试5] 树图");
        
        List<Map<String, Object>> mobileKids = List.of(
            treeNode("购物 APP", null),
            treeNode("微信小程序", null));
        Map<String, Object> mobile = treeNode("移动端", new ArrayList<>(mobileKids));

        List<Map<String, Object>> desktopKids = List.of(
            treeNode("Web 商城", null),
            treeNode("商家后台", null));
        Map<String, Object> desktop = treeNode("桌面端", new ArrayList<>(desktopKids));

        List<Map<String, Object>> cloudKids = List.of(
            treeNode("订单编排", null),
            treeNode("库存同步", null),
            treeNode("对账服务", null));
        Map<String, Object> cloud = treeNode("云服务 / 中台", new ArrayList<>(cloudKids));

        List<Map<String, Object>> top = new ArrayList<>();
        top.add(mobile);
        top.add(desktop);
        top.add(cloud);
        Map<String, Object> root = treeNode("电商平台 · 技术域", top);

        List<Map<String, Object>> data = new ArrayList<>();
        data.add(root);
        
        IPlot plot = Plots.of(800, 600);
        plot.tree(data)
            .title("树图", "三层：域 → 渠道 / 服务 → 叶子能力");
        
        boolean success = generateImage(plot, "complex_05_tree.png");
        Assertions.assertTrue(success, "树图图片生成失败");
    }
    
    @Test
    @Order(6)
    @DisplayName("测试关系图")
    void testGraph() {
        System.out.println("\n[测试6] 关系图");

        String[] graphNames = {
            "API 网关", "用户中心", "订单服务", "商品目录", "支付渠道", "仓储库存", "推荐引擎"
        };
        List<Map<String, Object>> nodes = new ArrayList<>();
        for (String nm : graphNames) {
            Map<String, Object> node = new HashMap<>();
            node.put("name", nm);
            nodes.add(node);
        }

        List<Map<String, Object>> links = new ArrayList<>();
        graphLink(links, 0, 1);
        graphLink(links, 0, 2);
        graphLink(links, 0, 3);
        graphLink(links, 0, 6);
        graphLink(links, 1, 2);
        graphLink(links, 2, 4);
        graphLink(links, 2, 5);
        graphLink(links, 3, 5);
        graphLink(links, 4, 5);
        graphLink(links, 5, 6);

        IPlot plot = Plots.of(800, 600);
        plot.graph(nodes, links)
            .title("关系图", "微服务依赖示意 · 7 结点 10 边");
        
        boolean success = generateImage(plot, "complex_06_graph.png");
        Assertions.assertTrue(success, "关系图图片生成失败");
    }
    
    @Test
    @Order(7)
    @DisplayName("测试平行坐标图")
    void testParallel() {
        System.out.println("\n[测试7] 平行坐标图");
        
        double[][] data = {
            {1, 2, 3, 4, 5},
            {2, 3, 4, 5, 6},
            {3, 4, 5, 6, 7},
            {3, 2, 5, 4, 6}
        };
        IMatrix<Double> matrix = Linalg.matrix(data);
        List<String> dimensions = List.of("Dim1", "Dim2", "Dim3", "Dim4", "Dim5");
        
        IPlot plot = Plots.of(800, 600);
        plot.parallel(matrix, dimensions)
            .title("平行坐标图");
        
        boolean success = generateImage(plot, "complex_07_parallel.png");
        Assertions.assertTrue(success, "平行坐标图图片生成失败");
    }
    
    @Test
    @Order(8)
    @DisplayName("测试不同主题复杂图表")
    void testComplexChartsWithThemes() {
        System.out.println("\n[测试8] 不同主题复杂图表");
        
        double[] funnelData = {100, 75, 50, 25};
        List<String> funnelLabels = List.of("A", "B", "C", "D");
        String[] themes = {"default", "dark", "vintage"};
        
        int successCount = 0;
        for (String theme : themes) {
            IPlot plot = Plots.of(800, 600, theme);
            plot.funnel(Linalg.vector(funnelData), funnelLabels)
                .title("漏斗图 - " + theme);
            
            if (generateImage(plot, "complex_08_theme_" + theme + ".png")) {
                successCount++;
            }
        }
        
        Assertions.assertTrue(successCount >= 2, "至少应生成2个主题图片");
    }
    
    @AfterAll
    static void tearDown() {
        printSummary("复杂图表测试");
    }
}
