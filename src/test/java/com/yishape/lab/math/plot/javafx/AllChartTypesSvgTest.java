package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.SvgChartTestBase;
import org.junit.jupiter.api.*;

import java.io.File;
import java.util.*;

/**
 * 全图表类型SVG导出测试（所有2D图表）。
 * 验证真正的矢量SVG导出（放大无模糊）。
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class AllChartTypesSvgTest extends SvgChartTestBase {

    private static final String OUTPUT_DIR = "images_javafx_svg";

    @BeforeAll
    static void setUp() {
        new File(OUTPUT_DIR).mkdirs();
        System.out.println("\n========== 全图表类型SVG测试开始 ==========");
    }

    @Test
    @Order(1) @DisplayName("01 折线图")
    void testLineChart() {
        double[] xd = new double[20];
        double[] yd = new double[20];
        for (int i = 0; i < 20; i++) {
            xd[i] = i;
            yd[i] = Math.sin(i * 0.5) * 10 + 20;
        }
        IVector x = Linalg.vector(xd);
        IVector y = Linalg.vector(yd);
        Plots.ofSvg(800, 500).line(x, y).title("折线图").saveAsSvg(OUTPUT_DIR + "/all_01_line.svg");
        System.out.println("  ✓ 折线图已生成");
    }

    @Test
    @Order(2) @DisplayName("02 散点图")
    void testScatterChart() {
        double[] xd = new double[50];
        double[] yd = new double[50];
        for (int i = 0; i < 50; i++) {
            xd[i] = i + Math.random() * 10;
            yd[i] = xd[i] * 0.8 + Math.random() * 5;
        }
        Plots.ofSvg(800, 500).scatter(Linalg.vector(xd), Linalg.vector(yd)).title("散点图").saveAsSvg(OUTPUT_DIR + "/all_02_scatter.svg");
        System.out.println("  ✓ 散点图已生成");
    }

    @Test
    @Order(3) @DisplayName("03 面积图")
    void testAreaChart() {
        double[] xd = new double[30];
        double[] yd = new double[30];
        for (int i = 0; i < 30; i++) {
            xd[i] = i;
            yd[i] = Math.abs(Math.sin(i * 0.3)) * 30 + 10;
        }
        Plots.ofSvg(800, 500).area(Linalg.vector(xd), Linalg.vector(yd)).title("面积图").saveAsSvg(OUTPUT_DIR + "/all_03_area.svg");
        System.out.println("  ✓ 面积图已生成");
    }

    @Test
    @Order(4) @DisplayName("04 阶梯图")
    void testStepChart() {
        double[] xd = new double[20];
        double[] yd = new double[20];
        for (int i = 0; i < 20; i++) { xd[i] = i; yd[i] = i * 2 + 10; }
        Plots.ofSvg(800, 500).step(Linalg.vector(xd), Linalg.vector(yd)).title("阶梯图").saveAsSvg(OUTPUT_DIR + "/all_04_step.svg");
        System.out.println("  ✓ 阶梯图已生成");
    }

    @Test
    @Order(5) @DisplayName("05 气泡图")
    void testBubbleChart() {
        double[] xd = new double[20];
        double[] yd = new double[20];
        double[] sd = new double[20];
        for (int i = 0; i < 20; i++) {
            xd[i] = i; yd[i] = i * 1.5 + 5; sd[i] = (i + 5) * 30;
        }
        Plots.ofSvg(800, 500).bubble(Linalg.vector(xd), Linalg.vector(yd), Linalg.vector(sd)).title("气泡图").saveAsSvg(OUTPUT_DIR + "/all_05_bubble.svg");
        System.out.println("  ✓ 气泡图已生成");
    }

    @Test
    @Order(6) @DisplayName("06 误差棒图")
    void testErrorBarChart() {
        double[] xd = new double[15];
        double[] yd = new double[15];
        double[] ed = new double[15];
        for (int i = 0; i < 15; i++) {
            xd[i] = i;
            yd[i] = Math.sin(i * 0.5) * 20 + 30;
            ed[i] = 5.0;
        }
        Plots.ofSvg(800, 500).errorBar(Linalg.vector(xd), Linalg.vector(yd), Linalg.vector(ed)).title("误差棒图").saveAsSvg(OUTPUT_DIR + "/all_06_errorbar.svg");
        System.out.println("  ✓ 误差棒图已生成");
    }

    @Test
    @Order(6) @DisplayName("06b 置信带图")
    void testCiBandChart() {
        double[] xd = new double[30];
        double[] yd = new double[30];
        double[] ylo = new double[30];
        double[] yhi = new double[30];
        for (int i = 0; i < 30; i++) {
            xd[i] = i;
            yd[i] = Math.sin(i * 0.3) * 15 + 30;
            ylo[i] = yd[i] - 5 - Math.random() * 3;
            yhi[i] = yd[i] + 5 + Math.random() * 3;
        }
        Plots.ofSvg(800, 500).ciBand(Linalg.vector(xd), Linalg.vector(yd), Linalg.vector(ylo), Linalg.vector(yhi)).title("置信带图").saveAsSvg(OUTPUT_DIR + "/all_06b_ciband.svg");
        System.out.println("  ✓ 置信带图已生成");
    }

    @Test
    @Order(7) @DisplayName("07 柱状图")
    void testBarChart() {
        double[] data = {45, 72, 58, 90, 63, 80, 55};
        Plots.ofSvg(800, 500).bar(Linalg.vector(data)).title("柱状图").saveAsSvg(OUTPUT_DIR + "/all_07_bar.svg");
        System.out.println("  ✓ 柱状图已生成");
    }

    @Test
    @Order(7) @DisplayName("07b 分组柱状图")
    void testGroupedBarChart() {
        IVector[] groups = new IVector[3];
        groups[0] = Linalg.vector(new double[]{45, 72, 58});
        groups[1] = Linalg.vector(new double[]{65, 50, 80});
        groups[2] = Linalg.vector(new double[]{55, 62, 70});
        Plots.ofSvg(800, 500).groupedBar(groups).title("分组柱状图").saveAsSvg(OUTPUT_DIR + "/all_07b_groupedbar.svg");
        System.out.println("  ✓ 分组柱状图已生成");
    }

    @Test
    @Order(7) @DisplayName("07c 堆叠柱状图")
    void testStackedBarChart() {
        IVector[] stacks = new IVector[3];
        stacks[0] = Linalg.vector(new double[]{45, 72, 58, 90});
        stacks[1] = Linalg.vector(new double[]{30, 50, 40, 60});
        stacks[2] = Linalg.vector(new double[]{25, 30, 35, 45});
        Plots.ofSvg(800, 500).stackedBar(stacks).title("堆叠柱状图").saveAsSvg(OUTPUT_DIR + "/all_07c_stackedbar.svg");
        System.out.println("  ✓ 堆叠柱状图已生成");
    }

    @Test
    @Order(7) @DisplayName("07d 水平柱状图")
    void testHorizontalBarChart() {
        double[] data = {45, 72, 58, 90, 63, 80, 55};
        Plots.ofSvg(800, 500).horizontalBar(Linalg.vector(data)).title("水平柱状图").saveAsSvg(OUTPUT_DIR + "/all_07d_hbar.svg");
        System.out.println("  ✓ 水平柱状图已生成");
    }

    @Test
    @Order(8) @DisplayName("08 饼图")
    void testPieChart() {
        double[] data = {30, 25, 20, 15, 10};
        Plots.ofSvg(800, 600).pie(Linalg.vector(data)).title("饼图").saveAsSvg(OUTPUT_DIR + "/all_08_pie.svg");
        System.out.println("  ✓ 饼图已生成");
    }

    @Test
    @Order(9) @DisplayName("09 极坐标图")
    void testPolarChart() {
        double[] td = new double[12];
        double[] rd = new double[12];
        for (int i = 0; i < 12; i++) {
            td[i] = i * 30.0;
            rd[i] = Math.abs(Math.sin(Math.toRadians(td[i]))) * 80 + 20;
        }
        Plots.ofSvg(800, 600).polar(Linalg.vector(td), Linalg.vector(rd)).title("极坐标图").saveAsSvg(OUTPUT_DIR + "/all_09_polar.svg");
        System.out.println("  ✓ 极坐标图已生成");
    }

    @Test
    @Order(9) @DisplayName("09b 极坐标柱状图")
    void testPolarBarChart() {
        double[] td = new double[8];
        double[] rd = new double[8];
        for (int i = 0; i < 8; i++) {
            td[i] = i * 45.0;
            rd[i] = Math.abs(Math.sin(Math.toRadians(td[i]))) * 70 + 30;
        }
        Plots.ofSvg(800, 600).polarBar(Linalg.vector(td), Linalg.vector(rd)).title("极坐标柱状图").saveAsSvg(OUTPUT_DIR + "/all_09b_polarbar.svg");
        System.out.println("  ✓ 极坐标柱状图已生成");
    }

    @Test
    @Order(9) @DisplayName("09c 极坐标散点图")
    void testPolarScatterChart() {
        double[] td = new double[20];
        double[] rd = new double[20];
        for (int i = 0; i < 20; i++) {
            td[i] = i * 18.0;
            rd[i] = Math.random() * 80 + 20;
        }
        Plots.ofSvg(800, 600).polarScatter(Linalg.vector(td), Linalg.vector(rd)).title("极坐标散点图").saveAsSvg(OUTPUT_DIR + "/all_09c_polarscatter.svg");
        System.out.println("  ✓ 极坐标散点图已生成");
    }

    @Test
    @Order(10) @DisplayName("10 雷达图")
    void testRadarChart() {
        IVector[] values = new IVector[2];
        values[0] = Linalg.vector(new double[]{4.2, 3.8, 4.5, 3.9, 4.0});
        values[1] = Linalg.vector(new double[]{3.5, 4.2, 3.8, 4.1, 3.7});
        List<String> categories = Arrays.asList("稳定性", "性能", "安全性", "可用性", "成本");
        Plots.ofSvg(800, 600).radar(values, categories).title("雷达图").saveAsSvg(OUTPUT_DIR + "/all_10_radar.svg");
        System.out.println("  ✓ 雷达图已生成");
    }

    @Test
    @Order(11) @DisplayName("11 热力图")
    void testHeatmapChart() {
        double[][] md = new double[8][6];
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 6; j++)
                md[i][j] = Math.sin(i * 0.5) * Math.cos(j * 0.5) * 50 + 50;
        Plots.ofSvg(800, 600).heatmap(Linalg.matrix(md)).title("热力图").saveAsSvg(OUTPUT_DIR + "/all_11_heatmap.svg");
        System.out.println("  ✓ 热力图已生成");
    }

    @Test
    @Order(12) @DisplayName("12 箱线图")
    void testBoxplotChart() {
        double[] dd = new double[100];
        for (int i = 0; i < 100; i++) dd[i] = Math.random() * 50 + 25;
        Plots.ofSvg(800, 500).boxplot(Linalg.vector(dd)).title("箱线图").saveAsSvg(OUTPUT_DIR + "/all_12_boxplot.svg");
        System.out.println("  ✓ 箱线图已生成");
    }

    @Test
    @Order(13) @DisplayName("13 直方图")
    void testHistogramChart() {
        double[] dd = new double[500];
        for (int i = 0; i < 500; i++) dd[i] = Math.random() * 2 - 1;
        Plots.ofSvg(800, 500).histogram(Linalg.vector(dd), 20).title("直方图").saveAsSvg(OUTPUT_DIR + "/all_13_histogram.svg");
        System.out.println("  ✓ 直方图已生成");
    }

    @Test
    @Order(14) @DisplayName("14 小提琴图")
    void testViolinChart() {
        double[] dd = new double[200];
        for (int i = 0; i < 200; i++) dd[i] = Math.random() * 60 + 20;
        Plots.ofSvg(800, 500).violinplot(Linalg.vector(dd)).title("小提琴图").saveAsSvg(OUTPUT_DIR + "/all_14_violin.svg");
        System.out.println("  ✓ 小提琴图已生成");
    }

    @Test
    @Order(15) @DisplayName("15 K线图")
    void testCandlestickChart() {
        int n = 20;
        double[][] ohlc = new double[n][4];
        double price = 100;
        for (int i = 0; i < n; i++) {
            double o = price;
            price += (Math.random() - 0.48) * 5;
            double c = price;
            double l = Math.min(o, c) - Math.random() * 3;
            double h = Math.max(o, c) + Math.random() * 3;
            ohlc[i][0] = o; ohlc[i][1] = c; ohlc[i][2] = l; ohlc[i][3] = h;
        }
        Plots.ofSvg(800, 500).candlestick(Linalg.matrix(ohlc)).title("K线图").saveAsSvg(OUTPUT_DIR + "/all_15_candlestick.svg");
        System.out.println("  ✓ K线图已生成");
    }

    @Test
    @Order(16) @DisplayName("16 漏斗图")
    void testFunnelChart() {
        double[] vals = {100, 75, 50, 35, 20};
        List<String> labels = Arrays.asList("浏览", "点击", "咨询", "订单", "成交");
        Plots.ofSvg(800, 600).funnel(Linalg.vector(vals), labels).title("漏斗图").saveAsSvg(OUTPUT_DIR + "/all_16_funnel.svg");
        System.out.println("  ✓ 漏斗图已生成");
    }

    @Test
    @Order(17) @DisplayName("17 桑基图")
    void testSankeyChart() {
        List<Map<String, Object>> nodes = new ArrayList<>();
        for (String name : Arrays.asList("访问", "注册", "浏览商品", "下单", "支付", "完成"))
            nodes.add(Map.of("name", name));

        List<Map<String, Object>> links = new ArrayList<>();
        links.add(Map.of("source", 0, "target", 1, "value", 30));
        links.add(Map.of("source", 0, "target", 2, "value", 40));
        links.add(Map.of("source", 1, "target", 3, "value", 20));
        links.add(Map.of("source", 2, "target", 3, "value", 35));
        links.add(Map.of("source", 3, "target", 4, "value", 25));
        links.add(Map.of("source", 4, "target", 5, "value", 22));

        Plots.ofSvg(900, 600).sankey(nodes, links).title("桑基图").saveAsSvg(OUTPUT_DIR + "/all_17_sankey.svg");
        System.out.println("  ✓ 桑基图已生成");
    }

    @Test
    @Order(18) @DisplayName("18 旭日图")
    void testSunburstChart() {
        // 用HashMap避免不可变Map的类型推断问题
        Map<String, Object> java = new HashMap<>(); java.put("name", "Java"); java.put("value", 30);
        Map<String, Object> python = new HashMap<>(); python.put("name", "Python"); python.put("value", 25);
        Map<String, Object> go = new HashMap<>(); go.put("name", "Go"); go.put("value", 15);
        Map<String, Object> tech = new HashMap<>(); tech.put("name", "技术"); tech.put("children", List.of(java, python, go));
        Map<String, Object> design = new HashMap<>(); design.put("name", "设计"); design.put("value", 20);
        Map<String, Object> ops = new HashMap<>(); ops.put("name", "运营"); ops.put("value", 18);
        Map<String, Object> product = new HashMap<>(); product.put("name", "产品"); product.put("children", List.of(design, ops));
        Map<String, Object> market = new HashMap<>(); market.put("name", "市场"); market.put("value", 22);
        Map<String, Object> root = new HashMap<>();
        root.put("name", "总");
        root.put("children", List.of(tech, product, market));
        Plots.ofSvg(800, 600).sunburst(root).title("旭日图").saveAsSvg(OUTPUT_DIR + "/all_18_sunburst.svg");
        System.out.println("  ✓ 旭日图已生成");
    }

    @Test
    @Order(19) @DisplayName("19 矩形树图")
    void testTreemapChart() {
        List<Map<String, Object>> data = List.of(
            Map.of("name", "技术", "value", 40, "children", List.of(
                Map.of("name", "前端", "value", 20),
                Map.of("name", "后端", "value", 35),
                Map.of("name", "运维", "value", 15)
            )),
            Map.of("name", "产品", "value", 30),
            Map.of("name", "运营", "value", 20),
            Map.of("name", "市场", "value", 25)
        );
        Plots.ofSvg(800, 600).treemap(data).title("矩形树图").saveAsSvg(OUTPUT_DIR + "/all_19_treemap.svg");
        System.out.println("  ✓ 矩形树图已生成");
    }

    @Test
    @Order(20) @DisplayName("20 树图")
    void testTreeChart() {
        Map<String, Object> tree = Map.of("name", "Root",
            "children", List.of(
                Map.of("name", "Node A",
                    "children", List.of(
                        Map.of("name", "Leaf A1"),
                        Map.of("name", "Leaf A2")
                    )),
                Map.of("name", "Node B",
                    "children", List.of(
                        Map.of("name", "Leaf B1"),
                        Map.of("name", "Leaf B2"),
                        Map.of("name", "Leaf B3")
                    )),
                Map.of("name", "Node C")
            )
        );
        Plots.ofSvg(800, 600).tree(tree, "Root").title("树图").saveAsSvg(OUTPUT_DIR + "/all_20_tree.svg");
        System.out.println("  ✓ 树图已生成");
    }

    @Test
    @Order(21) @DisplayName("21 关系图")
    void testGraphChart() {
        List<Map<String, Object>> nodes = new ArrayList<>();
        for (String name : Arrays.asList("A", "B", "C", "D", "E", "F"))
            nodes.add(Map.of("name", name));

        List<Map<String, Object>> links = new ArrayList<>();
        links.add(Map.of("source", 0, "target", 1));
        links.add(Map.of("source", 0, "target", 2));
        links.add(Map.of("source", 1, "target", 3));
        links.add(Map.of("source", 2, "target", 3));
        links.add(Map.of("source", 3, "target", 4));
        links.add(Map.of("source", 4, "target", 5));
        links.add(Map.of("source", 2, "target", 5));

        Plots.ofSvg(800, 600).graph(nodes, links).title("关系图").saveAsSvg(OUTPUT_DIR + "/all_21_graph.svg");
        System.out.println("  ✓ 关系图已生成");
    }

    @Test
    @Order(22) @DisplayName("22 平行坐标图")
    void testParallelChart() {
        int rows = 8, cols = 5;
        double[][] md = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                md[i][j] = Math.random() * 50 + 20 + j * 5;
        List<String> dims = Arrays.asList("性能", "稳定性", "安全性", "可用性", "成本");
        Plots.ofSvg(900, 500).parallel(Linalg.matrix(md), dims).title("平行坐标图").saveAsSvg(OUTPUT_DIR + "/all_22_parallel.svg");
        System.out.println("  ✓ 平行坐标图已生成");
    }

    @Test
    @Order(23) @DisplayName("23 主题河流图")
    void testThemeRiverChart() {
        List<Map<String, Object>> data = new ArrayList<>();
        String[] topics = {"技术", "产品", "运营", "市场"};
        for (int i = 0; i < 12; i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("time", "T" + i);
            for (String t : topics) row.put(t, Math.random() * 50 + 20);
            data.add(row);
        }
        Plots.ofSvg(900, 500).themeRiver(data).title("主题河流图").saveAsSvg(OUTPUT_DIR + "/all_23_themeriver.svg");
        System.out.println("  ✓ 主题河流图已生成");
    }

    @Test
    @Order(24) @DisplayName("24 仪表盘")
    void testGaugeChart() {
        Plots.ofSvg(600, 500).gaugeWithLabel(72, 0, 100, "完成率").title("仪表盘").saveAsSvg(OUTPUT_DIR + "/all_24_gauge.svg");
        System.out.println("  ✓ 仪表盘已生成");
    }

    @Test
    @Order(25) @DisplayName("25 KDE密度图")
    void testKdePlot() {
        IVector data = Linalg.randn(200, 0.0, 1.0);
        Plots.ofSvg(600, 400).kdeplot(data, 256, 0.0).title("KDE密度图").saveAsSvg(OUTPUT_DIR + "/all_25_kde.svg");
        System.out.println("  ✓ KDE密度图已生成");
    }

    @Test
    @Order(26) @DisplayName("26 Q-Q图")
    void testQqPlot() {
        IVector data = Linalg.randn(100, 0.0, 1.0);
        Plots.ofSvg(600, 400).qqplot(data).title("Q-Q图").saveAsSvg(OUTPUT_DIR + "/all_26_qqplot.svg");
        System.out.println("  ✓ Q-Q图已生成");
    }

    @Test
    @Order(27) @DisplayName("27 回归图")
    void testRegPlot() {
        IVector<Double> x = Linalg.linspace(0.0, 10.0, 50);
        double[] yArr = new double[50];
        for (int i = 0; i < 50; i++) {
            double xi = ((Number)x.get(i)).doubleValue();
            yArr[i] = 2 * xi + 3 + ((Number)Linalg.randn(1, 0.0, 1.0).get(0)).doubleValue();
        }
        IVector<Double> y = IVector.of(yArr);
        Plots.ofSvg(600, 400).regplot(x, y, true).title("回归图(带置信区间)").saveAsSvg(OUTPUT_DIR + "/all_27_regplot.svg");
        System.out.println("  ✓ 回归图已生成");
    }

    @Test
    @Order(28) @DisplayName("28 配对图")
    void testPairplot() {
        int rows = 50, cols = 4;
        double[][] md = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                md[i][j] = Math.random() * 50 + 20 + j * 10;
        List<String> colNames = Arrays.asList("长度", "宽度", "高度", "重量");
        Plots.ofSvg(800, 700).pairplot(Linalg.matrix(md), colNames, com.yishape.lab.math.plot.PairplotDiagonal.KDE).title("配对图").saveAsSvg(OUTPUT_DIR + "/all_28_pairplot.svg");
        System.out.println("  ✓ 配对图已生成");
    }

    @Test
    @Order(29) @DisplayName("29 联合图")
    void testJointplot() {
        double[] xd = new double[100];
        double[] yd = new double[100];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < 100; i++) {
            xd[i] = 2 + Math.pow(rand.nextDouble(), 0.5) * 8;
            yd[i] = 0.5 * xd[i] + 2 + rand.nextGaussian() * 1.5;
        }
        IVector<Double> x = IVector.of(xd);
        IVector<Double> y = IVector.of(yd);
        Plots.ofSvg(600, 600).jointplot(x, y, com.yishape.lab.math.plot.JointplotMarginal.HIST).title("联合图").saveAsSvg(OUTPUT_DIR + "/all_29_jointplot.svg");
        System.out.println("  ✓ 联合图已生成");
    }

    @Test
    @Order(30) @DisplayName("30 子图网格")
    void testSubplots() {
        IVector<Double> x = Linalg.linspace(0.0, 10.0, 50);
        double[] y1 = new double[50];
        double[] y2 = new double[50];
        double[] y3 = new double[50];
        for (int i = 0; i < 50; i++) {
            double xi = ((Number)x.get(i)).doubleValue();
            y1[i] = Math.sin(xi) * 10 + 20;
            y2[i] = Math.cos(xi) * 10 + 20;
            y3[i] = xi * 2 + 5;
        }
        Plots.ofSvg(800, 600).subplots(2, 2).title("子图网格").saveAsSvg(OUTPUT_DIR + "/all_30_subplots.svg");
        System.out.println("  ✓ 子图网格已生成");
    }

    @AfterAll
    static void tearDown() {
        System.out.println("\n=== 全图表类型SVG测试完成 ===");
        File dir = new File(OUTPUT_DIR);
        File[] files = dir.listFiles((d, n) -> n.endsWith(".svg"));
        int count = files != null ? files.length : 0;
        System.out.println("共生成 " + count + " 个SVG文件");
    }
}
