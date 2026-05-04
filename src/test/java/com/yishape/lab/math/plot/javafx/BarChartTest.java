package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

import java.util.List;

/**
 * 柱状图测试类
 * 
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class BarChartTest extends JavaFxChartTestBase {
    
    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 柱状图测试开始 ==========");
    }
    
    @Test
    @Order(1)
    @DisplayName("测试基础柱状图")
    void testBasicBarChart() {
        System.out.println("\n[测试1] 基础柱状图");
        
        double[] data = {45, 60, 75, 50, 85, 40, 95};
        
        IPlot plot = Plots.of(800, 600).bar(Linalg.vector(data))
            .title("基础柱状图")
            .xlabel("类别")
            .ylabel("数值");
        
        boolean success = generateImage(plot, "bar_01_basic.png");
        Assertions.assertTrue(success, "基础柱状图图片生成失败");
    }
    
    @Test
    @Order(2)
    @DisplayName("测试带标签柱状图")
    void testBarChartWithLabels() {
        System.out.println("\n[测试2] 带标签柱状图");
        
        double[] data = {120, 200, 150, 80, 170};
        List<String> labels = List.of("一月", "二月", "三月", "四月", "五月");
        
        IPlot plot = Plots.of(800, 600);
        plot.bar(labels, Linalg.vector(data))
            .title("月度销售柱状图", "2024年上半年")
            .xlabel("月份")
            .ylabel("销售额(万元)");
        
        boolean success = generateImage(plot, "bar_02_with_labels.png");
        Assertions.assertTrue(success, "带标签柱状图图片生成失败");
    }
    
    @Test
    @Order(3)
    @DisplayName("测试不同主题柱状图")
    void testBarChartWithThemes() {
        System.out.println("\n[测试3] 不同主题柱状图");
        
        double[] data = {30, 50, 70, 40, 60};
        List<String> labels = List.of("A", "B", "C", "D", "E");
        String[] themes = {"default", "dark", "business", "academic"};
        
        int successCount = 0;
        for (String theme : themes) {
            IPlot plot = Plots.of(800, 600, theme);
            plot.bar(labels, Linalg.vector(data))
                .title("柱状图 - " + theme);
            
            if (generateImage(plot, "bar_03_theme_" + theme + ".png")) {
                successCount++;
            }
        }
        
        Assertions.assertTrue(successCount >= 3, "至少应生成3个主题图片");
    }
    
    @Test
    @Order(4)
    @DisplayName("测试分组柱状图")
    void testGroupedBarChart() {
        System.out.println("\n[测试4] 分组柱状图");
        
        // 每个季度都有A和B两组数据，用于并排对比
        // 数据顺序：Q1-A, Q1-B, Q2-A, Q2-B, Q3-A, Q3-B, Q4-A, Q4-B, Q5-A, Q5-B
        double[] data = {100, 80, 150, 120, 110, 140, 160, 170, 140, 130};
        List<String> labels = List.of(
            "Q1", "Q1", "Q2", "Q2", "Q3", "Q3", "Q4", "Q4", "Q5", "Q5"
        );
        List<String> hue = List.of(
            "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"
        );
        
        IPlot plot = Plots.of(800, 600);
        plot.bar(labels, Linalg.vector(data), hue)
            .title("分组柱状图", "A组 vs B组季度对比");
        
        boolean success = generateImage(plot, "bar_04_grouped.png");
        Assertions.assertTrue(success, "分组柱状图图片生成失败");
    }
    
    @AfterAll
    static void tearDown() {
        printSummary("柱状图测试");
    }
}
