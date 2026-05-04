package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

import java.util.ArrayList;
import java.util.List;

/**
 * 统计图表测试类（箱线图、小提琴图、K线图）
 * 
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class StatisticalChartTest extends JavaFxChartTestBase {
    
    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 统计图表测试开始 ==========");
    }
    
    @Test
    @Order(1)
    @DisplayName("测试箱线图")
    void testBoxplot() {
        System.out.println("\n[测试1] 箱线图");
        
        double[] data = {12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50};
        
        IPlot plot = Plots.of(800, 600);
        plot.boxplot(Linalg.vector(data))
            .title("箱线图");
        
        boolean success = generateImage(plot, "stat_01_boxplot.svg");
        Assertions.assertTrue(success, "箱线图图片生成失败");
    }
    
    @Test
    @Order(2)
    @DisplayName("测试带标签箱线图")
    void testBoxplotWithLabels() {
        System.out.println("\n[测试2] 带标签箱线图");
        
        double[] data = {15, 25, 35, 45, 55, 65, 75, 85, 95};
        List<String> labels = List.of("Group A");
        
        IPlot plot = Plots.of(800, 600);
        plot.boxplot(Linalg.vector(data), labels)
            .title("带标签箱线图");
        
        boolean success = generateImage(plot, "stat_02_boxplot_labeled.svg");
        Assertions.assertTrue(success, "带标签箱线图图片生成失败");
    }
    
    @Test
    @Order(3)
    @DisplayName("测试小提琴图")
    void testViolinplot() {
        System.out.println("\n[测试3] 小提琴图");
        
        // 生成随机数据
        double[] data = new double[50];
        for (int i = 0; i < 50; i++) {
            data[i] = 10 + Math.random() * 30;
        }
        
        IPlot plot = Plots.of(800, 600);
        plot.violinplot(Linalg.vector(data))
            .title("小提琴图");
        
        boolean success = generateImage(plot, "stat_03_violin.svg");
        Assertions.assertTrue(success, "小提琴图图片生成失败");
    }
    
    @Test
    @Order(4)
    @DisplayName("测试带标签小提琴图")
    void testViolinplotWithLabels() {
        System.out.println("\n[测试4] 带标签小提琴图");
        
        double[] data = new double[60];
        for (int i = 0; i < 60; i++) {
            data[i] = 20 + 25 * (Math.random() - 0.5);
        }
        List<String> labels = List.of("Dataset 1");
        
        IPlot plot = Plots.of(800, 600);
        plot.violinplot(Linalg.vector(data), labels)
            .title("带标签小提琴图");
        
        boolean success = generateImage(plot, "stat_04_violin_labeled.svg");
        Assertions.assertTrue(success, "带标签小提琴图图片生成失败");
    }
    
    @Test
    @Order(5)
    @DisplayName("测试K线图")
    void testCandlestick() {
        System.out.println("\n[测试5] K线图");
        
        // 创建OHLC数据: [开盘, 收盘, 最低, 最高]
        double[][] ohlcData = {
            {10, 12, 9, 13},
            {12, 11, 10, 14},
            {11, 15, 11, 16},
            {15, 14, 13, 17},
            {14, 16, 14, 18},
            {16, 15, 14, 19},
            {15, 18, 15, 20}
        };
        IMatrix<Double> data = Linalg.matrix(ohlcData);
        List<String> dates = List.of("周一", "周二", "周三", "周四", "周五", "周六", "周日");
        
        IPlot plot = Plots.of(800, 600);
        plot.candlestick(data, dates)
            .title("K线图", "周度股价走势");
        
        boolean success = generateImage(plot, "stat_05_candlestick.svg");
        Assertions.assertTrue(success, "K线图图片生成失败");
    }
    
    @Test
    @Order(6)
    @DisplayName("测试分组小提琴图")
    void testGroupedViolin() {
        System.out.println("\n[测试6] 分组小提琴图");

        // 生成三组数据（不同分布）
        double[] groupA = {25, 28, 30, 32, 35, 27, 29, 31, 33, 26};
        double[] groupB = {15, 18, 20, 22, 25, 17, 19, 21, 23, 16};
        double[] groupC = {35, 38, 40, 42, 45, 37, 39, 41, 43, 36};

        // 合并数据
        double[] allData = new double[groupA.length + groupB.length + groupC.length];
        System.arraycopy(groupA, 0, allData, 0, groupA.length);
        System.arraycopy(groupB, 0, allData, groupA.length, groupB.length);
        System.arraycopy(groupC, 0, allData, groupA.length + groupB.length, groupC.length);

        // 创建标签
        List<String> labels = new ArrayList<>();
        for (int i = 0; i < groupA.length; i++) labels.add("A");
        for (int i = 0; i < groupB.length; i++) labels.add("B");
        for (int i = 0; i < groupC.length; i++) labels.add("C");

        // 创建分组小提琴图
        IPlot plot = Plots.of(1000, 600);
        plot.violinplot(Linalg.vector(allData), labels)
            .title("分组小提琴图", "三组数据分布对比");

        boolean success = generateImage(plot, "stat_06_grouped_violin.svg");
        Assertions.assertTrue(success, "分组小提琴图图片生成失败");
    }

    @Test
    @Order(7)
    @DisplayName("测试不同主题统计图")
    void testStatisticalChartsWithThemes() {
        System.out.println("\n[测试7] 不同主题统计图");
        
        double[] boxData = {20, 30, 40, 50, 60, 70, 80, 90};
        String[] themes = {"default", "academic", "business"};
        
        int successCount = 0;
        for (String theme : themes) {
            IPlot plot = Plots.of(800, 600, theme);
            plot.boxplot(Linalg.vector(boxData))
                .title("箱线图 - " + theme);
            
            if (generateImage(plot, "stat_07_theme_" + theme + ".svg")) {
                successCount++;
            }
        }
        
        Assertions.assertTrue(successCount >= 2, "至少应生成2个主题图片");
    }
    
    @AfterAll
    static void tearDown() {
        printSummary("统计图表测试");
    }
}
