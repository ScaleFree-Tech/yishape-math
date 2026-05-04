package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.javafx.base.SvgChartTestBase;
import com.yishape.lab.math.plot.svg.SvgPlot;
import org.junit.jupiter.api.*;

import java.util.ArrayList;
import java.util.List;

/**
 * 统计图表SVG矢量输出测试（箱线图、直方图）。
 * 使用 {@link SvgPlot} 生成真正的矢量 SVG，放大无模糊。
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class StatisticalChartSvgTest extends SvgChartTestBase {

    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 统计图表SVG测试开始 ==========");
    }

    @Test
    @Order(1)
    @DisplayName("SVG箱线图")
    void testBoxplot() {
        System.out.println("\n[SVG测试1] 箱线图");
        double[] data = {12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50};
        SvgPlot plot = new SvgPlot(800, 600);
        plot.boxplot(Linalg.vector(data)).title("箱线图");
        boolean success = generateImage(plot, "stat_01_boxplot.svg");
        Assertions.assertTrue(success, "箱线图SVG生成失败");
    }

    @Test
    @Order(2)
    @DisplayName("SVG带标签箱线图")
    void testBoxplotWithLabels() {
        System.out.println("\n[SVG测试2] 带标签箱线图");
        double[] data = {15, 25, 35, 45, 55, 65, 75, 85, 95};
        SvgPlot plot = new SvgPlot(800, 600);
        plot.boxplot(Linalg.vector(data), List.of("Group A")).title("带标签箱线图");
        boolean success = generateImage(plot, "stat_02_boxplot_labeled.svg");
        Assertions.assertTrue(success, "带标签箱线图SVG生成失败");
    }

    @Test
    @Order(3)
    @DisplayName("SVG直方图")
    void testHistogram() {
        System.out.println("\n[SVG测试3] 直方图");
        double[] data = new double[50];
        for (int i = 0; i < 50; i++) data[i] = 10 + Math.random() * 30;
        SvgPlot plot = new SvgPlot(800, 600);
        plot.histogram(Linalg.vector(data)).title("直方图");
        boolean success = generateImage(plot, "stat_03_histogram.svg");
        Assertions.assertTrue(success, "直方图SVG生成失败");
    }

    @Test
    @Order(4)
    @DisplayName("SVG小提琴图")
    void testViolinplot() {
        System.out.println("\n[SVG测试4] 小提琴图");
        double[] data = new double[50];
        for (int i = 0; i < 50; i++) data[i] = 10 + Math.random() * 30;
        SvgPlot plot = new SvgPlot(800, 600);
        plot.violinplot(Linalg.vector(data)).title("小提琴图");
        boolean success = generateImage(plot, "stat_04_violin.svg");
        Assertions.assertTrue(success, "小提琴图SVG生成失败");
    }

    @AfterAll
    static void tearDown() {
        printSummary("统计图表SVG测试");
    }
}
