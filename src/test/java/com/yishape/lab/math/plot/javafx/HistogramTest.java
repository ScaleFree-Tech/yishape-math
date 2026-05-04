package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

/**
 * 直方图测试类
 * 
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class HistogramTest extends JavaFxChartTestBase {

    /** Iris virginica 花瓣长（cm），UCI 前 50 条按文件顺序。 */
    private static final double[] IRIS_VIRGINICA_PETAL_LEN = {
        6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0,
        5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4,
        5.6, 5.1, 5.1, 5.4, 5.5, 5.3, 5.4, 5.2, 5.1, 5.2,
    };

    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 直方图测试开始 ==========");
    }
    
    @Test
    @Order(1)
    @DisplayName("测试基础直方图")
    void testBasicHistogram() {
        System.out.println("\n[测试1] 基础直方图");
        IPlot plot = Plots.of(800, 600);
        plot.hist(Linalg.vector(IRIS_VIRGINICA_PETAL_LEN), false)
            .title("直方图", "Iris virginica 花瓣长 cm")
            .xlabel("Petal length")
            .ylabel("Count");
        
        boolean success = generateImage(plot, "hist_01_basic.png");
        Assertions.assertTrue(success, "基础直方图图片生成失败");
    }
    
    @Test
    @Order(2)
    @DisplayName("测试带拟合线的直方图")
    void testHistogramWithFitting() {
        System.out.println("\n[测试2] 带拟合线的直方图");
        double[] nhanesBmi = new double[180];
        java.util.Random r = new java.util.Random(20240103L);
        for (int i = 0; i < nhanesBmi.length; i++) {
            double z = r.nextGaussian();
            nhanesBmi[i] = 26.5 + 5.2 * z;
            if (nhanesBmi[i] < 15) {
                nhanesBmi[i] = 15;
            }
            if (nhanesBmi[i] > 48) {
                nhanesBmi[i] = 48;
            }
        }
        IPlot plot = Plots.of(800, 600);
        plot.hist(Linalg.vector(nhanesBmi), true)
            .title("BMI 分布示意", "正态合成 · kg/m²")
            .xlabel("BMI")
            .ylabel("频数");
        
        boolean success = generateImage(plot, "hist_02_with_fitting.png");
        Assertions.assertTrue(success, "带拟合线直方图图片生成失败");
    }
    
    @Test
    @Order(3)
    @DisplayName("测试不同数据分布直方图")
    void testHistogramWithDifferentDistributions() {
        System.out.println("\n[测试3] 不同数据分布直方图");
        // Old Faithful 喷发间隔（分钟）经典集前 80 个观测值量级
        double[] faithful = {
            79, 54, 74, 62, 85, 55, 88, 85, 51, 54, 84, 78, 47, 83, 52, 62, 84, 52, 79, 51,
            47, 78, 69, 74, 83, 55, 76, 78, 79, 73, 77, 66, 80, 74, 52, 48, 80, 59, 90, 82,
            63, 62, 84, 86, 77, 79, 88, 63, 68, 86, 77, 76, 86, 67, 81, 75, 73, 76, 76, 76,
            76, 89, 80, 73, 77, 84, 85, 80, 69, 76, 75, 85, 72, 82, 74, 80, 49, 88, 79, 82,
        };
        IPlot plot1 = Plots.of(800, 600);
        plot1.hist(Linalg.vector(faithful), false)
            .title("单峰右偏：间歇泉间隔", "分钟·示意摘录");
        generateImage(plot1, "hist_03_uniform.png");
        // 双峰：两簇正态合成（例如两批仪器量程不同）
        double[] stipend = new double[120];
        java.util.Random r2 = new java.util.Random(42L);
        for (int i = 0; i < 60; i++) {
            stipend[i] = 45 + r2.nextDouble() * 18;
        }
        for (int i = 60; i < 120; i++) {
            stipend[i] = 110 + r2.nextDouble() * 35;
        }
        IPlot plot2 = Plots.of(800, 600);
        plot2.hist(Linalg.vector(stipend), true)
            .title("双峰分布", "两簇正态混合·示意");
        boolean success = generateImage(plot2, "hist_04_bimodal.png");
        
        Assertions.assertTrue(success, "不同分布直方图图片生成失败");
    }
    
    @AfterAll
    static void tearDown() {
        printSummary("直方图测试");
    }
}
