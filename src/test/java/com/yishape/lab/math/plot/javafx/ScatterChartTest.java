package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

import java.util.List;

/**
 * 散点图测试类
 * 
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ScatterChartTest extends JavaFxChartTestBase {

    /** mtcars 前 14：车重 vs 油耗 */
    private static final double[] MTCARS_WT = {
        2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440, 3.440, 4.070, 3.730, 3.780,
    };
    private static final double[] MTCARS_MPG = {
        21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2,
    };

    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 散点图测试开始 ==========");
    }
    
    @Test
    @Order(1)
    @DisplayName("测试基础散点图")
    void testBasicScatterChart() {
        System.out.println("\n[测试1] 基础散点图");
        IPlot plot = Plots.of(800, 600);
        plot.scatter(Linalg.vector(MTCARS_WT), Linalg.vector(MTCARS_MPG))
            .title("mtcars：油耗 vs 车重", "前 14 行")
            .xlabel("Weight (1000 lb)")
            .ylabel("MPG");
        
        boolean success = generateImage(plot, "scatter_01_basic.png");
        Assertions.assertTrue(success, "基础散点图图片生成失败");
    }
    
    @Test
    @Order(2)
    @DisplayName("测试分组散点图")
    void testGroupedScatterChart() {
        System.out.println("\n[测试2] 分组散点图");
        // Iris versicolor / virginica：萼片长 × 萼片宽（cm），各取 20 株，UCI 顺序片段
        double[] sepalL = {
            7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6,
            6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.3, 6.6, 6.9, 6.0,
        };
        double[] sepalW = {
            3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 2.7, 2.8, 2.7, 3.3,
            3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3.0, 2.5, 2.8, 2.9, 2.7, 2.9, 2.8, 2.9,
        };
        List<String> hue = new java.util.ArrayList<>();
        for (int i = 0; i < 20; i++) {
            hue.add("versicolor");
        }
        for (int i = 0; i < 20; i++) {
            hue.add("virginica");
        }
        
        IPlot plot = Plots.of(800, 600);
        plot.scatter(Linalg.vector(sepalL), Linalg.vector(sepalW), hue)
            .title("Iris：萼片长 vs 宽", "versicolor / virginica 各 20 株");
        
        boolean success = generateImage(plot, "scatter_02_grouped.png");
        Assertions.assertTrue(success, "分组散点图图片生成失败");
    }
    
    @Test
    @Order(3)
    @DisplayName("测试不同主题散点图")
    void testScatterChartWithThemes() {
        System.out.println("\n[测试3] 不同主题散点图");
        
        double[] hp = {110, 110, 93, 110, 175, 105, 245, 62, 95, 123};
        double[] qsec = {16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20.00, 22.90, 18.30};
        String[] themes = {"default", "dark", "seaborn"};

        int successCount = 0;
        for (String theme : themes) {
            IPlot plot = Plots.of(800, 600, theme);
            plot.scatter(Linalg.vector(hp), Linalg.vector(qsec))
                .title("散点图 - " + theme, "mtcars：1/4 英里时间 vs 马力")
                .xlabel("HP")
                .ylabel("qsec (s)");
            
            if (generateImage(plot, "scatter_03_theme_" + theme + ".png")) {
                successCount++;
            }
        }
        
        Assertions.assertTrue(successCount >= 2, "至少应生成2个主题图片");
    }
    
    @AfterAll
    static void tearDown() {
        printSummary("散点图测试");
    }
}
