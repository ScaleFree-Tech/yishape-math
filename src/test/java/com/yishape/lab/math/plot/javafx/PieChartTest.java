package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

/**
 * 饼图测试类
 * 
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class PieChartTest extends JavaFxChartTestBase {
    
    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 饼图测试开始 ==========");
    }
    
    @Test
    @Order(1)
    @DisplayName("测试基础饼图")
    void testBasicPieChart() {
        System.out.println("\n[测试1] 基础饼图");
        
        double[] data = {30, 25, 20, 15, 10};
        
        IPlot plot = Plots.of(800, 600);
        plot.pie(Linalg.vector(data))
            .title("基础饼图");
        
        boolean success = generateImage(plot, "pie_01_basic.png");
        Assertions.assertTrue(success, "基础饼图图片生成失败");
    }
    
    @Test
    @Order(2)
    @DisplayName("测试不同数据饼图")
    void testPieChartWithDifferentData() {
        System.out.println("\n[测试2] 不同数据饼图");
        
        // 均匀分布
        double[] data1 = {20, 20, 20, 20, 20};
        IPlot plot1 = Plots.of(800, 600);
        plot1.pie(Linalg.vector(data1))
            .title("均匀分布饼图");
        generateImage(plot1, "pie_02_uniform.png");
        
        // 单一主导
        double[] data2 = {70, 10, 10, 5, 5};
        IPlot plot2 = Plots.of(800, 600);
        plot2.pie(Linalg.vector(data2))
            .title("单一主导饼图");
        boolean success = generateImage(plot2, "pie_03_dominant.png");
        
        Assertions.assertTrue(success, "不同数据饼图图片生成失败");
    }
    
    @Test
    @Order(3)
    @DisplayName("测试不同主题饼图")
    void testPieChartWithThemes() {
        System.out.println("\n[测试3] 不同主题饼图");
        
        double[] data = {35, 25, 20, 15, 5};
        String[] themes = {"default", "rainbow", "vintage"};
        
        int successCount = 0;
        for (String theme : themes) {
            IPlot plot = Plots.of(800, 600, theme);
            plot.pie(Linalg.vector(data))
                .title("饼图 - " + theme);
            
            if (generateImage(plot, "pie_04_theme_" + theme + ".png")) {
                successCount++;
            }
        }
        
        Assertions.assertTrue(successCount >= 2, "至少应生成2个主题图片");
    }
    
    @AfterAll
    static void tearDown() {
        printSummary("饼图测试");
    }
}
