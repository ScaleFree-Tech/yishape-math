package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

import java.util.List;

/**
 * 极坐标图表测试类
 * 
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class PolarChartTest extends JavaFxChartTestBase {
    
    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 极坐标图表测试开始 ==========");
    }
    
    @Test
    @Order(1)
    @DisplayName("测试极坐标柱状图")
    void testPolarBarChart() {
        System.out.println("\n[测试1] 极坐标柱状图");
        
        double[] data = {80, 65, 90, 75, 85, 70};
        List<String> categories = List.of("A", "B", "C", "D", "E", "F");
        
        IPlot plot = Plots.of(800, 600);
        plot.polarBar(Linalg.vector(data), categories)
            .title("极坐标柱状图");
        
        boolean success = generateImage(plot, "polar_01_bar.png");
        Assertions.assertTrue(success, "极坐标柱状图图片生成失败");
    }
    
    @Test
    @Order(2)
    @DisplayName("测试极坐标线图")
    void testPolarLineChart() {
        System.out.println("\n[测试2] 极坐标线图");
        
        double[] data = {60, 80, 70, 90, 65, 85};
        List<String> categories = List.of("N", "NE", "E", "SE", "S", "SW");
        
        IPlot plot = Plots.of(800, 600);
        plot.polarLine(Linalg.vector(data), categories)
            .title("极坐标线图");
        
        boolean success = generateImage(plot, "polar_02_line.png");
        Assertions.assertTrue(success, "极坐标线图图片生成失败");
    }
    
    @Test
    @Order(3)
    @DisplayName("测试极坐标散点图")
    void testPolarScatterChart() {
        System.out.println("\n[测试3] 极坐标散点图");
        
        double[] data = {50, 70, 60, 80, 55, 75};
        List<String> categories = List.of("Q1", "Q2", "Q3", "Q4", "Q5", "Q6");
        
        IPlot plot = Plots.of(800, 600);
        plot.polarScatter(Linalg.vector(data), categories)
            .title("极坐标散点图");
        
        boolean success = generateImage(plot, "polar_03_scatter.png");
        Assertions.assertTrue(success, "极坐标散点图图片生成失败");
    }
    
    @Test
    @Order(4)
    @DisplayName("测试不同主题极坐标图")
    void testPolarChartWithThemes() {
        System.out.println("\n[测试4] 不同主题极坐标图");
        
        double[] data = {75, 85, 60, 90, 70, 80};
        List<String> categories = List.of("1", "2", "3", "4", "5", "6");
        String[] themes = {"default", "dark", "rainbow"};
        
        int successCount = 0;
        for (String theme : themes) {
            IPlot plot = Plots.of(800, 600, theme);
            plot.polarBar(Linalg.vector(data), categories)
                .title("极坐标图 - " + theme);
            
            if (generateImage(plot, "polar_05_theme_" + theme + ".png")) {
                successCount++;
            }
        }
        
        Assertions.assertTrue(successCount >= 2, "至少应生成2个主题图片");
    }
    
    @AfterAll
    static void tearDown() {
        printSummary("极坐标图表测试");
    }
}
