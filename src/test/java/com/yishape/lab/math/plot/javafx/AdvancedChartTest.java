package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

import java.util.List;

/**
 * 高级图表测试类（雷达图、热力图、仪表盘）
 * 
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class AdvancedChartTest extends JavaFxChartTestBase {
    
    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 高级图表测试开始 ==========");
    }
    
    @Test
    @Order(1)
    @DisplayName("测试雷达图")
    void testRadarChart() {
        System.out.println("\n[测试1] 雷达图");
        
        double[] data = {80, 90, 75, 85, 70, 95};
        List<String> indicators = List.of("性能", "质量", "价格", "服务", "创新", "可靠性");
        
        IPlot plot = Plots.of(800, 600);
        plot.radar(Linalg.vector(data), indicators)
            .title("雷达图", "产品综合评价");
        
        boolean success = generateImage(plot, "advanced_01_radar.png");
        Assertions.assertTrue(success, "雷达图图片生成失败");
    }
    
    @Test
    @Order(2)
    @DisplayName("测试不同数据雷达图")
    void testRadarChartWithDifferentData() {
        System.out.println("\n[测试2] 不同数据雷达图");
        
        // 高均衡数据
        double[] balancedData = {85, 85, 85, 85, 85};
        List<String> indicators = List.of("A", "B", "C", "D", "E");
        IPlot plot1 = Plots.of(800, 600);
        plot1.radar(Linalg.vector(balancedData), indicators)
            .title("均衡雷达图");
        generateImage(plot1, "advanced_02_radar_balanced.png");
        
        // 不均衡数据
        double[] unbalancedData = {30, 40, 90, 80, 20};
        IPlot plot2 = Plots.of(800, 600);
        plot2.radar(Linalg.vector(unbalancedData), indicators)
            .title("不均衡雷达图");
        boolean success = generateImage(plot2, "advanced_03_radar_unbalanced.png");
        
        Assertions.assertTrue(success, "不同数据雷达图图片生成失败");
    }
    
    @Test
    @Order(3)
    @DisplayName("测试热力图")
    void testHeatmap() {
        System.out.println("\n[测试3] 热力图");
        
        double[][] data = {
            {1, 2, 3, 4, 5},
            {2, 4, 6, 8, 10},
            {3, 6, 9, 12, 15},
            {4, 8, 12, 16, 20},
            {5, 10, 15, 20, 25}
        };
        IMatrix<Double> matrix = Linalg.matrix(data);
        List<String> xLabels = List.of("A", "B", "C", "D", "E");
        List<String> yLabels = List.of("1", "2", "3", "4", "5");
        
        IPlot plot = Plots.of(800, 600);
        plot.heatmap(matrix, xLabels, yLabels)
            .title("热力图", "相关性矩阵");
        
        boolean success = generateImage(plot, "advanced_04_heatmap.png");
        Assertions.assertTrue(success, "热力图图片生成失败");
    }
    
    @Test
    @Order(4)
    @DisplayName("测试不同数据热力图")
    void testHeatmapWithDifferentData() {
        System.out.println("\n[测试4] 不同数据热力图");
        
        // 随机数据热力图
        double[][] randomData = new double[6][8];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 8; j++) {
                randomData[i][j] = Math.random() * 100;
            }
        }
        IMatrix<Double> matrix = Linalg.matrix(randomData);
        
        IPlot plot = Plots.of(800, 600);
        plot.heatmap(matrix, null, null)
            .title("随机数据热力图");
        
        boolean success = generateImage(plot, "advanced_05_heatmap_random.png");
        Assertions.assertTrue(success, "不同数据热力图图片生成失败");
    }
    
    @Test
    @Order(5)
    @DisplayName("测试仪表盘")
    void testGauge() {
        System.out.println("\n[测试5] 仪表盘");
        
        IPlot plot = Plots.of(800, 600);
        plot.gauge(75, 100, 0)
            .title("仪表盘", "完成度75%");
        
        boolean success = generateImage(plot, "advanced_06_gauge.png");
        Assertions.assertTrue(success, "仪表盘图片生成失败");
    }
    
    @Test
    @Order(6)
    @DisplayName("测试不同数值仪表盘")
    void testGaugeWithDifferentValues() {
        System.out.println("\n[测试6] 不同数值仪表盘");
        
        // 低数值
        IPlot plot1 = Plots.of(800, 600);
        plot1.gauge(25, 100, 0)
            .title("仪表盘 - 25%");
        generateImage(plot1, "advanced_07_gauge_25.png");
        
        // 中等数值
        IPlot plot2 = Plots.of(800, 600);
        plot2.gauge(50, 100, 0)
            .title("仪表盘 - 50%");
        generateImage(plot2, "advanced_08_gauge_50.png");
        
        // 高数值
        IPlot plot3 = Plots.of(800, 600);
        plot3.gauge(90, 100, 0)
            .title("仪表盘 - 90%");
        boolean success = generateImage(plot3, "advanced_09_gauge_90.png");
        
        Assertions.assertTrue(success, "不同数值仪表盘图片生成失败");
    }
    
    @Test
    @Order(7)
    @DisplayName("测试不同主题高级图表")
    void testAdvancedChartsWithThemes() {
        System.out.println("\n[测试7] 不同主题高级图表");
        
        double[] radarData = {80, 85, 70, 90, 75};
        List<String> indicators = List.of("A", "B", "C", "D", "E");
        String[] themes = {"default", "dark", "futuristic"};
        
        int successCount = 0;
        for (String theme : themes) {
            IPlot plot = Plots.of(800, 600, theme);
            plot.radar(Linalg.vector(radarData), indicators)
                .title("雷达图 - " + theme);
            
            if (generateImage(plot, "advanced_10_theme_" + theme + ".png")) {
                successCount++;
            }
        }
        
        Assertions.assertTrue(successCount >= 2, "至少应生成2个主题图片");
    }
    
    @AfterAll
    static void tearDown() {
        printSummary("高级图表测试");
    }
}
