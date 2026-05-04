package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.IPlot;
import com.yishape.lab.math.plot.Plots;
import com.yishape.lab.math.plot.javafx.base.JavaFxChartTestBase;
import org.junit.jupiter.api.*;

import java.util.List;

/**
 * 线图测试类
 * 测试线图的所有功能
 * 
 * @author lteb2
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class LineChartTest extends JavaFxChartTestBase {

    /** Mauna Loa 月均 CO₂（ppm），NOAA GML，2019 自然年 12 个月近似公布的月均值。 */
    private static final double[] CO2_MONTH_INDEX = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    private static final double[] CO2_PPM = {
        410.83, 411.75, 411.97, 413.32, 414.66, 415.58,
        416.08, 414.38, 411.51, 413.30, 414.74, 416.06,
    };

    @BeforeAll
    static void setUp() {
        initialize();
        resetCounters();
        System.out.println("\n========== 线图测试开始 ==========");
    }
    
    @Test
    @Order(1)
    @DisplayName("测试基础双向量线图")
    void testBasicLineChart() {
        System.out.println("\n[测试1] 基础双向量线图");
        IPlot plot = Plots.of(800, 600);
        plot.line(Linalg.vector(CO2_MONTH_INDEX), Linalg.vector(CO2_PPM))
            .title("Mauna Loa CO₂ 月均值", "ppm · 2019 示意")
            .xlabel("Month of year (1–12)")
            .ylabel("CO₂ (ppm)");
        
        boolean success = generateImage(plot, "line_01_basic.png");
        Assertions.assertTrue(success, "基础线图图片生成失败");
    }
    
    @Test
    @Order(2)
    @DisplayName("测试单向量线图")
    void testSingleVectorLineChart() {
        System.out.println("\n[测试2] 单向量线图");
        // BTC/USD 日收盘示意（量级真实，非逐笔链上数据）
        double[] yData = {
            43200, 43850, 42100, 40980, 41520, 42800, 44100,
            43650, 45200, 44800, 46100, 47500, 46800, 48200,
        };
        
        IPlot plot = Plots.of(800, 600);
        plot.line(Linalg.vector(yData))
            .title("单向量线图", "BTCUSD 收盘示意 · 序号为交易日");
        
        boolean success = generateImage(plot, "line_02_single_vector.png");
        Assertions.assertTrue(success, "单向量线图图片生成失败");
    }
    
    @Test
    @Order(3)
    @DisplayName("测试多系列线图")
    void testMultiSeriesLineChart() {
        System.out.println("\n[测试3] 多系列线图");
        // 同一周内两台风机功率（MW），按 12h 采样
        double[] xData = {0, 12, 24, 36, 48, 60, 72, 84, 96};
        double[] yData = {2.1, 2.8, 1.9, 3.2, 2.6, 1.4, 2.0, 3.0, 2.3};
        List<String> hue = List.of("Unit_A", "Unit_A", "Unit_A", "Unit_B", "Unit_B", "Unit_A", "Unit_A", "Unit_B", "Unit_B");
        
        IPlot plot = Plots.of(800, 600);
        plot.line(Linalg.vector(xData), Linalg.vector(yData), hue)
            .title("风电场出力", "hue = 机组")
            .xlabel("Hour from start")
            .ylabel("Power (MW)");
        
        boolean success = generateImage(plot, "line_03_multi_series.png");
        Assertions.assertTrue(success, "多系列线图图片生成失败");
    }
    
    @Test
    @Order(4)
    @DisplayName("测试不同主题线图")
    void testLineChartWithThemes() {
        System.out.println("\n[测试4] 不同主题线图");
        
        double[] week = {1, 2, 3, 4, 5};
        double[] visitorsK = {142, 156, 148, 173, 161};
        String[] themes = {"default", "dark", "light", "academic", "rainbow"};

        int successCount = 0;
        for (String theme : themes) {
            IPlot plot = Plots.of(800, 600, theme);
            plot.line(Linalg.vector(week), Linalg.vector(visitorsK))
                .title("线图 - " + theme + "主题", "博物馆日客流（千人·示意）");
            
            if (generateImage(plot, "line_04_theme_" + theme + ".png")) {
                successCount++;
            }
        }
        
        Assertions.assertTrue(successCount >= themes.length - 1, 
            "至少应生成 " + (themes.length - 1) + " 个主题图片，实际成功 " + successCount + " 个");
    }
    
    @Test
    @Order(5)
    @DisplayName("测试不同尺寸线图")
    void testLineChartWithSizes() {
        System.out.println("\n[测试5] 不同尺寸线图");
        
        double[] xData = {2019, 2020, 2021, 2022, 2023};
        double[] yData = {2.1, 2.3, 2.8, 3.0, 3.2};
        int[][] sizes = {{400, 300}, {800, 600}, {1200, 900}};
        
        int successCount = 0;
        for (int i = 0; i < sizes.length; i++) {
            IPlot plot = Plots.of(sizes[i][0], sizes[i][1]);
            plot.line(Linalg.vector(xData), Linalg.vector(yData))
                .title("线图 - " + sizes[i][0] + "x" + sizes[i][1], "全球光伏年装机 TW·示意");
            
            if (generateImage(plot, "line_05_size_" + sizes[i][0] + "x" + sizes[i][1] + ".png")) {
                successCount++;
            }
        }
        
        Assertions.assertTrue(successCount >= 2, 
            "至少应生成2个尺寸图片，实际成功 " + successCount + " 个");
    }
    
    @AfterAll
    static void tearDown() {
        printSummary("线图测试");
    }
}
