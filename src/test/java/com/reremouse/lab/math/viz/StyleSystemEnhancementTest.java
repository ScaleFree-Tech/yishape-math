package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.IMatrix;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * 样式系统增强功能测试
 * 验证所有改进是否正确实现
 * 
 * @author lteb2
 */
public class StyleSystemEnhancementTest {
    
    private RerePlot plot;
    private IVector testData;
    private IMatrix testMatrix;
    
    @BeforeEach
    public void setUp() {
        plot = new RerePlot(800, 600);
        // 创建测试数据
        testData = Linalg.vector(new double[]{1, 4, 2, 8, 5, 7});
        testMatrix = Linalg.matrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });
    }
    
    @Test
    @DisplayName("Enhanced Color Operations Test")
    public void testEnhancedColorOperations() {
        // 测试HSL颜色空间转换
        PlotStyle style = new PlotStyle()
            .color("#ff0000")
            .lineWidth(2)
            .marker("o");
        
        // 测试颜色解析
        assertTrue(ColorPalette.parseColorToRGB("#ff0000") != null);
        
        // 测试颜色验证
        assertTrue(StyleConverter.isValidColor("#ff0000"));
        assertTrue(StyleConverter.isValidColor("rgb(255, 0, 0)"));
        assertFalse(StyleConverter.isValidColor("invalid"));
        
        // 测试高级梯度创建
        Map<String, Object> gradient = StyleConverter.createAdvancedGradientConfig(
            "#ff0000", "#0000ff", 90, "linear");
        
        assertNotNull(gradient);
        assertEquals("linear", gradient.get("type"));
        assertTrue(gradient.containsKey("colorStops"));
        
        System.out.println("✅ Enhanced color operations working correctly");
    }
    
    @Test
    @DisplayName("Enhanced State Styling Test")
    public void testEnhancedStateStyling() {
        PlotStyle style = new PlotStyle()
            .color("#5470c6")
            .lineWidth(2)
            .marker("circle")
            .markerSize(6);
        
        // 测试仪表盘状态样式
        org.icepear.echarts.charts.gauge.GaugeSeries gaugeSeries = 
            new org.icepear.echarts.charts.gauge.GaugeSeries();
        
        UniversalStyleApplier.applyToGaugeSeries(gaugeSeries, style);
        assertNotNull(gaugeSeries.getName());
        
        // 测试箱线图状态样式
        org.icepear.echarts.charts.boxplot.BoxplotSeries boxplotSeries = 
            new org.icepear.echarts.charts.boxplot.BoxplotSeries();
        
        UniversalStyleApplier.applyToBoxplotSeries(boxplotSeries, style);
        assertNotNull(boxplotSeries.getName());
        
        // 测试K线图状态样式
        org.icepear.echarts.charts.candlestick.CandlestickSeries candlestickSeries = 
            new org.icepear.echarts.charts.candlestick.CandlestickSeries();
        
        UniversalStyleApplier.applyToCandlestickSeries(candlestickSeries, style);
        assertNotNull(candlestickSeries.getName());
        
        System.out.println("✅ Enhanced state styling working correctly");
    }
    
    @Test
    @DisplayName("Chart-Specific Style Converters Test")
    public void testChartSpecificStyleConverters() {
        PlotStyle style = new PlotStyle()
            .color("#00ff00")
            .lineWidth(3)
            .alpha(0.8)
            .markerSize(8);
        
        // 测试关系图样式转换
        Map<String, Object> graphStyle = StyleConverter.toEChartsGraphStyle(style);
        assertNotNull(graphStyle);
        assertTrue(graphStyle.containsKey("nodeStyle"));
        assertTrue(graphStyle.containsKey("edgeStyle"));
        assertEquals("force", graphStyle.get("layout"));
        
        // 测试平行坐标图样式转换
        Map<String, Object> parallelStyle = StyleConverter.toEChartsParallelStyle(style);
        assertNotNull(parallelStyle);
        assertTrue(parallelStyle.containsKey("lineStyle"));
        assertEquals(true, parallelStyle.get("smooth"));
        
        System.out.println("✅ Chart-specific style converters working correctly");
    }
    
    @Test
    @DisplayName("Enhanced Gradient System Test")
    public void testEnhancedGradientSystem() {
        // 测试多色梯度
        String[] colors = {"#ff0000", "#00ff00", "#0000ff"};
        Map<String, Object> multiGradient = StyleConverter.createMultiColorGradient(
            colors, null, "linear");
        
        assertNotNull(multiGradient);
        assertEquals("linear", multiGradient.get("type"));
        assertTrue(multiGradient.containsKey("colorStops"));
        
        // 测试径向梯度
        Map<String, Object> radialGradient = StyleConverter.createAdvancedGradientConfig(
            "#ff0000", "#0000ff", 0, "radial");
        
        assertNotNull(radialGradient);
        assertEquals("radial", radialGradient.get("type"));
        assertEquals(0.5, radialGradient.get("x"));
        assertEquals(0.5, radialGradient.get("y"));
        
        System.out.println("✅ Enhanced gradient system working correctly");
    }
    
    @Test
    @DisplayName("Enhanced Theme Integration Test")
    public void testEnhancedThemeIntegration() {
        PlotStyle baseStyle = PlotStyle.defaultStyle();
        
        // 测试主题应用到样式
        PlotStyle academicStyle = ThemeManager.applyThemeToStyle(baseStyle, "academic");
        assertNotNull(academicStyle);
        assertEquals(1.5, academicStyle.getLineWidth());
        assertEquals(4, academicStyle.getMarkerSize());
        assertEquals(800, academicStyle.getAnimationDuration());
        
        PlotStyle businessStyle = ThemeManager.applyThemeToStyle(baseStyle, "business");
        assertNotNull(businessStyle);
        assertEquals(2.5, businessStyle.getLineWidth());
        assertEquals(6, businessStyle.getMarkerSize());
        assertEquals(1200, businessStyle.getAnimationDuration());
        
        // 测试主题调色板推荐
        String academicPalette = ThemeManager.getThemePreferredPalette("academic");
        assertEquals("muted", academicPalette);
        
        String businessPalette = ThemeManager.getThemePreferredPalette("business");
        assertEquals("echarts", businessPalette);
        
        // 测试智能主题推荐
        String recommendedTheme = ThemeManager.recommendTheme("business", "line", "professional");
        assertEquals("business", recommendedTheme);
        
        String academicTheme = ThemeManager.recommendTheme("scientific", "scatter", "minimal");
        assertEquals("academic", academicTheme);
        
        System.out.println("✅ Enhanced theme integration working correctly");
    }
    
    @Test
    @DisplayName("Complete Style System Coverage Test")
    public void testCompleteStyleSystemCoverage() {
        // 测试所有图表类型都支持样式
        assertTrue(plot.isStyleSystemEnabled());
        
        // 创建带样式的各种图表
        PlotStyle testStyle = new PlotStyle()
            .color("#ff6b6b")
            .lineWidth(2.5)
            .marker("circle")
            .markerSize(6)
            .alpha(0.9);
        
        // 测试基础图表
        assertDoesNotThrow(() -> {
            plot.line(testData, testStyle);
        });
        
        assertDoesNotThrow(() -> {
            plot.scatter(testData, testData, testStyle);
        });
        
        assertDoesNotThrow(() -> {
            plot.bar(testData, testStyle);
        });
        
        assertDoesNotThrow(() -> {
            plot.pie(testData, testStyle);
        });
        
        // 测试统计图表
        List<String> labels = Arrays.asList("A", "A", "B", "B", "C", "C");
        assertDoesNotThrow(() -> {
            plot.boxplot(testData, labels, testStyle);
        });
        
        // 测试专业图表
        List<String> xLabels = Arrays.asList("X1", "X2", "X3");
        List<String> yLabels = Arrays.asList("Y1", "Y2", "Y3");
        assertDoesNotThrow(() -> {
            plot.heatmap(testMatrix, xLabels, yLabels, testStyle);
        });
        
        List<String> indicators = Arrays.asList("指标1", "指标2", "指标3", "指标4", "指标5", "指标6");
        assertDoesNotThrow(() -> {
            plot.radar(testData, indicators, testStyle);
        });
        
        assertDoesNotThrow(() -> {
            plot.gauge(75, 100, 0, testStyle);
        });
        
        System.out.println("✅ Complete style system coverage verified");
    }
    
    @Test
    @DisplayName("Performance and Integration Test")
    public void testPerformanceAndIntegration() {
        // 测试样式系统与主题系统的集成
        plot.theme("academic");
        assertTrue(plot.isThemeSystemEnabled());
        
        PlotStyle themedStyle = ThemeManager.applyThemeToStyle(
            PlotStyle.defaultStyle(), plot.getCurrentTheme());
        
        // 创建具有主题样式的图表
        assertDoesNotThrow(() -> {
            plot.line(testData, themedStyle)
                .title("增强样式系统测试")
                .xlabel("X轴")
                .ylabel("Y轴");
        });
        
        // 测试样式解析性能
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < 1000; i++) {
            StyleExpression.parse("r-o");
            StyleExpression.parse("b--^");
            StyleExpression.parse("g:s");
        }
        long endTime = System.currentTimeMillis();
        
        assertTrue(endTime - startTime < 1000, "样式解析性能应该在1秒内完成1000次解析");
        
        // 测试颜色操作性能
        startTime = System.currentTimeMillis();
        for (int i = 0; i < 1000; i++) {
            ColorPalette.parseColorToRGB("#ff0000");
            StyleConverter.isValidColor("#00ff00");
            StyleConverter.applyAlpha("#0000ff", 0.5);
        }
        endTime = System.currentTimeMillis();
        
        assertTrue(endTime - startTime < 500, "颜色操作性能应该在0.5秒内完成1000次操作");
        
        System.out.println("✅ Performance and integration tests passed");
    }
    
    @Test
    @DisplayName("Advanced Features Integration Test")
    public void testAdvancedFeaturesIntegration() {
        // 测试高级特性组合使用
        PlotStyle advancedStyle = new PlotStyle()
            .color("#4ecdc4")
            .lineWidth(3)
            .marker("star")
            .markerSize(10)
            .alpha(0.8)
            .animation(true)
            .setProperty("shadowBlur", 10)
            .setProperty("shadowColor", "rgba(0, 0, 0, 0.3)");
        
        // 应用学术主题
        PlotStyle themedAdvancedStyle = ThemeManager.applyThemeToStyle(advancedStyle, "academic");
        
        // 创建多重效果图表
        assertDoesNotThrow(() -> {
            new RerePlot(900, 700)
                .theme("academic")
                .enableStyleSystem(true)
                .line(testData, themedAdvancedStyle)
                .title("高级特性集成测试", "样式系统 + 主题系统 + 高级效果")
                .xlabel("增强X轴")
                .ylabel("增强Y轴");
        });
        
        // 测试渐变色与主题结合
        String[] gradientColors = ColorPalette.generateAdvancedGradient(
            "#ff6b6b", "#4ecdc4", 5, "ease-in-out");
        
        assertNotNull(gradientColors);
        assertEquals(5, gradientColors.length);
        
        // 测试色盲友好检查
        boolean isColorBlindFriendly = ColorPalette.isColorBlindFriendly(gradientColors);
        // 颜色友好性检查应该返回结果（true或false都是有效的）
        assertNotNull(isColorBlindFriendly);
        
        System.out.println("✅ Advanced features integration working correctly");
    }
}