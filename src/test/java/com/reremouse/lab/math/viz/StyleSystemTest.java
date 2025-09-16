package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.linalg.RereDoubleVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import com.reremouse.lab.math.viz.PlotStyle;
import com.reremouse.lab.math.viz.StyleExpression;
import com.reremouse.lab.math.viz.ColorPalette;
import com.reremouse.lab.math.linalg.IDoubleVector;

/**
 * 样式系统测试类
 * 测试PlotStyle、StyleExpression、ColorPalette等新样式功能
 * 
 * @author lteb2
 */
public class StyleSystemTest {
    
    private IDoubleVector testX;
    private IDoubleVector testY;
    
    @BeforeEach
    void setUp() {
        double[] xData = {1, 2, 3, 4, 5};
        double[] yData = {2, 4, 6, 8, 10};
        testX = new RereDoubleVector(xData);
        testY = new RereDoubleVector(yData);
    }
    
    @Test
    void testPlotStyleCreation() {
        System.out.println("=== 测试PlotStyle创建 ===");
        
        // 测试默认样式
        PlotStyle defaultStyle = PlotStyle.defaultStyle();
        assertNotNull(defaultStyle);
        assertEquals("#5470c6", defaultStyle.getColor());
        assertEquals("solid", defaultStyle.getLineStyle());
        assertEquals(2.0f, defaultStyle.getLineWidth());
        assertEquals("o", defaultStyle.getMarker());
        
        System.out.println("默认样式: " + defaultStyle);
        
        // 测试自定义样式
        PlotStyle customStyle = new PlotStyle()
                .color("#FF0000")
                .lineStyle("dashed")
                .lineWidth(3.0f)
                .marker("s")
                .markerSize(8.0f)
                .alpha(0.8f)
                .label("测试数据");
        
        assertEquals("#FF0000", customStyle.getColor());
        assertEquals("dashed", customStyle.getLineStyle());
        assertEquals(3.0f, customStyle.getLineWidth());
        assertEquals("s", customStyle.getMarker());
        assertEquals(8.0f, customStyle.getMarkerSize());
        assertEquals(0.8f, customStyle.getAlpha());
        assertEquals("测试数据", customStyle.getLabel());
        
        System.out.println("自定义样式: " + customStyle);
    }
    
    @Test
    void testStyleExpressionParsing() {
        System.out.println("\n=== 测试StyleExpression解析 ===");
        
        // 测试简单样式表达式
        testStyleExpression("r-", "#FF0000", "solid", "circle");
        testStyleExpression("b--", "#0000FF", "dashed", "circle");
        testStyleExpression("g:", "#00FF00", "dotted", "circle");
        testStyleExpression("ko", "#000000", "solid", "circle");
        testStyleExpression("rs", "#FF0000", "solid", "rect");
        testStyleExpression("b^", "#0000FF", "solid", "triangle");
        
        // 测试复合表达式
        testStyleExpression("r--o", "#FF0000", "dashed", "circle");
        testStyleExpression("g:s", "#00FF00", "dotted", "rect");
        
        // 测试C0-C9颜色
        PlotStyle c0Style = StyleExpression.parse("C0-");
        assertEquals("#1f77b4", c0Style.getColor());
        assertEquals("solid", c0Style.getLineStyle());
        
        // 测试十六进制颜色
        PlotStyle hexStyle = StyleExpression.parse("#FF5733-o");
        assertEquals("#FF5733", hexStyle.getColor());
        assertEquals("solid", hexStyle.getLineStyle());
        assertEquals("circle", hexStyle.getMarker());
        
        System.out.println("样式表达式解析测试完成");
    }
    
    private void testStyleExpression(String expression, String expectedColor, 
                                   String expectedLineStyle, String expectedMarker) {
        PlotStyle style = StyleExpression.parse(expression);
        assertNotNull(style);
        
        if (expectedColor != null) {
            assertEquals(expectedColor, style.getColor(), 
                "表达式 '" + expression + "' 的颜色不匹配");
        }
        if (expectedLineStyle != null) {
            assertEquals(expectedLineStyle, style.getLineStyle(), 
                "表达式 '" + expression + "' 的线条样式不匹配");
        }
        if (expectedMarker != null) {
            assertEquals(expectedMarker, style.getMarker(), 
                "表达式 '" + expression + "' 的标记不匹配");
        }
        
        System.out.println("  " + expression + " -> " + style);
    }
    
    @Test
    void testColorPalette() {
        System.out.println("\n=== 测试ColorPalette ===");
        
        // 测试基础颜色
        assertEquals("#FF0000", ColorPalette.parseColor("r"));
        assertEquals("#FF0000", ColorPalette.parseColor("red"));
        assertEquals("#0000FF", ColorPalette.parseColor("b"));
        assertEquals("#0000FF", ColorPalette.parseColor("blue"));
        
        // 测试C0-C9颜色
        assertEquals("#1f77b4", ColorPalette.parseColor("C0"));
        assertEquals("#ff7f0e", ColorPalette.parseColor("C1"));
        
        // 测试十六进制颜色
        assertEquals("#FF5733", ColorPalette.parseColor("#FF5733"));
        
        // 测试调色板
        String[] echartsPalette = ColorPalette.getPalette("echarts");
        assertNotNull(echartsPalette);
        assertTrue(echartsPalette.length > 0);
        
        String[] matplotlibPalette = ColorPalette.getPalette("matplotlib");
        assertNotNull(matplotlibPalette);
        assertEquals(10, matplotlibPalette.length);
        
        System.out.println("ECharts调色板: " + java.util.Arrays.toString(echartsPalette));
        System.out.println("Matplotlib调色板: " + java.util.Arrays.toString(matplotlibPalette));
        
        // 测试调色板颜色获取
        for (int i = 0; i < 5; i++) {
            String color = ColorPalette.getColor("matplotlib", i);
            assertNotNull(color);
            System.out.println("Matplotlib[" + i + "]: " + color);
        }
    }
    
    @Test
    void testStyleConverter() {
        System.out.println("\n=== 测试StyleConverter ===");
        
        PlotStyle style = new PlotStyle()
                .color("#FF0000")
                .lineStyle("dashed")
                .lineWidth(3.0f)
                .marker("s")
                .alpha(0.7f);
        
        // 测试线条样式转换
        org.icepear.echarts.components.series.LineStyle lineStyle = 
                StyleConverter.toEChartsLineStyle(style);
        assertNotNull(lineStyle);
        
        // 测试区域样式转换
        org.icepear.echarts.charts.line.LineAreaStyle areaStyle = 
                StyleConverter.toEChartsAreaStyle(style);
        assertNotNull(areaStyle);
        
        // 测试标记转换
        String symbol = StyleConverter.convertMarkerToSymbol("s");
        assertEquals("rect", symbol);
        
        symbol = StyleConverter.convertMarkerToSymbol("o");
        assertEquals("circle", symbol);
        
        symbol = StyleConverter.convertMarkerToSymbol("^");
        assertEquals("triangle", symbol);
        
        // 测试颜色验证
        assertTrue(StyleConverter.isValidColor("#FF0000"));
        assertTrue(StyleConverter.isValidColor("#F00"));
        assertTrue(StyleConverter.isValidColor("rgb(255, 0, 0)"));
        assertTrue(StyleConverter.isValidColor("rgba(255, 0, 0, 0.5)"));
        assertFalse(StyleConverter.isValidColor("invalid"));
        
        System.out.println("StyleConverter测试完成");
    }
    
    @Test
    void testRerePlotStyleIntegration() {
        System.out.println("\n=== 测试RerePlot样式集成 ===");
        
        try {
            RerePlot plot = new RerePlot();
            
            // 测试样式系统是否启用
            assertTrue(plot.isStyleSystemEnabled());
            
            // 测试设置默认样式
            PlotStyle customDefault = PlotStyle.withColor("#FF0000");
            plot.setDefaultStyle(customDefault);
            assertEquals(customDefault, plot.getDefaultStyle());
            
            // 测试设置调色板
            plot.setPalette("matplotlib");
            assertEquals("matplotlib", plot.getPalette());
            
            // 测试样式字符串解析
            PlotStyle parsedStyle = plot.parseStyle("r-o");
            assertNotNull(parsedStyle);
            assertEquals("#FF0000", parsedStyle.getColor());
            assertEquals("solid", parsedStyle.getLineStyle());
            assertEquals("circle", parsedStyle.getMarker());
            
            // 测试创建自定义样式
            PlotStyle createdStyle = RerePlot.createStyle("#00FF00", "dashed", "s");
            assertEquals("#00FF00", createdStyle.getColor());
            assertEquals("dashed", createdStyle.getLineStyle());
            assertEquals("s", createdStyle.getMarker());
            
            System.out.println("RerePlot样式集成测试完成");
            
        } catch (Exception e) {
            System.err.println("RerePlot样式集成测试出错: " + e.getMessage());
        }
    }
    
    @Test
    void testStyleStringValidation() {
        System.out.println("\n=== 测试样式字符串验证 ===");
        
        // 有效的样式字符串
        assertTrue(StyleExpression.isValidStyleString("r-"));
        assertTrue(StyleExpression.isValidStyleString("b--o"));
        assertTrue(StyleExpression.isValidStyleString("g:^"));
        assertTrue(StyleExpression.isValidStyleString("#FF0000-s"));
        assertTrue(StyleExpression.isValidStyleString("C0"));
        assertTrue(StyleExpression.isValidStyleString(""));
        
        // 获取样式描述
        String desc = StyleExpression.getStyleDescription("r-o");
        assertNotNull(desc);
        System.out.println("'r-o' 描述: " + desc);
        
        desc = StyleExpression.getStyleDescription("b--^");
        assertNotNull(desc);
        System.out.println("'b--^' 描述: " + desc);
        
        System.out.println("样式字符串验证测试完成");
    }
    
    @Test
    void testPresetStyles() {
        System.out.println("\n=== 测试预设样式 ===");
        
        var presets = StyleExpression.createPresetStyles();
        assertNotNull(presets);
        assertFalse(presets.isEmpty());
        
        // 测试预设样式
        PlotStyle redLine = presets.get("red_line");
        assertNotNull(redLine);
        assertEquals("#FF0000", redLine.getColor());
        assertEquals("solid", redLine.getLineStyle());
        
        PlotStyle blueDashed = presets.get("blue_dashed");
        assertNotNull(blueDashed);
        assertEquals("#0000FF", blueDashed.getColor());
        assertEquals("dashed", blueDashed.getLineStyle());
        
        // 输出所有预设样式
        System.out.println("预设样式列表:");
        presets.forEach((name, style) -> {
            System.out.println("  " + name + ": " + style);
        });
        
        System.out.println("预设样式测试完成");
    }
}