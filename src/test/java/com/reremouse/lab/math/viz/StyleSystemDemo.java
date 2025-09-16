package com.reremouse.lab.math.viz;

import com.reremouse.lab.math.linalg.RereDoubleVector;
import com.reremouse.lab.math.viz.RerePlot;
import com.reremouse.lab.math.linalg.IDoubleVector;

/**
 * 样式系统演示类
 * 展示新的样式功能，包括matplotlib风格的样式表达式
 * 
 * @author lteb2
 */
public class StyleSystemDemo {
    
    public static void main(String[] args) {
        System.out.println("=== YiShape-Math 样式系统演示 ===");
        
        // 创建测试数据
        double[] xData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        double[] yData1 = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
        double[] yData2 = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
        
        IDoubleVector x = new RereDoubleVector(xData);
        IDoubleVector y1 = new RereDoubleVector(yData1);
        IDoubleVector y2 = new RereDoubleVector(yData2);
        
        demonstrateBasicStyleExpressions(x, y1, y2);
        demonstrateColorPalettes(x, y1, y2);
        demonstrateAdvancedStyling(x, y1, y2);
        demonstrateCompatibility(x, y1, y2);
        
        System.out.println("\n=== 演示完成 ===");
        System.out.println("所有图表文件已生成，可以在浏览器中查看效果。");
    }
    
    /**
     * 演示基础样式表达式
     */
    private static void demonstrateBasicStyleExpressions(IDoubleVector x, IDoubleVector y1, IDoubleVector y2) {
        System.out.println("\n1. 基础样式表达式演示");
        
        try {
            // 红色实线
            RerePlot plot1 = new RerePlot(800, 600);
            plot1.setTitle("基础样式表达式", "matplotlib风格的样式字符串");
            plot1.setXlabel("X轴");
            plot1.setYlabel("Y轴");
            plot1.line(x, y1, "r-");  // 红色实线
            plot1.saveAsHtml("style_red_solid.html");
            System.out.println("   红色实线 (r-): style_red_solid.html");
            
            // 蓝色虚线
            RerePlot plot2 = new RerePlot(800, 600);
            plot2.setTitle("蓝色虚线演示");
            plot2.line(x, y1, "b--");  // 蓝色虚线
            plot2.saveAsHtml("style_blue_dashed.html");
            System.out.println("   蓝色虚线 (b--): style_blue_dashed.html");
            
            // 绿色点线
            RerePlot plot3 = new RerePlot(800, 600);
            plot3.setTitle("绿色点线演示");
            plot3.line(x, y1, "g:");   // 绿色点线
            plot3.saveAsHtml("style_green_dotted.html");
            System.out.println("   绿色点线 (g:): style_green_dotted.html");
            
            // 黑色圆圈标记
            RerePlot plot4 = new RerePlot(800, 600);
            plot4.setTitle("黑色圆圈标记演示");
            plot4.scatter(x, y1, "ko"); // 黑色圆圈
            plot4.saveAsHtml("style_black_circles.html");
            System.out.println("   黑色圆圈 (ko): style_black_circles.html");
            
            // 红色方形标记
            RerePlot plot5 = new RerePlot(800, 600);
            plot5.setTitle("红色方形标记演示");
            plot5.scatter(x, y1, "rs"); // 红色方形
            plot5.saveAsHtml("style_red_squares.html");
            System.out.println("   红色方形 (rs): style_red_squares.html");
            
            // 复合样式：蓝色虚线带圆圈
            RerePlot plot6 = new RerePlot(800, 600);
            plot6.setTitle("复合样式演示", "蓝色虚线带圆圈标记");
            plot6.line(x, y1, "b--o"); // 蓝色虚线带圆圈
            plot6.saveAsHtml("style_blue_dashed_circles.html");
            System.out.println("   蓝色虚线圆圈 (b--o): style_blue_dashed_circles.html");
            
        } catch (Exception e) {
            System.err.println("   基础样式演示出错: " + e.getMessage());
        }
    }
    
    /**
     * 演示颜色调色板
     */
    private static void demonstrateColorPalettes(IDoubleVector x, IDoubleVector y1, IDoubleVector y2) {
        System.out.println("\n2. 颜色调色板演示");
        
        try {
            // matplotlib调色板
            RerePlot plot1 = new RerePlot(800, 600);
            plot1.setTitle("Matplotlib调色板", "C0-C9颜色序列");
            plot1.setPalette("matplotlib");
            plot1.line(x, y1, "C0-");  // matplotlib第0个颜色
            plot1.saveAsHtml("palette_matplotlib_c0.html");
            System.out.println("   Matplotlib C0: palette_matplotlib_c0.html");
            
            // 十六进制颜色
            RerePlot plot2 = new RerePlot(800, 600);
            plot2.setTitle("十六进制颜色演示");
            plot2.line(x, y1, "#FF5733-"); // 橙红色
            plot2.saveAsHtml("palette_hex_color.html");
            System.out.println("   十六进制颜色: palette_hex_color.html");
            
        } catch (Exception e) {
            System.err.println("   调色板演示出错: " + e.getMessage());
        }
    }
    
    /**
     * 演示高级样式功能
     */
    private static void demonstrateAdvancedStyling(IDoubleVector x, IDoubleVector y1, IDoubleVector y2) {
        System.out.println("\n3. 高级样式功能演示");
        
        try {
            // 自定义PlotStyle对象
            PlotStyle customStyle = new PlotStyle()
                    .color("#FF6B6B")
                    .lineStyle("dashed")
                    .lineWidth(3.0f)
                    .marker("s")
                    .markerSize(8.0f)
                    .alpha(0.8f)
                    .label("自定义样式数据");
            
            RerePlot plot1 = new RerePlot(800, 600);
            plot1.setTitle("自定义PlotStyle对象", "完全控制样式参数");
            plot1.line(x, y1, customStyle);
            plot1.saveAsHtml("advanced_custom_style.html");
            System.out.println("   自定义样式: advanced_custom_style.html");
            
            // 渐变和透明度效果
            PlotStyle transparentStyle = PlotStyle.withColor("#4ECDC4")
                    .alpha(0.6f)
                    .lineWidth(4.0f)
                    .label("半透明效果");
            
            RerePlot plot2 = new RerePlot(800, 600);
            plot2.setTitle("透明度效果演示");
            plot2.line(x, y1, transparentStyle);
            plot2.saveAsHtml("advanced_transparency.html");
            System.out.println("   透明度效果: advanced_transparency.html");
            
            // 多种预设样式
            var presets = StyleExpression.createPresetStyles();
            RerePlot plot3 = new RerePlot(800, 600);
            plot3.setTitle("预设样式演示", "使用内置的样式预设");
            plot3.line(x, y1, presets.get("red_line"));
            plot3.saveAsHtml("advanced_preset_styles.html");
            System.out.println("   预设样式: advanced_preset_styles.html");
            
        } catch (Exception e) {
            System.err.println("   高级样式演示出错: " + e.getMessage());
        }
    }
    
    /**
     * 演示向后兼容性
     */
    private static void demonstrateCompatibility(IDoubleVector x, IDoubleVector y1, IDoubleVector y2) {
        System.out.println("\n4. 向后兼容性演示");
        
        try {
            // 原有方法仍然工作
            RerePlot plot1 = new RerePlot(800, 600);
            plot1.setTitle("传统方法", "保持向后兼容");
            plot1.line(x, y1);  // 原有方法，无样式参数
            plot1.saveAsHtml("compatibility_traditional.html");
            System.out.println("   传统方法: compatibility_traditional.html");
            
            // 禁用样式系统
            RerePlot plot2 = new RerePlot(800, 600);
            plot2.enableStyleSystem(false);
            plot2.setTitle("禁用样式系统");
            plot2.line(x, y1, "r-"); // 样式字符串被忽略
            plot2.saveAsHtml("compatibility_disabled.html");
            System.out.println("   禁用样式系统: compatibility_disabled.html");
            
            // 混合使用
            RerePlot plot3 = new RerePlot(800, 600);
            plot3.setTitle("混合使用演示", "新旧方法结合");
            plot3.enableStyleSystem(true);
            plot3.line(x, y1, "g--"); // 新样式方法
            plot3.saveAsHtml("compatibility_mixed.html");
            System.out.println("   混合使用: compatibility_mixed.html");
            
        } catch (Exception e) {
            System.err.println("   兼容性演示出错: " + e.getMessage());
        }
    }
    
    /**
     * 打印样式系统信息
     */
    @SuppressWarnings("unused")
    private static void printStyleSystemInfo() {
        System.out.println("\n=== 样式系统信息 ===");
        
        // 可用调色板
        System.out.println("可用调色板: " + ColorPalette.getAvailablePalettes());
        
        // 预设样式
        System.out.println("\n预设样式:");
        var presets = StyleExpression.createPresetStyles();
        presets.forEach((name, style) -> 
            System.out.println("  " + name + ": " + style));
        
        // 支持的样式表达式
        System.out.println("\n支持的样式表达式示例:");
        String[] examples = {"r-", "b--", "g:", "ko", "rs", "b^", "#FF0000-o", "C0", "C1--"};
        for (String example : examples) {
            if (StyleExpression.isValidStyleString(example)) {
                System.out.println("  " + example + " -> " + 
                    StyleExpression.getStyleDescription(example));
            }
        }
    }
}