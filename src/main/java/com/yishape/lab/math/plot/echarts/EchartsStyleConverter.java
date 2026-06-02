package com.yishape.lab.math.plot.echarts;

import com.yishape.lab.math.plot.ColorPalette;
import com.yishape.lab.math.plot.PlotStyle;
import org.icepear.echarts.components.series.LineStyle;
import org.icepear.echarts.components.series.ItemStyle;
import org.icepear.echarts.charts.line.LineAreaStyle;
import org.icepear.echarts.charts.bar.BarItemStyle;
import org.icepear.echarts.charts.pie.PieItemStyle;
import org.icepear.echarts.charts.candlestick.CandlestickItemStyle;
import org.icepear.echarts.charts.treemap.TreemapSeriesItemStyle;
import org.icepear.echarts.charts.sunburst.SunburstItemStyle;

import java.io.Serializable;
import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

/**
 * 增强的样式转换工具类，负责将PlotStyle转换为ECharts相关的样式配置
 * 支持完整的ECharts样式特性，包括itemStyle、emphasis、blur、select等状态样式
 * 
 * @author lteb2
 */
public class EchartsStyleConverter implements Serializable {
    
    /**
     * 将PlotStyle转换为ECharts LineStyle（增强版）
     * @param plotStyle 绘图样式
     * @return ECharts线条样式
     */
    public static LineStyle toEChartsLineStyle(PlotStyle plotStyle) {
        LineStyle lineStyle = new LineStyle();
        
        if (plotStyle.getColor() != null) {
            lineStyle.setColor(plotStyle.getColor());
        }
        
        lineStyle.setWidth((int) plotStyle.getLineWidth());
        
        // 转换线条类型
        String lineType = convertLineStyleType(plotStyle.getLineStyle());
        lineStyle.setType(lineType);
        
        // 设置透明度
        if (plotStyle.getAlpha() < 1.0) {
            lineStyle.setOpacity(plotStyle.getAlpha());
        }
        
        // 设置线条端点样式
        if (plotStyle.getLineCap() != null) {
            lineStyle.setCap(convertLineCap(plotStyle.getLineCap()));
        }
        
        // 设置线条连接样式
        if (plotStyle.getLineJoin() != null) {
            lineStyle.setJoin(convertLineJoin(plotStyle.getLineJoin()));
        }
        
        // 设置阴影效果
        applyShadowToLineStyle(lineStyle, plotStyle);
        
        return lineStyle;
    }
    
    /**
     * 将PlotStyle转换为ECharts ItemStyle（通用）
     * @param plotStyle 绘图样式
     * @return ECharts通用项样式
     */
    public static ItemStyle toEChartsItemStyle(PlotStyle plotStyle) {
        ItemStyle itemStyle = new ItemStyle();
        
        // 设置颜色
        if (plotStyle.getColor() != null) {
            itemStyle.setColor(plotStyle.getColor());
        }
        
        // 设置透明度
        if (plotStyle.getAlpha() < 1.0) {
            itemStyle.setOpacity(plotStyle.getAlpha());
        }
        
        // 设置边框样式
        if (plotStyle.getEdgeColor() != null) {
            itemStyle.setBorderColor(plotStyle.getEdgeColor());
        }
        
        if (plotStyle.getLineWidth() > 0) {
            itemStyle.setBorderWidth(plotStyle.getLineWidth());
        }
        
        // 设置边框类型
        if (plotStyle.getLineStyle() != null) {
            itemStyle.setBorderType(convertBorderType(plotStyle.getLineStyle()));
        }
        
        // 设置阴影效果
        applyShadowToItemStyle(itemStyle, plotStyle);
        
        return itemStyle;
    }
    
    /**
     * 将PlotStyle转换为ECharts BarItemStyle
     * @param plotStyle 绘图样式
     * @return ECharts柱状图项样式
     */
    public static BarItemStyle toEChartsBarItemStyle(PlotStyle plotStyle) {
        BarItemStyle itemStyle = new BarItemStyle();
        
        // 设置基础颜色
        if (plotStyle.getColor() != null) {
            itemStyle.setColor(plotStyle.getColor());
        }
        
        // 设置透明度
        if (plotStyle.getAlpha() < 1.0) {
            itemStyle.setOpacity(plotStyle.getAlpha());
        }
        
        // 设置边框样式
        if (plotStyle.getEdgeColor() != null) {
            itemStyle.setBorderColor(plotStyle.getEdgeColor());
        }
        
        if (plotStyle.getLineWidth() > 0) {
            itemStyle.setBorderWidth(plotStyle.getLineWidth());
        }
        
        // 设置边框类型
        if (plotStyle.getLineStyle() != null) {
            itemStyle.setBorderType(convertBorderType(plotStyle.getLineStyle()));
        }
        
        // 设置阴影效果
        applyShadowToBarItemStyle(itemStyle, plotStyle);
        
        return itemStyle;
    }
    
    /**
     * 将PlotStyle转换为ECharts PieItemStyle
     * @param plotStyle 绘图样式
     * @return ECharts饼图项样式
     */
    public static PieItemStyle toEChartsPieItemStyle(PlotStyle plotStyle) {
        PieItemStyle itemStyle = new PieItemStyle();
        
        // 设置基础颜色
        if (plotStyle.getColor() != null) {
            itemStyle.setColor(plotStyle.getColor());
        }
        
        // 设置透明度
        if (plotStyle.getAlpha() < 1.0) {
            itemStyle.setOpacity(plotStyle.getAlpha());
        }
        
        // 设置边框样式
        if (plotStyle.getEdgeColor() != null) {
            itemStyle.setBorderColor(plotStyle.getEdgeColor());
        }
        
        if (plotStyle.getLineWidth() > 0) {
            itemStyle.setBorderWidth(plotStyle.getLineWidth());
        }
        
        // 设置阴影效果
        applyShadowToPieItemStyle(itemStyle, plotStyle);
        
        return itemStyle;
    }
    
    /**
     * 将PlotStyle转换为ECharts CandlestickItemStyle
     * @param plotStyle 绘图样式
     * @return ECharts K线图项样式
     */
    public static CandlestickItemStyle toEChartsCandlestickItemStyle(PlotStyle plotStyle) {
        CandlestickItemStyle itemStyle = new CandlestickItemStyle();
        
        // 设置基础颜色
        if (plotStyle.getColor() != null) {
            itemStyle.setColor(plotStyle.getColor());
        }
        
        // 设置透明度
        if (plotStyle.getAlpha() < 1.0) {
            itemStyle.setOpacity(plotStyle.getAlpha());
        }
        
        // 设置边框样式
        if (plotStyle.getEdgeColor() != null) {
            itemStyle.setBorderColor(plotStyle.getEdgeColor());
        }
        
        if (plotStyle.getLineWidth() > 0) {
            itemStyle.setBorderWidth(plotStyle.getLineWidth());
        }
        
        // 设置阴影效果
        applyShadowToCandlestickItemStyle(itemStyle, plotStyle);
        
        return itemStyle;
    }
    
    /**
     * 将PlotStyle转换为ECharts TreemapSeriesItemStyle
     * @param plotStyle 绘图样式
     * @return ECharts矩形树图项样式
     */
    public static TreemapSeriesItemStyle toEChartsTreemapItemStyle(PlotStyle plotStyle) {
        TreemapSeriesItemStyle itemStyle = new TreemapSeriesItemStyle();
        
        // 设置基础颜色
        if (plotStyle.getColor() != null) {
            itemStyle.setColor(plotStyle.getColor());
        }
        
        // 设置透明度
        if (plotStyle.getAlpha() < 1.0) {
            itemStyle.setOpacity(plotStyle.getAlpha());
        }
        
        // 设置边框样式
        if (plotStyle.getEdgeColor() != null) {
            itemStyle.setBorderColor(plotStyle.getEdgeColor());
        }
        
        if (plotStyle.getLineWidth() > 0) {
            itemStyle.setBorderWidth(plotStyle.getLineWidth());
        }
        
        // 设置阴影效果
        applyShadowToTreemapItemStyle(itemStyle, plotStyle);
        
        return itemStyle;
    }
    
    /**
     * 将PlotStyle转换为ECharts SunburstItemStyle
     * @param plotStyle 绘图样式
     * @return ECharts旭日图项样式
     */
    public static SunburstItemStyle toEChartsSunburstItemStyle(PlotStyle plotStyle) {
        SunburstItemStyle itemStyle = new SunburstItemStyle();
        
        // 设置基础颜色
        if (plotStyle.getColor() != null) {
            itemStyle.setColor(plotStyle.getColor());
        }
        
        // 设置透明度
        if (plotStyle.getAlpha() < 1.0) {
            itemStyle.setOpacity(plotStyle.getAlpha());
        }
        
        // 设置边框样式
        if (plotStyle.getEdgeColor() != null) {
            itemStyle.setBorderColor(plotStyle.getEdgeColor());
        }
        
        if (plotStyle.getLineWidth() > 0) {
            itemStyle.setBorderWidth(plotStyle.getLineWidth());
        }
        
        // 设置阴影效果
        applyShadowToSunburstItemStyle(itemStyle, plotStyle);
        
        return itemStyle;
    }
    
    /**
     * 将PlotStyle转换为ECharts热力图特定样式
     * @param plotStyle 绘图样式
     * @return 热力图样式配置
     */
    public static Map<String, Object> toEChartsHeatmapStyle(PlotStyle plotStyle) {
        Map<String, Object> heatmapStyle = new HashMap<>();
        
        // 设置基础颜色
        if (plotStyle.getColor() != null) {
            heatmapStyle.put("color", plotStyle.getColor());
        }
        
        // 设置透明度
        if (plotStyle.getAlpha() < 1.0) {
            heatmapStyle.put("opacity", plotStyle.getAlpha());
        }
        
        // 设置边框样式
        if (plotStyle.getEdgeColor() != null) {
            heatmapStyle.put("borderColor", plotStyle.getEdgeColor());
            heatmapStyle.put("borderWidth", plotStyle.getLineWidth());
        }
        
        // 设置渐变效果
        Object gradientConfig = plotStyle.getProperty("gradient");
        if (gradientConfig != null) {
            heatmapStyle.put("color", gradientConfig);
        }
        
        // 设置阴影效果
        applyShadowToMap(heatmapStyle, plotStyle);
        
        return heatmapStyle;
    }
    /**
     * 将PlotStyle转换为ECharts AreaStyle（用于填充）
     * @param plotStyle 绘图样式
     * @return ECharts区域样式
     */
    public static LineAreaStyle toEChartsAreaStyle(PlotStyle plotStyle) {
        LineAreaStyle areaStyle = new LineAreaStyle();
        
        if (plotStyle.getFaceColor() != null) {
            areaStyle.setColor(plotStyle.getFaceColor());
        }
        
        areaStyle.setOpacity(plotStyle.getAlpha());
        
        return areaStyle;
    }
    
    /**
     * 应用阴影效果到Map配置
     * @param styleMap 样式Map对象
     * @param plotStyle 绘图样式
     */
    private static void applyShadowToMap(Map<String, Object> styleMap, PlotStyle plotStyle) {
        // 检查是否有阴影配置
        Object shadowBlur = plotStyle.getProperty("shadowBlur");
        Object shadowColor = plotStyle.getProperty("shadowColor");
        Object shadowOffsetX = plotStyle.getProperty("shadowOffsetX");
        Object shadowOffsetY = plotStyle.getProperty("shadowOffsetY");
        
        if (shadowBlur != null) {
            styleMap.put("shadowBlur", shadowBlur);
        }
        
        if (shadowColor != null) {
            styleMap.put("shadowColor", shadowColor);
        }
        
        if (shadowOffsetX != null) {
            styleMap.put("shadowOffsetX", shadowOffsetX);
        }
        
        if (shadowOffsetY != null) {
            styleMap.put("shadowOffsetY", shadowOffsetY);
        }
    }
    
    /**
     * 为仪表盘创建渐变色配置
     * @param plotStyle 绘图样式
     * @return 渐变色配置
     */
    private static Object createGradientForGauge(PlotStyle plotStyle) {
        if (plotStyle.getColor() != null) {
            // 创建仪表盘特有的渐变配置
            List<List<Object>> gradientStops = new ArrayList<>();
            gradientStops.add(List.of(0.3, plotStyle.getColor()));
            gradientStops.add(List.of(0.7, lightenColor(plotStyle.getColor(), 0.2)));
            gradientStops.add(List.of(1, plotStyle.getColor()));
            return gradientStops;
        }
        
        // 默认渐变
        List<List<Object>> defaultGradient = new ArrayList<>();
        defaultGradient.add(List.of(0.3, "#67e0e3"));
        defaultGradient.add(List.of(0.7, "#37a2da"));
        defaultGradient.add(List.of(1, "#fd666d"));
        return defaultGradient;
    }
    
    /**
     * 为关系图创建特定样式配置
     * @param plotStyle 绘图样式
     * @return 关系图样式配置
     */
    public static Map<String, Object> toEChartsGraphStyle(PlotStyle plotStyle) {
        Map<String, Object> graphStyle = new HashMap<>();
        
        // 设置节点样式
        Map<String, Object> nodeStyle = new HashMap<>();
        if (plotStyle.getColor() != null) {
            nodeStyle.put("color", plotStyle.getColor());
        }
        nodeStyle.put("symbolSize", plotStyle.getMarkerSize());
        
        // 设置边样式
        Map<String, Object> edgeStyle = new HashMap<>();
        if (plotStyle.getEdgeColor() != null) {
            edgeStyle.put("color", plotStyle.getEdgeColor());
        }
        edgeStyle.put("width", plotStyle.getLineWidth());
        edgeStyle.put("opacity", plotStyle.getAlpha());
        
        graphStyle.put("nodeStyle", nodeStyle);
        graphStyle.put("edgeStyle", edgeStyle);
        
        // 设置布局配置
        graphStyle.put("layout", "force");
        graphStyle.put("roam", true);
        graphStyle.put("focusNodeAdjacency", true);
        
        return graphStyle;
    }
    
    /**
     * 为平行坐标图创建特定样式配置
     * @param plotStyle 绘图样式
     * @return 平行坐标图样式配置
     */
    public static Map<String, Object> toEChartsParallelStyle(PlotStyle plotStyle) {
        Map<String, Object> parallelStyle = new HashMap<>();
        
        // 设置线条样式
        Map<String, Object> lineStyle = new HashMap<>();
        if (plotStyle.getColor() != null) {
            lineStyle.put("color", plotStyle.getColor());
        }
        lineStyle.put("width", plotStyle.getLineWidth());
        lineStyle.put("opacity", plotStyle.getAlpha());
        lineStyle.put("type", convertLineStyleType(plotStyle.getLineStyle()));
        
        parallelStyle.put("lineStyle", lineStyle);
        
        // 设置平滑效果
        parallelStyle.put("smooth", true);
        
        return parallelStyle;
    }
    
    /**
     * 增亮颜色（HSL色彩空间实现）
     * @param color 原始颜色
     * @param factor 增亮因子 (0-1)
     * @return 增亮后的颜色
     */
    private static String lightenColor(String color, double factor) {
        if (color == null || !color.startsWith("#")) {
            return color;
        }
        
        try {
            int[] rgb = ColorPalette.parseColorToRGB(color);
            if (rgb == null) return color;
            
            // 转换到HSL色彩空间
            double[] hsl = rgbToHsl(rgb[0], rgb[1], rgb[2]);
            
            // 增加亮度
            hsl[2] = Math.min(1.0, hsl[2] + (1.0 - hsl[2]) * factor);
            
            // 转换回RGB
            int[] newRgb = hslToRgb(hsl[0], hsl[1], hsl[2]);
            
            return String.format("#%02x%02x%02x", newRgb[0], newRgb[1], newRgb[2]);
        } catch (Exception e) {
            return color;
        }
    }
    
    /**
     * RGB转HSL色彩空间
     * @param r 红色值 (0-255)
     * @param g 绿色值 (0-255)
     * @param b 蓝色值 (0-255)
     * @return HSL数组 [H(0-360), S(0-1), L(0-1)]
     */
    private static double[] rgbToHsl(int r, int g, int b) {
        double rNorm = r / 255.0;
        double gNorm = g / 255.0;
        double bNorm = b / 255.0;
        
        double max = Math.max(Math.max(rNorm, gNorm), bNorm);
        double min = Math.min(Math.min(rNorm, gNorm), bNorm);
        
        double h, s, l;
        l = (max + min) / 2.0;
        
        if (max == min) {
            h = s = 0; // achromatic
        } else {
            double d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            
            if (max == rNorm) {
                h = (gNorm - bNorm) / d + (gNorm < bNorm ? 6 : 0);
            } else if (max == gNorm) {
                h = (bNorm - rNorm) / d + 2;
            } else {
                h = (rNorm - gNorm) / d + 4;
            }
            h /= 6;
        }
        
        return new double[]{h * 360, s, l};
    }
    
    /**
     * HSL转RGB色彩空间
     * @param h 色相 (0-360)
     * @param s 饱和度 (0-1)
     * @param l 亮度 (0-1)
     * @return RGB数组 [R, G, B]
     */
    private static int[] hslToRgb(double h, double s, double l) {
        h = h / 360.0;
        
        double r, g, b;
        
        if (s == 0) {
            r = g = b = l; // achromatic
        } else {
            double q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            double p = 2 * l - q;
            r = hueToRgb(p, q, h + 1.0/3);
            g = hueToRgb(p, q, h);
            b = hueToRgb(p, q, h - 1.0/3);
        }
        
        return new int[]{
            Math.max(0, Math.min(255, (int) Math.round(r * 255))),
            Math.max(0, Math.min(255, (int) Math.round(g * 255))),
            Math.max(0, Math.min(255, (int) Math.round(b * 255)))
        };
    }
    
    /**
     * 色相转RGB分量
     */
    private static double hueToRgb(double p, double q, double t) {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1.0/6) return p + (q - p) * 6 * t;
        if (t < 1.0/2) return q;
        if (t < 2.0/3) return p + (q - p) * (2.0/3 - t) * 6;
        return p;
    }
    
    /**
     * 转换线条样式类型
     * @param plotLineStyle 绘图线条样式
     * @return ECharts线条类型
     */
    private static String convertLineStyleType(String plotLineStyle) {
        if (plotLineStyle == null) {
            return "solid";
        }
        
        switch (plotLineStyle.toLowerCase()) {
            case "solid":
                return "solid";
            case "dashed":
                return "dashed";
            case "dotted":
                return "dotted";
            case "dashdot":
                return "solid"; // ECharts不直接支持dashdot，使用solid
            default:
                return "solid";
        }
    }
    
    /**
     * 转换标记样式为ECharts符号
     * @param marker 标记样式
     * @return ECharts符号
     */
    public static String convertMarkerToSymbol(String marker) {
        if (marker == null) {
            return "circle";
        }
        
        switch (marker.toLowerCase()) {
            case "circle":
            case "o":
                return "circle";
            case "rect":
            case "square":
            case "s":
                return "rect";
            case "triangle":
            case "^":
            case "v":
            case "<":
            case ">":
                return "triangle";
            case "diamond":
            case "d":
                return "diamond";
            case "star":
            case "*":
                return "star";
            case "plus":
            case "+":
                return "plus";
            case "cross":
            case "x":
                return "cross";
            case "line":
            case "|":
            case "_":
                return "line";
            default:
                return "circle";
        }
    }
    
    /**
     * 获取对比色（用于文本等）
     * @param backgroundColor 背景颜色
     * @return 对比色
     */
    public static String getContrastColor(String backgroundColor) {
        return ColorPalette.getContrastColor(backgroundColor);
    }
    
    /**
     * 将颜色透明度应用到颜色值
     * @param color 原始颜色
     * @param alpha 透明度 (0.0-1.0)
     * @return 带透明度的颜色
     */
    public static String applyAlpha(String color, double alpha) {
        if (color == null) {
            return null;
        }
        
        // 如果已经是rgba格式，直接返回
        if (color.toLowerCase().startsWith("rgba(")) {
            return color;
        }
        
        // 如果是rgb格式，转换为rgba
        if (color.toLowerCase().startsWith("rgb(")) {
            String rgbPart = color.substring(4, color.length() - 1);
            return String.format("rgba(%s, %.2f)", rgbPart, alpha);
        }
        
        // 如果是十六进制格式，转换为rgba
        if (color.startsWith("#")) {
            String hex = color.substring(1);
            if (hex.length() == 3) {
                // 扩展3位十六进制到6位
                hex = String.format("%c%c%c%c%c%c", 
                    hex.charAt(0), hex.charAt(0),
                    hex.charAt(1), hex.charAt(1),
                    hex.charAt(2), hex.charAt(2));
            }
            
            if (hex.length() == 6) {
                try {
                    int r = Integer.parseInt(hex.substring(0, 2), 16);
                    int g = Integer.parseInt(hex.substring(2, 4), 16);
                    int b = Integer.parseInt(hex.substring(4, 6), 16);
                    return String.format("rgba(%d, %d, %d, %.2f)", r, g, b, alpha);
                } catch (NumberFormatException e) {
                    // 如果解析失败，返回原始颜色
                    return color;
                }
            }
        }
        
        // 其他格式直接返回原始颜色
        return color;
    }
    
    /**
     * 生成系列颜色数组
     * @param baseStyle 基础样式
     * @param count 需要的颜色数量
     * @param paletteName 调色板名称
     * @return 颜色数组
     */
    public static String[] generateSeriesColors(PlotStyle baseStyle, int count, String paletteName) {
        String[] colors = new String[count];
        
        if (baseStyle != null && baseStyle.getColor() != null && count == 1) {
            colors[0] = baseStyle.getColor();
            return colors;
        }
        
        String[] palette = ColorPalette.getPalette(paletteName != null ? paletteName : "default");
        
        for (int i = 0; i < count; i++) {
            colors[i] = palette[i % palette.length];
        }
        
        return colors;
    }
    
    /**
     * 创建高级渐变色配置（支持多种渐变类型）
     * @param startColor 起始颜色
     * @param endColor 结束颜色
     * @param direction 渐变方向 (0-360度)
     * @param type 渐变类型："linear", "radial", "conic"
     * @return 高级渐变色配置对象
     */
    public static Map<String, Object> createAdvancedGradientConfig(String startColor, String endColor, 
                                                                   int direction, String type) {
        Map<String, Object> gradient = new HashMap<>();
        
        switch (type.toLowerCase()) {
            case "radial":
                gradient.put("type", "radial");
                gradient.put("x", 0.5);
                gradient.put("y", 0.5);
                gradient.put("r", 0.5);
                break;
                
            case "conic":
                gradient.put("type", "conic");
                gradient.put("x", 0.5);
                gradient.put("y", 0.5);
                gradient.put("rotation", direction);
                break;
                
            default: // linear
                gradient.put("type", "linear");
                // 根据方向计算端点
                double radians = Math.toRadians(direction);
                gradient.put("x", 0);
                gradient.put("y", 0);
                gradient.put("x2", Math.cos(radians));
                gradient.put("y2", Math.sin(radians));
                break;
        }
        
        // 设置颜色停止点
        List<Map<String, Object>> colorStops = new ArrayList<>();
        
        Map<String, Object> startStop = new HashMap<>();
        startStop.put("offset", 0);
        startStop.put("color", startColor);
        colorStops.add(startStop);
        
        Map<String, Object> endStop = new HashMap<>();
        endStop.put("offset", 1);
        endStop.put("color", endColor);
        colorStops.add(endStop);
        
        gradient.put("colorStops", colorStops);
        
        return gradient;
    }
    
    /**
     * 创建多色渐变配置
     * @param colors 颜色数组
     * @param positions 位置数组（可选，如果为null则均匀分布）
     * @param type 渐变类型
     * @return 多色渐变配置
     */
    public static Map<String, Object> createMultiColorGradient(String[] colors, double[] positions, String type) {
        Map<String, Object> gradient = new HashMap<>();
        gradient.put("type", type.toLowerCase());
        
        // 如果没有提供位置，均匀分布
        if (positions == null) {
            positions = new double[colors.length];
            for (int i = 0; i < colors.length; i++) {
                positions[i] = (double) i / (colors.length - 1);
            }
        }
        
        List<Map<String, Object>> colorStops = new ArrayList<>();
        for (int i = 0; i < colors.length; i++) {
            Map<String, Object> stop = new HashMap<>();
            stop.put("offset", positions[i]);
            stop.put("color", colors[i]);
            colorStops.add(stop);
        }
        
        gradient.put("colorStops", colorStops);
        
        return gradient;
    }
    
    /**
     * 验证颜色值是否有效
     * @param color 颜色值
     * @return 是否有效
     */
    public static boolean isValidColor(String color) {
        if (color == null || color.trim().isEmpty()) {
            return false;
        }
        
        String trimmed = color.trim();
        
        // 检查十六进制颜色
        if (trimmed.matches("^#[0-9A-Fa-f]{6}$") || trimmed.matches("^#[0-9A-Fa-f]{3}$")) {
            return true;
        }
        
        // 检查RGB颜色
        if (trimmed.matches("^rgb\\(\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\s*\\)$")) {
            return true;
        }
        
        // 检查RGBA颜色
        if (trimmed.matches("^rgba\\(\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*[0-9.]+\\s*\\)$")) {
            return true;
        }
        
        // 检查颜色名称
        return ColorPalette.COLOR_NAME_MAP.containsKey(trimmed.toLowerCase());
    }
    
    // ========== 辅助方法 ==========
    
    /**
     * 转换线条端点样式
     * @param lineCap 线条端点样式
     * @return ECharts线条端点样式
     */
    private static String convertLineCap(String lineCap) {
        if (lineCap == null) return "butt";
        
        switch (lineCap.toLowerCase()) {
            case "butt":
                return "butt";
            case "round":
                return "round";
            case "square":
                return "square";
            default:
                return "butt";
        }
    }
    
    /**
     * 转换线条连接样式
     * @param lineJoin 线条连接样式
     * @return ECharts线条连接样式
     */
    private static String convertLineJoin(String lineJoin) {
        if (lineJoin == null) return "miter";
        
        switch (lineJoin.toLowerCase()) {
            case "miter":
                return "miter";
            case "round":
                return "round";
            case "bevel":
                return "bevel";
            default:
                return "miter";
        }
    }
    
    /**
     * 转换边框类型
     * @param lineStyle 线条样式
     * @return ECharts边框类型
     */
    private static String convertBorderType(String lineStyle) {
        if (lineStyle == null) return "solid";
        
        switch (lineStyle.toLowerCase()) {
            case "solid":
                return "solid";
            case "dashed":
                return "dashed";
            case "dotted":
                return "dotted";
            case "dashdot":
                return "dashed"; // ECharts不直接支持dashdot，使用dashed
            default:
                return "solid";
        }
    }
    
    /**
     * 应用阴影效果到LineStyle
     * @param lineStyle 线条样式对象
     * @param plotStyle 绘图样式
     */
    private static void applyShadowToLineStyle(LineStyle lineStyle, PlotStyle plotStyle) {
        // 检查是否有阴影配置
        Object shadowBlur = plotStyle.getProperty("shadowBlur");
        Object shadowColor = plotStyle.getProperty("shadowColor");
        Object shadowOffsetX = plotStyle.getProperty("shadowOffsetX");
        Object shadowOffsetY = plotStyle.getProperty("shadowOffsetY");
        
        if (shadowBlur != null) {
            lineStyle.setShadowBlur((Number) shadowBlur);
        }
        
        if (shadowColor != null) {
            lineStyle.setShadowColor((String) shadowColor);
        }
        
        if (shadowOffsetX != null) {
            lineStyle.setShadowOffsetX((Number) shadowOffsetX);
        }
        
        if (shadowOffsetY != null) {
            lineStyle.setShadowOffsetY((Number) shadowOffsetY);
        }
    }
    
    /**
     * 应用阴影效果到ItemStyle
     * @param itemStyle 项样式对象
     * @param plotStyle 绘图样式
     */
    private static void applyShadowToItemStyle(ItemStyle itemStyle, PlotStyle plotStyle) {
        // 检查是否有阴影配置
        Object shadowBlur = plotStyle.getProperty("shadowBlur");
        Object shadowColor = plotStyle.getProperty("shadowColor");
        Object shadowOffsetX = plotStyle.getProperty("shadowOffsetX");
        Object shadowOffsetY = plotStyle.getProperty("shadowOffsetY");
        
        if (shadowBlur != null) {
            itemStyle.setShadowBlur((Number) shadowBlur);
        }
        
        if (shadowColor != null) {
            itemStyle.setShadowColor((String) shadowColor);
        }
        
        if (shadowOffsetX != null) {
            itemStyle.setShadowOffsetX((Number) shadowOffsetX);
        }
        
        if (shadowOffsetY != null) {
            itemStyle.setShadowOffsetY((Number) shadowOffsetY);
        }
    }
    
    /**
     * 应用阴影效果到BarItemStyle
     * @param itemStyle 柱状图项样式对象
     * @param plotStyle 绘图样式
     */
    private static void applyShadowToBarItemStyle(BarItemStyle itemStyle, PlotStyle plotStyle) {
        // 检查是否有阴影配置
        Object shadowBlur = plotStyle.getProperty("shadowBlur");
        Object shadowColor = plotStyle.getProperty("shadowColor");
        Object shadowOffsetX = plotStyle.getProperty("shadowOffsetX");
        Object shadowOffsetY = plotStyle.getProperty("shadowOffsetY");
        
        if (shadowBlur != null) {
            itemStyle.setShadowBlur((Number) shadowBlur);
        }
        
        if (shadowColor != null) {
            itemStyle.setShadowColor((String) shadowColor);
        }
        
        if (shadowOffsetX != null) {
            itemStyle.setShadowOffsetX((Number) shadowOffsetX);
        }
        
        if (shadowOffsetY != null) {
            itemStyle.setShadowOffsetY((Number) shadowOffsetY);
        }
    }
    
    /**
     * 应用阴影效果到PieItemStyle
     * @param itemStyle 饼图项样式对象
     * @param plotStyle 绘图样式
     */
    private static void applyShadowToPieItemStyle(PieItemStyle itemStyle, PlotStyle plotStyle) {
        // 检查是否有阴影配置
        Object shadowBlur = plotStyle.getProperty("shadowBlur");
        Object shadowColor = plotStyle.getProperty("shadowColor");
        Object shadowOffsetX = plotStyle.getProperty("shadowOffsetX");
        Object shadowOffsetY = plotStyle.getProperty("shadowOffsetY");
        
        if (shadowBlur != null) {
            itemStyle.setShadowBlur((Number) shadowBlur);
        }
        
        if (shadowColor != null) {
            itemStyle.setShadowColor((String) shadowColor);
        }
        
        if (shadowOffsetX != null) {
            itemStyle.setShadowOffsetX((Number) shadowOffsetX);
        }
        
        if (shadowOffsetY != null) {
            itemStyle.setShadowOffsetY((Number) shadowOffsetY);
        }
    }
    
    /**
     * 应用阴影效果到CandlestickItemStyle
     * @param itemStyle K线图项样式对象
     * @param plotStyle 绘图样式
     */
    private static void applyShadowToCandlestickItemStyle(CandlestickItemStyle itemStyle, PlotStyle plotStyle) {
        // 检查是否有阴影配置
        Object shadowBlur = plotStyle.getProperty("shadowBlur");
        Object shadowColor = plotStyle.getProperty("shadowColor");
        Object shadowOffsetX = plotStyle.getProperty("shadowOffsetX");
        Object shadowOffsetY = plotStyle.getProperty("shadowOffsetY");
        
        if (shadowBlur != null) {
            itemStyle.setShadowBlur((Number) shadowBlur);
        }
        
        if (shadowColor != null) {
            itemStyle.setShadowColor((String) shadowColor);
        }
        
        if (shadowOffsetX != null) {
            itemStyle.setShadowOffsetX((Number) shadowOffsetX);
        }
        
        if (shadowOffsetY != null) {
            itemStyle.setShadowOffsetY((Number) shadowOffsetY);
        }
    }
    
    /**
     * 应用阴影效果到TreemapSeriesItemStyle
     * @param itemStyle 矩形树图项样式对象
     * @param plotStyle 绘图样式
     */
    private static void applyShadowToTreemapItemStyle(TreemapSeriesItemStyle itemStyle, PlotStyle plotStyle) {
        // 检查是否有阴影配置
        Object shadowBlur = plotStyle.getProperty("shadowBlur");
        Object shadowColor = plotStyle.getProperty("shadowColor");
        Object shadowOffsetX = plotStyle.getProperty("shadowOffsetX");
        Object shadowOffsetY = plotStyle.getProperty("shadowOffsetY");
        
        if (shadowBlur != null) {
            itemStyle.setShadowBlur((Number) shadowBlur);
        }
        
        if (shadowColor != null) {
            itemStyle.setShadowColor((String) shadowColor);
        }
        
        if (shadowOffsetX != null) {
            itemStyle.setShadowOffsetX((Number) shadowOffsetX);
        }
        
        if (shadowOffsetY != null) {
            itemStyle.setShadowOffsetY((Number) shadowOffsetY);
        }
    }
    
    /**
     * 应用阴影效果到SunburstItemStyle
     * @param itemStyle 旭日图项样式对象
     * @param plotStyle 绘图样式
     */
    private static void applyShadowToSunburstItemStyle(SunburstItemStyle itemStyle, PlotStyle plotStyle) {
        // 检查是否有阴影配置
        Object shadowBlur = plotStyle.getProperty("shadowBlur");
        Object shadowColor = plotStyle.getProperty("shadowColor");
        Object shadowOffsetX = plotStyle.getProperty("shadowOffsetX");
        Object shadowOffsetY = plotStyle.getProperty("shadowOffsetY");
        
        if (shadowBlur != null) {
            itemStyle.setShadowBlur((Number) shadowBlur);
        }
        
        if (shadowColor != null) {
            itemStyle.setShadowColor((String) shadowColor);
        }
        
        if (shadowOffsetX != null) {
            itemStyle.setShadowOffsetX((Number) shadowOffsetX);
        }
        
        if (shadowOffsetY != null) {
            itemStyle.setShadowOffsetY((Number) shadowOffsetY);
        }
    }
    
    /**
     * 创建渐变色配置
     * @param startColor 起始颜色
     * @param endColor 结束颜色
     * @param direction 渐变方向 (0-360度)
     * @return 渐变色配置对象
     */
    public static Map<String, Object> createGradientConfig(String startColor, String endColor, int direction) {
        Map<String, Object> gradient = new HashMap<>();
        gradient.put("type", "linear");
        gradient.put("x", 0);
        gradient.put("y", 0);
        gradient.put("x2", direction == 0 || direction == 180 ? 0 : 1);
        gradient.put("y2", direction == 90 || direction == 270 ? 0 : 1);
        
        Map<String, Object> colorStops = new HashMap<>();
        colorStops.put("0", startColor);
        colorStops.put("1", endColor);
        gradient.put("colorStops", colorStops);
        
        return gradient;
    }
    
    /**
     * 创建径向渐变色配置
     * @param startColor 起始颜色
     * @param endColor 结束颜色
     * @param centerX 中心X坐标 (0-1)
     * @param centerY 中心Y坐标 (0-1)
     * @param radius 半径 (0-1)
     * @return 径向渐变色配置对象
     */
    public static Map<String, Object> createRadialGradientConfig(String startColor, String endColor, 
                                                                double centerX, double centerY, double radius) {
        Map<String, Object> gradient = new HashMap<>();
        gradient.put("type", "radial");
        gradient.put("x", centerX);
        gradient.put("y", centerY);
        gradient.put("r", radius);
        
        Map<String, Object> colorStops = new HashMap<>();
        colorStops.put("0", startColor);
        colorStops.put("1", endColor);
        gradient.put("colorStops", colorStops);
        
        return gradient;
    }
    
    /**
     * 创建高亮状态样式
     * @param plotStyle 基础样式
     * @return 高亮状态样式
     */
    public static ItemStyle createEmphasisStyle(PlotStyle plotStyle) {
        ItemStyle emphasisStyle = toEChartsItemStyle(plotStyle);
        
        // 高亮时增加亮度
        if (plotStyle.getColor() != null) {
            String brightColor = brightenColor(plotStyle.getColor(), 0.2);
            emphasisStyle.setColor(brightColor);
        }
        
        // 增加边框宽度
        if (plotStyle.getLineWidth() > 0) {
            emphasisStyle.setBorderWidth(plotStyle.getLineWidth() + 1);
        }
        
        // 增加阴影效果
        emphasisStyle.setShadowBlur(10);
        emphasisStyle.setShadowColor("rgba(0, 0, 0, 0.3)");
        
        return emphasisStyle;
    }
    
    /**
     * 创建选中状态样式
     * @param plotStyle 基础样式
     * @return 选中状态样式
     */
    public static ItemStyle createSelectStyle(PlotStyle plotStyle) {
        ItemStyle selectStyle = toEChartsItemStyle(plotStyle);
        
        // 选中时使用对比色
        if (plotStyle.getColor() != null) {
            String contrastColor = getContrastColor(plotStyle.getColor());
            selectStyle.setColor(contrastColor);
        }
        
        // 增加边框宽度和特殊颜色
        selectStyle.setBorderWidth(plotStyle.getLineWidth() + 2);
        selectStyle.setBorderColor("#FF6B6B");
        
        return selectStyle;
    }
    
    /**
     * 创建模糊状态样式
     * @param plotStyle 基础样式
     * @return 模糊状态样式
     */
    public static ItemStyle createBlurStyle(PlotStyle plotStyle) {
        ItemStyle blurStyle = toEChartsItemStyle(plotStyle);
        
        // 模糊时降低透明度
        blurStyle.setOpacity(plotStyle.getAlpha() * 0.3);
        
        // 降低颜色饱和度
        if (plotStyle.getColor() != null) {
            String desaturatedColor = desaturateColor(plotStyle.getColor(), 0.5);
            blurStyle.setColor(desaturatedColor);
        }
        
        return blurStyle;
    }
    
    /**
     * 增加颜色亮度实现（使用HSL色彩空间）
     * @param color 原始颜色
     * @param factor 亮度因子 (0-1)
     * @return 增亮后的颜色
     */
    private static String brightenColor(String color, double factor) {
        if (color == null || !color.startsWith("#")) {
            return color;
        }
        
        try {
            int[] rgb = ColorPalette.parseColorToRGB(color);
            if (rgb == null) return color;
            
            // 转换到HSL色彩空间
            double[] hsl = rgbToHsl(rgb[0], rgb[1], rgb[2]);
            
            // 增加亮度
            hsl[2] = Math.min(1.0, hsl[2] + (1.0 - hsl[2]) * factor);
            
            // 转换回RGB
            int[] newRgb = hslToRgb(hsl[0], hsl[1], hsl[2]);
            
            return String.format("#%02x%02x%02x", newRgb[0], newRgb[1], newRgb[2]);
        } catch (Exception e) {
            return color;
        }
    }
    
    /**
     * 降低颜色饱和度实现（使用HSL色彩空间）
     * @param color 原始颜色
     * @param factor 饱和度因子 (0-1)
     * @return 降饱和后的颜色
     */
    private static String desaturateColor(String color, double factor) {
        if (color == null || !color.startsWith("#")) {
            return color;
        }
        
        try {
            int[] rgb = ColorPalette.parseColorToRGB(color);
            if (rgb == null) return color;
            
            // 转换到HSL色彩空间
            double[] hsl = rgbToHsl(rgb[0], rgb[1], rgb[2]);
            
            // 降低饱和度
            hsl[1] = Math.max(0.0, hsl[1] * (1.0 - factor));
            
            // 转换回RGB
            int[] newRgb = hslToRgb(hsl[0], hsl[1], hsl[2]);
            
            return String.format("#%02x%02x%02x", newRgb[0], newRgb[1], newRgb[2]);
        } catch (Exception e) {
            return color;
        }
    }
}