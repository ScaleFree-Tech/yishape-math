package com.reremouse.lab.math.viz;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * 统一的图表样式参数类，提供类似matplotlib的样式控制
 * 支持线条样式、颜色、标记等完整的样式定制
 * 
 * @author lteb2
 */
public class PlotStyle implements Serializable {
    
    // ========== 颜色相关 ==========
    private String color;           // 主要颜色
    private String faceColor;       // 填充颜色
    private String edgeColor;       // 边缘颜色
    private double alpha;            // 透明度 (0.0-1.0)
    
    // ========== 线条样式 ==========
    private String lineStyle;      // 线条样式: "solid", "dashed", "dotted", "dashdot"
    private double lineWidth;       // 线条宽度
    private String lineCap;        // 线条端点样式: "butt", "round", "square"
    private String lineJoin;       // 线条连接样式: "miter", "round", "bevel"
    
    // ========== 标记样式 ==========
    private String marker;         // 标记样式: "o", "s", "^", "v", "<", ">", "d", "*", "+", "x"
    private double markerSize;      // 标记大小
    private String markerColor;    // 标记颜色
    private String markerEdgeColor; // 标记边缘颜色
    private double markerEdgeWidth; // 标记边缘宽度
    private double markerAlpha;     // 标记透明度
    
    // ========== 填充样式 ==========
    private String fillStyle;     // 填充样式: "solid", "none", "pattern"
    private String hatchPattern;   // 填充图案: "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"
    
    // ========== 字体样式 ==========
    private String fontFamily;    // 字体族
    private double fontSize;       // 字体大小
    private String fontWeight;    // 字体粗细: "normal", "bold", "light"
    private String fontStyle;     // 字体样式: "normal", "italic", "oblique"
    
    // ========== 图例和标签 ==========
    private String label;         // 数据标签
    private boolean showLegend;   // 是否显示图例
    private String legendLocation; // 图例位置
    
    // ========== 动画效果 ==========
    private boolean enableAnimation; // 是否启用动画
    private int animationDuration;   // 动画持续时间(ms)
    private String animationEasing;  // 动画缓动函数
    
    // ========== 扩展属性 ==========
    private Map<String, Object> extraProperties; // 扩展属性
    
    /**
     * 默认构造函数，使用默认样式
     */
    public PlotStyle() {
        // 设置默认值
        this.color = "#5470c6";
        this.faceColor = "#5470c6";
        this.edgeColor = "#333333";
        this.alpha = 1.0f;
        
        this.lineStyle = "solid";
        this.lineWidth = 2.0f;
        this.lineCap = "butt";
        this.lineJoin = "miter";
        
        this.marker = "o";
        this.markerSize = 6.0f;
        this.markerColor = "#5470c6";
        this.markerEdgeColor = "#333333";
        this.markerEdgeWidth = 1.0f;
        this.markerAlpha = 1.0f;
        
        this.fillStyle = "solid";
        this.hatchPattern = "";
        
        this.fontFamily = "Arial";
        this.fontSize = 12.0f;
        this.fontWeight = "normal";
        this.fontStyle = "normal";
        
        this.label = "";
        this.showLegend = true;
        this.legendLocation = "top-right";
        
        this.enableAnimation = true;
        this.animationDuration = 1000;
        this.animationEasing = "cubicOut";
        
        this.extraProperties = new HashMap<>();
    }
    
    /**
     * 复制构造函数
     * @param other 要复制的样式
     */
    public PlotStyle(PlotStyle other) {
        this.color = other.color;
        this.faceColor = other.faceColor;
        this.edgeColor = other.edgeColor;
        this.alpha = other.alpha;
        
        this.lineStyle = other.lineStyle;
        this.lineWidth = other.lineWidth;
        this.lineCap = other.lineCap;
        this.lineJoin = other.lineJoin;
        
        this.marker = other.marker;
        this.markerSize = other.markerSize;
        this.markerColor = other.markerColor;
        this.markerEdgeColor = other.markerEdgeColor;
        this.markerEdgeWidth = other.markerEdgeWidth;
        this.markerAlpha = other.markerAlpha;
        
        this.fillStyle = other.fillStyle;
        this.hatchPattern = other.hatchPattern;
        
        this.fontFamily = other.fontFamily;
        this.fontSize = other.fontSize;
        this.fontWeight = other.fontWeight;
        this.fontStyle = other.fontStyle;
        
        this.label = other.label;
        this.showLegend = other.showLegend;
        this.legendLocation = other.legendLocation;
        
        this.enableAnimation = other.enableAnimation;
        this.animationDuration = other.animationDuration;
        this.animationEasing = other.animationEasing;
        
        this.extraProperties = new HashMap<>(other.extraProperties);
    }
    
    // ========== 流式API方法 ==========
    
    /**
     * 设置主要颜色
     * @param color 颜色值（支持十六进制、RGB、颜色名称）
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle color(String color) {
        this.color = color;
        return this;
    }
    
    /**
     * 设置填充颜色
     * @param faceColor 填充颜色
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle faceColor(String faceColor) {
        this.faceColor = faceColor;
        return this;
    }
    
    /**
     * 设置透明度
     * @param alpha 透明度值 (0.0-1.0)
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle alpha(double alpha) {
        this.alpha = Math.max(0.0f, Math.min(1.0f, alpha));
        return this;
    }
    
    /**
     * 设置线条样式
     * @param lineStyle 线条样式 ("solid", "dashed", "dotted", "dashdot")
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle lineStyle(String lineStyle) {
        this.lineStyle = lineStyle;
        return this;
    }
    
    /**
     * 设置线条宽度
     * @param lineWidth 线条宽度
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle lineWidth(double lineWidth) {
        this.lineWidth = Math.max(0.0f, lineWidth);
        return this;
    }
    
    /**
     * 设置标记样式
     * @param marker 标记样式
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle marker(String marker) {
        this.marker = marker;
        return this;
    }
    
    /**
     * 设置标记大小
     * @param markerSize 标记大小
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle markerSize(double markerSize) {
        this.markerSize = Math.max(0.0f, markerSize);
        return this;
    }
    
    /**
     * 设置数据标签
     * @param label 标签文本
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle label(String label) {
        this.label = label;
        return this;
    }
    
    /**
     * 设置字体大小
     * @param fontSize 字体大小
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle fontSize(double fontSize) {
        this.fontSize = Math.max(1.0f, fontSize);
        return this;
    }
    
    /**
     * 设置动画效果
     * @param enableAnimation 是否启用动画
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle animation(boolean enableAnimation) {
        this.enableAnimation = enableAnimation;
        return this;
    }
    
    /**
     * 设置扩展属性
     * @param key 属性名
     * @param value 属性值
     * @return 当前样式对象，支持链式调用
     */
    public PlotStyle setProperty(String key, Object value) {
        this.extraProperties.put(key, value);
        return this;
    }
    
    // ========== 静态工厂方法 ==========
    
    /**
     * 创建默认样式
     * @return 默认样式对象
     */
    public static PlotStyle defaultStyle() {
        return new PlotStyle();
    }
    
    /**
     * 创建指定颜色的样式
     * @param color 主要颜色
     * @return 样式对象
     */
    public static PlotStyle withColor(String color) {
        return new PlotStyle().color(color);
    }
    
    /**
     * 创建简单线条样式
     * @param color 颜色
     * @param lineStyle 线条样式
     * @param lineWidth 线条宽度
     * @return 样式对象
     */
    public static PlotStyle line(String color, String lineStyle, double lineWidth) {
        return new PlotStyle()
                .color(color)
                .lineStyle(lineStyle)
                .lineWidth(lineWidth);
    }
    
    /**
     * 创建标记样式
     * @param color 颜色
     * @param marker 标记形状
     * @param markerSize 标记大小
     * @return 样式对象
     */
    public static PlotStyle scatter(String color, String marker, double markerSize) {
        return new PlotStyle()
                .color(color)
                .marker(marker)
                .markerSize(markerSize);
    }
    
    // ========== Getter/Setter 方法 ==========
    
    public String getColor() { return color; }
    public void setColor(String color) { this.color = color; }
    
    public String getFaceColor() { return faceColor; }
    public void setFaceColor(String faceColor) { this.faceColor = faceColor; }
    
    public String getEdgeColor() { return edgeColor; }
    public void setEdgeColor(String edgeColor) { this.edgeColor = edgeColor; }
    
    public double getAlpha() { return alpha; }
    public void setAlpha(double alpha) { this.alpha = alpha; }
    
    public String getLineStyle() { return lineStyle; }
    public void setLineStyle(String lineStyle) { this.lineStyle = lineStyle; }
    
    public double getLineWidth() { return lineWidth; }
    public void setLineWidth(double lineWidth) { this.lineWidth = lineWidth; }
    
    public String getLineCap() { return lineCap; }
    public void setLineCap(String lineCap) { this.lineCap = lineCap; }
    
    public String getLineJoin() { return lineJoin; }
    public void setLineJoin(String lineJoin) { this.lineJoin = lineJoin; }
    
    public String getMarker() { return marker; }
    public void setMarker(String marker) { this.marker = marker; }
    
    public double getMarkerSize() { return markerSize; }
    public void setMarkerSize(double markerSize) { this.markerSize = markerSize; }
    
    public String getMarkerColor() { return markerColor; }
    public void setMarkerColor(String markerColor) { this.markerColor = markerColor; }
    
    public String getMarkerEdgeColor() { return markerEdgeColor; }
    public void setMarkerEdgeColor(String markerEdgeColor) { this.markerEdgeColor = markerEdgeColor; }
    
    public double getMarkerEdgeWidth() { return markerEdgeWidth; }
    public void setMarkerEdgeWidth(double markerEdgeWidth) { this.markerEdgeWidth = markerEdgeWidth; }
    
    public double getMarkerAlpha() { return markerAlpha; }
    public void setMarkerAlpha(double markerAlpha) { this.markerAlpha = markerAlpha; }
    
    public String getFillStyle() { return fillStyle; }
    public void setFillStyle(String fillStyle) { this.fillStyle = fillStyle; }
    
    public String getHatchPattern() { return hatchPattern; }
    public void setHatchPattern(String hatchPattern) { this.hatchPattern = hatchPattern; }
    
    public String getFontFamily() { return fontFamily; }
    public void setFontFamily(String fontFamily) { this.fontFamily = fontFamily; }
    
    public double getFontSize() { return fontSize; }
    public void setFontSize(double fontSize) { this.fontSize = fontSize; }
    
    public String getFontWeight() { return fontWeight; }
    public void setFontWeight(String fontWeight) { this.fontWeight = fontWeight; }
    
    public String getFontStyle() { return fontStyle; }
    public void setFontStyle(String fontStyle) { this.fontStyle = fontStyle; }
    
    public String getLabel() { return label; }
    public void setLabel(String label) { this.label = label; }
    
    public boolean isShowLegend() { return showLegend; }
    public void setShowLegend(boolean showLegend) { this.showLegend = showLegend; }
    
    public String getLegendLocation() { return legendLocation; }
    public void setLegendLocation(String legendLocation) { this.legendLocation = legendLocation; }
    
    public boolean isEnableAnimation() { return enableAnimation; }
    public void setEnableAnimation(boolean enableAnimation) { this.enableAnimation = enableAnimation; }
    
    public int getAnimationDuration() { return animationDuration; }
    public void setAnimationDuration(int animationDuration) { this.animationDuration = animationDuration; }
    
    public String getAnimationEasing() { return animationEasing; }
    public void setAnimationEasing(String animationEasing) { this.animationEasing = animationEasing; }
    
    public Map<String, Object> getExtraProperties() { return extraProperties; }
    public void setExtraProperties(Map<String, Object> extraProperties) { this.extraProperties = extraProperties; }
    
    public Object getProperty(String key) { return extraProperties.get(key); }
    
    @Override
    public String toString() {
        return String.format("PlotStyle{color='%s', lineStyle='%s', lineWidth=%.1f, marker='%s', markerSize=%.1f}", 
                color, lineStyle, lineWidth, marker, markerSize);
    }
}