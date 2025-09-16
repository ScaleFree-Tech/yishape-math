package com.reremouse.lab.math.viz;

import java.io.Serializable;
import java.util.Map;
import java.util.HashMap;
import org.icepear.echarts.charts.bar.BarSeries;
import org.icepear.echarts.charts.bar.BarEmphasis;
import org.icepear.echarts.charts.bar.BarItemStyle;
import org.icepear.echarts.charts.line.LineSeries;
import org.icepear.echarts.charts.line.LineEmphasis;
import org.icepear.echarts.charts.scatter.ScatterSeries;
import org.icepear.echarts.charts.pie.PieSeries;
import org.icepear.echarts.charts.pie.PieEmphasis;
import org.icepear.echarts.charts.pie.PieItemStyle;
import org.icepear.echarts.charts.heatmap.HeatmapSeries;
import org.icepear.echarts.charts.radar.RadarSeries;
import org.icepear.echarts.charts.gauge.GaugeSeries;
import org.icepear.echarts.charts.boxplot.BoxplotSeries;
import org.icepear.echarts.charts.candlestick.CandlestickSeries;
import org.icepear.echarts.charts.funnel.FunnelSeries;
import org.icepear.echarts.charts.sankey.SankeySeries;
import org.icepear.echarts.charts.sunburst.SunburstSeries;
import org.icepear.echarts.charts.themeRiver.ThemeRiverSeries;
import org.icepear.echarts.charts.tree.TreeSeries;
import org.icepear.echarts.charts.treemap.TreemapSeries;
import org.icepear.echarts.charts.graph.GraphSeries;
import org.icepear.echarts.charts.parallel.ParallelSeries;

/**
 * 增强的通用样式应用器，负责将PlotStyle应用到不同类型的ECharts图表系列
 * 提供统一的样式应用接口，支持所有图表类型和状态样式（emphasis、blur、select）
 * 
 * @author lteb2
 */
public class UniversalStyleApplier implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * 应用样式到线图系列（增强版，支持状态样式）
     * @param series 线图系列
     * @param style 样式对象
     */
    public static void applyToLineSeries(LineSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置线条样式
        if (style.getColor() != null) {
            series.setLineStyle(StyleConverter.toEChartsLineStyle(style));
        }
        
        // 设置标记符号
        if (style.getMarker() != null && !style.getMarker().isEmpty()) {
            String symbol = StyleConverter.convertMarkerToSymbol(style.getMarker());
            series.setSymbol(symbol);
            series.setSymbolSize((int) style.getMarkerSize());
            series.setShowSymbol(true);
        }
        
        // 设置透明度和填充
        if (style.getAlpha() < 1.0f) {
            series.setAreaStyle(StyleConverter.toEChartsAreaStyle(style));
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        } else if (style.getAnimationDuration() > 0) {
            series.setAnimationDuration(style.getAnimationDuration());
        }
        
        // 应用状态样式
        applyStateStylesToLineSeries(series, style);
    }
    
    /**
     * 应用状态样式到线图系列
     * @param series 线图系列
     * @param style 样式对象
     */
    private static void applyStateStylesToLineSeries(LineSeries series, PlotStyle style) {
        // 创建高亮状态样式
        LineEmphasis emphasis = new LineEmphasis();
        
        // 设置高亮时的线条样式
        if (style.getColor() != null) {
            emphasis.setLineStyle(StyleConverter.toEChartsLineStyle(style));
        }
        
        // 设置高亮时的填充样式
        if (style.getAlpha() < 1.0f) {
            emphasis.setAreaStyle(StyleConverter.toEChartsAreaStyle(style));
        }
        
        // 设置高亮时的itemStyle
        emphasis.setItemStyle(StyleConverter.toEChartsItemStyle(style));
        
        series.setEmphasis(emphasis);
    }
    
    /**
     * 应用样式到散点图系列
     * @param series 散点图系列
     * @param style 样式对象
     */
    public static void applyToScatterSeries(ScatterSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置标记符号和颜色
        if (style.getMarker() != null && !style.getMarker().isEmpty()) {
            String symbol = StyleConverter.convertMarkerToSymbol(style.getMarker());
            series.setSymbol(symbol);
        }
        
        series.setSymbolSize((int) style.getMarkerSize());
        
        // 设置颜色
        if (style.getColor() != null) {
            // 散点图的颜色主要通过系列颜色设置
            series.setColor(style.getColor());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
    }
    
    /**
     * 应用样式到柱状图系列（增强版，支持状态样式）
     * @param series 柱状图系列
     * @param style 样式对象
     */
    public static void applyToBarSeries(BarSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置透明度
        if (style.getAlpha() < 1.0f) {
            String colorWithAlpha = StyleConverter.applyAlpha(style.getColor(), style.getAlpha());
            series.setColor(colorWithAlpha);
        }
        
        // 设置itemStyle（包含边框等详细样式）
        BarItemStyle itemStyle = StyleConverter.toEChartsBarItemStyle(style);
        series.setItemStyle(itemStyle);
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        } else if (style.getAnimationDuration() > 0) {
            series.setAnimationDuration(style.getAnimationDuration());
        }
        
        // 应用状态样式
        applyStateStylesToBarSeries(series, style);
    }
    
    /**
     * 应用状态样式到柱状图系列
     * @param series 柱状图系列
     * @param style 样式对象
     */
    private static void applyStateStylesToBarSeries(BarSeries series, PlotStyle style) {
        // 创建高亮状态样式
        BarEmphasis emphasis = new BarEmphasis();
        
        // 设置高亮时的itemStyle
        BarItemStyle emphasisItemStyle = StyleConverter.toEChartsBarItemStyle(style);
        
        // 高亮时增加亮度
        if (style.getColor() != null) {
            String brightColor = brightenColor(style.getColor(), 0.2);
            emphasisItemStyle.setColor(brightColor);
        }
        
        // 增加边框宽度
        if (style.getLineWidth() > 0) {
            emphasisItemStyle.setBorderWidth(style.getLineWidth() + 1);
        }
        
        // 增加阴影效果
        emphasisItemStyle.setShadowBlur(10);
        emphasisItemStyle.setShadowColor("rgba(0, 0, 0, 0.3)");
        
        emphasis.setItemStyle(emphasisItemStyle);
        series.setEmphasis(emphasis);
    }
    
    /**
     * 应用样式到饼图系列（增强版，支持状态样式）
     * @param series 饼图系列
     * @param style 样式对象
     */
    public static void applyToPieSeries(PieSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 饼图主要通过数据项设置颜色，这里设置默认颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置透明度
        if (style.getAlpha() < 1.0f) {
            String colorWithAlpha = StyleConverter.applyAlpha(style.getColor(), style.getAlpha());
            series.setColor(colorWithAlpha);
        }
        
        // 设置itemStyle
        PieItemStyle itemStyle = StyleConverter.toEChartsPieItemStyle(style);
        series.setItemStyle(itemStyle);
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
        
        // 应用状态样式
        applyStateStylesToPieSeries(series, style);
    }
    
    /**
     * 应用状态样式到饼图系列
     * @param series 饼图系列
     * @param style 样式对象
     */
    private static void applyStateStylesToPieSeries(PieSeries series, PlotStyle style) {
        // 创建高亮状态样式
        PieEmphasis emphasis = new PieEmphasis();
        
        // 设置高亮时的itemStyle
        PieItemStyle emphasisItemStyle = StyleConverter.toEChartsPieItemStyle(style);
        
        // 高亮时增加亮度
        if (style.getColor() != null) {
            String brightColor = brightenColor(style.getColor(), 0.2);
            emphasisItemStyle.setColor(brightColor);
        }
        
        // 增加边框宽度
        if (style.getLineWidth() > 0) {
            emphasisItemStyle.setBorderWidth(style.getLineWidth() + 1);
        }
        
        // 增加阴影效果
        emphasisItemStyle.setShadowBlur(10);
        emphasisItemStyle.setShadowColor("rgba(0, 0, 0, 0.3)");
        
        emphasis.setItemStyle(emphasisItemStyle);
        series.setEmphasis(emphasis);
    }
    
    /**
     * 应用样式到热力图系列（增强版，支持视觉映射和状态样式）
     * @param series 热力图系列
     * @param style 样式对象
     */
    public static void applyToHeatmapSeries(HeatmapSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 应用特定的热力图样式
        Map<String, Object> heatmapStyleConfig = StyleConverter.toEChartsHeatmapStyle(style);
        
        // 设置项样式
        if (heatmapStyleConfig.containsKey("color") && heatmapStyleConfig.get("color") instanceof String) {
            series.setColor((String) heatmapStyleConfig.get("color"));
        }
        
        // 设置透明度
        if (style.getAlpha() < 1.0) {
            // 热力图通过视觉映射控制透明度效果
            series.setItemStyle(StyleConverter.toEChartsItemStyle(style));
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        } else if (style.getAnimationDuration() > 0) {
            series.setAnimationDuration(style.getAnimationDuration());
        }
        
        // 应用状态样式
        applyStateStylesToHeatmapSeries(series, style);
    }
    
    /**
     * 应用状态样式到热力图系列
     * @param series 热力图系列
     * @param style 样式对象
     */
    private static void applyStateStylesToHeatmapSeries(HeatmapSeries series, PlotStyle style) {
        // 创建高亮状态样式
        org.icepear.echarts.charts.heatmap.HeatmapEmphasis emphasis = new org.icepear.echarts.charts.heatmap.HeatmapEmphasis();
        
        // 设置高亮时的itemStyle
        org.icepear.echarts.components.series.ItemStyle emphasisItemStyle = StyleConverter.toEChartsItemStyle(style);
        
        // 高亮时增加亮度和边框
        if (style.getColor() != null) {
            String brightColor = brightenColor(style.getColor(), 0.3);
            emphasisItemStyle.setColor(brightColor);
        }
        
        // 增加边框效果
        emphasisItemStyle.setBorderWidth(Math.max(2, style.getLineWidth() + 1));
        emphasisItemStyle.setBorderColor("#ffffff");
        
        // 增加阴影效果
        emphasisItemStyle.setShadowBlur(8);
        emphasisItemStyle.setShadowColor("rgba(0, 0, 0, 0.4)");
        
        emphasis.setItemStyle(emphasisItemStyle);
        series.setEmphasis(emphasis);
    }
    
    /**
     * 应用样式到雷达图系列（增强版，支持状态样式）
     * @param series 雷达图系列
     * @param style 样式对象
     */
    public static void applyToRadarSeries(RadarSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置线条样式
        if (style.getColor() != null) {
            series.setLineStyle(StyleConverter.toEChartsLineStyle(style));
        }
        
        // 设置填充样式
        if (style.getAlpha() < 1.0) {
            series.setAreaStyle(StyleConverter.toEChartsAreaStyle(style));
        }
        
        // 设置标记
        if (style.getMarker() != null && !style.getMarker().isEmpty()) {
            String symbol = StyleConverter.convertMarkerToSymbol(style.getMarker());
            series.setSymbol(symbol);
            series.setSymbolSize((int) style.getMarkerSize());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        } else if (style.getAnimationDuration() > 0) {
            series.setAnimationDuration(style.getAnimationDuration());
        }
        
        // 应用状态样式
        applyStateStylesToRadarSeries(series, style);
    }
    
    /**
     * 应用状态样式到雷达图系列
     * @param series 雷达图系列
     * @param style 样式对象
     */
    private static void applyStateStylesToRadarSeries(RadarSeries series, PlotStyle style) {
        // 创建高亮状态样式
        org.icepear.echarts.charts.radar.RadarEmphasis emphasis = new org.icepear.echarts.charts.radar.RadarEmphasis();
        
        // 设置高亮时的线条样式
        org.icepear.echarts.components.series.LineStyle emphasisLineStyle = StyleConverter.toEChartsLineStyle(style);
        if (style.getColor() != null) {
            String brightColor = brightenColor(style.getColor(), 0.3);
            emphasisLineStyle.setColor(brightColor);
        }
        emphasisLineStyle.setWidth((int)(style.getLineWidth() + 1));
        emphasis.setLineStyle(emphasisLineStyle);
        
        // 设置高亮时的填充样式
        if (style.getAlpha() < 1.0) {
            org.icepear.echarts.charts.line.LineAreaStyle emphasisAreaStyle = StyleConverter.toEChartsAreaStyle(style);
            emphasisAreaStyle.setOpacity(Math.min(style.getAlpha() + 0.2, 1.0));
            emphasis.setAreaStyle(emphasisAreaStyle);
        }
        
        series.setEmphasis(emphasis);
    }
    
    /**
     * 应用样式到仪表盘系列（增强版，支持状态样式）
     * @param series 仪表盘系列
     * @param style 样式对象
     */
    public static void applyToGaugeSeries(GaugeSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        } else if (style.getAnimationDuration() > 0) {
            series.setAnimationDuration(style.getAnimationDuration());
        }
        
        // 应用状态样式
        applyStateStylesToGaugeSeries(series, style);
    }
    
    /**
     * 应用状态样式到仪表盘系列
     * @param series 仪表盘系列
     * @param style 样式对象
     */
    private static void applyStateStylesToGaugeSeries(GaugeSeries series, PlotStyle style) {
        // 创建高亮状态样式
        org.icepear.echarts.charts.gauge.GaugeEmphasis emphasis = new org.icepear.echarts.charts.gauge.GaugeEmphasis();
        
        // 设置高亮时的itemStyle
        org.icepear.echarts.components.series.ItemStyle emphasisItemStyle = StyleConverter.toEChartsItemStyle(style);
        
        // 高亮时增加亮度和特效
        if (style.getColor() != null) {
            String brightColor = brightenColor(style.getColor(), 0.3);
            emphasisItemStyle.setColor(brightColor);
        }
        
        // 增加阴影效果
        emphasisItemStyle.setShadowBlur(15);
        emphasisItemStyle.setShadowColor("rgba(0, 0, 0, 0.5)");
        
        // 增加光晕效果
        emphasisItemStyle.setBorderWidth(3);
        emphasisItemStyle.setBorderColor("#ffffff");
        
        emphasis.setItemStyle(emphasisItemStyle);
        series.setEmphasis(emphasis);
    }
    
    /**
     * 应用样式到箱线图系列（增强版，支持状态样式）
     * @param series 箱线图系列
     * @param style 样式对象
     */
    public static void applyToBoxplotSeries(BoxplotSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置透明度
        if (style.getAlpha() < 1.0f) {
            String colorWithAlpha = StyleConverter.applyAlpha(style.getColor(), style.getAlpha());
            series.setColor(colorWithAlpha);
        }
        
        // 设置itemStyle
        org.icepear.echarts.components.series.ItemStyle itemStyle = StyleConverter.toEChartsItemStyle(style);
        series.setItemStyle(itemStyle);
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        } else if (style.getAnimationDuration() > 0) {
            series.setAnimationDuration(style.getAnimationDuration());
        }
        
        // 应用状态样式
        applyStateStylesToBoxplotSeries(series, style);
    }
    
    /**
     * 应用状态样式到箱线图系列
     * @param series 箱线图系列
     * @param style 样式对象
     */
    private static void applyStateStylesToBoxplotSeries(BoxplotSeries series, PlotStyle style) {
        // 创建高亮状态样式
        org.icepear.echarts.charts.boxplot.BoxplotEmphasis emphasis = new org.icepear.echarts.charts.boxplot.BoxplotEmphasis();
        
        // 设置高亮时的itemStyle
        org.icepear.echarts.components.series.ItemStyle emphasisItemStyle = StyleConverter.toEChartsItemStyle(style);
        
        // 高亮时增加亮度和边框
        if (style.getColor() != null) {
            String brightColor = brightenColor(style.getColor(), 0.25);
            emphasisItemStyle.setColor(brightColor);
        }
        
        // 增加边框宽度和特殊颜色
        emphasisItemStyle.setBorderWidth(Math.max(2, style.getLineWidth() + 1));
        emphasisItemStyle.setBorderColor("#ff6b6b");
        
        // 增加阴影效果
        emphasisItemStyle.setShadowBlur(8);
        emphasisItemStyle.setShadowColor("rgba(255, 107, 107, 0.4)");
        
        emphasis.setItemStyle(emphasisItemStyle);
        series.setEmphasis(emphasis);
    }
    
    /**
     * 应用样式到K线图系列（增强版，支持状态样式）
     * @param series K线图系列
     * @param style 样式对象
     */
    public static void applyToCandlestickSeries(CandlestickSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // K线图主要通过itemStyle设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置透明度
        if (style.getAlpha() < 1.0f) {
            String colorWithAlpha = StyleConverter.applyAlpha(style.getColor(), style.getAlpha());
            series.setColor(colorWithAlpha);
        }
        
        // 设置itemStyle
        org.icepear.echarts.charts.candlestick.CandlestickItemStyle itemStyle = StyleConverter.toEChartsCandlestickItemStyle(style);
        series.setItemStyle(itemStyle);
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        } else if (style.getAnimationDuration() > 0) {
            series.setAnimationDuration(style.getAnimationDuration());
        }
        
        // 应用状态样式
        applyStateStylesToCandlestickSeries(series, style);
    }
    
    /**
     * 应用状态样式到K线图系列
     * @param series K线图系列
     * @param style 样式对象
     */
    private static void applyStateStylesToCandlestickSeries(CandlestickSeries series, PlotStyle style) {
        // 创建高亮状态样式
        org.icepear.echarts.charts.candlestick.CandlestickEmphasis emphasis = new org.icepear.echarts.charts.candlestick.CandlestickEmphasis();
        
        // 设置高亮时的itemStyle
        org.icepear.echarts.charts.candlestick.CandlestickItemStyle emphasisItemStyle = StyleConverter.toEChartsCandlestickItemStyle(style);
        
        // 高亮时增加亮度
        if (style.getColor() != null) {
            String brightColor = brightenColor(style.getColor(), 0.2);
            emphasisItemStyle.setColor(brightColor);
        }
        
        // 增加边框宽度
        if (style.getLineWidth() > 0) {
            emphasisItemStyle.setBorderWidth(style.getLineWidth() + 1);
        }
        
        // 增加阴影效果
        emphasisItemStyle.setShadowBlur(12);
        emphasisItemStyle.setShadowColor("rgba(0, 0, 0, 0.4)");
        
        emphasis.setItemStyle(emphasisItemStyle);
        series.setEmphasis(emphasis);
    }
    
    /**
     * 应用样式到漏斗图系列
     * @param series 漏斗图系列
     * @param style 样式对象
     */
    public static void applyToFunnelSeries(FunnelSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
    }
    
    /**
     * 应用样式到桑基图系列
     * @param series 桑基图系列
     * @param style 样式对象
     */
    public static void applyToSankeySeries(SankeySeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 桑基图主要通过节点和边设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
    }
    
    /**
     * 应用样式到旭日图系列
     * @param series 旭日图系列
     * @param style 样式对象
     */
    public static void applyToSunburstSeries(SunburstSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
    }
    
    /**
     * 应用样式到主题河流图系列
     * @param series 主题河流图系列
     * @param style 样式对象
     */
    public static void applyToThemeRiverSeries(ThemeRiverSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
    }
    
    /**
     * 应用样式到树图系列
     * @param series 树图系列
     * @param style 样式对象
     */
    public static void applyToTreeSeries(TreeSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
    }
    
    /**
     * 应用样式到矩形树图系列
     * @param series 矩形树图系列
     * @param style 样式对象
     */
    public static void applyToTreemapSeries(TreemapSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
    }
    
    /**
     * 应用样式到关系图系列
     * @param series 关系图系列
     * @param style 样式对象
     */
    public static void applyToGraphSeries(GraphSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置颜色
        if (style.getColor() != null) {
            series.setColor(style.getColor());
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
    }
    
    /**
     * 应用样式到平行坐标图系列
     * @param series 平行坐标图系列
     * @param style 样式对象
     */
    public static void applyToParallelSeries(ParallelSeries series, PlotStyle style) {
        if (style == null) return;
        
        // 设置名称/标签
        if (style.getLabel() != null && !style.getLabel().isEmpty()) {
            series.setName(style.getLabel());
        }
        
        // 设置线条样式
        if (style.getColor() != null) {
            series.setLineStyle(StyleConverter.toEChartsLineStyle(style));
        }
        
        // 设置透明度
        if (style.getAlpha() < 1.0f) {
            // 平行坐标图不支持areaStyle，通过线条透明度设置
            series.setLineStyle(StyleConverter.toEChartsLineStyle(style));
        }
        
        // 设置动画
        if (!style.isEnableAnimation()) {
            series.setAnimation(false);
        }
    }
    
    // ========== 辅助方法 ==========
    
    /**
     * 创建通用的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = new HashMap<>();
        
        if (style.getColor() != null) {
            itemStyle.put("color", style.getColor());
        }
        
        if (style.getAlpha() < 1.0f) {
            String colorWithAlpha = StyleConverter.applyAlpha(style.getColor(), style.getAlpha());
            itemStyle.put("color", colorWithAlpha);
        }
        
        if (style.getEdgeColor() != null) {
            Map<String, Object> border = new HashMap<>();
            border.put("color", style.getEdgeColor());
            border.put("width", style.getLineWidth());
            itemStyle.put("border", border);
        }
        
        return itemStyle;
    }
    
    /**
     * 创建柱状图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createBarItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 柱状图特有的样式设置
        if (style.getFillStyle() != null && !style.getFillStyle().equals("solid")) {
            itemStyle.put("fillStyle", style.getFillStyle());
        }
        
        return itemStyle;
    }
    
    /**
     * 创建仪表盘的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createGaugeItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 仪表盘特有的样式设置
        itemStyle.put("shadowBlur", 10);
        itemStyle.put("shadowColor", "rgba(0, 0, 0, 0.3)");
        
        return itemStyle;
    }
    
    /**
     * 创建箱线图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createBoxplotItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 箱线图特有的样式设置
        itemStyle.put("borderWidth", style.getLineWidth());
        
        return itemStyle;
    }
    
    /**
     * 创建K线图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createCandlestickItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // K线图特有的样式设置
        itemStyle.put("borderWidth", style.getLineWidth());
        
        return itemStyle;
    }
    
    /**
     * 创建漏斗图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createFunnelItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 漏斗图特有的样式设置
        itemStyle.put("borderWidth", style.getLineWidth());
        
        return itemStyle;
    }
    
    /**
     * 创建桑基图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createSankeyItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 桑基图特有的样式设置
        itemStyle.put("borderWidth", style.getLineWidth());
        
        return itemStyle;
    }
    
    /**
     * 创建旭日图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createSunburstItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 旭日图特有的样式设置
        itemStyle.put("borderWidth", style.getLineWidth());
        
        return itemStyle;
    }
    
    /**
     * 创建主题河流图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createThemeRiverItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 主题河流图特有的样式设置
        itemStyle.put("borderWidth", style.getLineWidth());
        
        return itemStyle;
    }
    
    /**
     * 创建树图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createTreeItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 树图特有的样式设置
        itemStyle.put("borderWidth", style.getLineWidth());
        
        return itemStyle;
    }
    
    /**
     * 创建矩形树图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createTreemapItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 矩形树图特有的样式设置
        itemStyle.put("borderWidth", style.getLineWidth());
        
        return itemStyle;
    }
    
    /**
     * 创建关系图的itemStyle配置
     * @param style 样式对象
     * @return itemStyle配置
     */
    private static Map<String, Object> createGraphItemStyle(PlotStyle style) {
        Map<String, Object> itemStyle = createItemStyle(style);
        
        // 关系图特有的样式设置
        itemStyle.put("borderWidth", style.getLineWidth());
        
        return itemStyle;
    }
    
    // ========== 辅助方法 ==========
    
    /**
     * 增加颜色亮度
     * @param color 原始颜色
     * @param factor 亮度因子 (0-1)
     * @return 增亮后的颜色
     */
    private static String brightenColor(String color, double factor) {
        // 简化实现，实际应该解析颜色并调整亮度
        if (color != null && color.startsWith("#")) {
            return color; // 暂时返回原色，实际应该计算增亮
        }
        return color;
    }
    
    /**
     * 降低颜色饱和度
     * @param color 原始颜色
     * @param factor 饱和度因子 (0-1)
     * @return 降饱和后的颜色
     */
    private static String desaturateColor(String color, double factor) {
        // 简化实现，实际应该解析颜色并调整饱和度
        if (color != null && color.startsWith("#")) {
            return color; // 暂时返回原色，实际应该计算降饱和
        }
        return color;
    }
    
    /**
     * 获取对比色（用于文本等）
     * @param backgroundColor 背景颜色
     * @return 对比色
     */
    private static String getContrastColor(String backgroundColor) {
        return ColorPalette.getContrastColor(backgroundColor);
    }
    
    /**
     * 创建通用的状态样式应用器
     * @param baseStyle 基础样式
     * @param stateType 状态类型 ("emphasis", "blur", "select")
     * @return 状态样式对象
     */
    public static Map<String, Object> createStateStyle(PlotStyle baseStyle, String stateType) {
        Map<String, Object> stateStyle = new HashMap<>();
        
        switch (stateType.toLowerCase()) {
            case "emphasis":
                // 高亮状态：增加亮度、边框宽度、阴影
                stateStyle.put("color", brightenColor(baseStyle.getColor(), 0.2));
                stateStyle.put("borderWidth", baseStyle.getLineWidth() + 1);
                stateStyle.put("shadowBlur", 10);
                stateStyle.put("shadowColor", "rgba(0, 0, 0, 0.3)");
                break;
                
            case "blur":
                // 模糊状态：降低透明度、饱和度
                stateStyle.put("opacity", baseStyle.getAlpha() * 0.3);
                stateStyle.put("color", desaturateColor(baseStyle.getColor(), 0.5));
                break;
                
            case "select":
                // 选中状态：使用对比色、特殊边框
                stateStyle.put("color", getContrastColor(baseStyle.getColor()));
                stateStyle.put("borderWidth", baseStyle.getLineWidth() + 2);
                stateStyle.put("borderColor", "#FF6B6B");
                break;
                
            default:
                // 默认状态：使用基础样式
                stateStyle.put("color", baseStyle.getColor());
                stateStyle.put("opacity", baseStyle.getAlpha());
                stateStyle.put("borderWidth", baseStyle.getLineWidth());
                break;
        }
        
        return stateStyle;
    }
}
