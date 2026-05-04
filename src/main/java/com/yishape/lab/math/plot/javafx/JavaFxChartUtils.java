package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.AxisTicks;
import com.yishape.lab.math.plot.PlotAxisScale;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

import java.util.List;
import java.util.Locale;

/**
 * JavaFX图表工具类
 * 提供通用的图表绘制工具方法
 * 
 * @author lteb2
 */
public class JavaFxChartUtils {

    private static double clampPositive(double v, double floor) {
        if (v <= 0.0 || Double.isNaN(v)) {
            return floor;
        }
        return v;
    }

    /** Range in data space; for LOG uses log10 of positive bounds. */
    public static double mapDataToUnit(double v, double min, double max, PlotAxisScale scale) {
        if (scale == PlotAxisScale.LOG) {
            double a = Math.log10(clampPositive(min, 1e-300));
            double b = Math.log10(clampPositive(max, 1e-300));
            if (b <= a) {
                b = a + 1e-6;
            }
            return (Math.log10(clampPositive(v, Math.pow(10, a))) - a) / (b - a);
        }
        if (max == min) {
            return 0.5;
        }
        return (v - min) / (max - min);
    }

    public static double mapXToPixel(JavaFxChartRenderer.ChartConfig config,
                                     double dataX, double xMin, double xMax) {
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double u = mapDataToUnit(dataX, xMin, xMax, config.xAxisScale);
        return config.paddingLeft + u * chartWidth;
    }

    public static double mapYToPixel(JavaFxChartRenderer.ChartConfig config,
                                     double dataY, double yMin, double yMax,
                                     PlotAxisScale yScale) {
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;
        double u = mapDataToUnit(dataY, yMin, yMax, yScale);
        return config.height - config.paddingBottom - u * chartHeight;
    }

    /**
     * 扩展笛卡尔坐标轴绘制：对数刻度标签 + 可选右侧副 Y 轴。
     */
    public static void drawAxesCartesian(GraphicsContext gc, JavaFxChartRenderer.ChartConfig config,
                                        double xMin, double xMax,
                                        double yMin, double yMax,
                                        Double y2Min, Double y2Max,
                                        JavaFxThemeManager themeManager) {
        Color axisColor = parseColorFromConfig(themeManager.getCurrentConfig().get("axisColor"), Color.DARKGRAY);
        Color gridColor = parseColorFromConfig(themeManager.getCurrentConfig().get("gridColor"), Color.LIGHTGRAY);

        Object showGridValue = themeManager.getCurrentConfig().get("showGrid");
        boolean themeWantsGrid = showGridValue == null || Boolean.TRUE.equals(showGridValue);
        boolean showGrid = config.showGrid && themeWantsGrid;

        gc.setLineWidth(1);
        if (showGrid) {
            gc.save();
            gc.setStroke(gridColor);
            gc.setLineWidth(0.85);
            gc.setGlobalAlpha(0.72);
            int xGrid = effectiveTickDivisions(config, config.xAxisTicks, 5);
            for (int i = 0; i <= xGrid; i++) {
                double frac = i / (double) xGrid;
                double xv = config.xAxisScale == PlotAxisScale.LOG
                    ? Math.pow(10, Math.log10(Math.max(xMin, 1e-300))
                    + frac * (Math.log10(Math.max(xMax, xMin * 1.0001)) - Math.log10(Math.max(xMin, 1e-300))))
                    : xMin + frac * (xMax - xMin);
                double px = mapXToPixel(config, xv, xMin, xMax);
                gc.strokeLine(px, config.paddingTop, px, config.height - config.paddingBottom);
            }
            int yGrid = effectiveTickDivisions(config, config.yAxisTicks, 5);
            for (int i = 0; i <= yGrid; i++) {
                double frac = i / (double) yGrid;
                double yv = config.yAxisScale == PlotAxisScale.LOG
                    ? Math.pow(10, Math.log10(Math.max(yMin, 1e-300))
                    + frac * (Math.log10(Math.max(yMax, yMin * 1.0001)) - Math.log10(Math.max(yMin, 1e-300))))
                    : yMin + frac * (yMax - yMin);
                double py = mapYToPixel(config, yv, yMin, yMax, config.yAxisScale);
                gc.strokeLine(config.paddingLeft, py, config.width - config.paddingRight, py);
            }
            gc.restore();
        }

        gc.setStroke(axisColor);
        gc.setLineWidth(1.35);
        gc.strokeLine(config.paddingLeft, config.height - config.paddingBottom,
            config.width - config.paddingRight, config.height - config.paddingBottom);
        gc.strokeLine(config.paddingLeft, config.paddingTop,
            config.paddingLeft, config.height - config.paddingBottom);

        drawCartesianAxisLabels(gc, config, xMin, xMax, yMin, yMax, themeManager);

        if (y2Min != null && y2Max != null && config.y2AxisLabel != null && !config.y2AxisLabel.isEmpty()) {
            gc.strokeLine(config.width - config.paddingRight, config.paddingTop,
                config.width - config.paddingRight, config.height - config.paddingBottom);
            gc.setFill(themeManager.getTextColor());
            gc.setFont(themeManager.getLabelFont());
            gc.setTextAlign(TextAlignment.LEFT);
            int ticks = 5;
            for (int i = 0; i <= ticks; i++) {
                double yv = y2Min + (y2Max - y2Min) * i / ticks;
                double py = mapYToPixel(config, yv, y2Min, y2Max, PlotAxisScale.LINEAR);
                gc.fillText(formatNumber(yv), config.width - config.paddingRight + 8, py + 5);
            }
            gc.save();
            gc.translate(config.width - 15, config.height / 2.0);
            gc.rotate(90);
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText(config.y2AxisLabel, 0, 0);
            gc.restore();
        }
    }

    private static void drawCartesianAxisLabels(GraphicsContext gc, JavaFxChartRenderer.ChartConfig config,
                                                double xMin, double xMax,
                                                double yMin, double yMax,
                                                JavaFxThemeManager themeManager) {
        Color textColor = themeManager.getTextColor();
        Font labelFont = themeManager.getLabelFont();
        gc.setFill(textColor);
        gc.setFont(labelFont);

        gc.setTextAlign(TextAlignment.CENTER);
        if (config.xAxisTicks != null && config.xAxisTicks.hasTickValues()) {
            IVector<?> xv = config.xAxisTicks.getTickValues();
            int n = xv.length();
            int xTickCount = Math.max(1, n - 1);
            List<String> labels = config.xAxisTicks.getTickLabels();
            for (int i = 0; i < n; i++) {
                double value = axisNumberAt(xv, i);
                double px = mapXToPixel(config, value, xMin, xMax);
                String text = labels != null && i < labels.size() && !labels.get(i).isEmpty()
                    ? labels.get(i) : formatNumber(value);
                gc.fillText(text, px, config.height - config.paddingBottom + 20);
            }
        } else {
            int xTicks = effectiveTickDivisions(config, config.xAxisTicks, 5);
            for (int i = 0; i <= xTicks; i++) {
                double value = config.xAxisScale == PlotAxisScale.LOG
                    ? Math.pow(10, Math.log10(Math.max(xMin, 1e-300))
                    + (i / (double) xTicks) * (Math.log10(Math.max(xMax, xMin * 1.0001))
                    - Math.log10(Math.max(xMin, 1e-300))))
                    : xMin + (xMax - xMin) * i / xTicks;
                double px = mapXToPixel(config, value, xMin, xMax);
                gc.fillText(formatNumber(value), px, config.height - config.paddingBottom + 20);
            }
        }

        gc.setTextAlign(TextAlignment.RIGHT);
        if (config.yAxisTicks != null && config.yAxisTicks.hasTickValues()) {
            IVector<?> yv = config.yAxisTicks.getTickValues();
            int n = yv.length();
            int yTickCount = Math.max(1, n - 1);
            List<String> labels = config.yAxisTicks.getTickLabels();
            for (int i = 0; i < n; i++) {
                double value = axisNumberAt(yv, i);
                double py = mapYToPixel(config, value, yMin, yMax, config.yAxisScale);
                String text = labels != null && i < labels.size() && !labels.get(i).isEmpty()
                    ? labels.get(i) : formatNumber(value);
                gc.fillText(text, config.paddingLeft - 10, py + 5);
            }
        } else {
            int yTicks = effectiveTickDivisions(config, config.yAxisTicks, 5);
            for (int i = 0; i <= yTicks; i++) {
                double value = config.yAxisScale == PlotAxisScale.LOG
                    ? Math.pow(10, Math.log10(Math.max(yMin, 1e-300))
                    + (i / (double) yTicks) * (Math.log10(Math.max(yMax, yMin * 1.0001))
                    - Math.log10(Math.max(yMin, 1e-300))))
                    : yMin + (yMax - yMin) * i / yTicks;
                double py = mapYToPixel(config, value, yMin, yMax, config.yAxisScale);
                gc.fillText(formatNumber(value), config.paddingLeft - 10, py + 5);
            }
        }

        if (!config.xlabel.isEmpty()) {
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText(config.xlabel, config.width / 2.0, config.height - 10);
        }
        if (!config.ylabel.isEmpty()) {
            gc.save();
            gc.translate(20, config.height / 2.0);
            gc.rotate(-90);
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText(config.ylabel, 0, 0);
            gc.restore();
        }
    }

    /**
     * 计算数据范围
     * @param dataList 数据列表
     * @param isX 是否计算X轴范围
     * @return [min, max] 范围数组
     */
    public static double[] calculateRange(List<JavaFxChartRenderer.SeriesData> dataList, boolean isX) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (JavaFxChartRenderer.SeriesData series : dataList) {
            IVector<Double> data = isX ? series.x : series.y;
            if (data == null) continue;
            
            for (int i = 0; i < data.length(); i++) {
                double value = data.get(i);
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
        }
        
        if (min == Double.MAX_VALUE) {
            min = 0;
            max = 1;
        }
        
        // 添加边距
        double margin = (max - min) * 0.1;
        if (margin == 0) margin = Math.abs(min) * 0.1;
        if (margin == 0) margin = 1;
        
        return new double[]{min - margin, max + margin};
    }
    
    /**
     * 计算矩阵数据范围
     * @param data 数据矩阵
     * @return [min, max] 范围数组
     */
    public static double[] calculateMatrixRange(double[][] data) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (double[] row : data) {
            for (double value : row) {
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
        }
        
        if (min == Double.MAX_VALUE) {
            min = 0;
            max = 1;
        }
        
        return new double[]{min, max};
    }

    private static int axisTickCount(AxisTicks ticks, int fallbackDivisions) {
        if (ticks != null && ticks.hasTickValues()) {
            int n = ticks.getTickValues().length();
            return Math.max(1, n - 1);
        }
        return fallbackDivisions;
    }

    /** jointplot 等小窗格：限制网格线/刻度段数，减轻标签重叠 */
    private static int effectiveTickDivisions(JavaFxChartRenderer.ChartConfig config,
                                              AxisTicks ticks, int fallbackDivisions) {
        if (config.maxAxisTicks > 0) {
            return Math.max(2, config.maxAxisTicks);
        }
        return axisTickCount(ticks, fallbackDivisions);
    }

    private static double axisNumberAt(IVector<?> v, int i) {
        Object o = v.get(i);
        if (o instanceof Number) {
            return ((Number) o).doubleValue();
        }
        return Double.parseDouble(String.valueOf(o));
    }

    /**
     * 绘制图例（线型/虚线、散点、置信带、面积、误差棒等与序列类型一致的示意符号）。
     */
    public static void drawLegend(GraphicsContext gc, List<JavaFxChartRenderer.SeriesData> dataList,
                                  JavaFxChartRenderer.ChartConfig config, JavaFxThemeManager themeManager) {
        if (dataList.size() <= 1) {
            return;
        }

        gc.save();
        try {
            gc.setGlobalAlpha(1.0);
            gc.setLineDashes();

            double itemHeight = 25;
            int n = dataList.size();
            double legendNameWidth = 100;
            for (JavaFxChartRenderer.SeriesData s : dataList) {
                if (s.name != null) {
                    legendNameWidth = Math.max(legendNameWidth, Math.min(300, s.name.length() * 7.5));
                }
            }
            double legendWidth = legendNameWidth;
            double legendHeight = n * itemHeight + 20;

            double legendX = config.width - config.paddingRight - legendWidth + 20;
            double legendY = config.paddingTop + 30;

            if (legendX < config.paddingLeft) {
                legendX = config.paddingLeft + 10;
            }

            Color legendBorder = parseColorFromConfig(themeManager.getCurrentConfig().get("axisColor"),
                Color.web("#cbd5e1"));
            Color bg = themeManager.getBackgroundColor();
            gc.setFill(new Color(bg.getRed(), bg.getGreen(), bg.getBlue(), 0.94));
            gc.fillRoundRect(legendX - 8, legendY - 8, legendWidth + 16, legendHeight + 16, 10, 10);
            gc.setStroke(legendBorder);
            gc.setLineWidth(0.9);
            gc.setLineDashes();
            gc.strokeRoundRect(legendX - 8, legendY - 8, legendWidth + 16, legendHeight + 16, 10, 10);

            String[] palette = themeManager.getColorPalette();

            for (int i = 0; i < n; i++) {
                JavaFxChartRenderer.SeriesData series = dataList.get(i);
                Color color = series.style != null && series.style.getColor() != null
                    ? JavaFxStyleApplier.parseColor(series.style.getColor())
                    : Color.web(palette[i % palette.length]);

                double yRow = legendY + i * itemHeight;
                double cy = yRow + 7.5;
                String type = series.type != null ? series.type : "line";

                gc.save();
                try {
                    switch (type) {
                        case "scatter" -> drawLegendSwatchScatter(gc, color, legendX, cy, series);
                        case "ci_band" -> drawLegendSwatchCiBand(gc, color, legendX, yRow);
                        case "area" -> drawLegendSwatchArea(gc, color, legendX, yRow);
                        case "errorbar" -> drawLegendSwatchErrorbar(gc, color, legendX, cy);
                        case "step", "line" -> drawLegendSwatchLine(gc, color, legendX, cy, series);
                        default -> drawLegendSwatchLine(gc, color, legendX, cy, series);
                    }
                } finally {
                    gc.restore();
                }

                gc.setFill(themeManager.getTextColor());
                gc.setFont(themeManager.getLabelFont());
                gc.setTextAlign(TextAlignment.LEFT);
                gc.setTextBaseline(javafx.geometry.VPos.CENTER);
                gc.setLineDashes();
                gc.fillText(series.name, legendX + 22, legendY + i * itemHeight + 7.5);
            }
        } finally {
            gc.restore();
        }
    }

    private static void drawLegendSwatchLine(GraphicsContext gc, Color color, double legendX, double cy,
                                            JavaFxChartRenderer.SeriesData series) {
        gc.setStroke(color);
        double lw = series.style != null && series.style.getLineWidth() > 0 ? series.style.getLineWidth() : 1.75;
        gc.setLineWidth(Math.min(lw, 2.6));
        String ls = series.style != null ? series.style.getLineStyle() : null;
        JavaFxStyleApplier.applyLineDashPattern(gc, ls);
        gc.setLineCap(StrokeLineCap.ROUND);
        gc.strokeLine(legendX + 1.5, cy, legendX + 13.5, cy);
    }

    private static void drawLegendSwatchScatter(GraphicsContext gc, Color color, double legendX, double cy,
                                               JavaFxChartRenderer.SeriesData series) {
        gc.setLineDashes();
        gc.setStroke(color);
        gc.setLineWidth(1);
        double r = series.style != null ? Math.max(2.5, series.style.getMarkerSize() * 0.55) : 4;
        gc.setFill(color);
        double cx = legendX + 7.5;
        gc.fillOval(cx - r / 2, cy - r / 2, r, r);
        gc.strokeOval(cx - r / 2, cy - r / 2, r, r);
    }

    private static void drawLegendSwatchCiBand(GraphicsContext gc, Color color, double legendX, double yRow) {
        Color fill = Color.color(color.getRed(), color.getGreen(), color.getBlue(), 0.35);
        gc.setFill(fill);
        gc.fillRoundRect(legendX, yRow + 2, 15, 11, 3, 3);
        gc.setStroke(Color.color(color.getRed(), color.getGreen(), color.getBlue(), 0.65));
        gc.setLineWidth(0.85);
        gc.setLineDashes();
        gc.strokeRoundRect(legendX, yRow + 2, 15, 11, 3, 3);
    }

    private static void drawLegendSwatchArea(GraphicsContext gc, Color color, double legendX, double yRow) {
        Color fill = Color.color(color.getRed(), color.getGreen(), color.getBlue(), 0.45);
        gc.setFill(fill);
        gc.fillRoundRect(legendX, yRow + 2, 15, 11, 3, 3);
        gc.setStroke(color);
        gc.setLineWidth(0.9);
        gc.setLineDashes();
        gc.strokeRoundRect(legendX, yRow + 2, 15, 11, 3, 3);
    }

    private static void drawLegendSwatchErrorbar(GraphicsContext gc, Color color, double legendX, double cy) {
        gc.setStroke(color);
        gc.setLineWidth(1.15);
        gc.setLineDashes();
        double x = legendX + 7.5;
        gc.strokeLine(x, cy - 5, x, cy + 5);
        gc.strokeLine(x - 3, cy - 5, x + 3, cy - 5);
        gc.strokeLine(x - 3, cy + 5, x + 3, cy + 5);
    }
    
    /**
     * 绘制标题
     * @param gc GraphicsContext
     * @param config 图表配置
     * @param themeManager 主题管理器
     */
    public static void drawTitle(GraphicsContext gc, JavaFxChartRenderer.ChartConfig config,
                                  JavaFxThemeManager themeManager) {
        if (config.title.isEmpty()) return;
        
        gc.setFill(themeManager.getTextColor());
        gc.setFont(themeManager.getTitleFont());
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(config.title, config.width / 2.0, 36);
        
        if (!config.subtitle.isEmpty()) {
            gc.setFill(themeManager.getMutedTextColor());
            gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 12.5));
            double subY = themeManager.getTitleFont().getSize() >= 19 ? 58 : 56;
            gc.fillText(config.subtitle, config.width / 2.0, subY);
        }
    }
    
    /**
     * 插值颜色
     * @param t 插值参数 (0-1)
     * @param start 起始颜色
     * @param middle 中间颜色
     * @param end 结束颜色
     * @return 插值后的颜色
     */
    public static Color interpolateColor(double t, Color start, Color middle, Color end) {
        if (t < 0.5) {
            double s = t * 2;
            return new Color(
                start.getRed() + (middle.getRed() - start.getRed()) * s,
                start.getGreen() + (middle.getGreen() - start.getGreen()) * s,
                start.getBlue() + (middle.getBlue() - start.getBlue()) * s,
                start.getOpacity() + (middle.getOpacity() - start.getOpacity()) * s
            );
        } else {
            double s = (t - 0.5) * 2;
            return new Color(
                middle.getRed() + (end.getRed() - middle.getRed()) * s,
                middle.getGreen() + (end.getGreen() - middle.getGreen()) * s,
                middle.getBlue() + (end.getBlue() - middle.getBlue()) * s,
                middle.getOpacity() + (end.getOpacity() - middle.getOpacity()) * s
            );
        }
    }
    
    /**
     * 归一化值到范围
     * @param value 值
     * @param min 最小值
     * @param max 最大值
     * @return 归一化值 (0-1)
     */
    public static double normalize(double value, double min, double max) {
        if (max == min) return 0;
        return (value - min) / (max - min);
    }
    
    /**
     * 映射值到像素坐标
     * @param value 数据值
     * @param min 数据最小值
     * @param max 数据最大值
     * @param pixelMin 像素最小值
     * @param pixelMax 像素最大值
     * @return 像素坐标
     */
    public static double mapToPixel(double value, double min, double max, double pixelMin, double pixelMax) {
        double normalized = normalize(value, min, max);
        return pixelMin + normalized * (pixelMax - pixelMin);
    }
    
    /**
     * 从配置中解析颜色
     * @param colorValue 配置中的颜色值（Color对象或String）
     * @param defaultColor 默认颜色
     * @return Color对象
     */
    public static Color parseColorFromConfig(Object colorValue, Color defaultColor) {
        if (colorValue == null) {
            return defaultColor;
        }
        if (colorValue instanceof Color) {
            return (Color) colorValue;
        }
        if (colorValue instanceof String) {
            try {
                return Color.web((String) colorValue);
            } catch (Exception e) {
                return defaultColor;
            }
        }
        return defaultColor;
    }
    
    /**
     * 坐标轴刻度等场景的数字格式化：整数优先；非整数使用约 3 位有效数字，
     * 避免 KDE 密度被 {@code %.1f} 打成 {@code 0.0}，同时避免过长小数造成拥挤。
     */
    public static String formatNumber(double value) {
        if (Double.isNaN(value)) {
            return "NaN";
        }
        if (Double.isInfinite(value)) {
            return value > 0 ? "inf" : "-inf";
        }
        if (value == 0.0) {
            return "0";
        }
        long asLong = Math.round(value);
        if (Math.abs(value - asLong) < 1e-9 && Math.abs(asLong) <= 1_000_000_000_000L) {
            return String.valueOf(asLong);
        }
        return compactNonIntegerLabel(value);
    }

    /**
     * 非整数轴标签：3 位有效数字为主；极小正数用 {@code %.1e} 缩短宽度。
     */
    private static String compactNonIntegerLabel(double value) {
        double av = Math.abs(value);
        String s;
        if (av > 0 && av < 1e-3) {
            s = String.format(Locale.ROOT, "%.1e", value);
        } else {
            s = String.format(Locale.ROOT, "%.3g", value);
        }
        int e = Math.max(s.indexOf('e'), s.indexOf('E'));
        if (e >= 0) {
            return s;
        }
        int dot = s.indexOf('.');
        if (dot < 0) {
            return s;
        }
        int end = s.length();
        while (end > dot + 1 && s.charAt(end - 1) == '0') {
            end--;
        }
        if (end > dot && s.charAt(end - 1) == '.') {
            end--;
        }
        return s.substring(0, end);
    }
    
    /**
     * 绘制圆角矩形
     * @param gc GraphicsContext
     * @param x X坐标
     * @param y Y坐标
     * @param width 宽度
     * @param height 高度
     * @param radius 圆角半径
     */
    public static void drawRoundedRect(GraphicsContext gc, double x, double y, 
                                       double width, double height, double radius) {
        gc.beginPath();
        gc.moveTo(x + radius, y);
        gc.lineTo(x + width - radius, y);
        gc.arcTo(x + width, y, x + width, y + radius, radius);
        gc.lineTo(x + width, y + height - radius);
        gc.arcTo(x + width, y + height, x + width - radius, y + height, radius);
        gc.lineTo(x + radius, y + height);
        gc.arcTo(x, y + height, x, y + height - radius, radius);
        gc.lineTo(x, y + radius);
        gc.arcTo(x, y, x + radius, y, radius);
        gc.closePath();
        gc.fill();
        gc.stroke();
    }

    /**
     * 注册用于悬停/点击的笛卡尔像素点（逻辑坐标系，与绘图一致）
     */
    public static void registerHit(JavaFxChartRenderer.ChartConfig config, double px, double py,
                                   double dataX, double dataY, String seriesName,
                                   int seriesIndex, int pointIndex) {
        if (config.hitTestHandler == null) {
            return;
        }
        config.hitTestHandler.addDataPoint(new JavaFxInteractionHandler.DataPoint(
            px, py, dataX, dataY, seriesName, seriesIndex, pointIndex));
    }
}
