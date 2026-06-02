package com.yishape.lab.math.plot.javafx;

import com.yishape.lab.math.plot.ColorPalette;
import com.yishape.lab.math.plot.PlotStyle;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.scene.shape.StrokeLineJoin;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;

import java.io.Serializable;

/**
 * JavaFX样式应用器
 * 将PlotStyle应用到JavaFX GraphicsContext
 * 
 * @author lteb2
 */
public class JavaFxStyleApplier implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * 应用样式到GraphicsContext
     * @param gc GraphicsContext
     * @param style PlotStyle
     */
    public static void applyStyle(GraphicsContext gc, PlotStyle style) {
        if (style == null) return;
        
        // 应用颜色
        if (style.getColor() != null) {
            Color color = parseColor(style.getColor());
            gc.setFill(color);
            gc.setStroke(color);
        }
        
        // 应用线条宽度
        if (style.getLineWidth() > 0) {
            gc.setLineWidth(style.getLineWidth());
        }
        
        // 应用线条样式
        if (style.getLineStyle() != null) {
            applyLineStyle(gc, style.getLineStyle());
        }
        
        // 应用线条端点样式
        if (style.getLineCap() != null) {
            applyLineCap(gc, style.getLineCap());
        }
        
        // 应用线条连接样式
        if (style.getLineJoin() != null) {
            applyLineJoin(gc, style.getLineJoin());
        }
        
        // 应用字体
        if (style.getFontFamily() != null || style.getFontSize() > 0) {
            applyFont(gc, style);
        }
        
        // 应用透明度
        if (style.getAlpha() < 1.0 && style.getAlpha() >= 0) {
            gc.setGlobalAlpha(style.getAlpha());
        } else {
            gc.setGlobalAlpha(1.0);
        }
    }
    
    /**
     * 应用填充样式
     * @param gc GraphicsContext
     * @param style PlotStyle
     */
    public static void applyFillStyle(GraphicsContext gc, PlotStyle style) {
        if (style == null) return;
        
        if (style.getFaceColor() != null) {
            gc.setFill(parseColor(style.getFaceColor()));
        } else if (style.getColor() != null) {
            gc.setFill(parseColor(style.getColor()));
        }
        
        // 应用填充透明度
        if (style.getAlpha() < 1.0 && style.getAlpha() >= 0) {
            gc.setGlobalAlpha(style.getAlpha());
        }
    }
    
    /**
     * 应用描边样式
     * @param gc GraphicsContext
     * @param style PlotStyle
     */
    public static void applyStrokeStyle(GraphicsContext gc, PlotStyle style) {
        if (style == null) return;
        
        if (style.getEdgeColor() != null) {
            gc.setStroke(parseColor(style.getEdgeColor()));
        } else if (style.getColor() != null) {
            gc.setStroke(parseColor(style.getColor()));
        }
        
        if (style.getLineWidth() > 0) {
            gc.setLineWidth(style.getLineWidth());
        }
    }
    
    /**
     * 应用标记样式
     * @param gc GraphicsContext
     * @param style PlotStyle
     */
    public static void applyMarkerStyle(GraphicsContext gc, PlotStyle style) {
        if (style == null) return;
        
        if (style.getMarkerColor() != null) {
            gc.setFill(parseColor(style.getMarkerColor()));
        } else if (style.getColor() != null) {
            gc.setFill(parseColor(style.getColor()));
        }
        
        if (style.getMarkerAlpha() < 1.0 && style.getMarkerAlpha() >= 0) {
            gc.setGlobalAlpha(style.getMarkerAlpha());
        }
    }
    
    /**
     * 仅设置描边虚线相位（不影响颜色/线宽）。用于图例等与 {@link #applyStyle} 隔离的绘制。
     *
     * @param lineStyle {@code solid}、{@code dashed}、{@code dotted}、{@code dashdot} 等，{@code null} 视为实线
     */
    public static void applyLineDashPattern(GraphicsContext gc, String lineStyle) {
        if (lineStyle == null || lineStyle.isEmpty()) {
            gc.setLineDashes();
            return;
        }
        applyLineStyle(gc, lineStyle);
    }

    /**
     * 应用线条样式
     * @param gc GraphicsContext
     * @param lineStyle 线条样式字符串
     */
    private static void applyLineStyle(GraphicsContext gc, String lineStyle) {
        switch (lineStyle.toLowerCase()) {
            case "solid":
            case "-":
                gc.setLineDashes();
                break;
            case "dashed":
            case "--":
                gc.setLineDashes(5, 5);
                break;
            case "dotted":
            case ":":
                gc.setLineDashes(2, 2);
                break;
            case "dashdot":
            case "-.":
                gc.setLineDashes(10, 5, 2, 5);
                break;
            default:
                gc.setLineDashes();
                break;
        }
    }
    
    /**
     * 应用线条端点样式
     * @param gc GraphicsContext
     * @param lineCap 端点样式
     */
    private static void applyLineCap(GraphicsContext gc, String lineCap) {
        switch (lineCap.toLowerCase()) {
            case "round":
                gc.setLineCap(StrokeLineCap.ROUND);
                break;
            case "square":
                gc.setLineCap(StrokeLineCap.SQUARE);
                break;
            case "butt":
            default:
                gc.setLineCap(StrokeLineCap.BUTT);
                break;
        }
    }
    
    /**
     * 应用线条连接样式
     * @param gc GraphicsContext
     * @param lineJoin 连接样式
     */
    private static void applyLineJoin(GraphicsContext gc, String lineJoin) {
        switch (lineJoin.toLowerCase()) {
            case "round":
                gc.setLineJoin(StrokeLineJoin.ROUND);
                break;
            case "bevel":
                gc.setLineJoin(StrokeLineJoin.BEVEL);
                break;
            case "miter":
            default:
                gc.setLineJoin(StrokeLineJoin.MITER);
                break;
        }
    }
    
    /**
     * 应用字体
     * @param gc GraphicsContext
     * @param style PlotStyle
     */
    private static void applyFont(GraphicsContext gc, PlotStyle style) {
        String fontFamily = style.getFontFamily();
        if (fontFamily == null || fontFamily.isEmpty()) {
            fontFamily = "Arial";
        }
        
        double fontSize = style.getFontSize();
        if (fontSize <= 0) {
            fontSize = 12;
        }
        
        FontWeight weight = FontWeight.NORMAL;
        if ("bold".equalsIgnoreCase(style.getFontWeight())) {
            weight = FontWeight.BOLD;
        }
        
        gc.setFont(Font.font(fontFamily, weight, fontSize));
    }
    
    /**
     * 解析颜色字符串
     * @param colorSpec 颜色规范
     * @return Color对象
     */
    public static Color parseColor(String colorSpec) {
        if (colorSpec == null || colorSpec.isEmpty()) {
            return Color.BLACK;
        }
        
        try {
            // 处理带透明度的十六进制颜色
            if (colorSpec.startsWith("#")) {
                if (colorSpec.length() == 9) { // #RRGGBBAA
                    String rgb = colorSpec.substring(0, 7);
                    int alpha = Integer.parseInt(colorSpec.substring(7, 9), 16);
                    Color base = Color.web(rgb);
                    return new Color(base.getRed(), base.getGreen(), base.getBlue(), alpha / 255.0);
                }
                return Color.web(colorSpec);
            }
            
            // 处理颜色名称
            String parsedColor = ColorPalette.parseColor(colorSpec);
            return Color.web(parsedColor);
        } catch (Exception e) {
            return Color.BLACK;
        }
    }
    
    /**
     * 获取标记形状
     * @param markerSpec 标记规范
     * @return 标记类型
     */
    public static MarkerType getMarkerType(String markerSpec) {
        if (markerSpec == null) return MarkerType.CIRCLE;
        
        switch (markerSpec.toLowerCase()) {
            case "o":
            case "circle":
                return MarkerType.CIRCLE;
            case "s":
            case "square":
                return MarkerType.SQUARE;
            case "^":
            case "triangle":
            case "triangle_up":
                return MarkerType.TRIANGLE_UP;
            case "v":
            case "triangle_down":
                return MarkerType.TRIANGLE_DOWN;
            case "d":
            case "diamond":
                return MarkerType.DIAMOND;
            case "*":
            case "star":
                return MarkerType.STAR;
            case "+":
            case "plus":
                return MarkerType.PLUS;
            case "x":
            case "cross":
                return MarkerType.CROSS;
            default:
                return MarkerType.CIRCLE;
        }
    }
    
    /**
     * 绘制标记
     * @param gc GraphicsContext
     * @param x X坐标
     * @param y Y坐标
     * @param markerType 标记类型
     * @param size 标记大小
     */
    public static void drawMarker(GraphicsContext gc, double x, double y, 
                                  MarkerType markerType, double size) {
        switch (markerType) {
            case CIRCLE:
                gc.fillOval(x - size / 2, y - size / 2, size, size);
                gc.strokeOval(x - size / 2, y - size / 2, size, size);
                break;
            case SQUARE:
                gc.fillRect(x - size / 2, y - size / 2, size, size);
                gc.strokeRect(x - size / 2, y - size / 2, size, size);
                break;
            case TRIANGLE_UP:
                gc.fillPolygon(
                    new double[]{x, x - size / 2, x + size / 2},
                    new double[]{y - size / 2, y + size / 2, y + size / 2},
                    3
                );
                gc.strokePolygon(
                    new double[]{x, x - size / 2, x + size / 2},
                    new double[]{y - size / 2, y + size / 2, y + size / 2},
                    3
                );
                break;
            case TRIANGLE_DOWN:
                gc.fillPolygon(
                    new double[]{x, x - size / 2, x + size / 2},
                    new double[]{y + size / 2, y - size / 2, y - size / 2},
                    3
                );
                gc.strokePolygon(
                    new double[]{x, x - size / 2, x + size / 2},
                    new double[]{y + size / 2, y - size / 2, y - size / 2},
                    3
                );
                break;
            case DIAMOND:
                gc.fillPolygon(
                    new double[]{x, x + size / 2, x, x - size / 2},
                    new double[]{y - size / 2, y, y + size / 2, y},
                    4
                );
                gc.strokePolygon(
                    new double[]{x, x + size / 2, x, x - size / 2},
                    new double[]{y - size / 2, y, y + size / 2, y},
                    4
                );
                break;
            case STAR:
                drawStar(gc, x, y, size);
                break;
            case PLUS:
                gc.strokeLine(x - size / 2, y, x + size / 2, y);
                gc.strokeLine(x, y - size / 2, x, y + size / 2);
                break;
            case CROSS:
                gc.strokeLine(x - size / 2, y - size / 2, x + size / 2, y + size / 2);
                gc.strokeLine(x + size / 2, y - size / 2, x - size / 2, y + size / 2);
                break;
        }
    }
    
    /**
     * 绘制星形标记
     * @param gc GraphicsContext
     * @param x X坐标
     * @param y Y坐标
     * @param size 大小
     */
    private static void drawStar(GraphicsContext gc, double x, double y, double size) {
        int points = 5;
        double[] xPoints = new double[points * 2];
        double[] yPoints = new double[points * 2];
        
        double outerRadius = size / 2;
        double innerRadius = size / 4;
        
        for (int i = 0; i < points * 2; i++) {
            double angle = Math.PI / 2 + i * Math.PI / points;
            double radius = (i % 2 == 0) ? outerRadius : innerRadius;
            xPoints[i] = x + Math.cos(angle) * radius;
            yPoints[i] = y - Math.sin(angle) * radius;
        }
        
        gc.fillPolygon(xPoints, yPoints, points * 2);
        gc.strokePolygon(xPoints, yPoints, points * 2);
    }
    
    /**
     * 标记类型枚举
     */
    public enum MarkerType {
        CIRCLE, SQUARE, TRIANGLE_UP, TRIANGLE_DOWN, DIAMOND, STAR, PLUS, CROSS
    }
}
