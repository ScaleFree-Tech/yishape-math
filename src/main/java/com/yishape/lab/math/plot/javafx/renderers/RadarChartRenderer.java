package com.yishape.lab.math.plot.javafx.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

import java.util.List;

/**
 * 雷达图渲染器
 * 
 * @author lteb2
 */
public class RadarChartRenderer implements JavaFxChartRenderer {
    
    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) return;
        
        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);
        SeriesData series = data.get(0);
        
        if (series.y == null) return;
        
        int numIndicators = series.y.length();
        
        // 清空画布
        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        
        // 绘制标题
        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        
        // 计算雷达图参数
        double centerX = config.width / 2.0;
        double centerY = config.height / 2.0 + 20;
        double maxRadius = Math.min(config.width, config.height) / 2.0 - 120;
        
        double angleStep = 2 * Math.PI / numIndicators;
        
        String[] palette = themeManager.getColorPalette();
        Color gridLineColor = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("gridColor"), Color.LIGHTGRAY);
        
        // 绘制网格多边形
        gc.setStroke(gridLineColor);
        gc.setLineWidth(1);
        
        for (int level = 1; level <= 5; level++) {
            double r = maxRadius * level / 5;
            double[] xPoints = new double[numIndicators];
            double[] yPoints = new double[numIndicators];
            
            for (int i = 0; i < numIndicators; i++) {
                double angle = i * angleStep - Math.PI / 2;
                xPoints[i] = centerX + Math.cos(angle) * r;
                yPoints[i] = centerY + Math.sin(angle) * r;
            }
            
            gc.strokePolygon(xPoints, yPoints, numIndicators);
        }
        
        // 绘制指标轴和标签
        gc.setFill(themeManager.getTextColor());
        gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 11));
        
        for (int i = 0; i < numIndicators; i++) {
            double angle = i * angleStep - Math.PI / 2;
            double x = centerX + Math.cos(angle) * maxRadius;
            double y = centerY + Math.sin(angle) * maxRadius;
            
            gc.setStroke(gridLineColor);
            gc.strokeLine(centerX, centerY, x, y);
            
            // 绘制标签
            double labelX = centerX + Math.cos(angle) * (maxRadius + 30);
            double labelY = centerY + Math.sin(angle) * (maxRadius + 30);
            gc.setTextAlign(TextAlignment.CENTER);
            
            String label = series.labels != null && i < series.labels.size()
                ? series.labels.get(i) : "指标" + (i + 1);
            gc.fillText(label, labelX, labelY);
        }
        
        // 绘制数据区域
        double[] xPoints = new double[numIndicators];
        double[] yPoints = new double[numIndicators];
        
        for (int i = 0; i < numIndicators; i++) {
            double angle = i * angleStep - Math.PI / 2;
            double value = series.y.get(i);
            double r = (value / 100.0) * maxRadius;
            xPoints[i] = centerX + Math.cos(angle) * r;
            yPoints[i] = centerY + Math.sin(angle) * r;
        }
        
        String areaColor = palette.length > 0 ? palette[0] : "#5470c6";
        gc.setFill(Color.web(areaColor, 0.4));
        gc.setStroke(Color.web(areaColor));
        gc.setLineWidth(2);
        gc.fillPolygon(xPoints, yPoints, numIndicators);
        gc.strokePolygon(xPoints, yPoints, numIndicators);
        
        // 绘制数据点（使用主题调色板轮转，便于区分主题）
        for (int i = 0; i < numIndicators; i++) {
            String c = palette[i % palette.length];
            gc.setFill(Color.web(c));
            gc.setStroke(JavaFxChartUtils.parseColorFromConfig(
                themeManager.getCurrentConfig().get("axisColor"), Color.DARKGRAY));
            gc.setLineWidth(1);
            gc.fillOval(xPoints[i] - 4, yPoints[i] - 4, 8, 8);
            gc.strokeOval(xPoints[i] - 4, yPoints[i] - 4, 8, 8);
        }
        JavaFxChartUtils.drawCartesianAxisTitles(gc, config, themeManager);
    }
    
    @Override
    public String getChartType() {
        return "radar";
    }
    
    @Override
    public boolean supportsAnimation() {
        return true;
    }
    
    @Override
    public int getAnimationDuration() {
        return 1200;
    }
}
