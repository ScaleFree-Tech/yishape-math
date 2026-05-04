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
 * 极坐标图表渲染器（支持柱状图、线图、散点图）
 * 
 * @author lteb2
 */
public class PolarChartRenderer implements JavaFxChartRenderer {
    
    private String polarType; // "bar", "line", "scatter"
    
    public PolarChartRenderer(String polarType) {
        this.polarType = polarType;
    }
    
    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) return;
        
        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);
        SeriesData series = data.get(0);
        
        if (series.y == null) return;
        
        // 计算最大值
        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < series.y.length(); i++) {
            maxValue = Math.max(maxValue, series.y.get(i));
        }
        
        // 清空画布
        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        
        // 绘制标题
        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        
        // 计算极坐标参数
        double centerX = config.width / 2.0;
        double centerY = config.height / 2.0 + 20;
        double maxRadius = Math.min(config.width, config.height) / 2.0 - 100;
        
        int numCategories = series.y.length();
        double angleStep = 2 * Math.PI / numCategories;
        
        String[] palette = themeManager.getColorPalette();
        
        // 绘制网格圆
        gc.setStroke(Color.LIGHTGRAY);
        gc.setLineWidth(1);
        for (int i = 1; i <= 5; i++) {
            double r = maxRadius * i / 5;
            gc.strokeOval(centerX - r, centerY - r, r * 2, r * 2);
        }
        
        // 绘制射线和标签
        gc.setFill(themeManager.getTextColor());
        gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 10));
        
        for (int i = 0; i < numCategories; i++) {
            double angle = i * angleStep - Math.PI / 2;
            double x = centerX + Math.cos(angle) * maxRadius;
            double y = centerY + Math.sin(angle) * maxRadius;
            
            gc.setStroke(Color.LIGHTGRAY);
            gc.strokeLine(centerX, centerY, x, y);
            
            // 绘制标签
            double labelX = centerX + Math.cos(angle) * (maxRadius + 30);
            double labelY = centerY + Math.sin(angle) * (maxRadius + 30);
            gc.setTextAlign(TextAlignment.CENTER);
            String label = series.labels != null && i < series.labels.size()
                ? series.labels.get(i) : "Item " + (i + 1);
            gc.fillText(label, labelX, labelY);
        }
        
        // 根据类型绘制数据
        if ("bar".equals(polarType) || "line".equals(polarType)) {
            drawPolarBarOrLine(gc, series, centerX, centerY, maxRadius, 
                              angleStep, maxValue, palette);
        } else if ("scatter".equals(polarType)) {
            drawPolarScatter(gc, series, centerX, centerY, maxRadius, 
                           angleStep, maxValue, palette);
        }
    }
    
    private void drawPolarBarOrLine(GraphicsContext gc, SeriesData series,
                                   double centerX, double centerY, double maxRadius,
                                   double angleStep, double maxValue, String[] palette) {
        int numCategories = series.y.length();
        
        if ("line".equals(polarType)) {
            gc.setFill(Color.web(palette[0], 0.3));
            gc.setStroke(Color.web(palette[0]));
            gc.setLineWidth(2);
            
            double[] xPoints = new double[numCategories];
            double[] yPoints = new double[numCategories];
            
            for (int i = 0; i < numCategories; i++) {
                double angle = i * angleStep - Math.PI / 2;
                double r = (series.y.get(i) / maxValue) * maxRadius;
                xPoints[i] = centerX + Math.cos(angle) * r;
                yPoints[i] = centerY + Math.sin(angle) * r;
            }
            
            gc.fillPolygon(xPoints, yPoints, numCategories);
            gc.strokePolygon(xPoints, yPoints, numCategories);
            
            // 绘制数据点
            gc.setFill(Color.web(palette[0]));
            for (int i = 0; i < numCategories; i++) {
                gc.fillOval(xPoints[i] - 4, yPoints[i] - 4, 8, 8);
            }
        } else {
            // 极坐标柱状图
            for (int i = 0; i < numCategories; i++) {
                double angle = i * angleStep - Math.PI / 2;
                double r = (series.y.get(i) / maxValue) * maxRadius;
                double endX = centerX + Math.cos(angle) * r;
                double endY = centerY + Math.sin(angle) * r;
                
                gc.setStroke(Color.web(palette[i % palette.length]));
                gc.setLineWidth(8);
                gc.strokeLine(centerX, centerY, endX, endY);
            }
        }
    }
    
    private void drawPolarScatter(GraphicsContext gc, SeriesData series,
                                  double centerX, double centerY, double maxRadius,
                                  double angleStep, double maxValue, String[] palette) {
        int numCategories = series.y.length();
        
        for (int i = 0; i < numCategories; i++) {
            double angle = i * angleStep - Math.PI / 2;
            double r = (series.y.get(i) / maxValue) * maxRadius;
            double x = centerX + Math.cos(angle) * r;
            double y = centerY + Math.sin(angle) * r;
            
            gc.setFill(Color.web(palette[i % palette.length]));
            gc.fillOval(x - 5, y - 5, 10, 10);
        }
    }
    
    @Override
    public String getChartType() {
        return "polar_" + polarType;
    }
    
    @Override
    public boolean supportsAnimation() {
        return true;
    }
    
    @Override
    public int getAnimationDuration() {
        return 1000;
    }
}
