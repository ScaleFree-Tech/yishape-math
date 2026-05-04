package com.yishape.lab.math.plot.javafx.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.shape.ArcType;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.TextAlignment;

import java.util.List;

/**
 * 仪表盘渲染器
 * 
 * @author lteb2
 */
public class GaugeRenderer implements JavaFxChartRenderer {
    
    @SuppressWarnings("unchecked")
    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) return;
        
        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);
        SeriesData series = data.get(0);
        
        // 获取数据
        double value = ((Number) series.extraData.getOrDefault("value", 0.0)).doubleValue();
        double max = ((Number) series.extraData.getOrDefault("max", 100.0)).doubleValue();
        double min = ((Number) series.extraData.getOrDefault("min", 0.0)).doubleValue();
        
        // 清空画布
        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        
        // 绘制标题
        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        
        // 计算仪表盘参数
        double centerX = config.width / 2.0;
        double centerY = config.height / 2.0 + 50;
        double radius = Math.min(config.width - 100, config.height - 150) / 2.0;
        
        String[] palette = themeManager.getColorPalette();
        
        // 绘制背景弧
        gc.setStroke(Color.LIGHTGRAY);
        gc.setLineWidth(20);
        gc.strokeArc(centerX - radius, centerY - radius, radius * 2, radius * 2, 0, 180, ArcType.OPEN);
        
        // 绘制刻度
        gc.setFill(themeManager.getTextColor());
        gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 12));
        gc.setTextAlign(TextAlignment.CENTER);
        
        for (int i = 0; i <= 10; i++) {
            double angle = Math.PI * i / 10;
            double tickX = centerX + Math.cos(Math.PI - angle) * (radius + 30);
            double tickY = centerY - Math.sin(angle) * (radius + 30);
            double tickValue = min + (max - min) * i / 10;
            gc.fillText(String.format("%.0f", tickValue), tickX, tickY);
        }
        
        // 绘制值弧
        double percentage = (value - min) / (max - min);
        double angle = percentage * 180;
        
        gc.setStroke(Color.web(palette[0]));
        gc.strokeArc(centerX - radius, centerY - radius, radius * 2, radius * 2, 0, angle, ArcType.OPEN);
        
        // 绘制指针
        double needleAngle = Math.PI - angle * Math.PI / 180;
        double needleX = centerX + Math.cos(needleAngle) * radius;
        double needleY = centerY - Math.sin(needleAngle) * radius;
        
        gc.setStroke(Color.RED);
        gc.setLineWidth(3);
        gc.strokeLine(centerX, centerY, needleX, needleY);
        
        // 绘制中心圆点
        gc.setFill(Color.DARKGRAY);
        gc.fillOval(centerX - 8, centerY - 8, 16, 16);
        
        // 绘制中心值
        gc.setFill(Color.web(palette[0]));
        gc.setFont(Font.font(themeManager.getTitleFont().getFamily(), FontWeight.BOLD, 36));
        gc.setTextAlign(TextAlignment.CENTER);
        gc.fillText(String.format("%.1f", value), centerX, centerY - 30);
        
        // 绘制单位标签
        gc.setFill(themeManager.getTextColor());
        gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 14));
        gc.fillText(series.name, centerX, centerY + 20);
    }
    
    @Override
    public String getChartType() {
        return "gauge";
    }
    
    @Override
    public boolean supportsAnimation() {
        return true;
    }
    
    @Override
    public int getAnimationDuration() {
        return 1500;
    }
}
