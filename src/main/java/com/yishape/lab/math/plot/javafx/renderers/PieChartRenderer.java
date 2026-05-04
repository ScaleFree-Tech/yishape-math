package com.yishape.lab.math.plot.javafx.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.shape.ArcType;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

import java.util.List;

/**
 * 饼图渲染器
 * 
 * @author lteb2
 */
public class PieChartRenderer implements JavaFxChartRenderer {
    
    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) return;
        
        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);
        SeriesData series = data.get(0);
        
        if (series.y == null) return;
        
        // 计算总和
        double total = 0;
        for (int i = 0; i < series.y.length(); i++) {
            total += series.y.get(i);
        }
        
        // 清空画布
        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        
        // 绘制标题
        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        
        // 计算饼图参数
        double centerX = config.width / 2.0;
        double centerY = config.height / 2.0 + 20;
        double radius = Math.min(config.width - 200, config.height - 150) / 2.0;
        
        String[] palette = themeManager.getColorPalette();
        double startAngle = 0;
        
        // 绘制饼图扇区
        for (int i = 0; i < series.y.length(); i++) {
            double value = series.y.get(i);
            double angle = (value / total) * 360;
            
            // 绘制扇区
            gc.setFill(Color.web(palette[i % palette.length]));
            gc.fillArc(centerX - radius, centerY - radius, radius * 2, radius * 2,
                      startAngle, angle, ArcType.ROUND);
            
            // 绘制边框
            gc.setStroke(Color.WHITE);
            gc.setLineWidth(2);
            gc.strokeArc(centerX - radius, centerY - radius, radius * 2, radius * 2,
                        startAngle, angle, ArcType.ROUND);
            
            // 计算标签位置
            double midAngle = Math.toRadians(startAngle + angle / 2);
            double labelX = centerX + Math.cos(midAngle) * (radius + 30);
            double labelY = centerY + Math.sin(midAngle) * (radius + 30);
            
            // 绘制百分比标签
            gc.setFill(themeManager.getTextColor());
            gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 10));
            gc.setTextAlign(TextAlignment.CENTER);
            String percentLabel = String.format("%.1f%%", (value / total) * 100);
            gc.fillText(percentLabel, labelX, labelY);
            
            startAngle += angle;
        }
        
        // 绘制图例
        drawPieLegend(gc, series, palette, centerX, centerY, radius, themeManager);
    }
    
    private void drawPieLegend(GraphicsContext gc, SeriesData series, String[] palette,
                               double centerX, double centerY, double radius,
                               JavaFxThemeManager themeManager) {
        double legendX = centerX + radius + 60;
        double legendY = centerY - series.y.length() * 12;
        
        gc.setFont(themeManager.getLabelFont());
        
        for (int i = 0; i < series.y.length(); i++) {
            // 绘制颜色块
            gc.setFill(Color.web(palette[i % palette.length]));
            gc.fillRect(legendX, legendY + i * 25, 15, 15);
            gc.setStroke(themeManager.getTextColor());
            gc.setLineWidth(1);
            gc.strokeRect(legendX, legendY + i * 25, 15, 15);

            // 绘制标签文本 - 放在颜色块右侧，垂直居中对齐
            gc.setFill(themeManager.getTextColor());
            gc.setFont(themeManager.getLabelFont());
            gc.setTextAlign(TextAlignment.LEFT);
            gc.setTextBaseline(javafx.geometry.VPos.CENTER);
            String label = series.labels != null && i < series.labels.size()
                ? series.labels.get(i)
                : "Item " + (i + 1);
            gc.fillText(label, legendX + 25, legendY + i * 25 + 7.5);
        }
    }
    
    @Override
    public String getChartType() {
        return "pie";
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
