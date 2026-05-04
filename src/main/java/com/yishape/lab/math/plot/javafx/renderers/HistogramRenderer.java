package com.yishape.lab.math.plot.javafx.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import com.yishape.lab.math.linalg.IVector;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

import java.util.List;

/**
 * 直方图渲染器
 * 
 * @author lteb2
 */
public class HistogramRenderer implements JavaFxChartRenderer {
    
    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) return;
        
        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);
        SeriesData series = data.get(0);
        
        if (series.y == null) return;
        
        // 计算直方图参数
        int bins = Math.min(20, (int) Math.sqrt(series.y.length()));
        Integer binsOverride = (Integer) series.extraData.get("bins");
        if (binsOverride != null && binsOverride > 0) {
            bins = Math.min(binsOverride, 500);
        }
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (int i = 0; i < series.y.length(); i++) {
            double val = series.y.get(i);
            min = Math.min(min, val);
            max = Math.max(max, val);
        }

        if (config.axisLockX != null && config.axisLockX.length == 2) {
            min = config.axisLockX[0];
            max = config.axisLockX[1];
            if (max <= min) {
                max = min + 1;
            }
        }
        
        double binWidth = (max - min) / bins;
        int[] counts = new int[bins];
        
        for (int i = 0; i < series.y.length(); i++) {
            double value = series.y.get(i);
            int binIndex = (int) Math.min((value - min) / binWidth, bins - 1);
            counts[binIndex]++;
        }
        
        double maxCount = 0;
        for (int count : counts) {
            maxCount = Math.max(maxCount, count);
        }
        
        // 清空画布
        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        
        // 绘制标题
        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        
        // 绘制坐标轴（使用 drawAxesCartesian 获得更好的网格和标签处理）
        JavaFxChartUtils.drawAxesCartesian(gc, config, min, max, 0, maxCount, null, null, themeManager);
        
        // 计算柱状图参数
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;
        double barWidth = chartWidth / bins * 0.95;
        
        String[] palette = themeManager.getColorPalette();
        Color barColor = Color.web(palette[0]);
        
        gc.setFill(barColor);
        gc.setStroke(themeManager.getTextColor());
        gc.setLineWidth(1);
        
        // 绘制直方图柱
        for (int i = 0; i < bins; i++) {
            double barHeight = (counts[i] / maxCount) * chartHeight * 0.9;
            double x = config.paddingLeft + i * (chartWidth / bins);
            double y = config.height - config.paddingBottom - barHeight;
            
            gc.fillRect(x, y, barWidth, barHeight);
            gc.strokeRect(x, y, barWidth, barHeight);

            double binCenter = min + (i + 0.5) * binWidth;
            double hitX = x + barWidth / 2;
            double hitY = y + barHeight / 2;
            JavaFxChartUtils.registerHit(config, hitX, hitY, binCenter, counts[i], series.name, 0, i);
        }
        
        // 绘制KDE拟合曲线
        Boolean fittingLine = (Boolean) series.extraData.get("fittingLine");
        if (fittingLine != null && fittingLine) {
            drawKDECurve(gc, series.y, min, max, config, chartHeight, maxCount, themeManager);
        }
    }
    
    /**
     * 绘制KDE（核密度估计）曲线及半透明填充
     */
    private void drawKDECurve(GraphicsContext gc, IVector<Double> data, double min, double max,
                               ChartConfig config, double chartHeight, double maxCount,
                               JavaFxThemeManager themeManager) {
        int numPoints = 200;
        // 使用Silverman规则计算带宽
        double std = calculateStd(data);
        double iqr = calculateIQR(data);
        double bandwidth = 0.9 * Math.min(std, iqr / 1.34) * Math.pow(data.length(), -0.2);
        if (bandwidth <= 0 || Double.isNaN(bandwidth)) {
            bandwidth = (max - min) / 20;
        }

        // 计算所有点的KDE值
        double[] kdeValues = new double[numPoints];
        double maxKde = 0;

        for (int i = 0; i < numPoints; i++) {
            double x = min + (max - min) * i / (numPoints - 1);
            double kde = 0;

            // 高斯核函数
            for (int j = 0; j < data.length(); j++) {
                double u = (x - data.get(j)) / bandwidth;
                kde += Math.exp(-0.5 * u * u);
            }
            kde /= (data.length() * bandwidth * Math.sqrt(2 * Math.PI));

            kdeValues[i] = kde;
            maxKde = Math.max(maxKde, kde);
        }

        // 计算缩放因子，使KDE峰值达到直方图最大高度的60%
        // 直方图最高柱子高度 = chartHeight * 0.9
        // 目标KDE峰值高度 = chartHeight * 0.9 * 0.6 = chartHeight * 0.54
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double targetPixelHeight = chartHeight * 0.54;
        double scaleFactor = targetPixelHeight / maxKde;

        // 计算底部Y坐标（X轴位置）
        double baselineY = config.height - config.paddingBottom;

        // 存储所有点的坐标
        double[] pxPoints = new double[numPoints];
        double[] pyPoints = new double[numPoints];

        for (int i = 0; i < numPoints; i++) {
            double x = min + (max - min) * i / (numPoints - 1);
            pxPoints[i] = config.paddingLeft + (x - min) / (max - min) * chartWidth;
            double scaledKde = kdeValues[i] * scaleFactor;
            pyPoints[i] = baselineY - scaledKde;
        }

        String[] pal = themeManager.getColorPalette();
        String accentHex = pal != null && pal.length > 1 ? pal[1] : (pal != null && pal.length > 0 ? pal[0] : "#dd8452");
        Color accent = Color.web(accentHex);
        Color fill = new Color(accent.getRed(), accent.getGreen(), accent.getBlue(), 0.24);

        gc.setFill(fill);
        gc.beginPath();
        gc.moveTo(pxPoints[0], baselineY);
        for (int i = 0; i < numPoints; i++) {
            gc.lineTo(pxPoints[i], pyPoints[i]);
        }
        gc.lineTo(pxPoints[numPoints - 1], baselineY);
        gc.closePath();
        gc.fill();

        gc.setStroke(accent);
        gc.setLineWidth(2.2);
        gc.beginPath();
        gc.moveTo(pxPoints[0], pyPoints[0]);
        for (int i = 1; i < numPoints; i++) {
            gc.lineTo(pxPoints[i], pyPoints[i]);
        }
        gc.stroke();
    }

    private double calculateStd(IVector<Double> data) {
        double mean = 0;
        for (int i = 0; i < data.length(); i++) {
            mean += data.get(i);
        }
        mean /= data.length();

        double variance = 0;
        for (int i = 0; i < data.length(); i++) {
            double diff = data.get(i) - mean;
            variance += diff * diff;
        }
        variance /= data.length();
        return Math.sqrt(variance);
    }

    private double calculateIQR(IVector<Double> data) {
        double[] sorted = new double[data.length()];
        for (int i = 0; i < data.length(); i++) {
            sorted[i] = data.get(i);
        }
        java.util.Arrays.sort(sorted);

        int q1Index = sorted.length / 4;
        int q3Index = 3 * sorted.length / 4;
        return sorted[q3Index] - sorted[q1Index];
    }
    
    @Override
    public String getChartType() {
        return "histogram";
    }
    
    @Override
    public boolean supportsAnimation() {
        return true;
    }
    
    @Override
    public int getAnimationDuration() {
        return 800;
    }
}
