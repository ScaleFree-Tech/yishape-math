package com.yishape.lab.math.plot.javafx.renderers;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.plot.AxisTicks;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxStyleApplier;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.TextAlignment;

import java.util.ArrayList;
import java.util.List;

/**
 * 统计图表渲染器（箱线图、小提琴图、K线图）
 * 
 * @author lteb2
 */
public class StatisticalChartRenderer implements JavaFxChartRenderer {
    
    private String chartType; // "boxplot", "violin", "candlestick"
    
    public StatisticalChartRenderer(String chartType) {
        this.chartType = chartType;
    }
    
    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) return;
        
        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);
        
        // 清空画布
        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        
        // 绘制标题
        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        
        switch (chartType) {
            case "boxplot":
                renderBoxplot(gc, data.get(0), config, themeManager);
                break;
            case "violin":
                if (data.size() > 1) {
                    renderMultiViolin(gc, data, config, themeManager);
                } else {
                    renderViolin(gc, data.get(0), config, themeManager);
                }
                break;
            case "candlestick":
                renderCandlestick(gc, data.get(0), config, themeManager);
                break;
        }
    }
    
    private void renderBoxplot(GraphicsContext gc, SeriesData series,
                              ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null) return;

        double[] sorted = new double[series.y.length()];
        for (int i = 0; i < series.y.length(); i++) {
            sorted[i] = series.y.get(i);
        }
        java.util.Arrays.sort(sorted);

        double dataMin = sorted[0];
        double dataMax = sorted[sorted.length - 1];
        double q1 = quantileLinear(sorted, 0.25);
        double median = quantileLinear(sorted, 0.50);
        double q3 = quantileLinear(sorted, 0.75);
        double iqr = q3 - q1;

        // 1.5×IQR 规则确定须的边界（seaborn 标准）
        double whiskerLo = Math.max(dataMin, q1 - 1.5 * iqr);
        double whiskerHi = Math.min(dataMax, q3 + 1.5 * iqr);

        // 显示范围：以须端点为基准，用须的实际跨度计算 padding，避免离群点干扰
        double whiskerRange = whiskerHi - whiskerLo;
        double displayMin = whiskerLo - whiskerRange * 0.08;
        double displayMax = whiskerHi + whiskerRange * 0.08;
        // 须端点不超出显示范围（使用须跨度而非数据全距，避免离群点导致 padding 不足）
        if (whiskerLo < displayMin) displayMin = whiskerLo - whiskerRange * 0.04;
        if (whiskerHi > displayMax) displayMax = whiskerHi + whiskerRange * 0.04;

        final double dMin = displayMin, dMax = displayMax;
        final double cHeight = config.height - config.paddingTop - config.paddingBottom;
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double boxWidth = Math.min(90, chartWidth / 3.5);
        double centerX = config.paddingLeft + chartWidth / 2;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;

        // 数据 → 像素映射（Y 轴向上为正，但像素 Y 向下为正，所以用减法）
        java.util.function.DoubleUnaryOperator dataToY = v -> {
            double u = (v - dMin) / (dMax - dMin);
            return config.height - config.paddingBottom - u * cHeight;
        };

        double boxTopY    = dataToY.applyAsDouble(q3);
        double boxBotY    = dataToY.applyAsDouble(q1);
        double medianY    = dataToY.applyAsDouble(median);
        double whiskerTopY = dataToY.applyAsDouble(whiskerHi);
        double whiskerBotY = dataToY.applyAsDouble(whiskerLo);

        String[] palette = themeManager.getColorPalette();
        String primaryHex = palette.length > 0 ? palette[0] : "#5470c6";
        Color boxStroke = Color.web(primaryHex);
        Color boxFill = Color.web(primaryHex, 0.28);

        // 绘制箱体
        gc.setFill(boxFill);
        gc.setStroke(boxStroke);
        gc.setLineWidth(1.8);
        gc.fillRect(centerX - boxWidth / 2, boxTopY, boxWidth, boxBotY - boxTopY);
        gc.strokeRect(centerX - boxWidth / 2, boxTopY, boxWidth, boxBotY - boxTopY);

        // 中位数线（比箱体略粗）
        gc.setLineWidth(2.5);
        gc.strokeLine(centerX - boxWidth / 2, medianY, centerX + boxWidth / 2, medianY);

        // 上下须线
        gc.setLineWidth(1.4);
        gc.strokeLine(centerX, boxTopY, centerX, whiskerTopY);
        gc.strokeLine(centerX, boxBotY, centerX, whiskerBotY);

        // 须端帽线（seaborn 风格）
        gc.strokeLine(centerX - boxWidth * 0.32, whiskerTopY, centerX + boxWidth * 0.32, whiskerTopY);
        gc.strokeLine(centerX - boxWidth * 0.32, whiskerBotY, centerX + boxWidth * 0.32, whiskerBotY);

        // 单独数据点（落在 1.5×IQR 之外的异常值，用圆点标记）
        double capLoY = dataToY.applyAsDouble(whiskerLo);
        double capHiY = dataToY.applyAsDouble(whiskerHi);
        gc.setFill(boxStroke);
        gc.setStroke(boxStroke);
        gc.setLineWidth(1);
        for (double v : sorted) {
            if (v < whiskerLo || v > whiskerHi) {
                double vy = dataToY.applyAsDouble(v);
                // 防止与须线重叠：只在 cap 范围之外画
                if (vy < capHiY - 1 || vy > capLoY + 1) {
                    gc.fillOval(centerX - 3.5, vy - 3.5, 7, 7);
                    gc.strokeOval(centerX - 3.5, vy - 3.5, 7, 7);
                }
            }
        }

        // Y 轴（须端点落在刻度范围内）
        drawYAxisOnly(gc, config, new double[]{displayMin, displayMax}, themeManager);
    }

    /**
     * 只绘制Y轴（用于箱线图、小提琴图等单变量图表）
     */
    private void drawYAxisOnly(GraphicsContext gc, ChartConfig config,
                                double[] yRange, JavaFxThemeManager themeManager) {
        Color axisColor = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("axisColor"), Color.DARKGRAY);
        Color gridColor = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("gridColor"), Color.LIGHTGRAY);

        double chartHeight = config.height - config.paddingTop - config.paddingBottom;

        // 绘制Y轴网格线
        gc.setStroke(gridColor);
        gc.setLineWidth(1);
        int yTicks = 5;
        for (int i = 0; i <= yTicks; i++) {
            double y = config.height - config.paddingBottom - chartHeight * i / yTicks;
            gc.strokeLine(config.paddingLeft, y, config.width - config.paddingRight, y);
        }

        // 绘制轴线
        gc.setStroke(axisColor);
        gc.setLineWidth(2);

        // X轴线（底部）
        gc.strokeLine(config.paddingLeft, config.height - config.paddingBottom,
                     config.width - config.paddingRight, config.height - config.paddingBottom);
        // Y轴线（左侧）
        gc.strokeLine(config.paddingLeft, config.paddingTop,
                     config.paddingLeft, config.height - config.paddingBottom);

        // 绘制Y轴标签
        gc.setFill(themeManager.getTextColor());
        gc.setFont(themeManager.getLabelFont());
        gc.setTextAlign(TextAlignment.RIGHT);

        for (int i = 0; i <= yTicks; i++) {
            double value = yRange[0] + (yRange[1] - yRange[0]) * i / yTicks;
            double y = config.height - config.paddingBottom - chartHeight * i / yTicks;
            gc.fillText(JavaFxChartUtils.formatNumber(value), config.paddingLeft - 10, y + 5);
        }
    }

    private void renderViolin(GraphicsContext gc, SeriesData series,
                             ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null || series.y.length() == 0) return;

        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double centerX = config.paddingLeft + chartWidth / 2;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;

        // 计算统计数据
        double[] sorted = new double[series.y.length()];
        for (int i = 0; i < series.y.length(); i++) {
            sorted[i] = series.y.get(i);
        }
        java.util.Arrays.sort(sorted);

        double min = sorted[0];
        double max = sorted[sorted.length - 1];
        double q1 = quantileLinear(sorted, 0.25);
        double median = quantileLinear(sorted, 0.50);
        double q3 = quantileLinear(sorted, 0.75);
        double iqr = q3 - q1;

        String[] palette = themeManager.getColorPalette();
        // seaborn风格：更柔和的颜色
        Color violinColor = Color.web(palette[0], 0.6);
        Color strokeColor = Color.web(palette[0], 0.8);

        // 计算KDE - 使用更多点使曲线更平滑
        int numPoints = 500;
        // 使用Scott规则计算带宽: h = 1.06 * σ * n^(-1/5)
        double std = calculateStd(sorted);
        double bandwidth = 1.06 * Math.min(std, iqr / 1.34) * Math.pow(series.y.length(), -0.2);
        if (bandwidth <= 0 || Double.isNaN(bandwidth)) bandwidth = (max - min) / 30;

        // 扩展KDE计算范围，超出数据范围3倍带宽，使边缘密度能降到0
        double kdeMin = min - 3 * bandwidth;
        double kdeMax = max + 3 * bandwidth;
        double kdeRange = kdeMax - kdeMin;

        double[] kdeValues = new double[numPoints];
        double maxKde = 0;

        // 计算KDE值 - 在扩展范围内
        for (int i = 0; i < numPoints; i++) {
            double y = kdeMin + kdeRange * i / (numPoints - 1);
            double kde = 0;
            for (int j = 0; j < sorted.length; j++) {
                double u = (y - sorted[j]) / bandwidth;
                kde += Math.exp(-0.5 * u * u);
            }
            kde /= (sorted.length * bandwidth * Math.sqrt(2 * Math.PI));
            kdeValues[i] = kde;
            maxKde = Math.max(maxKde, kde);
        }

        // 小提琴最大宽度 - 更窄更优雅
        double maxViolinWidth = Math.min(50, chartWidth / 8);

        // 预计算所有点的坐标
        double[] rightX = new double[numPoints];
        double[] py = new double[numPoints];

        for (int i = 0; i < numPoints; i++) {
            double y = kdeMin + kdeRange * i / (numPoints - 1);
            // 使用扩展后的范围来计算Y坐标
            py[i] = config.height - config.paddingBottom - ((y - kdeMin) / kdeRange) * chartHeight;
            double width = (kdeValues[i] / maxKde) * maxViolinWidth;
            rightX[i] = centerX + width;
        }

        // 绘制小提琴形状 - 使用平滑路径
        gc.setFill(violinColor);
        gc.beginPath();

        // 右侧曲线（从下往上）
        gc.moveTo(centerX, py[0]);
        for (int i = 1; i < numPoints; i++) {
            gc.lineTo(rightX[i], py[i]);
        }

        // 顶部弧（回到中心）
        gc.lineTo(centerX, py[numPoints - 1]);

        // 左侧曲线（从上往下）
        for (int i = numPoints - 2; i >= 0; i--) {
            double width = (kdeValues[i] / maxKde) * maxViolinWidth;
            gc.lineTo(centerX - width, py[i]);
        }

        // 底部弧（回到起点）
        gc.lineTo(centerX, py[0]);
        gc.closePath();

        // 填充和描边
        gc.fill();
        gc.setStroke(strokeColor);
        gc.setLineWidth(1.5);
        gc.stroke();

        // 绘制内部箱线图 - seaborn风格
        double boxWidth = maxViolinWidth * 0.15;
        // 使用扩展后的范围计算Y坐标
        double q1Y = config.height - config.paddingBottom - ((q1 - kdeMin) / kdeRange) * chartHeight;
        double medianY = config.height - config.paddingBottom - ((median - kdeMin) / kdeRange) * chartHeight;
        double q3Y = config.height - config.paddingBottom - ((q3 - kdeMin) / kdeRange) * chartHeight;

        Color innerStroke = themeManager.getTextColor();
        Color whiskerStroke = violinWhiskerColor(themeManager);

        // 绘制箱体（Q1到Q3）- 半透明
        gc.setFill(violinInnerBoxFill(themeManager));
        gc.fillRect(centerX - boxWidth / 2, q3Y, boxWidth, q1Y - q3Y);
        gc.setStroke(innerStroke);
        gc.setLineWidth(1.2);
        gc.strokeRect(centerX - boxWidth / 2, q3Y, boxWidth, q1Y - q3Y);

        // 绘制中位数线
        gc.setStroke(innerStroke);
        gc.setLineWidth(2);
        gc.strokeLine(centerX - boxWidth / 2 - 2, medianY, centerX + boxWidth / 2 + 2, medianY);

        // 绘制须线
        double whiskerWidth = boxWidth * 0.6;
        gc.setStroke(whiskerStroke);
        gc.setLineWidth(1);

        // 计算须的位置（使用1.5*IQR规则）
        double lowerWhisker = Math.max(min, q1 - 1.5 * iqr);
        double upperWhisker = Math.min(max, q3 + 1.5 * iqr);

        double lowerY = config.height - config.paddingBottom - ((lowerWhisker - kdeMin) / kdeRange) * chartHeight;
        double upperY = config.height - config.paddingBottom - ((upperWhisker - kdeMin) / kdeRange) * chartHeight;

        // 上须
        gc.strokeLine(centerX, q3Y, centerX, upperY);
        gc.strokeLine(centerX - whiskerWidth / 2, upperY, centerX + whiskerWidth / 2, upperY);

        // 下须
        gc.strokeLine(centerX, q1Y, centerX, lowerY);
        gc.strokeLine(centerX - whiskerWidth / 2, lowerY, centerX + whiskerWidth / 2, lowerY);

        // 绘制Y轴坐标轴（小提琴图只有Y轴有意义）
        drawYAxisOnly(gc, config, new double[]{kdeMin, kdeMax}, themeManager);
    }

    /**
     * 分位数线性插值（Type 7：R 和 Excel 默认使用的 "inverse CDF" 法）。
     * 当 n 较小时结果与简单索引法不同，能更好处理小样本。
     */
    private static double quantileLinear(double[] sorted, double p) {
        if (sorted == null || sorted.length == 0) {
            return Double.NaN;
        }
        if (sorted.length == 1) {
            return sorted[0];
        }
        double n = sorted.length;
        double pos = p * (n - 1);
        int i = (int) Math.floor(pos);
        int j = (int) Math.ceil(pos);
        if (i == j) {
            return sorted[i];
        }
        double weight = pos - i;
        return sorted[i] * (1 - weight) + sorted[j] * weight;
    }

    private double calculateStd(double[] data) {
        double mean = 0;
        for (double v : data) mean += v;
        mean /= data.length;

        double variance = 0;
        for (double v : data) {
            double diff = v - mean;
            variance += diff * diff;
        }
        variance /= data.length;
        return Math.sqrt(variance);
    }

    /**
     * 渲染多组小提琴图（并排显示）
     */
    private void renderMultiViolin(GraphicsContext gc, List<SeriesData> data,
                                    ChartConfig config, JavaFxThemeManager themeManager) {
        int numGroups = data.size();
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;

        // 计算所有数据的整体范围（用于统一Y轴）
        double globalMin = Double.MAX_VALUE;
        double globalMax = Double.MIN_VALUE;
        for (SeriesData series : data) {
            if (series.y == null || series.y.length() == 0) continue;
            for (int i = 0; i < series.y.length(); i++) {
                double v = series.y.get(i);
                globalMin = Math.min(globalMin, v);
                globalMax = Math.max(globalMax, v);
            }
        }

        // 计算全局带宽（基于所有数据）
        double globalStd = 0;
        double globalMean = 0;
        int totalCount = 0;
        for (SeriesData series : data) {
            if (series.y == null) continue;
            for (int i = 0; i < series.y.length(); i++) {
                globalMean += series.y.get(i);
                totalCount++;
            }
        }
        globalMean /= totalCount;

        for (SeriesData series : data) {
            if (series.y == null) continue;
            for (int i = 0; i < series.y.length(); i++) {
                double diff = series.y.get(i) - globalMean;
                globalStd += diff * diff;
            }
        }
        globalStd = Math.sqrt(globalStd / totalCount);

        double bandwidth = 1.06 * globalStd * Math.pow(totalCount, -0.2);
        if (bandwidth <= 0 || Double.isNaN(bandwidth)) bandwidth = (globalMax - globalMin) / 30;

        // 扩展范围使边缘密度降到0
        double kdeMin = globalMin - 3 * bandwidth;
        double kdeMax = globalMax + 3 * bandwidth;
        double kdeRange = kdeMax - kdeMin;

        // 计算每个小提琴的位置
        double groupWidth = chartWidth / numGroups;
        double violinMaxWidth = Math.min(groupWidth * 0.7, 60);

        String[] palette = themeManager.getColorPalette();

        // 绘制每个小提琴
        for (int g = 0; g < numGroups; g++) {
            SeriesData series = data.get(g);
            if (series.y == null || series.y.length() == 0) continue;

            double centerX = config.paddingLeft + g * groupWidth + groupWidth / 2;

            // 计算该组的KDE
            int numPoints = 300;
            double[] kdeValues = new double[numPoints];
            double maxKde = 0;

            for (int i = 0; i < numPoints; i++) {
                double y = kdeMin + kdeRange * i / (numPoints - 1);
                double kde = 0;
                for (int j = 0; j < series.y.length(); j++) {
                    double u = (y - series.y.get(j)) / bandwidth;
                    kde += Math.exp(-0.5 * u * u);
                }
                kde /= (series.y.length() * bandwidth * Math.sqrt(2 * Math.PI));
                kdeValues[i] = kde;
                maxKde = Math.max(maxKde, kde);
            }

            // 颜色：与图例一致——优先用序列上已赋值的 PlotStyle（分组小提琴在 JavaFxPlot 中按组着色）
            Color baseRgb = resolveMultiViolinSeriesColor(series, g, palette);
            Color violinColor = new Color(baseRgb.getRed(), baseRgb.getGreen(), baseRgb.getBlue(), 0.6);
            Color strokeColor = new Color(baseRgb.getRed(), baseRgb.getGreen(), baseRgb.getBlue(), 0.8);

            // 预计算坐标
            double[] py = new double[numPoints];
            for (int i = 0; i < numPoints; i++) {
                double y = kdeMin + kdeRange * i / (numPoints - 1);
                py[i] = config.height - config.paddingBottom - ((y - kdeMin) / kdeRange) * chartHeight;
            }

            // 绘制小提琴
            gc.setFill(violinColor);
            gc.beginPath();

            // 右侧曲线
            gc.moveTo(centerX, py[0]);
            for (int i = 1; i < numPoints; i++) {
                double width = (kdeValues[i] / maxKde) * violinMaxWidth;
                gc.lineTo(centerX + width, py[i]);
            }

            // 顶部
            gc.lineTo(centerX, py[numPoints - 1]);

            // 左侧曲线
            for (int i = numPoints - 2; i >= 0; i--) {
                double width = (kdeValues[i] / maxKde) * violinMaxWidth;
                gc.lineTo(centerX - width, py[i]);
            }

            gc.closePath();
            gc.fill();
            gc.setStroke(strokeColor);
            gc.setLineWidth(1.5);
            gc.stroke();

            // 绘制内部箱线图（seaborn风格）
            // 计算该组的统计数据
            double[] sorted = new double[series.y.length()];
            for (int i = 0; i < series.y.length(); i++) {
                sorted[i] = series.y.get(i);
            }
            java.util.Arrays.sort(sorted);

            double sMin = sorted[0];
            double sMax = sorted[sorted.length - 1];
            double sQ1 = sorted[sorted.length / 4];
            double sMedian = sorted[sorted.length / 2];
            double sQ3 = sorted[3 * sorted.length / 4];
            double sIqr = sQ3 - sQ1;

            double boxWidth = violinMaxWidth * 0.12;

            // 计算Y坐标（使用全局范围）
            double q1Y = config.height - config.paddingBottom - ((sQ1 - kdeMin) / kdeRange) * chartHeight;
            double medianY = config.height - config.paddingBottom - ((sMedian - kdeMin) / kdeRange) * chartHeight;
            double q3Y = config.height - config.paddingBottom - ((sQ3 - kdeMin) / kdeRange) * chartHeight;

            Color innerStroke = themeManager.getTextColor();
            Color whiskerStroke = violinWhiskerColor(themeManager);

            // 绘制箱体（Q1到Q3）
            gc.setFill(violinInnerBoxFill(themeManager));
            gc.fillRect(centerX - boxWidth / 2, q3Y, boxWidth, q1Y - q3Y);
            gc.setStroke(innerStroke);
            gc.setLineWidth(1);
            gc.strokeRect(centerX - boxWidth / 2, q3Y, boxWidth, q1Y - q3Y);

            // 绘制中位数线
            gc.setStroke(innerStroke);
            gc.setLineWidth(1.5);
            gc.strokeLine(centerX - boxWidth / 2 - 1, medianY, centerX + boxWidth / 2 + 1, medianY);

            // 计算须的位置（1.5*IQR规则）
            double lowerWhisker = Math.max(sMin, sQ1 - 1.5 * sIqr);
            double upperWhisker = Math.min(sMax, sQ3 + 1.5 * sIqr);

            double lowerY = config.height - config.paddingBottom - ((lowerWhisker - kdeMin) / kdeRange) * chartHeight;
            double upperY = config.height - config.paddingBottom - ((upperWhisker - kdeMin) / kdeRange) * chartHeight;

            // 绘制须线
            double whiskerWidth = boxWidth * 0.5;
            gc.setStroke(whiskerStroke);
            gc.setLineWidth(0.8);

            // 上须
            gc.strokeLine(centerX, q3Y, centerX, upperY);
            gc.strokeLine(centerX - whiskerWidth / 2, upperY, centerX + whiskerWidth / 2, upperY);

            // 下须
            gc.strokeLine(centerX, q1Y, centerX, lowerY);
            gc.strokeLine(centerX - whiskerWidth / 2, lowerY, centerX + whiskerWidth / 2, lowerY);

            // 绘制组标签
            if (series.name != null && !series.name.isEmpty()) {
                gc.setFill(themeManager.getTextColor());
                gc.setFont(themeManager.getLabelFont());
                gc.setTextAlign(javafx.scene.text.TextAlignment.CENTER);
                gc.fillText(series.name, centerX, config.height - config.paddingBottom + 20);
            }
        }

        // 绘制统一的Y轴（分组小提琴图只有Y轴有意义，X轴显示组名）
        drawYAxisOnly(gc, config, new double[]{kdeMin, kdeMax}, themeManager);

        // 绘制图例
        JavaFxChartUtils.drawLegend(gc, data, config, themeManager);
    }

    /**
     * 多组小提琴每条序列的颜色：与 {@link JavaFxChartUtils#drawLegend} 一致（优先 {@code series.style.getColor()}）。
     */
    private static Color resolveMultiViolinSeriesColor(SeriesData series, int listIndex, String[] palette) {
        if (series.style != null && series.style.getColor() != null && !series.style.getColor().isEmpty()) {
            Color c = JavaFxStyleApplier.parseColor(series.style.getColor());
            if (c != null) {
                return new Color(c.getRed(), c.getGreen(), c.getBlue(), 1.0);
            }
        }
        Object gi = series.extraData != null ? series.extraData.get("groupIndex") : null;
        int p = gi instanceof Integer ? (Integer) gi : listIndex;
        int plen = palette != null && palette.length > 0 ? palette.length : 1;
        String hex = palette != null && palette.length > 0 ? palette[p % plen] : "#5470c6";
        Color c = Color.web(hex);
        return new Color(c.getRed(), c.getGreen(), c.getBlue(), 1.0);
    }

    private static double clamp01(double x) {
        return Math.min(1.0, Math.max(0.0, x));
    }

    private static Color violinInnerBoxFill(JavaFxThemeManager tm) {
        Color bg = tm.getBackgroundColor();
        double lum = 0.2126 * bg.getRed() + 0.7152 * bg.getGreen() + 0.0722 * bg.getBlue();
        if (lum > 0.55) {
            return new Color(1, 1, 1, 0.93);
        }
        return new Color(
            clamp01(bg.getRed() + 0.14),
            clamp01(bg.getGreen() + 0.14),
            clamp01(bg.getBlue() + 0.16),
            0.92);
    }

    private static Color violinWhiskerColor(JavaFxThemeManager tm) {
        Color t = tm.getTextColor();
        return new Color(t.getRed(), t.getGreen(), t.getBlue(), 0.74);
    }

    /** K 线实体填充：略透明；影线/描边用同色系略深，贴近其它图表的调色板观感。 */
    private static Color candleBodyFill(Color base) {
        return new Color(base.getRed(), base.getGreen(), base.getBlue(), 0.74);
    }

    private static Color candleBodyStroke(Color base) {
        double f = 0.82;
        return new Color(
            clamp01(base.getRed() * f),
            clamp01(base.getGreen() * f),
            clamp01(base.getBlue() * f),
            1.0);
    }

    /**
     * K 线涨/跌色：与 A 股常见习惯一致（涨红、跌绿），使用低饱和柔和色，避免荧光大红大绿。
     * 不直接使用主题高饱和 palette，以免刺眼影线色与个股语义冲突。
     */
    private static Color[] candleRiseFallColors() {
        return new Color[] {
            Color.web("#c45b5b"), // 涨：砖红
            Color.web("#4d8f72")  // 跌：青绿
        };
    }

    /** 线性轴「整步长」边界与步长，便于价格刻度（返回 lo, hi, step）。 */
    private static double[] niceLinearAxis(double dataMin, double dataMax, int targetDivisions) {
        if (!Double.isFinite(dataMin) || !Double.isFinite(dataMax)) {
            return new double[] {0.0, 1.0, 0.2};
        }
        double lo = dataMin;
        double hi = dataMax;
        if (hi < lo) {
            double t = lo;
            lo = hi;
            hi = t;
        }
        if (hi - lo < 1e-15) {
            double pad = Math.abs(hi) > 1e-6 ? Math.abs(hi) * 0.08 : 0.5;
            return new double[] {hi - pad, hi + pad, pad > 0 ? pad / 2 : 0.25};
        }
        double span = hi - lo;
        double rawStep = span / Math.max(2, targetDivisions - 1);
        double exp = Math.floor(Math.log10(rawStep));
        double pow = Math.pow(10.0, exp);
        double err = rawStep / pow;
        double nf;
        if (err <= 1.0) {
            nf = 1.0;
        } else if (err <= 2.0) {
            nf = 2.0;
        } else if (err <= 5.0) {
            nf = 5.0;
        } else {
            nf = 10.0;
        }
        double step = nf * pow;
        double axisLo = Math.floor(lo / step) * step;
        double axisHi = Math.ceil(hi / step) * step;
        if (axisHi <= axisLo) {
            axisHi = axisLo + step;
        }
        return new double[] {axisLo, axisHi, step};
    }

    @SuppressWarnings("unchecked")
    private void renderCandlestick(GraphicsContext gc, JavaFxChartRenderer.SeriesData series,
                                  JavaFxChartRenderer.ChartConfig config, JavaFxThemeManager themeManager) {
        IMatrix<Double> matrix = (IMatrix<Double>) series.extraData.get("matrixData");
        if (matrix == null || matrix.getColNum() < 4) return;
        
        int numCandles = matrix.getRowNum();
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double candleWidth = chartWidth / numCandles * 0.8;
        double candleSpacing = chartWidth / numCandles;
        
        double minPrice = Double.MAX_VALUE;
        double maxPrice = Double.MIN_VALUE;
        
        for (int i = 0; i < numCandles; i++) {
            double low = matrix.get(i, 2);
            double high = matrix.get(i, 3);
            minPrice = Math.min(minPrice, low);
            maxPrice = Math.max(maxPrice, high);
        }
        
        double[] axis = niceLinearAxis(minPrice, maxPrice, 6);
        double axisLo = axis[0];
        double axisHi = axis[1];
        double tickStep = axis[2];
        double priceRange = axisHi - axisLo;
        if (priceRange <= 1e-15) {
            priceRange = 1.0;
        }
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;

        Color[] upDown = candleRiseFallColors();

        for (int i = 0; i < numCandles; i++) {
            double open = matrix.get(i, 0);
            double close = matrix.get(i, 1);
            double low = matrix.get(i, 2);
            double high = matrix.get(i, 3);
            
            boolean isRising = close >= open;
            Color base = isRising ? upDown[0] : upDown[1];
            Color bodyColor = candleBodyFill(base);
            Color strokeColor = candleBodyStroke(base);
            
            double x = config.paddingLeft + i * candleSpacing + candleSpacing * 0.1;
            
            double openY = config.height - config.paddingBottom - ((open - axisLo) / priceRange) * chartHeight;
            double closeY = config.height - config.paddingBottom - ((close - axisLo) / priceRange) * chartHeight;
            double lowY = config.height - config.paddingBottom - ((low - axisLo) / priceRange) * chartHeight;
            double highY = config.height - config.paddingBottom - ((high - axisLo) / priceRange) * chartHeight;
            
            gc.setStroke(strokeColor);
            gc.setLineWidth(1.15);
            gc.strokeLine(x + candleWidth / 2, lowY, x + candleWidth / 2, highY);
            
            gc.setFill(bodyColor);
            double bodyTop = Math.min(openY, closeY);
            double bodyHeight = Math.abs(closeY - openY);
            if (bodyHeight < 3) {
                bodyHeight = 3;
            }
            gc.fillRect(x, bodyTop, candleWidth, bodyHeight);
            gc.setStroke(strokeColor);
            gc.setLineWidth(1);
            gc.strokeRect(x, bodyTop, candleWidth, bodyHeight);
        }

        AxisTicks prevXTicks = config.xAxisTicks;
        AxisTicks prevYTicks = config.yAxisTicks;
        boolean prevCenter = config.centerCategoryXLabels;
        try {
            double[] tickX = new double[numCandles];
            for (int i = 0; i < numCandles; i++) {
                tickX[i] = i;
            }
            List<String> labs = new ArrayList<>();
            if (series.labels != null) {
                labs.addAll(series.labels);
            }
            while (labs.size() < numCandles) {
                labs.add(String.valueOf(labs.size()));
            }
            if (labs.size() > numCandles) {
                labs = new ArrayList<>(labs.subList(0, numCandles));
            }
            int steps = (int) Math.round((axisHi - axisLo) / tickStep);
            steps = Math.max(1, Math.min(15, steps));
            double[] yVals = new double[steps + 1];
            for (int k = 0; k <= steps; k++) {
                yVals[k] = axisLo + k * tickStep;
            }
            config.xAxisTicks = new AxisTicks(Linalg.vector(tickX), labs);
            config.yAxisTicks = new AxisTicks(Linalg.vector(yVals), new ArrayList<>());
            config.centerCategoryXLabels = true;
            double xmax = Math.max(0.0, numCandles - 1);
            JavaFxChartUtils.drawAxesCartesian(gc, config, 0, xmax, axisLo, axisHi, null, null, themeManager);
        } finally {
            config.xAxisTicks = prevXTicks;
            config.yAxisTicks = prevYTicks;
            config.centerCategoryXLabels = prevCenter;
        }
    }

    @Override
    public String getChartType() {
        return chartType;
    }
    
    @Override
    public boolean supportsAnimation() {
        return false;
    }
    
    @Override
    public int getAnimationDuration() {
        return 0;
    }
}
