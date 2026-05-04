package com.yishape.lab.math.plot.javafx.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxStyleApplier;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.TextAlignment;

import java.util.List;

/**
 * 柱状图渲染器
 * 
 * @author lteb2
 */
public class BarChartRenderer implements JavaFxChartRenderer {
    
    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) return;
        
        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);

        boolean horizontal = Boolean.TRUE.equals(data.get(0).extraData.get("horizontal"));
        boolean stacked = Boolean.TRUE.equals(data.get(0).extraData.get("stacked"));

        if (stacked) {
            renderStackedBars(gc, data, config, themeManager, horizontal);
            return;
        }
        if (horizontal) {
            renderHorizontalBars(gc, data, config, themeManager);
            return;
        }
        
        // 判断是否为分组柱状图
        boolean isGrouped = data.size() > 1;

        // 与 Seaborn/matplotlib 一致：Y 轴包含 0，负值自基线向下延伸
        double[] yRange = computeVerticalBarYRange(data);
        
        // 清空画布
        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);

        // 绘制标题
        JavaFxChartUtils.drawTitle(gc, config, themeManager);

        // 计算柱状图参数
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;

        String[] palette = themeManager.getColorPalette();

        if (isGrouped) {
            // 分组柱状图渲染 - 使用自定义坐标轴（只显示Y轴）
            drawBarChartAxes(gc, config, yRange, themeManager);
            renderGroupedBars(gc, data, config, themeManager, yRange, chartWidth, chartHeight, palette);
        } else {
            // 普通柱状图渲染 - 使用自定义坐标轴（只显示Y轴和分类标签）
            drawBarChartAxes(gc, config, yRange, themeManager);
            renderSimpleBars(gc, data.get(0), config, themeManager, yRange, chartWidth, chartHeight, palette);
        }
    }

    /**
     * 垂直柱图 Y 轴范围：必含 0；全非负时从 0 起算，全非正时到 0 止，有正有负时在两侧留边距。
     */
    private static double[] computeVerticalBarYRange(List<SeriesData> data) {
        double minV = Double.POSITIVE_INFINITY;
        double maxV = Double.NEGATIVE_INFINITY;
        for (SeriesData s : data) {
            if (s.y == null) continue;
            for (int i = 0; i < s.y.length(); i++) {
                double v = s.y.get(i);
                if (!Double.isFinite(v)) continue;
                minV = Math.min(minV, v);
                maxV = Math.max(maxV, v);
            }
        }
        if (minV == Double.POSITIVE_INFINITY) {
            return new double[] { 0, 1 };
        }
        double pad;
        if (minV >= 0) {
            double hi = maxV;
            double span = Math.max(hi, 1e-15);
            pad = span * 0.08;
            return new double[] { 0, hi + pad };
        }
        if (maxV <= 0) {
            double lo = minV;
            double span = Math.max(-lo, 1e-15);
            pad = span * 0.08;
            return new double[] { lo - pad, 0 };
        }
        double span = maxV - minV;
        pad = span * 0.08;
        return new double[] { minV - pad, maxV + pad };
    }

    private static double yPixelForValue(double v, double[] yRange, ChartConfig config, double chartHeight) {
        double span = yRange[1] - yRange[0];
        if (span <= 1e-30) span = 1;
        double t = (v - yRange[0]) / span;
        return config.height - config.paddingBottom - t * chartHeight;
    }

    /** 横向柱图：数值映射到 X 像素（与垂直柱共用 {@code computeVerticalBarYRange} 的区间逻辑） */
    private static double xPixelForBarhValue(double v, double[] valueRange, ChartConfig config, double plotW) {
        double span = valueRange[1] - valueRange[0];
        if (span <= 1e-30) span = 1;
        double t = (v - valueRange[0]) / span;
        return config.paddingLeft + t * plotW;
    }

    /**
     * 绘制柱状图专用的坐标轴（只显示Y轴和分类X轴）
     */
    private void drawBarChartAxes(GraphicsContext gc, ChartConfig config,
                                    double[] yRange, JavaFxThemeManager themeManager) {
        Color axisColor = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("axisColor"), Color.DARKGRAY);
        Color gridColor = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("gridColor"), Color.LIGHTGRAY);

        // 绘制Y轴网格线
        gc.setStroke(gridColor);
        gc.setLineWidth(1);
        int yTicks = 5;
        for (int i = 0; i <= yTicks; i++) {
            double y = config.height - config.paddingBottom -
                      (config.height - config.paddingTop - config.paddingBottom) * i / yTicks;
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
            double y = config.height - config.paddingBottom -
                      (config.height - config.paddingTop - config.paddingBottom) * i / yTicks;
            gc.fillText(JavaFxChartUtils.formatNumber(value), config.paddingLeft - 10, y + 5);
        }
    }
    
    private void renderSimpleBars(GraphicsContext gc, SeriesData series, ChartConfig config,
                                   JavaFxThemeManager themeManager, double[] yRange,
                                   double chartWidth, double chartHeight, String[] palette) {
        if (series.y == null) return;
        
        int numBars = series.y.length();
        double barWidth = chartWidth / numBars * 0.8;
        double barSpacing = chartWidth / numBars * 0.2;
        
        double y0 = yPixelForValue(0, yRange, config, chartHeight);

        // 绘制柱状图
        for (int i = 0; i < numBars; i++) {
            double value = series.y.get(i);
            double yVal = yPixelForValue(value, yRange, config, chartHeight);
            double top = Math.min(yVal, y0);
            double barHeight = Math.abs(yVal - y0);
            if (value != 0 && barHeight < 1) {
                barHeight = 1;
            }
            double x = config.paddingLeft + i * (barWidth + barSpacing) + barSpacing / 2;
            double y = top;

            // 绘制柱子
            Color barColor = Color.web(palette[i % palette.length]);
            gc.setFill(barColor);
            gc.fillRect(x, y, barWidth, barHeight);

            // 绘制边框 - 使用主题文本颜色
            gc.setStroke(themeManager.getTextColor());
            gc.setLineWidth(1);
            gc.strokeRect(x, y, barWidth, barHeight);

            double hitPx = x + barWidth / 2;
            double hitPy = y + barHeight / 2;
            JavaFxChartUtils.registerHit(config, hitPx, hitPy, i, value, series.name, 0, i);

            // 绘制X轴标签
            if (series.labels != null && i < series.labels.size()) {
                gc.setFill(themeManager.getTextColor());
                gc.setFont(themeManager.getLabelFont());
                gc.setTextAlign(TextAlignment.CENTER);
                gc.fillText(series.labels.get(i), x + barWidth / 2,
                           config.height - config.paddingBottom + 20);
            }

            // 绘制数值标签：正值在柱顶上方，负值在柱底下方（与常见 bar 图一致）
            gc.setFill(themeManager.getTextColor());
            gc.setTextAlign(TextAlignment.CENTER);
            double labelY = value >= 0 ? y - 5 : y + barHeight + 14;
            gc.fillText(String.format("%.1f", value), x + barWidth / 2, labelY);
        }
    }
    
    private void renderGroupedBars(GraphicsContext gc, List<SeriesData> data, ChartConfig config,
                                    JavaFxThemeManager themeManager, double[] yRange,
                                    double chartWidth, double chartHeight, String[] palette) {
        int numGroups = data.size();
        if (numGroups == 0) return;

        // 从第一个series获取X轴标签列表
        SeriesData firstSeries = data.get(0);
        int numXLabels = (firstSeries.labels != null) ? firstSeries.labels.size() :
                        (firstSeries.y != null ? firstSeries.y.length() : 0);

        if (numXLabels == 0) return;

        // 计算柱子宽度和位置
        double groupWidth = chartWidth / numXLabels;
        double barWidth = groupWidth * 0.8 / numGroups;
        double groupSpacing = groupWidth * 0.2;

        double y0 = yPixelForValue(0, yRange, config, chartHeight);

        // 绘制每个series的柱子
        for (int g = 0; g < numGroups; g++) {
            SeriesData series = data.get(g);
            if (series.y == null) continue;

            // 从series.style获取颜色
            Color groupColor = series.style != null && series.style.getColor() != null ?
                JavaFxStyleApplier.parseColor(series.style.getColor()) :
                Color.web(palette[g % palette.length]);
            gc.setFill(groupColor);

            // 遍历每个X位置
            for (int i = 0; i < series.y.length() && i < numXLabels; i++) {
                double value = series.y.get(i);
                // 值为0时不绘制（表示该组在该X位置无数据）
                if (value == 0) continue;

                double yVal = yPixelForValue(value, yRange, config, chartHeight);
                double top = Math.min(yVal, y0);
                double barHeight = Math.abs(yVal - y0);
                if (barHeight < 1) {
                    barHeight = 1;
                }

                // X位置: 组起始 + 组内偏移
                double groupStartX = config.paddingLeft + i * groupWidth + groupSpacing / 2;
                double x = groupStartX + g * barWidth;
                double y = top;

                gc.fillRect(x, y, barWidth - 2, barHeight);
                gc.setStroke(themeManager.getTextColor());
                gc.strokeRect(x, y, barWidth - 2, barHeight);

                double hitPx = x + (barWidth - 2) / 2;
                double hitPy = y + barHeight / 2;
                JavaFxChartUtils.registerHit(config, hitPx, hitPy, i, value, series.name, g, i);
            }
        }

        // 绘制X轴标签 - 在分组中心位置
        gc.setFill(themeManager.getTextColor());
        gc.setFont(themeManager.getLabelFont());
        gc.setTextAlign(TextAlignment.CENTER);

        for (int i = 0; i < numXLabels; i++) {
            double groupStartX = config.paddingLeft + i * groupWidth + groupSpacing / 2;
            double x = groupStartX + (numGroups * barWidth) / 2 - barWidth / 2;

            String label = "";
            if (firstSeries.labels != null && i < firstSeries.labels.size()) {
                label = firstSeries.labels.get(i);
            } else {
                label = String.valueOf(i + 1);
            }
            gc.fillText(label, x, config.height - config.paddingBottom + 20);
        }

        // 绘制图例
        JavaFxChartUtils.drawLegend(gc, data, config, themeManager);
    }

    private void renderStackedBars(GraphicsContext gc, List<SeriesData> data, ChartConfig config,
                                   JavaFxThemeManager themeManager, boolean horizontal) {
        if (horizontal) {
            renderStackedHorizontal(gc, data, config, themeManager);
            return;
        }
        SeriesData first = data.get(0);
        if (first.y == null || first.labels == null) return;
        int nCat = Math.min(first.labels.size(), first.y.length());
        int nLay = data.size();
        double[] sum = new double[nCat];
        for (int j = 0; j < nCat; j++) {
            for (int g = 0; g < nLay; g++) {
                SeriesData s = data.get(g);
                if (s.y != null && j < s.y.length()) {
                    sum[j] += Math.max(0, s.y.get(j));
                }
            }
        }
        double maxVal = Double.MIN_VALUE;
        for (double v : sum) maxVal = Math.max(maxVal, v);
        if (maxVal <= 0) maxVal = 1;
        double[] yRange = new double[] { 0, maxVal * 1.08 };

        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        drawBarChartAxes(gc, config, yRange, themeManager);

        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;
        String[] palette = themeManager.getColorPalette();
        double barWidth = chartWidth / nCat * 0.8;
        double barSpacing = chartWidth / nCat * 0.2;

        for (int j = 0; j < nCat; j++) {
            double acc = 0;
            double x = config.paddingLeft + j * (barWidth + barSpacing) + barSpacing / 2;
            for (int g = 0; g < nLay; g++) {
                SeriesData series = data.get(g);
                if (series.y == null || j >= series.y.length()) continue;
                double v = Math.max(0, series.y.get(j));
                if (v <= 0) continue;
                double bottomH = (acc / yRange[1]) * chartHeight;
                acc += v;
                double topH = (acc / yRange[1]) * chartHeight;
                double h = topH - bottomH;
                double y = config.height - config.paddingBottom - topH;
                Color c = series.style != null && series.style.getColor() != null
                    ? com.yishape.lab.math.plot.javafx.JavaFxStyleApplier.parseColor(series.style.getColor())
                    : Color.web(palette[g % palette.length]);
                gc.setFill(c);
                gc.fillRect(x, y, barWidth, h);
                gc.setStroke(themeManager.getTextColor());
                gc.strokeRect(x, y, barWidth, h);
                double midY = y + h / 2;
                JavaFxChartUtils.registerHit(config, x + barWidth / 2, midY, j, v, series.name, g, j);
            }
            gc.setFill(themeManager.getTextColor());
            gc.setFont(themeManager.getLabelFont());
            gc.setTextAlign(TextAlignment.CENTER);
            gc.fillText(first.labels.get(j), x + barWidth / 2, config.height - config.paddingBottom + 20);
        }
        JavaFxChartUtils.drawLegend(gc, data, config, themeManager);
    }

    private void renderStackedHorizontal(GraphicsContext gc, List<SeriesData> data, ChartConfig config,
                                        JavaFxThemeManager themeManager) {
        SeriesData first = data.get(0);
        if (first.y == null || first.labels == null) return;
        int nCat = Math.min(first.labels.size(), first.y.length());
        int nLay = data.size();
        double[] sum = new double[nCat];
        for (int j = 0; j < nCat; j++) {
            for (int g = 0; g < nLay; g++) {
                SeriesData s = data.get(g);
                if (s.y != null && j < s.y.length()) {
                    sum[j] += Math.max(0, s.y.get(j));
                }
            }
        }
        double maxVal = 0;
        for (double v : sum) maxVal = Math.max(maxVal, v);
        if (maxVal <= 0) maxVal = 1;

        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        JavaFxChartUtils.drawTitle(gc, config, themeManager);

        double chartW = config.width - config.paddingLeft - config.paddingRight;
        double chartH = config.height - config.paddingTop - config.paddingBottom;
        Color axisColor = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("axisColor"), Color.DARKGRAY);
        gc.setStroke(axisColor);
        gc.setLineWidth(2);
        gc.strokeLine(config.paddingLeft, config.height - config.paddingBottom,
            config.width - config.paddingRight, config.height - config.paddingBottom);
        gc.strokeLine(config.paddingLeft, config.paddingTop, config.paddingLeft, config.height - config.paddingBottom);

        String[] palette = themeManager.getColorPalette();
        double rowH = chartH / nCat * 0.75;
        double gap = chartH / nCat * 0.25;

        for (int j = 0; j < nCat; j++) {
            double y = config.paddingTop + j * (rowH + gap);
            double acc = 0;
            for (int g = 0; g < nLay; g++) {
                SeriesData series = data.get(g);
                if (series.y == null || j >= series.y.length()) continue;
                double v = Math.max(0, series.y.get(j));
                if (v <= 0) continue;
                double x0 = config.paddingLeft + (acc / maxVal) * chartW * 0.95;
                acc += v;
                double x1 = config.paddingLeft + (acc / maxVal) * chartW * 0.95;
                Color c = series.style != null && series.style.getColor() != null
                    ? com.yishape.lab.math.plot.javafx.JavaFxStyleApplier.parseColor(series.style.getColor())
                    : Color.web(palette[g % palette.length]);
                gc.setFill(c);
                gc.fillRect(x0, y, x1 - x0, rowH);
                gc.setStroke(themeManager.getTextColor());
                gc.strokeRect(x0, y, x1 - x0, rowH);
            }
            gc.setFill(themeManager.getTextColor());
            gc.setFont(themeManager.getLabelFont());
            gc.setTextAlign(TextAlignment.RIGHT);
            gc.fillText(first.labels.get(j), config.paddingLeft - 8, y + rowH / 2 + 4);
        }
        JavaFxChartUtils.drawLegend(gc, data, config, themeManager);
    }

    private void renderHorizontalBars(GraphicsContext gc, List<SeriesData> data, ChartConfig config,
                                      JavaFxThemeManager themeManager) {
        SeriesData series = data.get(0);
        if (series.y == null || series.labels == null) return;
        int n = Math.min(series.y.length(), series.labels.size());
        List<SeriesData> one = List.of(series);
        double[] xRange = computeVerticalBarYRange(one);

        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        JavaFxChartUtils.drawTitle(gc, config, themeManager);
        double chartW = config.width - config.paddingLeft - config.paddingRight;
        double chartH = config.height - config.paddingTop - config.paddingBottom;
        Color axisColor = JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("axisColor"), Color.DARKGRAY);
        gc.setStroke(axisColor);
        gc.strokeLine(config.paddingLeft, config.height - config.paddingBottom,
            config.width - config.paddingRight, config.height - config.paddingBottom);
        gc.strokeLine(config.paddingLeft, config.paddingTop, config.paddingLeft, config.height - config.paddingBottom);

        String[] palette = themeManager.getColorPalette();
        double rowH = chartH / n * 0.72;
        double gap = chartH / n * 0.28;
        boolean uniformColor = Boolean.TRUE.equals(series.extraData.get("uniformBarColor"));
        double plotW = chartW * 0.92;
        double x0line = xPixelForBarhValue(0, xRange, config, plotW);
        for (int i = 0; i < n; i++) {
            double v = series.y.get(i);
            double y = config.paddingTop + i * (rowH + gap);
            double xVal = xPixelForBarhValue(v, xRange, config, plotW);
            double left = Math.min(x0line, xVal);
            double w = Math.abs(xVal - x0line);
            if (v != 0 && w < 1) {
                w = 1;
            }
            gc.setFill(Color.web(uniformColor ? palette[0] : palette[i % palette.length]));
            gc.fillRect(left, y, w, rowH);
            gc.setStroke(themeManager.getTextColor());
            gc.strokeRect(left, y, w, rowH);
            gc.setFill(themeManager.getTextColor());
            gc.setFont(themeManager.getLabelFont());
            gc.setTextAlign(TextAlignment.RIGHT);
            gc.fillText(series.labels.get(i), config.paddingLeft - 8, y + rowH / 2 + 4);
            gc.setTextAlign(TextAlignment.CENTER);
            double labelX = v >= 0 ? Math.max(x0line, xVal) + 18 : Math.min(x0line, xVal) - 18;
            gc.fillText(String.format("%.1f", v), labelX, y + rowH / 2 + 4);
            JavaFxChartUtils.registerHit(config, left + w / 2, y + rowH / 2, i, v, series.name, 0, i);
        }
    }
    
    @Override
    public String getChartType() {
        return "bar";
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
