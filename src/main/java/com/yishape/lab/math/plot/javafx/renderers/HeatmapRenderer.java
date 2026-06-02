package com.yishape.lab.math.plot.javafx.renderers;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer;
import com.yishape.lab.math.plot.javafx.JavaFxChartUtils;
import com.yishape.lab.math.plot.javafx.JavaFxInteractionHandler;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

import java.util.List;

/**
 * 热力图渲染器
 *
 * @author lteb2
 */
public class HeatmapRenderer implements JavaFxChartRenderer {

    /** 颜色条最小宽度（像素） */
    private static final double COLORBAR_MIN_WIDTH = 16;
    /** 颜色条最大宽度（像素） */
    private static final double COLORBAR_MAX_WIDTH = 26;
    /** 单元格间缝隙（像素） */
    private static final double CELL_GAP = 1;

    @SuppressWarnings("unchecked")
    @Override
    public void render(GraphicsContext gc, List<SeriesData> data, ChartConfig config) {
        if (data.isEmpty()) return;

        JavaFxThemeManager themeManager = new JavaFxThemeManager(config.theme);
        SeriesData series = data.get(0);

        IMatrix<Double> matrix = (IMatrix<Double>) series.extraData.get("matrixData");
        if (matrix == null) return;

        List<String> xLabels = (List<String>) series.extraData.get("xLabels");
        List<String> yLabels = (List<String>) series.extraData.get("yLabels");

        int rows = matrix.getRowNum();
        int cols = matrix.getColNum();

        double minValue = Double.MAX_VALUE;
        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v = matrix.get(i, j);
                minValue = Math.min(minValue, v);
                maxValue = Math.max(maxValue, v);
            }
        }

        gc.setFill(themeManager.getBackgroundColor());
        gc.fillRect(0, 0, config.width, config.height);
        JavaFxChartUtils.drawTitle(gc, config, themeManager);

        // 计算绘图区：左右各留颜色条空间
        double colorBarWidth = Math.min(COLORBAR_MAX_WIDTH,
            Math.max(COLORBAR_MIN_WIDTH, (config.width - config.paddingLeft - config.paddingRight) / 25.0));
        double chartWidth = config.width - config.paddingLeft - config.paddingRight - colorBarWidth - 10;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;
        double cellWidth = chartWidth / cols;
        double cellHeight = chartHeight / rows;

        String[] pal = themeManager.getColorPalette();
        Color heatCold = Color.web(pal[0]);
        Color heatWarm = Color.web(pal.length > 2 ? pal[2] : pal[pal.length - 1]);
        double valueRange = maxValue - minValue;

        // 绘制单元格
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = matrix.get(i, j);
                double normalized = valueRange > 0 ? (value - minValue) / valueRange : 0.5;
                Color cellColor = heatCold.interpolate(heatWarm, normalized);

                double x = config.paddingLeft + j * cellWidth;
                double y = config.paddingTop + i * cellHeight;
                double cw = cellWidth - CELL_GAP;
                double ch = cellHeight - CELL_GAP;

                gc.setFill(cellColor);
                gc.fillRect(x, y, cw, ch);

                // 大单元格内显示数值
                if (cw > 30 && ch > 16) {
                    gc.setFill(contrastingLabelColor(cellColor));
                    gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), Math.min(11, Math.min(cw, ch) * 0.18)));
                    gc.setTextAlign(TextAlignment.CENTER);
                    gc.fillText(formatHeatmapValue(value), x + cw / 2, y + ch / 2 + 4);
                }

                if (config.hitTestHandler != null) {
                    double cx = x + cw / 2;
                    double cy = y + ch / 2;
                    // matrix.get(i, j): i = row (Y), j = col (X)
                    JavaFxInteractionHandler.DataPoint dp =
                        new JavaFxInteractionHandler.DataPoint(cx, cy, j, i, "Heatmap", i, j);
                    dp.label = String.format("(%s, %s): %s",
                        yLabels != null && i < yLabels.size() ? yLabels.get(i) : String.valueOf(i),
                        xLabels != null && j < xLabels.size() ? xLabels.get(j) : String.valueOf(j),
                        formatHeatmapValue(value));
                    config.hitTestHandler.addDataPoint(dp);
                }
            }
        }

        gc.setStroke(JavaFxChartUtils.parseColorFromConfig(
            themeManager.getCurrentConfig().get("axisColor"), Color.web("#94a3b8")));
        gc.setLineWidth(1);
        gc.strokeRect(config.paddingLeft, config.paddingTop, chartWidth, chartHeight);

        // X轴标签
        if (xLabels != null) {
            gc.setFill(themeManager.getTextColor());
            gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 10));
            gc.setTextAlign(TextAlignment.CENTER);
            for (int j = 0; j < cols && j < xLabels.size(); j++) {
                gc.fillText(xLabels.get(j),
                    config.paddingLeft + j * cellWidth + cellWidth / 2,
                    config.height - config.paddingBottom + 20);
            }
        }

        // Y轴标签
        if (yLabels != null) {
            gc.setFill(themeManager.getTextColor());
            gc.setFont(Font.font(themeManager.getLabelFont().getFamily(), 10));
            gc.setTextAlign(TextAlignment.RIGHT);
            for (int i = 0; i < rows && i < yLabels.size(); i++) {
                gc.fillText(yLabels.get(i),
                    config.paddingLeft - 6,
                    config.paddingTop + i * cellHeight + cellHeight / 2 + 5);
            }
        }

        JavaFxChartUtils.drawCartesianAxisTitles(gc, config, themeManager);

        // 颜色条（位于图表右侧，不遮挡）
        drawColorBar(gc, config, minValue, maxValue, heatCold, heatWarm, colorBarWidth, chartHeight);
    }

    private void drawColorBar(GraphicsContext gc, ChartConfig config,
                             double minValue, double maxValue,
                             Color cold, Color warm, double barWidth, double chartHeight) {
        double barX = config.paddingLeft + (config.width - config.paddingLeft - config.paddingRight - barWidth - 10);
        double barY = config.paddingTop;
        int segments = 60;

        gc.save();
        for (int i = 0; i < segments; i++) {
            double t = i / (double) segments;
            gc.setFill(cold.interpolate(warm, t));
            double sy = barY + i * (chartHeight / segments);
            gc.fillRect(barX, sy, barWidth, chartHeight / segments + 0.5);
        }
        gc.restore();

        gc.setStroke(JavaFxChartUtils.parseColorFromConfig(
            config.theme != null ? new JavaFxThemeManager(config.theme).getCurrentConfig().get("axisColor") : null,
            Color.web("#94a3b8")));
        gc.setLineWidth(1);
        gc.strokeRect(barX, barY, barWidth, chartHeight);

        gc.setFill(new JavaFxThemeManager(config.theme).getTextColor());
        gc.setFont(Font.font(new JavaFxThemeManager(config.theme).getLabelFont().getFamily(), 9));
        gc.setTextAlign(TextAlignment.LEFT);
        gc.fillText(formatHeatmapValue(maxValue), barX + barWidth + 5, barY + 9);
        gc.fillText(formatHeatmapValue(minValue), barX + barWidth + 5, barY + chartHeight);
    }

    /** 根据单元格背景色返回对比明显的文字颜色 */
    private static Color contrastingLabelColor(Color bg) {
        double lum = 0.2126 * bg.getRed() + 0.7152 * bg.getGreen() + 0.0722 * bg.getBlue();
        return lum > 0.5 ? Color.BLACK : Color.WHITE;
    }

    private static String formatHeatmapValue(double v) {
        if (Math.abs(v) >= 1e4 || (Math.abs(v) < 0.01 && v != 0)) {
            return String.format("%.1e", v);
        }
        return String.format("%.2f", v).replaceAll("0+$", "").replaceAll("\\.$", "");
    }

    @Override
    public String getChartType() {
        return "heatmap";
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
