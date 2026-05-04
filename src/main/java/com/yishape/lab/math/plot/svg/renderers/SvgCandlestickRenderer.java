package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.List;

/**
 * K线图SVG渲染器（涨红跌绿）。
 */
public class SvgCandlestickRenderer extends AbstractSvgChartRenderer {

    private static final String RISE_COLOR = "#c45b5b"; // 涨
    private static final String FALL_COLOR = "#4d8f72";  // 跌

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        IMatrix<Double> matrix = (IMatrix<Double>) series.extraData.get("matrixData");
        if (matrix == null || matrix.getColNum() < 4) return;

        int n = matrix.getRowNum();
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        double candleWidth = chartWidth / n * 0.8;
        double spacing = chartWidth / n;

        double minPrice = Double.MAX_VALUE, maxPrice = Double.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            minPrice = Math.min(minPrice, matrix.get(i, 2));
            maxPrice = Math.max(maxPrice, matrix.get(i, 3));
        }

        double[] axis = niceAxis(minPrice, maxPrice, 6);
        double lo = axis[0], hi = axis[1], step = axis[2];
        double priceRange = hi - lo;
        if (priceRange <= 1e-15) priceRange = 1.0;
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;
        double cLeft = config.paddingLeft, cBottom = config.height - config.paddingBottom;

        // 坐标轴
        drawAxes(sb, config, 0, Math.max(1, n - 1), lo, hi,
                 config.xlabel, config.ylabel);

        for (int i = 0; i < n; i++) {
            double open = matrix.get(i, 0), close = matrix.get(i, 1);
            double low = matrix.get(i, 2), high = matrix.get(i, 3);
            boolean rising = close >= open;
            String color = rising ? RISE_COLOR : FALL_COLOR;

            double x = cLeft + i * spacing + spacing * 0.1;
            double openY = dataToY(config, open, lo, hi);
            double closeY = dataToY(config, close, lo, hi);
            double lowY = dataToY(config, low, lo, hi);
            double highY = dataToY(config, high, lo, hi);

            // 影线（上下须）
            sb.append("<line x1=\"").append((int)(x + candleWidth / 2)).append("\" y1=\"").append((int)highY)
              .append("\" x2=\"").append((int)(x + candleWidth / 2)).append("\" y2=\"").append((int)lowY)
              .append("\" stroke=\"").append(color).append("\" stroke-width=\"1.15\"/>\n");

            // 实体
            double bodyTop = Math.min(openY, closeY);
            double bodyH = Math.max(3, Math.abs(closeY - openY));
            sb.append("<rect x=\"").append((int)x).append("\" y=\"").append((int)bodyTop)
              .append("\" width=\"").append((int)candleWidth).append("\" height=\"").append((int)bodyH)
              .append("\" fill=\"").append(color).append("\" stroke=\"").append(color).append("\" stroke-width=\"1\"/>\n");
        }
    }

    private double[] niceAxis(double dataMin, double dataMax, int targetDivisions) {
        if (!Double.isFinite(dataMin) || !Double.isFinite(dataMax)) return new double[]{0.0, 1.0, 0.2};
        if (dataMax < dataMin) { double t = dataMin; dataMin = dataMax; dataMax = t; }
        if (dataMax - dataMin < 1e-15) {
            double pad = Math.abs(dataMax) > 1e-6 ? Math.abs(dataMax) * 0.08 : 0.5;
            return new double[]{dataMax - pad, dataMax + pad, pad / 2};
        }
        double span = dataMax - dataMin;
        double rawStep = span / Math.max(2, targetDivisions - 1);
        double exp = Math.floor(Math.log10(rawStep));
        double pow = Math.pow(10.0, exp);
        double err = rawStep / pow;
        double nf = err <= 1.0 ? 1.0 : err <= 2.0 ? 2.0 : err <= 5.0 ? 5.0 : 10.0;
        double step = nf * pow;
        double axisLo = (float) Math.floor(dataMin / step) * step;
        double axisHi = (float) Math.ceil(dataMax / step) * step;
        if (axisHi <= axisLo) axisHi = axisLo + step;
        return new double[]{axisLo, axisHi, step};
    }
}
