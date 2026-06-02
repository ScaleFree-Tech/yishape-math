package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.List;

/**
 * 平行坐标图SVG渲染器。
 */
public class SvgParallelRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        IMatrix<Double> matrix = (IMatrix<Double>) series.extraData.get("matrixData");
        List<String> dimList = (List<String>) series.extraData.get("dimensions");

        int numDimensions = 5;
        String[] dimensions = new String[]{"Dim1", "Dim2", "Dim3", "Dim4", "Dim5"};
        boolean demoData = matrix == null || matrix.getColNum() <= 0 || matrix.getRowNum() <= 0;

        if (!demoData) {
            numDimensions = matrix.getColNum();
            dimensions = new String[numDimensions];
            for (int i = 0; i < numDimensions; i++) {
                if (dimList != null && i < dimList.size() && dimList.get(i) != null) {
                    dimensions[i] = dimList.get(i);
                } else {
                    dimensions[i] = "Dim" + (i + 1);
                }
            }
        }

        double chartLeft = config.paddingLeft;
        double chartRight = config.width - config.paddingRight;
        double chartTop = config.paddingTop + 34;
        double dimLabelReserve = 54;
        double chartBottom = config.height - config.paddingBottom - dimLabelReserve;
        double plotH = Math.max(40, chartBottom - chartTop);

        double[] dimensionX = new double[numDimensions];
        if (numDimensions == 1) {
            dimensionX[0] = (chartLeft + chartRight) / 2;
        } else {
            for (int i = 0; i < numDimensions; i++) {
                dimensionX[i] = chartLeft + (chartRight - chartLeft) * i / (numDimensions - 1);
            }
        }

        double[] dmin = new double[numDimensions];
        double[] dmax = new double[numDimensions];
        for (int i = 0; i < numDimensions; i++) {
            dmin[i] = Double.MAX_VALUE;
            dmax[i] = -Double.MAX_VALUE;
        }

        if (!demoData) {
            for (int i = 0; i < matrix.getRowNum(); i++) {
                for (int j = 0; j < numDimensions; j++) {
                    double v = matrix.get(i, j);
                    dmin[j] = Math.min(dmin[j], v);
                    dmax[j] = Math.max(dmax[j], v);
                }
            }
        } else {
            // Demo data fallback
            for (int j = 0; j < numDimensions; j++) {
                dmin[j] = 0;
                dmax[j] = 100;
            }
        }

        // 确保范围有效
        for (int j = 0; j < numDimensions; j++) {
            if (!Double.isFinite(dmin[j]) || !Double.isFinite(dmax[j]) || dmax[j] <= dmin[j]) {
                dmin[j] = 0;
                dmax[j] = 100;
            }
        }

        // 绘制网格线
        sb.append("<g stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\" opacity=\"0.7\">\n");
        for (int i = 0; i < numDimensions; i++) {
            sb.append("  <line x1=\"").append((int)dimensionX[i]).append("\" y1=\"")
              .append((int)chartTop).append("\" x2=\"").append((int)dimensionX[i]).append("\" y2=\"")
              .append((int)chartBottom).append("\"/>\n");
        }
        int gridLines = 5;
        for (int i = 0; i <= gridLines; i++) {
            double y = chartTop + (chartBottom - chartTop) * i / gridLines;
            sb.append("  <line x1=\"").append((int)chartLeft).append("\" y1=\"")
              .append((int)y).append("\" x2=\"").append((int)chartRight).append("\" y2=\"")
              .append((int)y).append("\"/>\n");
        }
        sb.append("</g>\n");

        // 绘制维度标签
        for (int i = 0; i < numDimensions; i++) {
            sb.append("<text x=\"").append((int)dimensionX[i]).append("\" y=\"")
              .append((int)(chartBottom + dimLabelReserve * 0.65)).append("\" text-anchor=\"middle\" class=\"axis-label\">")
              .append(escXml(dimensions[i])).append("</text>\n");
        }

        // 绘制数据线
        int numRows = demoData ? 8 : matrix.getRowNum();
        for (int r = 0; r < numRows; r++) {
            String color = colorPalette[r % colorPalette.length];
            sb.append("<polyline points=\"");
            for (int i = 0; i < numDimensions; i++) {
                double v = demoData ? Math.random() * 80 + 10 : matrix.get(r, i);
                double norm = (v - dmin[i]) / (dmax[i] - dmin[i]);
                double y = chartBottom - norm * plotH;
                if (i > 0) sb.append(" ");
                sb.append((int)dimensionX[i]).append(",").append((int)y);
            }
            sb.append("\" fill=\"none\" stroke=\"").append(color)
              .append("\" stroke-width=\"1.5\" opacity=\"0.7\"/>\n");
        }

        // 绘制坐标轴
        for (int i = 0; i < numDimensions; i++) {
            sb.append("<line x1=\"").append((int)dimensionX[i]).append("\" y1=\"")
              .append((int)chartTop).append("\" x2=\"").append((int)dimensionX[i]).append("\" y2=\"")
              .append((int)chartBottom).append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1.5\"/>\n");
            // 刻度标签
            for (int t = 0; t <= 4; t++) {
                double v = dmin[i] + (dmax[i] - dmin[i]) * (4 - t) / 4.0;
                double y = chartTop + plotH * t / 4.0;
                sb.append("<text x=\"").append((int)(dimensionX[i] - 4)).append("\" y=\"")
                  .append((int)(y + 4)).append("\" text-anchor=\"end\" class=\"tick-label\">")
                  .append(formatTickLabel(v)).append("</text>\n");
            }
        }
    }
}
