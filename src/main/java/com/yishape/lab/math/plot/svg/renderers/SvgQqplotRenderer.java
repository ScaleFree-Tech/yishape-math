package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.*;

/**
 * Q-Q plot SVG renderer - scatter plot against theoretical quantiles.
 */
public class SvgQqplotRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        double[] sorted = series.extraData != null
            ? (double[]) series.extraData.get("sortedData") : null;

        if (sorted == null) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("Q-Q plot requires sorted data").append("</text>\n");
            return;
        }

        double cLeft = config.paddingLeft;
        double cRight = config.width - config.paddingRight;
        double cTop = config.paddingTop;
        double cBottom = config.height - config.paddingBottom;
        double cWidth = cRight - cLeft;
        double cHeight = cBottom - cTop;

        int n = sorted.length;
        double[] theoretical = new double[n];
        for (int i = 0; i < n; i++) {
            // Standard normal quantiles for the expected distribution
            theoretical[i] = quantileNormal((i + 0.5) / n);
        }

        double dataMin = sorted[0], dataMax = sorted[n - 1];
        double theoMin = theoretical[0], theoMax = theoretical[n - 1];

        // Draw grid
        int xTicks = Math.min(axisTickCount(config.width), 8);
        int yTicks = Math.min(axisTickCount(config.height), 8);
        sb.append("<g stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\" opacity=\"0.7\">\n");
        for (int i = 0; i <= yTicks; i++) {
            double y = cTop + i * cHeight / yTicks;
            sb.append("  <line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)y)
              .append("\" x2=\"").append((int)cRight).append("\" y2=\"").append((int)y).append("\"/>\n");
        }
        for (int i = 0; i <= xTicks; i++) {
            double x = cLeft + i * cWidth / xTicks;
            sb.append("  <line x1=\"").append((int)x).append("\" y1=\"").append((int)cTop)
              .append("\" x2=\"").append((int)x).append("\" y2=\"").append((int)cBottom).append("\"/>\n");
        }
        sb.append("</g>\n");

        // Draw axes
        sb.append("<line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)cBottom)
          .append("\" x2=\"").append((int)cRight).append("\" y2=\"").append((int)cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");
        sb.append("<line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)cTop)
          .append("\" x2=\"").append((int)cLeft).append("\" y2=\"").append((int)cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");

        // X axis ticks (theoretical quantiles)
        for (int i = 0; i <= xTicks; i++) {
            double x = cLeft + i * cWidth / xTicks;
            double val = theoMin + i * (theoMax - theoMin) / xTicks;
            sb.append("<text x=\"").append((int)x).append("\" y=\"").append((int)(cBottom + 16))
              .append("\" text-anchor=\"middle\" class=\"tick-label\">")
              .append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int)x).append("\" y1=\"").append((int)cBottom)
              .append("\" x2=\"").append((int)x).append("\" y2=\"").append((int)(cBottom + 4))
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        // Y axis ticks (sample quantiles)
        for (int i = 0; i <= yTicks; i++) {
            double y = cBottom - i * cHeight / yTicks;
            double val = dataMin + i * (dataMax - dataMin) / yTicks;
            sb.append("<text x=\"").append((int)(cLeft - 6)).append("\" y=\"").append((int)(y + 4))
              .append("\" text-anchor=\"end\" class=\"tick-label\">")
              .append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)y)
              .append("\" x2=\"").append((int)(cLeft - 4)).append("\" y2=\"").append((int)y)
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        // Axis labels（与 JavaFX qqplot 一致：未设置时用默认名，否则用 config）
        String xLab = (config.xlabel != null && !config.xlabel.isEmpty()) ? config.xlabel : "Theoretical Quantiles";
        String yLab = (config.ylabel != null && !config.ylabel.isEmpty()) ? config.ylabel : "Sample Quantiles";
        sb.append("<text x=\"").append((int)(cLeft + cWidth / 2)).append("\" y=\"").append((int)(config.height - 8))
          .append("\" text-anchor=\"middle\" class=\"axis-label\">").append(escXml(xLab)).append("</text>\n");
        sb.append("<text x=\"14\" y=\"").append((int)(cTop + cHeight / 2))
          .append("\" text-anchor=\"middle\" class=\"axis-label\" transform=\"rotate(-90, 14, ")
          .append((int)(cTop + cHeight / 2)).append(")\">").append(escXml(yLab)).append("</text>\n");

        // Draw diagonal reference line (data on theoretical quantiles)
        double refX1 = cLeft + (theoretical[0] - theoMin) / (theoMax - theoMin) * cWidth;
        double refY1 = cBottom - (sorted[0] - dataMin) / (dataMax - dataMin) * cHeight;
        double refX2 = cLeft + (theoretical[n - 1] - theoMin) / (theoMax - theoMin) * cWidth;
        double refY2 = cBottom - (sorted[n - 1] - dataMin) / (dataMax - dataMin) * cHeight;
        sb.append("<line x1=\"").append((int)refX1).append("\" y1=\"").append((int)refY1)
          .append("\" x2=\"").append((int)refX2).append("\" y2=\"").append((int)refY2)
          .append("\" stroke=\"#94a3b8\" stroke-width=\"1.5\" stroke-dasharray=\"5,3\"/>\n");

        // Draw scatter points
        for (int i = 0; i < n; i++) {
            double px = cLeft + (theoretical[i] - theoMin) / (theoMax - theoMin) * cWidth;
            double py = cBottom - (sorted[i] - dataMin) / (dataMax - dataMin) * cHeight;
            sb.append("<circle cx=\"").append((int)px).append("\" cy=\"").append((int)py)
              .append("\" r=\"4\" fill=\"").append(colorPalette[0]).append("\"/>\n");
        }
    }

    // Standard normal quantile approximation (Abramowitz-Stegun approximation)
    private double quantileNormal(double p) {
        if (p <= 0) return -10;
        if (p >= 1) return 10;
        double sign = (p < 0.5) ? -1 : 1;
        double pp = (p < 0.5) ? p : 1 - p;
        double t = Math.sqrt(Math.log(1 / (pp * pp)));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.001353;
        double d3 = 0.000244;
        double z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
        return sign * z;
    }
}
