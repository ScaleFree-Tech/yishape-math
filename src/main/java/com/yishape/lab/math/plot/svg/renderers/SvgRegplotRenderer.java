package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.*;

/**
 * Regression plot SVG renderer - scatter with regression line and optional confidence band.
 */
public class SvgRegplotRenderer extends AbstractSvgChartRenderer {

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        @SuppressWarnings("unchecked")
        double[] xVals = series.extraData != null
            ? (double[]) series.extraData.get("xData") : null;
        @SuppressWarnings("unchecked")
        double[] yVals = series.extraData != null
            ? (double[]) series.extraData.get("yData") : null;
        @SuppressWarnings("unchecked")
        double[] yErrLow = series.extraData != null
            ? (double[]) series.extraData.get("yErrLow") : null;
        @SuppressWarnings("unchecked")
        double[] yErrHigh = series.extraData != null
            ? (double[]) series.extraData.get("yErrHigh") : null;

        if (xVals == null || yVals == null) {
            sb.append("<text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#9ca3af\" font-size=\"13\">")
              .append("Regplot requires x and y data").append("</text>\n");
            return;
        }

        double cLeft = config.paddingLeft;
        double cRight = config.width - config.paddingRight;
        double cTop = config.paddingTop;
        double cBottom = config.height - config.paddingBottom;
        double cWidth = cRight - cLeft;
        double cHeight = cBottom - cTop;

        double xMin = min(xVals), xMax = max(xVals);
        double yMin = min(yVals), yMax = max(yVals);
        if (yErrLow != null) yMin = Math.min(yMin, min(yErrLow));
        if (yErrHigh != null) yMax = Math.max(yMax, max(yErrHigh));

        double xRange = xMax - xMin;
        double yRange = yMax - yMin;
        if (xRange == 0) xRange = 1;
        if (yRange == 0) yRange = 1;

        // Grid
        int xTicks = Math.min(axisTickCount(config.width), 8);
        int yTicks = Math.min(axisTickCount(config.height), 8);
        sb.append("<g stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\" opacity=\"0.7\">\n");
        for (int i = 0; i <= yTicks; i++) {
            double yy = cTop + i * cHeight / yTicks;
            sb.append("  <line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)yy)
              .append("\" x2=\"").append((int)cRight).append("\" y2=\"").append((int)yy).append("\"/>\n");
        }
        for (int i = 0; i <= xTicks; i++) {
            double xx = cLeft + i * cWidth / xTicks;
            sb.append("  <line x1=\"").append((int)xx).append("\" y1=\"").append((int)cTop)
              .append("\" x2=\"").append((int)xx).append("\" y2=\"").append((int)cBottom).append("\"/>\n");
        }
        sb.append("</g>\n");

        // Axes
        sb.append("<line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)cBottom)
          .append("\" x2=\"").append((int)cRight).append("\" y2=\"").append((int)cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");
        sb.append("<line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)cTop)
          .append("\" x2=\"").append((int)cLeft).append("\" y2=\"").append((int)cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");

        // X ticks
        for (int i = 0; i <= xTicks; i++) {
            double xx = cLeft + i * cWidth / xTicks;
            double val = xMin + i * xRange / xTicks;
            sb.append("<text x=\"").append((int)xx).append("\" y=\"").append((int)(cBottom + 16))
              .append("\" text-anchor=\"middle\" class=\"tick-label\">").append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int)xx).append("\" y1=\"").append((int)cBottom)
              .append("\" x2=\"").append((int)xx).append("\" y2=\"").append((int)(cBottom + 4))
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        // Y ticks
        for (int i = 0; i <= yTicks; i++) {
            double yy = cBottom - i * cHeight / yTicks;
            double val = yMin + i * yRange / yTicks;
            sb.append("<text x=\"").append((int)(cLeft - 6)).append("\" y=\"").append((int)(yy + 4))
              .append("\" text-anchor=\"end\" class=\"tick-label\">").append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)yy)
              .append("\" x2=\"").append((int)(cLeft - 4)).append("\" y2=\"").append((int)yy)
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        // Confidence band
        if (yErrLow != null && yErrHigh != null) {
            int bandPoints = yErrLow.length;
            StringBuilder bandPath = new StringBuilder();
            // Forward along upper band
            for (int i = 0; i < bandPoints; i++) {
                double px = cLeft + (i / (bandPoints - 1.0)) * cWidth;
                double pyHigh = cBottom - (yErrHigh[i] - yMin) / yRange * cHeight;
                if (i == 0) bandPath.append("M").append((int)px).append(",").append((int)pyHigh);
                else bandPath.append("L").append((int)px).append(",").append((int)pyHigh);
            }
            // Backward along lower band to close
            for (int i = bandPoints - 1; i >= 0; i--) {
                double px = cLeft + (i / (bandPoints - 1.0)) * cWidth;
                double pyLow = cBottom - (yErrLow[i] - yMin) / yRange * cHeight;
                bandPath.append("L").append((int)px).append(",").append((int)pyLow);
            }
            bandPath.append("Z");
            sb.append("<path d=\"").append(bandPath).append("\" fill=\"").append(colorPalette[0])
              .append("\" opacity=\"0.35\"/>\n");
        }

        // Regression line
        @SuppressWarnings("unchecked")
        double[] xLine = (double[]) (series.extraData != null ? series.extraData.get("xLine") : null);
        @SuppressWarnings("unchecked")
        double[] yLine = (double[]) (series.extraData != null ? series.extraData.get("yLine") : null);
        if (xLine != null && yLine != null && xLine.length > 1) {
            StringBuilder path = new StringBuilder();
            for (int i = 0; i < xLine.length; i++) {
                double px = cLeft + (xLine[i] - xMin) / xRange * cWidth;
                double py = cBottom - (yLine[i] - yMin) / yRange * cHeight;
                if (i == 0) path.append("M").append((int)px).append(",").append((int)py);
                else path.append("L").append((int)px).append(",").append((int)py);
            }
            sb.append("<path d=\"").append(path).append("\" fill=\"none\" stroke=\"")
              .append(colorPalette[0]).append("\" stroke-width=\"2\"/>\n");
        }

        // Scatter points
        for (int i = 0; i < xVals.length; i++) {
            double px = cLeft + (xVals[i] - xMin) / xRange * cWidth;
            double py = cBottom - (yVals[i] - yMin) / yRange * cHeight;
            sb.append("<circle cx=\"").append((int)px).append("\" cy=\"").append((int)py)
              .append("\" r=\"4\" fill=\"").append(colorPalette[0]).append("\" opacity=\"0.6\"/>\n");
        }
    }

    private double min(double[] arr) {
        double m = arr[0];
        for (double v : arr) m = Math.min(m, v);
        return m;
    }

    private double max(double[] arr) {
        double m = arr[0];
        for (double v : arr) m = Math.max(m, v);
        return m;
    }
}
