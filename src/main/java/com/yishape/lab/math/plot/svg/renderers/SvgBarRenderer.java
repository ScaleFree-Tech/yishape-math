package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.ArrayList;
import java.util.List;

/**
 * 柱状图SVG渲染器（支持普通柱状图、分组柱状图、堆叠柱状图、横向柱状图）。
 */
public class SvgBarRenderer extends AbstractSvgChartRenderer {

    @Override
    public String renderMulti(List<SeriesData> dataList, ChartConfig config, String theme) {
        if (dataList.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        buildHeader(sb, config);
        themeManager = new JavaFxThemeManager(theme);
        applyTheme(themeManager);
        renderBackground(sb, config);
        renderTitle(sb, config);
        renderMultiSeries(sb, dataList, config, themeManager);
        buildFooter(sb);
        return sb.toString();
    }

    private void renderMultiSeries(StringBuilder sb, List<SeriesData> dataList,
                                  ChartConfig config, JavaFxThemeManager themeManager) {
        if (dataList.isEmpty()) return;
        boolean horizontal = Boolean.TRUE.equals(dataList.get(0).extraData.get("horizontal"));
        boolean stacked = Boolean.TRUE.equals(dataList.get(0).extraData.get("stacked"));
        boolean grouped = dataList.size() > 1;

        double maxValue = Double.NEGATIVE_INFINITY;
        if (stacked) {
            // For stacked bars, compute max as sum of all stacks at any position
            int n = dataList.get(0).y.length();
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (SeriesData s : dataList) sum += s.y.get(i);
                maxValue = Math.max(maxValue, sum);
            }
        } else {
            for (SeriesData s : dataList) {
                if (s.y == null) continue;
                for (int i = 0; i < s.y.length(); i++) maxValue = Math.max(maxValue, s.y.get(i));
            }
        }
        if (!Double.isFinite(maxValue) || maxValue <= 0) maxValue = 1;
        double yMax = maxValue <= 0 ? maxValue * 1.1 : Math.max(maxValue * 1.1, 0.1);

        double cLeft = config.paddingLeft;
        double cBottom = config.height - config.paddingBottom;
        double cTop = config.paddingTop;
        double chartWidth = config.width - cLeft - config.paddingRight;
        double chartHeight = cBottom - cTop;

        if (horizontal) {
            drawHorizontalBarAxes(sb, chartWidth, chartHeight, cLeft, cTop, cBottom, yMax);
            renderHorizontalBars(sb, dataList.get(0), config, chartWidth, chartHeight, cLeft, cBottom, yMax);
            appendAxisTitleLabels(sb, config);
            return;
        }

        // 网格线（竖向柱：水平线）
        int yTicks = 5;
        sb.append("<g stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\" opacity=\"0.7\">\n");
        for (int i = 0; i <= yTicks; i++) {
            double y = cBottom - i * chartHeight / yTicks;
            sb.append("  <line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)y)
              .append("\" x2=\"").append((int)(config.width - config.paddingRight)).append("\" y2=\"").append((int)y).append("\"/>\n");
        }
        sb.append("</g>\n");

        // Y轴
        sb.append("<line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)cTop)
          .append("\" x2=\"").append((int)cLeft).append("\" y2=\"").append((int)cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");
        // X轴
        sb.append("<line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)cBottom)
          .append("\" x2=\"").append((int)(config.width - config.paddingRight)).append("\" y2=\"").append((int)cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");

        // Y轴刻度标签
        for (int i = 0; i <= yTicks; i++) {
            double val = yMax * i / yTicks;
            double y = cBottom - i * chartHeight / yTicks;
            sb.append("<text x=\"").append((int)(cLeft - 6)).append("\" y=\"").append((int)(y + 4))
              .append("\" text-anchor=\"end\" class=\"tick-label\">").append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int)cLeft).append("\" y1=\"").append((int)y)
              .append("\" x2=\"").append((int)(cLeft - 4)).append("\" y2=\"").append((int)y)
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        if (stacked) {
            renderStackedBars(sb, dataList, config, chartWidth, chartHeight, cLeft, cBottom, yMax);
        } else if (grouped) {
            renderGroupedBars(sb, dataList, config, chartWidth, chartHeight, cLeft, cBottom, yMax);
        } else {
            renderSimpleBars(sb, dataList.get(0), config, chartWidth, chartHeight, cLeft, cBottom, yMax);
        }
        appendAxisTitleLabels(sb, config);
    }

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        List<SeriesData> data = new ArrayList<>();
        data.add(series);
        renderMultiSeries(sb, data, config, themeManager);
    }

    private void renderSimpleBars(StringBuilder sb, SeriesData s, ChartConfig config,
                                  double chartWidth, double chartHeight,
                                  double cLeft, double cBottom, double yMax) {
        int n = s.y.length();
        double barWidth = chartWidth / n * 0.8;
        for (int i = 0; i < n; i++) {
            double val = s.y.get(i);
            double barHeight = (val / yMax) * chartHeight;
            double x = cLeft + i * (chartWidth / n) + (chartWidth / n - barWidth) / 2;
            double y = cBottom - barHeight;
            String color = colorPalette[i % colorPalette.length];
            sb.append("<rect x=\"").append((int)x).append("\" y=\"").append((int)y)
              .append("\" width=\"").append((int)barWidth).append("\" height=\"").append((int)barHeight)
              .append("\" fill=\"").append(color).append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }
    }

    private void renderGroupedBars(StringBuilder sb, List<SeriesData> data,
                                   ChartConfig config, double chartWidth, double chartHeight,
                                   double cLeft, double cBottom, double yMax) {
        int groups = data.size();
        int n = data.get(0).y.length();
        double groupWidth = chartWidth / n * 0.85;
        double barWidth = groupWidth / groups * 0.85;
        for (int gi = 0; gi < groups; gi++) {
            SeriesData s = data.get(gi);
            for (int i = 0; i < n; i++) {
                double val = s.y.get(i);
                double barHeight = (val / yMax) * chartHeight;
                double x = cLeft + i * groupWidth + gi * barWidth + (groupWidth - groups * barWidth) / 2;
                double y = cBottom - barHeight;
                String color = colorPalette[gi % colorPalette.length];
                sb.append("<rect x=\"").append((int)x).append("\" y=\"").append((int)y)
                  .append("\" width=\"").append((int)barWidth).append("\" height=\"").append((int)barHeight)
                  .append("\" fill=\"").append(color).append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.7\"/>\n");
            }
        }
    }

    private void renderStackedBars(StringBuilder sb, List<SeriesData> data,
                                    ChartConfig config, double chartWidth, double chartHeight,
                                    double cLeft, double cBottom, double yMax) {
        int n = data.get(0).y.length();
        double barWidth = chartWidth / n * 0.85;
        for (int i = 0; i < n; i++) {
            double stackTop = cBottom;
            for (SeriesData s : data) {
                double val = s.y.get(i);
                double barHeight = (val / yMax) * chartHeight;
                double y = stackTop - barHeight;
                String color = colorPalette[data.indexOf(s) % colorPalette.length];
                sb.append("<rect x=\"").append((int)(cLeft + i * barWidth + barWidth * 0.075))
                  .append("\" y=\"").append((int)y)
                  .append("\" width=\"").append((int)(barWidth * 0.85))
                  .append("\" height=\"").append((int)barHeight)
                  .append("\" fill=\"").append(color).append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.7\"/>\n");
                stackTop = y;
            }
        }
    }

    /** 横向柱图：数值在 X 方向，刻度与网格画在底部 X 轴（不再把数值刻度画在左侧 Y 轴）。 */
    private void drawHorizontalBarAxes(StringBuilder sb, double chartWidth, double chartHeight,
                                       double cLeft, double cTop, double cBottom, double yMax) {
        int xTicks = 5;
        sb.append("<g stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\" opacity=\"0.7\">\n");
        for (int i = 0; i <= xTicks; i++) {
            double xx = cLeft + i * chartWidth / xTicks;
            sb.append("  <line x1=\"").append((int) xx).append("\" y1=\"").append((int) cTop)
              .append("\" x2=\"").append((int) xx).append("\" y2=\"").append((int) cBottom).append("\"/>\n");
        }
        sb.append("</g>\n");
        sb.append("<line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) cTop)
          .append("\" x2=\"").append((int) cLeft).append("\" y2=\"").append((int) cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");
        sb.append("<line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) cBottom)
          .append("\" x2=\"").append((int) (cLeft + chartWidth)).append("\" y2=\"").append((int) cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");
        for (int i = 0; i <= xTicks; i++) {
            double val = yMax * i / xTicks;
            double xx = cLeft + i * chartWidth / xTicks;
            sb.append("<text x=\"").append((int) xx).append("\" y=\"").append((int) (cBottom + 16))
              .append("\" text-anchor=\"middle\" class=\"tick-label\">").append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int) xx).append("\" y1=\"").append((int) cBottom)
              .append("\" x2=\"").append((int) xx).append("\" y2=\"").append((int) (cBottom + 4))
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }
    }

    private void renderHorizontalBars(StringBuilder sb, SeriesData s, ChartConfig config,
                                      double chartWidth, double chartHeight,
                                      double cLeft, double cBottom, double yMax) {
        int n = s.y.length();
        double barHeight = chartHeight / n * 0.8;
        for (int i = 0; i < n; i++) {
            double val = s.y.get(i);
            double barWidth = (val / yMax) * chartWidth;
            double y = cBottom - (i + 0.5) * (chartHeight / n) - barHeight / 2;
            String color = colorPalette[i % colorPalette.length];
            sb.append("<rect x=\"").append((int)cLeft).append("\" y=\"").append((int)y)
              .append("\" width=\"").append((int)barWidth).append("\" height=\"").append((int)barHeight)
              .append("\" fill=\"").append(color).append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
            if (s.labels != null && i < s.labels.size()) {
                sb.append("<text x=\"").append((int) (cLeft - 8)).append("\" y=\"")
                  .append((int) (y + barHeight / 2 + 4)).append("\" text-anchor=\"end\" class=\"tick-label\" font-size=\"10\">")
                  .append(escXml(s.labels.get(i))).append("</text>\n");
            }
        }
    }
}
