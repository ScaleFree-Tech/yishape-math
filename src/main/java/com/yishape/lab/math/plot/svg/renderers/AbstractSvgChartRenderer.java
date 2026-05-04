package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

import java.util.List;
import java.util.Locale;

/**
 * SVG渲染器基类：复用JavaFX渲染器的计算逻辑，输出SVG文本。
 * 每个具体渲染器只需实现 {@link #renderSvgContent(StringBuilder, SeriesData, ChartConfig, JavaFxThemeManager)}。
 */
public abstract class AbstractSvgChartRenderer {

    /** 主题色板（长度≥3） */
    protected String[] colorPalette;
    /** 背景色 */
    protected String backgroundColor = "#ffffff";
    /** 文字色 */
    protected String textColor = "#374151";
    /** 坐标轴色 */
    protected String axisColor = "#9ca3af";
    /** 网格色 */
    protected String gridColor = "#e5e7eb";
    /** 标签字体 */
    protected String labelFontFamily = "Arial, sans-serif";
    protected double labelFontSize = 12;
    /** 当前主题管理器（供子类在 renderSvgContent 中使用） */
    protected JavaFxThemeManager themeManager;

    // ==================== 公开方法 ====================

    public final String render(SeriesData series, ChartConfig config, String theme) {
        return renderMulti(java.util.List.of(series), config, theme);
    }

    /** 多序列渲染（默认调用单序列render，可被子类覆盖） */
    public String renderMulti(List<SeriesData> seriesList, ChartConfig config, String theme) {
        if (seriesList.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        buildHeader(sb, config);
        themeManager = new JavaFxThemeManager(theme);
        applyTheme(themeManager);
        renderBackground(sb, config);
        renderTitle(sb, config);
        for (SeriesData s : seriesList) {
            renderSvgContent(sb, s, config, themeManager);
        }
        buildFooter(sb);
        return sb.toString();
    }

    // ==================== 可覆盖方法 ====================

    /** 渲染图表具体内容（柱、线、点等） */
    protected abstract void renderSvgContent(StringBuilder sb, SeriesData series,
                                             ChartConfig config, JavaFxThemeManager theme);

    // ==================== 通用SVG构建工具 ====================

    protected void buildHeader(StringBuilder sb, ChartConfig config) {
        sb.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        sb.append("<svg xmlns=\"http://www.w3.org/2000/svg\" ")
          .append("width=\"").append(config.width).append("\" ")
          .append("height=\"").append(config.height).append("\" ")
          .append("viewBox=\"0 0 ").append(config.width).append(" ").append(config.height).append("\">\n");
        sb.append("<defs>\n");
        sb.append("  <style>\n");
        sb.append("    .title { font-family: ").append(fontCss(labelFontFamily))
          .append("; font-size: 15px; font-weight: bold; fill: ").append(textColor).append("; }\n");
        sb.append("    .subtitle { font-family: ").append(fontCss(labelFontFamily))
          .append("; font-size: 11px; fill: ").append(textColor).append("; opacity: 0.7; }\n");
        sb.append("    .axis-label { font-family: ").append(fontCss(labelFontFamily))
          .append("; font-size: 11px; fill: ").append(textColor).append("; }\n");
        sb.append("    .tick-label { font-family: ").append(fontCss(labelFontFamily))
          .append("; font-size: 10px; fill: ").append(textColor).append("; }\n");
        sb.append("  </style>\n");
        sb.append("</defs>\n");
    }

    protected void buildFooter(StringBuilder sb) {
        sb.append("</svg>\n");
    }

    protected void renderBackground(StringBuilder sb, ChartConfig config) {
        sb.append("<rect width=\"").append(config.width).append("\" height=\"").append(config.height)
          .append("\" fill=\"").append(backgroundColor).append("\"/>\n");
    }

    protected void renderTitle(StringBuilder sb, ChartConfig config) {
        double cx = config.width / 2.0;
        if (config.title != null && !config.title.isEmpty()) {
            sb.append("<text x=\"").append((int) cx).append("\" y=\"30\" ")
              .append("text-anchor=\"middle\" class=\"title\">")
              .append(escXml(config.title)).append("</text>\n");
        }
        if (config.subtitle != null && !config.subtitle.isEmpty()) {
            sb.append("<text x=\"").append((int) cx).append("\" y=\"48\" ")
              .append("text-anchor=\"middle\" class=\"subtitle\">")
              .append(escXml(config.subtitle)).append("</text>\n");
        }
    }

    /**
     * Draw only left + bottom axis lines (no top/right), with tick labels.
     */
    protected void drawBoxOnly(StringBuilder sb, ChartConfig config,
                              double xMin, double xMax,
                              double yMin, double yMax,
                              String xLabel, String yLabel) {
        double cLeft = config.paddingLeft;
        double cRight = config.width - config.paddingRight;
        double cTop = config.paddingTop;
        double cBottom = config.height - config.paddingBottom;
        double cWidth = cRight - cLeft;
        double cHeight = cBottom - cTop;

        // Grid lines
        int xTicks = Math.min(axisTickCount(config.width), 10);
        int yTicks = Math.min(axisTickCount(config.height), 8);
        if (config.showGrid) {
            sb.append("<g stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\" opacity=\"0.7\">\n");
            for (int i = 0; i <= yTicks; i++) {
                double y = cTop + i * cHeight / yTicks;
                sb.append("  <line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) y)
                  .append("\" x2=\"").append((int) cRight).append("\" y2=\"").append((int) y).append("\"/>\n");
            }
            for (int i = 0; i <= xTicks; i++) {
                double x = cLeft + i * cWidth / xTicks;
                sb.append("  <line x1=\"").append((int) x).append("\" y1=\"").append((int) cTop)
                  .append("\" x2=\"").append((int) x).append("\" y2=\"").append((int) cBottom).append("\"/>\n");
            }
            sb.append("</g>\n");
        }

        // X axis (bottom only)
        sb.append("<line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) cBottom)
          .append("\" x2=\"").append((int) cRight).append("\" y2=\"").append((int) cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");
        // Y axis (left only)
        sb.append("<line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) cTop)
          .append("\" x2=\"").append((int) cLeft).append("\" y2=\"").append((int) cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");

        // X axis tick labels
        for (int i = 0; i <= xTicks; i++) {
            double x = cLeft + i * cWidth / xTicks;
            double val = xMin + i * (xMax - xMin) / xTicks;
            sb.append("<text x=\"").append((int) x).append("\" y=\"").append((int) (cBottom + 16))
              .append("\" text-anchor=\"middle\" class=\"tick-label\">")
              .append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int) x).append("\" y1=\"").append((int) cBottom)
              .append("\" x2=\"").append((int) x).append("\" y2=\"").append((int) (cBottom + 4))
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        // Y axis tick labels
        for (int i = 0; i <= yTicks; i++) {
            double y = cBottom - i * cHeight / yTicks;
            double val = yMin + i * (yMax - yMin) / yTicks;
            sb.append("<text x=\"").append((int) (cLeft - 6)).append("\" y=\"").append((int) (y + 4))
              .append("\" text-anchor=\"end\" class=\"tick-label\">")
              .append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) y)
              .append("\" x2=\"").append((int) (cLeft - 4)).append("\" y2=\"").append((int) y)
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        // Axis labels
        if (xLabel != null && !xLabel.isEmpty()) {
            sb.append("<text x=\"").append((int) (cLeft + cWidth / 2)).append("\" y=\"")
              .append(config.height - 8).append("\" text-anchor=\"middle\" class=\"axis-label\">")
              .append(escXml(xLabel)).append("</text>\n");
        }
        if (yLabel != null && !yLabel.isEmpty()) {
            sb.append("<text x=\"14\" y=\"").append((int) (cTop + cHeight / 2))
              .append("\" text-anchor=\"middle\" class=\"axis-label\" ")
              .append("transform=\"rotate(-90, 14, ").append((int) (cTop + cHeight / 2)).append(")\">")
              .append(escXml(yLabel)).append("</text>\n");
        }
    }

    /**
     * Draw inner box-only axes for a sub-cell region (left+bottom only, no top/right).
     * @param sb SVG string builder
     * @param cellLeft left edge of the cell
     * @param cellTop top edge of the cell
     * @param cellRight right edge of the cell
     * @param cellBottom bottom edge of the cell
     * @param xMin data min for x
     * @param xMax data max for x
     * @param yMin data min for y
     * @param yMax data max for y
     * @param numXTicks number of x ticks
     * @param numYTicks number of y ticks
     */
    protected void drawInnerBox(StringBuilder sb,
                                 double cellLeft, double cellTop, double cellRight, double cellBottom,
                                 double xMin, double xMax, double yMin, double yMax,
                                 int numXTicks, int numYTicks) {
        drawInnerBox(sb, cellLeft, cellTop, cellRight, cellBottom, xMin, xMax, yMin, yMax,
            numXTicks, numYTicks, false);
    }

    /**
     * @param xTickLabelsInside 为 true 时 X 轴数字画在轴线之上（散点区内），避免轴线紧挨下方直方图时与柱条重叠。
     */
    protected void drawInnerBox(StringBuilder sb,
                                 double cellLeft, double cellTop, double cellRight, double cellBottom,
                                 double xMin, double xMax, double yMin, double yMax,
                                 int numXTicks, int numYTicks, boolean xTickLabelsInside) {
        double pad = 8;
        double cellW = cellRight - cellLeft;
        double cellH = cellBottom - cellTop;

        // X axis (bottom only)
        sb.append("<line x1=\"").append((int)cellLeft).append("\" y1=\"").append((int)cellBottom)
          .append("\" x2=\"").append((int)cellRight).append("\" y2=\"").append((int)cellBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");
        // Y axis (left only)
        sb.append("<line x1=\"").append((int)cellLeft).append("\" y1=\"").append((int)cellTop)
          .append("\" x2=\"").append((int)cellLeft).append("\" y2=\"").append((int)cellBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.5\"/>\n");

        // X tick labels
        for (int t = 0; t <= numXTicks; t++) {
            double tfrac = (double) t / numXTicks;
            double tx = cellLeft + pad + tfrac * (cellW - 2 * pad);
            double tickVal = xMin + tfrac * (xMax - xMin);
            int tickY2 = xTickLabelsInside ? (int) (cellBottom - 3) : (int) (cellBottom + 3);
            sb.append("<line x1=\"").append((int) tx).append("\" y1=\"").append((int) cellBottom)
              .append("\" x2=\"").append((int) tx).append("\" y2=\"").append(tickY2)
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.7\"/>\n");
            int textY = xTickLabelsInside ? (int) (cellBottom - 5) : (int) (cellBottom + 10);
            sb.append("<text x=\"").append((int)tx).append("\" y=\"").append(textY)
              .append("\" text-anchor=\"middle\" class=\"tick-label\" font-size=\"8\"")
              .append(xTickLabelsInside ? " dominant-baseline=\"ideographic\"" : "")
              .append(">")
              .append(formatTickLabel(tickVal)).append("</text>\n");
        }

        // Y tick labels
        for (int t = 0; t <= numYTicks; t++) {
            double tfrac = (double) t / numYTicks;
            double ty = cellBottom - pad - tfrac * (cellH - 2 * pad);
            double tickVal = yMax - tfrac * (yMax - yMin);
            sb.append("<text x=\"").append((int)(cellLeft - 3)).append("\" y=\"").append((int)ty)
              .append("\" text-anchor=\"end\" class=\"tick-label\" font-size=\"8\" dy=\".3em\">")
              .append(formatTickLabel(tickVal)).append("</text>\n");
        }
    }

    /**
     * Pairplot 风格：X 刻度写在单元格最底行（labelBaselineY），轴线在 axisY，避免与下方子图重叠。
     */
    protected void drawInsetBottomXTicksInCell(StringBuilder sb, double plotLeft, double plotRight, double axisY,
                                               double labelBaselineY, double xMin, double xMax, int numTicks) {
        if (numTicks < 1) {
            numTicks = 1;
        }
        double pad = 3;
        double span = Math.max(1e-9, plotRight - plotLeft - 2 * pad);
        for (int t = 0; t <= numTicks; t++) {
            double frac = t / (double) numTicks;
            double tx = plotLeft + pad + frac * span;
            double val = xMin + frac * (xMax - xMin);
            sb.append("<line x1=\"").append((int) tx).append("\" y1=\"").append((int) axisY)
              .append("\" x2=\"").append((int) tx).append("\" y2=\"").append((int) (axisY - 3))
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.6\"/>\n");
            sb.append("<text x=\"").append((int) tx).append("\" y=\"").append((int) labelBaselineY)
              .append("\" text-anchor=\"middle\" class=\"tick-label\" font-size=\"7\" dominant-baseline=\"ideographic\">")
              .append(formatTickLabel(val)).append("</text>\n");
        }
    }

    /**
     * Pairplot 风格：Y 刻度写在轴线右侧（labelX 略大于 axisX），避免与左侧子图重叠。
     */
    protected void drawInsetLeftYTicksInCell(StringBuilder sb, double axisX, double plotTop, double plotBottom,
                                               double labelX, double yMin, double yMax, int numTicks) {
        if (numTicks < 1) {
            numTicks = 1;
        }
        double pad = 3;
        double span = Math.max(1e-9, plotBottom - plotTop - 2 * pad);
        for (int t = 0; t <= numTicks; t++) {
            double frac = t / (double) numTicks;
            double ty = plotBottom - pad - frac * span;
            double val = yMax - frac * (yMax - yMin);
            sb.append("<line x1=\"").append((int) axisX).append("\" y1=\"").append((int) ty)
              .append("\" x2=\"").append((int) (axisX + 3)).append("\" y2=\"").append((int) ty)
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.6\"/>\n");
            sb.append("<text x=\"").append((int) labelX).append("\" y=\"").append((int) (ty + 3))
              .append("\" text-anchor=\"start\" class=\"tick-label\" font-size=\"7\" dy=\".3em\">")
              .append(formatTickLabel(val)).append("</text>\n");
        }
    }

    /** 仅底部 X 数值刻度（pairplot 底行等，不重复画轴线） */
    protected void drawBottomXTicksOnly(StringBuilder sb, double cellLeft, double cellRight, double cellBottom,
                                        double xMin, double xMax, int numTicks) {
        if (numTicks < 1) {
            numTicks = 1;
        }
        double pad = 6;
        double iw = Math.max(1e-9, cellRight - cellLeft - 2 * pad);
        for (int t = 0; t <= numTicks; t++) {
            double frac = t / (double) numTicks;
            double tx = cellLeft + pad + frac * iw;
            double val = xMin + frac * (xMax - xMin);
            sb.append("<line x1=\"").append((int) tx).append("\" y1=\"").append((int) cellBottom)
              .append("\" x2=\"").append((int) tx).append("\" y2=\"").append((int) (cellBottom + 3))
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.7\"/>\n");
            sb.append("<text x=\"").append((int) tx).append("\" y=\"").append((int) (cellBottom + 11))
              .append("\" text-anchor=\"middle\" class=\"tick-label\" font-size=\"7\">")
              .append(formatTickLabel(val)).append("</text>\n");
        }
    }

    /** 仅左侧 Y 数值刻度（pairplot 左列等） */
    protected void drawLeftYTicksOnly(StringBuilder sb, double cellLeft, double cellTop, double cellBottom,
                                      double yMin, double yMax, int numTicks) {
        if (numTicks < 1) {
            numTicks = 1;
        }
        double pad = 6;
        double ih = Math.max(1e-9, cellBottom - cellTop - 2 * pad);
        for (int t = 0; t <= numTicks; t++) {
            double frac = t / (double) numTicks;
            double ty = cellBottom - pad - frac * ih;
            double val = yMax - frac * (yMax - yMin);
            sb.append("<line x1=\"").append((int) cellLeft).append("\" y1=\"").append((int) ty)
              .append("\" x2=\"").append((int) (cellLeft - 3)).append("\" y2=\"").append((int) ty)
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.7\"/>\n");
            sb.append("<text x=\"").append((int) (cellLeft - 5)).append("\" y=\"").append((int) (ty + 3))
              .append("\" text-anchor=\"end\" class=\"tick-label\" font-size=\"7\" dy=\".3em\">")
              .append(formatTickLabel(val)).append("</text>\n");
        }
    }

    /**
     * 绘制笛卡尔坐标轴（带网格、刻度标签、轴标签）。
     * 等同于JavaFX版的drawAxesCartesian。
     */
    protected void drawAxes(StringBuilder sb, ChartConfig config,
                            double xMin, double xMax,
                            double yMin, double yMax,
                            String xLabel, String yLabel) {
        double cLeft = config.paddingLeft;
        double cRight = config.width - config.paddingRight;
        double cTop = config.paddingTop;
        double cBottom = config.height - config.paddingBottom;
        double cWidth = cRight - cLeft;
        double cHeight = cBottom - cTop;

        // 网格线
        int xTicks = Math.min(axisTickCount(config.width), 10);
        int yTicks = Math.min(axisTickCount(config.height), 8);
        if (config.showGrid) {
            sb.append("<g stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\" opacity=\"0.7\">\n");
            for (int i = 0; i <= yTicks; i++) {
                double y = cTop + i * cHeight / yTicks;
                sb.append("  <line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) y)
                  .append("\" x2=\"").append((int) cRight).append("\" y2=\"").append((int) y).append("\"/>\n");
            }
            for (int i = 0; i <= xTicks; i++) {
                double x = cLeft + i * cWidth / xTicks;
                sb.append("  <line x1=\"").append((int) x).append("\" y1=\"").append((int) cTop)
                  .append("\" x2=\"").append((int) x).append("\" y2=\"").append((int) cBottom).append("\"/>\n");
            }
            sb.append("</g>\n");
        }

        // X轴
        sb.append("<line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) cBottom)
          .append("\" x2=\"").append((int) cRight).append("\" y2=\"").append((int) cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");
        // Y轴
        sb.append("<line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) cTop)
          .append("\" x2=\"").append((int) cLeft).append("\" y2=\"").append((int) cBottom)
          .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"1\"/>\n");

        // X轴刻度标签
        for (int i = 0; i <= xTicks; i++) {
            double x = cLeft + i * cWidth / xTicks;
            double val = xMin + i * (xMax - xMin) / xTicks;
            sb.append("<text x=\"").append((int) x).append("\" y=\"").append((int) (cBottom + 16))
              .append("\" text-anchor=\"middle\" class=\"tick-label\">")
              .append(formatTickLabel(val)).append("</text>\n");
            // 小刻度
            sb.append("<line x1=\"").append((int) x).append("\" y1=\"").append((int) cBottom)
              .append("\" x2=\"").append((int) x).append("\" y2=\"").append((int) (cBottom + 4))
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        // Y轴刻度标签
        for (int i = 0; i <= yTicks; i++) {
            double y = cBottom - i * cHeight / yTicks;
            double val = yMin + i * (yMax - yMin) / yTicks;
            sb.append("<text x=\"").append((int) (cLeft - 6)).append("\" y=\"").append((int) (y + 4))
              .append("\" text-anchor=\"end\" class=\"tick-label\">")
              .append(formatTickLabel(val)).append("</text>\n");
            sb.append("<line x1=\"").append((int) cLeft).append("\" y1=\"").append((int) y)
              .append("\" x2=\"").append((int) (cLeft - 4)).append("\" y2=\"").append((int) y)
              .append("\" stroke=\"").append(axisColor).append("\" stroke-width=\"0.8\"/>\n");
        }

        // 轴标签
        if (xLabel != null && !xLabel.isEmpty()) {
            sb.append("<text x=\"").append((int) (cLeft + cWidth / 2)).append("\" y=\"")
              .append(config.height - 8).append("\" text-anchor=\"middle\" class=\"axis-label\">")
              .append(escXml(xLabel)).append("</text>\n");
        }
        if (yLabel != null && !yLabel.isEmpty()) {
            sb.append("<text x=\"14\" y=\"").append((int) (cTop + cHeight / 2))
              .append("\" text-anchor=\"middle\" class=\"axis-label\" ")
              .append("transform=\"rotate(-90, 14, ").append((int) (cTop + cHeight / 2)).append(")\">")
              .append(escXml(yLabel)).append("</text>\n");
        }
    }

    // ==================== 数据映射工具 ====================

    protected double dataToX(ChartConfig config, double xVal, double xMin, double xMax) {
        double chartWidth = config.width - config.paddingLeft - config.paddingRight;
        return config.paddingLeft + (xVal - xMin) / (xMax - xMin) * chartWidth;
    }

    protected double dataToY(ChartConfig config, double yVal, double yMin, double yMax) {
        double chartHeight = config.height - config.paddingTop - config.paddingBottom;
        return config.height - config.paddingBottom - (yVal - yMin) / (yMax - yMin) * chartHeight;
    }

    // ==================== 分位数计算（seaborn标准，Type 7线性插值） ====================

    protected double quantileLinear(double[] sorted, double p) {
        if (sorted.length == 1) return sorted[0];
        double N = sorted.length;
        double pos = (N - 1) * p;
        int i = (int) Math.floor(pos);
        double weight = pos - i;
        if (i >= N - 1) return sorted[(int) N - 1];
        return sorted[i] * (1 - weight) + sorted[i + 1] * weight;
    }

    // ==================== 刻度数量 & 格式化 ====================

    protected int axisTickCount(int pixelLength) {
        if (pixelLength <= 200) return 4;
        if (pixelLength <= 400) return 6;
        return 8;
    }

    protected String formatTickLabel(double v) {
        if (Math.abs(v) >= 1e4 || (Math.abs(v) < 0.01 && v != 0)) {
            return String.format(Locale.US, "%.1e", v);
        }
        String s = String.format(Locale.US, "%.2f", v);
        return s.replaceAll("0+$", "").replaceAll("\\.$", "");
    }

    // ==================== 主题 ====================

    public void applyTheme(JavaFxThemeManager tm) {
        var cfg = tm.getCurrentConfig();
        colorPalette = tm.getColorPalette();
        backgroundColor = str(cfg.get("backgroundColor"), "#ffffff");
        textColor = str(cfg.get("textColor"), "#374151");
        axisColor = str(cfg.get("axisColor"), "#9ca3af");
        gridColor = str(cfg.get("gridColor"), "#e5e7eb");
        labelFontFamily = str(cfg.get("labelFontFamily"), "Arial, sans-serif");
        labelFontSize = 12;
    }

    private String str(Object v, String def) {
        return v == null ? def : v.toString();
    }

    // ==================== XML工具 ====================

    protected String escXml(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                 .replace("\"", "&quot;").replace("'", "&apos;");
    }

    protected String fontCss(String family) {
        return family.contains(",") ? family : "\"" + family + "\"";
    }

    // ==================== 共享的图表计算（供子类调用） ====================

    /** 计算箱线图须范围 */
    protected double[] computeWhiskerRange(double[] sorted, double q1, double median, double q3) {
        double dataMin = sorted[0];
        double dataMax = sorted[sorted.length - 1];
        double iqr = q3 - q1;
        double whiskerLo = Math.max(dataMin, q1 - 1.5 * iqr);
        double whiskerHi = Math.min(dataMax, q3 + 1.5 * iqr);
        return new double[]{whiskerLo, whiskerHi, dataMin, dataMax};
    }

    protected double[] computeDataRange(double[] values) {
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : values) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return new double[]{min, max};
    }
}
