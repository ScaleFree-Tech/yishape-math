package com.yishape.lab.math.plot.svg.renderers;

import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.ChartConfig;
import com.yishape.lab.math.plot.javafx.JavaFxChartRenderer.SeriesData;
import com.yishape.lab.math.plot.javafx.JavaFxThemeManager;

/**
 * 极坐标图表 SVG（柱/线/散点）：按 {@link SeriesData#x} 为角位置（度或弧度）、{@link SeriesData#y} 为径向值；
 * 绘制同心圆与角度射线，并标注径向刻度与角度刻度。
 */
public class SvgPolarRenderer extends AbstractSvgChartRenderer {

    private final String polarType; // "bar", "line", "scatter"

    public SvgPolarRenderer() { this("line"); }
    public SvgPolarRenderer(String polarType) { this.polarType = polarType; }

    /** 与数据同长时为角位置；若 max|x|&gt;2π 视为度，否则视为弧度。无 x 时均分整圆。 */
    private static double[] sampleAngles(SeriesData series) {
        int n = series.y.length();
        double[] ang = new double[n];
        if (series.x != null && series.x.length() == n) {
            double maxAbs = 0;
            for (int i = 0; i < n; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(series.x.get(i)));
            }
            boolean asDegrees = maxAbs > 2 * Math.PI + 0.01;
            for (int i = 0; i < n; i++) {
                double th = series.x.get(i);
                ang[i] = (asDegrees ? Math.toRadians(th) : th) - Math.PI / 2;
            }
        } else {
            double step = 2 * Math.PI / Math.max(1, n);
            for (int i = 0; i < n; i++) {
                ang[i] = i * step - Math.PI / 2;
            }
        }
        return ang;
    }

    private static double bisect(double a, double b) {
        return (a + b) / 2;
    }

    /** 第 i 个扇区的左右角边界（弧度，已含 −π/2 偏移） */
    private static double wedgeStart(double[] ang, int i) {
        if (ang.length == 1) {
            return ang[0] - Math.PI;
        }
        if (i == 0) {
            return bisect(ang[0] - (ang[1] - ang[0]), ang[0]);
        }
        return bisect(ang[i - 1], ang[i]);
    }

    private static double wedgeEnd(double[] ang, int i) {
        if (ang.length == 1) {
            return ang[0] + Math.PI;
        }
        if (i == ang.length - 1) {
            return bisect(ang[i], ang[i] + (ang[i] - ang[i - 1]));
        }
        return bisect(ang[i], ang[i + 1]);
    }

    @Override
    protected void renderSvgContent(StringBuilder sb, SeriesData series,
                                   ChartConfig config, JavaFxThemeManager themeManager) {
        if (series.y == null) return;
        int n = series.y.length();
        if (n == 0) return;

        double maxValue = Double.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            maxValue = Math.max(maxValue, series.y.get(i));
        }
        if (maxValue <= 0 || !Double.isFinite(maxValue)) {
            maxValue = 1;
        }

        double cx = config.width / 2.0;
        double cy = config.height / 2.0 + 18;
        double maxR = Math.min(config.width, config.height) / 2.0 - 88;
        if (maxR < 40) {
            maxR = 40;
        }

        double[] ang = sampleAngles(series);

        // 同心圆 + 径向刻度（沿右侧水平方向标注）
        int rings = 5;
        for (int level = 1; level <= rings; level++) {
            double r = maxR * level / rings;
            sb.append("<circle cx=\"").append((int) cx).append("\" cy=\"").append((int) cy)
              .append("\" r=\"").append((int) r).append("\" fill=\"none\" stroke=\"")
              .append(gridColor).append("\" stroke-width=\"0.8\"/>\n");
            double rv = maxValue * level / rings;
            sb.append("<text x=\"").append((int) (cx + r + 4)).append("\" y=\"").append((int) (cy + 4))
              .append("\" class=\"tick-label\" font-size=\"9\" fill=\"").append(textColor)
              .append("\">").append(formatTickLabel(rv)).append("</text>\n");
        }

        // 射线 + 角度刻度（外圈）
        for (int i = 0; i < n; i++) {
            double a = ang[i];
            double ex = cx + Math.cos(a) * maxR;
            double ey = cy + Math.sin(a) * maxR;
            sb.append("<line x1=\"").append((int) cx).append("\" y1=\"").append((int) cy)
              .append("\" x2=\"").append((int) ex).append("\" y2=\"").append((int) ey)
              .append("\" stroke=\"").append(gridColor).append("\" stroke-width=\"0.8\"/>\n");
            double lx = cx + Math.cos(a) * (maxR + 12);
            double ly = cy + Math.sin(a) * (maxR + 12);
            String lab;
            if (series.x != null && i < series.x.length()) {
                double xv = series.x.get(i);
                lab = Math.abs(xv - (int) xv) < 1e-6 ? String.valueOf((int) xv) : formatTickLabel(xv);
            } else {
                lab = String.valueOf(i);
            }
            sb.append("<text x=\"").append((int) lx).append("\" y=\"").append((int) ly)
              .append("\" text-anchor=\"middle\" class=\"tick-label\" font-size=\"8\" fill=\"")
              .append(textColor).append("\" dy=\".3em\">").append(escXml(lab)).append("</text>\n");
        }

        String color = colorPalette[0];
        if ("bar".equals(polarType)) {
            for (int i = 0; i < n; i++) {
                double a0 = wedgeStart(ang, i);
                double a1 = wedgeEnd(ang, i);
                double r = (series.y.get(i) / maxValue) * maxR;
                StringBuilder d = new StringBuilder();
                d.append("M").append(String.format("%.2f,%.2f", cx, cy));
                int steps = 16;
                for (int k = 0; k <= steps; k++) {
                    double t = a0 + (a1 - a0) * k / steps;
                    d.append(" L").append(String.format("%.2f,%.2f",
                        cx + Math.cos(t) * r, cy + Math.sin(t) * r));
                }
                d.append(" Z");
                sb.append("<path d=\"").append(d).append("\" fill=\"").append(color)
                  .append("\" opacity=\"0.75\" stroke=\"").append(axisColor)
                  .append("\" stroke-width=\"0.6\"/>\n");
            }
        } else if ("line".equals(polarType)) {
            sb.append("<polyline points=\"");
            for (int i = 0; i < n; i++) {
                double r = (series.y.get(i) / maxValue) * maxR;
                double px = cx + Math.cos(ang[i]) * r;
                double py = cy + Math.sin(ang[i]) * r;
                if (i > 0) sb.append(" ");
                sb.append(String.format("%.1f,%.1f", px, py));
            }
            sb.append("\" fill=\"none\" stroke=\"").append(color).append("\" stroke-width=\"2\"/>\n");
        } else {
            for (int i = 0; i < n; i++) {
                double r = (series.y.get(i) / maxValue) * maxR;
                double px = cx + Math.cos(ang[i]) * r;
                double py = cy + Math.sin(ang[i]) * r;
                sb.append("<circle cx=\"").append((int) px).append("\" cy=\"").append((int) py)
                  .append("\" r=\"4\" fill=\"").append(color).append("\"/>\n");
            }
        }
    }
}
