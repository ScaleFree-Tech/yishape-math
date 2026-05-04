package com.yishape.lab.math.plot.echarts.j3d;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * 将当前 {@link Echarts3dPlot} 状态编译为浏览器端 option 字面量（非严格 JSON）。
 */
final class Echarts3dOptionJson {

    private Echarts3dOptionJson() {
    }

    static String build(List<Echarts3dSeriesConfig> seriesList,
                         Map<String, Object> sceneConfig,
                         Map<String, String> labels) {
        StringBuilder opt = new StringBuilder();
        opt.append("{");

        String title = labels.get("title");
        if (title != null && !title.isEmpty()) {
            opt.append("title: { text: '").append(Echarts3dEscapes.jsSingleQuoted(title)).append("'");
            String subtitle = labels.get("subtitle");
            if (subtitle != null && !subtitle.isEmpty()) {
                opt.append(", subtext: '").append(Echarts3dEscapes.jsSingleQuoted(subtitle)).append("'");
            }
            opt.append(" },\n");
        }

        opt.append("tooltip: { show: true },\n");
        opt.append("toolbox: { feature: { saveAsImage: {}, restore: {} } },\n");

        if (seriesList.size() > 1) {
            opt.append("legend: { data: [");
            for (int i = 0; i < seriesList.size(); i++) {
                if (i > 0) opt.append(", ");
                opt.append("'").append(Echarts3dEscapes.jsSingleQuoted(seriesList.get(i).name)).append("'");
            }
            opt.append("] },\n");
        }

        opt.append("xAxis3D: { name: '").append(Echarts3dEscapes.jsSingleQuoted(labels.getOrDefault("x", "X"))).append("', type: 'value' },\n");
        opt.append("yAxis3D: { name: '").append(Echarts3dEscapes.jsSingleQuoted(labels.getOrDefault("y", "Y"))).append("', type: 'value' },\n");
        opt.append("zAxis3D: { name: '").append(Echarts3dEscapes.jsSingleQuoted(labels.getOrDefault("z", "Z"))).append("', type: 'value' },\n");

        opt.append("grid3D: {\n");
        opt.append("  boxWidth: 100, boxDepth: 100, boxHeight: 100,\n");
        @SuppressWarnings("unchecked")
        Map<String, Object> viewControl = (Map<String, Object>) sceneConfig.get("viewControl");
        if (viewControl != null) {
            opt.append("  viewControl: {\n");
            viewControl.forEach((k, v) -> {
                opt.append("    ").append(k).append(": ");
                if (v instanceof String) opt.append("'").append(v).append("'");
                else opt.append(v);
                opt.append(",\n");
            });
            opt.append("  },\n");
        }
        opt.append("  light: { main: { intensity: 1.2, shadow: true }, ambient: { intensity: 0.4 } }\n");
        opt.append("},\n");

        opt.append("series: [\n");
        for (int i = 0; i < seriesList.size(); i++) {
            if (i > 0) opt.append(",\n");
            appendSeries(opt, seriesList.get(i));
        }
        opt.append("\n]\n");

        opt.append("}");
        return opt.toString();
    }

    private static void appendSeries(StringBuilder opt, Echarts3dSeriesConfig s) {
        opt.append("  {\n");
        opt.append("    type: '").append(s.type).append("',\n");
        opt.append("    name: '").append(Echarts3dEscapes.jsSingleQuoted(s.name)).append("',\n");

        if (s.data != null && s.data.length > 0) {
            opt.append("    data: ");
            appendDataArray(opt, s.data, s.symbolSize);
            opt.append(",\n");
        }

        if (s.symbolSize > 0) {
            opt.append("    symbolSize: ").append(s.symbolSize).append(",\n");
        } else if (s.symbolSize < 0) {
            opt.append("    symbolSize: function(data) { return data[3] || 8; },\n");
        }

        if (s.color != null) {
            opt.append("    itemStyle: { color: '").append(s.color).append("' },\n");
        }

        if (s.wireframe) {
            opt.append("    wireframe: { show: true },\n");
        }

        if (!s.shading) {
            opt.append("    shading: 'none',\n");
        }

        if (s.opacity < 1.0) {
            opt.append("    itemStyle: { opacity: ").append(s.opacity).append(" },\n");
        }

        s.extra.forEach((k, v) -> {
            opt.append("    ").append(k).append(": ");
            appendJsonValue(opt, v);
            opt.append(",\n");
        });

        opt.append("    emphasis: { itemStyle: { borderWidth: 2, borderColor: '#fff' } }\n");
        opt.append("  }");
    }

    private static void appendDataArray(StringBuilder opt, double[][] data, double symbolSize) {
        opt.append("[");
        for (int i = 0; i < data.length; i++) {
            if (i > 0) opt.append(", ");
            opt.append("[");
            for (int j = 0; j < data[i].length; j++) {
                if (j > 0) opt.append(", ");
                opt.append(formatDouble(data[i][j]));
            }
            opt.append("]");
        }
        opt.append("]");
    }

    private static void appendJsonValue(StringBuilder opt, Object v) {
        if (v instanceof String) {
            opt.append("'").append(Echarts3dEscapes.jsSingleQuoted((String) v)).append("'");
        } else if (v instanceof Number || v instanceof Boolean) {
            opt.append(v);
        } else if (v instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> m = (Map<String, Object>) v;
            opt.append("{");
            int i = 0;
            for (Map.Entry<String, Object> e : m.entrySet()) {
                if (i++ > 0) opt.append(", ");
                opt.append(e.getKey()).append(": ");
                appendJsonValue(opt, e.getValue());
            }
            opt.append("}");
        } else if (v instanceof Object[] arr) {
            opt.append("[");
            for (int i = 0; i < arr.length; i++) {
                if (i > 0) opt.append(", ");
                appendJsonValue(opt, arr[i]);
            }
            opt.append("]");
        } else {
            opt.append("null");
        }
    }

    static String formatDouble(double d) {
        if (Double.isNaN(d)) return "null";
        if (Double.isInfinite(d)) return d > 0 ? "Infinity" : "-Infinity";
        return String.format(Locale.US, "%.6f", d);
    }
}
