package com.yishape.lab.math.plot.echarts.j3d;

import java.util.List;
import java.util.Map;

/**
 * 拼装完整 HTML / 嵌入式脚本片段，对应 JavaFX {@link com.yishape.lab.math.plot.javafx.j3d.JavaFx3dSceneSupport} 的导出职责。
 */
final class Echarts3dHtmlTemplate {

    static final String CDN_ECHARTS = "https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js";
    static final String CDN_ECHARTS_GL = "https://cdn.jsdelivr.net/npm/echarts-gl@2.0.9/dist/echarts-gl.min.js";

    private Echarts3dHtmlTemplate() {
    }

    static String fullPage(List<Echarts3dSeriesConfig> seriesList,
                            Map<String, Object> sceneConfig,
                            Map<String, String> labels,
                            String themeName) {
        StringBuilder html = new StringBuilder();
        html.append("<!DOCTYPE html>\n");
        html.append("<html>\n<head>\n");
        html.append("<meta charset=\"UTF-8\">\n");
        html.append("<title>").append(Echarts3dEscapes.html(labels.getOrDefault("title", "3D Chart"))).append("</title>\n");
        html.append("<script src=\"").append(CDN_ECHARTS).append("\"></script>\n");
        html.append("<script src=\"").append(CDN_ECHARTS_GL).append("\"></script>\n");
        html.append("<style>\n");
        html.append("body { margin: 0; padding: 0; overflow: hidden; }\n");
        html.append("#chart { width: 100vw; height: 100vh; }\n");
        html.append("</style>\n");
        html.append("</head>\n<body>\n");
        html.append("<div id=\"chart\"></div>\n");
        html.append("<script>\n");
        html.append(bootstrapChartScript(seriesList, sceneConfig, labels, themeName));
        html.append("</script>\n");
        html.append("</body>\n</html>");
        return html.toString();
    }

    static String bootstrapChartScript(List<Echarts3dSeriesConfig> seriesList,
                                     Map<String, Object> sceneConfig,
                                     Map<String, String> labels,
                                     String themeName) {
        StringBuilder js = new StringBuilder();
        js.append("var chart = echarts.init(document.getElementById('chart'), '")
                .append(themeName != null ? themeName : "")
                .append("');\n");
        js.append("var option = ").append(Echarts3dOptionJson.build(seriesList, sceneConfig, labels)).append(";\n");
        js.append("chart.setOption(option);\n");
        js.append("window.addEventListener('resize', function() { chart.resize(); });\n");
        return js.toString();
    }
}
