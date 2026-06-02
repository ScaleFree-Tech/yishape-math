package com.yishape.lab.math.plot.echarts.j3d;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * 单行 ECharts GL 系列的序列化快照（{@link Echarts3dPlot} 内部状态）。
 *
 * @author lteb2
 */
final class Echarts3dSeriesConfig implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    String type;
    String name;
    double[][] data;
    String symbol = "circle";
    double symbolSize = 8;
    String color;
    double opacity = 0.8;
    boolean wireframe = false;
    boolean shading = true;
    final Map<String, Object> extra = new HashMap<>();

    Echarts3dSeriesConfig(String type, String name) {
        this.type = type;
        this.name = name;
    }
}
