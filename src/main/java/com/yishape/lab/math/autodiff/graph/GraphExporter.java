package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.impl.RereDiffMatrix;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Exports autodiff computation graphs to JSON for HPC / external backends.
 * 将自动微分计算图导出为 JSON，供 HPC 或外部后端使用。
 */
public final class GraphExporter {

    private GraphExporter() {
    }

    /**
     * Exports a vector-based computation graph to JSON.
     */
    public static String toJson(RereDiffVector root) {
        List<RereDiffVector> order = new ArrayList<>();
        Set<RereDiffVector> visited = new HashSet<>();
        root.buildTopo(order, visited);

        Map<RereDiffVector, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < order.size(); i++) {
            indexMap.put(order.get(i), i);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{\"nodes\":[");
        for (int i = 0; i < order.size(); i++) {
            RereDiffVector v = order.get(i);
            if (i > 0) sb.append(',');
            appendVectorNode(sb, v, i, indexMap);
        }
        sb.append("]}");
        return sb.toString();
    }

    /**
     * Exports a matrix-based computation graph to JSON.
     */
    public static String toJson(RereDiffMatrix root) {
        List<RereDiffMatrix> order = new ArrayList<>();
        Set<RereDiffMatrix> visited = new HashSet<>();
        root.buildTopo(order, visited);

        Map<RereDiffMatrix, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < order.size(); i++) {
            indexMap.put(order.get(i), i);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{\"nodes\":[");
        for (int i = 0; i < order.size(); i++) {
            RereDiffMatrix v = order.get(i);
            if (i > 0) sb.append(',');
            appendMatrixNode(sb, v, i, indexMap);
        }
        sb.append("]}");
        return sb.toString();
    }

    private static void appendVectorNode(StringBuilder sb, RereDiffVector v, int id,
            Map<RereDiffVector, Integer> indexMap) {
        sb.append("{\"id\":").append(id);
        sb.append(",\"shape\":[").append(v.value.size()).append(']');
        appendOpTag(sb, v);
        if (v.isLeaf) {
            appendLeafData(sb, v.value.getData());
        }
        appendScalarParam(sb, v.scalarParam);
        appendScalarParam2(sb, v.scalarParam2);
        appendInputs(sb, v.inputs, indexMap);
        sb.append('}');
    }

    private static void appendMatrixNode(StringBuilder sb, RereDiffMatrix v, int id,
            Map<RereDiffMatrix, Integer> indexMap) {
        sb.append("{\"id\":").append(id);
        sb.append(",\"shape\":[").append(v.value.rows()).append(',').append(v.value.cols()).append(']');
        appendOpTag(sb, v);
        if (v.isLeaf) {
            double[][] d = v.value.getData();
            sb.append(",\"data\":[");
            for (int i = 0; i < d.length; i++) {
                if (i > 0) sb.append(',');
                sb.append('[');
                double[] row = d[i];
                for (int j = 0; j < row.length; j++) {
                    if (j > 0) sb.append(',');
                    sb.append(row[j]);
                }
                sb.append(']');
            }
            sb.append(']');
        }
        appendScalarParam(sb, v.scalarParam);
        appendMatrixInputs(sb, v.inputs, indexMap);
        sb.append('}');
    }

    private static void appendOpTag(StringBuilder sb, RereDiffVector v) {
        sb.append(",\"op\":\"");
        sb.append(v.opTag != null ? v.opTag : (v.isLeaf ? "leaf" : "unknown"));
        sb.append('"');
    }

    private static void appendOpTag(StringBuilder sb, RereDiffMatrix v) {
        sb.append(",\"op\":\"");
        sb.append(v.opTag != null ? v.opTag : (v.isLeaf ? "leaf" : "unknown"));
        sb.append('"');
    }

    private static void appendLeafData(StringBuilder sb, double[] data) {
        sb.append(",\"data\":[");
        for (int i = 0; i < data.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(data[i]);
        }
        sb.append(']');
    }

    private static void appendScalarParam(StringBuilder sb, double scalar) {
        if (!Double.isNaN(scalar)) {
            sb.append(",\"scalar\":").append(scalar);
        }
    }

    private static void appendScalarParam2(StringBuilder sb, double param2) {
        if (!Double.isNaN(param2)) {
            sb.append(",\"param2\":").append(param2);
        }
    }

    private static void appendInputs(StringBuilder sb, List<RereDiffVector> inputs,
            Map<RereDiffVector, Integer> indexMap) {
        if (inputs != null && !inputs.isEmpty()) {
            sb.append(",\"inputs\":[");
            for (int j = 0; j < inputs.size(); j++) {
                if (j > 0) sb.append(',');
                sb.append(indexMap.get(inputs.get(j)));
            }
            sb.append(']');
        }
    }

    private static void appendMatrixInputs(StringBuilder sb, List<RereDiffMatrix> inputs,
            Map<RereDiffMatrix, Integer> indexMap) {
        if (inputs != null && !inputs.isEmpty()) {
            sb.append(",\"inputs\":[");
            for (int j = 0; j < inputs.size(); j++) {
                if (j > 0) sb.append(',');
                sb.append(indexMap.get(inputs.get(j)));
            }
            sb.append(']');
        }
    }
}
