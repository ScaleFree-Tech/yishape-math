package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.impl.RereDiffMatrix;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Exports autodiff computation graphs to JSON for HPC / external backends.
 * Operates on the tensor graph via {@link RereDiffVector#tensor}.
 */
public final class GraphExporter {

    private GraphExporter() {
    }

    /**
     * Exports a vector-based computation graph to JSON.
     */
    public static String toJson(RereDiffVector root) {
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.tensor.buildTopo(order, visited);

        Map<RereDiffTensor, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < order.size(); i++) {
            indexMap.put(order.get(i), i);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{\"nodes\":[");
        for (int i = 0; i < order.size(); i++) {
            RereDiffTensor t = order.get(i);
            if (i > 0) sb.append(',');
            appendTensorNode(sb, t, i, indexMap);
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

    private static void appendTensorNode(StringBuilder sb, RereDiffTensor t, int id,
            Map<RereDiffTensor, Integer> indexMap) {
        sb.append("{\"id\":").append(id);
        if (t.exportShape != null) {
            sb.append(",\"shape\":[");
            for (int i = 0; i < t.exportShape.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(t.exportShape[i]);
            }
            sb.append(']');
        } else {
            sb.append(",\"shape\":[").append(t.value.totalSize()).append(']');
        }
        sb.append(",\"op\":\"");
        sb.append(t.opTag != null ? t.opTag : (t.isLeaf ? "leaf" : "unknown"));
        sb.append('"');
        if (t.isLeaf) {
            appendLeafData(sb, t.value.toDoubleArray());
        }
        appendScalarParam(sb, t.scalarParam);
        appendScalarParam2(sb, t.scalarParam2);
        if (t.backwardIndices != null && t.backwardIndices.length > 0) {
            sb.append(",\"indices\":[");
            for (int i = 0; i < t.backwardIndices.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(t.backwardIndices[i]);
            }
            sb.append(']');
        }
        if (t.inputs != null && !t.inputs.isEmpty()) {
            sb.append(",\"inputs\":[");
            for (int j = 0; j < t.inputs.size(); j++) {
                if (j > 0) sb.append(',');
                sb.append(indexMap.get(t.inputs.get(j)));
            }
            sb.append(']');
        }
        sb.append('}');
    }

    private static void appendMatrixNode(StringBuilder sb, RereDiffMatrix v, int id,
            Map<RereDiffMatrix, Integer> indexMap) {
        sb.append("{\"id\":").append(id);
        sb.append(",\"shape\":[").append(v.value.rows()).append(',').append(v.value.cols()).append(']');
        sb.append(",\"op\":\"");
        sb.append(v.opTag != null ? v.opTag : (v.isLeaf ? "leaf" : "unknown"));
        sb.append('"');
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
        appendScalarParam2(sb, v.scalarParam2);
        if (v.inputs != null && !v.inputs.isEmpty()) {
            sb.append(",\"inputs\":[");
            for (int j = 0; j < v.inputs.size(); j++) {
                if (j > 0) sb.append(',');
                sb.append(indexMap.get(v.inputs.get(j)));
            }
            sb.append(']');
        }
        sb.append('}');
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
        if (!Double.isNaN(scalar) && !Double.isInfinite(scalar)) {
            sb.append(",\"scalar\":").append(scalar);
        }
    }

    private static void appendScalarParam2(StringBuilder sb, double param2) {
        if (!Double.isNaN(param2) && !Double.isInfinite(param2)) {
            sb.append(",\"param2\":").append(param2);
        }
    }
}
