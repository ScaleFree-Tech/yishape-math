package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * Exports tensor-based autodiff computation graphs to JSON for HPC / GPU / external backends.
 *
 * <p>Mirrors {@link GraphExporter} but operates on {@link RereDiffTensor} nodes.
 * Tensors naturally carry shape information via their {@code shape()} method,
 * which is serialized into the JSON for the Rust/GPU backends to use.</p>
 *
 * <p>This is the first step toward making the tensor AD graph directly exportable
 * to GPU/HPC backends without going through the vector-graph bridge.</p>
 */
public final class TensorGraphExporter {

    private TensorGraphExporter() {
    }

    /**
     * Exports a tensor-based computation graph to JSON.
     *
     * @param root the root (output) node of the computation graph
     * @return JSON string compatible with HPC/GPU backends
     */
    public static String toJson(RereDiffTensor root) {
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        Map<RereDiffTensor, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < order.size(); i++) {
            indexMap.put(order.get(i), i);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("{\"nodes\":[");
        for (int i = 0; i < order.size(); i++) {
            RereDiffTensor v = order.get(i);
            if (i > 0) sb.append(',');
            appendNode(sb, v, i, indexMap);
        }
        sb.append("]}");
        return sb.toString();
    }

    private static void appendNode(StringBuilder sb, RereDiffTensor v, int id,
                                    Map<RereDiffTensor, Integer> indexMap) {
        sb.append("{\"id\":").append(id);

        // Shape: use exportShape if set (fused pattern nodes), otherwise tensor's own shape
        int[] shape = v.exportShape != null ? v.exportShape : v.shape();
        sb.append(",\"shape\":[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(shape[i]);
        }
        sb.append(']');

        // Operation tag
        sb.append(",\"op\":\"");
        sb.append(v.opTag != null ? v.opTag : (v.isLeaf ? "leaf" : "unknown"));
        sb.append('"');

        // Leaf data
        if (v.isLeaf) {
            double[] data = v.value.toDoubleArray();
            sb.append(",\"data\":[");
            for (int i = 0; i < data.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(data[i]);
            }
            sb.append(']');
        }

        // Scalar parameters
        if (!Double.isNaN(v.scalarParam) && !Double.isInfinite(v.scalarParam)) {
            sb.append(",\"scalar\":").append(v.scalarParam);
        }
        if (!Double.isNaN(v.scalarParam2) && !Double.isInfinite(v.scalarParam2)) {
            sb.append(",\"param2\":").append(v.scalarParam2);
        }

        // Backward indices (e.g. MaxPool2d argmax)
        if (v.backwardIndices != null && v.backwardIndices.length > 0) {
            sb.append(",\"indices\":[");
            for (int i = 0; i < v.backwardIndices.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(v.backwardIndices[i]);
            }
            sb.append(']');
        }

        // Input references
        if (v.inputs != null && !v.inputs.isEmpty()) {
            List<RereDiffTensor> inputs = new ArrayList<>();
            for (RereDiffTensor inp : v.inputs) {
                if (inp != null && indexMap.containsKey(inp)) {
                    inputs.add(inp);
                }
            }
            if (!inputs.isEmpty()) {
                sb.append(",\"inputs\":[");
                for (int j = 0; j < inputs.size(); j++) {
                    if (j > 0) sb.append(',');
                    sb.append(indexMap.get(inputs.get(j)));
                }
                sb.append(']');
            }
        }

        sb.append('}');
    }
}
