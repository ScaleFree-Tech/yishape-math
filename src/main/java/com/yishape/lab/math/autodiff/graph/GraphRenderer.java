package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.impl.RereDiffMatrix;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Renders autodiff DAGs as Graphviz DOT for debugging.
 * Operates on the tensor graph via {@link RereDiffVector#tensor}.
 */
public class GraphRenderer {

    private GraphRenderer() {}

    public static String renderVector(RereDiffVector root) {
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.tensor.buildTopo(order, visited);

        StringBuilder sb = new StringBuilder();
        sb.append("digraph AD {\n");
        sb.append("  rankdir=BT;\n");
        sb.append("  node [shape=box, style=filled, fontname=\"Courier\"];\n");

        for (int idx = 0; idx < order.size(); idx++) {
            RereDiffTensor t = order.get(idx);
            String id = "n" + Integer.toUnsignedString(System.identityHashCode(t));
            long size = t.value().totalSize();
            String label = "node" + idx + "\\n" + (t.isLeaf() ? "leaf" : "op") + " [" + size + "]";
            String color = t.isLeaf() ? "lightblue" : "lightyellow";
            sb.append("  ").append(id).append(" [label=\"").append(label)
                    .append("\", fillcolor=").append(color).append("];\n");

            if (t.inputs() != null) {
                for (RereDiffTensor in : t.inputs()) {
                    String inId = "n" + Integer.toUnsignedString(System.identityHashCode(in));
                    sb.append("  ").append(id).append(" -> ").append(inId).append(";\n");
                }
            }
        }

        sb.append("}\n");
        return sb.toString();
    }

    public static String renderMatrix(RereDiffMatrix root) {
        List<RereDiffMatrix> order = new ArrayList<>();
        Set<RereDiffMatrix> visited = new HashSet<>();
        root.buildTopo(order, visited);

        StringBuilder sb = new StringBuilder();
        sb.append("digraph AutodiffMatrix {\n");
        sb.append("  rankdir=BT;\n");
        sb.append("  node [shape=box, style=filled, fontname=\"Courier\"];\n");

        for (int idx = 0; idx < order.size(); idx++) {
            RereDiffMatrix v = order.get(idx);
            String id = "m" + System.identityHashCode(v);
            String label = matrixNodeLabel(v, idx);
            String color = v.isLeaf ? "lightblue" : "lightyellow";
            sb.append("  ").append(id).append(" [label=\"").append(label)
                    .append("\", fillcolor=").append(color).append("];\n");

            if (v.inputs != null) {
                for (RereDiffMatrix in : v.inputs) {
                    String inId = "m" + Integer.toUnsignedString(System.identityHashCode(in));
                    sb.append("  ").append(id).append(" -> ").append(inId).append(";\n");
                }
            }
        }

        sb.append("}\n");
        return sb.toString();
    }

    private static String matrixNodeLabel(RereDiffMatrix v, int idx) {
        String shape = "[" + v.value.rows() + "x" + v.value.cols() + "]";
        String kind = v.isLeaf ? "leaf" : "op";
        return "node" + idx + "\\n" + kind + " " + shape;
    }
}
