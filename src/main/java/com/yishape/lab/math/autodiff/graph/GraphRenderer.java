package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.impl.RereDiffMatrix;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Renders autodiff DAGs as Graphviz DOT for debugging.
 * 将自动微分 DAG 渲染为 Graphviz DOT，便于调试可视化。
 */
public class GraphRenderer {

    private GraphRenderer() {}

    public static String renderVector(RereDiffVector root) {
        List<RereDiffVector> order = new ArrayList<>();
        Set<RereDiffVector> visited = new HashSet<>();
        root.buildTopo(order, visited);

        StringBuilder sb = new StringBuilder();
        sb.append("digraph AD {\n");
        sb.append("  rankdir=BT;\n");
        sb.append("  node [shape=box, style=filled, fontname=\"Courier\"];\n");

        for (int idx = 0; idx < order.size(); idx++) {
            RereDiffVector v = order.get(idx);
            String id = "n" + System.identityHashCode(v);
            String label = nodeLabel(v, idx);
            String color = v.isLeaf ? "lightblue" : "lightyellow";
            sb.append("  ").append(id).append(" [label=\"").append(label)
                    .append("\", fillcolor=").append(color).append("];\n");

            for (RereDiffVector in : v.inputs) {
                String inId = "n" + System.identityHashCode(in);
                sb.append("  ").append(id).append(" -> ").append(inId).append(";\n");
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

            for (RereDiffMatrix in : v.inputs) {
                String inId = "m" + System.identityHashCode(in);
                sb.append("  ").append(id).append(" -> ").append(inId).append(";\n");
            }
        }

        sb.append("}\n");
        return sb.toString();
    }

    private static String nodeLabel(RereDiffVector v, int idx) {
        int size = v.value.size();
        String shape = "[" + size + "]";
        String kind = v.isLeaf ? "leaf" : "op";
        return "node" + idx + "\\n" + kind + " " + shape;
    }

    private static String matrixNodeLabel(RereDiffMatrix v, int idx) {
        String shape = "[" + v.value.rows() + "x" + v.value.cols() + "]";
        String kind = v.isLeaf ? "leaf" : "op";
        return "node" + idx + "\\n" + kind + " " + shape;
    }
}
