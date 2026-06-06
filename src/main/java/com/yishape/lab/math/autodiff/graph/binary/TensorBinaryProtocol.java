package com.yishape.lab.math.autodiff.graph.binary;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Tensor-native binary graph protocol — mirrors {@link BinaryProtocol} but operates on
 * {@link RereDiffTensor} nodes. Tensors carry shape information via {@code shape()} which
 * is serialized naturally, supporting N-D tensors without flattening.
 *
 * <p>Format (identical to BinaryProtocol — all little-endian):
 * <pre>
 * Header: u32 magic="YSGP", u32 version=1, u32 num_nodes
 *
 * Per node:
 *   u16 flags   — bit0=has_data, bit1=has_scalar, bit2=has_param2, bit3=has_indices, bit4=is_leaf
 *   u16 op_len  — length of op string in bytes (UTF-8)
 *   u32 id
 *   [op_len] op_string (UTF-8, not null-terminated)
 *   u16 num_dims
 *   [num_dims × u32] shape
 *   u16 num_inputs
 *   [num_inputs × u32] inputs (refer to other nodes by id)
 *   if has_scalar: f64 (raw IEEE 754 bits)
 *   if has_param2: f64
 *   if has_data: u32 data_len, [data_len × f64]
 *   if has_indices: u32 indices_len, [indices_len × i32]
 *
 * Result (Rust→Java):
 *   f64 loss
 *   u32 num_grads
 *   for each: u32 grad_len, [grad_len × f64]
 * </pre>
 */
public final class TensorBinaryProtocol {

    /** Magic number: "YSGP" in little-endian ASCII */
    public static final int MAGIC = 0x50535359;

    public static final int VERSION = 1;

    // Flag bits
    public static final int FLAG_HAS_DATA    = 1 << 0;
    public static final int FLAG_HAS_SCALAR  = 1 << 1;
    public static final int FLAG_HAS_PARAM2  = 1 << 2;
    public static final int FLAG_HAS_INDICES = 1 << 3;
    public static final int FLAG_IS_LEAF     = 1 << 4;

    private TensorBinaryProtocol() {}

    // ── Graph Export ──

    /**
     * Serializes a tensor computation graph rooted at {@code root} into a binary buffer.
     *
     * @param root   the root of the computation graph
     * @param order  nodes in topological order (output positions = indices in this list)
     * @return ByteBuffer containing the serialized graph
     */
    public static ByteBuffer serializeGraph(RereDiffTensor root, List<RereDiffTensor> order) {
        java.util.Map<RereDiffTensor, Integer> posMap = new java.util.HashMap<>();
        for (int i = 0; i < order.size(); i++) posMap.put(order.get(i), i);

        // Pass 1: compute total size
        int size = 12; // header: magic + version + num_nodes
        for (RereDiffTensor v : order) {
            size += nodeSize(v);
        }

        ByteBuffer buf = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);

        // Header
        buf.putInt(MAGIC);
        buf.putInt(VERSION);
        buf.putInt(order.size());

        // Nodes
        for (RereDiffTensor v : order) {
            writeNode(buf, v, posMap);
        }

        buf.flip();
        return buf;
    }

    // ── Result Serialization (shared with BinaryProtocol) ──

    /**
     * Serializes graph execution result (loss + gradients) into a binary buffer.
     */
    public static ByteBuffer serializeResult(double loss, List<double[]> grads) {
        int size = 8 + 4; // loss (f64) + num_grads (u32)
        for (double[] g : grads) {
            size += 4 + g.length * 8; // grad_len (u32) + data (f64[])
        }

        ByteBuffer buf = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        buf.putDouble(loss);
        buf.putInt(grads.size());
        for (double[] g : grads) {
            buf.putInt(g.length);
            for (double v : g) buf.putDouble(v);
        }
        buf.flip();
        return buf;
    }

    // ── Result Deserialization ──

    /**
     * Parses a binary result buffer into loss + gradients.
     * Used on the Java side to read GPU/HPC results.
     */
    public static BinaryProtocol.GraphResult deserializeResult(ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        double loss = buf.getDouble();
        int numGrads = buf.getInt();
        List<double[]> grads = new ArrayList<>(numGrads);
        for (int i = 0; i < numGrads; i++) {
            int len = buf.getInt();
            double[] g = new double[len];
            for (int j = 0; j < len; j++) g[j] = buf.getDouble();
            grads.add(g);
        }
        return new BinaryProtocol.GraphResult(loss, grads);
    }

    // ── Internal: node sizing ──

    private static int nodeSize(RereDiffTensor v) {
        int s = 2 + 2 + 4; // flags(u16) + op_len(u16) + id(u32)
        byte[] opBytes = opTag(v).getBytes(StandardCharsets.UTF_8);
        s += opBytes.length;
        s += 2; // num_dims (u16)
        int[] shape = v.exportShape != null ? v.exportShape : v.shape();
        s += shape.length * 4; // shape (u32[])
        s += 2; // num_inputs (u16)
        if (v.inputs != null) s += v.inputs.size() * 4; // inputs (u32[])

        if (!Double.isNaN(v.scalarParam)) s += 8; // f64
        if (!Double.isNaN(v.scalarParam2)) s += 8; // f64
        if (v.isLeaf) s += 4 + v.totalSize() * 8; // data_len (u32) + data (f64[])
        if (v.backwardIndices != null && v.backwardIndices.length > 0)
            s += 4 + v.backwardIndices.length * 4; // indices_len + indices (i32[])

        return s;
    }

    // ── Internal: node writing ──

    private static void writeNode(ByteBuffer buf, RereDiffTensor v,
                                   java.util.Map<RereDiffTensor, Integer> posMap) {
        byte[] opBytes = opTag(v).getBytes(StandardCharsets.UTF_8);
        int[] shape = v.exportShape != null ? v.exportShape : v.shape();

        // Flags
        int flags = (v.isLeaf ? FLAG_IS_LEAF : 0)
                  | (v.isLeaf ? FLAG_HAS_DATA : 0)
                  | (!Double.isNaN(v.scalarParam) ? FLAG_HAS_SCALAR : 0)
                  | (!Double.isNaN(v.scalarParam2) ? FLAG_HAS_PARAM2 : 0)
                  | (v.backwardIndices != null && v.backwardIndices.length > 0 ? FLAG_HAS_INDICES : 0);
        buf.putShort((short) flags);
        buf.putShort((short) opBytes.length);
        buf.putInt(posMap.getOrDefault(v, -1)); // id = position in order
        buf.put(opBytes);
        buf.putShort((short) shape.length);
        for (int d : shape) buf.putInt(d);

        // Input references with actual positions
        if (v.inputs != null && !v.inputs.isEmpty()) {
            buf.putShort((short) v.inputs.size());
            for (RereDiffTensor inp : v.inputs) {
                int inpPos = posMap.getOrDefault(inp, -1);
                buf.putInt(inpPos);
            }
        } else {
            buf.putShort((short) 0);
        }

        if ((flags & FLAG_HAS_SCALAR) != 0) buf.putDouble(v.scalarParam);
        if ((flags & FLAG_HAS_PARAM2) != 0) buf.putDouble(v.scalarParam2);
        if ((flags & FLAG_HAS_DATA) != 0) {
            double[] data = v.value.toDoubleArray();
            buf.putInt(data.length);
            for (double d : data) buf.putDouble(d);
        }
        if ((flags & FLAG_HAS_INDICES) != 0) {
            buf.putInt(v.backwardIndices.length);
            for (int idx : v.backwardIndices) buf.putInt(idx);
        }
    }

    // ── Internal: op tag resolution ──

    private static String opTag(RereDiffTensor v) {
        return v.opTag != null ? v.opTag : (v.isLeaf ? "leaf" : "unknown");
    }

    // ── Helpers for Rust FFI compatibility ──

    /**
     * Converts a direct ByteBuffer to a byte array for JNI/FFM transfer.
     * The buffer position is reset after this call.
     */
    public static byte[] toByteArray(ByteBuffer buf) {
        byte[] bytes = new byte[buf.remaining()];
        buf.get(bytes);
        buf.rewind();
        return bytes;
    }
}
