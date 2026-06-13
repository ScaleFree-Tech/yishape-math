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
            size += nodeSize(v, posMap);
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
            bulkPutDoubles(buf, g);
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

    private static int nodeSize(RereDiffTensor v, java.util.Map<RereDiffTensor, Integer> posMap) {
        int s = 2 + 2 + 4; // flags(u16) + op_len(u16) + id(u32)
        byte[] opBytes = opTag(v).getBytes(StandardCharsets.UTF_8);
        s += opBytes.length;
        s += 2; // num_dims (u16)
        int[] shape = v.serializationShape();
        s += shape.length * 4; // shape (u32[])
        s += 2; // num_inputs (u16)
        // Only count inputs that are in posMap (safe for Rust deserialization)
        if (v.inputs() != null) {
            for (RereDiffTensor inp : v.inputs()) {
                if (posMap.containsKey(inp)) s += 4; // inputs (u32[])
            }
        }

        if (!Double.isNaN(v.scalarParam())) s += 8; // f64
        if (!Double.isNaN(v.scalarParam2())) s += 8; // f64
        if (v.isLeaf()) s += 4 + v.totalSize() * 8; // data_len (u32) + data (f64[])
        if (v.backwardIndices() != null && v.backwardIndices().length > 0)
            s += 4 + v.backwardIndices().length * 4; // indices_len + indices (i32[])

        return s;
    }

    // ── Internal: node writing ──

    private static void writeNode(ByteBuffer buf, RereDiffTensor v,
                                   java.util.Map<RereDiffTensor, Integer> posMap) {
        byte[] opBytes = opTag(v).getBytes(StandardCharsets.UTF_8);
        int[] shape = v.serializationShape();

        // Flags
        int flags = (v.isLeaf() ? FLAG_IS_LEAF : 0)
                  | (v.isLeaf() ? FLAG_HAS_DATA : 0)
                  | (!Double.isNaN(v.scalarParam()) ? FLAG_HAS_SCALAR : 0)
                  | (!Double.isNaN(v.scalarParam2()) ? FLAG_HAS_PARAM2 : 0)
                  | (v.backwardIndices() != null && v.backwardIndices().length > 0 ? FLAG_HAS_INDICES : 0);
        buf.putShort((short) flags);
        buf.putShort((short) opBytes.length);
        buf.putInt(posMap.getOrDefault(v, -1)); // id = position in order
        buf.put(opBytes);
        buf.putShort((short) shape.length);
        for (int d : shape) buf.putInt(d);

        // Input references — only write inputs that are in posMap (matching JSON exporter behavior)
        if (v.inputs() != null && !v.inputs().isEmpty()) {
            int validCount = 0;
            for (RereDiffTensor inp : v.inputs()) {
                if (posMap.containsKey(inp)) validCount++;
            }
            buf.putShort((short) validCount);
            for (RereDiffTensor inp : v.inputs()) {
                if (posMap.containsKey(inp)) {
                    buf.putInt(posMap.get(inp));
                }
            }
        } else {
            buf.putShort((short) 0);
        }

        if ((flags & FLAG_HAS_SCALAR) != 0) buf.putDouble(v.scalarParam());
        if ((flags & FLAG_HAS_PARAM2) != 0) buf.putDouble(v.scalarParam2());
        if ((flags & FLAG_HAS_DATA) != 0) {
            double[] data = v.value().toDoubleArray();
            buf.putInt(data.length);
            bulkPutDoubles(buf, data);
        }
        if ((flags & FLAG_HAS_INDICES) != 0) {
            int[] indices = v.backwardIndices();
            buf.putInt(indices.length);
            bulkPutInts(buf, indices);
        }
    }

    // ── Internal: op tag resolution ──

    private static String opTag(RereDiffTensor v) {
        return v.opTag() != null ? v.opTag() : (v.isLeaf() ? "leaf" : "unknown");
    }

    // ── Helpers for Rust FFI compatibility ──

    /**
     * Converts a direct ByteBuffer to a byte array for JNI/FFM transfer.
     * The buffer position is reset after this call.
     */
    // ── Graph skeleton cache ──

    /**
     * Cached graph skeleton with leaf data offset tracking.
     * First pass records leaf positions via buffer parsing; subsequent calls
     * clone the skeleton byte[] and overwrite leaf data at known offsets,
     * avoiding topology re-serialization (~5-15ms saving per step).
     */
    public static final class CachedGraph {
        final byte[] skeleton;
        final int[] dataOffsets;  // [leafIdx] = byte offset of data_len field in skeleton
        private final int structureHash;

        CachedGraph(byte[] skeleton, int[] dataOffsets, int structureHash) {
            this.skeleton = skeleton;
            this.dataOffsets = dataOffsets;
            this.structureHash = structureHash;
        }

        public int structureHash() { return structureHash; }

        /**
         * Clone skeleton and overwrite leaf data at pre-recorded offsets.
         * @return ready-to-send byte[] (658KB clone ~0.1ms)
         */
        public byte[] updateLeafData(java.util.List<RereDiffTensor> order) {
            byte[] buf = skeleton.clone();
            int leafIdx = 0;
            for (RereDiffTensor v : order) {
                if (v.isLeaf()) {
                    double[] data = v.value().toDoubleArray();
                    int off = dataOffsets[leafIdx++];
                    // ⚠️ ByteBuffer.wrap() defaults to BIG_ENDIAN. The YSGP binary protocol
                    // uses LITTLE_ENDIAN. Without .order(LITTLE_ENDIAN), data_len is written
                    // as {0x00,0x00,0x00,0x80} instead of {0x80,0x00,0x00,0x00}, and Rust
                    // read_u32 LE interprets it as 0x80000000 (~2GB) → instant EOF crash.
                    java.nio.ByteBuffer.wrap(buf, off, 4)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN).putInt(data.length);
                    // Same LE fix for the double data values
                    java.nio.ByteBuffer db = java.nio.ByteBuffer.wrap(buf, off + 4, data.length * 8)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                    db.asDoubleBuffer().put(data);
                }
            }
            return buf;
        }
    }

    /**
     * First-pass serialization: records byte offsets of each leaf's data_len field.
     * The full buffer is serialized normally, then scanned to locate leaf data.
     */
    public static CachedGraph serializeGraphCached(RereDiffTensor root,
            java.util.List<RereDiffTensor> order, int structureHash) {
        ByteBuffer buf = serializeGraph(root, order);
        byte[] fullBytes = toByteArray(buf);

        // Count leaves
        int leafCount = 0;
        for (RereDiffTensor v : order) { if (v.isLeaf()) leafCount++; }
        int[] dataOffsets = new int[leafCount];

        // Scan buffer to find leaf data locations
        int pos = 12; // skip magic(4) + version(4) + num_nodes(4)
        int leafIdx = 0;
        for (int nodeIdx = 0; nodeIdx < order.size(); nodeIdx++) {
            int flags = ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF); pos += 2;
            int opLen = ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF); pos += 2;
            pos += 4; // id (u32)
            pos += opLen; // op string
            int numDims = ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF); pos += 2;
            pos += numDims * 4; // shape
            int numInputs = ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF); pos += 2;
            pos += numInputs * 4; // inputs
            if ((flags & FLAG_HAS_SCALAR) != 0) pos += 8; // scalar
            if ((flags & FLAG_HAS_PARAM2) != 0) pos += 8; // param2
            if ((flags & FLAG_HAS_DATA) != 0) { // has_data
                dataOffsets[leafIdx++] = pos;
                int dataLen = ((fullBytes[pos + 3] & 0xFF) << 24) | ((fullBytes[pos + 2] & 0xFF) << 16)
                            | ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF);
                pos += 4 + dataLen * 8;
            }
            if ((flags & FLAG_HAS_INDICES) != 0) { // has_indices
                int idxLen = ((fullBytes[pos + 3] & 0xFF) << 24) | ((fullBytes[pos + 2] & 0xFF) << 16)
                           | ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF);
                pos += 4 + idxLen * 4;
            }
        }
        return new CachedGraph(fullBytes, dataOffsets, structureHash);
    }

    // ── Bulk write helpers    // ── Bulk write helpers (avoid per-element loop overhead) ──

    /** Bulk-write double[] via DoubleBuffer view (single boundary check). */
    private static void bulkPutDoubles(ByteBuffer buf, double[] data) {
        int pos = buf.position();
        buf.asDoubleBuffer().put(data);
        buf.position(pos + data.length * 8);
    }

    /** Bulk-write int[] via IntBuffer view (single boundary check). */
    private static void bulkPutInts(ByteBuffer buf, int[] data) {
        int pos = buf.position();
        buf.asIntBuffer().put(data);
        buf.position(pos + data.length * 4);
    }

    public static byte[] toByteArray(ByteBuffer buf) {
        byte[] bytes = new byte[buf.remaining()];
        buf.get(bytes);
        buf.rewind();
        return bytes;
    }
}
