package com.yishape.lab.math.autodiff.graph.binary;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Tensor-native binary graph protocol — mirrors {@link BinaryProtocol} but operates on
 * {@link RereDiffTensor} nodes. Tensors carry shape information via {@code shape()} which
 * is serialized naturally, supporting N-D tensors without flattening.
 *
 * <h3>Format (all little-endian)</h3>
 * <pre>
 * Header: u32 magic="YSGP", u32 version, u32 num_nodes
 *
 * Per node:
 *   u16 flags   — see flag bits below
 *   u16 op_len  — length of op string in bytes (UTF-8)
 *   u32 id
 *   [op_len] op_string (UTF-8, not null-terminated)
 *   u16 num_dims
 *   [num_dims × u32] shape
 *   u16 num_inputs
 *   [num_inputs × u32] inputs (refer to other nodes by id)
 *   if has_input_shapes: u16 count, then per input: u16 ndim, [ndim × u32]
 *   if has_scalar: f64 (raw IEEE 754 bits)
 *   if has_param2: f64
 *   if has_data: u32 data_len, [data_len × f64]
 *   if has_indices: u32 indices_len, [indices_len × i32]
 *   [VERSION≥2 only:] u16 extension_block_len, [extension_block_len bytes]
 *
 * Result (Rust→Java):
 *   f64 loss
 *   u32 num_grads
 *   for each: u32 grad_len, [grad_len × f64]
 * </pre>
 *
 * <h3>Version policy</h3>
 * <ul>
 *   <li><b>V1</b> (wire default): fixed flag layout. Unknown flags cause parse errors on
 *       the Rust side if the crate doesn't handle them.</li>
 *   <li><b>V2</b> (opt-in via {@code -Dyishape.ysgp.version=2}): adds a trailing
 *       {@code u16 extension_block_len} per node. Extensions use inner TLV
 *       {@code (u16 type_tag, u16 data_len, data_len bytes)}. Unknown extensions
 *       are skipped safely.</li>
 * </ul>
 *
 * <h3>Flag metadata registry</h3>
 * All known flags are registered in {@link #FLAG_METADATA}. The scanner uses this
 * metadata to skip flag-dependent data correctly. Adding a new flag requires:
 * <ol>
 *   <li>Define the flag constant here</li>
 *   <li>Add a {@link FlagMeta} entry to {@link #FLAG_METADATA}</li>
 *   <li>Update {@code nodeSize()} and {@code writeNode()} to account for the new data</li>
 *   <li>Update the Rust parsers (GPU and HPC crates)</li>
 * </ol>
 */
public final class TensorBinaryProtocol {

    /** Magic number: "YSGP" in little-endian ASCII */
    public static final int MAGIC = 0x50535359;

    /** Current protocol version we can read (supports V1-V2). */
    public static final int VERSION = 2;

    /** Oldest protocol version we can read. */
    public static final int MIN_SUPPORTED_VERSION = 1;

    /**
     * Wire format version actually written. Default is 1 for backward
     * compatibility with existing Rust binaries (GPU/HPC). Opt into V2
     * (TLV extension blocks) via {@code -Dyishape.ysgp.version=2}.
     */
    /** Default wire format version (1 = backward-compat with old Rust). Overridable via system property or direct mutation for testing. */
    public static volatile int WIRE_VERSION = Integer.getInteger("yishape.ysgp.version", 1);

    // Flag bits — keep in sync with BinaryProtocol + Rust parsers
    public static final int FLAG_HAS_DATA    = 1 << 0;
    public static final int FLAG_HAS_SCALAR  = 1 << 1;
    public static final int FLAG_HAS_PARAM2  = 1 << 2;
    public static final int FLAG_HAS_INDICES = 1 << 3;
    public static final int FLAG_IS_LEAF     = 1 << 4;
    /** Input shapes are included after standard node fields (for broadcast mode selection). */
    public static final int FLAG_HAS_INPUT_SHAPES = 1 << 5;

    // ── Flag metadata registry (A2) ──

    /** Describes how the data for a flag bit is structured in the binary stream. */
    public enum FlagDataKind {
        /** No trailing data (flag is purely a marker, e.g., IS_LEAF). */
        NONE,
        /** Fixed-size trailing data (e.g., FLAG_HAS_SCALAR = 8 bytes f64). */
        FIXED,
        /** u32-prefixed variable-length f64 data (e.g., FLAG_HAS_DATA). */
        LENGTH_PREFIXED_U32,
        /** u32-prefixed variable-length i32 data (e.g., FLAG_HAS_INDICES). */
        LENGTH_PREFIXED_I32,
        /**
         * u16 count + per-input (u16 ndim + ndim×u32 shape).
         * Used by FLAG_HAS_INPUT_SHAPES for broadcast mode selection.
         */
        INPUT_SHAPES,
    }

    /** Metadata for each defined flag bit. */
    public record FlagMeta(int bit, FlagDataKind kind, int fixedLength, String description) {}

    /**
     * All known flag bits mapped by their integer value (1, 2, 4, 8, 16, 32).
     * The scanner uses this to skip flag-dependent data deterministically.
     * Unknown bits not in this map are an error in V1 and skippable in V2.
     */
    public static final Map<Integer, FlagMeta> FLAG_METADATA = Map.of(
        FLAG_HAS_DATA,         new FlagMeta(0, FlagDataKind.LENGTH_PREFIXED_U32,  0, "leaf data (f64[])"),
        FLAG_HAS_SCALAR,       new FlagMeta(1, FlagDataKind.FIXED,                8, "scalar param (f64)"),
        FLAG_HAS_PARAM2,       new FlagMeta(2, FlagDataKind.FIXED,                8, "second scalar param (f64)"),
        FLAG_HAS_INDICES,      new FlagMeta(3, FlagDataKind.LENGTH_PREFIXED_I32,  0, "backward indices (i32[])"),
        FLAG_IS_LEAF,          new FlagMeta(4, FlagDataKind.NONE,                 0, "is leaf node"),
        FLAG_HAS_INPUT_SHAPES, new FlagMeta(5, FlagDataKind.INPUT_SHAPES,         0, "input shapes for broadcast")
    );

    private TensorBinaryProtocol() {}

    // ── Graph Export ──

    /**
     * Serializes a tensor computation graph rooted at {@code root} into a binary buffer.
     * Writes {@link #WIRE_VERSION} (default V1 for backward compat with existing Rust).
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
        int[] nodeSizes = new int[order.size()];
        for (int i = 0; i < order.size(); i++) {
            nodeSizes[i] = nodeSize(order.get(i), posMap);
            size += nodeSizes[i];
        }

        ByteBuffer buf = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);

        // Header — write WIRE_VERSION for backward compat
        buf.putInt(MAGIC);
        buf.putInt(WIRE_VERSION);
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

    // ── Header validation ──

    /**
     * Validates YSGP graph binary buffer header (magic + version).
     * Delegates to {@link BinaryProtocol#validateHeader}.
     */
    public static void validateGraphHeader(ByteBuffer buf) {
        BinaryProtocol.validateHeader(buf);
        // num_nodes is read but not validated here — caller consumes it
    }

    // ── Flag-aware scanner helper (A3) ──

    /**
     * Skip flag-dependent data for one flag bit using the metadata registry.
     *
     * @param buf     the full byte array
     * @param pos     current buffer position
     * @param flagBit the flag bit value (1, 2, 4, 8, ...)
     * @param flags   the full flags u16 for this node
     * @return new buffer position after skipping this flag's data (if set)
     * @throws IllegalArgumentException if an unknown flag is set and we can't determine its size
     */
    public static int skipFlagData(byte[] buf, int pos, int flagBit, int flags, int wireVersion) {
        if ((flags & flagBit) == 0) return pos; // flag not set

        FlagMeta meta = FLAG_METADATA.get(flagBit);
        if (meta == null) {
            // Unknown flag — cannot determine data size in V1. In V2, unknown flags
            // go in the TLV extension block and we should never reach here for them.
            throw new IllegalArgumentException(
                "Unknown YSGP flag bit 0x" + Integer.toHexString(flagBit)
                + " at buffer position " + pos
                + " — protocol version " + wireVersion
                + " cannot skip unknown flag data. "
                + "Upgrade to V2 (TLV extensions) for forward-compatible flag handling.");
        }

        return switch (meta.kind()) {
            case NONE -> pos;
            case FIXED -> pos + meta.fixedLength();
            case LENGTH_PREFIXED_U32 -> {
                int len = ((buf[pos + 3] & 0xFF) << 24) | ((buf[pos + 2] & 0xFF) << 16)
                         | ((buf[pos + 1] & 0xFF) << 8) | (buf[pos] & 0xFF);
                yield pos + 4 + len * 8;
            }
            case LENGTH_PREFIXED_I32 -> {
                int len = ((buf[pos + 3] & 0xFF) << 24) | ((buf[pos + 2] & 0xFF) << 16)
                         | ((buf[pos + 1] & 0xFF) << 8) | (buf[pos] & 0xFF);
                yield pos + 4 + len * 4;
            }
            case INPUT_SHAPES -> {
                int count = ((buf[pos + 1] & 0xFF) << 8) | (buf[pos] & 0xFF);
                pos += 2;
                for (int i = 0; i < count; i++) {
                    int nd = ((buf[pos + 1] & 0xFF) << 8) | (buf[pos] & 0xFF);
                    pos += 2 + nd * 4;
                }
                yield pos;
            }
        };
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
        int validInputCount = 0;
        if (v.inputs() != null) {
            for (RereDiffTensor inp : v.inputs()) {
                if (posMap.containsKey(inp)) validInputCount++;
            }
        }
        s += validInputCount * 4; // inputs (u32[])

        // Input shapes: only for broadcast nodes (shape mismatch)
        if (v.inputs() != null) {
            boolean anyDiff = false;
            int[] vShape = v.serializationShape();
            for (RereDiffTensor inp : v.inputs()) {
                if (posMap.containsKey(inp) && !java.util.Arrays.equals(inp.shape(), vShape)) {
                    anyDiff = true;
                    break;
                }
            }
            if (anyDiff) {
                s += 2; // count field (u16)
                for (RereDiffTensor inp : v.inputs()) {
                    if (posMap.containsKey(inp)) {
                        int[] inShape = inp.shape();
                        s += 2 + inShape.length * 4;
                    }
                }
            }
        }

        if (!Double.isNaN(v.scalarParam())) s += 8; // f64
        if (!Double.isNaN(v.scalarParam2())) s += 8; // f64
        if (v.isLeaf()) s += 4 + v.totalSize() * 8; // data_len (u32) + data (f64[])
        if (v.backwardIndices() != null && v.backwardIndices().length > 0)
            s += 4 + v.backwardIndices().length * 4; // indices_len + indices (i32[])

        // V2: extension block — currently 2 bytes (u16 length = 0)
        if (WIRE_VERSION >= 2) {
            s += 2;
        }

        return s;
    }

    // ── Internal: node writing ──

    private static void writeNode(ByteBuffer buf, RereDiffTensor v,
                                   java.util.Map<RereDiffTensor, Integer> posMap) {
        byte[] opBytes = opTag(v).getBytes(StandardCharsets.UTF_8);
        int[] shape = v.serializationShape();

        // FLAG_HAS_INPUT_SHAPES: only set when at least one input shape differs
        // from the output shape — i.e., broadcasting is actually needed. Previously
        // this was set for every node with any valid input, which broke HPC Rust
        // parsers that don't know about bit 5 (unexpected EOF reading u32). GPU
        // Rust handles this flag correctly; HPC falls back to JSON on broadcast nodes.
        boolean hasInputShapes = false;
        if (v.inputs() != null) {
            for (RereDiffTensor inp : v.inputs()) {
                if (posMap.containsKey(inp)) {
                    int[] inShape = inp.shape();
                    if (!java.util.Arrays.equals(inShape, shape)) {
                        hasInputShapes = true;
                        break;
                    }
                }
            }
        }

        // Flags
        int flags = (v.isLeaf() ? FLAG_IS_LEAF : 0)
                  | (v.isLeaf() ? FLAG_HAS_DATA : 0)
                  | (!Double.isNaN(v.scalarParam()) ? FLAG_HAS_SCALAR : 0)
                  | (!Double.isNaN(v.scalarParam2()) ? FLAG_HAS_PARAM2 : 0)
                  | (v.backwardIndices() != null && v.backwardIndices().length > 0 ? FLAG_HAS_INDICES : 0)
                  | (hasInputShapes ? FLAG_HAS_INPUT_SHAPES : 0);

        int inputCount = 0;
        if (v.inputs() != null) {
            for (RereDiffTensor inp : v.inputs()) {
                if (posMap.containsKey(inp)) { inputCount++; }
            }
        }

        buf.putShort((short) flags);
        buf.putShort((short) opBytes.length);
        buf.putInt(posMap.getOrDefault(v, -1)); // id = position in order
        buf.put(opBytes);
        buf.putShort((short) shape.length);
        for (int d : shape) buf.putInt(d);

        // Input references — only write inputs that are in posMap (matching JSON exporter behavior)
        if (v.inputs() != null && !v.inputs().isEmpty()) {
            buf.putShort((short) inputCount);
            for (RereDiffTensor inp : v.inputs()) {
                if (posMap.containsKey(inp)) {
                    buf.putInt(posMap.get(inp));
                }
            }
        } else {
            buf.putShort((short) 0);
        }

        // Input shapes (for GPU/HPC broadcast mode selection)
        if (hasInputShapes) {
            buf.putShort((short) inputCount);
            for (RereDiffTensor inp : v.inputs()) {
                if (posMap.containsKey(inp)) {
                    int[] inShape = inp.shape();
                    buf.putShort((short) inShape.length);
                    for (int d : inShape) buf.putInt(d);
                }
            }
        }

        if ((flags & FLAG_HAS_SCALAR) != 0) { buf.putDouble(v.scalarParam()); }
        if ((flags & FLAG_HAS_PARAM2) != 0) { buf.putDouble(v.scalarParam2()); }
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

        // V2: extension block (TLV container for future flag additions)
        if (WIRE_VERSION >= 2) {
            buf.putShort((short) 0); // extension_block_length = 0 (no extensions yet)
        }
    }

    // ── Internal: op tag resolution ──

    private static String opTag(RereDiffTensor v) {
        return v.opTag() != null ? v.opTag() : (v.isLeaf() ? "leaf" : "unknown");
    }

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
                    if (leafIdx >= dataOffsets.length) {
                        throw new IllegalStateException(
                            "CachedGraph leaf count mismatch: graph has more leaves than cached skeleton");
                    }
                    double[] data = v.value().toDoubleArray();
                    int off = dataOffsets[leafIdx++];
                    int dataEnd = off + 4 + data.length * 8;
                    if (dataEnd > buf.length) {
                        throw new IllegalStateException(
                            "CachedGraph buffer overflow: leaf data (" + data.length
                            + " elements, " + dataEnd + " bytes) exceeds skeleton buffer ("
                            + buf.length + " bytes). Leaf tensor size increased since caching.");
                    }
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

        // Validate header
        int pos = 0;
        int magic = ((fullBytes[pos + 3] & 0xFF) << 24) | ((fullBytes[pos + 2] & 0xFF) << 16)
                  | ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF);
        if (magic != MAGIC) {
            throw new BinaryProtocol.ProtocolVersionException(
                "CachedGraph scanner: bad magic 0x" + Integer.toHexString(magic));
        }
        pos += 4;
        int wireVersion = ((fullBytes[pos + 3] & 0xFF) << 24) | ((fullBytes[pos + 2] & 0xFF) << 16)
                        | ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF);
        pos += 4;
        int numNodes = ((fullBytes[pos + 3] & 0xFF) << 24) | ((fullBytes[pos + 2] & 0xFF) << 16)
                      | ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF);
        pos += 4;

        // Count leaves
        int leafCount = 0;
        for (RereDiffTensor v : order) { if (v.isLeaf()) leafCount++; }
        int[] dataOffsets = new int[leafCount];

        // Scan buffer using flag metadata registry (A3)
        int leafIdx = 0;
        for (int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
            int flags = ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF); pos += 2;
            int opLen = ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF); pos += 2;
            pos += 4; // id (u32)
            pos += opLen; // op string
            int numDims = ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF); pos += 2;
            pos += numDims * 4; // shape
            int numInputs = ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF); pos += 2;
            pos += numInputs * 4; // inputs

            // Skip flag-dependent data using the metadata registry for all 16 bits
            for (int bit = 0; bit < 16; bit++) {
                int flagVal = 1 << bit;
                // FLAG_HAS_DATA needs special handling to capture leaf offsets
                if (flagVal == FLAG_HAS_DATA && (flags & FLAG_HAS_DATA) != 0) {
                    dataOffsets[leafIdx++] = pos;
                }
                pos = skipFlagData(fullBytes, pos, flagVal, flags, wireVersion);
            }

            // V2: skip extension block
            if (wireVersion >= 2) {
                int extLen = ((fullBytes[pos + 1] & 0xFF) << 8) | (fullBytes[pos] & 0xFF);
                pos += 2 + extLen;
            }
        }
        return new CachedGraph(fullBytes, dataOffsets, structureHash);
    }

    // ── Bulk write helpers (avoid per-element loop overhead) ──

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
