package com.yishape.lab.math.autodiff.graph.binary;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;


/**
 * Zero-dependency binary graph protocol replacing JSON for GPU/HPC graph execution.
 *
 * <h3>Format (all little-endian)</h3>
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
 *   if has_scalar: f64 (raw IEEE 754 bits — EXACT, no subnormal loss!)
 *   if has_param2: f64
 *   if has_data: u32 data_len, [data_len × f64]
 *   if has_indices: u32 indices_len, [indices_len × i32]
 *
 * Result (Rust→Java):
 *   f64 loss
 *   u32 num_grads
 *   for each: u32 grad_len, [grad_len × f64]
 * </pre>
 *
 * <h3>Key advantages over JSON</h3>
 * - scalar params stored as raw f64 bits → subnormal doubles survive round-trip perfectly
 * - No string parsing overhead → 10-100× faster
 * - Fixed offsets for fast field access
 * - No NaN/Infinity serialization issues
 * - Zero external dependencies
 */
public final class BinaryProtocol {

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
    public static volatile int WIRE_VERSION = Integer.getInteger("yishape.ysgp.version", 1);

    // Flag bits
    public static final int FLAG_HAS_DATA    = 1 << 0;
    public static final int FLAG_HAS_SCALAR  = 1 << 1;
    public static final int FLAG_HAS_PARAM2  = 1 << 2;
    public static final int FLAG_HAS_INDICES = 1 << 3;
    public static final int FLAG_IS_LEAF     = 1 << 4;

    private BinaryProtocol() {}

    // ── Result Serialization ──

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

    // ── Header Validation ──

    /**
     * Validates a YSGP binary buffer header (magic + version).
     * Call before {@link #deserializeResult} to fail-fast on corrupt/mismatched data.
     *
     * @throws ProtocolVersionException if magic or version is unsupported
     */
    public static void validateHeader(ByteBuffer buf) {
        buf.order(ByteOrder.LITTLE_ENDIAN);
        int magic = buf.getInt();
        if (magic != MAGIC) {
            throw new ProtocolVersionException(
                String.format("Invalid YSGP magic: expected 0x%08X, got 0x%08X", MAGIC, magic));
        }
        int version = buf.getInt();
        if (version < MIN_SUPPORTED_VERSION || version > VERSION) {
            throw new ProtocolVersionException(
                String.format("Unsupported YSGP protocol version %d (supported: %d-%d)",
                    version, MIN_SUPPORTED_VERSION, VERSION));
        }
    }

    /**
     * Thrown when a YSGP binary buffer has an unsupported magic or version.
     */
    public static final class ProtocolVersionException extends IllegalArgumentException {
        public ProtocolVersionException(String message) { super(message); }
    }

    // ── Result Deserialization ──

    /**
     * Parses a binary result buffer into loss + gradients.
     * Used on the Java side to read GPU/HPC results.
     */
    public static GraphResult deserializeResult(ByteBuffer buf) {
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
        return new GraphResult(loss, grads);
    }

    public record GraphResult(double loss, List<double[]> grads) {}

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
