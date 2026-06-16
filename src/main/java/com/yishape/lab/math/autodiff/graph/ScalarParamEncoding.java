package com.yishape.lab.math.autodiff.graph;

/**
 * Bit-packed encoding schemes for {@code scalarParam} and {@code scalarParam2}
 * double-precision fields on AD graph nodes.
 *
 * <h3>Why bit-packing in doubles?</h3>
 * The YSGP binary protocol reserves two {@code f64} fields per node for op
 * parameters. Rather than add a variable-length parameter block, integer
 * parameters are packed into the IEEE 754 bits of these doubles via
 * {@link Double#longBitsToDouble}. This keeps the node header fixed-size
 * (critical for HPC/GPU zero-copy buffer access) while carrying all needed
 * parameters.
 *
 * <h3>Cross-backend contract</h3>
 * Every encoding scheme here MUST match the decoding in the corresponding
 * Rust backend:
 * <ul>
 *   <li>GPU: {@code yishape_math_gpu/src/ops/graph.rs} — {@code forward_dispatch}</li>
 *   <li>HPC: {@code yishape_math_rust/src/graph.rs} — {@code forward_node}</li>
 * </ul>
 * Changes to any shift/mask MUST be synchronized across Java, GPU Rust,
 * and HPC Rust.
 *
 * <h3>Usage</h3>
 * <pre>
 *   // Packing
 *   node.setScalarParam(Pooling.packScalarParam(kH, kW, stride));
 *   node.setScalarParam2(Pooling.packScalarParam2(pad));
 *
 *   // Unpacking (e.g. in ExportShapeValidator)
 *   long bits = Double.doubleToRawLongBits(node.scalarParam());
 *   int kH = Pooling.unpackKH(bits);
 *   int kW = Pooling.unpackKW(bits);
 *   int stride = Pooling.unpackStride(bits);
 * </pre>
 */
public final class ScalarParamEncoding {

    private ScalarParamEncoding() {}

    /**
     * Encoding for pooling ops: maxpool2d, avgpool2d.
     *
     * <pre>
     * scalarParam  (64 bits):
     *   [unused:24] [kH:16] [kW:8] [stride:8]
     *
     * scalarParam2 (64 bits):
     *   [unused:48] [pad:16]
     * </pre>
     */
    public static final class Pooling {
        private Pooling() {}

        // scalarParam layout
        public static final int KH_SHIFT   = 16;
        public static final int KW_SHIFT   = 8;
        public static final int STRIDE_SHIFT = 0;
        public static final long KH_MASK   = 0xFFFFL;
        public static final long KW_MASK   = 0xFFL;
        public static final long STRIDE_MASK = 0xFFL;

        // scalarParam2 layout
        public static final int PAD_SHIFT  = 16;
        public static final long PAD_MASK  = 0xFFFFL;

        public static double packScalarParam(int kH, int kW, int stride) {
            return Double.longBitsToDouble(
                ((long) kH << KH_SHIFT) | ((long) kW << KW_SHIFT) | ((long) stride));
        }

        public static double packScalarParam2(int pad) {
            return Double.longBitsToDouble(((long) pad << PAD_SHIFT));
        }

        public static int unpackKH(long bits)       { return (int) ((bits >>> KH_SHIFT) & KH_MASK); }
        public static int unpackKW(long bits)       { return (int) ((bits >>> KW_SHIFT) & KW_MASK); }
        public static int unpackStride(long bits)   { return (int) (bits & STRIDE_MASK); }
        public static int unpackPad(long bits2)     { return (int) ((bits2 >>> PAD_SHIFT) & PAD_MASK); }
    }

    /**
     * Encoding for conv2d.
     *
     * <pre>
     * scalarParam  (64 bits):
     *   [dilation-1:24] [kH:16] [kW:8] [stride:8]
     *
     * scalarParam2 (64 bits):
     *   [unused:32] [pad:16] [outC:16]
     * </pre>
     */
    public static final class Conv2d {
        private Conv2d() {}

        // scalarParam layout
        public static final int DIL_SHIFT   = 40;
        public static final int KH_SHIFT    = 16;
        public static final int KW_SHIFT    = 8;
        public static final int STRIDE_SHIFT = 0;
        public static final long DIL_MASK   = 0xFFFFFFL;  // 24 bits for dilation-1
        public static final long KH_MASK    = 0xFFFFL;
        public static final long KW_MASK    = 0xFFL;
        public static final long STRIDE_MASK = 0xFFL;

        // scalarParam2 layout
        public static final int PAD_SHIFT   = 16;
        public static final int OUTC_SHIFT  = 0;
        public static final long PAD_MASK   = 0xFFFFL;
        public static final long OUTC_MASK  = 0xFFFFL;

        public static double packScalarParam(int dilation, int kH, int kW, int stride) {
            return Double.longBitsToDouble(
                ((long)(dilation - 1) << DIL_SHIFT) | ((long) kH << KH_SHIFT)
                | ((long) kW << KW_SHIFT) | ((long) stride));
        }

        public static double packScalarParam2(int pad, int outC) {
            return Double.longBitsToDouble(
                ((long) pad << PAD_SHIFT) | ((long) outC & OUTC_MASK));
        }

        public static int unpackDilation(long bits) { return (int) ((bits >>> DIL_SHIFT) & DIL_MASK) + 1; }
        public static int unpackKH(long bits)       { return (int) ((bits >>> KH_SHIFT) & KH_MASK); }
        public static int unpackKW(long bits)       { return (int) ((bits >>> KW_SHIFT) & KW_MASK); }
        public static int unpackStride(long bits)   { return (int) (bits & STRIDE_MASK); }
        public static int unpackPad(long bits2)     { return (int) ((bits2 >>> PAD_SHIFT) & PAD_MASK); }
        public static int unpackOutC(long bits2)    { return (int) (bits2 & OUTC_MASK); }
    }

    /**
     * Encoding for depthwiseConv1d.
     *
     * <pre>
     * scalarParam  (64 bits):
     *   [L:48] [kSize:16]
     *
     * scalarParam2 (64 bits):
     *   C (channel count, stored as raw double)
     * </pre>
     */
    public static final class DepthwiseConv1d {
        private DepthwiseConv1d() {}

        public static final int L_SHIFT = 16;
        public static final int KSIZE_SHIFT = 0;
        public static final long L_MASK = 0xFFFF_FFFF_FFFFL;
        public static final long KSIZE_MASK = 0xFFFFL;

        public static double packScalarParam(int L, int kSize) {
            return Double.longBitsToDouble(((long) L << L_SHIFT) | ((long) kSize));
        }

        public static int unpackL(long bits)     { return (int) ((bits >>> L_SHIFT) & L_MASK); }
        public static int unpackKSize(long bits) { return (int) (bits & KSIZE_MASK); }
    }

    /**
     * Encoding for gridSample.
     *
     * <pre>
     * scalarParam  (64 bits):
     *   [unused:32] [H:16] [W:16]
     *
     * scalarParam2 (64 bits):
     *   [unused:48] [padMode:8] [mode:8]
     * </pre>
     */
    public static final class GridSample {
        private GridSample() {}

        public static final int H_SHIFT   = 16;
        public static final int W_SHIFT   = 0;
        public static final long H_MASK   = 0xFFFFL;
        public static final long W_MASK   = 0xFFFFL;

        public static final int PADMODE_SHIFT = 8;
        public static final int MODE_SHIFT    = 0;
        public static final long PADMODE_MASK = 0xFFL;
        public static final long MODE_MASK    = 0xFFL;

        public static double packScalarParam(int H, int W) {
            return Double.longBitsToDouble(((long) H << H_SHIFT) | ((long) W));
        }

        public static double packScalarParam2(int padModeIdx, int modeIdx) {
            return Double.longBitsToDouble(
                ((long) padModeIdx << PADMODE_SHIFT) | ((long) modeIdx));
        }

        public static int unpackH(long bits)        { return (int) ((bits >>> H_SHIFT) & H_MASK); }
        public static int unpackW(long bits)        { return (int) (bits & W_MASK); }
        public static int unpackPadMode(long bits2) { return (int) ((bits2 >>> PADMODE_SHIFT) & PADMODE_MASK); }
        public static int unpackMode(long bits2)    { return (int) (bits2 & MODE_MASK); }
    }

    /**
     * Encoding for interpolate.
     *
     * <pre>
     * scalarParam  (64 bits):
     *   [H:32] [W:32]
     *
     * scalarParam2 (64 bits):
     *   0.0 = bilinear, 1.0 = nearest
     * </pre>
     */
    public static final class Interpolate {
        private Interpolate() {}

        public static final int H_SHIFT = 32;
        public static final int W_SHIFT = 0;
        public static final long H_MASK = 0xFFFF_FFFFL;
        public static final long W_MASK = 0xFFFF_FFFFL;

        public static double packScalarParam(int H, int W) {
            return Double.longBitsToDouble(((long) H << H_SHIFT) | ((long) W & W_MASK));
        }

        public static double packScalarParam2(boolean bilinear) {
            return bilinear ? 0.0 : 1.0;
        }

        public static int unpackH(long bits) { return (int) ((bits >>> H_SHIFT) & H_MASK); }
        public static int unpackW(long bits) { return (int) (bits & W_MASK); }
    }

    /**
     * Encoding for trapezoidalScan.
     *
     * <pre>
     * scalarParam  (64 bits):
     *   [AIsVec:62] [DIsScalar:1] [DeltaBroadcast:1]
     * </pre>
     */
    public static final class TrapezoidalScan {
        private TrapezoidalScan() {}

        public static final int A_IS_VEC_SHIFT     = 2;
        public static final int D_IS_SCALAR_SHIFT  = 1;
        public static final int DELTA_BCAST_SHIFT  = 0;

        public static double packScalarParam(boolean aIsVec, boolean dIsScalar, boolean deltaBroadcast) {
            return Double.longBitsToDouble(
                ((long) (aIsVec ? 1 : 0) << A_IS_VEC_SHIFT)
                | ((long) (dIsScalar ? 1 : 0) << D_IS_SCALAR_SHIFT)
                | (long) (deltaBroadcast ? 1 : 0));
        }

        public static boolean unpackAIsVec(long bits)      { return ((bits >>> A_IS_VEC_SHIFT) & 1) != 0; }
        public static boolean unpackDIsScalar(long bits)   { return ((bits >>> D_IS_SCALAR_SHIFT) & 1) != 0; }
        public static boolean unpackDeltaBroadcast(long bits) { return (bits & 1) != 0; }
    }

    /**
     * Encoding for reduce ops (sum/mean with dim argument).
     *
     * <pre>
     * scalarParam (64 bits):
     *   inner stride (raw integer stored as double bit pattern)
     *   NaN → flat reduce (stride = 1, contiguous)
     *
     * scalarParam2 (only for fused ops like reluMean):
     *   [unused:63] [keepdim:1] — or raw keepdim flag
     * </pre>
     *
     * <p>In fused reduction ops ({@code TensorFusedReductionOps}), scalarParam
     * stores the reduction dimension index, and scalarParam2 stores the keepdim
     * flag (1.0 = keep, 0.0 = don't keep).</p>
     */
    public static final class Reduce {
        private Reduce() {}

        /**
         * Returns the inner stride from a sum/mean node's scalarParam.
         * NaN → flat reduce (all elements contiguous).
         */
        public static int unpackInnerStride(double scalarParam) {
            return Double.isNaN(scalarParam) ? 1 : (int) scalarParam;
        }

        /** Pack inner stride as scalarParam. */
        public static double packInnerStride(int inner) {
            return (double) inner;
        }

        /** Pack reduction dimension for fused ops. */
        public static double packReduceDim(int dim) {
            return (double) dim;
        }

        /** Unpack reduction dimension from fused op scalarParam. */
        public static int unpackReduceDim(double scalarParam) {
            return (int) scalarParam;
        }

        /** Pack keepdim flag (1.0 = true, 0.0 = false). */
        public static double packKeepDim(boolean keepdim) {
            return keepdim ? 1.0 : 0.0;
        }

        /** Unpack keepdim flag. */
        public static boolean unpackKeepDim(double scalarParam2) {
            return scalarParam2 != 0.0;
        }
    }
}
