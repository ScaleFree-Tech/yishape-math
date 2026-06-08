package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Validates that tensor graph nodes carry sufficient shape metadata for
 * HPC/GPU backend execution BEFORE serialization.
 *
 * <h3>Purpose</h3>
 * The HPC (faer) and GPU (wgpu) backends derive input dimensions from the
 * serialized shape. For ops like maxpool2d, a bare output shape (4D) forces
 * the backend to reverse-engineer input dims via {@code (out-1)*stride+kh-2*pad},
 * which is wrong when stride doesn't divide the input evenly. This validator
 * catches such cases at export time — before a silent wrong-result bug reaches
 * production.
 *
 * <h3>Usage</h3>
 * <pre>
 *   ExportShapeValidator.Result r = ExportShapeValidator.validate(order);
 *   if (r.hasErrors()) {
 *       throw new IllegalStateException("Shape validation failed:\n" + r);
 *   }
 *   if (r.hasWarnings()) {
 *       log.warn("Shape warnings:\n{}", r);
 *   }
 * </pre>
 *
 * <h3>Enforcement policy</h3>
 * <ul>
 *   <li><b>ERROR</b> — serialization shape would cause the backend to produce
 *       wrong results (e.g. derived input dim ≠ actual). Export blocked.</li>
 *   <li><b>WARN</b> — shape is technically correct but fragile; recommend
 *       using richer metadata for robustness across stride/edge cases.</li>
 * </ul>
 */
public final class ExportShapeValidator {

    private ExportShapeValidator() {}

    // ── Public API ──

    /**
     * Validates all nodes in a topological graph order before backend export.
     *
     * @param order nodes in topological order (index = node id)
     * @return validation result with errors and warnings
     */
    public static Result validate(List<RereDiffTensor> order) {
        Result result = new Result();
        for (int i = 0; i < order.size(); i++) {
            RereDiffTensor v = order.get(i);
            String op = v.opTag();
            if (op == null) continue;

            switch (op) {
                case "maxpool2d":
                    validateMaxPool2d(v, order, i, result);
                    break;
                case "conv2d":
                    validateConv2d(v, result);
                    break;
                default:
                    // No shape-metadata requirements for other ops
                    break;
            }
        }
        return result;
    }

    /**
     * Validates and throws on errors. Convenience wrapper for call sites
     * that want fail-fast behavior.
     *
     * @throws IllegalStateException if validation errors are found
     */
    public static void validateOrThrow(List<RereDiffTensor> order) {
        Result r = validate(order);
        if (r.hasErrors()) {
            throw new IllegalStateException(
                "Graph export blocked by ExportShapeValidator — " +
                r.errors().size() + " error(s):\n" + r);
        }
    }

    // ── Per-op validators ──

    private static void validateMaxPool2d(RereDiffTensor node, List<RereDiffTensor> order,
                                          int nodeIdx, Result result) {
        int[] shape = node.serializationShape();
        long bits = Double.doubleToRawLongBits(node.scalarParam());
        int kh = (int) ((bits >> 16) & 0xFF);
        int stride = (int) (bits & 0xFF);
        long bits2 = Double.doubleToRawLongBits(node.scalarParam2());
        int pad = (int) ((bits2 >> 16) & 0xFFFF);

        if (shape.length >= 6) {
            // 6D: [B, C, inH, inW, outH, outW] — fully specified, verify consistency
            int B = shape[0], C = shape[1];
            int inH = shape[2], inW = shape[3];
            int outH = shape[4], outW = shape[5];

            // Verify derived out dims match the 6D out dims
            int expectedOutH = (inH + 2 * pad - kh) / stride + 1;
            int expectedOutW = (inW + 2 * pad - kh) / stride + 1;
            if (expectedOutH != outH || expectedOutW != outW) {
                result.addError(String.format(
                    "node#%d maxpool2d: 6D shape out dims [%d,%d] don't match derived [%d,%d] " +
                    "from inH=%d inW=%d kh=%d stride=%d pad=%d. " +
                    "shape=%s",
                    nodeIdx, outH, outW, expectedOutH, expectedOutW,
                    inH, inW, kh, stride, pad, Arrays.toString(shape)));
            } else {
                // Also verify shape matches input buffer size
                int inputSize = (int) node.totalSize();
                int expectedInputSize = B * C * inH * inW;
                // Note: totalSize is output size, not input. We need input node's size.
                verifyInputBufferSize(node, order, B, C, inH, inW, nodeIdx, result);
            }
        } else {
            // 4D: [B, C, outH, outW] — backend will derive inH/inW
            if (shape.length < 4) {
                result.addError(String.format(
                    "node#%d maxpool2d: shape must be at least 4D [B,C,outH,outW], got %dD: %s",
                    nodeIdx, shape.length, Arrays.toString(shape)));
                return;
            }
            int B = shape[0], C = shape[1];
            int outH = shape[2], outW = shape[3];

            // Derive what the backend WILL compute for input dims
            int derivedInH = (outH - 1) * stride + kh - 2 * pad;
            int derivedInW = (outW - 1) * stride + kh - 2 * pad;

            // Find the actual input size from the input node
            int actualInputSize = getInputBufferSize(node, order);
            if (actualInputSize > 0) {
                int expectedByDerived = B * C * derivedInH * derivedInW;
                if (expectedByDerived != actualInputSize) {
                    result.addError(String.format(
                        "node#%d maxpool2d: 4D shape causes backend to derive " +
                        "inH=%d inW=%d (via (out-1)*stride+kh-2*pad), but actual input " +
                        "buffer has %d elements (B=%d C=%d). Expected %d from derived dims, got %d. " +
                        "The input dimensions are NOT evenly divisible by stride=%d. " +
                        "Fix: set exportShape to [B,C,inH,inW,outH,outW] 6D format.%n" +
                        "  shape=%s",
                        nodeIdx, derivedInH, derivedInW, actualInputSize,
                        B, C, expectedByDerived, actualInputSize,
                        stride, Arrays.toString(shape)));
                } else {
                    // Derived matches, but still fragile — suggest 6D
                    result.addWarning(String.format(
                        "node#%d maxpool2d: using 4D shape (derived inH=%d inW=%d matches input). " +
                        "Consider setting exportShape to 6D [B,C,inH,inW,outH,outW] for robustness.",
                        nodeIdx, derivedInH, derivedInW));
                }
            }
        }
    }

    private static void validateConv2d(RereDiffTensor node, Result result) {
        int[] shape = node.serializationShape();
        if (shape.length < 4) {
            result.addError(String.format(
                "conv2d: shape must be at least 4D [B,outC,outH,outW], got %dD: %s",
                shape.length, Arrays.toString(shape)));
        }
    }

    // ── Helpers ──

    /**
     * Gets the total element count of the first input node's buffer.
     * Returns 0 if the input cannot be resolved.
     */
    private static int getInputBufferSize(RereDiffTensor node, List<RereDiffTensor> order) {
        if (node.inputs() == null || node.inputs().isEmpty()) return 0;
        RereDiffTensor input = node.inputs().get(0);
        if (input == null) return 0;
        return (int) input.totalSize();
    }

    private static void verifyInputBufferSize(RereDiffTensor node, List<RereDiffTensor> order,
                                               int B, int C, int inH, int inW,
                                               int nodeIdx, Result result) {
        int actualInputSize = getInputBufferSize(node, order);
        if (actualInputSize <= 0) return;
        int expected = B * C * inH * inW;
        if (expected != actualInputSize) {
            result.addError(String.format(
                "node#%d maxpool2d: 6D shape claims inH=%d inW=%d (expected input size=%d), " +
                "but actual input node buffer has %d elements. Mismatch=%d. " +
                "Check that exportShape inH/inW match the actual input dimensions.",
                nodeIdx, inH, inW, expected, actualInputSize, Math.abs(expected - actualInputSize)));
        }
    }

    // ── Result type ──

    public static class Result {
        private final List<String> errors = new ArrayList<>();
        private final List<String> warnings = new ArrayList<>();

        void addError(String msg) { errors.add(msg); }
        void addWarning(String msg) { warnings.add(msg); }

        public boolean hasErrors() { return !errors.isEmpty(); }
        public boolean hasWarnings() { return !warnings.isEmpty(); }
        public List<String> errors() { return errors; }
        public List<String> warnings() { return warnings; }

        public boolean isValid() { return !hasErrors(); }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            if (!errors.isEmpty()) {
                sb.append("ERRORS (").append(errors.size()).append("):\n");
                for (int i = 0; i < errors.size(); i++) {
                    sb.append("  [E").append(i + 1).append("] ").append(errors.get(i)).append('\n');
                }
            }
            if (!warnings.isEmpty()) {
                sb.append("WARNINGS (").append(warnings.size()).append("):\n");
                for (int i = 0; i < warnings.size(); i++) {
                    sb.append("  [W").append(i + 1).append("] ").append(warnings.get(i)).append('\n');
                }
            }
            if (errors.isEmpty() && warnings.isEmpty()) {
                sb.append("VALID (no issues)");
            }
            return sb.toString().stripTrailing();
        }
    }
}
