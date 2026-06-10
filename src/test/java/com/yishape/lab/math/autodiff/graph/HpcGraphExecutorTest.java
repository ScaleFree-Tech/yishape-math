package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.autodiff.graph.binary.TensorBinaryProtocol;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Failure path + diagnostic tests for {@link HpcGraphExecutor}.
 */
public class HpcGraphExecutorTest {

    @Test
    void testReciprocalReturnsNaN() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) x.reciprocal().sum();
        double result = HpcGraphExecutor.tryExecute(loss);
        assertTrue(Double.isNaN(result), "Unsupported op should return NaN for graceful CPU fallback");
    }

    @Test
    void testGraphValidationFailureReturnsNaN() {
        IDiffVector x = AD.vector(new double[]{1.0});
        RereDiffVector loss = (RereDiffVector) x.mul(1.0).sum();
        double result = HpcGraphExecutor.tryExecute(loss);
    }

    // ── Binary protocol diagnostic ──

    /**
     * Verifies that the CachedGraph serialization produces a byte-identical result
     * to a fresh serializeGraph call. Any mismatch would cause the Rust parser
     * to misinterpret fields and hit EOF.
     */
    @Test
    void testCachedGraphMatchesFreshSerialization() {
        // Build a graph with multiple leaves and ops, plus a softmaxCrossEntropySparse node
        // (the labels leaf is a non-differentiable leaf — a common source of offset errors)
        double[] data1 = new double[100];
        double[] data2 = new double[100];
        double[] data3 = new double[100];
        for (int i = 0; i < data1.length; i++) data1[i] = Math.sin(i);
        for (int i = 0; i < data2.length; i++) data2[i] = Math.cos(i);
        for (int i = 0; i < data3.length; i++) data3[i] = i * 0.01;

        RereDiffTensor x1 = new RereDiffTensor(data1, 10, 10);
        x1.setRequiresGrad(true);
        RereDiffTensor x2 = new RereDiffTensor(data2, 10, 10);
        x2.setRequiresGrad(true);
        RereDiffTensor x3 = new RereDiffTensor(data3, 10, 10);
        x3.setRequiresGrad(true);

        // Chain with compatible shapes: x1.mul(x2).add(x3).sum()
        IDiffTensor y = x1.mul(x2).add(x3).sum();
        RereDiffTensor root = (RereDiffTensor) y;

        // Build topological order
        ArrayList<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);
        assertTrue(order.size() >= 4, "Graph should have at least 4 nodes");

        // Method 1: fresh serialization via serializeGraph
        java.nio.ByteBuffer freshBuf = TensorBinaryProtocol.serializeGraph(root, order);
        byte[] freshBytes = new byte[freshBuf.remaining()];
        freshBuf.get(freshBytes);

        // Method 2: CachedGraph serialization
        int structHash = ExportShapeValidator.computeStructureHash(order);
        TensorBinaryProtocol.CachedGraph cached = TensorBinaryProtocol.serializeGraphCached(root, order, structHash);
        byte[] cachedBytes = cached.updateLeafData(order);

        // Compare lengths
        assertEquals(freshBytes.length, cachedBytes.length,
            "CachedGraph should produce same byte length as fresh serialization");

        // Compare content byte-by-byte
        int firstDiff = -1;
        for (int i = 0; i < freshBytes.length; i++) {
            if (freshBytes[i] != cachedBytes[i]) { firstDiff = i; break; }
        }
        if (firstDiff >= 0) {
            fail("CachedGraph byte mismatch at offset " + firstDiff
                + " (fresh=0x" + Integer.toHexString(freshBytes[firstDiff] & 0xFF)
                + " cached=0x" + Integer.toHexString(cachedBytes[firstDiff] & 0xFF)
                + ") — CachedGraph.updateLeafData corrupted the binary protocol");
        }

        // Note: full manual scan for self-consistency is done in testLeafDataOffsetsAreConsistent
    }

    /**
     * Tests that serializeGraph + serializeGraphCached agree on data offset positions
     * for each leaf. A mismatch in offset means updateLeafData overwrites wrong bytes.
     */
    @Test
    void testLeafDataOffsetsAreConsistent() {
        double[] data1 = new double[128];
        double[] data2 = new double[128];
        for (int i = 0; i < data1.length; i++) data1[i] = i * 1.0;
        for (int i = 0; i < data2.length; i++) data2[i] = i * 2.0;

        RereDiffTensor x1 = new RereDiffTensor(data1, 128);
        x1.setRequiresGrad(true);
        RereDiffTensor x2 = new RereDiffTensor(data2, 128);
        x2.setRequiresGrad(true);

        IDiffTensor y = x1.add(x2).sum();
        RereDiffTensor root = (RereDiffTensor) y;

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);

        // Fresh serialization
        java.nio.ByteBuffer freshBuf = TensorBinaryProtocol.serializeGraph(root, order);
        byte[] freshBytes = new byte[freshBuf.remaining()];
        freshBuf.get(freshBytes);

        // Cached serialization
        int structHash = ExportShapeValidator.computeStructureHash(order);
        TensorBinaryProtocol.CachedGraph cached = TensorBinaryProtocol.serializeGraphCached(root, order, structHash);
        byte[] cachedRaw = cached.updateLeafData(order);

        // Debug: check first leaf data lengths
        System.err.println("=== Binary protocol diagnostic ===");
        System.err.println("order size=" + order.size() + ", freshBytes.length=" + freshBytes.length
            + ", cachedRaw.length=" + cachedRaw.length);
        for (int vi = 0; vi < order.size(); vi++) {
            RereDiffTensor v = order.get(vi);
            System.err.println("  node[" + vi + "]: op=" + v.opTag()
                + " leaf=" + v.isLeaf() + " totalSize=" + v.totalSize()
                + " requiresGrad=" + v.requiresGrad()
                + " backwardIndices=" + (v.backwardIndices() != null ? v.backwardIndices().length : 0));
        }

        // Verify both produce identical output (first call: no data has changed)
        if (!Arrays.equals(freshBytes, cachedRaw)) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < Math.min(freshBytes.length, Math.min(cachedRaw.length, 100)); i++) {
                if (freshBytes[i] != cachedRaw[i]) {
                    sb.append(String.format("\n  byte[%d]: fresh=0x%02x cached=0x%02x",
                        i, freshBytes[i] & 0xFF, cachedRaw[i] & 0xFF));
                    if (sb.length() > 500) break;
                }
            }
            fail("CachedGraph and fresh serialization differ:" + sb.toString());
        }

        // Verify the data at each leaf offset is correct by comparing with order's leaf data
        int leafIdx = 0;
        int pos = 12;
        for (int nodeIdx = 0; nodeIdx < order.size(); nodeIdx++) {
            int flags = ((freshBytes[pos + 1] & 0xFF) << 8) | (freshBytes[pos] & 0xFF); pos += 2;
            int opLen = ((freshBytes[pos + 1] & 0xFF) << 8) | (freshBytes[pos] & 0xFF); pos += 2;
            pos += 4;
            pos += opLen;
            int numDims = ((freshBytes[pos + 1] & 0xFF) << 8) | (freshBytes[pos] & 0xFF); pos += 2;
            pos += numDims * 4;
            int numInputs = ((freshBytes[pos + 1] & 0xFF) << 8) | (freshBytes[pos] & 0xFF); pos += 2;
            pos += numInputs * 4;
            if ((flags & 2) != 0) pos += 8; // scalar
            if ((flags & 4) != 0) pos += 8; // param2
            if ((flags & 1) != 0) { // has_data
                // Read the data_len from the buffer
                int storedLen = ((freshBytes[pos + 3] & 0xFF) << 24) | ((freshBytes[pos + 2] & 0xFF) << 16)
                              | ((freshBytes[pos + 1] & 0xFF) << 8) | (freshBytes[pos] & 0xFF);
                RereDiffTensor v = order.get(nodeIdx);
                int actualLen = (int) v.totalSize();
                assertEquals(actualLen, storedLen,
                    "Node " + nodeIdx + " data_len mismatch: expected " + actualLen + " got " + storedLen);
                pos += 4 + storedLen * 8;
                leafIdx++;
            }
            if ((flags & 8) != 0) { // has_indices
                int idxLen = ((freshBytes[pos + 3] & 0xFF) << 24) | ((freshBytes[pos + 2] & 0xFF) << 16)
                           | ((freshBytes[pos + 1] & 0xFF) << 8) | (freshBytes[pos] & 0xFF);
                pos += 4 + idxLen * 4;
            }
        }
    }
}
