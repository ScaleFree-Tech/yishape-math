package com.yishape.lab.math.autodiff.graph.binary;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.graph.ExportShapeValidator;

/**
 * Validates that {@link TensorBinaryProtocol#serializeGraph} and
 * {@link TensorBinaryProtocol#serializeGraphCached} + {@link TensorBinaryProtocol.CachedGraph#updateLeafData}
 * produce byte-identical buffers.
 *
 * <p>This is the prerequisite for enabling GPU skeleton caching — the GPU
 * executor must be able to use the incremental path (clone skeleton + overwrite
 * leaf data) without changing the byte stream that Rust receives.</p>
 */
public class TensorBinaryProtocolRoundTripTest {

    private static byte[] serializeFresh(RereDiffTensor root, List<RereDiffTensor> order) {
        var buf = TensorBinaryProtocol.serializeGraph(root, order);
        byte[] data = new byte[buf.remaining()];
        buf.get(data);
        return data;
    }

    private static byte[] serializeCached(RereDiffTensor root, List<RereDiffTensor> order, int structHash) {
        var cached = TensorBinaryProtocol.serializeGraphCached(root, order, structHash);
        return cached.updateLeafData(order);
    }

    /**
     * Round-trip: fresh serialization vs cached+update for a simple add graph.
     */
    @Test
    void testSimpleAddGraph() {
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor b = AD.tensor(new double[]{5, 6, 7, 8}, 2, 2);
        IDiffTensor c = a.add(b);
        RereDiffTensor root = (RereDiffTensor) c;

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        root.buildTopo(order, new java.util.HashSet<>());

        int structHash = ExportShapeValidator.computeStructureHash(order);

        byte[] fresh = serializeFresh(root, order);
        byte[] cached = serializeCached(root, order, structHash);

        assertArrayEquals(fresh, cached,
            "Fresh serializeGraph must match serializeGraphCached + updateLeafData");
    }

    /**
     * Round-trip with leaf data changed between cache and update.
     * Creates two independent graphs with identical structure but different leaf values,
     * caches the skeleton from graph 1, then verifies that updateLeafData with graph 2's
     * order produces bytes matching a fresh serialization of graph 2.
     */
    @Test
    void testSimpleAddGraph_DataChanged() {
        // Create two graphs with identical structure but different leaf values
        IDiffTensor a1 = AD.tensor(new double[]{1, 2, 3}, 3);
        IDiffTensor b1 = AD.tensor(new double[]{4, 5, 6}, 3);
        IDiffTensor c1 = a1.add(b1);
        RereDiffTensor root1 = (RereDiffTensor) c1;

        ArrayList<RereDiffTensor> order1 = new ArrayList<>();
        root1.buildTopo(order1, new java.util.HashSet<>());
        int structHash = ExportShapeValidator.computeStructureHash(order1);

        // Cache skeleton with graph 1's data
        var cached = TensorBinaryProtocol.serializeGraphCached(root1, order1, structHash);

        // Now create a second graph with same structure but different data
        IDiffTensor a2 = AD.tensor(new double[]{10, 20, 30}, 3);
        IDiffTensor b2 = AD.tensor(new double[]{40, 50, 60}, 3);
        IDiffTensor c2 = a2.add(b2);
        RereDiffTensor root2 = (RereDiffTensor) c2;

        ArrayList<RereDiffTensor> order2 = new ArrayList<>();
        root2.buildTopo(order2, new java.util.HashSet<>());
        int structHash2 = ExportShapeValidator.computeStructureHash(order2);

        // Structure hash should match (same topology)
        assertEquals(structHash, structHash2,
            "Same topology should produce same structure hash");

        // Fresh serialization of graph 2
        byte[] fresh2 = serializeFresh(root2, order2);

        // Cached skeleton + update with graph 2's leaf data
        byte[] updated = cached.updateLeafData(order2);

        assertArrayEquals(fresh2, updated,
            "Cached skeleton + updateLeafData must match fresh serialization " +
            "when topology is identical (same structure hash)");
    }

    /**
     * Round-trip for a broadcast + sum dim graph (the op that caused
     * the previous GPU cache regression).
     */
    @Test
    void testBroadcastThenSumDim() {
        // Simulates: a = tensor([2,3]), b = broadcast_to([4,3]), c = sum(b, dim=0)
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor b = a.add(AD.tensor(new double[]{0, 0, 0}, 1, 3)); // broadcast add
        IDiffTensor c = b.sum(0, false);
        RereDiffTensor root = (RereDiffTensor) c;

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        root.buildTopo(order, new java.util.HashSet<>());

        int structHash = ExportShapeValidator.computeStructureHash(order);

        var cached = TensorBinaryProtocol.serializeGraphCached(root, order, structHash);
        byte[] fresh = serializeFresh(root, order);
        byte[] updated = cached.updateLeafData(order);

        assertArrayEquals(fresh, updated,
            "Broadcast+sum: fresh serialize must match cached.updateLeafData");
    }

    /**
     * Multiple rounds of updateLeafData should all match fresh serialization.
     */
    @Test
    void testMultipleUpdateRounds() {
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3}, 3);
        IDiffTensor b = a.mul(AD.tensor(new double[]{2, 2, 2}, 3));
        IDiffTensor c = b.sum();
        RereDiffTensor root = (RereDiffTensor) c;

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        root.buildTopo(order, new java.util.HashSet<>());

        int structHash = ExportShapeValidator.computeStructureHash(order);
        var cached = TensorBinaryProtocol.serializeGraphCached(root, order, structHash);

        // Round 1: same data
        assertArrayEquals(serializeFresh(root, order), cached.updateLeafData(order));

        // Round 2: change leaf data
        RereDiffTensor leaf = order.stream()
            .filter(v -> v.isLeaf() && v.totalSize() == 3)
            .findFirst().orElseThrow();
        leaf.value().set(10.0, 0);
        leaf.value().set(20.0, 1);
        leaf.value().set(30.0, 2);
        assertArrayEquals(serializeFresh(root, order), cached.updateLeafData(order),
            "Round 2 after data change");

        // Round 3: change again
        leaf.value().set(-1.0, 0);
        leaf.value().set(-2.0, 1);
        leaf.value().set(-3.0, 2);
        assertArrayEquals(serializeFresh(root, order), cached.updateLeafData(order),
            "Round 3 after data change");
    }

    /**
     * Round-trip for a graph with multiple leaves with different sizes.
     */
    @Test
    void testMultipleLeaves() {
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor b = AD.tensor(new double[]{5, 6, 7, 8}, 2, 2);
        IDiffTensor c = AD.tensor(new double[]{0.1, 0.2, 0.3, 0.4}, 2, 2);
        IDiffTensor d = a.add(b).mul(c);
        RereDiffTensor root = (RereDiffTensor) d;

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        root.buildTopo(order, new java.util.HashSet<>());

        int structHash = ExportShapeValidator.computeStructureHash(order);
        byte[] fresh = serializeFresh(root, order);
        byte[] cached = serializeCached(root, order, structHash);
        assertArrayEquals(fresh, cached, "Multiple leaves: fresh must match cached");
    }

    /**
     * Round-trip for a matrix multiplication graph.
     */
    @Test
    void testMmulGraph() {
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor b = AD.tensor(new double[]{7, 8, 9, 10, 11, 12}, 3, 2);
        IDiffTensor c = a.mmul(b);
        RereDiffTensor root = (RereDiffTensor) c;

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        root.buildTopo(order, new java.util.HashSet<>());

        int structHash = ExportShapeValidator.computeStructureHash(order);
        byte[] fresh = serializeFresh(root, order);
        byte[] cached = serializeCached(root, order, structHash);
        assertArrayEquals(fresh, cached, "mmul: fresh must match cached");
    }

    /**
     * Structure hash change should produce different bytes (sanity check).
     * When structure hash changes, the cache should be invalidated and a new
     * full serialization emitted — this test just verifies that different
     * graphs produce different cached output.
     */
    @Test
    void testStructureHashChange() {
        // Graph 1: simple add
        IDiffTensor a1 = AD.tensor(new double[]{1, 2, 3}, 3);
        IDiffTensor b1 = AD.tensor(new double[]{4, 5, 6}, 3);
        IDiffTensor c1 = a1.add(b1);
        RereDiffTensor root1 = (RereDiffTensor) c1;

        ArrayList<RereDiffTensor> order1 = new ArrayList<>();
        root1.buildTopo(order1, new java.util.HashSet<>());
        int hash1 = ExportShapeValidator.computeStructureHash(order1);

        // Graph 2: mul instead of add (different op tag)
        IDiffTensor a2 = AD.tensor(new double[]{1, 2, 3}, 3);
        IDiffTensor b2 = AD.tensor(new double[]{4, 5, 6}, 3);
        IDiffTensor c2 = a2.mul(b2);
        RereDiffTensor root2 = (RereDiffTensor) c2;

        ArrayList<RereDiffTensor> order2 = new ArrayList<>();
        root2.buildTopo(order2, new java.util.HashSet<>());
        int hash2 = ExportShapeValidator.computeStructureHash(order2);

        assertNotEquals(hash1, hash2,
            "add vs mul should produce different structure hashes");
        assertNotEquals(
            java.util.Arrays.hashCode(serializeFresh(root1, order1)),
            java.util.Arrays.hashCode(serializeFresh(root2, order2)),
            "add vs mul should produce different serialized bytes");
    }

    /**
     * NaN in leaf data should round-trip correctly.
     */
    @Test
    void testNaNLeafData() {
        IDiffTensor a = AD.tensor(new double[]{1, Double.NaN, 3}, 3);
        IDiffTensor b = a.mul(AD.tensor(new double[]{2, 2, 2}, 3));
        RereDiffTensor root = (RereDiffTensor) b;

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        root.buildTopo(order, new java.util.HashSet<>());

        int structHash = ExportShapeValidator.computeStructureHash(order);
        byte[] fresh = serializeFresh(root, order);
        byte[] cached = serializeCached(root, order, structHash);
        assertArrayEquals(fresh, cached, "NaN leaf data: fresh must match cached");
    }
}
