package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.graph.binary.TensorBinaryProtocol;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

/** Quick diagnostic: dump binary protocol data for broadcast-add test to a file. */
public class BinaryDebugDump {
    public static void main(String[] args) throws Exception {
        Random rng = new Random(0xCAFE_BEEF);
        int B = 32, C = 10;
        double[] d1 = new double[B * C];
        double[] d2 = new double[B];
        for (int i = 0; i < B * C; i++) d1[i] = rng.nextDouble() * 2.0 - 1.0;
        for (int i = 0; i < B; i++) d2[i] = rng.nextDouble() * 2.0 - 1.0;

        RereDiffTensor a = (RereDiffTensor) AD.leafTensor(d1, B, C);
        RereDiffTensor b = (RereDiffTensor) AD.leafTensor(d2, B);
        RereDiffTensor loss = (RereDiffTensor) a.add(b).sum();

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        loss.buildTopo(order, visited);

        int structureHash = ExportShapeValidator.computeStructureHash(order);
        TensorBinaryProtocol.CachedGraph cg = TensorBinaryProtocol.serializeGraphCached(loss, order, structureHash);
        byte[] data = cg.updateLeafData(order);

        // CPU reference
        loss.backward();
        double cpuLoss = loss.value().toDoubleArray()[0];
        double[] cpuGa = a.gradData().clone();
        double[] cpuGb = b.gradData().clone();

        System.out.printf("CPU loss=%.12f%n", cpuLoss);
        System.out.printf("d1 sum=%.6f d2 sum=%.6f expected=%.6f%n",
            sum(d1), sum(d2), sum(d1) + C * sum(d2));
        System.out.printf("Binary data size: %d bytes%n", data.length);

        // Write binary data to temp file for Rust test
        String path = System.getProperty("java.io.tmpdir") + "/simple_graph.bin";
        Files.write(Paths.get(path), data);
        System.out.println("Binary data written to: " + path);
    }
    static double sum(double[] a) { double s = 0; for (double v : a) s += v; return s; }
}
