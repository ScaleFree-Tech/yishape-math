package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import java.util.Arrays;

/**
 * Test if there's a static caching issue.
 */
public class AttentionTrace2 {
    public static void main(String[] args) {
        double[] q1 = {0.5, 1.2, -0.3, 0.8};
        double[] q2 = {0.51, 1.2, -0.3, 0.8};
        double[] k = {1.0, -0.5, 0.2, 0.7};
        double[] v = {2.0, 1.0, 0.0, -1.0};

        // Test 1: Call with q1, then q2 (original order)
        System.out.println("=== Test 1: q1 then q2 ===");
        test(q1, k, v, "q1");
        test(q2, k, v, "q2");

        // Test 2: Call with q2, then q1 (reverse order)
        System.out.println("=== Test 2: q2 then q1 ===");
        test(q2, k, v, "q2");
        test(q1, k, v, "q1");

        // Test 3: Interleave fresh Q with fresh K,V
        System.out.println("=== Test 3: interleaved ===");
        test(q1, k, v, "q1-a");
        test(q2, k, v, "q2-a");
        test(q1.clone(), k.clone(), v.clone(), "q1-b");
        test(q2.clone(), k.clone(), v.clone(), "q2-b");
    }

    static void test(double[] qArr, double[] kArr, double[] vArr, String label) {
        RereDiffTensor qt = new RereDiffTensor(qArr, 1, 2, 2);
        RereDiffTensor kt = new RereDiffTensor(kArr, 1, 2, 2);
        RereDiffTensor vt = new RereDiffTensor(vArr, 1, 2, 2);
        IDiffTensor out = qt.scaledDotProductAttention(kt, vt, null, 0.0);
        double[] vals = out.toDoubleArray();
        double sum = 0;
        for (double x : vals) sum += x;
        System.out.println(label + ": out=" + Arrays.toString(vals) + " sum=" + sum);
    }
}
