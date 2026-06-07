package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Direct trace: create tensors, call forward, print everything.
 */
public class AttentionDebug2 {
    public static void main(String[] args) {
        test("Q1", new double[]{0.5, 1.2, -0.3, 0.8});
        test("Q2", new double[]{0.5 + 0.01, 1.2, -0.3, 0.8});
    }

    static void test(String label, double[] qData) {
        RereDiffTensor qt = new RereDiffTensor(qData, 1, 2, 2);
        RereDiffTensor kt = new RereDiffTensor(new double[]{1.0, -0.5, 0.2, 0.7}, 1, 2, 2);
        RereDiffTensor vt = new RereDiffTensor(new double[]{2.0, 1.0, 0.0, -1.0}, 1, 2, 2);

        // Print the raw data as stored in the tensor
        double[] rawQ = qt.toDoubleArray();
        System.out.println(label + ": rawQ=" + java.util.Arrays.toString(rawQ));
        System.out.println(label + ": rawK=" + java.util.Arrays.toString(kt.toDoubleArray()));
        System.out.println(label + ": rawV=" + java.util.Arrays.toString(vt.toDoubleArray()));

        // Direct computation with DoubleFlatGemm — what should happen
        int seqQ = 2, dk = 2, seqK = 2, dv = 2;
        double scale = 1.0 / Math.sqrt(dk);
        double[] kT = com.yishape.lab.math.compute.DoubleFlatGemm.flatTranspose(kt.toDoubleArray(), seqK, dk);
        double[] scoresDirect = com.yishape.lab.math.compute.DoubleFlatGemm.flatMmul(rawQ, seqQ, dk, kT, seqK);
        System.out.println(label + ": scoresDirect (raw)=" + java.util.Arrays.toString(scoresDirect));

        // Scale scores
        for (int i = 0; i < scoresDirect.length; i++) scoresDirect[i] *= scale;

        // Softmax
        double[] attnDirect = new double[seqQ * seqK];
        for (int q = 0; q < seqQ; q++) {
            int rowOff = q * seqK;
            double maxVal = -Double.MAX_VALUE;
            for (int j = 0; j < seqK; j++) maxVal = Math.max(maxVal, scoresDirect[rowOff + j]);
            double sumExp = 0;
            for (int j = 0; j < seqK; j++) {
                double e = Math.exp(scoresDirect[rowOff + j] - maxVal);
                attnDirect[rowOff + j] = e;
                sumExp += e;
            }
            double invSum = 1.0 / sumExp;
            for (int j = 0; j < seqK; j++) attnDirect[rowOff + j] *= invSum;
        }
        System.out.println(label + ": attnDirect=" + java.util.Arrays.toString(attnDirect));

        IDiffTensor out = qt.scaledDotProductAttention(kt, vt, null, 0.0);
        double[] outVal = out.toDoubleArray();
        double sum = 0;
        for (double v : outVal) sum += v;
        System.out.println(label + ": ACTUAL out=" + java.util.Arrays.toString(outVal) + " sum=" + sum);
        System.out.println();
    }
}
