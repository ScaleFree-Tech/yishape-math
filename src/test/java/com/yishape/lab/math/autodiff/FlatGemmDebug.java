package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.compute.DoubleFlatGemm;

/**
 * Test DoubleFlatGemm directly with attention test values.
 */
public class FlatGemmDebug {
    public static void main(String[] args) {
        double[] qData1 = {0.5, 1.2, -0.3, 0.8};
        double[] qData2 = {0.51, 1.2, -0.3, 0.8};
        double[] kData = {1.0, -0.5, 0.2, 0.7};
        double[] vData = {2.0, 1.0, 0.0, -1.0};

        int seqQ = 2, dk = 2, seqK = 2, dv = 2;
        double scale = 1.0 / Math.sqrt(dk);

        compute("qData1", qData1, kData, vData, seqQ, dk, seqK, dv, scale);
        compute("qData2", qData2, kData, vData, seqQ, dk, seqK, dv, scale);
    }

    static void compute(String label, double[] qd, double[] kd, double[] vd, int seqQ, int dk, int seqK, int dv, double scale) {
        // qSlice = [qd[0..3]]
        double[] qSlice = java.util.Arrays.copyOfRange(qd, 0, 4);
        double[] kSlice = java.util.Arrays.copyOfRange(kd, 0, 4);
        System.out.println(label + ": qSlice=" + java.util.Arrays.toString(qSlice));
        System.out.println(label + ": kSlice=" + java.util.Arrays.toString(kSlice));

        double[] kT = DoubleFlatGemm.flatTranspose(kSlice, seqK, dk);
        System.out.println(label + ": kT=" + java.util.Arrays.toString(kT));

        double[] scoresB = DoubleFlatGemm.flatMmul(qSlice, seqQ, dk, kT, seqK);
        System.out.println(label + ": scoresB=" + java.util.Arrays.toString(scoresB));

        // Apply scale
        for (int i = 0; i < scoresB.length; i++) scoresB[i] *= scale;
        System.out.println(label + ": scaled=" + java.util.Arrays.toString(scoresB));

        // Softmax
        double[] attn = new double[seqQ * seqK];
        for (int q = 0; q < seqQ; q++) {
            int rowOff = q * seqK;
            double maxVal = -Double.MAX_VALUE;
            for (int j = 0; j < seqK; j++) maxVal = Math.max(maxVal, scoresB[rowOff + j]);
            double sumExp = 0;
            for (int j = 0; j < seqK; j++) {
                double e = Math.exp(scoresB[rowOff + j] - maxVal);
                attn[rowOff + j] = e;
                sumExp += e;
            }
            double invSum = 1.0 / sumExp;
            for (int j = 0; j < seqK; j++) attn[rowOff + j] *= invSum;
        }
        System.out.println(label + ": attn=" + java.util.Arrays.toString(attn));

        // Output = attn @ V
        double[] out = DoubleFlatGemm.flatMmul(attn, seqQ, seqK, vd, dv);
        double sum = 0;
        for (double v : out) sum += v;
        System.out.println(label + ": out=" + java.util.Arrays.toString(out) + " sum=" + sum);
        System.out.println();
    }
}
