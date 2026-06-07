package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.compute.DoubleFlatGemm;
import java.util.Arrays;

/**
 * Final debug test that manually replicates what the attention method does,
 * verifying each intermediate step.
 */
public class AttentionTrace {
    public static void main(String[] args) {
        for (int round = 0; round < 2; round++) {
            double[] qArr = round == 0
                ? new double[]{0.5, 1.2, -0.3, 0.8}
                : new double[]{0.5 + 0.01, 1.2, -0.3, 0.8};
            double[] kArr = new double[]{1.0, -0.5, 0.2, 0.7};
            double[] vArr = new double[]{2.0, 1.0, 0.0, -1.0};

            System.out.println("=== Round " + round + " ===");

            // Step 1: Create tensors (same as in AttentionDebug2)
            RereDiffTensor qt = new RereDiffTensor(qArr, 1, 2, 2);
            RereDiffTensor kt = new RereDiffTensor(kArr, 1, 2, 2);
            RereDiffTensor vt = new RereDiffTensor(vArr, 1, 2, 2);

            // Step 2: Extract data (same as attention method does)
            double[] qd = qt.toDoubleArray();
            double[] kd = kt.toDoubleArray();
            double[] vd = vt.toDoubleArray();
            System.out.println("  qd=" + Arrays.toString(qd));
            System.out.println("  kd=" + Arrays.toString(kd));
            System.out.println("  vd=" + Arrays.toString(vd));

            // Step 3: Extract slices
            int batch = 1, seqQ = 2, dk = 2, seqK = 2, dv = 2;
            int qStride = 4, kStride = 4, vStride = 4;
            int scoresStride = seqQ * seqK; // 4
            double scale = 1.0 / Math.sqrt(dk);

            double[] qSlice = Arrays.copyOfRange(qd, 0, 4);
            double[] kSlice = Arrays.copyOfRange(kd, 0, 4);
            System.out.println("  qSlice=" + Arrays.toString(qSlice));
            System.out.println("  kSlice=" + Arrays.toString(kSlice));

            // Step 4: Compute scores
            double[] kT = DoubleFlatGemm.flatTranspose(kSlice, seqK, dk);
            double[] scoresB = DoubleFlatGemm.flatMmul(qSlice, seqQ, dk, kT, seqK);
            System.out.println("  kT=" + Arrays.toString(kT));
            System.out.println("  scoresB (raw)=" + Arrays.toString(scoresB));

            // Step 5: Apply scale (BUG: currently applied to separate scores[] array)
            double[] scores = new double[4];
            for (int i = 0; i < 4; i++) scores[i] = scoresB[i] * scale;
            System.out.println("  scores (scaled)=" + Arrays.toString(scores));
            System.out.println("  scoresB (after scale loop, should be unchanged)=" + Arrays.toString(scoresB));

            // Step 6: Softmax on scoresB (BUG: should be on scores!)
            double[] attn = new double[4];
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
            System.out.println("  attn (softmax on RAW scoresB)=" + Arrays.toString(attn));

            // Step 7: Correct softmax (on scaled scores)
            double[] attnCorrect = new double[4];
            for (int q = 0; q < seqQ; q++) {
                int rowOff = q * seqK;
                double maxVal = -Double.MAX_VALUE;
                for (int j = 0; j < seqK; j++) maxVal = Math.max(maxVal, scores[rowOff + j]);
                double sumExp = 0;
                for (int j = 0; j < seqK; j++) {
                    double e = Math.exp(scores[rowOff + j] - maxVal);
                    attnCorrect[rowOff + j] = e;
                    sumExp += e;
                }
                double invSum = 1.0 / sumExp;
                for (int j = 0; j < seqK; j++) attnCorrect[rowOff + j] *= invSum;
            }
            System.out.println("  attnCorrect (softmax on SCALED scores)=" + Arrays.toString(attnCorrect));

            // Step 8: Output using BUGGY attention
            double[] outBug = DoubleFlatGemm.flatMmul(attn, seqQ, seqK, vd, dv);
            double[] outCorrect = DoubleFlatGemm.flatMmul(attnCorrect, seqQ, seqK, vd, dv);

            double sumBug = 0, sumCorrect = 0;
            for (double v : outBug) sumBug += v;
            for (double v : outCorrect) sumCorrect += v;
            System.out.println("  outBug=" + Arrays.toString(outBug) + " sum=" + sumBug);
            System.out.println("  outCorrect=" + Arrays.toString(outCorrect) + " sum=" + sumCorrect);

            // Step 9: What does the ACTUAL method return?
            IDiffTensor actual = qt.scaledDotProductAttention(kt, vt, null, 0.0);
            double[] actualArr = actual.toDoubleArray();
            double actualSum = 0;
            for (double v : actualArr) actualSum += v;
            System.out.println("  ACTUAL=" + Arrays.toString(actualArr) + " sum=" + actualSum);
            System.out.println();
        }
    }
}
