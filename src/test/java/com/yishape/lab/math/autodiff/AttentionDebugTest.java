package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Standalone debug to trace attention forward computation.
 */
public class AttentionDebugTest {
    public static void main(String[] args) {
        // Scenario: tiny attention, perturb Q[0] and check if output changes
        double[] qData1 = {0.5, 1.2, -0.3, 0.8};
        double[] qData2 = {0.5 + 0.01, 1.2, -0.3, 0.8};
        double[] kData = {1.0, -0.5, 0.2, 0.7};
        double[] vData = {2.0, 1.0, 0.0, -1.0};

        double loss1 = scalarLoss(qData1, kData, vData);
        double loss2 = scalarLoss(qData2, kData, vData);

        System.out.println("scalarLoss 1 = " + loss1);
        System.out.println("scalarLoss 2 = " + loss2);
        System.out.println("diff = " + (loss2 - loss1));
        System.out.println();

        // Check output tensor directly
        System.out.println("=== Check output tensors directly ===");
        checkOutputTensor(qData1, qData2, kData, vData);
        System.out.println();

        // Manual trace
        System.out.println("=== Manual trace for qData1 ===");
        manualForward(qData1, kData, vData);
        System.out.println();
        System.out.println("=== Manual trace for qData2 ===");
        manualForward(qData2, kData, vData);
    }

    static double scalarLoss(double[] qData, double[] kData, double[] vData) {
        RereDiffTensor qt = new RereDiffTensor(qData, 1, 2, 2);
        RereDiffTensor kt = new RereDiffTensor(kData, 1, 2, 2);
        RereDiffTensor vt = new RereDiffTensor(vData, 1, 2, 2);
        IDiffTensor out = qt.scaledDotProductAttention(kt, vt, null, 0.0);
        return out.sum().item();
    }

    static void checkOutputTensor(double[] qData1, double[] qData2, double[] kData, double[] vData) {
        RereDiffTensor qt1 = new RereDiffTensor(qData1, 1, 2, 2);
        RereDiffTensor kt1 = new RereDiffTensor(kData, 1, 2, 2);
        RereDiffTensor vt1 = new RereDiffTensor(vData, 1, 2, 2);
        IDiffTensor out1 = qt1.scaledDotProductAttention(kt1, vt1, null, 0.0);

        RereDiffTensor qt2 = new RereDiffTensor(qData2, 1, 2, 2);
        RereDiffTensor kt2 = new RereDiffTensor(kData, 1, 2, 2);
        RereDiffTensor vt2 = new RereDiffTensor(vData, 1, 2, 2);
        IDiffTensor out2 = qt2.scaledDotProductAttention(kt2, vt2, null, 0.0);

        double[] outVal1 = out1.toDoubleArray();
        double[] outVal2 = out2.toDoubleArray();
        System.out.println("out1 values: " + java.util.Arrays.toString(outVal1));
        System.out.println("out2 values: " + java.util.Arrays.toString(outVal2));

        double sum1 = 0, sum2 = 0;
        for (double v : outVal1) sum1 += v;
        for (double v : outVal2) sum2 += v;
        System.out.println("sum1=" + sum1 + " sum2=" + sum2 + " diff=" + (sum2 - sum1));
    }

    static void manualForward(double[] qd, double[] kd, double[] vd) {
        int seqQ = 2, dk = 2, seqK = 2, dv = 2;
        double scale = 1.0 / Math.sqrt(dk);
        System.out.println("scale = " + scale);

        // Q@K^T
        // Q = [[qd[0], qd[1]], [qd[2], qd[3]]]
        // K = [[kd[0], kd[1]], [kd[2], kd[3]]]
        System.out.println("Q = [[" + qd[0] + ", " + qd[1] + "], [" + qd[2] + ", " + qd[3] + "]]");
        System.out.println("K = [[" + kd[0] + ", " + kd[1] + "], [" + kd[2] + ", " + kd[3] + "]]");
        System.out.println("V = [[" + vd[0] + ", " + vd[1] + "], [" + vd[2] + ", " + vd[3] + "]]");

        // scores = Q @ K^T
        double s00 = qd[0]*kd[0] + qd[1]*kd[1];
        double s01 = qd[0]*kd[2] + qd[1]*kd[3];
        double s10 = qd[2]*kd[0] + qd[3]*kd[1];
        double s11 = qd[2]*kd[2] + qd[3]*kd[3];
        System.out.println("Q@K^T = [[" + s00 + ", " + s01 + "], [" + s10 + ", " + s11 + "]]");

        // Scaled
        double ss00 = s00 * scale, ss01 = s01 * scale;
        double ss10 = s10 * scale, ss11 = s11 * scale;
        System.out.println("Scaled = [[" + ss00 + ", " + ss01 + "], [" + ss10 + ", " + ss11 + "]]");

        // Softmax (row-wise)
        // Row 0
        double max0 = Math.max(ss00, ss01);
        double e00 = Math.exp(ss00 - max0), e01 = Math.exp(ss01 - max0);
        double sum0 = e00 + e01;
        double a00 = e00 / sum0, a01 = e01 / sum0;
        // Row 1
        double max1 = Math.max(ss10, ss11);
        double e10 = Math.exp(ss10 - max1), e11 = Math.exp(ss11 - max1);
        double sum1 = e10 + e11;
        double a10 = e10 / sum1, a11 = e11 / sum1;
        System.out.println("Attn = [[" + a00 + ", " + a01 + "], [" + a10 + ", " + a11 + "]]");

        // Output = Attn @ V
        double o00 = a00*vd[0] + a01*vd[2];
        double o01 = a00*vd[1] + a01*vd[3];
        double o10 = a10*vd[0] + a11*vd[2];
        double o11 = a10*vd[1] + a11*vd[3];
        System.out.println("Out = [[" + o00 + ", " + o01 + "], [" + o10 + ", " + o11 + "]]");

        double sum = o00 + o01 + o10 + o11;
        System.out.println("Sum = " + sum);
    }
}
