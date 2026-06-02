package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.IDiffMatrix;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;

/**
 * Numerical gradient verification for DML-style loss functions using autodiff.
 */
public class ADDmlGradientTest {

    private static final double TOL = 1e-4;

    // ---- LDML-style pairwise BCE loss ----

    private IDiffVector buildLdmlLoss(IDiffVector w, double[][] uData, double[] labels,
            int r, int d, double bias) {
        IDiffMatrix L = w.reshape(r, d);
        IDiffMatrix U = AD.matrix(uData);
        IDiffMatrix Lu = U.matmul(L.transpose());
        IDiffVector dist = Lu.square().matmul(AD.ones(r));
        IDiffVector z = dist.rsub(bias);
        IDiffVector sigZ = z.sigmoid();
        IDiffVector labelVar = AD.vector(labels);
        IDiffVector logSig = sigZ.log();
        IDiffVector logOneMinusSig = sigZ.rsub(1.0).log();
        IDiffVector term1 = logSig.mul(labelVar);
        IDiffVector term2 = logOneMinusSig.mul(labelVar.rsub(1.0));
        int P = uData.length;
        return term1.add(term2).sum().mul(-1.0 / P);
    }

    @Test
    void testLdmlGradientNumerical() {
        double[][] uData = { { 1, 0 }, { 0, 1 }, { 1, 1 }, { -1, 0 } };
        double[] labels = { 1, 0, 1, 0 };
        int r = 2, d = 2;
        double bias = 0.5;
        double[] wArr = { 0.5, 0, 0, 0.5 };

        double eps = 1e-5;
        for (int i = 0; i < wArr.length; i++) {
            double[] wp = wArr.clone();
            wp[i] += eps;
            IDiffVector vp = AD.vector(wp);
            double fp = buildLdmlLoss(vp, uData, labels, r, d, bias).getValue().get(0);

            double[] wm = wArr.clone();
            wm[i] -= eps;
            IDiffVector vm = AD.vector(wm);
            double fm = buildLdmlLoss(vm, uData, labels, r, d, bias).getValue().get(0);

            double numGrad = (fp - fm) / (2 * eps);

            IDiffVector v = AD.vector(wArr);
            IDiffVector loss = buildLdmlLoss(v, uData, labels, r, d, bias);
            loss.backward();

            assertEquals(numGrad, v.getGradient().get(i), TOL);
        }
    }

    // ---- LMNN-style triplet hinge loss ----

    private IDiffVector buildLmnnLoss(IDiffVector w, double[][] u1Data, double[][] u2Data,
            int r, int d, double margin) {
        IDiffMatrix L = w.reshape(r, d);
        IDiffMatrix LT = L.transpose();
        IDiffMatrix U1 = AD.matrix(u1Data);
        IDiffMatrix U2 = AD.matrix(u2Data);
        IDiffVector ones_r = AD.ones(r);
        IDiffVector dist1 = U1.matmul(LT).square().matmul(ones_r);
        IDiffVector dist2 = U2.matmul(LT).square().matmul(ones_r);
        return dist1.add(margin).sub(dist2).relu().mean();
    }

    @Test
    void testLmnnGradientNumerical() {
        double[][] u1Data = { { 1, 0 }, { 2, 1 } };
        double[][] u2Data = { { 0, 1 }, { 0, 2 } };
        int r = 2, d = 2;
        double margin = 1.0;
        double[] wArr = { 1, 0, 0, 1 };

        double eps = 1e-5;
        for (int i = 0; i < wArr.length; i++) {
            double[] wp = wArr.clone();
            wp[i] += eps;
            IDiffVector vp = AD.vector(wp);
            double fp = buildLmnnLoss(vp, u1Data, u2Data, r, d, margin).getValue().get(0);

            double[] wm = wArr.clone();
            wm[i] -= eps;
            IDiffVector vm = AD.vector(wm);
            double fm = buildLmnnLoss(vm, u1Data, u2Data, r, d, margin).getValue().get(0);

            double numGrad = (fp - fm) / (2 * eps);

            IDiffVector v = AD.vector(wArr);
            IDiffVector loss = buildLmnnLoss(v, u1Data, u2Data, r, d, margin);
            loss.backward();

            assertEquals(numGrad, v.getGradient().get(i), TOL);
        }
    }

    // ---- DML-eig style exp-quadratic loss ----

    private IDiffVector buildDmleigLoss(IDiffVector w, double[][] uData, int d, double mu) {
        IDiffMatrix M = w.reshape(d, d);
        IDiffMatrix U = AD.matrix(uData);
        IDiffMatrix UM = U.matmul(M);
        IDiffVector quads = UM.mul(U).matmul(AD.ones(d));
        return quads.div(-mu).exp().mean();
    }

    @Test
    void testDmleigGradientNumerical() {
        double[][] uData = { { 1, 0 }, { 0, 1 }, { 1, 1 } };
        int d = 2;
        double mu = 0.5;
        double[] wArr = { 2, 0, 0, 2 };

        double eps = 1e-5;
        for (int i = 0; i < wArr.length; i++) {
            double[] wp = wArr.clone();
            wp[i] += eps;
            IDiffVector vp = AD.vector(wp);
            double fp = buildDmleigLoss(vp, uData, d, mu).getValue().get(0);

            double[] wm = wArr.clone();
            wm[i] -= eps;
            IDiffVector vm = AD.vector(wm);
            double fm = buildDmleigLoss(vm, uData, d, mu).getValue().get(0);

            double numGrad = (fp - fm) / (2 * eps);

            IDiffVector v = AD.vector(wArr);
            IDiffVector loss = buildDmleigLoss(v, uData, d, mu);
            loss.backward();

            assertEquals(numGrad, v.getGradient().get(i), TOL);
        }
    }

    // ---- Edge cases ----

    @Test
    void testLdmlGradientScalarCase() {
        // r=1 case: L is 1×d
        double[][] uData = { { 2.0, 3.0 }, { 1.0, 0.5 } };
        double[] labels = { 1, 0 };
        int r = 1, d = 2;
        double bias = 0.0;
        double[] wArr = { 0.3, 0.7 };

        double eps = 1e-5;
        for (int i = 0; i < wArr.length; i++) {
            double[] wp = wArr.clone();
            wp[i] += eps;
            IDiffVector vp = AD.vector(wp);
            double fp = buildLdmlLoss(vp, uData, labels, r, d, bias).getValue().get(0);

            double[] wm = wArr.clone();
            wm[i] -= eps;
            IDiffVector vm = AD.vector(wm);
            double fm = buildLdmlLoss(vm, uData, labels, r, d, bias).getValue().get(0);

            double numGrad = (fp - fm) / (2 * eps);

            IDiffVector v = AD.vector(wArr);
            IDiffVector loss = buildLdmlLoss(v, uData, labels, r, d, bias);
            loss.backward();

            assertEquals(numGrad, v.getGradient().get(i), 1e-3);
        }
    }

    @Test
    void testLmnnGradientAllInactive() {
        // All triplets have hinge <= 0 (large margin, dist1 << dist2)
        double[][] u1Data = { { 0.1, 0.1 } };
        double[][] u2Data = { { 3, 3 } };
        int r = 2, d = 2;
        double margin = 0.01;
        double[] wArr = { 1, 0, 0, 1 };

        IDiffVector v = AD.vector(wArr);
        IDiffVector loss = buildLmnnLoss(v, u1Data, u2Data, r, d, margin);
        loss.backward();
        // All gradients should be zero (no active triplets)
        for (int i = 0; i < wArr.length; i++) {
            assertEquals(0.0, v.getGradient().get(i), 1e-10);
        }
    }
}
