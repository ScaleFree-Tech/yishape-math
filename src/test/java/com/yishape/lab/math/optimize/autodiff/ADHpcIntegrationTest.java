package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.optimize.IOptimizer;

/**
 * End-to-end HPC graph execution tests. All tests are designed to pass
 * regardless of HPC availability — if native runtime is missing, the
 * Java fallback path is exercised transparently.
 */
public class ADHpcIntegrationTest {

    private static final double TOL = 1e-10;

    @Test
    void testHpcSquareSumGradient() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector loss = x.pow(2).sum();

        boolean hpcUsed = AD.tryHpcExecute(loss);
        if (!hpcUsed) {
            loss.backward();
        }

        double[] grad = x.getGradient().getData();
        assertArrayEquals(new double[] { 2, 4, 6 }, grad, TOL);
    }

    @Test
    void testHpcExpSumGradient() {
        IDiffVector x = AD.vector(new double[] { 0, 1, 2 });
        IDiffVector loss = x.exp().sum();

        boolean hpcUsed = AD.tryHpcExecute(loss);
        if (!hpcUsed) {
            loss.backward();
        }

        double[] grad = x.getGradient().getData();
        assertEquals(Math.exp(0), grad[0], TOL);
        assertEquals(Math.exp(1), grad[1], TOL);
        assertEquals(Math.exp(2), grad[2], TOL);
    }

    @Test
    void testHpcLogisticLoss() {
        // Binary logistic regression loss: -log(sigmoid(y * dot(w, x)))
        IDiffVector w = AD.vector(new double[] { 0.5, -0.3, 0.1 });
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, -1.0 });
        double yTrue = 1.0;

        IDiffVector logit = w.dot(x);
        // loss = -log(sigmoid(y * logit)) = log(1 + exp(-y * logit))
        IDiffVector yVec = AD.vector(yTrue);
        IDiffVector negYLogit = yVec.mul(-1).mul(logit);
        IDiffVector loss = negYLogit.exp().add(1).log();

        boolean hpcUsed = AD.tryHpcExecute(loss);
        if (!hpcUsed) {
            loss.backward();
        }
        double[] autoGrad = w.getGradient().getData();
        assertNotNull(autoGrad);

        // Verify with numerical gradient
        double[] wData = { 0.5, -0.3, 0.1 };
        double eps = 1e-6;
        for (int i = 0; i < wData.length; i++) {
            wData[i] += eps;
            double fp = logisticLoss(wData, new double[] { 1, 2, -1 }, 1.0);
            wData[i] -= 2 * eps;
            double fm = logisticLoss(wData, new double[] { 1, 2, -1 }, 1.0);
            wData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGrad[i], 1e-5);
        }
    }

    private double logisticLoss(double[] w, double[] x, double y) {
        double logit = 0;
        for (int i = 0; i < w.length; i++) {
            logit += w[i] * x[i];
        }
        return Math.log(1 + Math.exp(-y * logit));
    }

    @Test
    void testHpcMultiLeafGraph() {
        IDiffVector a = AD.vector(new double[] { 1.0, 2.0 });
        IDiffVector b = AD.vector(new double[] { 3.0, 4.0 });
        IDiffVector loss = a.mul(b).sum();

        boolean hpcUsed = AD.tryHpcExecute(loss);
        if (!hpcUsed) {
            loss.backward();
        }

        double[] gradA = a.getGradient().getData();
        double[] gradB = b.getGradient().getData();
        assertArrayEquals(new double[] { 3, 4 }, gradA, TOL);
        assertArrayEquals(new double[] { 1, 2 }, gradB, TOL);
    }

    @Test
    void testHpcFallbackToJava() {
        // Even without HPC, backward() produces correct gradients
        IDiffVector x = AD.vector(new double[] { 1, 2, 3, 4 });
        IDiffVector loss = x.square().mean();

        boolean hpcUsed = AD.tryHpcExecute(loss);
        if (!hpcUsed) {
            loss.backward();
        }

        double[] grad = x.getGradient().getData();
        int n = 4;
        for (int i = 0; i < n; i++) {
            assertEquals(2.0 * (i + 1) / n, grad[i], TOL);
        }
    }

    @Test
    void testOptimizeWithHpc() {
        // Minimize f(x) = sum((x - target)^2)
        double[] target = { 3.0, 4.0 };
        IDoubleVector initX = IDoubleVector.of(new double[] { 0.0, 0.0 });

        java.util.function.Function<IDiffVector, IDiffVector> lossBuilder = x -> {
            IDiffVector t = AD.vector(target);
            return x.sub(t).square().sum();
        };

        IOptimizer optimizer = com.yishape.lab.math.optimize.Opts.lbfgs();
        var result = AD.optimize(initX, lossBuilder, optimizer);

        assertNotNull(result);
        double[] xOpt = ((IDoubleVector) result.getOptimalPoint()).getData();
        assertEquals(target[0], xOpt[0], 1e-3);
        assertEquals(target[1], xOpt[1], 1e-3);
    }
}
