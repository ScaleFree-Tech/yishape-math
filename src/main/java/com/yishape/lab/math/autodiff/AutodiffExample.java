package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.util.YishapeLogger;

public class AutodiffExample {

    private static final YishapeLogger log = YishapeLogger.getLogger(AutodiffExample.class);

    public static void main(String[] args) {
        demoGradientDescent();
        demoOptimizerIntegration();
        demoMatrixAutodiff();
    }

    static void demoGradientDescent() {
        log.info("=== 1. 手动梯度下降: y = x^2 - 2 最小值 ===");

        double lr = 0.1;
        IDiffVector x = AD.vector(3.0);

        for (int i = 0; i < 10; i++) {
            x.zeroGradient();
            IDiffVector y = x.pow(2).sub(2);
            y.backward();
            double grad = x.getGradient().get(0);
            log.info("  iter " + (i + 1) + ": x=" + String.format("%.6f", x.getValue().get(0))
                    + ", grad=" + String.format("%.6f", grad));
            x = AD.vector(x.getValue().get(0) - lr * grad);
        }
        log.info("  结果: x=" + String.format("%.8f", x.getValue().get(0))
                + ", y=" + String.format("%.8f", x.getValue().get(0) * x.getValue().get(0) - 2));
    }

    static void demoOptimizerIntegration() {
        log.info("\n=== 2. 一行训练: AD.optimize() + L-BFGS ===");

        OptResult result = AD.optimize(
                Linalg.vector(3.0,2.9),
                x -> x.pow(2).sum().sub(2),
                Opts.lbfgs());

        log.info("  x_opt=" + result.getOptimalPoint()
                + ", f_opt=" + String.format("%.10f", result.getOptimalValue())
                + ", 迭代次数=" + result.getIterations());
    }

    static void demoMatrixAutodiff() {
        log.info("\n=== 3. 矩阵 autodiff: matmul + transpose ===");

        IDiffMatrix W = AD.matrix(new double[][] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0 });
        IDiffVector z = W.matmul(x);
        IDiffVector loss = z.pow(2).sum();

        loss.backward();

        log.info("  W = [[1,2],[3,4]], x = [1,2]");
        log.info("  z = W @ x = " + vecStr(z.getValue()));
        log.info("  loss = sum(z^2) = " + String.format("%.4f", loss.getValue().get(0)));
        log.info("  d(loss)/dx = " + vecStr(x.getGradient()));
        log.info("  d(loss)/dW = " + matStr(W.getGradient()));
    }

    private static String vecStr(IDoubleVector v) {
        StringBuilder sb = new StringBuilder("[");
        double[] d = v.getData();
        for (int i = 0; i < d.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(String.format("%.2f", d[i]));
        }
        return sb.append("]").toString();
    }

    private static String matStr(IDoubleMatrix m) {
        StringBuilder sb = new StringBuilder("[");
        double[][] d = m.getData();
        for (int i = 0; i < d.length; i++) {
            if (i > 0) sb.append("; ");
            for (int j = 0; j < d[i].length; j++) {
                if (j > 0) sb.append(",");
                sb.append(String.format("%.2f", d[i][j]));
            }
        }
        return sb.append("]").toString();
    }
}
