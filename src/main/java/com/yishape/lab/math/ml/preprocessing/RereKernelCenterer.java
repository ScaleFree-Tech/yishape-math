package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;

/**
 * KernelCenterer (核矩阵中心化)
 * <p>
 * 对核矩阵（Gram matrix）进行中心化处理。
 * 用于将核矩阵变换为具有零均值的核矩阵。
 * K_centered = K - 1_n K - K 1_n + 1_n K 1_n
 * </p>
 *
 * @author lteb2
 */
public class RereKernelCenterer implements ITransform<Double> {

    private IMatrix feature;
    private boolean isFloat = false;
    private boolean trained = false;

    @Override
    public boolean ifTrained() {
        return trained;
    }

    @Override
    public ITransform<Double> fit(IMatrix feature) {
        if (feature == null || feature.rows() == 0 || feature.cols() == 0) {
            throw new IllegalArgumentException("Feature matrix cannot be null or empty");
        }
        if (feature.rows() != feature.cols()) {
            throw new IllegalArgumentException("Kernel matrix must be square");
        }
        this.feature = feature;
        this.isFloat = feature instanceof IFloatMatrix;
        trained = true;
        return this;
    }

    @Override
    public IMatrix transform(IMatrix K) {
        if (!trained) {
            throw new IllegalStateException("Centerer must be fitted before transform");
        }
        if (K == null || K.rows() != K.cols()) {
            throw new IllegalArgumentException("Kernel matrix must be square");
        }

        int n = K.rows();

        if (isFloat) {
            IMatrix<Float> ones = IMatrix.ones(n, n).toFloatMatrix();
            IMatrix<Float> Kf = K.toFloatMatrix();
            IMatrix<Float> KCentered = Kf.sub(ones.mmul(Kf).divideByScalar((float) n));
            KCentered = KCentered.sub(Kf.mmul(ones).divideByScalar((float) n));
            KCentered = KCentered.add(ones.mmul(Kf).mmul(ones).divideByScalar((float) (n * n)));
            return KCentered;
        } else {
            IMatrix<Double> ones = IMatrix.ones(n, n);
            IMatrix<Double> Kd = K.toDoubleMatrix();
            IMatrix<Double> KCentered = Kd.sub(ones.mmul(Kd).divideByScalar((double) n));
            KCentered = KCentered.sub(Kd.mmul(ones).divideByScalar((double) n));
            KCentered = KCentered.add(ones.mmul(Kd).mmul(ones).divideByScalar((double) (n * n)));
            return KCentered;
        }
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }
}
