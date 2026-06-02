package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;

/**
 * Binarizer (阈值二值化预处理器)
 * <p>
 * 将特征值按照阈值二值化为0或1。
 * 公式：X_binary = 1 if X >= threshold else 0
 * </p>
 *
 * @author lteb2
 */
public class RereBinarizer implements ITransform<Double> {

    private IMatrix feature;
    private boolean isFloat = false;
    private double threshold = 0.0;

    public RereBinarizer() {
    }

    public RereBinarizer(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public boolean ifTrained() {
        return feature != null;
    }

    @Override
    public ITransform<Double> fit(IMatrix feature) {
        if (feature == null || feature.rows() == 0 || feature.cols() == 0) {
            throw new IllegalArgumentException("Feature matrix cannot be null or empty");
        }
        this.feature = feature;
        this.isFloat = feature instanceof IFloatMatrix;
        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (feature == null) {
            throw new IllegalArgumentException("Feature matrix cannot be null");
        }

        int rows = feature.rows();
        int cols = feature.cols();

        if (isFloat) {
            float[][] result = new float[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] = feature.get(i, j) >= threshold ? 1.0f : 0.0f;
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i][j] = feature.get(i, j) >= threshold ? 1.0 : 0.0;
                }
            }
            return IMatrix.of(result);
        }
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    public double getThreshold() {
        return threshold;
    }
}
