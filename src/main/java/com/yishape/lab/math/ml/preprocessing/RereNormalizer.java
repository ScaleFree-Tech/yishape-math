package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.IDoubleMatrix;

/**
 * Normalizer (逐样本归一化)
 * <p>
 * 对每个样本（行）独立进行归一化，使其范数为指定值。
 * 与 StandardScaler/MinMaxScaler 的列级别归一化不同，Normalizer 是行级别归一化。
 * </p>
 * <p>
 * 使用系统内部 API (IVector.norm1(), IVector.norm2(), IVector.normInf()) 进行计算。
 * </p>
 * <p>
 * 支持的范数类型：
 * - L1: 元素绝对值之和为1
 * - L2: 欧几里得范数为1
 * - MAX: 最大绝对值为1
 * </p>
 *
 * @author lteb2
 */
public class RereNormalizer implements ITransform<Double> {

    public enum Norm { L1, L2, MAX }

    private IMatrix<Double> feature;
    private boolean isFloat = false;
    private Norm norm = Norm.L2;

    public RereNormalizer() {
    }

    public RereNormalizer(Norm norm) {
        this.norm = norm;
    }

    public static RereNormalizer l1() {
        return new RereNormalizer(Norm.L1);
    }

    public static RereNormalizer l2() {
        return new RereNormalizer(Norm.L2);
    }

    public static RereNormalizer max() {
        return new RereNormalizer(Norm.MAX);
    }

    @Override
    public boolean ifTrained() {
        return feature != null;
    }

    @Override
    @SuppressWarnings("unchecked")
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
        if (feature == null || feature.rows() == 0 || feature.cols() == 0) {
            throw new IllegalArgumentException("Feature matrix cannot be null or empty");
        }

        int rows = feature.rows();
        int cols = feature.cols();

        if (isFloat) {
            float[][] result = new float[rows][cols];
            for (int i = 0; i < rows; i++) {
                IVector<Double> row = feature.getRow(i);
                double normVal;
                normVal = switch (norm) {
                    case L1 -> row.norm1Value();
                    case L2 -> row.norm2Value();
                    case MAX -> row.normInf();
                    default -> 0;
                };
                if (normVal == 0) {
                    for (int j = 0; j < cols; j++) {
                        result[i][j] = 0.0f;
                    }
                } else {
                    for (int j = 0; j < cols; j++) {
                        result[i][j] = (float) (feature.get(i, j) / normVal);
                    }
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                IVector<Double> row = feature.getRow(i);
                double normVal;
                normVal = switch (norm) {
                    case L1 -> row.norm1Value();
                    case L2 -> row.norm2Value();
                    case MAX -> row.normInf();
                    default -> 0;
                };
                if (normVal == 0) {
                    for (int j = 0; j < cols; j++) {
                        result[i][j] = 0.0;
                    }
                } else {
                    for (int j = 0; j < cols; j++) {
                        result[i][j] = feature.get(i, j) / normVal;
                    }
                }
            }
            return IMatrix.of(result);
        }
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    public Norm getNorm() {
        return norm;
    }
}
