package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;

/**
 * MaxAbs 归一化预处理器
 * <p>
 * 按每个特征的最大绝对值进行缩放，使特征值位于 [-1, 1] 区间。
 * 公式：X_scaled = X / |X_max|，其中 X_max 是每列的最大绝对值
 * </p>
 * <p>
 * 自动识别输入数据类型（Double/Float），输出与输入类型一致。
 * </p>
 *
 * @author lteb2
 */
public class RereMaxAbsScaler implements IRereScaler<Double> {

    private IMatrix feature;
    private boolean isFloat = false;
    private double[] colMaxAbs;
    private boolean trained = false;

    public RereMaxAbsScaler() {
    }

    @Override
    public boolean ifTrained() {
        return trained;
    }

    @Override
    public ITransform<Double> fit(IMatrix feature) {
        if (feature == null || feature.rows() == 0 || feature.cols() == 0) {
            throw new IllegalArgumentException("Feature matrix cannot be null or empty");
        }
        this.feature = feature;
        isFloat = feature instanceof IFloatMatrix;
        int n = feature.cols();

        var colMinVec = feature.colMins().abs();
        var colMaxVec = feature.colMaxs();
        colMaxAbs = new double[n];
        for (int j = 0; j < n; j++) {
            colMaxAbs[j] = Math.max(colMinVec.get(j), colMaxVec.get(j));
        }
        trained = true;
        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        if (feature == null || feature.cols() != colMaxAbs.length) {
            throw new IllegalArgumentException("Feature matrix dimensions do not match fitted data");
        }

        int rows = feature.rows();
        int cols = feature.cols();

        if (isFloat) {
            float[][] result = new float[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    if (colMaxAbs[j] == 0) {
                        result[i][j] = 0.0f;
                    } else {
                        result[i][j] = (float) (feature.get(i, j) / colMaxAbs[j]);
                    }
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    if (colMaxAbs[j] == 0) {
                        result[i][j] = 0.0;
                    } else {
                        result[i][j] = feature.get(i, j) / colMaxAbs[j];
                    }
                }
            }
            return IMatrix.of(result);
        }
    }

    @Override
    public IMatrix inverseTransform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Scaler must be fitted before inverse transform");
        }
        if (feature == null) {
            throw new IllegalArgumentException("Feature matrix cannot be null");
        }

        int rows = feature.rows();
        int cols = feature.cols();

        if (isFloat) {
            float[][] result = new float[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    result[i][j] = (float) (feature.get(i, j) * colMaxAbs[j]);
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    result[i][j] = feature.get(i, j) * colMaxAbs[j];
                }
            }
            return IMatrix.of(result);
        }
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    public double[] getColMaxAbs() {
        return colMaxAbs;
    }
}
