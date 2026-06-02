package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;

/**
 * Min-Max 归一化预处理器
 * <p>
 * 将特征缩放到指定范围 [min, max]，默认范围为 [0, 1]。
 * 公式：X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
 * </p>
 * <p>
 * 自动识别输入数据类型（Double/Float），输出与输入类型一致。
 * </p>
 *
 * @author lteb2
 */
public class RereMinMaxScaler implements IRereScaler<Double> {

    private IMatrix feature;
    private boolean isFloat = false;
    private double[] colMin;
    private double[] colMax;
    private double targetMin = 0.0;
    private double targetMax = 1.0;
    private boolean trained = false;

    public RereMinMaxScaler() {
    }

    public RereMinMaxScaler(double targetMin, double targetMax) {
        this.targetMin = targetMin;
        this.targetMax = targetMax;
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

        var colMinVec = feature.colMins();
        var colMaxVec = feature.colMaxs();
        colMin = new double[n];
        colMax = new double[n];
        for (int j = 0; j < n; j++) {
            colMin[j] = colMinVec.get(j);
            colMax[j] = colMaxVec.get(j);
        }
        trained = true;
        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        if (feature == null || feature.cols() != colMin.length) {
            throw new IllegalArgumentException("Feature matrix dimensions do not match fitted data");
        }

        int rows = feature.rows();
        int cols = feature.cols();
        double range = targetMax - targetMin;

        if (isFloat) {
            float[][] result = new float[rows][cols];
            for (int j = 0; j < cols; j++) {
                float dataRange = (float) (colMax[j] - colMin[j]);
                for (int i = 0; i < rows; i++) {
                    if (dataRange == 0) {
                        result[i][j] = (float) targetMin;
                    } else {
                        result[i][j] = (float) ((feature.get(i, j) - colMin[j]) / dataRange * range + targetMin);
                    }
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                double dataRange = colMax[j] - colMin[j];
                for (int i = 0; i < rows; i++) {
                    if (dataRange == 0) {
                        result[i][j] = targetMin;
                    } else {
                        result[i][j] = (feature.get(i, j) - colMin[j]) / dataRange * range + targetMin;
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
        double range = targetMax - targetMin;

        if (isFloat) {
            float[][] result = new float[rows][cols];
            for (int j = 0; j < cols; j++) {
                float dataRange = (float) (colMax[j] - colMin[j]);
                for (int i = 0; i < rows; i++) {
                    if (range == 0) {
                        result[i][j] = (float) colMin[j];
                    } else {
                        result[i][j] = (float) (feature.get(i, j) / range * dataRange + colMin[j]);
                    }
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                double dataRange = colMax[j] - colMin[j];
                for (int i = 0; i < rows; i++) {
                    if (range == 0) {
                        result[i][j] = colMin[j];
                    } else {
                        result[i][j] = feature.get(i, j) / range * dataRange + colMin[j];
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

    public double[] getColMin() {
        return colMin;
    }

    public double[] getColMax() {
        return colMax;
    }

    public double getTargetMin() {
        return targetMin;
    }

    public double getTargetMax() {
        return targetMax;
    }
}
