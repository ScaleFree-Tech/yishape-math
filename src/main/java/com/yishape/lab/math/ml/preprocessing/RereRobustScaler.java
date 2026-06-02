package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.IVector;

/**
 * RobustScaler (鲁棒标准化预处理器)
 * <p>
 * 使用中位数和四分位距(IQR)进行缩放，对异常值具有较强的鲁棒性。
 * 公式：X_scaled = (X - median) / IQR
 * 其中 IQR = Q3 - Q1 (第三四分位数 - 第一四分位数)
 * </p>
 * <p>
 * 使用系统内部 API (IVector.median(), IVector.percentile()) 进行计算。
 * </p>
 * <p>
 * 自动识别输入数据类型（Double/Float），输出与输入类型一致。
 * </p>
 *
 * @author lteb2
 */
public class RereRobustScaler implements IRereScaler<Double> {

    private IMatrix<Double> feature;
    private boolean isFloat = false;
    private double[] colMedian;
    private double[] colIQR;
    private boolean trained = false;

    public RereRobustScaler() {
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
//        isFloat = feature instanceof IFloatMatrix;
        isFloat = false;
        int n = feature.cols();
        colMedian = new double[n];
        colIQR = new double[n];

        for (int j = 0; j < n; j++) {
            // 使用系统 API: IVector.median() 和 IVector.percentile()
            IVector<Double> col = feature.getColumn(j);
            colMedian[j] = col.median();
            double q1 = col.percentile(25.0);
            double q3 = col.percentile(75.0);
            colIQR[j] = q3 - q1;
        }
        trained = true;
        return this;
    }

    @Override
    public IMatrix<Double> transform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        if (feature == null || feature.cols() != colMedian.length) {
            throw new IllegalArgumentException("Feature matrix dimensions do not match fitted data");
        }

        int rows = feature.rows();
        int cols = feature.cols();

        if (isFloat) {
            float[][] result = new float[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    if (colIQR[j] == 0) {
                        result[i][j] = 0.0f;
                    } else {
                        result[i][j] = (float) ((feature.get(i, j) - colMedian[j]) / colIQR[j]);
                    }
                }
            }
            return IMatrix.of(result).toDoubleMatrix();
        } else {
            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    if (colIQR[j] == 0) {
                        result[i][j] = 0.0;
                    } else {
                        result[i][j] = (feature.get(i, j) - colMedian[j]) / colIQR[j];
                    }
                }
            }
            return IMatrix.of(result);
        }
    }

    @Override
    public IMatrix<Double> inverseTransform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Scaler must be fitted before inverse transform");
        }
        if (feature == null) {
            throw new IllegalArgumentException("Feature matrix cannot be null");
        }

        int rows = feature.rows();
        int cols = feature.cols();

            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    result[i][j] = feature.get(i, j) * colIQR[j] + colMedian[j];
                }
            }
            return IMatrix.of(result);
        
    }

    @Override
    public IMatrix<Double> getFeature() {
        return feature;
    }

    public double[] getColMedian() {
        return colMedian;
    }

    public double[] getColIQR() {
        return colIQR;
    }
}
