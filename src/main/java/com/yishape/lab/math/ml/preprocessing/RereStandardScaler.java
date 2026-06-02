package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.data.Column;
import com.yishape.lab.math.data.ColumnType;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.util.ArrayList;
import java.util.List;

/**
 * StandardScaler (Z-Score 标准化预处理器)
 * <p>
 * 将特征标准化为均值为0、标准差为1的分布。
 * 公式：X_scaled = (X - mean) / std
 * </p>
 * <p>
 * 使用系统内部 API (IMatrix.colMeans(), IVector.std(ddof)) 进行计算，
 * 默认使用样本标准差 (ddof=1)，符合统计学常识。
 * </p>
 * <p>
 * 自动识别输入数据类型（Double/Float），输出与输入类型一致。
 * </p>
 * <p>
 * 支持 DataFrame 操作：通过 {@link IDataFrameTransform} 接口，
 * 可直接对 DataFrame 指定列进行 fit/transform。
 * </p>
 *
 * @author lteb2
 */
public class RereStandardScaler implements IRereScaler<Double>, IDataFrameTransform<Double> {

    private IMatrix<Double> feature;
    private boolean isFloat = false;
    private double[] colMean;
    private double[] colStd;
    private boolean trained = false;
    private boolean withStd = true;
    private String[] columns;

    public RereStandardScaler() {
    }

    public RereStandardScaler(boolean withStd) {
        this.withStd = withStd;
    }

    public RereStandardScaler columns(String... columns) {
        this.columns = columns;
        return this;
    }

    public boolean isWithStd() {
        return withStd;
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

        // 使用系统 API: IMatrix.colMeans() 获取每列均值
        IVector<Double> means = feature.colMeans();
        colMean = new double[n];
        for (int j = 0; j < n; j++) {
            colMean[j] = means.get(j);
        }

        if (withStd) {
            colStd = new double[n];
            for (int j = 0; j < n; j++) {
                // 使用系统 API: IVector.std(1) 获取样本标准差 (ddof=1)
                double std = feature.getColumn(j).stdValue(1);
                colStd[j] = (std == 0) ? 1.0 : std;
            }
        } else {
            colStd = new double[n];
            java.util.Arrays.fill(colStd, 1.0);
        }

        trained = true;
        return this;
    }

    @Override
    public IMatrix<Double> transform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }
        if (feature == null || feature.cols() != colMean.length) {
            throw new IllegalArgumentException("Feature matrix dimensions do not match fitted data");
        }

        int rows = feature.rows();
        int cols = feature.cols();

            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    result[i][j] = (feature.get(i, j) - colMean[j]) / colStd[j];
                }
            }
            return IMatrix.of(result);
        
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
                    result[i][j] = feature.get(i, j) * colStd[j] + colMean[j];
                }
            }
            return IMatrix.of(result);
        
    }

    @Override
    public IMatrix<Double> getFeature() {
        return feature;
    }

    public double[] getColMean() {
        return colMean;
    }

    public double[] getColStd() {
        return colStd;
    }

    // ==================== IDataFrameTransform 实现 ====================

    @Override
    public String[] getInputColumnNames() {
        return columns != null ? columns : new String[0];
    }

    @Override
    public String[] getOutputColumnNames() {
        return columns;
    }

    @Override
    public DataFrame transform(DataFrame df, String... colNames) {
        if (!trained) {
            throw new IllegalStateException("Scaler must be fitted before transform");
        }

        DataFrame result = new DataFrame();

        for (Column col : df.getColumns()) {
            boolean isTarget = false;
            int targetIdx = -1;
            for (int i = 0; i < colNames.length; i++) {
                if (col.getName().equals(colNames[i])) {
                    isTarget = true;
                    targetIdx = i;
                    break;
                }
            }

            if (isTarget) {
                Column newCol = new Column();
                newCol.setName(col.getName());
                newCol.setColumnType(ColumnType.Numeric);
                List<Object> data = new ArrayList<>();
                for (Object v : col.getData()) {
                    if (v instanceof Number num) {
                        double val = num.doubleValue();
                        data.add((val - colMean[targetIdx]) / colStd[targetIdx]);
                    } else {
                        data.add(v);
                    }
                }
                newCol.setData(data);
                result.addColumn(newCol);
            } else {
                result.addColumn(col);
            }
        }

        return result;
    }

    @Override
    public DataFrame fitTransform(DataFrame df, String... columns) {
        this.columns = columns;
        DataFrame subset = df.selectColumns(List.of(columns));
        fit(subset.toMatrix());
        return transform(df, columns);
    }

    @Override
    public IDataFrameTransform<Double> fit(DataFrame df, String... columns) {
        this.columns = columns;
        DataFrame subset = df.selectColumns(List.of(columns));
        return (IDataFrameTransform<Double>) fit(subset.toMatrix());
    }
}
