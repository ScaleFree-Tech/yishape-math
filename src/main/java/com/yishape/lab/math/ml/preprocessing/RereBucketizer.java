package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.IVector;
import java.util.*;

/**
 * Bucketizer (分箱预处理器)
 * <p>
 * 将连续特征离散化到分箱中。
 * 支持等宽分箱和等频分箱。
 * </p>
 *
 * @author lteb2
 */
public class RereBucketizer implements ITransform<Double> {

    public enum Strategy { FIXED_WIDTH, QUANTILE }

    private IMatrix feature;
    private Strategy strategy = Strategy.FIXED_WIDTH;
    private int nBins = 5;
    private double[] binEdges;
    private double[] colMin;
    private double[] colMax;
    private boolean trained = false;

    public RereBucketizer() {
    }

    public RereBucketizer(int nBins) {
        this.nBins = nBins;
    }

    public RereBucketizer(Strategy strategy, int nBins) {
        this.strategy = strategy;
        this.nBins = nBins;
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
        int nFeatures = feature.cols();
        colMin = new double[nFeatures];
        colMax = new double[nFeatures];

        IVector<?> colMinVec = feature.colMins();
        IVector<?> colMaxVec = feature.colMaxs();
        for (int j = 0; j < nFeatures; j++) {
            colMin[j] = colMinVec.get(j);
            colMax[j] = colMaxVec.get(j);
        }

        if (strategy == Strategy.FIXED_WIDTH) {
            computeFixedWidthEdges();
        } else {
            computeQuantileEdges();
        }

        trained = true;
        return this;
    }

    private void computeFixedWidthEdges() {
        int nFeatures = feature.cols();
        binEdges = new double[(nBins + 1) * nFeatures];

        for (int j = 0; j < nFeatures; j++) {
            double range = colMax[j] - colMin[j];
            if (range == 0) {
                for (int k = 0; k <= nBins; k++) {
                    binEdges[j * (nBins + 1) + k] = colMin[j];
                }
            } else {
                double binWidth = range / nBins;
                for (int k = 0; k <= nBins; k++) {
                    binEdges[j * (nBins + 1) + k] = colMin[j] + k * binWidth;
                }
            }
        }
    }

    private void computeQuantileEdges() {
        int nFeatures = feature.cols();
        int nSamples = feature.rows();
        binEdges = new double[(nBins + 1) * nFeatures];

        for (int j = 0; j < nFeatures; j++) {
            double[] values = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                values[i] = feature.get(i, j);
            }
            Arrays.sort(values);

            for (int k = 0; k <= nBins; k++) {
                int idx = (int) ((k / (double) nBins) * (nSamples - 1));
                binEdges[j * (nBins + 1) + k] = values[idx];
            }
        }
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Bucketizer must be fitted before transform");
        }
        if (feature == null) {
            throw new IllegalArgumentException("Feature matrix cannot be null");
        }

        int rows = feature.rows();
        int cols = feature.cols();
        boolean isFloat = feature instanceof IFloatMatrix;

        if (isFloat) {
            float[][] result = new float[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    double val = feature.get(i, j);
                    result[i][j] = findBin(val, j);
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    double val = feature.get(i, j);
                    result[i][j] = findBin(val, j);
                }
            }
            return IMatrix.of(result);
        }
    }

    private int findBin(double value, int colIdx) {
        int offset = colIdx * (nBins + 1);
        for (int k = 0; k < nBins; k++) {
            double lower = binEdges[offset + k];
            double upper = binEdges[offset + k + 1];
            if (value >= lower && (value < upper || k == nBins - 1)) {
                return k;
            }
        }
        return nBins - 1;
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    public Strategy getStrategy() {
        return strategy;
    }

    public int getNBins() {
        return nBins;
    }

    public double[] getBinEdges() {
        return binEdges;
    }
}
