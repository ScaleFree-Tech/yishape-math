package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;
import java.util.Arrays;

/**
 * QuantileTransformer (分位数变换预处理器)
 * <p>
 * 将特征变换为均匀分布或正态分布的分位数。
 * 对于每个特征，变换后的值基于其经验累积分布函数。
 * </p>
 *
 * @author lteb2
 */
public class RereQuantileTransformer implements ITransform<Double> {

    public enum OutputDistribution { UNIFORM, NORMAL }

    private IMatrix<?> feature;
    private OutputDistribution outputDistribution = OutputDistribution.UNIFORM;
    private double[][] quantiles;
    private double[][] references;
    private int nQuantiles = 1000;
    private boolean trained = false;

    public RereQuantileTransformer() {
    }

    public RereQuantileTransformer(OutputDistribution outputDistribution) {
        this.outputDistribution = outputDistribution;
    }

    public RereQuantileTransformer(int nQuantiles) {
        this.nQuantiles = nQuantiles;
    }

    public RereQuantileTransformer(OutputDistribution outputDistribution, int nQuantiles) {
        this.outputDistribution = outputDistribution;
        this.nQuantiles = nQuantiles;
    }

    @Override
    public boolean ifTrained() {
        return trained;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ITransform<Double> fit(IMatrix feature) {
        if (feature == null || feature.rows() == 0 || feature.cols() == 0) {
            throw new IllegalArgumentException("Feature matrix cannot be null or empty");
        }

        this.feature = feature;
        int nSamples = feature.rows();
        int nFeatures = feature.cols();

        quantiles = new double[nFeatures][nQuantiles];
        references = new double[nFeatures][nQuantiles];

        for (int j = 0; j < nFeatures; j++) {
            double[] values = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                values[i] = feature.get(i, j);
            }
            Arrays.sort(values);

            for (int k = 0; k < nQuantiles; k++) {
                double frac = (k + 1.0) / nQuantiles;
                int idx = (int) (frac * (nSamples - 1));
                double alpha = frac * (nSamples - 1) - idx;
                quantiles[j][k] = idx < nSamples - 1
                    ? values[idx] * (1 - alpha) + values[idx + 1] * alpha
                    : values[idx];
            }

            for (int k = 0; k < nQuantiles; k++) {
                references[j][k] = (k + 1.0) / nQuantiles;
            }

            if (outputDistribution == OutputDistribution.NORMAL) {
                for (int k = 0; k < nQuantiles; k++) {
                    references[j][k] = normalInverseCDF((k + 1.0) / nQuantiles);
                }
            }
        }

        trained = true;
        return this;
    }

    private double normalInverseCDF(double p) {
        if (p <= 0) return Double.NEGATIVE_INFINITY;
        if (p >= 1) return Double.POSITIVE_INFINITY;

        double[] a = {
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00
        };
        double[] b = {
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01
        };
        double[] c = {
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00
        };
        double[] d = {
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00
        };

        double pLow = 0.02425;
        double pHigh = 1 - pLow;

        double q, r;

        if (p < pLow) {
            q = Math.sqrt(-2 * Math.log(p));
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
        } else if (p <= pHigh) {
            q = p - 0.5;
            r = q * q;
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
        } else {
            q = Math.sqrt(-2 * Math.log(1 - p));
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
        }
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Transformer must be fitted before transform");
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
                    result[i][j] = (float) computeECDF(val, quantiles[j], references[j]);
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    double val = feature.get(i, j);
                    result[i][j] = computeECDF(val, quantiles[j], references[j]);
                }
            }
            return IMatrix.of(result);
        }
    }

    private double computeECDF(double x, double[] sortedValues, double[] refQuantiles) {
        int n = sortedValues.length;
        int idx = Arrays.binarySearch(sortedValues, x);
        if (idx < 0) {
            idx = -idx - 1;
        }
        idx = Math.min(idx, n - 1);
        return refQuantiles[Math.max(0, Math.min(idx, n - 1))];
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    public OutputDistribution getOutputDistribution() {
        return outputDistribution;
    }

    public int getNQuantiles() {
        return nQuantiles;
    }
}
