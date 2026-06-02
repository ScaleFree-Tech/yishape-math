package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;

/**
 * PowerTransformer (幂变换预处理器)
 * <p>
 * 支持两种幂变换方法：
 * - Yeo-Johnson: 适用于包含负值的数据
 * - Box-Cox: 仅适用于严格正值的数据
 * </p>
 * <p>
 * 变换后，数据趋向于服从正态分布。
 * </p>
 *
 * @author lteb2
 */
public class RerePowerTransformer implements ITransform<Double> {

    public enum Method { YEO_JOHNSON, BOX_COX }

    private IMatrix<Double> feature;
    private Method method = Method.YEO_JOHNSON;
    private double[] lambdas;
    private boolean trained = false;

    public RerePowerTransformer() {
    }

    public RerePowerTransformer(Method method) {
        this.method = method;
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

        if (method == Method.BOX_COX) {
            for (int i = 0; i < feature.rows(); i++) {
                for (int j = 0; j < feature.cols(); j++) {
                    if (feature.get(i, j) <= 0) {
                        throw new IllegalArgumentException(
                            "Box-Cox transformation requires strictly positive values");
                    }
                }
            }
        }

        this.feature = feature;
        int n = feature.cols();
        int m = feature.rows();
        lambdas = new double[n];

        for (int j = 0; j < n; j++) {
            lambdas[j] = optimizeLambda(feature.getColumn(j).toDoubleArray());
        }

        trained = true;
        return this;
    }

    private double optimizeLambda(double[] values) {
        double minLambda = -5.0;
        double maxLambda = 5.0;
        double bestLambda = 0.0;
        double bestLogLikelihood = Double.NEGATIVE_INFINITY;

        for (int i = 0; i <= 1000; i++) {
            double lambda = minLambda + (maxLambda - minLambda) * i / 1000.0;
            double logLikelihood = computeLogLikelihood(values, lambda);
            if (logLikelihood > bestLogLikelihood) {
                bestLogLikelihood = logLikelihood;
                bestLambda = lambda;
            }
        }
        return bestLambda;
    }

    private double computeLogLikelihood(double[] values, double lambda) {
        if (method == Method.BOX_COX) {
            double sum = 0.0;
            for (double v : values) {
                if (v <= 0) return Double.NEGATIVE_INFINITY;
                sum += Math.pow(v, lambda);
            }
            double logSum = Math.log(sum / values.length);
            double ll = (lambda - 1) * logSum * values.length;
            return ll;
        } else {
            return computeYeoJohnsonLogLikelihood(values, lambda);
        }
    }

    private double computeYeoJohnsonLogLikelihood(double[] values, double lambda) {
        int n = values.length;
        double[] transformed = new double[n];
        for (int i = 0; i < n; i++) {
            transformed[i] = yeoJohnson(values[i], lambda);
        }

        double mean = 0.0;
        for (double v : transformed) {
            mean += v;
        }
        mean /= n;

        double variance = 0.0;
        for (double v : transformed) {
            variance += (v - mean) * (v - mean);
        }
        variance /= n;

        if (variance <= 0) return Double.NEGATIVE_INFINITY;

        return -n / 2.0 * Math.log(2 * Math.PI * variance);
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
                    result[i][j] = (float) (method == Method.BOX_COX
                        ? boxCox(val, lambdas[j])
                        : yeoJohnson(val, lambdas[j]));
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[rows][cols];
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    double val = feature.get(i, j);
                    result[i][j] = method == Method.BOX_COX
                        ? boxCox(val, lambdas[j])
                        : yeoJohnson(val, lambdas[j]);
                }
            }
            return IMatrix.of(result);
        }
    }

    private double boxCox(double x, double lambda) {
        if (lambda == 0) {
            return Math.log(x);
        }
        return (Math.pow(x, lambda) - 1) / lambda;
    }

    private double yeoJohnson(double x, double lambda) {
        if (x >= 0) {
            if (lambda == 0) {
                return Math.log(x + 1);
            }
            return (Math.pow(x + 1, lambda) - 1) / lambda;
        } else {
            if (lambda == 2) {
                return -Math.log(-x + 1);
            }
            return -((Math.pow(-x + 1, 2 - lambda) - 1) / (2 - lambda));
        }
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    public Method getMethod() {
        return method;
    }

    public double[] getLambdas() {
        return lambdas;
    }
}
