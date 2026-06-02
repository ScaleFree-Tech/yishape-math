package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;
import java.util.ArrayList;
import java.util.List;

/**
 * PolynomialFeatures (多项式特征生成器)
 * <p>
 * 从原始特征生成多项式特征，支持生成交互项和高阶项。
 * </p>
 * <p>
 * 例如对于2个特征 [x1, x2]，degree=2 时生成：
 * [1, x1, x2, x1^2, x1*x2, x2^2]
 * </p>
 *
 * @author lteb2
 */
public class RerePolynomialFeatures implements ITransform<Double> {

    private IMatrix feature;
    private int degree = 2;
    private boolean includeBias = true;
    private boolean interactionOnly = false;

    public RerePolynomialFeatures() {
    }

    public RerePolynomialFeatures(int degree) {
        this.degree = degree;
    }

    public RerePolynomialFeatures(int degree, boolean includeBias, boolean interactionOnly) {
        this.degree = degree;
        this.includeBias = includeBias;
        this.interactionOnly = interactionOnly;
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
        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (feature == null || feature.rows() == 0 || feature.cols() == 0) {
            throw new IllegalArgumentException("Feature matrix cannot be null or empty");
        }

        int nSamples = feature.rows();
        int nOriginalFeatures = feature.cols();
        boolean isFloat = feature instanceof IFloatMatrix;

        List<double[]> polyFeatures = generatePolynomialFeatures(feature, nOriginalFeatures, nSamples);

        if (isFloat) {
            float[][] result = new float[nSamples][polyFeatures.size()];
            for (int j = 0; j < polyFeatures.size(); j++) {
                for (int i = 0; i < nSamples; i++) {
                    result[i][j] = (float) polyFeatures.get(j)[i];
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[nSamples][polyFeatures.size()];
            for (int j = 0; j < polyFeatures.size(); j++) {
                for (int i = 0; i < nSamples; i++) {
                    result[i][j] = polyFeatures.get(j)[i];
                }
            }
            return IMatrix.of(result);
        }
    }

    private List<double[]> generatePolynomialFeatures(IMatrix<Double> feature,
        int nOriginalFeatures, int nSamples) {

        List<double[]> features = new ArrayList<>();

        if (includeBias) {
            double[] bias = new double[nSamples];
            java.util.Arrays.fill(bias, 1.0);
            features.add(bias);
        }

        List<int[]> combinations = generateCombinations(nOriginalFeatures, degree, interactionOnly);

        for (int[] combo : combinations) {
            double[] values = new double[nSamples];
            for (int s = 0; s < nSamples; s++) {
                double val = 1.0;
                for (int idx : combo) {
                    val *= feature.get(s, idx);
                }
                values[s] = val;
            }
            features.add(values);
        }

        return features;
    }

    private List<int[]> generateCombinations(int nFeatures, int degree, boolean interactionOnly) {
        List<int[]> combinations = new ArrayList<>();
        generateCombinationsRecursive(new int[degree], 0, 0, nFeatures, degree, interactionOnly, combinations);
        return combinations;
    }

    private void generateCombinationsRecursive(int[] current, int depth, int start,
        int nFeatures, int degree, boolean interactionOnly, List<int[]> result) {

        if (depth == degree) {
            int[] combo = new int[degree];
            System.arraycopy(current, 0, combo, 0, degree);
            result.add(combo);
            return;
        }

        int actualStart = interactionOnly ? start : 0;
        for (int i = actualStart; i < nFeatures; i++) {
            current[depth] = i;
            generateCombinationsRecursive(current, depth + 1, i + 1, nFeatures, degree, interactionOnly, result);
        }
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    public int getDegree() {
        return degree;
    }

    public boolean isIncludeBias() {
        return includeBias;
    }

    public boolean isInteractionOnly() {
        return interactionOnly;
    }
}
