package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;
import java.util.*;

/**
 * LabelBinarizer (标签二值化预处理器)
 * <p>
 * 将标签数组二值化。对于多标签问题，每个标签对应一列。
 * 支持单个标签和多个标签的情况。
 * </p>
 *
 * @author lteb2
 */
public class RereLabelBinarizer implements ITransform<Double> {

    private IMatrix feature;
    private List<Double> classes;
    private Map<Double, Integer> classToIndex;
    private boolean trained = false;

    public RereLabelBinarizer() {
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

        Set<Double> uniqueValues = new HashSet<>();
        for (int i = 0; i < feature.rows(); i++) {
            for (int j = 0; j < feature.cols(); j++) {
                uniqueValues.add(feature.get(i, j));
            }
        }

        classes = new ArrayList<>(uniqueValues);
        Collections.sort(classes);

        classToIndex = new HashMap<>();
        for (int i = 0; i < classes.size(); i++) {
            classToIndex.put(classes.get(i), i);
        }

        trained = true;
        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Binarizer must be fitted before transform");
        }
        if (feature == null || feature.rows() == 0) {
            throw new IllegalArgumentException("Feature matrix cannot be null or empty");
        }

        int nSamples = feature.rows();
        int nClasses = classes.size();
        boolean isFloat = feature instanceof IFloatMatrix;

        if (isFloat) {
            float[][] result = new float[nSamples][nClasses];
            for (int i = 0; i < nSamples; i++) {
                Set<Integer> activeClasses = new HashSet<>();
                for (int j = 0; j < feature.cols(); j++) {
                    Double label = feature.get(i, j);
                    Integer idx = classToIndex.get(label);
                    if (idx != null) {
                        activeClasses.add(idx);
                    }
                }
                for (Integer idx : activeClasses) {
                    result[i][idx] = 1.0f;
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[nSamples][nClasses];
            for (int i = 0; i < nSamples; i++) {
                Set<Integer> activeClasses = new HashSet<>();
                for (int j = 0; j < feature.cols(); j++) {
                    Double label = feature.get(i, j);
                    Integer idx = classToIndex.get(label);
                    if (idx != null) {
                        activeClasses.add(idx);
                    }
                }
                for (Integer idx : activeClasses) {
                    result[i][idx] = 1.0;
                }
            }
            return IMatrix.of(result);
        }
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    public List<Double> getClasses() {
        return classes;
    }

    public IMatrix inverseTransform(IMatrix Y) {
        if (!trained) {
            throw new IllegalStateException("Binarizer must be fitted before inverse transform");
        }

        int nSamples = Y.rows();
        int nFeatures = Y.cols();
        boolean isFloat = Y instanceof IFloatMatrix;

        if (isFloat) {
            float[][] result = new float[nSamples][1];
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    if (Y.get(i, j) > 0.5) {
                        result[i][0] = classes.get(j).floatValue();
                        break;
                    }
                }
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[nSamples][1];
            for (int i = 0; i < nSamples; i++) {
                for (int j = 0; j < nFeatures; j++) {
                    if (Y.get(i, j) > 0.5) {
                        result[i][0] = classes.get(j);
                        break;
                    }
                }
            }
            return IMatrix.of(result);
        }
    }
}
