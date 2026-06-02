package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.data.Column;
import com.yishape.lab.math.data.ColumnType;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.Linalg;

import java.util.*;

/**
 * OneHotEncoder (独热编码预处理器)
 * <p>
 * 将分类特征编码为独热格式。每个类别对应一列。
 * 支持处理未知类别（可通过 handleUnknown 配置）。
 * </p>
 * <p>
 * 支持 DataFrame 操作：通过 {@link IDataFrameTransform} 接口，
 * 可直接对 DataFrame 指定列进行 fit/transform。
 * </p>
 *
 * @author lteb2
 */
public class RereOneHotEncoder implements ITransform<Double>, IDataFrameTransform<Double> {

    public enum HandleUnknown { IGNORE, ERROR }

    private IMatrix feature;
    private List<Double>[] categories;
    private Map<Integer, Map<Double, Integer>> categoryToIndex;
    private HandleUnknown handleUnknown = HandleUnknown.ERROR;
    private boolean trained = false;
    private String[] columns;

    // DataFrame 专用：存储原始类别值（支持 String 等非数值类型）
    private List<?>[] dfCategories;
    private Map<Integer, Map<Object, Integer>> dfCategoryToIndex;

    @Override
    public boolean ifTrained() {
        return trained;
    }

    @SuppressWarnings("unchecked")
    @Override
    public ITransform<Double> fit(IMatrix feature) {
        if (feature == null || feature.rows() == 0 || feature.cols() == 0) {
            throw new IllegalArgumentException("Feature matrix cannot be null or empty");
        }

        this.feature = feature;
        int nFeatures = feature.cols();

        categories = new List[nFeatures];
        categoryToIndex = new HashMap<>();

        for (int j = 0; j < nFeatures; j++) {
            Set<Double> uniqueValues = new HashSet<>();
            for (int i = 0; i < feature.rows(); i++) {
                uniqueValues.add(feature.get(i, j));
            }

            List<Double> sortedCategories = new ArrayList<>(uniqueValues);
            Collections.sort(sortedCategories);
            categories[j] = sortedCategories;

            Map<Double, Integer> catMap = new HashMap<>();
            for (int i = 0; i < sortedCategories.size(); i++) {
                catMap.put(sortedCategories.get(i), i);
            }
            categoryToIndex.put(j, catMap);
        }

        trained = true;
        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!trained) {
            throw new IllegalStateException("Encoder must be fitted before transform");
        }
        if (feature == null || feature.rows() == 0 || feature.cols() != this.feature.cols()) {
            throw new IllegalArgumentException("Feature matrix dimensions do not match fitted data");
        }

        int nSamples = feature.rows();
        int[] nCategoriesPerFeature = new int[this.feature.cols()];
        int totalCategories = 0;

        for (int j = 0; j < this.feature.cols(); j++) {
            nCategoriesPerFeature[j] = categories[j].size();
            totalCategories += nCategoriesPerFeature[j];
        }

        boolean isFloat = feature instanceof IFloatMatrix;

        if (isFloat) {
            float[][] result = new float[nSamples][totalCategories];
            int columnOffset = 0;
            for (int j = 0; j < this.feature.cols(); j++) {
                Map<Double, Integer> catMap = categoryToIndex.get(j);
                for (int i = 0; i < nSamples; i++) {
                    double value = feature.get(i, j);
                    Integer idx = catMap.get(value);
                    if (idx == null) {
                        if (handleUnknown == HandleUnknown.ERROR) {
                            throw new IllegalArgumentException("Unknown category: " + value);
                        }
                    } else {
                        result[i][columnOffset + idx] = 1.0f;
                    }
                }
                columnOffset += nCategoriesPerFeature[j];
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[nSamples][totalCategories];
            int columnOffset = 0;
            for (int j = 0; j < this.feature.cols(); j++) {
                Map<Double, Integer> catMap = categoryToIndex.get(j);
                for (int i = 0; i < nSamples; i++) {
                    double value = feature.get(i, j);
                    Integer idx = catMap.get(value);
                    if (idx == null) {
                        if (handleUnknown == HandleUnknown.ERROR) {
                            throw new IllegalArgumentException("Unknown category: " + value);
                        }
                    } else {
                        result[i][columnOffset + idx] = 1.0;
                    }
                }
                columnOffset += nCategoriesPerFeature[j];
            }
            return IMatrix.of(result);
        }
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    public List<Double>[] getCategories() {
        return categories;
    }

    public HandleUnknown getHandleUnknown() {
        return handleUnknown;
    }

    public void setHandleUnknown(HandleUnknown handleUnknown) {
        this.handleUnknown = handleUnknown;
    }

    public RereOneHotEncoder columns(String... columns) {
        this.columns = columns;
        return this;
    }

    // ==================== IDataFrameTransform 实现 ====================

    @Override
    public String[] getInputColumnNames() {
        return columns != null ? columns : new String[0];
    }

    public String[][] getFeatureNames() {
        if (!trained || categories == null) return null;
        String[][] names = new String[categories.length][];
        for (int j = 0; j < categories.length; j++) {
            List<Double> cats = categories[j];
            names[j] = new String[cats.size()];
            for (int k = 0; k < cats.size(); k++) {
                names[j][k] = columns[j] + "_" + cats.get(k);
            }
        }
        return names;
    }

    public int[] getCategoryCounts() {
        if (categories == null) return new int[0];
        int[] counts = new int[categories.length];
        for (int j = 0; j < categories.length; j++) {
            counts[j] = categories[j].size();
        }
        return counts;
    }

    @Override
    public DataFrame transform(DataFrame df, String... colNames) {
        if (!trained) {
            throw new IllegalStateException("Encoder must be fitted before transform");
        }

        DataFrame result = new DataFrame();

        // 复制未编码的列
        for (Column col : df.getColumns()) {
            boolean isTarget = false;
            for (String encCol : columns) {
                if (col.getName().equals(encCol)) {
                    isTarget = true;
                    break;
                }
            }
            if (!isTarget) {
                result.addColumn(col);
            }
        }

        // 添加编码后的列
        for (int j = 0; j < dfCategories.length; j++) {
            List<?> cats = dfCategories[j];
            Column original = df.getColumnByName(columns[j]);

            for (Object cat : cats) {
                String colName = columns[j] + "_" + cat;
                Column newCol = new Column();
                newCol.setName(colName);
                newCol.setColumnType(ColumnType.Numeric);

                List<Object> colData = new ArrayList<>();
                for (Object v : original.getData()) {
                    if (Objects.equals(v, cat)) {
                        colData.add(1.0);
                    } else {
                        colData.add(0.0);
                    }
                }
                newCol.setData(colData);
                result.addColumn(newCol);
            }
        }

        return result;
    }

    @Override
    public DataFrame fitTransform(DataFrame df, String... colNames) {
        fit(df, colNames);
        return transform(df, colNames);
    }

    @Override
    public IDataFrameTransform<Double> fit(DataFrame df, String... colNames) {
        this.columns = colNames;
        int nFeatures = colNames.length;

        dfCategories = new List[nFeatures];
        dfCategoryToIndex = new HashMap<>();

        for (int j = 0; j < nFeatures; j++) {
            Column col = df.getColumnByName(colNames[j]);
            if (col == null) {
                throw new IllegalArgumentException("列不存在: " + colNames[j]);
            }

            List<Object> uniqueValues = new ArrayList<>();
            for (Object v : col.getData()) {
                if (v != null && !uniqueValues.contains(v)) {
                    uniqueValues.add(v);
                }
            }

            // 数值类型排序，字符串保持顺序
            List<Object> sortedCategories = new ArrayList<>(uniqueValues);
            boolean allNumeric = true;
            for (Object v : sortedCategories) {
                if (!(v instanceof Number)) {
                    allNumeric = false;
                    break;
                }
            }
            if (allNumeric) {
                sortedCategories.sort((a, b) -> {
                    double da = ((Number) a).doubleValue();
                    double db = ((Number) b).doubleValue();
                    return Double.compare(da, db);
                });
            }

            dfCategories[j] = sortedCategories;

            Map<Object, Integer> catMap = new HashMap<>();
            for (int i = 0; i < sortedCategories.size(); i++) {
                catMap.put(sortedCategories.get(i), i);
            }
            dfCategoryToIndex.put(j, catMap);
        }

        // 同时更新 Matrix 版本用的 categories
        categories = new List[nFeatures];
        categoryToIndex = new HashMap<>();
        for (int j = 0; j < nFeatures; j++) {
            List<Double> numericCats = new ArrayList<>();
            Map<Double, Integer> numCatMap = new HashMap<>();
            int idx = 0;
            for (Object cat : dfCategories[j]) {
                if (cat instanceof Number num) {
                    numericCats.add(num.doubleValue());
                    numCatMap.put(num.doubleValue(), idx++);
                }
            }
            categories[j] = numericCats;
            categoryToIndex.put(j, numCatMap);
        }

        trained = true;
        return this;
    }

    @SuppressWarnings("unchecked")
    public IMatrix inverseTransform(IMatrix Y) {
        if (!trained) {
            throw new IllegalStateException("Encoder must be fitted before inverse transform");
        }

        int nSamples = Y.rows();
        int nFeatures = categories.length;
        boolean isFloat = Y instanceof IFloatMatrix;

        if (isFloat) {
            float[][] result = new float[nSamples][nFeatures];
            int columnOffset = 0;
            for (int j = 0; j < nFeatures; j++) {
                for (int i = 0; i < nSamples; i++) {
                    int activeIdx = -1;
                    for (int k = 0; k < categories[j].size(); k++) {
                        if (Y.get(i, columnOffset + k) > 0.5) {
                            activeIdx = k;
                            break;
                        }
                    }
                    if (activeIdx >= 0) {
                        result[i][j] = categories[j].get(activeIdx).floatValue();
                    }
                }
                columnOffset += categories[j].size();
            }
            return IMatrix.of(result);
        } else {
            double[][] result = new double[nSamples][nFeatures];
            int columnOffset = 0;
            for (int j = 0; j < nFeatures; j++) {
                for (int i = 0; i < nSamples; i++) {
                    int activeIdx = -1;
                    for (int k = 0; k < categories[j].size(); k++) {
                        if (Y.get(i, columnOffset + k) > 0.5) {
                            activeIdx = k;
                            break;
                        }
                    }
                    if (activeIdx >= 0) {
                        result[i][j] = categories[j].get(activeIdx);
                    }
                }
                columnOffset += categories[j].size();
            }
            return IMatrix.of(result);
        }
    }
}
