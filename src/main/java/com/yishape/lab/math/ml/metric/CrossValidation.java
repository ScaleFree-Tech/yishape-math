package com.yishape.lab.math.ml.metric;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.cls.ClassificationResult;
import java.io.Serializable;

import java.util.*;
import com.yishape.lab.math.ml.cls.IClassifier;

/**
 * 交叉验证工具类
 * <p>
 * 提供多种交叉验证方法，用于分类器模型的训练和评估。 支持K折交叉验证、分层K折交叉验证、随机分割等多种验证方式。
 * 与ClassificationMetrics配合使用，提供全面的模型评估。
 * </p>
 *
 * @author yishape
 * @version 1.0
 * @since 1.0
 */
public class CrossValidation implements Serializable {

    // ==================== 交叉验证类型枚举 ====================
    /**
     * 交叉验证类型
     */
    public enum CrossValidationType {
        /**
         * K折交叉验证
         */
        K_FOLD,
        /**
         * 分层K折交叉验证 (保持各类别比例)
         */
        STRATIFIED_K_FOLD,
        /**
         * 随机分割交叉验证
         */
        RANDOM_SPLIT,
        /**
         * 留一法交叉验证
         */
        LEAVE_ONE_OUT,
        /**
         * 多次K折交叉验证
         */
        REPEATED_K_FOLD
    }

    // ==================== 静态交叉验证方法 ====================
    
    
    /**
     * K折交叉验证 K-Fold Cross Validation
     *
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param k 折数
     * @return 交叉验证结果
     */
    public static CrossValidationResult kFoldCrossValidation(IClassifier classifier,
            IMatrix<Double> X, String[] y, int k) {
        return crossValidate(classifier, X, y, CrossValidationType.K_FOLD,
                createKFoldSplits(y.length, k), null);
    }

    /**
     * K折交叉验证 K-Fold Cross Validation
     *
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param k 折数
     * @param logger
     * @return 交叉验证结果
     */
    public static CrossValidationResult kFoldCrossValidation(IClassifier classifier,
            IMatrix<Double> X, String[] y, int k, CrossValidationLogger logger) {
        return crossValidate(classifier, X, y, CrossValidationType.K_FOLD,
                createKFoldSplits(y.length, k), logger);
    }

    /**
     * 分层K折交叉验证 Stratified K-Fold Cross Validation
     *
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param k 折数
     * @return 交叉验证结果
     */
    public static CrossValidationResult stratifiedKFoldCrossValidation(IClassifier classifier,
            IMatrix<Double> X, String[] y, int k) {
        return crossValidate(classifier, X, y, CrossValidationType.STRATIFIED_K_FOLD,
                createStratifiedKFoldSplits(y, k),null);
    }

    /**
     * 随机分割交叉验证 Random Split Cross Validation
     *
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param testSize 测试集比例 (0.0-1.0)
     * @param nSplits 分割次数
     * @param randomSeed 随机种子
     * @return 交叉验证结果
     */
    public static CrossValidationResult randomSplitCrossValidation(IClassifier classifier,
            IMatrix<Double> X, String[] y,
            double testSize, int nSplits,
            int randomSeed) {
        return crossValidate(classifier, X, y, CrossValidationType.RANDOM_SPLIT,
                createRandomSplits(y.length, testSize, nSplits, randomSeed),null);
    }

    /**
     * 留一法交叉验证 Leave-One-Out Cross Validation
     *
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @return 交叉验证结果
     */
    public static CrossValidationResult leaveOneOutCrossValidation(IClassifier classifier,
            IMatrix<Double> X, String[] y) {
        return crossValidate(classifier, X, y, CrossValidationType.LEAVE_ONE_OUT,
                createLeaveOneOutSplits(y.length),null);
    }

    /**
     * 多次K折交叉验证 Repeated K-Fold Cross Validation
     *
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param k 折数
     * @param nRepeats 重复次数
     * @param randomSeed 随机种子
     * @return 交叉验证结果
     */
    public static CrossValidationResult repeatedKFoldCrossValidation(IClassifier classifier,
            IMatrix<Double> X, String[] y,
            int k, int nRepeats,
            int randomSeed) {
        return crossValidate(classifier, X, y, CrossValidationType.REPEATED_K_FOLD,
                createRepeatedKFoldSplits(y.length, k, nRepeats, randomSeed),null);
    }

    // ==================== 通用交叉验证方法 ====================
    /**
     * 通用交叉验证方法
     *
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param validationType 验证类型
     * @param splits 分割列表
     * @return 交叉验证结果
     */
    private static CrossValidationResult crossValidate(IClassifier classifier, IMatrix<Double> X, String[] y,
            CrossValidationType validationType,
            List<IndexPair>[] splits, CrossValidationLogger logger) {
        if (classifier == null || X == null || y == null) {
            throw new IllegalArgumentException("分类器、特征矩阵和标签不能为空");
        }

        if (y.length != X.getRowNum()) {
            throw new IllegalArgumentException("标签数量与特征矩阵行数不一致");
        }

        List<Double> accuracyScores = new ArrayList<>();
        List<Double> f1Scores = new ArrayList<>();
        List<Double> precisionScores = new ArrayList<>();
        List<Double> recallScores = new ArrayList<>();
        List<Double> aucScores = new ArrayList<>();
        List<Long> trainingTimes = new ArrayList<>();
        List<Long> predictionTimes = new ArrayList<>();
        int k=0;
        for (List<IndexPair> foldSplits : splits) {
            k++;
            CrossValidationResult foldResult = singleFoldValidation(classifier, X, y, foldSplits);

            //用于检验过程中的日志输出
            if (logger != null) {
                logger.log(k,foldResult);
            }
            
            
            accuracyScores.addAll(foldResult.getAccuracyScores());
            f1Scores.addAll(foldResult.getF1Scores());
            precisionScores.addAll(foldResult.getPrecisionScores());
            recallScores.addAll(foldResult.getRecallScores());
            aucScores.addAll(foldResult.getAucScores());
            trainingTimes.addAll(foldResult.getTrainingTimes());
            predictionTimes.addAll(foldResult.getPredictionTimes());
        }

        return new CrossValidationResult(accuracyScores, f1Scores, precisionScores, recallScores,
                aucScores, trainingTimes, predictionTimes, validationType);
    }


    /**
     * 单折验证
     */
    private static CrossValidationResult singleFoldValidation(IClassifier classifier, IMatrix<Double> X,
            String[] y, List<IndexPair> splits) {
        List<Double> accuracyScores = new ArrayList<>();
        List<Double> f1Scores = new ArrayList<>();
        List<Double> precisionScores = new ArrayList<>();
        List<Double> recallScores = new ArrayList<>();
        List<Double> aucScores = new ArrayList<>();
        List<Long> trainingTimes = new ArrayList<>();
        List<Long> predictionTimes = new ArrayList<>();
        
        for (IndexPair split : splits) {
            long trainStartTime = System.currentTimeMillis();

            // 提取训练集
            IMatrix<Double> trainX = extractRows(X, split.trainIndices);
            String[] trainY = extractLabels(y, split.trainIndices);

            // 训练模型
            ClassificationResult result = classifier.fit(trainX, trainY);
            
            long trainEndTime = System.currentTimeMillis();

            // 提取测试集
            IMatrix<Double> testX = extractRows(X, split.testIndices);
            String[] testY = extractLabels(y, split.testIndices);

            long predStartTime = System.currentTimeMillis();

            // 进行预测
            String[] predictions = new String[testX.getRowNum()];

            for (int i = 0; i < testX.getRowNum(); i++) {
                IVector<Double> row = testX.getRow(i);
                predictions[i] = classifier.predict(row);
            }

            long predEndTime = System.currentTimeMillis();

            // 计算评估指标 (不使用概率，避免硬编码)
            ClassificationMetrics metrics = ClassificationMetrics.compute(testY, predictions);

            accuracyScores.add(metrics.getAccuracy());
            f1Scores.add(metrics.getWeightedF1());
            precisionScores.add(metrics.getWeightedPrecision());
            recallScores.add(metrics.getWeightedRecall());
            aucScores.add(metrics.getAuc());
            trainingTimes.add(trainEndTime - trainStartTime);
            predictionTimes.add(predEndTime - predStartTime);
        }

        return new CrossValidationResult(accuracyScores, f1Scores, precisionScores, recallScores,
                aucScores, trainingTimes, predictionTimes, CrossValidationType.K_FOLD);
    }

    // ==================== 数据分割方法 ====================
    /**
     * 索引对
     */
    private static class IndexPair {

        final int[] trainIndices;
        final int[] testIndices;

        IndexPair(int[] trainIndices, int[] testIndices) {
            this.trainIndices = trainIndices;
            this.testIndices = testIndices;
        }
    }

    /**
     * 创建K折分割
     */
    private static List<IndexPair>[] createKFoldSplits(int nSamples, int k) {
        return createKFoldSplits(nSamples, k, new Random(42)); // 使用固定种子确保可重现
    }

    /**
     * 创建K折分割 (带随机种子)
     */
    private static List<IndexPair>[] createKFoldSplits(int nSamples, int k, Random random) {
        List<IndexPair>[] splits = new ArrayList[k];

        // 创建索引数组
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            indices[i] = i;
        }

        // 随机打乱
        shuffleArray(indices, random);

        int foldSize = nSamples / k;
        int remainder = nSamples % k;

        for (int i = 0; i < k; i++) {
            List<IndexPair> splitList = new ArrayList<>();

            int startIdx = i * foldSize + Math.min(i, remainder);
            int endIdx = (i + 1) * foldSize + Math.min(i + 1, remainder);

            // 当前折的测试集
            int[] testIndices = new int[endIdx - startIdx];
            System.arraycopy(indices, startIdx, testIndices, 0, testIndices.length);

            // 其余作为训练集
            int[] trainIndices = new int[nSamples - testIndices.length];
            int trainIdx = 0;
            for (int j = 0; j < nSamples; j++) {
                if (j < startIdx || j >= endIdx) {
                    trainIndices[trainIdx++] = indices[j];
                }
            }

            splitList.add(new IndexPair(trainIndices, testIndices));
            splits[i] = splitList;
        }

        return splits;
    }

    /**
     * 创建分层K折分割
     */
    private static List<IndexPair>[] createStratifiedKFoldSplits(String[] y, int k) {
        return createStratifiedKFoldSplits(y, k, new Random(42)); // 使用固定种子确保可重现
    }

    /**
     * 创建分层K折分割 (带随机种子)
     */
    private static List<IndexPair>[] createStratifiedKFoldSplits(String[] y, int k, Random random) {
        // 按类别分组
        Map<String, List<Integer>> classIndices = new HashMap<>();
        for (int i = 0; i < y.length; i++) {
            classIndices.computeIfAbsent(y[i], key -> new ArrayList<>()).add(i);
        }

        List<IndexPair>[] splits = new ArrayList[k];

        for (int fold = 0; fold < k; fold++) {
            List<IndexPair> splitList = new ArrayList<>();
            List<Integer> trainIndices = new ArrayList<>();
            List<Integer> testIndices = new ArrayList<>();

            // 对每个类别进行分层分割
            for (List<Integer> indices : classIndices.values()) {
                // 随机打乱该类别的索引
                List<Integer> shuffledIndices = new ArrayList<>(indices);
                Collections.shuffle(shuffledIndices, random);

                int classSize = shuffledIndices.size();
                int foldSize = classSize / k;
                int remainder = classSize % k;

                int startIdx = fold * foldSize + Math.min(fold, remainder);
                int endIdx = (fold + 1) * foldSize + Math.min(fold + 1, remainder);

                // 当前折的测试集
                for (int i = startIdx; i < endIdx && i < classSize; i++) {
                    testIndices.add(shuffledIndices.get(i));
                }

                // 其余作为训练集
                for (int i = 0; i < classSize; i++) {
                    if (i < startIdx || i >= endIdx) {
                        trainIndices.add(shuffledIndices.get(i));
                    }
                }
            }

            splitList.add(new IndexPair(
                    trainIndices.stream().mapToInt(Integer::intValue).toArray(),
                    testIndices.stream().mapToInt(Integer::intValue).toArray()
            ));
            splits[fold] = splitList;
        }

        return splits;
    }

    /**
     * 创建随机分割
     */
    private static List<IndexPair>[] createRandomSplits(int nSamples, double testSize,
            int nSplits, int randomSeed) {
        Random random = new Random(randomSeed);
        List<IndexPair>[] splits = new ArrayList[nSplits];

        for (int i = 0; i < nSplits; i++) {
            List<IndexPair> splitList = new ArrayList<>();

            // 生成随机分割
            int[] indices = new int[nSamples];
            for (int j = 0; j < nSamples; j++) {
                indices[j] = j;
            }

            // 打乱数组
            shuffleArray(indices, random);

            int testCount = (int) (nSamples * testSize);

            int[] testIndices = new int[testCount];
            int[] trainIndices = new int[nSamples - testCount];

            System.arraycopy(indices, 0, testIndices, 0, testCount);
            System.arraycopy(indices, testCount, trainIndices, 0, nSamples - testCount);

            splitList.add(new IndexPair(trainIndices, testIndices));
            splits[i] = splitList;
        }

        return splits;
    }

    /**
     * 创建留一法分割
     */
    private static List<IndexPair>[] createLeaveOneOutSplits(int nSamples) {
        List<IndexPair>[] splits = new ArrayList[nSamples];

        for (int i = 0; i < nSamples; i++) {
            List<IndexPair> splitList = new ArrayList<>();

            int[] trainIndices = new int[nSamples - 1];
            int[] testIndices = new int[]{i};

            int trainIdx = 0;
            for (int j = 0; j < nSamples; j++) {
                if (j != i) {
                    trainIndices[trainIdx++] = j;
                }
            }

            splitList.add(new IndexPair(trainIndices, testIndices));
            splits[i] = splitList;
        }

        return splits;
    }

    /**
     * 创建重复K折分割
     */
    private static List<IndexPair>[] createRepeatedKFoldSplits(int nSamples, int k,
            int nRepeats, int randomSeed) {
        Random random = new Random(randomSeed);
        List<IndexPair> allSplits = new ArrayList<>();

        for (int repeat = 0; repeat < nRepeats; repeat++) {
            List<IndexPair>[] kFoldSplits = createKFoldSplits(nSamples, k, random);
            for (List<IndexPair> split : kFoldSplits) {
                allSplits.addAll(split);
            }
        }

        @SuppressWarnings("unchecked")
        List<IndexPair>[] result = new ArrayList[allSplits.size()];
        return allSplits.toArray(result);
    }

    // ==================== 工具方法 ====================
    /**
     * 数组打乱
     */
    private static void shuffleArray(int[] array) {
        Random random = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    /**
     * 数组打乱 (带随机种子)
     */
    private static void shuffleArray(int[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    /**
     * 提取矩阵行
     */
    private static IMatrix<Double> extractRows(IMatrix<Double> X, int[] indices) {
        int nRows = indices.length;
        int nCols = X.getColNum();

        double[][] data = new double[nRows][nCols];
        for (int i = 0; i < nRows; i++) {
            int rowIdx = indices[i];
            for (int j = 0; j < nCols; j++) {
                data[i][j] = X.get(rowIdx, j);
            }
        }

        return IMatrix.of(data);
    }

    /**
     * 提取标签
     */
    private static String[] extractLabels(String[] y, int[] indices) {
        String[] subset = new String[indices.length];
        for (int i = 0; i < indices.length; i++) {
            subset[i] = y[indices[i]];
        }
        return subset;
    }
}
