package com.yishape.lab.math.ml.metric;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.cls.IClassification;
import com.yishape.lab.math.ml.cls.ClassificationResult;

import java.util.*;

/**
 * 交叉验证工具类
 * <p>
 * 提供多种交叉验证方法，用于分类器模型的训练和评估。
 * 支持K折交叉验证、分层K折交叉验证、随机分割等多种验证方式。
 * 与ClassificationMetrics配合使用，提供全面的模型评估。
 * </p>
 * 
 * @author yishape
 * @version 1.0
 * @since 1.0
 */
public class CrossValidation {
    
    // ==================== 交叉验证类型枚举 ====================
    
    /**
     * 交叉验证类型
     */
    public enum CrossValidationType {
        /** K折交叉验证 */
        K_FOLD,
        /** 分层K折交叉验证 (保持各类别比例) */
        STRATIFIED_K_FOLD,
        /** 随机分割交叉验证 */
        RANDOM_SPLIT,
        /** 留一法交叉验证 */
        LEAVE_ONE_OUT,
        /** 多次K折交叉验证 */
        REPEATED_K_FOLD
    }
    
    // ==================== 交叉验证结果类 ====================
    
    /**
     * 交叉验证结果
     */
    public static class CrossValidationResult {
        /** 每次折叠的准确率 */
        private final List<Double> accuracyScores;
        
        /** 每次折叠的F1分数 */
        private final List<Double> f1Scores;
        
        /** 每次折叠的精确率 */
        private final List<Double> precisionScores;
        
        /** 每次折叠的召回率 */
        private final List<Double> recallScores;
        
        /** 每次折叠的AUC (如果是二分类) */
        private final List<Double> aucScores;
        
        /** 平均准确率 */
        private final double meanAccuracy;
        
        /** 准确率标准差 */
        private final double stdAccuracy;
        
        /** 平均F1分数 */
        private final double meanF1;
        
        /** F1分数标准差 */
        private final double stdF1;
        
        /** 平均精确率 */
        private final double meanPrecision;
        
        /** 平均召回率 */
        private final double meanRecall;
        
        /** 平均AUC */
        private final double meanAuc;
        
        /** 总验证次数 */
        private final int totalFolds;
        
        /** 使用的验证类型 */
        private final CrossValidationType validationType;
        
        /** 每折的训练时间 (毫秒) */
        private final List<Long> trainingTimes;
        
        /** 每折的预测时间 (毫秒) */
        private final List<Long> predictionTimes;
        
        /**
         * 构造函数
         */
        public CrossValidationResult(List<Double> accuracyScores, List<Double> f1Scores,
                                   List<Double> precisionScores, List<Double> recallScores,
                                   List<Double> aucScores, List<Long> trainingTimes,
                                   List<Long> predictionTimes, CrossValidationType validationType) {
            this.accuracyScores = new ArrayList<>(accuracyScores);
            this.f1Scores = new ArrayList<>(f1Scores);
            this.precisionScores = new ArrayList<>(precisionScores);
            this.recallScores = new ArrayList<>(recallScores);
            this.aucScores = new ArrayList<>(aucScores);
            this.trainingTimes = new ArrayList<>(trainingTimes);
            this.predictionTimes = new ArrayList<>(predictionTimes);
            this.validationType = validationType;
            this.totalFolds = accuracyScores.size();
            
            // 计算统计指标
            this.meanAccuracy = calculateMean(accuracyScores);
            this.stdAccuracy = calculateStd(accuracyScores, meanAccuracy);
            this.meanF1 = calculateMean(f1Scores);
            this.stdF1 = calculateStd(f1Scores, meanF1);
            this.meanPrecision = calculateMean(precisionScores);
            this.meanRecall = calculateMean(recallScores);
            this.meanAuc = aucScores.isEmpty() ? -1.0 : calculateMean(aucScores);
        }
        
        /**
         * 计算平均值
         */
        private double calculateMean(List<Double> values) {
            if (values.isEmpty()) return 0.0;
            return values.stream().mapToDouble(Double::doubleValue).sum() / values.size();
        }
        
        /**
         * 计算标准差
         */
        private double calculateStd(List<Double> values, double mean) {
            if (values.size() <= 1) return 0.0;
            double variance = values.stream()
                .mapToDouble(v -> (v - mean) * (v - mean))
                .sum() / (values.size() - 1);
            return Math.sqrt(variance);
        }
        
        // ==================== Getter方法 ====================
        
        public List<Double> getAccuracyScores() { return new ArrayList<>(accuracyScores); }
        public List<Double> getF1Scores() { return new ArrayList<>(f1Scores); }
        public List<Double> getPrecisionScores() { return new ArrayList<>(precisionScores); }
        public List<Double> getRecallScores() { return new ArrayList<>(recallScores); }
        public List<Double> getAucScores() { return new ArrayList<>(aucScores); }
        public List<Long> getTrainingTimes() { return new ArrayList<>(trainingTimes); }
        public List<Long> getPredictionTimes() { return new ArrayList<>(predictionTimes); }
        
        public double getMeanAccuracy() { return meanAccuracy; }
        public double getStdAccuracy() { return stdAccuracy; }
        public double getMeanF1() { return meanF1; }
        public double getStdF1() { return stdF1; }
        public double getMeanPrecision() { return meanPrecision; }
        public double getMeanRecall() { return meanRecall; }
        public double getMeanAuc() { return meanAuc; }
        public int getTotalFolds() { return totalFolds; }
        public CrossValidationType getValidationType() { return validationType; }
        
        /**
         * 获取95%置信区间
         */
        public double[] getAccuracy95Percentile() {
            return getPercentile(accuracyScores, 2.5, 97.5);
        }
        
        /**
         * 获取百分位数
         */
        private double[] getPercentile(List<Double> values, double... percentiles) {
            if (values.isEmpty()) return new double[percentiles.length];
            
            List<Double> sortedValues = new ArrayList<>(values);
            Collections.sort(sortedValues);
            
            double[] result = new double[percentiles.length];
            for (int i = 0; i < percentiles.length; i++) {
                double p = percentiles[i];
                double index = p / 100.0 * (sortedValues.size() - 1);
                int lower = (int) Math.floor(index);
                int upper = (int) Math.ceil(index);
                
                if (lower == upper) {
                    result[i] = sortedValues.get(lower);
                } else {
                    double weight = index - lower;
                    result[i] = sortedValues.get(lower) * (1 - weight) + sortedValues.get(upper) * weight;
                }
            }
            return result;
        }
        
        /**
         * 获取详细报告
         */
        public String getDetailedReport() {
            StringBuilder sb = new StringBuilder();
            sb.append("=================== 交叉验证结果报告 ===================\n");
            sb.append(String.format("验证类型: %s\n", validationType));
            sb.append(String.format("验证次数: %d\n", totalFolds));
            
            sb.append("\n=== 准确率统计 ===\n");
            sb.append(String.format("平均值: %.4f ± %.4f\n", meanAccuracy, stdAccuracy));
            sb.append(String.format("最小值: %.4f\n", Collections.min(accuracyScores)));
            sb.append(String.format("最大值: %.4f\n", Collections.max(accuracyScores)));
            sb.append(String.format("95%%置信区间: [%.4f, %.4f]\n", 
                    getAccuracy95Percentile()[0], getAccuracy95Percentile()[1]));
            
            sb.append("\n=== F1分数统计 ===\n");
            sb.append(String.format("平均值: %.4f ± %.4f\n", meanF1, stdF1));
            sb.append(String.format("最小值: %.4f\n", Collections.min(f1Scores)));
            sb.append(String.format("最大值: %.4f\n", Collections.max(f1Scores)));
            
            sb.append("\n=== 精确率和召回率 ===\n");
            sb.append(String.format("平均精确率: %.4f\n", meanPrecision));
            sb.append(String.format("平均召回率: %.4f\n", meanRecall));
            
            if (meanAuc >= 0) {
                sb.append("\n=== AUC统计 ===\n");
                sb.append(String.format("平均AUC: %.4f\n", meanAuc));
                sb.append(String.format("最小AUC: %.4f\n", Collections.min(aucScores)));
                sb.append(String.format("最大AUC: %.4f\n", Collections.max(aucScores)));
            }
            
            sb.append("\n=== 时间性能 ===\n");
            long totalTrainingTime = trainingTimes.stream().mapToLong(Long::longValue).sum();
            long totalPredictionTime = predictionTimes.stream().mapToLong(Long::longValue).sum();
            sb.append(String.format("总训练时间: %d ms\n", totalTrainingTime));
            sb.append(String.format("总预测时间: %d ms\n", totalPredictionTime));
            sb.append(String.format("平均每折训练时间: %.2f ms\n", 
                    (double) totalTrainingTime / totalFolds));
            sb.append(String.format("平均每折预测时间: %.2f ms\n", 
                    (double) totalPredictionTime / totalFolds));
            
            sb.append("\n=== 每折详细结果 ===\n");
            sb.append(String.format("%-6s %-10s %-10s %-10s %-10s %-10s %-10s\n",
                    "折数", "准确率", "F1分数", "精确率", "召回率", "AUC", "训练时间(ms)"));
            sb.append(String.format("%-6s %-10s %-10s %-10s %-10s %-10s %-10s\n",
                    "----", "------", "-------", "-------", "-------", "---", "----------"));
            
            for (int i = 0; i < totalFolds; i++) {
                sb.append(String.format("%-6d %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10d\n",
                        i + 1, accuracyScores.get(i), f1Scores.get(i),
                        precisionScores.get(i), recallScores.get(i),
                        aucScores.isEmpty() ? -1.0 : aucScores.get(i),
                        trainingTimes.get(i)));
            }
            
            return sb.toString();
        }
        
        @Override
        public String toString() {
            return String.format("CrossValidationResult{meanAccuracy=%.4f±%.4f, meanF1=%.4f±%.4f, folds=%d}",
                    meanAccuracy, stdAccuracy, meanF1, stdF1, totalFolds);
        }
    }
    
    // ==================== 静态交叉验证方法 ====================
    
    /**
     * K折交叉验证
     * K-Fold Cross Validation
     * 
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param k 折数
     * @return 交叉验证结果
     */
    public static CrossValidationResult kFoldCrossValidation(IClassification classifier, 
                                                            IMatrix<Double> X, String[] y, int k) {
        return crossValidate(classifier, X, y, CrossValidationType.K_FOLD, 
                createKFoldSplits(y.length, k));
    }
    
    /**
     * 分层K折交叉验证
     * Stratified K-Fold Cross Validation
     * 
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param k 折数
     * @return 交叉验证结果
     */
    public static CrossValidationResult stratifiedKFoldCrossValidation(IClassification classifier,
                                                                      IMatrix<Double> X, String[] y, int k) {
        return crossValidate(classifier, X, y, CrossValidationType.STRATIFIED_K_FOLD,
                createStratifiedKFoldSplits(y, k));
    }
    
    /**
     * 随机分割交叉验证
     * Random Split Cross Validation
     * 
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param testSize 测试集比例 (0.0-1.0)
     * @param nSplits 分割次数
     * @param randomSeed 随机种子
     * @return 交叉验证结果
     */
    public static CrossValidationResult randomSplitCrossValidation(IClassification classifier,
                                                                  IMatrix<Double> X, String[] y,
                                                                  double testSize, int nSplits,
                                                                  int randomSeed) {
        return crossValidate(classifier, X, y, CrossValidationType.RANDOM_SPLIT,
                createRandomSplits(y.length, testSize, nSplits, randomSeed));
    }
    
    /**
     * 留一法交叉验证
     * Leave-One-Out Cross Validation
     * 
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @return 交叉验证结果
     */
    public static CrossValidationResult leaveOneOutCrossValidation(IClassification classifier,
                                                                 IMatrix<Double> X, String[] y) {
        return crossValidate(classifier, X, y, CrossValidationType.LEAVE_ONE_OUT,
                createLeaveOneOutSplits(y.length));
    }
    
    /**
     * 多次K折交叉验证
     * Repeated K-Fold Cross Validation
     * 
     * @param classifier 分类器
     * @param X 特征矩阵
     * @param y 标签数组
     * @param k 折数
     * @param nRepeats 重复次数
     * @param randomSeed 随机种子
     * @return 交叉验证结果
     */
    public static CrossValidationResult repeatedKFoldCrossValidation(IClassification classifier,
                                                                    IMatrix<Double> X, String[] y,
                                                                    int k, int nRepeats,
                                                                    int randomSeed) {
        return crossValidate(classifier, X, y, CrossValidationType.REPEATED_K_FOLD,
                createRepeatedKFoldSplits(y.length, k, nRepeats, randomSeed));
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
    private static CrossValidationResult crossValidate(IClassification classifier, IMatrix<Double> X, String[] y,
                                                      CrossValidationType validationType,
                                                      List<IndexPair>[] splits) {
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
        
        for (List<IndexPair> foldSplits : splits) {
            CrossValidationResult foldResult = singleFoldValidation(classifier, X, y, foldSplits);
            
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
    private static CrossValidationResult singleFoldValidation(IClassification classifier, IMatrix<Double> X,
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
            double[] probabilities = new double[testX.getRowNum()];
            
            for (int i = 0; i < testX.getRowNum(); i++) {
                IVector<Double> row = testX.getRow(i);
                predictions[i] = classifier.predict(row);
                // 如果分类器有概率预测能力，这里需要根据具体分类器实现
                // 简化示例：假设正类概率为0.5
                probabilities[i] = 0.5;
            }
            
            long predEndTime = System.currentTimeMillis();
            
            // 计算评估指标
            ClassificationMetrics metrics = ClassificationMetrics.compute(testY, predictions, probabilities);
            
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
        List<IndexPair>[] splits = new ArrayList[k];
        
        // 创建索引数组
        int[] indices = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            indices[i] = i;
        }
        
        // 简单随机打乱 (这里可以改进为更复杂的随机算法)
        shuffleArray(indices);
        
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
                int classSize = indices.size();
                int foldSize = classSize / k;
                int remainder = classSize % k;
                
                int startIdx = fold * foldSize + Math.min(fold, remainder);
                int endIdx = (fold + 1) * foldSize + Math.min(fold + 1, remainder);
                
                // 当前折的测试集
                for (int i = startIdx; i < endIdx && i < classSize; i++) {
                    testIndices.add(indices.get(i));
                }
                
                // 其余作为训练集
                for (int i = 0; i < classSize; i++) {
                    if (i < startIdx || i >= endIdx) {
                        trainIndices.add(indices.get(i));
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
        List<IndexPair> allSplits = new ArrayList<>();
        
        for (int repeat = 0; repeat < nRepeats; repeat++) {
            List<IndexPair>[] kFoldSplits = createKFoldSplits(nSamples, k);
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