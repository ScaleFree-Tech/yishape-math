package com.yishape.lab.math.ml.clustering;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.io.*;
import java.util.*;

/**
 * K-means++聚类算法实现
 * 提供改进的初始化策略和数值稳定性保证
 * 
 * @author reremouse
 */
public class KMeansPlusPlus implements IClustering, ISerializableModel {

    private static final Logger log = LoggerFactory.getLogger(KMeansPlusPlus.class);

    
    private static final long serialVersionUID = 1L;
    
    // 算法参数
    private static final int MAX_ITERATIONS = 100;
    private static final double CONVERGENCE_THRESHOLD = 1e-6;
    private static final double MIN_DISTANCE_THRESHOLD = 1e-8;
    private static final int MAX_INIT_ATTEMPTS = 10;
    
    // 数值稳定性参数
    private static final double NUMERICAL_STABILITY_EPS = 1e-12;
    private static final double MIN_CLUSTER_SIZE_RATIO = 0.01; // 最小聚类大小比例
    
    private final Random random;
    private final int maxIterations;
    private final double convergenceThreshold;
    
    // 算法状态
    private int numClusters = 3;
    private List<IVector<Double>> clusterCenters;
    private int[] labels;
    private double inertia;
    private boolean converged;
    private int iterations;
    private int dimension;
    private Map<String, Object> parameters = new HashMap<>();
    
    /**
     * 构造函数
     */
    public KMeansPlusPlus() {
        this(new Random(), MAX_ITERATIONS, CONVERGENCE_THRESHOLD);
    }
    
    /**
     * 构造函数
     * @param seed 随机种子
     */
    public KMeansPlusPlus(long seed) {
        this(new Random(seed), MAX_ITERATIONS, CONVERGENCE_THRESHOLD);
    }
    
    /**
     * 构造函数
     * @param random 随机数生成器
     * @param maxIterations 最大迭代次数
     * @param convergenceThreshold 收敛阈值
     */
    public KMeansPlusPlus(Random random, int maxIterations, double convergenceThreshold) {
        this.random = random;
        this.maxIterations = maxIterations;
        this.convergenceThreshold = convergenceThreshold;
    }
    
    /**
     * K-means++初始化聚类中心（适配Vector列表接口）
     * @param data 数据向量列表
     * @param k 聚类数量
     * @return 初始化的聚类中心列表
     */
    public List<IVector<Double>> initializeCenters(List<IVector<Double>> data, int k) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("数据列表不能为空");
        }
        
        // 转换为矩阵格式
        IMatrix<Double> dataMatrix = convertToMatrix(data);
        IMatrix<Double> centerMatrix = initializeCenters(dataMatrix, k);
        
        // 转换回向量列表
        return convertToVectorList(centerMatrix);
    }
    
    /**
     * K-means++初始化聚类中心
     * @param data 数据矩阵 (n x d)
     * @param k 聚类数量
     * @return 初始化的聚类中心 (k x d)
     */
    public IMatrix<Double> initializeCenters(IMatrix<Double> data, int k) {
        if (data == null || data.getRowNum() == 0 || data.getColNum() == 0) {
            throw new IllegalArgumentException("数据矩阵不能为空");
        }
        if (k <= 0 || k > data.getRowNum()) {
            throw new IllegalArgumentException("聚类数量必须在1到数据点数量之间");
        }
        
        int n = data.getRowNum();
        int d = data.getColNum();
        
        // 尝试多次初始化，选择最好的结果
        IMatrix<Double> bestCenters = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        
        for (int attempt = 0; attempt < MAX_INIT_ATTEMPTS; attempt++) {
            try {
                IMatrix<Double> centers = performKMeansPlusPlusInit(data, k);
                double score = evaluateInitialization(data, centers);
                
                if (score > bestScore) {
                    bestScore = score;
                    bestCenters = centers;
                }
            } catch (Exception e) {
                // 如果某次初始化失败，继续尝试
                continue;
            }
        }
        
        if (bestCenters == null) {
            // 如果所有尝试都失败，使用随机初始化作为后备
            return randomInitialization(data, k);
        }
        
        return bestCenters;
    }
    
    /**
     * 执行K-means++初始化
     */
    private IMatrix<Double> performKMeansPlusPlusInit(IMatrix<Double> data, int k) {
        int n = data.getRowNum();
        int d = data.getColNum();
        
        IMatrix<Double> centers = Linalg.zeros(k, d);
        boolean[] chosen = new boolean[n];
        
        // 1. 随机选择第一个中心
        int firstCenter = random.nextInt(n);
        for (int j = 0; j < d; j++) {
            centers.set(0, j, data.get(firstCenter, j));
        }
        chosen[firstCenter] = true;
        
        // 2. 依次选择剩余的中心
        for (int centerIdx = 1; centerIdx < k; centerIdx++) {
            double[] distances = new double[n];
            double totalDistance = 0.0;
            
            // 计算每个点到最近中心的距离平方
            for (int i = 0; i < n; i++) {
                if (chosen[i]) {
                    distances[i] = 0.0;
                    continue;
                }
                
                double minDistSq = Double.POSITIVE_INFINITY;
                for (int c = 0; c < centerIdx; c++) {
                    double distSq = computeSquaredDistance(data, i, centers, c);
                    minDistSq = Math.min(minDistSq, distSq);
                }
                
                // 添加数值稳定性保护
                distances[i] = Math.max(minDistSq, NUMERICAL_STABILITY_EPS);
                totalDistance += distances[i];
            }
            
            // 3. 按概率选择下一个中心
            if (totalDistance < NUMERICAL_STABILITY_EPS) {
                // 如果总距离太小，随机选择
                int nextCenter = selectRandomUnchosen(chosen);
                if (nextCenter == -1) break;
                
                for (int j = 0; j < d; j++) {
                    centers.set(centerIdx, j, data.get(nextCenter, j));
                }
                chosen[nextCenter] = true;
            } else {
                double target = random.nextDouble() * totalDistance;
                double cumulative = 0.0;
                int selectedPoint = -1;
                
                for (int i = 0; i < n; i++) {
                    if (chosen[i]) continue;
                    
                    cumulative += distances[i];
                    if (cumulative >= target) {
                        selectedPoint = i;
                        break;
                    }
                }
                
                // 安全检查
                if (selectedPoint == -1) {
                    selectedPoint = selectRandomUnchosen(chosen);
                }
                
                if (selectedPoint != -1) {
                    for (int j = 0; j < d; j++) {
                        centers.set(centerIdx, j, data.get(selectedPoint, j));
                    }
                    chosen[selectedPoint] = true;
                }
            }
        }
        
        return centers;
    }
    
    /**
     * 随机初始化聚类中心
     */
    private IMatrix<Double> randomInitialization(IMatrix<Double> data, int k) {
        int n = data.getRowNum();
        int d = data.getColNum();
        
        IMatrix<Double> centers = Linalg.zeros(k, d);
        Set<Integer> selected = new HashSet<>();
        
        for (int i = 0; i < k; i++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (selected.contains(idx));
            selected.add(idx);
            
            for (int j = 0; j < d; j++) {
                centers.set(i, j, data.get(idx, j));
            }
        }
        
        return centers;
    }
    
    /**
     * 评估初始化质量
     */
    private double evaluateInitialization(IMatrix<Double> data, IMatrix<Double> centers) {
        int n = data.getRowNum();
        int k = centers.getRowNum();
        
        double totalDistance = 0.0;
        for (int i = 0; i < n; i++) {
            double minDist = Double.POSITIVE_INFINITY;
            for (int c = 0; c < k; c++) {
                double dist = computeSquaredDistance(data, i, centers, c);
                minDist = Math.min(minDist, dist);
            }
            totalDistance += minDist;
        }
        
        return -totalDistance; // 负值因为我们要最大化距离
    }
    
    /**
     * 计算数据点到聚类中心的平方距离
     */
    private double computeSquaredDistance(IMatrix<Double> data, int dataIdx, IMatrix<Double> centers, int centerIdx) {
        double sum = 0.0;
        int d = data.getColNum();
        for (int j = 0; j < d; j++) {
            double diff = data.get(dataIdx, j) - centers.get(centerIdx, j);
            sum += diff * diff;
        }
        return sum;
    }
    
    /**
     * 选择一个未被选中的随机点
     */
    private int selectRandomUnchosen(boolean[] chosen) {
        List<Integer> unchosen = new ArrayList<>();
        for (int i = 0; i < chosen.length; i++) {
            if (!chosen[i]) {
                unchosen.add(i);
            }
        }
        
        if (unchosen.isEmpty()) {
            return -1;
        }
        
        return unchosen.get(random.nextInt(unchosen.size()));
    }
    
    /**
     * 执行K-means聚类算法
     */
    private KMeansResult performKMeansClustering(IMatrix<Double> data, IMatrix<Double> initialCenters) {
        int n = data.getRowNum();
        int d = data.getColNum();
        int k = initialCenters.getRowNum();
        
        // 复制初始中心
        IMatrix<Double> centers = initialCenters.copy();
        
        int[] assignments = new int[n];
        boolean converged = false;
        int iterations = 0;
        double previousInertia = Double.POSITIVE_INFINITY;
        
        while (!converged && iterations < maxIterations) {
            // 分配步骤：为每个数据点分配最近的聚类中心
            for (int i = 0; i < n; i++) {
                double minDistance = Double.POSITIVE_INFINITY;
                int bestCluster = 0;
                
                for (int c = 0; c < k; c++) {
                    double distance = computeSquaredDistance(data, i, centers, c);
                    if (distance < minDistance) {
                        minDistance = distance;
                        bestCluster = c;
                    }
                }
                
                assignments[i] = bestCluster;
            }
            
            // 更新步骤：重新计算聚类中心
            IMatrix<Double> newCenters = Linalg.zeros(k, d);
            int[] clusterSizes = new int[k];
            
            for (int i = 0; i < n; i++) {
                int cluster = assignments[i];
                clusterSizes[cluster]++;
                for (int j = 0; j < d; j++) {
                    newCenters.set(cluster, j, newCenters.get(cluster, j) + data.get(i, j));
                }
            }
            
            // 计算新的聚类中心
            for (int c = 0; c < k; c++) {
                if (clusterSizes[c] > 0) {
                    for (int j = 0; j < d; j++) {
                        newCenters.set(c, j, newCenters.get(c, j) / clusterSizes[c]);
                    }
                }
            }
            
            // 检查收敛性
            double inertia = 0.0;
            for (int i = 0; i < n; i++) {
                int cluster = assignments[i];
                inertia += computeSquaredDistance(data, i, newCenters, cluster);
            }
            
            double inertiaChange = Math.abs(previousInertia - inertia);
            converged = inertiaChange < convergenceThreshold;
            previousInertia = inertia;
            
            centers = newCenters;
            iterations++;
        }
        
        return new KMeansResult(centers, assignments, previousInertia, converged, iterations);
    }
    
    /**
     * 内部聚类方法
     */
    private void clusterInternal(IMatrix<Double> data, int numClusters) {
        if (data == null || data.getRowNum() == 0) {
            throw new IllegalArgumentException("数据矩阵不能为空");
        }
        
        int n = data.getRowNum();
        int d = data.getColNum();
        this.dimension = d;
        this.numClusters = numClusters;
        
        if (numClusters <= 0 || numClusters > n) {
            throw new IllegalArgumentException("聚类数量必须在1到数据点数量之间");
        }
        
        // 初始化聚类中心
        IMatrix<Double> initialCentersMatrix = initializeCenters(data, numClusters);
        List<IVector<Double>> initialCenters = convertToVectorList(initialCentersMatrix);
        
        // 执行K-means聚类
        KMeansResult result = performKMeansClustering(data, initialCentersMatrix);
        
        // 保存结果
        this.clusterCenters = convertToVectorList(result.getCenters());
        this.labels = result.getAssignments().clone();
        this.inertia = result.getInertia();
        this.converged = result.isConverged();
        this.iterations = result.getIterations();
        
        // 更新参数信息
        this.parameters.put("algorithm", "K-Means++");
        this.parameters.put("maxIterations", maxIterations);
        this.parameters.put("convergenceThreshold", convergenceThreshold);
    }
    
    /**
     * 将向量列表转换为矩阵
     */
    private IMatrix<Double> convertToMatrix(List<IVector<Double>> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("向量列表不能为空");
        }
        
        int rows = vectors.size();
        int cols = vectors.get(0).size();
        IMatrix<Double> matrix = Linalg.zeros(rows, cols);
        
        for (int i = 0; i < rows; i++) {
            IVector<Double> row = vectors.get(i);
            if (row.size() != cols) {
                throw new IllegalArgumentException("所有向量必须具有相同的维度");
            }
            for (int j = 0; j < cols; j++) {
                matrix.set(i, j, row.get(j));
            }
        }
        
        return matrix;
    }
    
    /**
     * 将矩阵转换为向量列表
     */
    private List<IVector<Double>> convertToVectorList(IMatrix<Double> matrixx) {
        List<IVector<Double>> vectors = new ArrayList<>();
        int rows = matrixx.getRowNum();
        int cols = matrixx.getColNum();
        
        for (int i = 0; i < rows; i++) {
            IVector<Double> vector = Linalg.zeros(cols);
            for (int j = 0; j < cols; j++) {
                vector.set(j, matrixx.get(i, j));
            }
            vectors.add(vector);
        }
        
        return vectors;
    }
    
    /**
     * K-means聚类结果
     */
    public static class KMeansResult {
        private final IMatrix<Double> centers;
        private final int[] assignments;
        private final double inertia;
        private final boolean converged;
        private final int iterations;
        
        public KMeansResult(IMatrix<Double> centers, int[] assignments, double inertia, boolean converged, int iterations) {
            this.centers = centers;
            this.assignments = assignments;
            this.inertia = inertia;
            this.converged = converged;
            this.iterations = iterations;
        }
        
        public IMatrix<Double> getCenters() { return centers; }
        public int[] getAssignments() { return assignments; }
        public double getInertia() { return inertia; }
        public boolean isConverged() { return converged; }
        public int getIterations() { return iterations; }
    }
    
    // ========== IClustering接口实现 ==========
    
    @Override
    public IClustering fit(List<IVector<Double>> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("数据列表不能为空");
        }
        
        IMatrix<Double> dataMatrix = convertToMatrix(data);
        clusterInternal(dataMatrix, this.numClusters);
        return this;
    }
    
    @Override
    public IClustering fit(IMatrix<Double> data) {
        if (data == null || data.getRowNum() == 0) {
            throw new IllegalArgumentException("数据矩阵不能为空");
        }
        
        clusterInternal(data, this.numClusters);
        return this;
    }
    
    @Override
    public int[] fitPredict(List<IVector<Double>> data) {
        fit(data);
        return getLabels();
    }
    
    @Override
    public int[] fitPredict(IMatrix<Double> data) {
        fit(data);
        return getLabels();
    }
    
    @Override
    public int[] predict(List<IVector<Double>> data) {
        if (clusterCenters == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        if (data == null) {
            throw new IllegalArgumentException("数据不能为空");
        }
        
        int[] predictions = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            predictions[i] = predict(data.get(i));
        }
        return predictions;
    }
    
    @Override
    public int predict(IVector<Double> dataPoint) {
        if (clusterCenters == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        if (dataPoint == null) {
            throw new IllegalArgumentException("数据点不能为空");
        }
        
        double minDistance = Double.POSITIVE_INFINITY;
        int bestCluster = 0;
        
        for (int i = 0; i < clusterCenters.size(); i++) {
            double distance = dataPoint.euclideanDistance(clusterCenters.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                bestCluster = i;
            }
        }
        
        return bestCluster;
    }
    

    
    @Override
    public List<IVector<Double>> getClusterCenters() {
        return clusterCenters;
    }
    
    @Override
    public int[] getLabels() {
        if (labels == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        return labels.clone();
    }
    
    @Override
    public int getNumClusters() {
        return numClusters;
    }
    
    @Override
    public int getDimension() {
        return dimension;
    }
    
    @Override
    public double getInertia() {
        return inertia;
    }
    
    @Override
    public boolean isConverged() {
        return converged;
    }
    
    @Override
    public int getIterations() {
        return iterations;
    }
    
    @Override
    public ClusteringMetrics evaluateQuality(List<IVector<Double>> data) {
        if (clusterCenters == null || labels == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        return ClusteringMetrics.compute(data, clusterCenters, labels);
    }
    
    @Override
    public String getAlgorithmName() {
        return "K-Means++";
    }
    
    @Override
    public void setParameters(Map<String, Object> params) {
        this.parameters.putAll(params);
        
        // 更新特定参数
        if (params.containsKey("numClusters")) {
            int newNumClusters = (Integer) params.get("numClusters");
            if (newNumClusters <= 0) {
                throw new IllegalArgumentException("聚类数量必须大于0");
            }
            this.numClusters = newNumClusters;
        }
        if (params.containsKey("k")) {
            int newK = (Integer) params.get("k");
            if (newK <= 0) {
                throw new IllegalArgumentException("聚类数量必须大于0");
            }
            this.numClusters = newK;
        }
    }
    
    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new HashMap<>(parameters);
        params.put("numClusters", numClusters);
        params.put("maxIterations", maxIterations);
        params.put("convergenceThreshold", convergenceThreshold);
        params.put("algorithmName", getAlgorithmName());
        return params;
    }
    
    /**
     * 将模型保存在本地
     * @param path 保存路径
     */
    @Override
    public void save(String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        } catch (IOException e) {
            log.error("exception", e);
        }
    }
}