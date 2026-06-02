package com.yishape.lab.math.ml.clu;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.IRichReport;
import com.yishape.lab.util.ReportBuilder;

import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * 聚类评估指标类 / Clustering Evaluation Metrics Class
 * <p>
 * 提供多种聚类质量评估指标的计算。
 * Provides computation of various clustering quality evaluation metrics.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class ClusteringMetrics implements IRichReport {
    
    /** 惯性（类内平方和）/ Inertia (within-cluster sum of squares) */
    private final double inertia;
    
    /** 轮廓系数 / Silhouette coefficient */
    private final double silhouetteScore;
    
    /** Calinski-Harabasz指数 / Calinski-Harabasz index */
    private final double calinskiHarabaszIndex;
    
    /** Davies-Bouldin指数 / Davies-Bouldin index */
    private final double daviesBouldinIndex;
    
    /** 类间距离 / Between-cluster distance */
    private final double betweenClusterDistance;
    
    /** 类内距离 / Within-cluster distance */
    private final double withinClusterDistance;
    
    /** 附加指标 / Additional metrics */
    private final Map<String, Double> additionalMetrics;
    
    /**
     * 构造函数
     * Constructor
     */
    public ClusteringMetrics(double inertia, double silhouetteScore, double calinskiHarabaszIndex,
                           double daviesBouldinIndex, double betweenClusterDistance, 
                           double withinClusterDistance, Map<String, Double> additionalMetrics) {
        this.inertia = inertia;
        this.silhouetteScore = silhouetteScore;
        this.calinskiHarabaszIndex = calinskiHarabaszIndex;
        this.daviesBouldinIndex = daviesBouldinIndex;
        this.betweenClusterDistance = betweenClusterDistance;
        this.withinClusterDistance = withinClusterDistance;
        this.additionalMetrics = additionalMetrics != null ? 
            new HashMap<>(additionalMetrics) : new HashMap<>();
    }
    
    /**
     * 计算聚类评估指标
     * Compute clustering evaluation metrics
     * 
     * @param data 原始数据 / Original data
     * @param centers 聚类中心 / Cluster centers
     * @param labels 聚类标签 / Cluster labels
     * @return 评估指标 / Evaluation metrics
     */
    public static ClusteringMetrics compute(List<IVector<Double>> data, List<IVector<Double>> centers, int[] labels) {
        if (data == null || data.isEmpty() || centers == null || centers.isEmpty() || labels == null) {
            throw new IllegalArgumentException("数据、聚类中心和聚类标签不能为空");
        }
        
        int n = data.size();
        int k = centers.size();
        
        // 计算惯性
        double inertia = computeInertia(data, centers, labels);
        
        // 计算轮廓系数
        double silhouetteScore = computeSilhouetteScore(data, labels, k);
        
        // 计算Calinski-Harabasz指数
        double calinskiHarabaszIndex = computeCalinskiHarabaszIndex(data, centers, labels, k);
        
        // 计算Davies-Bouldin指数
        double daviesBouldinIndex = computeDaviesBouldinIndex(data, centers, labels, k);
        
        // 计算类间和类内距离
        double betweenClusterDistance = computeBetweenClusterDistance(centers);
        double withinClusterDistance = computeWithinClusterDistance(data, centers, labels);
        
        return new ClusteringMetrics(inertia, silhouetteScore, calinskiHarabaszIndex,
                                   daviesBouldinIndex, betweenClusterDistance, 
                                   withinClusterDistance, null);
    }
    

    
    /**
     * 计算惯性（类内平方和）
     * Compute inertia (within-cluster sum of squares)
     */
    private static double computeInertia(List<IVector<Double>> data, 
                                       List<IVector<Double>> centers, int[] labels) {
        double inertia = 0.0;
        for (int i = 0; i < data.size(); i++) {
            IVector<Double> point = data.get(i);
            IVector<Double> center = centers.get(labels[i]);
            double distance = euclideanDistance(point, center);
            inertia += distance * distance;
        }
        return inertia;
    }
    
    /**
     * 计算轮廓系数
     * Compute silhouette score
     */
    private static double computeSilhouetteScore(List<IVector<Double>> data, int[] labels, int k) {
        int n = data.size();
        double totalSilhouette = 0.0;
        
        for (int i = 0; i < n; i++) {
            double a = computeIntraClusterDistance(data, labels, i);
            double b = computeNearestClusterDistance(data, labels, i, k);
            
            double silhouette = (b - a) / Math.max(a, b);
            totalSilhouette += silhouette;
        }
        
        return totalSilhouette / n;
    }
    
    /**
     * 计算类内平均距离
     * Compute intra-cluster average distance
     */
    private static double computeIntraClusterDistance(List<IVector<Double>> data, int[] labels, int pointIndex) {
        IVector<Double> point = data.get(pointIndex);
        int cluster = labels[pointIndex];
        
        double totalDistance = 0.0;
        int count = 0;
        
        for (int i = 0; i < data.size(); i++) {
            if (i != pointIndex && labels[i] == cluster) {
                totalDistance += euclideanDistance(point, data.get(i));
                count++;
            }
        }
        
        return count > 0 ? totalDistance / count : 0.0;
    }
    
    /**
     * 计算到最近其他聚类的平均距离
     * Compute average distance to nearest other cluster
     */
    private static double computeNearestClusterDistance(List<IVector<Double>> data, int[] labels, 
                                                      int pointIndex, int k) {
        IVector<Double> point = data.get(pointIndex);
        int currentCluster = labels[pointIndex];
        
        double minAvgDistance = Double.POSITIVE_INFINITY;
        
        for (int cluster = 0; cluster < k; cluster++) {
            if (cluster == currentCluster) continue;
            
            double totalDistance = 0.0;
            int count = 0;
            
            for (int i = 0; i < data.size(); i++) {
                if (labels[i] == cluster) {
                    totalDistance += euclideanDistance(point, data.get(i));
                    count++;
                }
            }
            
            if (count > 0) {
                double avgDistance = totalDistance / count;
                minAvgDistance = Math.min(minAvgDistance, avgDistance);
            }
        }
        
        return minAvgDistance == Double.POSITIVE_INFINITY ? 0.0 : minAvgDistance;
    }
    
    /**
     * 计算Calinski-Harabasz指数
     * Compute Calinski-Harabasz index
     */
    private static double computeCalinskiHarabaszIndex(List<IVector<Double>> data, 
                                                     List<IVector<Double>> centers, 
                                                     int[] labels, int k) {
        int n = data.size();
        if (k <= 1 || n <= k) return 0.0;
        
        // 计算全局中心
        IVector<Double> globalCenter = computeGlobalCenter(data);
        
        // 计算类间平方和
        double betweenSS = 0.0;
        for (int i = 0; i < k; i++) {
            int clusterSize = 0;
            for (int label : labels) {
                if (label == i) clusterSize++;
            }
            if (clusterSize > 0) {
                double distance = euclideanDistance(centers.get(i), globalCenter);
                betweenSS += clusterSize * distance * distance;
            }
        }
        
        // 计算类内平方和
        double withinSS = computeInertia(data, centers, labels);
        
        if (withinSS == 0.0) return 0.0;
        
        return (betweenSS / (k - 1)) / (withinSS / (n - k));
    }
    
    /**
     * 计算Davies-Bouldin指数
     * Compute Davies-Bouldin index
     */
    private static double computeDaviesBouldinIndex(List<IVector<Double>> data, 
                                                  List<IVector<Double>> centers, 
                                                  int[] labels, int k) {
        double[] clusterScatter = new double[k];
        
        // 计算每个聚类的散布度
        for (int i = 0; i < k; i++) {
            double totalDistance = 0.0;
            int count = 0;
            
            for (int j = 0; j < data.size(); j++) {
                if (labels[j] == i) {
                    totalDistance += euclideanDistance(data.get(j), centers.get(i));
                    count++;
                }
            }
            
            clusterScatter[i] = count > 0 ? totalDistance / count : 0.0;
        }
        
        // 计算Davies-Bouldin指数
        double totalDB = 0.0;
        for (int i = 0; i < k; i++) {
            double maxRatio = 0.0;
            for (int j = 0; j < k; j++) {
                if (i != j) {
                    double centerDistance = euclideanDistance(centers.get(i), centers.get(j));
                    if (centerDistance > 0) {
                        double ratio = (clusterScatter[i] + clusterScatter[j]) / centerDistance;
                        maxRatio = Math.max(maxRatio, ratio);
                    }
                }
            }
            totalDB += maxRatio;
        }
        
        return totalDB / k;
    }
    
    /**
     * 计算类间距离
     * Compute between-cluster distance
     */
    private static double computeBetweenClusterDistance(List<IVector<Double>> centers) {
        int k = centers.size();
        if (k <= 1) return 0.0;
        
        double totalDistance = 0.0;
        int count = 0;
        
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                totalDistance += euclideanDistance(centers.get(i), centers.get(j));
                count++;
            }
        }
        
        return count > 0 ? totalDistance / count : 0.0;
    }
    
    /**
     * 计算类内距离
     * Compute within-cluster distance
     */
    private static double computeWithinClusterDistance(List<IVector<Double>> data, 
                                                     List<IVector<Double>> centers, 
                                                     int[] labels) {
        double totalDistance = 0.0;
        int count = 0;
        
        for (int i = 0; i < data.size(); i++) {
            totalDistance += euclideanDistance(data.get(i), centers.get(labels[i]));
            count++;
        }
        
        return count > 0 ? totalDistance / count : 0.0;
    }
    
    /**
     * 计算全局中心
     * Compute global center
     */
    private static IVector<Double> computeGlobalCenter(List<IVector<Double>> data) {
        if (data.isEmpty()) return null;
        
        int dimension = data.get(0).size();
        IVector<Double> center = Linalg.zeros(dimension);
        
        for (IVector<Double> point : data) {
            center = center.add(point);
        }
        
        return center.divideByScalar((double) data.size());
    }
    
    /**
     * 计算欧几里得距离
     * Compute Euclidean distance
     */
    private static double euclideanDistance(IVector<Double> v1, IVector<Double> v2) {
        double sum = 0.0;
        for (int i = 0; i < v1.size(); i++) {
            double diff = v1.get(i) - v2.get(i);
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    // Getters
    public double getInertia() { return inertia; }
    public double getSilhouetteScore() { return silhouetteScore; }
    public double getCalinskiHarabaszIndex() { return calinskiHarabaszIndex; }
    public double getDaviesBouldinIndex() { return daviesBouldinIndex; }
    public double getBetweenClusterDistance() { return betweenClusterDistance; }
    public double getWithinClusterDistance() { return withinClusterDistance; }
    public Map<String, Double> getAdditionalMetrics() { return new HashMap<>(additionalMetrics); }
    
    @Override
    public String toReport() {
        ReportBuilder rb = new ReportBuilder("Clustering Metrics");
        rb.kv("Inertia", String.format("%.6f", inertia));
        rb.kv("Silhouette Score", silhouetteScore);
        rb.kv("Calinski-Harabasz Index", calinskiHarabaszIndex);
        rb.kv("Davies-Bouldin Index", daviesBouldinIndex);
        rb.kv("Between-cluster Dist", betweenClusterDistance);
        rb.kv("Within-cluster Dist", withinClusterDistance);
        if (!additionalMetrics.isEmpty()) {
            rb.h2("Additional Metrics");
            for (Map.Entry<String, Double> e : additionalMetrics.entrySet()) {
                rb.kv(e.getKey(), e.getValue());
            }
        }
        return rb.build();
    }

    @Override
    public String toBriefReport() {
        return String.format("Clustering | Silhouette=%.4f | CH=%.4f | DB=%.4f",
                silhouetteScore, calinskiHarabaszIndex, daviesBouldinIndex);
    }

    @Override
    public String toString() {
        return String.format("ClusteringMetrics{inertia=%.6f, silhouette=%.6f, CH=%.6f, DB=%.6f, between=%.6f, within=%.6f}",
                inertia, silhouetteScore, calinskiHarabaszIndex, daviesBouldinIndex,
                betweenClusterDistance, withinClusterDistance);
    }
}