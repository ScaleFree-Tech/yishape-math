package com.yishape.lab.math.plot.javafx.j3d;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 3D数据降采样和LOD（细节层次）工具类。
 * <p>
 * 用于处理大量数据点时的性能优化，支持：
 * <ul>
 *   <li>随机采样：随机选择指定数量的点</li>
 *   <li>均匀采样：按固定间隔采样</li>
 *   <li>分层采样：根据密度自适应采样</li>
 *   <li>LOD选择：根据视图距离选择细节层次</li>
 * </ul>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public final class DataSamplingUtils {

    private DataSamplingUtils() {
        // 工具类
    }

    /** 默认最大渲染点数 */
    public static final int DEFAULT_MAX_POINTS = 5000;

    /** 性能优化的数据点阈值 */
    public static final int PERFORMANCE_THRESHOLD = 2000;

    /**
     * 自动采样：根据数据量自动选择采样策略。
     *
     * @param data 原始数据点 [x, y, z]
     * @param maxPoints 最大保留点数
     * @return 采样后的数据点
     */
    public static double[][] autoSample(double[][] data, int maxPoints) {
        if (data == null || data.length <= maxPoints) {
            return data;
        }

        int n = data.length;

        // 根据数据量选择策略
        if (n > 100000) {
            // 大量数据：随机采样
            return randomSample(data, maxPoints, 42L);
        } else if (n > 10000) {
            // 中等数据：分层采样
            return stratifiedSample(data, maxPoints);
        } else {
            // 小数据：均匀采样
            return uniformSample(data, maxPoints);
        }
    }

    /**
     * 随机采样：随机选择指定数量的点。
     *
     * @param data 原始数据
     * @param targetSize 目标大小
     * @param seed 随机种子（可复现）
     * @return 采样后的数据
     */
    public static double[][] randomSample(double[][] data, int targetSize, long seed) {
        if (data == null || data.length <= targetSize) return data;

        Random rand = new Random(seed);
        double[][] result = new double[targetSize][];

        // Fisher-Yates shuffle的简化版
        boolean[] selected = new boolean[data.length];
        int count = 0;

        while (count < targetSize) {
            int idx = rand.nextInt(data.length);
            if (!selected[idx]) {
                selected[idx] = true;
                result[count++] = data[idx];
            }
        }

        return result;
    }

    /**
     * 均匀采样：按固定间隔选择点。
     *
     * @param data 原始数据
     * @param targetSize 目标大小
     * @return 采样后的数据
     */
    public static double[][] uniformSample(double[][] data, int targetSize) {
        if (data == null || data.length <= targetSize) return data;

        int step = data.length / targetSize;
        if (step < 1) step = 1;

        List<double[]> result = new ArrayList<>(targetSize);
        for (int i = 0; i < data.length; i += step) {
            result.add(data[i]);
            if (result.size() >= targetSize) break;
        }

        return result.toArray(new double[0][]);
    }

    /**
     * 分层采样：根据数据密度进行自适应采样。
     * 在密集区域采样更多点，稀疏区域采样较少点。
     *
     * @param data 原始数据
     * @param targetSize 目标大小
     * @return 采样后的数据
     */
    public static double[][] stratifiedSample(double[][] data, int targetSize) {
        if (data == null || data.length <= targetSize) return data;

        // 计算数据边界
        double[] bounds = CoordinateMapper.calculateBounds(java.util.Arrays.asList(data));
        double rangeX = bounds[1] - bounds[0];
        double rangeY = bounds[3] - bounds[2];
        double rangeZ = bounds[5] - bounds[4];

        // 创建网格进行空间划分
        int gridSize = (int) Math.ceil(Math.cbrt(targetSize));
        double cellSizeX = rangeX / gridSize;
        double cellSizeY = rangeY / gridSize;
        double cellSizeZ = rangeZ / gridSize;

        // 按网格分组
        List<double[]>[][][] grid = new ArrayList[gridSize][gridSize][gridSize];
        for (double[] point : data) {
            int gx = Math.min((int) ((point[0] - bounds[0]) / cellSizeX), gridSize - 1);
            int gy = Math.min((int) ((point[1] - bounds[2]) / cellSizeY), gridSize - 1);
            int gz = Math.min((int) ((point[2] - bounds[4]) / cellSizeZ), gridSize - 1);

            if (grid[gx][gy][gz] == null) {
                grid[gx][gy][gz] = new ArrayList<>();
            }
            grid[gx][gy][gz].add(point);
        }

        // 从每个网格中采样
        List<double[]> result = new ArrayList<>(targetSize);
        int pointsPerCell = targetSize / (gridSize * gridSize * gridSize) + 1;

        Random rand = new Random(42L);
        for (int x = 0; x < gridSize; x++) {
            for (int y = 0; y < gridSize; y++) {
                for (int z = 0; z < gridSize; z++) {
                    List<double[]> cell = grid[x][y][z];
                    if (cell == null || cell.isEmpty()) continue;

                    // 从该格子中随机选取若干点
                    int sampleCount = Math.min(cell.size(), pointsPerCell);
                    for (int i = 0; i < sampleCount; i++) {
                        int idx = rand.nextInt(cell.size());
                        result.add(cell.get(idx));
                        if (result.size() >= targetSize) break;
                    }
                    if (result.size() >= targetSize) break;
                }
                if (result.size() >= targetSize) break;
            }
            if (result.size() >= targetSize) break;
        }

        return result.toArray(new double[0][]);
    }

    /**
     * 简化折线：使用Ramer-Douglas-Peucker算法简化3D折线。
     *
     * @param points 原始点序列
     * @param epsilon 简化容差（越大简化越多）
     * @return 简化后的点序列
     */
    public static double[][] simplifyLine(double[][] points, double epsilon) {
        if (points == null || points.length <= 2 || epsilon <= 0) return points;

        boolean[] keep = new boolean[points.length];
        keep[0] = true;
        keep[points.length - 1] = true;

        rdpRecursive(points, 0, points.length - 1, epsilon, keep);

        List<double[]> result = new ArrayList<>();
        for (int i = 0; i < points.length; i++) {
            if (keep[i]) result.add(points[i]);
        }

        return result.toArray(new double[0][]);
    }

    private static void rdpRecursive(double[][] points, int start, int end,
                                       double epsilon, boolean[] keep) {
        if (end <= start + 1) return;

        double maxDist = 0;
        int maxIdx = start;

        for (int i = start + 1; i < end; i++) {
            double dist = pointToLineDistance(points[i], points[start], points[end]);
            if (dist > maxDist) {
                maxDist = dist;
                maxIdx = i;
            }
        }

        if (maxDist > epsilon) {
            keep[maxIdx] = true;
            rdpRecursive(points, start, maxIdx, epsilon, keep);
            rdpRecursive(points, maxIdx, end, epsilon, keep);
        }
    }

    private static double pointToLineDistance(double[] p, double[] lineStart, double[] lineEnd) {
        double[] v = new double[]{
                lineEnd[0] - lineStart[0],
                lineEnd[1] - lineStart[1],
                lineEnd[2] - lineStart[2]
        };
        double[] w = new double[]{
                p[0] - lineStart[0],
                p[1] - lineStart[1],
                p[2] - lineStart[2]
        };

        double lenSq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
        if (lenSq < 1e-12) {
            return Math.sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
        }

        double t = Math.max(0, Math.min(1, (w[0]*v[0] + w[1]*v[1] + w[2]*v[2]) / lenSq));
        double[] projection = new double[]{
                lineStart[0] + t * v[0],
                lineStart[1] + t * v[1],
                lineStart[2] + t * v[2]
        };

        double dx = p[0] - projection[0];
        double dy = p[1] - projection[1];
        double dz = p[2] - projection[2];

        return Math.sqrt(dx*dx + dy*dy + dz*dz);
    }

    /**
     * LOD级别枚举
     */
    public enum LodLevel {
        HIGH(1.0),      // 全细节
        MEDIUM(0.5),    // 50%采样
        LOW(0.25),      // 25%采样
        VERY_LOW(0.1)   // 10%采样
        ;

        private final double ratio;

        LodLevel(double ratio) { this.ratio = ratio; }
        public double getRatio() { return ratio; }
    }

    /**
     * 根据LOD级别采样数据。
     *
     * @param data 原始数据
     * @param level LOD级别
     * @return 采样后的数据
     */
    public static double[][] lodSample(double[][] data, LodLevel level) {
        if (data == null || level == null || level == LodLevel.HIGH) return data;
        int targetSize = Math.max(10, (int) (data.length * level.getRatio()));
        return uniformSample(data, targetSize);
    }

    /**
     * 根据相机距离计算推荐的LOD级别。
     *
     * @param cameraDistance 相机距离
     * @param maxDistance 最大可视距离
     * @return 推荐的LOD级别
     */
    public static LodLevel calculateLodLevel(double cameraDistance, double maxDistance) {
        double ratio = cameraDistance / maxDistance;
        if (ratio < 0.25) return LodLevel.HIGH;
        if (ratio < 0.5) return LodLevel.MEDIUM;
        if (ratio < 0.75) return LodLevel.LOW;
        return LodLevel.VERY_LOW;
    }

    /**
     * 检查是否需要降采样。
     *
     * @param dataSize 数据点数量
     * @return true如果需要降采样
     */
    public static boolean needsSampling(int dataSize) {
        return dataSize > PERFORMANCE_THRESHOLD;
    }

    /**
     * 建议的最大渲染点数（基于可用内存）。
     *
     * @return 建议的最大点数
     */
    public static int recommendMaxPoints() {
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long freeMemory = runtime.freeMemory();

        // 保守估计：每点约50字节，最多使用10%可用内存
        long availableMemory = Math.min(freeMemory, maxMemory / 4);
        int maxPoints = (int) (availableMemory / 50);

        return Math.min(maxPoints, DEFAULT_MAX_POINTS);
    }
}
