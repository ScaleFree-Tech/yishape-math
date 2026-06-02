package com.yishape.lab.math.optimize.mclp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 多目标线性规划工具类
 * Multi-Criteria Linear Programming Utility Class
 *
 * <p>提供MCLP相关的静态工具方法，包括：
 * - 极端点计算
 * - 权重生成
 * - Pareto前沿分析
 * - 规范化方法
 * - 约束转换
 *
 * @author lteb2
 */
public class MclpUtil {

    private MclpUtil() {
        // 工具类，禁止实例化
    }

    // ==================== 极端点计算 ====================

    /**
     * 计算多目标问题的极端点
     * 极端点是指在某个目标上达到最优而在其他目标上可能较差的解
     *
     * @param c 目标函数系数数组
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右端向量
     * @param baseSolver 底层单目标求解器
     * @return 极端点列表
     */
    public static Tuple2<List<IVector>, List<double[]>> computeExtremePoints(
            IVector[] c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
            com.yishape.lab.math.optimize.linpg.ILinProgSolver baseSolver) {

        List<IVector> extremePoints = new ArrayList<>();
        List<double[]> extremeObjectiveValues = new ArrayList<>();

        for (int i = 0; i < c.length; i++) {
            var result = baseSolver.solve(c[i], A_ub, b_ub, A_eq, b_eq);
            if (result.isConverged() && result.getOptimalPoint() != null) {
                IVector optimalX = result.getOptimalPoint();
                extremePoints.add(optimalX);

                double[] objVals = new double[c.length];
                for (int j = 0; j < c.length; j++) {
                    objVals[j] = c[j].innerProductValue(optimalX);
                }
                extremeObjectiveValues.add(objVals);
            }
        }

        return new Tuple2<>(extremePoints, extremeObjectiveValues);
    }

    // ==================== 理想点和反理想点 ====================

    /**
     * 计算理想点（各目标的单目标最优值）
     *
     * @param c 目标函数系数数组
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右端向量
     * @param baseSolver 底层求解器
     * @return 理想点数组
     */
    public static double[] computeIdealPoint(
            IVector[] c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
            com.yishape.lab.math.optimize.linpg.ILinProgSolver baseSolver) {

        double[] ideal = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            var result = baseSolver.solve(c[i], A_ub, b_ub, A_eq, b_eq);
            ideal[i] = result.isConverged() ? result.getOptimalValue() : Double.MAX_VALUE;
        }
        return ideal;
    }

    /**
     * 计算反理想点（各目标的单目标最差值）
     *
     * @param c 目标函数系数数组
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @param A_eq 等式约束矩阵
     * @param b_eq 等式约束右端向量
     * @param baseSolver 底层求解器
     * @return 反理想点数组
     */
    public static double[] computeNadirPoint(
            IVector[] c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
            com.yishape.lab.math.optimize.linpg.ILinProgSolver baseSolver) {

        double[] nadir = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            IVector negC = c[i].multiplyByScalar(-1.0);
            var result = baseSolver.solve(negC, A_ub, b_ub, A_eq, b_eq);
            nadir[i] = result.isConverged() ? -result.getOptimalValue() : Double.MAX_VALUE;
        }
        return nadir;
    }

    /**
     * 同时计算理想点和反理想点
     *
     * @return Tuple3(理想点, 反理想点, 各极端点)
     */
    public static Tuple3<double[], double[], List<double[]>> computeIdealAndNadirPoints(
            IVector[] c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
            com.yishape.lab.math.optimize.linpg.ILinProgSolver baseSolver) {

        List<double[]> extremeObjectiveValues = new ArrayList<>();
        double[] ideal = new double[c.length];
        double[] nadir = new double[c.length];

        // 初始化
        for (int i = 0; i < c.length; i++) {
            ideal[i] = Double.MAX_VALUE;
            nadir[i] = -Double.MAX_VALUE;
        }

        // 计算各目标的极端解
        for (int i = 0; i < c.length; i++) {
            var result = baseSolver.solve(c[i], A_ub, b_ub, A_eq, b_eq);
            if (result.isConverged()) {
                IVector optimalX = result.getOptimalPoint();
                double[] objVals = new double[c.length];
                for (int j = 0; j < c.length; j++) {
                    objVals[j] = c[j].innerProductValue(optimalX);
                }
                extremeObjectiveValues.add(objVals);

                // 更新理想点和反理想点
                for (int j = 0; j < c.length; j++) {
                    ideal[j] = Math.min(ideal[j], objVals[j]);
                    nadir[j] = Math.max(nadir[j], objVals[j]);
                }
            }

            // 计算反问题（最大化目标）
            IVector negC = c[i].multiplyByScalar(-1.0);
            result = baseSolver.solve(negC, A_ub, b_ub, A_eq, b_eq);
            if (result.isConverged()) {
                IVector optimalX = result.getOptimalPoint();
                double[] objVals = new double[c.length];
                for (int j = 0; j < c.length; j++) {
                    objVals[j] = c[j].innerProductValue(optimalX);
                }
                extremeObjectiveValues.add(objVals);

                for (int j = 0; j < c.length; j++) {
                    nadir[j] = Math.max(nadir[j], objVals[j]);
                }
            }
        }

        return new Tuple3<>(ideal, nadir, extremeObjectiveValues);
    }

    // ==================== 权重生成 ====================

    /**
     * 生成均匀分布的权重向量
     *
     * @param numObjectives 目标数量
     * @param numSamples 每维采样点数
     * @return 权重向量列表
     */
    public static List<double[]> generateUniformWeights(int numObjectives, int numSamples) {
        List<double[]> weights = new ArrayList<>();
        if (numSamples <= 1) {
            double[] w = new double[numObjectives];
            Arrays.fill(w, 1.0 / numObjectives);
            weights.add(w);
            return weights;
        }
        double step = 1.0 / (numSamples - 1);

        // 简化为双目标情况
        if (numObjectives == 2) {
            for (int i = 0; i < numSamples; i++) {
                double w1 = i * step;
                double w2 = 1.0 - w1;
                weights.add(new double[]{w1, w2});
            }
        } else {
            // 多目标情况：生成稀疏网格
            generateMultiDimensionalWeights(weights, new double[numObjectives], 0, numSamples);
        }

        return weights;
    }

    private static void generateMultiDimensionalWeights(List<double[]> weights, double[] current,
                                                       int dim, int numSamples) {
        if (dim == current.length) {
            double sum = 0.0;
            for (double w : current) sum += w;
            if (sum > 1e-12) {
                double[] normalized = new double[current.length];
                for (int i = 0; i < current.length; i++) {
                    normalized[i] = current[i] / sum;
                }
                weights.add(normalized);
            }
            return;
        }

        double step = 1.0 / (numSamples - 1);
        for (int i = 0; i < numSamples; i++) {
            current[dim] = i * step;
            generateMultiDimensionalWeights(weights, current, dim + 1, numSamples);
        }
    }

    /**
     * 生成随机权重向量
     *
     * @param numObjectives 目标数量
     * @param numSamples 样本数量
     * @param seed 随机种子
     * @return 权重向量列表
     */
    public static List<double[]> generateRandomWeights(int numObjectives, int numSamples, long seed) {
        List<double[]> weights = new ArrayList<>();
        java.util.Random random = new java.util.Random(seed);

        for (int i = 0; i < numSamples; i++) {
            double[] w = new double[numObjectives];
            double sum = 0.0;

            for (int j = 0; j < numObjectives; j++) {
                w[j] = random.nextDouble();
                sum += w[j];
            }

            for (int j = 0; j < numObjectives; j++) {
                w[j] /= sum;
            }

            weights.add(w);
        }

        return weights;
    }

    /**
     * 生成基于偏好的权重向量
     *
     * @param priorities 优先级数组（数值越大优先级越高）
     * @return 权重向量
     */
    public static double[] generatePriorityWeights(int[] priorities) {
        int n = priorities.length;
        double[] weights = new double[n];
        double sum = 0.0;

        for (int i = 0; i < n; i++) {
            weights[i] = Math.pow(2.0, priorities[i]);
            sum += weights[i];
        }

        for (int i = 0; i < n; i++) {
            weights[i] /= sum;
        }

        return weights;
    }

    // ==================== Pareto分析 ====================

    /**
     * 过滤Pareto最优解
     *
     * @param objectiveValues 目标函数值列表
     * @return 非支配解索引列表
     */
    public static List<Integer> filterParetoOptimal(List<double[]> objectiveValues) {
        List<Integer> paretoIndices = new ArrayList<>();

        for (int i = 0; i < objectiveValues.size(); i++) {
            boolean isDominated = false;

            for (int j = 0; j < objectiveValues.size(); j++) {
                if (i != j && MclpResult.paretoDominates(objectiveValues.get(j), objectiveValues.get(i)) > 0) {
                    isDominated = true;
                    break;
                }
            }

            if (!isDominated) {
                paretoIndices.add(i);
            }
        }

        return paretoIndices;
    }

    /**
     * 计算Pareto前沿覆盖率
     *
     * @param paretoFront Pareto前沿
     * @param idealPoint 理想点
     * @param nadirPoint 反理想点
     * @return 覆盖率
     */
    public static double computeParetoCoverage(List<double[]> paretoFront, double[] idealPoint, double[] nadirPoint) {
        if (paretoFront.isEmpty() || idealPoint == null || nadirPoint == null) {
            return 0.0;
        }

        double totalRange = 0.0;
        for (int i = 0; i < idealPoint.length; i++) {
            totalRange += (nadirPoint[i] - idealPoint[i]) * (nadirPoint[i] - idealPoint[i]);
        }
        totalRange = Math.sqrt(totalRange);

        return totalRange > 0 ? Math.min(1.0, paretoFront.size() / (5.0 * idealPoint.length)) : 0.0;
    }

    /**
     * 计算Pareto前沿的多样性（基于扩展多样性指标）
     *
     * @param paretoFront Pareto前沿
     * @param idealPoint 理想点
     * @param nadirPoint 反理想点
     * @return 多样性指标
     */
    public static double computeDiversityMetric(List<double[]> paretoFront, double[] idealPoint, double[] nadirPoint) {
        if (paretoFront.size() < 2) {
            return 0.0;
        }

        // 计算相邻解之间的平均距离
        List<double[]> sorted = new ArrayList<>(paretoFront);
        final int objIdx = 0;
        sorted.sort((a, b) -> Double.compare(a[objIdx], b[objIdx]));

        double totalDistance = 0.0;
        for (int i = 0; i < sorted.size() - 1; i++) {
            totalDistance += euclideanDistance(sorted.get(i), sorted.get(i + 1));
        }

        // 考虑目标空间范围的归一化
        double range = 0.0;
        for (int i = 0; i < idealPoint.length; i++) {
            range += (nadirPoint[i] - idealPoint[i]) * (nadirPoint[i] - idealPoint[i]);
        }
        range = Math.sqrt(range);

        return range > 0 ? totalDistance / range : 0.0;
    }

    /**
     * 计算欧氏距离
     */
    private static double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    // ==================== 规范化 ====================

    /**
     * 线性规范化目标函数值到[0,1]范围
     *
     * @param objectiveValues 原始目标函数值
     * @param idealPoint 理想点
     * @param nadirPoint 反理想点
     * @return 规范化后的值
     */
    public static double[][] normalizeObjectives(List<double[]> objectiveValues,
                                                 double[] idealPoint, double[] nadirPoint) {
        int numSolutions = objectiveValues.size();
        int numObjectives = objectiveValues.get(0).length;
        double[][] normalized = new double[numSolutions][numObjectives];

        for (int i = 0; i < numSolutions; i++) {
            for (int j = 0; j < numObjectives; j++) {
                double range = nadirPoint[j] - idealPoint[j];
                if (Math.abs(range) > 1e-12) {
                    normalized[i][j] = (objectiveValues.get(i)[j] - idealPoint[j]) / range;
                } else {
                    normalized[i][j] = 0.0;
                }
            }
        }

        return normalized;
    }

    /**
     * 向量化规范化
     *
     * @param objectiveValues 原始目标函数值
     * @return 规范化后的值
     */
    public static double[] normalizeVector(double[] objectiveValues) {
        double sumSquares = 0.0;
        for (double v : objectiveValues) {
            sumSquares += v * v;
        }
        double norm = Math.sqrt(sumSquares);

        if (norm < 1e-12) {
            return objectiveValues.clone();
        }

        double[] normalized = new double[objectiveValues.length];
        for (int i = 0; i < objectiveValues.length; i++) {
            normalized[i] = objectiveValues[i] / norm;
        }

        return normalized;
    }

    // ==================== 约束转换 ====================

    /**
     * 将不等式约束转换为等式约束（添加松弛/剩余变量）
     *
     * @param A_ub 不等式约束矩阵
     * @param b_ub 不等式约束右端向量
     * @return Tuple3(转换后的等式约束矩阵, 转换后的右端向量, 松弛/剩余变量数量)
     */
    public static Tuple3<IMatrix, IVector, Integer> convertUbToEqConstraints(
            IMatrix A_ub, IVector b_ub) {

        if (A_ub == null || b_ub == null) {
            return new Tuple3<>(null, null, 0);
        }

        int numConstraints = A_ub.rows();
        int numVariables = A_ub.cols();
        int numSlackVariables = numConstraints;

        // 构建扩展矩阵 [A_ub | I] 或 [A_ub | -I]
        double[][] extendedData = new double[numConstraints][numVariables + numSlackVariables];

        for (int i = 0; i < numConstraints; i++) {
            // 复制原约束系数
            for (int j = 0; j < numVariables; j++) {
                extendedData[i][j] = (Double) A_ub.get(i, j);
            }
            // 添加松弛变量系数（对于<=约束，添加正松弛变量）
            extendedData[i][numVariables + i] = 1.0;
        }

        IMatrix extendedAeq = Linalg.matrix(extendedData);
        IVector extendedBeq = b_ub.copy();

        return new Tuple3<>(extendedAeq, extendedBeq, numSlackVariables);
    }

    /**
     * 扩展变量向量（添加松弛变量）
     *
     * @param originalX 原变量向量
     * @param numSlackVariables 松弛变量数量
     * @return 扩展后的变量向量
     */
    public static IVector extendVariables(IVector originalX, int numSlackVariables) {
        int originalLen = originalX.length();
        double[] extendedData = new double[originalLen + numSlackVariables];

        for (int i = 0; i < originalLen; i++) {
            extendedData[i] = (Double) originalX.get(i);
        }

        // 松弛变量初始值为0
        for (int i = originalLen; i < extendedData.length; i++) {
            extendedData[i] = 0.0;
        }

        return Linalg.vector(extendedData);
    }

    /**
     * 从扩展解中提取原始变量
     *
     * @param extendedX 扩展解向量
     * @param numOriginalVariables 原始变量数量
     * @return 原始变量向量
     */
    public static IVector extractOriginalVariables(IVector extendedX, int numOriginalVariables) {
        double[] originalData = new double[numOriginalVariables];

        for (int i = 0; i < numOriginalVariables; i++) {
            originalData[i] = (Double) extendedX.get(i);
        }

        return Linalg.vector(originalData);
    }

    // ==================== 评估指标 ====================

    /**
     * 计算Hypervolume指标（仅适用于双目标问题）
     *
     * @param paretoFront Pareto前沿
     * @param referencePoint 参考点
     * @return Hypervolume值
     */
    public static double computeHypervolume(List<double[]> paretoFront, double[] referencePoint) {
        if (paretoFront.isEmpty() || referencePoint == null || referencePoint.length != 2) {
            return 0.0;
        }

        // 按第一个目标排序
        List<double[]> sorted = new ArrayList<>(paretoFront);
        sorted.sort((a, b) -> Double.compare(a[0], b[0]));

        double hypervolume = 0.0;
        for (double[] point : sorted) {
            double width = referencePoint[0] - point[0];
            double height = referencePoint[1] - point[1];
            if (width > 0 && height > 0) {
                hypervolume += width * height;
            }
        }

        return hypervolume;
    }

    /**
     * 计算反转指标（Inverted Generational Distance）
     *
     * @param paretoFront 获得的Pareto前沿
     * @param trueParetoFront 真实Pareto前沿
     * @return IGD值
     */
    public static double computeIGD(List<double[]> paretoFront, List<double[]> trueParetoFront) {
        if (trueParetoFront.isEmpty()) {
            return 0.0;
        }

        double totalDistance = 0.0;
        for (double[] truePoint : trueParetoFront) {
            double minDistance = Double.MAX_VALUE;
            for (double[] approxPoint : paretoFront) {
                double dist = euclideanDistance(truePoint, approxPoint);
                minDistance = Math.min(minDistance, dist);
            }
            totalDistance += minDistance;
        }

        return totalDistance / trueParetoFront.size();
    }

    /**
     * 计算代际距离（Generational Distance）
     *
     * @param paretoFront 获得的Pareto前沿
     * @param trueParetoFront 真实Pareto前沿
     * @return GD值
     */
    public static double computeGD(List<double[]> paretoFront, List<double[]> trueParetoFront) {
        if (paretoFront.isEmpty()) {
            return 0.0;
        }

        double totalDistance = 0.0;
        for (double[] approxPoint : paretoFront) {
            double minDistance = Double.MAX_VALUE;
            for (double[] truePoint : trueParetoFront) {
                double dist = euclideanDistance(approxPoint, truePoint);
                minDistance = Math.min(minDistance, dist);
            }
            totalDistance += minDistance;
        }

        return totalDistance / paretoFront.size();
    }

    // ==================== 决策支持 ====================

    /**
     * 使用TOPSIS选择最佳解
     *
     * @param solutions 解集
     * @param objectiveValues 目标函数值
     * @param weights 权重
     * @return 最佳解的索引
     */
    public static int selectBestByTopsis(List<IVector> solutions, List<double[]> objectiveValues, double[] weights) {
        if (solutions.isEmpty() || objectiveValues.isEmpty() || solutions.size() != objectiveValues.size()) {
            return -1;
        }

        int numSolutions = solutions.size();
        int numObjectives = objectiveValues.get(0).length;

        // 归一化：先计算理想点与反理想点
        double[] idealPoint = new double[numObjectives];
        double[] nadirPoint = new double[numObjectives];
        for (int j = 0; j < numObjectives; j++) {
            double minVal = Double.MAX_VALUE;
            double maxVal = -Double.MAX_VALUE;
            for (int i = 0; i < numSolutions; i++) {
                minVal = Math.min(minVal, objectiveValues.get(i)[j]);
                maxVal = Math.max(maxVal, objectiveValues.get(i)[j]);
            }
            idealPoint[j] = minVal;
            nadirPoint[j] = maxVal;
        }
        double[][] normalized = normalizeObjectives(objectiveValues, idealPoint, nadirPoint);

        // 加权
        double[][] weighted = new double[numSolutions][numObjectives];
        for (int i = 0; i < numSolutions; i++) {
            for (int j = 0; j < numObjectives; j++) {
                weighted[i][j] = normalized[i][j] * weights[j];
            }
        }

        // 计算理想解和负理想解
        double[] ideal = new double[numObjectives];
        double[] negativeIdeal = new double[numObjectives];

        for (int j = 0; j < numObjectives; j++) {
            double maxVal = -Double.MAX_VALUE;
            double minVal = Double.MAX_VALUE;
            for (int i = 0; i < numSolutions; i++) {
                maxVal = Math.max(maxVal, weighted[i][j]);
                minVal = Math.min(minVal, weighted[i][j]);
            }
            ideal[j] = maxVal;
            negativeIdeal[j] = minVal;
        }

        // 计算贴近度
        double[] closeness = new double[numSolutions];
        for (int i = 0; i < numSolutions; i++) {
            double distIdeal = euclideanDistance(weighted[i], ideal);
            double distNegIdeal = euclideanDistance(weighted[i], negativeIdeal);
            if (distIdeal + distNegIdeal > 1e-12) {
                closeness[i] = distNegIdeal / (distIdeal + distNegIdeal);
            }
        }

        // 选择最高贴近度
        int bestIndex = 0;
        double maxCloseness = closeness[0];
        for (int i = 1; i < numSolutions; i++) {
            if (closeness[i] > maxCloseness) {
                maxCloseness = closeness[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    /**
     * 根据决策者指定的目标值找到最接近的解
     *
     * @param solutions 解集
     * @param objectiveValues 目标函数值
     * @param targetValues 目标值
     * @return 最接近目标值的解索引
     */
    public static int selectClosestToTarget(List<IVector> solutions, List<double[]> objectiveValues, double[] targetValues) {
        if (solutions.isEmpty() || objectiveValues.isEmpty() || targetValues == null) {
            return -1;
        }

        int bestIndex = 0;
        double minDistance = euclideanDistance(objectiveValues.get(0), targetValues);

        for (int i = 1; i < objectiveValues.size(); i++) {
            double dist = euclideanDistance(objectiveValues.get(i), targetValues);
            if (dist < minDistance) {
                minDistance = dist;
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}
