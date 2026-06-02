package com.yishape.lab.math.optimize.mcqp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IQpSolver;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * 多目标二次规划工具类
 * Multi-Criteria Quadratic Programming Utility Class
 *
 * @author lteb2
 */
public class McqpUtil {

    private McqpUtil() {
        // 工具类，禁止实例化
    }

    // ==================== 极端点计算 ====================

    /**
     * 计算多目标二次规划问题的极端点
     */
    public static Tuple2<List<IVector>, List<double[]>> computeExtremePoints(
            IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
            IQpSolver baseSolver) {

        List<IVector> extremePoints = new ArrayList<>();
        List<double[]> extremeObjectiveValues = new ArrayList<>();

        for (int i = 0; i < c.length; i++) {
            var result = baseSolver.solve(Q[i], c[i], A_ub, b_ub, A_eq, b_eq, null);
            if (result.isConverged() && result.getOptimalPoint() != null) {
                IVector optimalX = result.getOptimalPoint();
                extremePoints.add(optimalX);

                double[] objVals = new double[c.length];
                for (int j = 0; j < c.length; j++) {
                    objVals[j] = computeQuadraticObjective(Q[j], c[j], optimalX);
                }
                extremeObjectiveValues.add(objVals);
            }
        }

        return new Tuple2<>(extremePoints, extremeObjectiveValues);
    }

    // ==================== 理想点和反理想点 ====================

    /**
     * 计算理想点（各目标的单目标最优值）
     */
    public static double[] computeIdealPoint(
            IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
            IQpSolver baseSolver) {

        double[] ideal = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            var result = baseSolver.solve(Q[i], c[i], A_ub, b_ub, A_eq, b_eq, null);
            ideal[i] = result.isConverged() ? result.getOptimalValue() : Double.NaN;
        }
        return ideal;
    }

    /**
     * 计算反理想点（各目标的单目标最差值）
     */
    public static double[] computeNadirPoint(
            IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
            IQpSolver baseSolver) {

        double[] nadir = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            IMatrix negQ = Q[i].multiplyByScalar(-1.0);
            IVector negC = c[i].multiplyByScalar(-1.0);
            var result = baseSolver.solve(negQ, negC, A_ub, b_ub, A_eq, b_eq, null);
            nadir[i] = result.isConverged() ? -result.getOptimalValue() : Double.NaN;
        }
        return nadir;
    }

    /**
     * 同时计算理想点和反理想点
     */
    public static Tuple3<double[], double[], List<double[]>> computeIdealAndNadirPoints(
            IMatrix[] Q, IVector[] c, IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq,
            IQpSolver baseSolver) {

        List<double[]> extremeObjectiveValues = new ArrayList<>();
        double[] ideal = new double[c.length];
        double[] nadir = new double[c.length];

        Arrays.fill(ideal, Double.MAX_VALUE);
        Arrays.fill(nadir, -Double.MAX_VALUE);

        for (int i = 0; i < c.length; i++) {
            var result = baseSolver.solve(Q[i], c[i], A_ub, b_ub, A_eq, b_eq, null);
            if (result.isConverged()) {
                IVector optimalX = result.getOptimalPoint();
                double[] objVals = new double[c.length];
                for (int j = 0; j < c.length; j++) {
                    objVals[j] = computeQuadraticObjective(Q[j], c[j], optimalX);
                }
                extremeObjectiveValues.add(objVals);

                for (int j = 0; j < c.length; j++) {
                    ideal[j] = Math.min(ideal[j], objVals[j]);
                    nadir[j] = Math.max(nadir[j], objVals[j]);
                }
            }

            IMatrix negQ = Q[i].multiplyByScalar(-1.0);
            IVector negC = c[i].multiplyByScalar(-1.0);
            result = baseSolver.solve(negQ, negC, A_ub, b_ub, A_eq, b_eq, null);
            if (result.isConverged()) {
                IVector optimalX = result.getOptimalPoint();
                double[] objVals = new double[c.length];
                for (int j = 0; j < c.length; j++) {
                    objVals[j] = computeQuadraticObjective(Q[j], c[j], optimalX);
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
     */
    public static List<double[]> generateUniformWeights(int numObjectives, int numSamples) {
        List<double[]> weights = new ArrayList<>();
        if (numSamples <= 1) {
            double[] w = new double[numObjectives];
            Arrays.fill(w, 1.0 / numObjectives);
            weights.add(w);
            return weights;
        }

        if (numObjectives == 2) {
            double step = 1.0 / (numSamples - 1);
            for (int i = 0; i < numSamples; i++) {
                weights.add(new double[]{i * step, 1.0 - i * step});
            }
        } else {
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
                for (int i = 0; i < current.length; i++) normalized[i] = current[i] / sum;
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
     */
    public static List<double[]> generateRandomWeights(int numObjectives, int numSamples, long seed) {
        List<double[]> weights = new ArrayList<>();
        Random random = new Random(seed);

        for (int i = 0; i < numSamples; i++) {
            double[] w = new double[numObjectives];
            double sum = 0.0;
            for (int j = 0; j < numObjectives; j++) {
                w[j] = random.nextDouble();
                sum += w[j];
            }
            for (int j = 0; j < numObjectives; j++) w[j] /= sum;
            weights.add(w);
        }
        return weights;
    }

    /**
     * 生成基于偏好的权重向量
     */
    public static double[] generatePriorityWeights(int[] priorities) {
        int n = priorities.length;
        double[] weights = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            weights[i] = Math.pow(2.0, priorities[i]);
            sum += weights[i];
        }
        for (int i = 0; i < n; i++) weights[i] /= sum;
        return weights;
    }

    // ==================== Pareto分析 ====================

    /**
     * 过滤Pareto最优解
     */
    public static List<Integer> filterParetoOptimal(List<double[]> objectiveValues) {
        List<Integer> paretoIndices = new ArrayList<>();
        for (int i = 0; i < objectiveValues.size(); i++) {
            boolean isDominated = false;
            for (int j = 0; j < objectiveValues.size(); j++) {
                if (i != j && McqpResult.paretoDominates(objectiveValues.get(j), objectiveValues.get(i)) > 0) {
                    isDominated = true;
                    break;
                }
            }
            if (!isDominated) paretoIndices.add(i);
        }
        return paretoIndices;
    }

    /**
     * 计算Pareto前沿覆盖率
     */
    public static double computeParetoCoverage(List<double[]> paretoFront, double[] idealPoint, double[] nadirPoint) {
        if (paretoFront.isEmpty() || idealPoint == null || nadirPoint == null) return 0.0;
        double totalRange = 0.0;
        for (int i = 0; i < idealPoint.length; i++) {
            double diff = nadirPoint[i] - idealPoint[i];
            totalRange += diff * diff;
        }
        totalRange = Math.sqrt(totalRange);
        return totalRange > 0 ? Math.min(1.0, paretoFront.size() / (5.0 * idealPoint.length)) : 0.0;
    }

    /**
     * 计算Pareto前沿的多样性
     */
    public static double computeDiversityMetric(List<double[]> paretoFront, double[] idealPoint, double[] nadirPoint) {
        if (paretoFront.size() < 2) return 0.0;
        List<double[]> sorted = new ArrayList<>(paretoFront);
        sorted.sort((a, b) -> Double.compare(a[0], b[0]));
        double totalDistance = 0.0;
        for (int i = 0; i < sorted.size() - 1; i++) {
            totalDistance += euclideanDistance(sorted.get(i), sorted.get(i + 1));
        }
        double range = euclideanDistance(idealPoint, nadirPoint);
        return range > 0 ? totalDistance / range : 0.0;
    }

    // ==================== 规范化 ====================

    /**
     * 线性规范化目标函数值到[0,1]范围
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

    // ==================== 评估指标 ====================

    /**
     * 计算Hypervolume指标（仅适用于双目标问题）
     */
    public static double computeHypervolume(List<double[]> paretoFront, double[] referencePoint) {
        if (paretoFront.isEmpty() || referencePoint == null || referencePoint.length != 2) return 0.0;
        List<double[]> sorted = new ArrayList<>(paretoFront);
        sorted.sort((a, b) -> Double.compare(a[0], b[0]));
        double hypervolume = 0.0;
        for (double[] point : sorted) {
            double width = referencePoint[0] - point[0];
            double height = referencePoint[1] - point[1];
            if (width > 0 && height > 0) hypervolume += width * height;
        }
        return hypervolume;
    }

    // ==================== 辅助方法 ====================

    /**
     * 计算二次目标函数值：1/2 * x^T * Q * x + c^T * x
     */
    public static double computeQuadraticObjective(IMatrix Q, IVector c, IVector x) {
        IVector qx = Q.mmul(x);
        double xqx = x.dotValue(qx);
        double cx = c.dotValue(x);
        return 0.5 * xqx + cx;
    }

    /**
     * 计算欧氏距离
     */
    public static double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * 计算欧氏距离（使用IVector）
     */
    public static double euclideanDistance(IVector a, IVector b) {
        double sum = 0.0;
        for (int i = 0; i < a.length(); i++) {
            double diff = ((Number) a.get(i)).doubleValue() - ((Number) b.get(i)).doubleValue();
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}
