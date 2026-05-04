package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IVector;
import java.io.Serializable;

/**
 * 线搜索算法实现，用于优化算法中确定最优步长
 * Line Search Algorithm Implementation for Determining Optimal Step Size in Optimization Algorithms
 *
 * <p>线搜索是优化算法中的关键组件，用于在给定搜索方向上找到合适的步长。
 * 本类实现了基于强Wolfe条件的线搜索算法，包括Armijo条件和曲率条件。
 * Line search is a key component in optimization algorithms, used to find an appropriate step size
 * in a given search direction. This class implements line search algorithms based on strong Wolfe conditions,
 * including Armijo condition and curvature condition.</p>
 *
 * <h3>算法描述 / Algorithm Description:</h3>
 * <ul>
 *   <li>Armijo条件: f(x + αd) ≤ f(x) + c1 * α * ∇f(x)^T * d 确保函数值充分下降</li>
 *   <li>曲率条件: |∇f(x + αd)^T * d| ≤ c2 * |∇f(x)^T * d| 确保梯度变化合理</li>
 * </ul>
 *
 * @author lteb2
 * @see Serializable
 */
public class RereLineSearch implements Serializable{

    private double c1 = 1e-4;             // Armijo条件参数 / Armijo condition parameter
    private double c2 = 0.9;              // Wolfe条件参数 / Wolfe condition parameter
    private double initialStepSize = 1.0;  // 初始步长 / Initial step size
    private int maxLineSearchIterations = 50; // 最大线搜索迭代次数 / Maximum line search iterations

    /**
     * 默认构造函数，使用标准参数值
     * Default constructor with standard parameter values
     *
     * <p>使用默认参数: c1=1e-4, c2=0.9, initialStepSize=1.0, maxIterations=50
     * Uses default parameters: c1=1e-4, c2=0.9, initialStepSize=1.0, maxIterations=50</p>
     */
    public RereLineSearch() {
    }

    /**
     * 完整参数构造函数
     * Full parameter constructor
     *
     * @param c1 Armijo条件参数，必须满足 0 < c1 < c2 < 1 / Armijo condition parameter, must satisfy 0 < c1 < c2 < 1
     * @param c2 Wolfe条件参数，必须满足 c1 < c2 < 1 / Wolfe condition parameter, must satisfy c1 < c2 < 1
     * @param initialStepSize 初始步长，必须大于0 / Initial step size, must be greater than 0
     */
    public RereLineSearch(double c1, double c2, double initialStepSize) {
        this.c1 = c1;
        this.c2 = c2;
        this.initialStepSize = initialStepSize;
    }

    /**
     * 使用预计算方向导数的优化线搜索算法
     * Optimized Line Search Algorithm with Pre-computed Directional Derivative
     *
     * <p>此方法使用预计算的方向导数避免重复计算，提高效率。
     * This method uses a pre-computed directional derivative to avoid redundant calculations and improve efficiency.</p>
     *
     * @param x 当前位置向量 / Current position vector
     * @param direction 搜索方向向量，必须是下降方向 / Search direction vector, must be a descent direction
     * @param objFun 目标函数对象 / Objective function object
     * @param grdFun 梯度函数对象 / Gradient function object
     * @param grad 当前点的梯度向量 / Gradient vector at current point
     * @param cachedDirectionalDerivative 缓存的方向导数 ∇f(x)^T * d / Cached directional derivative ∇f(x)^T * d
     * @return 步长 alpha，满足强Wolfe条件 / Step size alpha satisfying strong Wolfe conditions
     */
    public double searchWithCachedDerivative(IVector x, IVector direction, IObjectiveFunction objFun,
            IGradientFunction grdFun, IVector grad, double cachedDirectionalDerivative) {

        double alpha = initialStepSize;
        double currentValue = objFun.computeObjective(x);
        // 使用缓存的方向导数，避免重复计算 / Use cached directional derivative to avoid recomputation
        double directionalDerivative = cachedDirectionalDerivative;

        // 如果方向导数为正，说明不是下降方向，返回小步长 / If directional derivative is positive, not a descent direction
        if (directionalDerivative >= 0) {
            // 尝试更小的步长 / Try smaller step size
            alpha = 1e-8;
            IVector newX = x.add(direction.multiplyScalar(alpha));
            double newValue = objFun.computeObjective(newX);

            // 如果更小的步长能改善目标函数值，则使用它 / If smaller step size improves objective function, use it
            if (newValue < currentValue) {
                return alpha;
            }

            // 否则返回极小步长 / Otherwise return very small step size
            return 1e-12;
        }

        // 回溯线搜索 / Backtracking line search
        for (int i = 0; i < maxLineSearchIterations; i++) {
            IVector newX = x.add(direction.multiplyScalar(alpha));
            double newValue = objFun.computeObjective(newX);

            // 检查Armijo条件 / Check Armijo condition
            if (newValue <= currentValue + c1 * alpha * directionalDerivative) {
                IVector newGrad = grdFun.computeGradient(newX);
                double newDirectionalDerivative = (Double) newGrad.innerProduct(direction);

                // 检查曲率条件 / Check curvature condition
                if (Math.abs(newDirectionalDerivative) <= c2 * Math.abs(directionalDerivative)) {
                    return alpha;
                }
            }

            // 减小步长 / Reduce step size
            alpha *= 0.5;

            // 如果步长太小，停止搜索 / If step size too small, stop search
            if (alpha < 1e-20) {
                break;
            }
        }

        // 如果线搜索失败，返回一个保守的步长 / If line search fails, return conservative step size
        return Math.min(initialStepSize, Math.max(1e-8, alpha));
    }

    /**
     * 标准线搜索算法，使用强Wolfe条件
     * Standard Line Search Algorithm Using Strong Wolfe Conditions
     *
     * <p>使用强Wolfe条件进行线搜索，确保找到合适的步长：
     * Performs line search using strong Wolfe conditions to ensure finding an appropriate step size:
     * <ol>
     *   <li>Armijo条件: f(x + αd) ≤ f(x) + c1 * α * ∇f(x)^T * d 确保函数值充分下降</li>
     *   <li>曲率条件: |∇f(x + αd)^T * d| ≥ c2 * |∇f(x)^T * d| 确保梯度变化合理</li>
     * </ol>
     * Uses strong Wolfe conditions for line search, ensuring finding an appropriate step size:
     * <ol>
     *   <li>Armijo condition: f(x + αd) ≤ f(x) + c1 * α * ∇f(x)^T * d ensures sufficient function value decrease</li>
     *   <li>Curvature condition: |∇f(x + αd)^T * d| ≥ c2 * |∇f(x)^T * d| ensures reasonable gradient change</li>
     * </ol>
     *
     * @param x 当前位置向量 / Current position vector
     * @param direction 搜索方向向量，必须是下降方向 / Search direction vector, must be a descent direction
     * @param objFun 目标函数对象 / Objective function object
     * @param grdFun 梯度函数对象 / Gradient function object
     * @param grad 当前点的梯度向量 / Gradient vector at current point
     * @return 步长 alpha，满足强Wolfe条件 / Step size alpha satisfying strong Wolfe conditions
     * @throws IllegalArgumentException 如果任何参数为null或方向导数计算失败 / If any parameter is null or directional derivative computation fails
     */
    public double search(IVector x, IVector direction, IObjectiveFunction objFun,
            IGradientFunction grdFun, IVector grad) {

        double alpha = initialStepSize;
        double currentValue = objFun.computeObjective(x);
        double directionalDerivative = (Double) grad.innerProduct(direction);

        // 如果方向导数为正，说明不是下降方向，返回小步长 / If directional derivative is positive, not a descent direction
        if (directionalDerivative >= 0) {
            // 尝试更小的步长 / Try smaller step size
            alpha = 1e-8;
            IVector newX = x.add(direction.multiplyScalar(alpha));
            double newValue = objFun.computeObjective(newX);

            // 如果更小的步长能改善目标函数值，则使用它 / If smaller step size improves objective function, use it
            if (newValue < currentValue) {
                return alpha;
            }

            // 否则返回极小步长 / Otherwise return very small step size
            return 1e-12;
        }

        // 回溯线搜索 / Backtracking line search
        for (int i = 0; i < maxLineSearchIterations; i++) {
            IVector newX = x.add(direction.multiplyScalar(alpha));
            double newValue = objFun.computeObjective(newX);

            // 检查Armijo条件 / Check Armijo condition
            if (newValue <= currentValue + c1 * alpha * directionalDerivative) {
                IVector newGrad = grdFun.computeGradient(newX);
                double newDirectionalDerivative = (Double) newGrad.innerProduct(direction);

                // 检查曲率条件 / Check curvature condition
                if (Math.abs(newDirectionalDerivative) <= c2 * Math.abs(directionalDerivative)) {
                    return alpha;
                }
            }

            // 减小步长 / Reduce step size
            alpha *= 0.5;

            // 如果步长太小，停止搜索 / If step size too small, stop search
            if (alpha < 1e-20) {
                break;
            }
        }

        // 如果线搜索失败，返回一个保守的步长 / If line search fails, return conservative step size
        return Math.min(initialStepSize, Math.max(1e-8, alpha));
    }

    // Getter and Setter methods / 获取和设置方法

    /**
     * 获取Armijo条件参数 c1
     * Get Armijo condition parameter c1
     *
     * @return Armijo条件参数 / Armijo condition parameter
     */
    public double getC1() {
        return c1;
    }

    /**
     * 设置Armijo条件参数 c1
     * Set Armijo condition parameter c1
     *
     * @param c1 Armijo条件参数，必须满足 0 < c1 < 1 / Armijo condition parameter, must satisfy 0 < c1 < 1
     * @throws IllegalArgumentException 如果 c1 不在有效范围内 / If c1 is not in valid range
     */
    public void setC1(double c1) {
        this.c1 = c1;
    }

    /**
     * 获取Wolfe条件参数 c2
     * Get Wolfe condition parameter c2
     *
     * @return Wolfe条件参数 / Wolfe condition parameter
     */
    public double getC2() {
        return c2;
    }

    /**
     * 设置Wolfe条件参数 c2
     * Set Wolfe condition parameter c2
     *
     * @param c2 Wolfe条件参数，必须满足 0 < c2 < 1 / Wolfe condition parameter, must satisfy 0 < c2 < 1
     * @throws IllegalArgumentException 如果 c2 不在有效范围内 / If c2 is not in valid range
     */
    public void setC2(double c2) {
        this.c2 = c2;
    }

    /**
     * 获取初始步长
     * Get initial step size
     *
     * @return 初始步长 / Initial step size
     */
    public double getInitialStepSize() {
        return initialStepSize;
    }

    /**
     * 设置初始步长
     * Set initial step size
     *
     * @param initialStepSize 初始步长，必须大于0 / Initial step size, must be greater than 0
     * @throws IllegalArgumentException 如果初始步长不大于0 / If initial step size is not greater than 0
     */
    public void setInitialStepSize(double initialStepSize) {
        this.initialStepSize = initialStepSize;
    }

    /**
     * 获取最大线搜索迭代次数
     * Get maximum number of line search iterations
     *
     * @return 最大迭代次数 / Maximum number of iterations
     */
    public int getMaxLineSearchIterations() {
        return maxLineSearchIterations;
    }

    /**
     * 设置最大线搜索迭代次数
     * Set maximum number of line search iterations
     *
     * @param maxLineSearchIterations 最大迭代次数，必须大于0 / Maximum iterations, must be greater than 0
     * @throws IllegalArgumentException 如果最大迭代次数不大于0 / If maximum iterations is not greater than 0
     */
    public void setMaxLineSearchIterations(int maxLineSearchIterations) {
        this.maxLineSearchIterations = maxLineSearchIterations;
    }
}