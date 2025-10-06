package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IVector;

/**
 * 最优化中的线搜索器
 *
 * @author lteb2
 */
public class RereLineSearch {

    private double c1 = 1e-4;             // Armijo条件参数 / Armijo condition parameter
    private double c2 = 0.9;              // Wolfe条件参数 / Wolfe condition parameter
    private double initialStepSize = 1.0;  // 初始步长 / Initial step size
    private int maxLineSearchIterations = 50; // 最大线搜索迭代次数 / Maximum line search iterations

    public RereLineSearch() {
    }

    /**
     * 
     * @param c1 Armijo条件参数 / Armijo condition parameter
     * @param c2 Wolfe条件参数 / Wolfe condition parameter
     * @param initialStepSize 初始步长 / Initial step size
     */
    public RereLineSearch(double c1, double c2, double initialStepSize) {
        this.c1 = c1;
        this.c2 = c2;
        this.initialStepSize = initialStepSize;
    }

    /**
     * 线搜索算法
     * <p>
     * 使用强Wolfe条件进行线搜索，确保找到合适的步长： 1. Armijo条件：f(x + αd) ≤ f(x) + c1 * α *
     * ∇f(x)^T * d 2. 曲率条件：∇f(x + αd)^T * d ≥ c2 * ∇f(x)^T * d
     * </p>
     *
     * @param x 当前位置 / Current position
     * @param direction 搜索方向 / Search direction
     * @param objFun 目标函数 / Objective function
     * @param grdFun 梯度函数 / Gradient function
     * @param grad 当前梯度 / Current gradient
     * @return 步长 / Step size
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

    // Getter and Setter methods
    public double getC1() {
        return c1;
    }

    public void setC1(double c1) {
        this.c1 = c1;
    }

    public double getC2() {
        return c2;
    }

    public void setC2(double c2) {
        this.c2 = c2;
    }

    public double getInitialStepSize() {
        return initialStepSize;
    }

    public void setInitialStepSize(double initialStepSize) {
        this.initialStepSize = initialStepSize;
    }

    public int getMaxLineSearchIterations() {
        return maxLineSearchIterations;
    }

    public void setMaxLineSearchIterations(int maxLineSearchIterations) {
        this.maxLineSearchIterations = maxLineSearchIterations;
    }
}