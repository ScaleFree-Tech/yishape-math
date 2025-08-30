package com.reremouse.lab.math.optimize;

import com.reremouse.lab.math.IVector;

/**
 * 最优化中的线搜索器
 *
 * @author lteb2
 */
public class RereLineSearch {

    private float c1 = 1e-4f;             // Armijo条件参数 / Armijo condition parameter
    private float c2 = 0.9f;              // Wolfe条件参数 / Wolfe condition parameter
    private float initialStepSize = 1.0f;  // 初始步长 / Initial step size

    public RereLineSearch() {
    }

    /**
     * 
     * @param c1 Armijo条件参数 / Armijo condition parameter
     * @param c2 Wolfe条件参数 / Wolfe condition parameter
     * @param initialStepSize 初始步长 / Initial step size
     */
    public RereLineSearch(float c1, float c2, float initialStepSize) {
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
    public float search(IVector x, IVector direction, IObjectiveFunction objFun,
            IGradientFunction grdFun, IVector grad) {

        float alpha = initialStepSize;
        float currentValue = objFun.computeObjective(x);
        float directionalDerivative = grad.innerProduct(direction);

        // 如果方向导数为正，说明不是下降方向，返回小步长 / If directional derivative is positive, not a descent direction
        if (directionalDerivative >= 0) {
            return 1e-8f;
        }

        // 回溯线搜索 / Backtracking line search
        int maxLineSearchIterations = 50;
        for (int i = 0; i < maxLineSearchIterations; i++) {
            IVector newX = x.add(direction.multiplyScalar(alpha));
            float newValue = objFun.computeObjective(newX);

            // 检查Armijo条件 / Check Armijo condition
            if (newValue <= currentValue + c1 * alpha * directionalDerivative) {
                IVector newGrad = grdFun.computeGradient(newX);
                float newDirectionalDerivative = newGrad.innerProduct(direction);

                // 检查曲率条件 / Check curvature condition
                if (newDirectionalDerivative >= c2 * directionalDerivative) {
                    return alpha;
                }
            }

            // 减小步长 / Reduce step size
            alpha *= 0.5f;

            // 如果步长太小，停止搜索 / If step size too small, stop search
            if (alpha < 1e-10f) {
                break;
            }
        }

        return alpha;
    }

}
