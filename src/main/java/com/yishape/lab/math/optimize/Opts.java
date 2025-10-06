package com.yishape.lab.math.optimize;

import com.yishape.lab.math.optimize.linpg.IIntegerProg;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.newton.RereConjugateGradient;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.newton.RereOnlineAdam;
import com.yishape.lab.math.optimize.newton.RereOnlineSGD;

/**
 * Optimization Factory Class
 * 优化工厂类
 *
 * This class provides static factory methods for creating various optimization algorithm instances.
 * Provides access to different optimization techniques including unconstrained optimization,
 * online learning optimizers, and linear programming solvers.
 *
 * 该类提供静态工厂方法用于创建各种优化算法实例。
 * 提供对不同优化技术的访问，包括无约束优化、在线学习优化器和线性规划求解器。
 *
 * @author lteb2
 * @version 1.0
 * @since 2014
 */
public class Opts {

    /**
     * Create a Limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS) optimizer.
     * 创建有限内存BFGS优化器。
     *
     * LBFGS is a quasi-Newton method for unconstrained optimization that approximates
     * the Broyden–Fletcher–Goldfarb–Shanno algorithm using limited memory.
     *
     * LBFGS是一种用于无约束优化的拟牛顿法，使用有限内存逼近BFGS算法。
     *
     * @return LBFGS optimizer instance / LBFGS优化器实例
     */
    public static IOptimizer lbfgs() {
        return new RereLBFGS();
    }
    
    /**
     * Create a Conjugate Gradient optimizer.
     * 创建共轭梯度优化器。
     *
     * Conjugate Gradient method is an algorithm for the numerical solution of
     * particular systems of linear equations, and is often used for unconstrained optimization.
     *
     * 共轭梯度法是一种用于数值求解特定线性方程组的算法，常用于无约束优化。
     *
     * @return Conjugate Gradient optimizer instance / 共轭梯度优化器实例
     */
    public static IOptimizer conjugateGradient() {
        return new RereConjugateGradient();
    }
    
    
    
    /**
     * Create an Online Adam optimizer.
     * 创建在线Adam优化器。
     *
     * Adam is an adaptive learning rate optimization algorithm that's popular in
     * deep learning for training neural networks.
     *
     * Adam是一种自适应学习率优化算法，在深度学习中常用于训练神经网络。
     *
     * @return Online Adam optimizer instance / 在线Adam优化器实例
     */
    public static IOnlineOptimizer onlineAdam() {
        return new RereOnlineAdam();
    }
    
    /**
     * Create an Online Stochastic Gradient Descent (SGD) optimizer.
     * 创建在线随机梯度下降优化器。
     *
     * SGD is an iterative method for optimizing an objective function with suitable
     * smoothness properties, commonly used in machine learning.
     *
     * SGD是一种迭代方法，用于优化具有适当光滑性质的目标函数，常用于机器学习。
     *
     * @return Online SGD optimizer instance / 在线SGD优化器实例
     */
    public static IOnlineOptimizer onlineSGD() {
        return new RereOnlineSGD();
    }

    /**
     * Create an Integer Linear Programming solver.
     * 创建整数线性规划求解器。
     *
     * Integer programming involves optimizing a linear function subject to linear
     * constraints where some or all variables are required to be integers.
     *
     * 整数规划涉及在线性约束条件下优化线性函数，其中部分或全部变量必须为整数。
     *
     * @return Integer linear programming solver instance / 整数线性规划求解器实例
     */
    public static IIntegerProg intLinProgSolver() {
        return new RereIntegerProg();
    }

    /**
     * Create a Simplex Linear Programming solver.
     * 创建单纯形线性规划求解器。
     *
     * The simplex algorithm is a popular method for numerical solution of the linear
     * programming problem, solving optimization problems with linear constraints.
     *
     * 单纯形算法是一种用于数值求解线性规划问题的流行方法，求解具有线性约束的优化问题。
     *
     * @return Simplex linear programming solver instance / 单纯形线性规划求解器实例
     */
    public static ILinProgSolver simplexLinProgSolver() {
        return new RereSimplexLinProgSolver();
    }
    
    /**
     * Create an Interior Point Linear Programming solver.
     * 创建内点线性规划求解器。
     *
     * Interior point methods are a class of algorithms that solve linear and nonlinear
     * convex optimization problems by traversing the interior of the feasible region.
     *
     * 内点法是一类通过遍历可行域内部来求解线性和非线性凸优化问题的算法。
     *
     * @return Interior point linear programming solver instance / 内点线性规划求解器实例
     */
    public static ILinProgSolver interPointLinProgSolver() {
        return new RereSimplexLinProgSolver();
    }

}
