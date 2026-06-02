package com.yishape.lab.math.optimize;

import com.yishape.lab.math.optimize.linpg.IIntegerProg;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.InteriorPointLinProgSolver;
import com.yishape.lab.math.optimize.linpg.LpSolverType;
import com.yishape.lab.math.optimize.linpg.highs.HighsIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.newton.LBFGSType;
import static com.yishape.lab.math.optimize.newton.LBFGSType.Java;
import static com.yishape.lab.math.optimize.newton.LBFGSType.Rust;
import com.yishape.lab.math.optimize.newton.RereConjugateGradient;
import com.yishape.lab.math.optimize.newton.RereDFP;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.newton.RereOnlineAdam;
import com.yishape.lab.math.optimize.newton.RereOnlineAdamW;
import com.yishape.lab.math.optimize.newton.RereOnlineAdadelta;
import com.yishape.lab.math.optimize.newton.RereOnlineAdagrad;
import com.yishape.lab.math.optimize.newton.RereOnlineLAMB;
import com.yishape.lab.math.optimize.newton.RereOnlineRMSprop;
import com.yishape.lab.math.optimize.newton.RereOnlineSGD;
import com.yishape.lab.math.optimize.newton.RereSteepestDescent;
import com.yishape.lab.math.optimize.newton.RustLBFGS;
import com.yishape.lab.math.optimize.newton.RustOWLQN;
import java.io.Serializable;

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
public class Opts implements Serializable{

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
     * 
     * @param tolerance
     * @param maxSteps
     * @return 
     */
    public static IOptimizer lbfgs(double tolerance, int maxSteps) {
        return new RereLBFGS(10,tolerance,maxSteps);
    }

    /**
     * Create a Rust/HPC-accelerated L-BFGS optimizer.
     * 创建 Rust/HPC 加速的 L-BFGS 优化器。
     *
     * <p>This optimizer tries the native HPC path (gosh-lbfgs via Rust/FFM) first;
     * falls back to {@link RereLBFGS} when HPC is unavailable.</p>
     *
     * <p>该优化器优先尝试原生 HPC 路径（gosh-lbfgs / Rust / FFM），
     * HPC 不可用时自动回退到 {@link RereLBFGS}。</p>
     *
     * @param type Java | Rust
     * @return Rust/HPC-accelerated L-BFGS optimizer instance
     */
    public static IOptimizer lbfgs(LBFGSType type) {
        if(null == type)
            return lbfgs();
        else return switch (type) {
            case Rust -> new RustLBFGS();
            case Java -> new RereLBFGS();
            default -> lbfgs();
        };
    }
    
    /**
     * 
     * @param tolerance
     * @param maxSteps
     * @param type
     * @return 
     */
    public static IOptimizer lbfgs(double tolerance, int maxSteps,LBFGSType type) {
        if(null == type)
            return lbfgs();
        else return switch (type) {
            case Rust -> new RustLBFGS(tolerance,maxSteps);
            case Java -> new RereLBFGS(tolerance,maxSteps);
            default -> lbfgs();
        };
    }

    /**
     * Create a Rust/HPC-accelerated OWL-QN optimizer.
     * 创建 Rust/HPC 加速的 OWL-QN 优化器。
     *
     * <p>OWL-QN (Orthant-Wise Limited-memory Quasi-Newton) is the L1-regularized
     * variant of L-BFGS, suitable for problems requiring sparse solutions
     * (e.g., Lasso regression). Falls back to {@link RereLBFGS} when HPC is
     * unavailable.</p>
     *
     * <p>OWL-QN 是 L-BFGS 的 L1 正则化变体，适用于需要稀疏解的问题（如 Lasso 回归）。
     * HPC 不可用时自动回退到 {@link RereLBFGS}。</p>
     *
     * @return Rust/HPC-accelerated OWL-QN optimizer instance
     */
    public static IOptimizer owlqn() {
        return new RustOWLQN();
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
     * 经典拟牛顿法DFP，但难以应用于大规模问题
     * @return 
     */
    public static IOptimizer dfp() {
        return new RereDFP();
    }
    
    /**
     * 最速下降法，经典一阶优化法，但多数情况没有二阶（拟）牛顿法速度快，但部分场景仍有优势
     * @return 
     */
    public static IOptimizer steepestDescent() {
        return new RereSteepestDescent();
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
     * Create an Online Adam optimizer.
     * 创建在线Adam优化器。
     *
     * @param learningRate 学习率 / Learning rate
     * @param beta1 一阶矩衰减率，必须在[0, 1)范围内 / First moment decay rate, must be in [0, 1)
     * @param beta2 二阶矩衰减率，必须在[0, 1)范围内 / Second moment decay rate, must be in [0, 1)
     * @return 
     * @throws IllegalArgumentException 如果任何参数无效 / If any parameter is invalid
     */
    public static IOnlineOptimizer onlineAdam(double learningRate, double beta1, double beta2) {
        return new RereOnlineAdam(learningRate, beta1, beta2);
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
    /**
     * Create an Online AdamW optimizer (Adam + Decoupled Weight Decay).
     * AdamW 是现代深度学习训练的事实标准优化器，在 Adam 基础上将权重衰减与梯度更新解耦。
     */
    public static IOnlineOptimizer onlineAdamW() {
        return new RereOnlineAdamW();
    }

    public static IOnlineOptimizer onlineAdamW(double learningRate, double beta1, double beta2, double weightDecay) {
        return new RereOnlineAdamW(learningRate, beta1, beta2, weightDecay);
    }

    /**
     * Create an Online RMSprop optimizer.
     * RMSprop 使用梯度平方的指数移动平均调整学习率，适合非平稳目标。
     */
    public static IOnlineOptimizer onlineRMSprop() {
        return new RereOnlineRMSprop();
    }

    public static IOnlineOptimizer onlineRMSprop(double learningRate, double alpha) {
        return new RereOnlineRMSprop(learningRate, alpha);
    }

    public static IOnlineOptimizer onlineSGD() {
        return new RereOnlineSGD();
    }

    /**
     * Create an Online Adagrad optimizer.
     * Adagrad 使用累积平方梯度的自适应学习率，适合稀疏特征。
     */
    public static IOnlineOptimizer onlineAdagrad() {
        return new RereOnlineAdagrad();
    }

    public static IOnlineOptimizer onlineAdagrad(double learningRate) {
        return new RereOnlineAdagrad(learningRate);
    }

    /**
     * Create an Online Adadelta optimizer.
     * Adadelta 不需要初始学习率，使用参数更新的运行平均自适应学习率。
     */
    public static IOnlineOptimizer onlineAdadelta() {
        return new RereOnlineAdadelta();
    }

    public static IOnlineOptimizer onlineAdadelta(double rho) {
        return new RereOnlineAdadelta(rho);
    }

    /**
     * Create an Online LAMB optimizer.
     * LAMB (Layer-wise Adaptive Moments) 通过逐层信任比率缩放支持大批量训练。
     */
    public static IOnlineOptimizer onlineLAMB() {
        return new RereOnlineLAMB();
    }

    public static IOnlineOptimizer onlineLAMB(double learningRate, double beta1, double beta2, double weightDecay) {
        return new RereOnlineLAMB(learningRate, beta1, beta2, weightDecay);
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
        //默认使用RUST HIGHS版本，出现问题时回退JAVA版本
        return new HighsIntegerProg();
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
     * 创建线性规划求解器，使用默认类型（Highs）
     * @return 
     */
    public static ILinProgSolver linProgSolver() {
        return ILinProgSolver.of();
    }
    
    /**
     * 创建线性规划求解器，按指定类型
     * @param solverType 求解器类型
     * @return 
     */
    public static ILinProgSolver linProgSolver(LpSolverType solverType) {
        return ILinProgSolver.of(solverType);
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
        return new InteriorPointLinProgSolver();
    }

    /**
     * 多目标/多准则线性规划包装类
     */
    public static MclpWrapper mclp = new MclpWrapper();

    /**
     * 多目标/多准则二次规划包装类
     */
    public static McqpWrapper mcqp = new McqpWrapper();
}
