package com.yishape.lab.math.optimize.newton;

import com.yishape.lab.math.compute.hpc.HpcConfig;
import com.yishape.lab.math.compute.hpc.HpcOptimizers;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.RereDoubleVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.util.YishapeLogger;

/**
 * OWL-QN 优化器（HPC 加速版）。
 * <p>
 * OWL-QN（Orthant-Wise Limited-memory Quasi-Newton）是 L-BFGS 的 L1 正则化变体，
 * 适用于产生稀疏解的问题（如 Lasso 回归）。
 * 优先使用 HPC 原生路径（gosh-lbfgs / Rust），HPC 不可用时自动回退到纯 Java
 * {@link RereLBFGS}。
 * </p>
 */
public class RustOWLQN implements IOptimizer {

    private static final YishapeLogger LOG = YishapeLogger.getLogger(RustOWLQN.class);

    private int m;
    private double tolerance;
    private int maxIterations;
    private double orthantwiseC;

    /** 默认参数：m=10, tolerance=1e-6, maxIterations=1000, orthantwiseC=1.0 */
    public RustOWLQN() {
        this(10, 1e-6, 1000, 1.0);
    }

    /**
     * @param m             L-BFGS 历史校正数（通常 3~20）
     * @param tolerance     收敛容差
     * @param maxIterations 最大迭代次数
     * @param orthantwiseC  L1 正则化权重（≥0；0 等价于普通 L-BFGS）
     */
    public RustOWLQN(int m, double tolerance, int maxIterations, double orthantwiseC) {
        this.m = m;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
        this.orthantwiseC = orthantwiseC;
    }

    /**
     * 设置 L1 正则化权重。
     *
     * @return this
     */
    public RustOWLQN withOrthantwiseC(double orthantwiseC) {
        this.orthantwiseC = orthantwiseC;
        return this;
    }

    /**
     * 设置 L-BFGS 历史校正数。
     *
     * @return this
     */
    public RustOWLQN withM(int m) {
        this.m = m;
        return this;
    }

    /**
     * 设置收敛容差。
     *
     * @return this
     */
    public RustOWLQN withTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    /**
     * 设置最大迭代次数。
     *
     * @return this
     */
    public RustOWLQN withMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
        return this;
    }

    @Override
    public OptResult optimize(IVector initX, IObjectiveFunction objFun, IGradientFunction grdFun) {
        if (initX == null) {
            throw new IllegalArgumentException("初始点不能为空 / Initial point cannot be null");
        }
        if (objFun == null) {
            throw new IllegalArgumentException("目标函数不能为空 / Objective function cannot be null");
        }
        if (grdFun == null) {
            throw new IllegalArgumentException("梯度函数不能为空 / Gradient function cannot be null");
        }

        // HPC 路径
        if (HpcConfig.allowAttempts() && HpcOptimizers.isExtensionPresent()) {
            double[] xArr = initX.toDoubleArray();

            final int[] iterCount = {0};
            IGradientFunction countingGrd = v -> {
                iterCount[0]++;
                return grdFun.computeGradient(v);
            };

            long start = System.currentTimeMillis();
            HpcOptimizers.RLbfgsResult result = HpcOptimizers.tryOWLQN(
                    xArr, m, tolerance, maxIterations, orthantwiseC, objFun, countingGrd);
            long elapsed = System.currentTimeMillis() - start;

            if (result != null && result.ok()) {
                double[] xSol = result.x();
                RereDoubleVector optPoint = new RereDoubleVector(xSol);
                double initialValue = objFun.computeObjective(initX);
                LOG.info("RustOWLQN: HPC gosh-lbfgs OWL-QN converged in {} iterations, {}ms, f(x)={}, c={}",
                        iterCount[0], elapsed, result.fx(), orthantwiseC);
                return new OptResult.Builder(result.fx(), optPoint)
                        .initialPoint(initX)
                        .initialValue(initialValue)
                        .converged(true)
                        .convergenceReason("HPC OWL-QN converged with gosh-lbfgs")
                        .iterations(iterCount[0])
                        .gradientEvaluations(iterCount[0])
                        .maxIterations(maxIterations)
                        .tolerance(tolerance)
                        .executionTimeMs(elapsed)
                        .build();
            }
            if (result == null) {
                LOG.warn("RustOWLQN: HPC tryOWLQN returned null (native unavailable), falling back to RereLBFGS");
            } else {
                LOG.warn("RustOWLQN: HPC gosh-lbfgs OWL-QN returned status={}, falling back to RereLBFGS",
                        result.status());
            }
        } else {
            LOG.info("RustOWLQN: HPC not available (allowAttempts={}, extensionPresent={}), using RereLBFGS fallback",
                    HpcConfig.allowAttempts(), HpcOptimizers.isExtensionPresent());
        }

        // 纯 Java 回退：RereLBFGS 作为纯 Java OWL-QN 后备（标准 L-BFGS 近似，不支持 L1 正则）
        LOG.info("RustOWLQN: using RereLBFGS fallback (m={}, tol={}, maxIter={})",
                m, tolerance, maxIterations);
        return new RereLBFGS(m, tolerance, maxIterations).optimize(initX, objFun, grdFun);
    }
}
