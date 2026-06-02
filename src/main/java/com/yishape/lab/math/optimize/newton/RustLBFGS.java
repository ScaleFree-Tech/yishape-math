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
 * L-BFGS 优化器（HPC 加速版）。
 * <p>
 * 优先使用 HPC 原生路径（gosh-lbfgs / Rust），HPC 不可用时自动回退到纯 Java
 * {@link RereLBFGS}。API 风格与 {@link RereLBFGS} 一致，参数含义相同。
 * </p>
 */
public class RustLBFGS implements IOptimizer {

    private static final YishapeLogger LOG = YishapeLogger.getLogger(RustLBFGS.class);

    private int m;
    private double tolerance;
    private int maxIterations;

    /** 默认参数：m=10, tolerance=1e-6, maxIterations=1000 */
    public RustLBFGS() {
        this(10, 1e-6, 1000);
    }
    
    /**
     * @param tolerance     收敛容差
     * @param maxIterations 最大迭代次数
     */
    public RustLBFGS(double tolerance, int maxIterations) {
        this.m = 10;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
    }

    /**
     * @param m             L-BFGS 历史校正数（通常 3~20）
     * @param tolerance     收敛容差
     * @param maxIterations 最大迭代次数
     */
    public RustLBFGS(int m, double tolerance, int maxIterations) {
        this.m = m;
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
    }

    /**
     * 设置 L-BFGS 历史校正数。
     * @return this
     */
    public RustLBFGS withM(int m) {
        this.m = m;
        return this;
    }

    /**
     * 设置收敛容差。
     * @return this
     */
    public RustLBFGS withTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    /**
     * 设置最大迭代次数。
     * @return this
     */
    public RustLBFGS withMaxIterations(int maxIterations) {
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
            HpcOptimizers.RLbfgsResult result = HpcOptimizers.tryLBFGS(
                    xArr, m, tolerance, maxIterations, objFun, countingGrd);
            long elapsed = System.currentTimeMillis() - start;

            if (result != null && result.ok()) {
                double[] xSol = result.x();
                RereDoubleVector optPoint = new RereDoubleVector(xSol);
                double initialValue = objFun.computeObjective(initX);
                LOG.info("RustLBFGS: HPC gosh-lbfgs converged in {} iterations, {}ms, f(x)={}",
                        iterCount[0], elapsed, result.fx());
                return new OptResult.Builder(result.fx(), optPoint)
                        .initialPoint(initX)
                        .initialValue(initialValue)
                        .converged(true)
                        .convergenceReason("HPC L-BFGS converged with gosh-lbfgs")
                        .iterations(iterCount[0])
                        .gradientEvaluations(iterCount[0])
                        .maxIterations(maxIterations)
                        .tolerance(tolerance)
                        .executionTimeMs(elapsed)
                        .build();
            }
            if (result == null) {
                LOG.warn("RustLBFGS: HPC tryLBFGS returned null (native unavailable), falling back to RereLBFGS");
            } else {
                LOG.warn("RustLBFGS: HPC gosh-lbfgs returned status={}, falling back to RereLBFGS",
                        result.status());
            }
        } else {
            LOG.info("RustLBFGS: HPC not available (allowAttempts={}, extensionPresent={}), using RereLBFGS",
                    HpcConfig.allowAttempts(), HpcOptimizers.isExtensionPresent());
        }

        // 纯 Java 回退
        LOG.info("RustLBFGS: using RereLBFGS (m={}, tol={}, maxIter={})",
                m, tolerance, maxIterations);
        return new RereLBFGS(m, tolerance, maxIterations).optimize(initX, objFun, grdFun);
    }
}
