package com.yishape.lab.math.compute.hpc;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.RereDoubleVector;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

/**
 * 优化器 HPC 桥接：L-BFGS / OWL-QN 经反射调用 {@code yishape-math-hpc} 的 {@code YishapeHpc}。
 * classpath 未加入 HPC 扩展 JAR 时安全降级返回 {@code null}。
 */
public final class HpcOptimizers {

    private static final String HPC_CLASS = "com.yishape.lab.math.hpc.YishapeHpc";

    private static final Class<?> HPC;
    private static final Method M_IS_NATIVE;
    private static final Method M_LBFGS_MINIMIZE;
    private static final Method M_OWLQN_MINIMIZE;
    private static final Class<?> EVAL_FN_INTERFACE;

    static {
        Class<?> c = null;
        Method mNative = null;
        Method mLbfgs = null;
        Method mOwlqn = null;
        Class<?> evalFn = null;
        try {
            c = Class.forName(HPC_CLASS);
            mNative = c.getMethod("isNativeRuntimeAvailable");
            // 查找 LbfgsEvaluateFunction 内部接口
            for (Class<?> inner : c.getClasses()) {
                if ("LbfgsEvaluateFunction".equals(inner.getSimpleName())) {
                    evalFn = inner;
                    break;
                }
            }
            if (evalFn != null) {
                mLbfgs = c.getMethod("lbfgsMinimize",
                        double[].class, int.class, double.class, int.class,
                        evalFn);
                mOwlqn = c.getMethod("owlqnMinimize",
                        double[].class, int.class, double.class, int.class,
                        double.class, evalFn);
            }
        } catch (ReflectiveOperationException | LinkageError e) {
            c = null;
        }
        HPC = c;
        M_IS_NATIVE = mNative;
        M_LBFGS_MINIMIZE = mLbfgs;
        M_OWLQN_MINIMIZE = mOwlqn;
        EVAL_FN_INTERFACE = evalFn;
    }

    private HpcOptimizers() {
    }

    /** HPC 扩展 JAR 是否存在且已成功解析。 */
    public static boolean isExtensionPresent() {
        return HPC != null && M_LBFGS_MINIMIZE != null;
    }

    /** 原生运行时是否可用。 */
    public static boolean isNativeRuntimeAvailable() {
        if (M_IS_NATIVE == null) return false;
        try {
            Object v = M_IS_NATIVE.invoke(null);
            return v instanceof Boolean && (Boolean) v;
        } catch (ReflectiveOperationException | LinkageError e) {
            return false;
        }
    }

    /**
     * L-BFGS 最小化（HPC 路径）。
     *
     * @param x             in/out：初始点 → 解
     * @param m             历史校正数
     * @param epsilon       收敛容差
     * @param maxIterations 最大迭代次数（0 = 无限制）
     * @param objFun        目标函数
     * @param grdFun        梯度函数
     * @return 成功时 {@link RLbfgsResult#ok()} 为 true；HPC 不可用或失败时返回 null
     */
    public static RLbfgsResult tryLBFGS(
            double[] x, int m, double epsilon, int maxIterations,
            IObjectiveFunction objFun, IGradientFunction grdFun) {
        if (M_LBFGS_MINIMIZE == null || EVAL_FN_INTERFACE == null) return null;
        InvocationHandler handler = (proxy, method, args) -> {
            if ("evaluate".equals(method.getName()) && args != null && args.length == 2) {
                double[] xArr = (double[]) args[0];
                double[] gArr = (double[]) args[1];
                RereDoubleVector xVec = new RereDoubleVector(xArr);
                double fx = objFun.computeObjective(xVec);
                IVector<Double> gVec = grdFun.computeGradient(xVec);
                for (int i = 0; i < gArr.length; i++) {
                    gArr[i] = gVec.get(i);
                }
                return fx;
            }
            return 0.0;
        };
        try {
            Object evalProxy = Proxy.newProxyInstance(
                    EVAL_FN_INTERFACE.getClassLoader(),
                    new Class<?>[]{EVAL_FN_INTERFACE},
                    handler);
            Object res = M_LBFGS_MINIMIZE.invoke(null, x, m, epsilon, maxIterations, evalProxy);
            if (res == null) return null;
            return toLbfgsResult(res);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            return null;
        }
    }

    /**
     * OWL-QN 最小化（HPC 路径）。
     *
     * @param x             in/out：初始点 → 解
     * @param m             历史校正数
     * @param epsilon       收敛容差
     * @param maxIterations 最大迭代次数
     * @param orthantwiseC  L1 正则化权重
     * @param objFun        目标函数
     * @param grdFun        梯度函数
     * @return 成功时 {@link RLbfgsResult#ok()} 为 true；HPC 不可用或失败时返回 null
     */
    public static RLbfgsResult tryOWLQN(
            double[] x, int m, double epsilon, int maxIterations,
            double orthantwiseC,
            IObjectiveFunction objFun, IGradientFunction grdFun) {
        if (M_OWLQN_MINIMIZE == null || EVAL_FN_INTERFACE == null) return null;
        InvocationHandler handler = (proxy, method, args) -> {
            if ("evaluate".equals(method.getName()) && args != null && args.length == 2) {
                double[] xArr = (double[]) args[0];
                double[] gArr = (double[]) args[1];
                RereDoubleVector xVec = new RereDoubleVector(xArr);
                double fx = objFun.computeObjective(xVec);
                IVector<Double> gVec = grdFun.computeGradient(xVec);
                for (int i = 0; i < gArr.length; i++) {
                    gArr[i] = gVec.get(i);
                }
                return fx;
            }
            return 0.0;
        };
        try {
            Object evalProxy = Proxy.newProxyInstance(
                    EVAL_FN_INTERFACE.getClassLoader(),
                    new Class<?>[]{EVAL_FN_INTERFACE},
                    handler);
            Object res = M_OWLQN_MINIMIZE.invoke(null, x, m, epsilon, maxIterations, orthantwiseC, evalProxy);
            if (res == null) return null;
            return toLbfgsResult(res);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            return null;
        }
    }

    private static RLbfgsResult toLbfgsResult(Object res) throws ReflectiveOperationException {
        int st = (Integer) res.getClass().getMethod("status").invoke(res);
        double[] x = (double[]) res.getClass().getMethod("x").invoke(res);
        double fx = (Double) res.getClass().getMethod("fx").invoke(res);
        return new RLbfgsResult(st, x, fx);
    }

    /** L-BFGS / OWL-QN 结果。 */
    public static final class RLbfgsResult {
        private final int status;
        private final double[] x;
        private final double fx;

        public RLbfgsResult(int status, double[] x, double fx) {
            this.status = status;
            this.x = x;
            this.fx = fx;
        }

        public int status() {
            return status;
        }

        public double[] x() {
            return x;
        }

        public double fx() {
            return fx;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }
}
