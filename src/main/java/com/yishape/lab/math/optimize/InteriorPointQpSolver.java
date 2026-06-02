package com.yishape.lab.math.optimize;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.Objects;

import com.yishape.lab.util.YishapeLogger;

/**
 * 增强型内点法二次规划求解器
 * Enhanced Interior Point Method for Quadratic Programming
 *
 * <p>使用原始-对偶内点法（Primal-Dual Interior Point Method）求解凸二次规划问题。
 *
 * <p>标准形式：
 * minimize 1/2 * x^T * Q * x + c^T * x
 * subject to {@code A_ub * x <= b_ub}
 *            A_eq * x = b_eq
 *            x >= 0</p>
 *
 * @author lteb2
 */
public class InteriorPointQpSolver implements IQpSolver {

    private static final YishapeLogger log = YishapeLogger.getLogger(InteriorPointQpSolver.class);

    private double tolerance = 1e-8;
    private int maxIterations = 150;
    private double mu0 = 0.1;
    private boolean verbose = false;

    public InteriorPointQpSolver() {
    }

    public InteriorPointQpSolver(double tolerance, int maxIterations) {
        this.tolerance = tolerance;
        this.maxIterations = maxIterations;
    }

    public InteriorPointQpSolver withVerbose(boolean verbose) {
        this.verbose = verbose;
        return this;
    }

    @Override
    public OptResult solve(IMatrix Q, IVector c, IMatrix A_ub, IVector b_ub,
                          IMatrix A_eq, IVector b_eq, IVector initX) {
        long startTime = System.currentTimeMillis();

        Objects.requireNonNull(Q, "Q矩阵不能为null");
        Objects.requireNonNull(c, "c向量不能为null");

        int n = c.length();
        int numIneq = (A_ub != null) ? A_ub.rows() : 0;
        int numEq = (A_eq != null) ? A_eq.rows() : 0;

        // 如果无约束，使用共轭梯度法直接求解
        if (numIneq == 0 && numEq == 0) {
            return solveUnconstrained(Q, c, initX, startTime);
        }

        // 扩展变量：原始变量 x + 松弛变量 s (对于不等式约束)
        int totalVars = n + numIneq;

        // 构建扩展问题
        // minimize 1/2 * [x;s]^T * Q_ext * [x;s] + c_ext^T * [x;s]
        // s >= 0

        IMatrix Q_ext = buildExtendedQ(Q, n, numIneq);
        IVector c_ext = buildExtendedC(c, n, numIneq);

        // 构建扩展的不等式约束：A_ub * x <= b_ub  =>  A_ub * x + s = b_ub, s >= 0
        IMatrix A_ub_ext = buildExtendedAub(A_ub, n, numIneq);
        IVector b_ub_ext = b_ub != null ? b_ub.copy() : null;

        // 等式约束扩展（如果存在）
        IMatrix A_eq_ext = buildExtendedAeq(A_eq, n, numIneq);
        IVector b_eq_ext = b_eq != null ? b_eq.copy() : null;

        // 初始化
        IVector xExt = initX != null ? initX.copy() : Linalg.ones(totalVars);
        ensurePositive(xExt);

        // 对偶变量初始化
        IVector y = numEq > 0 ? Linalg.zeros(numEq) : null;
        IVector z = Linalg.ones(totalVars);
        ensurePositive(z);

        double mu = mu0;
        boolean converged = false;
        String convergenceReason = "";
        int iterations = 0;

        for (int iter = 0; iter < maxIterations; iter++) {
            iterations++;

            // 计算残差
            IVector r_p = computePrimalResidual(A_ub_ext, b_ub_ext, A_eq_ext, b_eq_ext, xExt);
            IVector r_d = computeDualResidual(Q_ext, c_ext, A_eq_ext, y, z, xExt);
            double r_cs = computeComplementarityGap(xExt, z, mu);

            double primalError = norm(r_p);
            double dualError = norm(r_d);

            if (verbose) {
                double obj = computeObjective(Q, c, extractFirstN(xExt, n));
                log.info(String.format("Iter %3d: |r_p|=%.2e |r_d|=%.2e gap=%.2e obj=%.6f",
                        iter, primalError, dualError, r_cs, obj));
            }

            // 检查收敛
            if (primalError < tolerance && dualError < tolerance && r_cs < tolerance) {
                converged = true;
                convergenceReason = "原始-对偶残差均收敛";
                break;
            }

            // 求解牛顿系统
            NewtonResult nr = solveNewtonSystem(Q_ext, A_ub_ext, A_eq_ext, xExt, z, c_ext, y, mu);

            if (!nr.feasible) {
                break;
            }

            // 计算步长
            double alphaX = computeStepSize(xExt, nr.dx, 1.0);
            double alphaZ = computeStepSize(z, nr.dz, 1.0);

            double alpha = Math.min(alphaX, alphaZ) * 0.95; // 安全因子

            // 更新变量
            xExt = xExt.add(nr.dx.multiplyByScalar(alpha));
            z = z.add(nr.dz.multiplyByScalar(alpha));

            if (y != null && nr.dy != null) {
                y = y.add(nr.dy.multiplyByScalar(alpha));
            }

            // 自适应障碍参数
            if (adaptiveMu) {
                mu = Math.max(1e-9, mu * 0.5);
            }
        }

        if (!converged) {
            convergenceReason = String.format("达到最大迭代次数%d", iterations);
        }

        // 提取原始变量
        IVector x = extractFirstN(xExt, n);
        double optimalValue = computeObjective(Q, c, x);
        double constraintViolation = computeConstraintViolation(A_ub, b_ub, A_eq, b_eq, x);

        return new OptResult.Builder(optimalValue, x)
                .converged(converged)
                .convergenceReason(convergenceReason)
                .iterations(iterations)
                .executionTimeMs(System.currentTimeMillis() - startTime)
                .constraintViolation(constraintViolation)
                .build();
    }

    /**
     * 求解无约束凸QP
     * 使用共轭梯度法求解 Q*x = -c
     */
    private OptResult solveUnconstrained(IMatrix Q, IVector c, IVector initX, long startTime) {
        // 求解无约束凸QP: min 1/2*x'*Q*x + c'*x
        // 最优解满足: Q*x = -c
        // 使用Q.solve()可自动路由到HPC优化路径
        IVector negC = c.multiplyByScalar(-1.0);
        IVector xOpt = Q.solve(negC);
        double optimalValue = computeObjective(Q, c, xOpt);

        return new OptResult.Builder(optimalValue, xOpt)
                .converged(true)
                .convergenceReason("线性系统求解完成")
                .iterations(1)
                .executionTimeMs(System.currentTimeMillis() - startTime)
                .build();
    }

    private void ensurePositive(IVector v) {
        for (int i = 0; i < v.length(); i++) {
            double val = ((Number) v.get(i)).doubleValue();
            if (val <= 0) v.set(i, 1.0);
        }
    }

    /**
     * 构建扩展Q矩阵
     */
    private IMatrix buildExtendedQ(IMatrix Q, int n, int numIneq) {
        int total = n + numIneq;
        double[][] data = new double[total][total];

        // 复制Q到左上角
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = ((Number) Q.get(i, j)).doubleValue();
            }
        }

        // 对角线元素（对于松弛变量）
        for (int i = n; i < total; i++) {
            data[i][i] = 1e-6; // 小正数保证正定性
        }

        return Linalg.matrix(data);
    }

    /**
     * 构建扩展c向量
     */
    private IVector buildExtendedC(IVector c, int n, int numIneq) {
        int total = n + numIneq;
        double[] data = new double[total];

        for (int i = 0; i < n; i++) {
            data[i] = ((Number) c.get(i)).doubleValue();
        }

        return Linalg.vector(data);
    }

    /**
     * 构建扩展不等式约束矩阵
     */
    private IMatrix buildExtendedAub(IMatrix A_ub, int n, int numIneq) {
        if (A_ub == null) {
            return null;
        }

        int m = A_ub.rows();
        int total = n + numIneq;
        double[][] data = new double[m][total];

        // 复制A_ub到左半部分
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = ((Number) A_ub.get(i, j)).doubleValue();
            }
        }

        // 添加单位矩阵块（对于松弛变量）
        for (int i = 0; i < numIneq; i++) {
            data[i][n + i] = 1.0;
        }

        return Linalg.matrix(data);
    }

    /**
     * 构建扩展等式约束矩阵
     */
    private IMatrix buildExtendedAeq(IMatrix A_eq, int n, int numIneq) {
        if (A_eq == null) {
            return null;
        }

        int m = A_eq.rows();
        int total = n + numIneq;
        double[][] data = new double[m][total];

        // 只取前n列（原始变量部分）
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = ((Number) A_eq.get(i, j)).doubleValue();
            }
        }

        return Linalg.matrix(data);
    }

    /**
     * 求解牛顿系统
     * [H   A'  I ][dx]   [-r_d]
     * [A   0   0 ][dy] = [-r_p]
     * [Z   0   X ][dz]   [-r_cs]
     */
    private NewtonResult solveNewtonSystem(IMatrix Q, IMatrix A_ub, IMatrix A_eq,
                                          IVector x, IVector z,
                                          IVector c, IVector y, double mu) {
        int n = x.length();
        int m_eq = (A_eq != null) ? A_eq.rows() : 0;

        // 计算修正Hessian：H = Q + diag(z/x)
        IMatrix H = computeModifiedHessian(Q, x, z);

        // 计算右端项
        IVector r_d = computeDualResidual(Q, c, A_eq, y, z, x);
        IVector r_p = computePrimalResidual(A_ub, null, A_eq, null, x);

        // 简化策略：先忽略等式约束的耦合，使用分解方法
        // 解 H * dx = -r_d 得到 dx
        IVector dx = solveSymmetricPositiveDefinite(H, r_d.multiplyByScalar(-1.0));

        // 计算 dz = -(z + r_cs/x) + (z/x)*dx
        double r_cs = computeComplementarityGap(x, z, mu);
        IVector dz = Linalg.zeros(n);
        for (int i = 0; i < n; i++) {
            double xi = Math.max(((Number) x.get(i)).doubleValue(), 1e-8);
            double zi = ((Number) z.get(i)).doubleValue();
            double dxi = ((Number) dx.get(i)).doubleValue();
            // dz_i = -(z_i + (r_cs - x_i*z_i)/x_i) + (z_i/x_i)*dx_i
            //      = -z_i - (r_cs - x_i*z_i)/x_i + (z_i/x_i)*dx_i
            //      = -r_cs/x_i + (z_i/x_i)*dx_i
            dz.set(i, -r_cs / n / xi + (zi / xi) * dxi);
        }

        // 如果有等式约束，使用投影校正 dx
        if (m_eq > 0 && A_eq != null) {
            IVector dy = solveForEqualityMultipliers(H, A_eq, r_p, dx);
            IVector corr = A_eq.transposeNew().mmul(dy);
            dx = dx.sub(corr);
            return new NewtonResult(true, dx, dz, dy);
        }

        return new NewtonResult(true, dx, dz, null);
    }

    /**
     * 计算修正Hessian
     */
    private IMatrix computeModifiedHessian(IMatrix Q, IVector x, IVector z) {
        int n = Q.rows();
        double[][] data = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = ((Number) Q.get(i, j)).doubleValue();
            }
        }

        for (int i = 0; i < n; i++) {
            double xi = Math.max(((Number) x.get(i)).doubleValue(), 1e-8);
            double zi = Math.max(((Number) z.get(i)).doubleValue(), 1e-8);
            data[i][i] += zi / xi;
        }

        return Linalg.matrix(data);
    }

    /**
     * 使用共轭梯度法求解对称正定系统 H*x = b
     */
    private IVector solveSymmetricPositiveDefinite(IMatrix H, IVector b) {
        int n = H.rows();
        double[][] hData = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                hData[i][j] = ((Number) H.get(i, j)).doubleValue();
            }
        }

        double[] x = new double[n];
        double[] r = new double[n];
        double[] p = new double[n];

        // x = 0, r = b
        for (int i = 0; i < n; i++) {
            x[i] = 0;
            r[i] = ((Number) b.get(i)).doubleValue();
            p[i] = r[i];
        }

        double rs_old = dot(r, r);

        for (int iter = 0; iter < n; iter++) {
            // Hp = H * p
            double[] hp = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    hp[i] += hData[i][j] * p[j];
                }
            }

            double pHp = dot(p, hp);
            if (Math.abs(pHp) < 1e-12) {
                break;
            }

            double alpha = rs_old / pHp;

            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * hp[i];
            }

            double rs_new = dot(r, r);
            if (Math.sqrt(rs_new) < 1e-8) {
                break;
            }

            double beta = rs_new / rs_old;
            for (int i = 0; i < n; i++) {
                p[i] = r[i] + beta * p[i];
            }

            rs_old = rs_new;
        }

        return Linalg.vector(x);
    }

    private double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private double dot(IVector a, IVector b) {
        double sum = 0;
        for (int i = 0; i < a.length(); i++) {
            sum += ((Number) a.get(i)).doubleValue() * ((Number) b.get(i)).doubleValue();
        }
        return sum;
    }

    /**
     * 计算等式约束乘子
     */
    private IVector solveForEqualityMultipliers(IMatrix H, IMatrix A_eq, IVector r_p, IVector dx) {
        int m = A_eq.rows();
        int n = H.rows();

        // 简化为：dy = (A * H^{-1} * A')^{-1} * (A * H^{-1} * (-r_d) - r_p)
        // 使用直接方法求解

        double[][] aData = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                aData[i][j] = ((Number) A_eq.get(i, j)).doubleValue();
            }
        }

        // 计算 A * H^{-1}
        double[][] ahInv = new double[m][n];
        for (int i = 0; i < m; i++) {
            IVector hiInvRow = solveSymmetricPositiveDefinite(H, extractRowAsVector(A_eq, i).multiplyByScalar(-1.0));
            for (int j = 0; j < n; j++) {
                ahInv[i][j] = ((Number) hiInvRow.get(j)).doubleValue();
            }
        }

        // 计算 AHA = A * H^{-1} * A'
        double[][] aha = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                for (int k = 0; k < n; k++) {
                    aha[i][j] += ahInv[i][k] * aData[j][k];
                }
            }
        }

        // 计算 RHS = A * H^{-1} * (-r_d) - r_p
        double[] rhs = new double[m];
        for (int i = 0; i < m; i++) {
            rhs[i] = -((Number) r_p.get(i)).doubleValue();
            for (int j = 0; j < n; j++) {
                rhs[i] -= ahInv[i][j] * ((Number) dx.get(j)).doubleValue();
            }
        }

        // 解 AHA * dy = rhs
        double[] dy = solveSmallSystem(aha, rhs);

        return Linalg.vector(dy);
    }

    private IVector extractRowAsVector(IMatrix A, int row) {
        double[] data = new double[A.cols()];
        for (int j = 0; j < A.cols(); j++) {
            data[j] = ((Number) A.get(row, j)).doubleValue();
        }
        return Linalg.vector(data);
    }

    /**
     * 解小型稠密线性系统
     */
    private double[] solveSmallSystem(double[][] A, double[] b) {
        int n = b.length;
        double[][] aug = new double[n][n + 1];

        // 构造增广矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i][j] = A[i][j];
            }
            aug[i][n] = b[i];
        }

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < n; col++) {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
                    maxRow = row;
                }
            }

            // Swap rows
            double[] temp = aug[col];
            aug[col] = aug[maxRow];
            aug[maxRow] = temp;

            if (Math.abs(aug[col][col]) < 1e-12) {
                continue;
            }

            // Eliminate
            for (int row = col + 1; row < n; row++) {
                double factor = aug[row][col] / aug[col][col];
                for (int j = col; j <= n; j++) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= aug[i][j] * x[j];
            }
            if (Math.abs(aug[i][i]) > 1e-12) {
                x[i] /= aug[i][i];
            }
        }

        return x;
    }

    /**
     * 计算原始可行性残差
     */
    private IVector computePrimalResidual(IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector x) {
        int m_ub = (A_ub != null) ? A_ub.rows() : 0;
        int m_eq = (A_eq != null) ? A_eq.rows() : 0;
        int m = m_ub + m_eq;

        if (m == 0) {
            return Linalg.zeros(1);
        }

        double[] residual = new double[m];
        int idx = 0;

        if (A_ub != null && b_ub != null) {
            for (int i = 0; i < m_ub; i++) {
                double val = 0;
                for (int j = 0; j < x.length(); j++) {
                    val += ((Number) A_ub.get(i, j)).doubleValue() * ((Number) x.get(j)).doubleValue();
                }
                residual[idx++] = val - ((Number) b_ub.get(i)).doubleValue();
            }
        }

        if (A_eq != null && b_eq != null) {
            for (int i = 0; i < m_eq; i++) {
                double val = 0;
                for (int j = 0; j < x.length(); j++) {
                    val += ((Number) A_eq.get(i, j)).doubleValue() * ((Number) x.get(j)).doubleValue();
                }
                residual[idx++] = val - ((Number) b_eq.get(i)).doubleValue();
            }
        }

        return Linalg.vector(residual);
    }

    /**
     * 计算对偶可行性残差: r_d = Q*x + c - A_eq'*y - z
     */
    private IVector computeDualResidual(IMatrix Q, IVector c, IMatrix A_eq, IVector y, IVector z, IVector x) {
        IVector qx = Q.mmul(x);
        IVector rd = qx.add(c);

        if (A_eq != null && y != null) {
            IVector aty = A_eq.transposeNew().mmul(y);
            rd = rd.sub(aty);
        }

        if (z != null) {
            rd = rd.sub(z);
        }

        return rd;
    }

    /**
     * 计算互补性间隙
     */
    private double computeComplementarityGap(IVector x, IVector z, double mu) {
        double gap = 0;
        for (int i = 0; i < x.length(); i++) {
            double xi = ((Number) x.get(i)).doubleValue();
            double zi = ((Number) z.get(i)).doubleValue();
            gap += xi * zi;
        }
        return gap - x.length() * mu;
    }

    /**
     * 计算目标函数值
     */
    private double computeObjective(IMatrix Q, IVector c, IVector x) {
        IVector qx = Q.mmul(x);
        double xqx = x.dotValue(qx);
        double cx = c.dotValue(x);
        return 0.5 * xqx + cx;
    }

    /**
     * 计算约束违反度
     */
    private double computeConstraintViolation(IMatrix A_ub, IVector b_ub, IMatrix A_eq, IVector b_eq, IVector x) {
        double maxViolation = 0.0;
        if (A_ub != null && b_ub != null) {
            for (int i = 0; i < A_ub.rows(); i++) {
                double val = 0;
                for (int j = 0; j < x.length(); j++) {
                    val += ((Number) A_ub.get(i, j)).doubleValue() * ((Number) x.get(j)).doubleValue();
                }
                double violation = val - ((Number) b_ub.get(i)).doubleValue();
                maxViolation = Math.max(maxViolation, violation);
            }
        }
        if (A_eq != null && b_eq != null) {
            for (int i = 0; i < A_eq.rows(); i++) {
                double val = 0;
                for (int j = 0; j < x.length(); j++) {
                    val += ((Number) A_eq.get(i, j)).doubleValue() * ((Number) x.get(j)).doubleValue();
                }
                double violation = Math.abs(val - ((Number) b_eq.get(i)).doubleValue());
                maxViolation = Math.max(maxViolation, violation);
            }
        }
        return maxViolation;
    }

    /**
     * 计算步长
     */
    private double computeStepSize(IVector v, IVector dv, double bound) {
        double alpha = bound;
        for (int i = 0; i < v.length(); i++) {
            double vi = ((Number) v.get(i)).doubleValue();
            double dvi = ((Number) dv.get(i)).doubleValue();
            if (dvi < -1e-12) {
                alpha = Math.min(alpha, -0.99 * vi / dvi);
            }
        }
        return Math.max(alpha, 1e-6);
    }

    private double norm(IVector v) {
        double sum = 0;
        for (int i = 0; i < v.length(); i++) {
            double vi = ((Number) v.get(i)).doubleValue();
            sum += vi * vi;
        }
        return Math.sqrt(sum);
    }

    /**
     * 提取向量前n个元素
     */
    private IVector extractFirstN(IVector v, int n) {
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = ((Number) v.get(i)).doubleValue();
        }
        return Linalg.vector(data);
    }

    private static class NewtonResult {
        boolean feasible;
        IVector dx, dz, dy;

        NewtonResult(boolean feasible) {
            this.feasible = feasible;
        }

        NewtonResult(boolean feasible, IVector dx, IVector dz, IVector dy) {
            this.feasible = feasible;
            this.dx = dx;
            this.dz = dz;
            this.dy = dy;
        }
    }

    private boolean adaptiveMu = true;
}
