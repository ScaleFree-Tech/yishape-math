package com.yishape.lab.math.compute.hpc;

import java.lang.reflect.Method;
import java.util.concurrent.ConcurrentHashMap;
import com.yishape.lab.util.YishapeLogger;

/**
 * <p><strong>YiShape Math HPC 桥接</strong>：大块双精度矩阵乘与部分分解经本类反射调用可选构件
 * {@code yishape-math-hpc}（底层可为 {@code yishape_math_rust}）。classpath 未加入
 * {@code com.yishape.lab.math.hpc} 时，本类各方法安全降级，不会抛出 {@link NoClassDefFoundError}。</p>
 *
 * <p>{@code solveSquare} 等较晚加入的入口单独探测：旧版扩展 JAR 缺少对应方法时仅该入口降级，
 * 矩阵乘、Cholesky、SVD、LP 等其余入口不受影响。</p>
 *
 * <p>开关与维度阈值见 {@link HpcConfig}（系统属性前缀 {@code yishape.hpc.…}）。</p>
 */
public final class HpcOptionalRuntime {

    private static final YishapeLogger logger = YishapeLogger.getLogger(HpcOptionalRuntime.class);

    private static final String HPC_CLASS = "com.yishape.lab.math.hpc.YishapeHpc";

    private static final Class<?> HPC;
    private static final Method M_IS_NATIVE;
    private static final Method M_TRY_MATMUL;

    // Cache for dynamic method lookups (tryActivationF64, tryElementwiseF64, etc.)
    // and result accessor methods (status, lLower, u, etc.)
    private static final ConcurrentHashMap<String, Method> METHOD_CACHE = new ConcurrentHashMap<>();

    private static Method cachedMethod(Class<?> clazz, String name, Class<?>... paramTypes) {
        String key = clazz.getName() + "." + name + ":" + java.util.Arrays.toString(paramTypes);
        return METHOD_CACHE.computeIfAbsent(key, k -> {
            try {
                return clazz.getMethod(name, paramTypes);
            } catch (NoSuchMethodException e) {
                return null;
            }
        });
    }
    private static final Method M_CHOLESKY;
    private static final Method M_SVD;
    private static final Method M_LP;
    private static final Method M_LP_MILP;
    private static final Method M_MILP_AVAILABLE;
    private static final Method M_SOLVE_SQUARE;
    // v0.5.0 新增
    private static final Method M_EIGEN_SYMMETRIC;
    private static final Method M_SOLVE_MULTI_RHS;
    private static final Method M_INVERSE;
    private static final Method M_QR;
    private static final Method M_LU;
    private static final Method M_EIGEN_NONSYMMETRIC;
    // v0.8.0: autodiff graph execution
    private static final Method M_EXECUTE_GRAPH;
    // v0.9.0: binary graph execution (YSGP protocol)
    private static final Method M_EXECUTE_GRAPH_BINARY;
    // v0.7.0: HNSW
    private static final Method M_HNSW_BUILD;
    private static final Method M_HNSW_ADD;
    private static final Method M_HNSW_SEARCH;
    private static final Method M_HNSW_GET;
    private static final Method M_HNSW_SIZE;
    private static final Method M_HNSW_SET_EF;
    private static final Method M_HNSW_FREE;
    // Cached accessor methods for HnswSearchResult record
    private static final Method M_RESULT_STATUS;
    private static final Method M_RESULT_IDS;
    private static final Method M_RESULT_DISTANCES;
    private static final Method M_RESULT_FOUND;

    static {
        Class<?> c = null;
        Method mNative = null;
        Method mMul = null;
        Method mChol = null;
        Method mSvd = null;
        Method mLp = null;
        Method mMilp = null;
        Method mMilpAvail = null;
        Method mSolveSquare = null;
        // v0.5.0 新增
        Method mEigenSym = null;
        Method mSolveMulti = null;
        Method mInverse = null;
        Method mQr = null;
        Method mLu = null;
        Method mEigenNonsym = null;
        try {
            c = Class.forName(HPC_CLASS);
            mNative = c.getMethod("isNativeRuntimeAvailable");
            mMul = c.getMethod("tryMatMul", double[][].class, double[][].class);
            mChol = c.getMethod("cholesky", double[][].class);
            mSvd = c.getMethod("svd", double[][].class);
            mLp = c.getMethod("lpNonnegative",
                    double[].class, double[][].class, double[].class,
                    double[][].class, double[].class);
            mMilp = c.getMethod("lpMixedIntegerNonnegative",
                    double[].class, double[][].class, double[].class,
                    double[][].class, double[].class, int[].class);
            mMilpAvail = c.getMethod("isMixedIntegerLpNativeAvailable");
        } catch (ReflectiveOperationException | LinkageError e) {
            c = null;
            mNative = mMul = mChol = mSvd = mLp = mMilp = mMilpAvail = null;
        }
        if (c != null) {
            try {
                mSolveSquare = c.getMethod("solveSquare", double[][].class, double[].class);
            } catch (ReflectiveOperationException e) {
                mSolveSquare = null;
            }
            // v0.5.0 新增（可选探测）
            try {
                mEigenSym = c.getMethod("eigenSymmetric", double[][].class);
            } catch (ReflectiveOperationException e) {
                mEigenSym = null;
            }
            try {
                mSolveMulti = c.getMethod("solveMultiRhs", double[][].class, double[][].class);
            } catch (ReflectiveOperationException e) {
                mSolveMulti = null;
            }
            try {
                mInverse = c.getMethod("inverse", double[][].class);
            } catch (ReflectiveOperationException e) {
                mInverse = null;
            }
            try {
                mQr = c.getMethod("qr", double[][].class);
            } catch (ReflectiveOperationException e) {
                mQr = null;
            }
            try {
                mLu = c.getMethod("lu", double[][].class);
            } catch (ReflectiveOperationException e) {
                mLu = null;
            }
            try {
                mEigenNonsym = c.getMethod("eigenNonsymmetric", double[][].class);
            } catch (ReflectiveOperationException e) {
                mEigenNonsym = null;
            }
            // v0.8.0: autodiff graph execution（可选探测）
            Method mExecuteGraph = null;
            try {
                mExecuteGraph = c.getMethod("executeGraph", String.class);
            } catch (ReflectiveOperationException e) {
                mExecuteGraph = null;
            }
            // v0.9.0: binary graph execution (YSGP protocol)
            Method mExecuteGraphBinary = null;
            try {
                mExecuteGraphBinary = c.getMethod("executeGraphBinary", byte[].class);
            } catch (ReflectiveOperationException e) {
                mExecuteGraphBinary = null;
            }
            // v0.7.0: HNSW（可选探测）
            Method mHnswBuild = null, mHnswAdd = null, mHnswSearch = null,
                    mHnswGet = null, mHnswSize = null, mHnswSetEf = null, mHnswFree = null;
            try {
                mHnswBuild = c.getMethod("hnswBuildF32", int.class, float[].class, long[].class,
                        int.class, int.class, int.class, int.class);
            } catch (ReflectiveOperationException e) {
                mHnswBuild = null;
            }
            try {
                mHnswAdd = c.getMethod("hnswAddF32", long.class, long.class, float[].class);
            } catch (ReflectiveOperationException e) {
                mHnswAdd = null;
            }
            try {
                mHnswSearch = c.getMethod("hnswSearchF32", long.class, float[].class, int.class);
            } catch (ReflectiveOperationException e) {
                mHnswSearch = null;
            }
            try {
                mHnswGet = c.getMethod("hnswGetF32", long.class, long.class, float[].class);
            } catch (ReflectiveOperationException e) {
                mHnswGet = null;
            }
            try {
                mHnswSize = c.getMethod("hnswSize", long.class);
            } catch (ReflectiveOperationException e) {
                mHnswSize = null;
            }
            try {
                mHnswSetEf = c.getMethod("hnswSetEf", long.class, int.class);
            } catch (ReflectiveOperationException e) {
                mHnswSetEf = null;
            }
            try {
                mHnswFree = c.getMethod("hnswFree", long.class);
            } catch (ReflectiveOperationException e) {
                mHnswFree = null;
            }
            M_EXECUTE_GRAPH = mExecuteGraph;
            M_EXECUTE_GRAPH_BINARY = mExecuteGraphBinary;
            M_HNSW_BUILD = mHnswBuild;
            M_HNSW_ADD = mHnswAdd;
            M_HNSW_SEARCH = mHnswSearch;
            M_HNSW_GET = mHnswGet;
            M_HNSW_SIZE = mHnswSize;
            M_HNSW_SET_EF = mHnswSetEf;
            M_HNSW_FREE = mHnswFree;
            // Cache HnswSearchResult accessor methods to avoid per-query getMethod()
            Method mResStatus = null, mResIds = null, mResDists = null, mResFound = null;
            try {
                Class<?> resClass = Class.forName("com.yishape.lab.math.hpc.HnswSearchResult");
                mResStatus = resClass.getMethod("status");
                mResIds    = resClass.getMethod("ids");
                mResDists  = resClass.getMethod("distances");
                mResFound  = resClass.getMethod("found");
            } catch (ReflectiveOperationException e) { /* optional */ }
            M_RESULT_STATUS    = mResStatus;
            M_RESULT_IDS       = mResIds;
            M_RESULT_DISTANCES = mResDists;
            M_RESULT_FOUND     = mResFound;
        } else {
            M_EXECUTE_GRAPH = null;
            M_EXECUTE_GRAPH_BINARY = null;
            M_HNSW_BUILD = M_HNSW_ADD = M_HNSW_SEARCH = M_HNSW_GET =
                    M_HNSW_SIZE = M_HNSW_SET_EF = M_HNSW_FREE = null;
            M_RESULT_STATUS = M_RESULT_IDS = M_RESULT_DISTANCES = M_RESULT_FOUND = null;
        }
        HPC = c;
        M_IS_NATIVE = mNative;
        M_TRY_MATMUL = mMul;
        M_CHOLESKY = mChol;
        M_SVD = mSvd;
        M_LP = mLp;
        M_LP_MILP = mMilp;
        M_MILP_AVAILABLE = mMilpAvail;
        M_SOLVE_SQUARE = mSolveSquare;
        M_EIGEN_SYMMETRIC = mEigenSym;
        M_SOLVE_MULTI_RHS = mSolveMulti;
        M_INVERSE = mInverse;
        M_QR = mQr;
        M_LU = mLu;
        M_EIGEN_NONSYMMETRIC = mEigenNonsym;
    }

    /**
     * 一次检测结果缓存，整个 JVM 运行周期内不重复检测。
     */
    private static volatile Boolean nativeRuntimeAvailable = null;

    private static void logHpcError(String method, Throwable e) {
        if (e instanceof ClassNotFoundException || e instanceof NoSuchMethodException) {
            return;
        }
        logger.warn("HPC bridge error in {}: {}", method, e.getMessage(), e);
    }

    /**
     * {@code com.yishape.lab.math.hpc} 是否在 classpath 上已成功解析。
     */
    public static boolean isExtensionPresent() {
        return HPC != null;
    }

    /**
     * 检测 HPC 原生运行时是否可用（首次调用检测并缓存，后续直接返回缓存结果）。
     */
    public static boolean isNativeRuntimeAvailable() {
        if (nativeRuntimeAvailable != null) {
            return nativeRuntimeAvailable;
        }
        if (M_IS_NATIVE == null) {
            nativeRuntimeAvailable = false;
            return false;
        }
        try {
            Object v = M_IS_NATIVE.invoke(null);
            nativeRuntimeAvailable = v instanceof Boolean && (Boolean) v;
        } catch (ReflectiveOperationException | LinkageError e) {
            logHpcError("isNativeRuntimeAvailable", e);
            nativeRuntimeAvailable = false;
        }
        if (nativeRuntimeAvailable) {
            logger.info("HPC native runtime detected and available");
        }
        return nativeRuntimeAvailable;
    }

    public static double[][] tryMatMul(double[][] a, double[][] b) {
        if (M_TRY_MATMUL == null) {
            return null;
        }
        try {
            Object out = M_TRY_MATMUL.invoke(null, a, b);
            return (out instanceof double[][]) ? (double[][]) out : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("tryMatMul", e);
            return null;
        }
    }

    public static RCholesky cholesky(double[][] a) {
        if (M_CHOLESKY == null) {
            return null;
        }
        try {
            Object res = M_CHOLESKY.invoke(null, (Object) a);
            if (res == null) {
                return null;
            }
            Class<?> rc = res.getClass();
            int st = (Integer) cachedMethod(rc, "status").invoke(res);
            double[][] l = (double[][]) cachedMethod(rc, "lLower").invoke(res);
            return new RCholesky(st, l);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("cholesky", e);
            return null;
        }
    }

    public static RSvd svd(double[][] a) {
        if (M_SVD == null) {
            return null;
        }
        try {
            Object res = M_SVD.invoke(null, (Object) a);
            if (res == null) {
                return null;
            }
            Class<?> rc = res.getClass();
            int st = (Integer) cachedMethod(rc, "status").invoke(res);
            double[][] u = (double[][]) cachedMethod(rc, "u").invoke(res);
            double[] s = (double[]) cachedMethod(rc, "singularValues").invoke(res);
            double[][] vt = (double[][]) cachedMethod(rc, "vt").invoke(res);
            return new RSvd(st, u, s, vt);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("svd", e);
            return null;
        }
    }

    /**
     * 稠密方阵 Ax=b（与 {@code YishapeHpc.solveSquare} 一致）；扩展 JAR 或方法不存在时返回 {@code null}。
     */
    public static RSolveSquare solveSquare(double[][] a, double[] b) {
        if (M_SOLVE_SQUARE == null) {
            return null;
        }
        try {
            Object res = M_SOLVE_SQUARE.invoke(null, a, b);
            if (res == null) {
                return null;
            }
            Class<?> cl = res.getClass();
            int st = (Integer) cachedMethod(cl, "status").invoke(res);
            boolean ok = (Boolean) cachedMethod(cl, "ok").invoke(res);
            Object xRaw;
            Method mX = cachedMethod(cl, "x");
            if (mX != null) {
                xRaw = mX.invoke(res);
            } else {
                xRaw = cachedMethod(cl, "getX").invoke(res);
            }
            double[] x = (double[]) xRaw;
            return new RSolveSquare(st, ok, x);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("solveSquare", e);
            return null;
        }
    }

    public static boolean isMixedIntegerLpNativeAvailable() {
        if (M_MILP_AVAILABLE == null) {
            return false;
        }
        try {
            Object v = M_MILP_AVAILABLE.invoke(null);
            return v instanceof Boolean && (Boolean) v;
        } catch (ReflectiveOperationException | LinkageError e) {
            logHpcError("isMixedIntegerLpNativeAvailable", e);
            return false;
        }
    }

    public static HpcDenseLpResult lpNonnegative(
            double[] c, double[][] aUb, double[] bUb, double[][] aEq, double[] bEq) {
        if (M_LP == null) {
            return null;
        }
        try {
            Object res = M_LP.invoke(null, c, aUb, bUb, aEq, bEq);
            return toLp(res);
        } catch (ReflectiveOperationException | LinkageError e) {
            logHpcError("lpNonnegative", e);
            return null;
        }
    }

    public static HpcDenseLpResult lpMixedIntegerNonnegative(
            double[] c, double[][] aUb, double[] bUb,
            double[][] aEq, double[] bEq, int[] integrality) {
        if (M_LP_MILP == null) {
            return null;
        }
        try {
            Object res = M_LP_MILP.invoke(null, c, aUb, bUb, aEq, bEq, integrality);
            return toLp(res);
        } catch (ReflectiveOperationException | LinkageError e) {
            logHpcError("lpMixedIntegerNonnegative", e);
            return null;
        }
    }

    private static HpcDenseLpResult toLp(Object res) {
        if (res == null) {
            return null;
        }
        try {
            Class<?> rc = res.getClass();
            int st = (Integer) cachedMethod(rc, "status").invoke(res);
            double obj = (Double) cachedMethod(rc, "objective").invoke(res);
            double[] x = (double[]) cachedMethod(rc, "x").invoke(res);
            return new HpcDenseLpResult(st, obj, x);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("toLp", e);
            return null;
        }
    }

    public static final class RSolveSquare {
        private final int status;
        private final boolean ok;
        private final double[] x;

        public RSolveSquare(int status, boolean ok, double[] x) {
            this.status = status;
            this.ok = ok;
            this.x = x;
        }

        public int status() {
            return status;
        }

        public boolean ok() {
            return ok;
        }

        public double[] x() {
            return x;
        }
    }

    public static final class RCholesky {
        private final int status;
        private final double[][] lLower;

        public RCholesky(int status, double[][] lLower) {
            this.status = status;
            this.lLower = lLower;
        }

        public int status() {
            return status;
        }

        public double[][] lLower() {
            return lLower;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }

    public static final class RSvd {
        private final int status;
        private final double[][] u;
        private final double[] singularValues;
        private final double[][] vt;

        public RSvd(int status, double[][] u, double[] singularValues, double[][] vt) {
            this.status = status;
            this.u = u;
            this.singularValues = singularValues;
            this.vt = vt;
        }

        public int status() {
            return status;
        }

        public double[][] u() {
            return u;
        }

        public double[] singularValues() {
            return singularValues;
        }

        public double[][] vt() {
            return vt;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }

    // ===================== v0.5.0 新增 =====================

    public static REigenSymmetric eigenSymmetric(double[][] a) {
        if (M_EIGEN_SYMMETRIC == null) {
            return null;
        }
        try {
            Object res = M_EIGEN_SYMMETRIC.invoke(null, (Object) a);
            if (res == null) {
                return null;
            }
            Class<?> rc = res.getClass();
            int st = (Integer) cachedMethod(rc, "status").invoke(res);
            double[] w = (double[]) cachedMethod(rc, "eigenvaluesAscending").invoke(res);
            double[][] vecs = (double[][]) cachedMethod(rc, "eigenvectors").invoke(res);
            return new REigenSymmetric(st, w, vecs);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("eigenSymmetric", e);
            return null;
        }
    }

    public static RSolveMultiRhs solveMultiRhs(double[][] a, double[][] b) {
        if (M_SOLVE_MULTI_RHS == null) {
            return null;
        }
        try {
            Object res = M_SOLVE_MULTI_RHS.invoke(null, (Object) a, (Object) b);
            if (res == null) {
                return null;
            }
            Class<?> rc = res.getClass();
            int st = (Integer) cachedMethod(rc, "status").invoke(res);
            double[][] x = (double[][]) cachedMethod(rc, "x").invoke(res);
            return new RSolveMultiRhs(st, x);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("solveMultiRhs", e);
            return null;
        }
    }

    public static RInverse inverse(double[][] a) {
        if (M_INVERSE == null) {
            return null;
        }
        try {
            Object res = M_INVERSE.invoke(null, (Object) a);
            if (res == null) {
                return null;
            }
            Class<?> rc = res.getClass();
            int st = (Integer) cachedMethod(rc, "status").invoke(res);
            double[][] inv = (double[][]) cachedMethod(rc, "inv").invoke(res);
            return new RInverse(st, inv);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("inverse", e);
            return null;
        }
    }

    public static RQr qr(double[][] a) {
        if (M_QR == null) {
            return null;
        }
        try {
            Object res = M_QR.invoke(null, (Object) a);
            if (res == null) {
                return null;
            }
            Class<?> rc = res.getClass();
            int st = (Integer) cachedMethod(rc, "status").invoke(res);
            double[][] q = (double[][]) cachedMethod(rc, "q").invoke(res);
            double[][] r = (double[][]) cachedMethod(rc, "r").invoke(res);
            return new RQr(st, q, r);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("qr", e);
            return null;
        }
    }

    public static RLu lu(double[][] a) {
        if (M_LU == null) {
            return null;
        }
        try {
            Object res = M_LU.invoke(null, (Object) a);
            if (res == null) {
                return null;
            }
            Class<?> rc = res.getClass();
            int st = (Integer) cachedMethod(rc, "status").invoke(res);
            double[][] l = (double[][]) cachedMethod(rc, "l").invoke(res);
            double[][] u = (double[][]) cachedMethod(rc, "u").invoke(res);
            int[] p = (int[]) cachedMethod(rc, "p").invoke(res);
            return new RLu(st, l, u, p);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("lu", e);
            return null;
        }
    }

    public static REigenNonsymmetric eigenNonsymmetric(double[][] a) {
        if (M_EIGEN_NONSYMMETRIC == null) {
            return null;
        }
        try {
            Object res = M_EIGEN_NONSYMMETRIC.invoke(null, (Object) a);
            if (res == null) {
                return null;
            }
            Class<?> rc = res.getClass();
            int st = (Integer) cachedMethod(rc, "status").invoke(res);
            double[] er = (double[]) cachedMethod(rc, "eigenvaluesReal").invoke(res);
            double[] ei = (double[]) cachedMethod(rc, "eigenvaluesImag").invoke(res);
            double[][] vr = (double[][]) cachedMethod(rc, "eigenvectorsReal").invoke(res);
            double[][] vi = (double[][]) cachedMethod(rc, "eigenvectorsImag").invoke(res);
            return new REigenNonsymmetric(st, er, ei, vr, vi);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("eigenNonsymmetric", e);
            return null;
        }
    }

    // ===================== v0.7.0: HNSW vector index =====================

    public static boolean isHnswNativeAvailable() {
        return HpcConfig.isHnswEnabled() && HPC != null && M_HNSW_BUILD != null;
    }

    public static Long hnswBuildF32(int dims, float[] data, long[] ids, int metricType,
            int m, int efConstruction, int efSearch) {
        if (M_HNSW_BUILD == null) {
            return null;
        }
        try {
            Object res = M_HNSW_BUILD.invoke(null, dims, data, ids, metricType, m, efConstruction, efSearch);
            return (res instanceof Long) ? (Long) res : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("hnswBuildF32", e);
            return null;
        }
    }

    public static Integer hnswAddF32(long handle, long id, float[] data) {
        if (M_HNSW_ADD == null) {
            return null;
        }
        try {
            Object res = M_HNSW_ADD.invoke(null, handle, id, data);
            return (res instanceof Integer) ? (Integer) res : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("hnswAddF32", e);
            return null;
        }
    }

    public static RHnswSearch hnswSearchF32(long handle, float[] query, int k) {
        if (M_HNSW_SEARCH == null) {
            return null;
        }
        try {
            Object res = M_HNSW_SEARCH.invoke(null, handle, query, k);
            if (res == null) {
                return null;
            }
            int    st    = (Integer) M_RESULT_STATUS.invoke(res);
            if (st != HpcAbiCodes.OK) {
                return new RHnswSearch(st, new long[0], new float[0], 0);
            }
            long[] ids       = (long[])  M_RESULT_IDS.invoke(res);
            float[] distances = (float[]) M_RESULT_DISTANCES.invoke(res);
            int    found     = (Integer) M_RESULT_FOUND.invoke(res);
            return new RHnswSearch(st, ids, distances, found);
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("hnswSearchF32", e);
            return null;
        }
    }

    public static Integer hnswGetF32(long handle, long id, float[] dataOut) {
        if (M_HNSW_GET == null) {
            return null;
        }
        try {
            Object res = M_HNSW_GET.invoke(null, handle, id, dataOut);
            return (res instanceof Integer) ? (Integer) res : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("hnswGetF32", e);
            return null;
        }
    }

    public static Integer hnswSize(long handle) {
        if (M_HNSW_SIZE == null) {
            return null;
        }
        try {
            Object res = M_HNSW_SIZE.invoke(null, handle);
            return (res instanceof Integer) ? (Integer) res : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("hnswSize", e);
            return null;
        }
    }

    public static Integer hnswSetEf(long handle, int ef) {
        if (M_HNSW_SET_EF == null) {
            return null;
        }
        try {
            Object res = M_HNSW_SET_EF.invoke(null, handle, ef);
            return (res instanceof Integer) ? (Integer) res : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("hnswSetEf", e);
            return null;
        }
    }

    public static Integer hnswFree(long handle) {
        if (M_HNSW_FREE == null) {
            return null;
        }
        try {
            Object res = M_HNSW_FREE.invoke(null, handle);
            return (res instanceof Integer) ? (Integer) res : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("hnswFree", e);
            return null;
        }
    }

    // ===================== v0.8.0: autodiff graph execution =====================

    /**
     * Execute an autodiff computation graph via native runtime.
     *
     * @param graphJson JSON graph definition from {@code GraphExporter.toJson()}
     * @return array of [loss, grads[0], grads[1], ...] or {@code null} on failure
     */
    public static double[][] tryExecuteGraph(String graphJson) {
        if (M_EXECUTE_GRAPH == null) {
            return null;
        }
        try {
            Object result = M_EXECUTE_GRAPH.invoke(null, graphJson);
            if (result == null) return null;
            // result is YishapeHpc.GraphResult record: loss() → double, grads() → double[][]
            Class<?> rc = result.getClass();
            double loss = (Double) cachedMethod(rc, "loss").invoke(result);
            double[][] grads = (double[][]) cachedMethod(rc, "grads").invoke(result);
            // Return as [loss_array, grad0, grad1, ...]
            double[][] out = new double[1 + grads.length][];
            out[0] = new double[] { loss };
            for (int i = 0; i < grads.length; i++) {
                out[i + 1] = grads[i];
            }
            return out;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("tryExecuteGraph", e);
            return null;
        }
    }

    /**
     * Binary graph execution via YSGP protocol. Returns raw result bytes, or null.
     */
    public static byte[] tryExecuteGraphBinary(byte[] data) {
        if (M_EXECUTE_GRAPH_BINARY == null) {
            return null;
        }
        try {
            Object out = M_EXECUTE_GRAPH_BINARY.invoke(null, (Object) data);
            return (out instanceof byte[] b) ? b : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logHpcError("tryExecuteGraphBinary", e);
            return null;
        }
    }

    // ==================== Element-wise / activation (f64) ====================

    /**
     * 尝试通过 HPC 原生库执行一元激活函数（double 版本）。
     *
     * @param opName HPC 方法名（如 "reluF64", "sigmoidF64", "geluF64"）
     * @param input  输入 double 数组
     * @return 结果 double 数组，失败返回 null
     */
    public static double[] tryActivationF64(String opName, double[] input) {
        if (!isNativeRuntimeAvailable() || HPC == null) return null;
        Method m = cachedMethod(HPC, opName, double[].class, double[].class);
        if (m == null) return null;
        try {
            double[] out = new double[input.length];
            int status = (Integer) m.invoke(null, input, out);
            if (status != 0) return null;
            return out;
        } catch (ReflectiveOperationException | LinkageError e) {
            logHpcError(opName, e);
            return null;
        }
    }

    /**
     * 尝试通过 HPC 原生库执行一元激活函数。
     * 内部将 float[] 转为 double[] 调用原生，再转回 float[]。
     *
     * @param opName HPC 方法名（如 "reluF64", "sigmoidF64", "geluF64"）
     * @param input  输入 float 数组
     * @return 结果 float 数组，失败返回 null
     */
    public static float[] tryFloatActivation(String opName, float[] input) {
        // HPC 无双精度浮点核，float→double→float 转换+三次临时分配抵消任何收益，直接走 SIMD
        return null;
    }

    /**
     * 尝试通过 HPC 原生库执行二元逐元素运算。
     *
     * @param opName HPC 方法名（如 "addF64", "mulF64"）
     * @param a      第一个输入 float 数组
     * @param b      第二个输入 float 数组
     * @return 结果 float 数组，失败返回 null
     */
    public static float[] tryFloatElementwise(String opName, float[] a, float[] b) {
        // HPC 无双精度浮点核，float→double→float 转换+临时分配抵消任何收益，直接走 SIMD
        return null;
    }

    /**
     * 尝试通过 HPC 原生库执行二元逐元素运算（double 版本）。
     *
     * @param opName HPC 方法名（如 "addF64", "mulF64"）
     * @param a      第一个输入 double 数组
     * @param b      第二个输入 double 数组
     * @return 结果 double 数组，失败返回 null
     */
    public static double[] tryElementwiseF64(String opName, double[] a, double[] b) {
        if (!isNativeRuntimeAvailable() || HPC == null) return null;
        Method m = cachedMethod(HPC, opName, double[].class, double[].class, double[].class);
        if (m == null) return null;
        try {
            double[] out = new double[a.length];
            int status = (Integer) m.invoke(null, a, b, out);
            if (status != 0) return null;
            return out;
        } catch (ReflectiveOperationException | LinkageError e) {
            logHpcError(opName, e);
            return null;
        }
    }

    public static final class RHnswSearch {
        private final int status;
        private final long[] ids;
        private final float[] distances;
        private final int found;

        public RHnswSearch(int status, long[] ids, float[] distances, int found) {
            this.status = status;
            this.ids = ids;
            this.distances = distances;
            this.found = found;
        }

        public int status() {
            return status;
        }

        public long[] ids() {
            return ids;
        }

        public float[] distances() {
            return distances;
        }

        public int found() {
            return found;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }

    public static final class REigenSymmetric {
        private final int status;
        private final double[] eigenvalues;
        private final double[][] eigenvectors;

        public REigenSymmetric(int status, double[] eigenvalues, double[][] eigenvectors) {
            this.status = status;
            this.eigenvalues = eigenvalues;
            this.eigenvectors = eigenvectors;
        }

        public int status() {
            return status;
        }

        public double[] eigenvalues() {
            return eigenvalues;
        }

        public double[][] eigenvectors() {
            return eigenvectors;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }

    public static final class RSolveMultiRhs {
        private final int status;
        private final double[][] x;

        public RSolveMultiRhs(int status, double[][] x) {
            this.status = status;
            this.x = x;
        }

        public int status() {
            return status;
        }

        public double[][] x() {
            return x;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }

    public static final class RInverse {
        private final int status;
        private final double[][] inv;

        public RInverse(int status, double[][] inv) {
            this.status = status;
            this.inv = inv;
        }

        public int status() {
            return status;
        }

        public double[][] inv() {
            return inv;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }

    public static final class RQr {
        private final int status;
        private final double[][] q;
        private final double[][] r;

        public RQr(int status, double[][] q, double[][] r) {
            this.status = status;
            this.q = q;
            this.r = r;
        }

        public int status() {
            return status;
        }

        public double[][] q() {
            return q;
        }

        public double[][] r() {
            return r;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }

    public static final class RLu {
        private final int status;
        private final double[][] l;
        private final double[][] u;
        private final int[] p;

        public RLu(int status, double[][] l, double[][] u, int[] p) {
            this.status = status;
            this.l = l;
            this.u = u;
            this.p = p;
        }

        public int status() {
            return status;
        }

        public double[][] l() {
            return l;
        }

        public double[][] u() {
            return u;
        }

        public int[] p() {
            return p;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }

    public static final class REigenNonsymmetric {
        private final int status;
        private final double[] eigenvaluesReal;
        private final double[] eigenvaluesImag;
        private final double[][] eigenvectorsReal;
        private final double[][] eigenvectorsImag;

        public REigenNonsymmetric(int status, double[] eigenvaluesReal, double[] eigenvaluesImag,
                                   double[][] eigenvectorsReal, double[][] eigenvectorsImag) {
            this.status = status;
            this.eigenvaluesReal = eigenvaluesReal;
            this.eigenvaluesImag = eigenvaluesImag;
            this.eigenvectorsReal = eigenvectorsReal;
            this.eigenvectorsImag = eigenvectorsImag;
        }

        public int status() {
            return status;
        }

        public double[] eigenvaluesReal() {
            return eigenvaluesReal;
        }

        public double[] eigenvaluesImag() {
            return eigenvaluesImag;
        }

        public double[][] eigenvectorsReal() {
            return eigenvectorsReal;
        }

        public double[][] eigenvectorsImag() {
            return eigenvectorsImag;
        }

        public boolean ok() {
            return status == HpcAbiCodes.OK;
        }
    }
}
