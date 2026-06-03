package com.yishape.lab.math.bench;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.ArrayRealVector;
import org.apache.commons.math4.legacy.linear.CholeskyDecomposition;
import org.apache.commons.math4.legacy.linear.EigenDecomposition;
import org.apache.commons.math4.legacy.linear.LUDecomposition;
import org.apache.commons.math4.legacy.linear.QRDecomposition;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.RealVector;
import org.apache.commons.math4.legacy.linear.SingularValueDecomposition;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.hpc.HpcConfig;
import com.yishape.lab.math.compute.hpc.HpcSwitch;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.hpc.*;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.newton.RereConjugateGradient;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.newton.RustLBFGS;
import com.yishape.lab.math.optimize.newton.RustOWLQN;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereDCT;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.stats.distribution.GammaDistribution;
import com.yishape.lab.math.stats.distribution.NormalDistribution;

/**
 * Comprehensive benchmark: Pure Java (API), HPC (API wrapper), HPC (Direct YishapeHpc.*).
 * Records timing AND numerical checksums for correctness cross-validation with NumPy.
 * Output: benchmarks/out/direct_hpc_bench.csv
 */
public final class DirectHpcBenchmarkMain {

    private static final int WARMUP = 2;
    private static final int REPEAT = 5;
    private static final long SEED = 42L;
    private static final Path OUT = Path.of("benchmarks/out/direct_hpc_bench.csv");

    private static boolean nativeOk;
    private static final ExecutorService LP_EXECUTOR = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "lp-simplex-worker");
        t.setDaemon(true);
        return t;
    });

    private DirectHpcBenchmarkMain() {}

    public static void main(String[] args) throws Exception {
        Files.createDirectories(OUT.getParent());
        nativeOk = YishapeHpc.isNativeRuntimeAvailable();
        System.out.println("META_JSON {\"java\":\"" + Runtime.version() + "\",\"hpcNative\":" + nativeOk + "}");

        List<Row> rows = new ArrayList<>();

        // ===== LINALG =====
        benchGemm(rows);
        benchSvd(rows);
        benchQr(rows);
        benchCholesky(rows);
        benchLu(rows);
        benchSolve(rows);
        benchInverse(rows);
        benchEigenSymmetric(rows);
        benchEigenGeneral(rows);

        // ===== OPTIMIZE =====
        benchOptimizers(rows);

        // ===== LP =====
        benchLpGrid(rows);
        benchLpSmall(rows);

        // ===== STATS / SIGNAL / VECTOR =====
        benchStats(rows);
        benchSignal(rows);
        benchVector(rows);

        writeCsv(rows);
        System.out.println("WROTE " + OUT.toAbsolutePath() + " (" + rows.size() + " rows)");
    }

    // ============ Infrastructure ============

    private static void setHpcApiMode(boolean enable) {
        if (enable) HpcSwitch.enable(); else HpcSwitch.disable();
        System.setProperty(HpcConfig.PROP_ENABLE, Boolean.toString(enable));
        if (enable) {
            System.setProperty(HpcConfig.PROP_GEMM_MIN_FLOPS, "0");
            System.setProperty(HpcConfig.PROP_CHOLESKY_MIN_DIM, "32");
            System.setProperty(HpcConfig.PROP_SVD_MIN_TOTAL_ELEMENTS, "1");
            System.setProperty(HpcConfig.PROP_SOLVE_MIN_DIM, "32");
            System.setProperty(HpcConfig.PROP_LU_MIN_DIM, "32");
            System.setProperty(HpcConfig.PROP_EIGEN_MIN_DIM, "32");
            System.setProperty(HpcConfig.PROP_INVERSE_MIN_DIM, "32");
            System.setProperty(HpcConfig.PROP_QR_MIN_TOTAL_ELEMENTS, "1");
        }
    }

    private static double benchMs(Runnable op, int runs) {
        for (int i = 0; i < WARMUP; i++) op.run();
        long[] ns = new long[runs];
        for (int i = 0; i < runs; i++) {
            long t0 = System.nanoTime();
            op.run();
            ns[i] = System.nanoTime() - t0;
        }
        Arrays.sort(ns);
        return ns[ns.length / 2] / 1_000_000.0;
    }

    /** Frobenius norm as numerical fingerprint for cross-validation */
    private static double frobeniusNorm(IMatrix<Double> m) {
        double sum = 0.0;
        for (int i = 0; i < m.rows(); i++)
            for (int j = 0; j < m.cols(); j++) {
                double v = m.get(i, j);
                sum += v * v;
            }
        return Math.sqrt(sum);
    }

    private static double vectorSum(IVector<Double> v) {
        double s = 0;
        for (int i = 0; i < v.size(); i++) s += v.get(i);
        return s;
    }

    private static double vectorNorm(IVector<Double> v) {
        double sum = 0;
        for (int i = 0; i < v.size(); i++) { double x = v.get(i); sum += x * x; }
        return Math.sqrt(sum);
    }

    private static String checksumStr(double v) {
        return String.format(Locale.ROOT, "%.12e", v);
    }

    // Matrix helpers
    private static IMatrix<Double> spdMatrix(int n, long seed) {
        IMatrix<Double> r = Linalg.rand(n, n, seed);
        IMatrix<Double> a = r.mmul(r.transpose()).add(Linalg.eye(n).multiplyByScalar((double) n));
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                double avg = (a.get(i, j) + a.get(j, i)) / 2.0;
                a.put(i, j, avg);
                a.put(j, i, avg);
            }
        return a;
    }

    private static IMatrix<Double> symMatrix(int n, long seed) {
        IMatrix<Double> m = Linalg.rand(n, n, seed);
        return m.add(m.transpose()).multiplyByScalar(0.5);
    }

    private static IMatrix<Double> genMatrix(int n, long seed) {
        return Linalg.rand(n, n, seed);
    }

    private static IMatrix<Double> hilbertMatrix(int n) {
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                d[i][j] = 1.0 / (i + j + 1);
        return Linalg.matrix(d);
    }

    private static IVector<Double> randVector(int n, long seed) {
        Random rng = new Random(seed);
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = rng.nextDouble();
        return Linalg.vector(d);
    }

    private static double[] fill(int n, double v) { double[] a = new double[n]; Arrays.fill(a, v); return a; }

    // ---- Commons Math 4 adapters ----

    private static RealMatrix toCM4(IMatrix<Double> m) {
        return new Array2DRowRealMatrix(m.toDoubleArray(), false);
    }

    private static RealVector toCM4Vec(IVector<Double> v) {
        return new ArrayRealVector(v.toDoubleArray(), false);
    }

    private static double frobeniusNormCM4(RealMatrix m) {
        double sum = 0;
        for (double[] row : m.getData())
            for (double v : row) sum += v * v;
        return Math.sqrt(sum);
    }

    private static double vectorSumCM4(RealVector v) {
        double s = 0;
        for (double x : v.toArray()) s += x;
        return s;
    }

    private static double vectorNormCM4(double[] a) {
        double sum = 0;
        for (double x : a) sum += x * x;
        return Math.sqrt(sum);
    }

    // ============ 1. GEMM ============

    private static void benchGemm(List<Row> rows) {
        System.out.println("\n=== GEMM ===");
        for (int n : new int[]{100, 500, 1000, 1500}) {
            IMatrix<Double> a = genMatrix(n, SEED);
            IMatrix<Double> b = genMatrix(n, SEED + 1);
            int runs = (n >= 1000) ? 1 : REPEAT;

            // Pure Java
            HpcSwitch.disable();
            double msP = benchMs(() -> a.mmul(b), runs);
            IMatrix<Double> rP = a.mmul(b);
            double csP = frobeniusNorm(rP);
            double resP = frobeniusNorm(rP); // For GEMM, checksum IS the result fingerprint
            rows.add(row("linalg", "mmul", n + "x" + n, "java_pure", msP, csP, 0));

            // HPC API
            setHpcApiMode(true);
            double msA = benchMs(() -> a.mmul(b), runs);
            IMatrix<Double> rA = a.mmul(b);
            double csA = frobeniusNorm(rA);
            rows.add(row("linalg", "mmul", n + "x" + n, "java_hpc_api", msA, csA, 0));

            // HPC Direct
            if (nativeOk) {
                double[][] ad = a.toDoubleArray();
                double[][] bd = b.toDoubleArray();
                double[] msD = {0};
                double[][] rd = benchMsDirect(() -> YishapeHpc.tryMatMul(ad, bd), runs, msD);
                if (rd != null) {
                    double csD = frobeniusNorm(Linalg.matrix(rd));
                    rows.add(row("linalg", "mmul", n + "x" + n, "java_hpc_direct", msD[0], csD, 0));
                }
            }

            // Commons Math 4
            {
                RealMatrix aC = toCM4(a);
                RealMatrix bC = toCM4(b);
                double msC = benchMs(() -> aC.multiply(bC), runs);
                RealMatrix rC = aC.multiply(bC);
                double csC = frobeniusNormCM4(rC);
                rows.add(row("linalg", "mmul", n + "x" + n, "java_pure_cm4", msC, csC, 0));
            }

            System.out.printf("  GEMM %d: pure=%.1f api=%.1f%n", n, msP, msA);
        }
    }

    // ============ 2. SVD ============

    private static void benchSvd(List<Row> rows) {
        System.out.println("\n=== SVD ===");
        for (int n : new int[]{50, 100, 200, 500, 1000}) {
            IMatrix<Double> a = genMatrix(n, SEED + 2 + n);
            int runs = (n >= 500) ? 1 : REPEAT;

            // Pure Java
            HpcSwitch.disable();
            double msP = benchMs(() -> a.svd(), runs);
            var svdP = a.svd();
            IVector<Double> sP = svdP._2;
            IMatrix<Double> uP = svdP._1, vtP = svdP._3;
            double csP = vectorSum(sP) + frobeniusNorm(uP) + frobeniusNorm(vtP);
            double resP = computeSvdResidual(a, uP, sP, vtP);
            rows.add(row("linalg", "svd", n + "x" + n, "java_pure", msP, csP, resP));

            // HPC API
            setHpcApiMode(true);
            double msA = benchMs(() -> a.svd(), runs);
            var svdA = a.svd();
            IVector<Double> sA = svdA._2;
            IMatrix<Double> uA = svdA._1, vtA = svdA._3;
            double csA = vectorSum(sA) + frobeniusNorm(uA) + frobeniusNorm(vtA);
            double resA = computeSvdResidual(a, uA, sA, vtA);
            rows.add(row("linalg", "svd", n + "x" + n, "java_hpc_api", msA, csA, resA));

            // HPC Direct
            if (nativeOk) {
                double[][] ad = a.toDoubleArray();
                double[] msD = {0};
                SvdResult rD = benchMsDirect(() -> YishapeHpc.svd(ad), runs, msD);
                if (rD != null && rD.ok()) {
                    IMatrix<Double> uD = Linalg.matrix(rD.u());
                    IMatrix<Double> vtD = Linalg.matrix(rD.vt());
                    double[] svals = rD.singularValues();
                    int k = Math.min(n, n);
                    IVector<Double> sDv = Linalg.vector(Arrays.copyOf(svals, k));
                    double csD = Arrays.stream(svals).sum() + frobeniusNorm(uD) + frobeniusNorm(vtD);
                    double resD = computeSvdResidual(a, uD, sDv, vtD);
                    rows.add(row("linalg", "svd", n + "x" + n, "java_hpc_direct", msD[0], csD, resD));
                }
            }

            // Commons Math 4
            {
                RealMatrix aC = toCM4(a);
                double msC = benchMs(() -> new SingularValueDecomposition(aC), runs);
                SingularValueDecomposition svdC = new SingularValueDecomposition(aC);
                double[] svals = svdC.getSingularValues();
                RealMatrix uC = svdC.getU();
                RealMatrix vtC = svdC.getV();
                RealMatrix sC = svdC.getS();
                double csC = Arrays.stream(svals).sum() + frobeniusNormCM4(uC) + frobeniusNormCM4(vtC);
                // compute residual: ||A - U*S*Vt|| / ||A||
                double[][] sd = new double[svals.length][svals.length];
                for (int i = 0; i < svals.length; i++) sd[i][i] = svals[i];
                RealMatrix sMat = new Array2DRowRealMatrix(sd, false);
                IMatrix<Double> recon = Linalg.matrix(uC.multiply(sMat).multiply(vtC).getData());
                double resC = frobeniusNorm(a.sub(recon)) / frobeniusNorm(a);
                rows.add(row("linalg", "svd", n + "x" + n, "java_pure_cm4", msC, csC, resC));
            }

            System.out.printf("  SVD %d: pure=%.1f api=%.1f%n", n, msP, msA);
        }
    }

    private static double computeSvdResidual(IMatrix<Double> a, IMatrix<Double> u, IVector<Double> s, IMatrix<Double> vt) {
        int k = s.size();
        double[][] sd = new double[k][k];
        for (int i = 0; i < k; i++) sd[i][i] = s.get(i);
        IMatrix<Double> sm = Linalg.matrix(sd);
        IMatrix<Double> recon = u.mmul(sm).mmul(vt);
        IMatrix<Double> diff = a.sub(recon);
        return frobeniusNorm(diff) / frobeniusNorm(a);
    }

    // ============ 3. QR ============

    private static void benchQr(List<Row> rows) {
        System.out.println("\n=== QR ===");
        for (int n : new int[]{50, 100, 200, 500, 1000}) {
            IMatrix<Double> a = genMatrix(n, SEED + 3 + n);
            int runs = (n >= 500) ? 1 : REPEAT;

            HpcSwitch.disable();
            double msP = benchMs(() -> a.qr(), runs);
            var qrP = a.qr();
            IMatrix<Double> qP = qrP._1, rP = qrP._2;
            double csP = frobeniusNorm(qP) + frobeniusNorm(rP);
            double resP = frobeniusNorm(a.sub(qP.mmul(rP))) / frobeniusNorm(a);
            rows.add(row("linalg", "qr", n + "x" + n, "java_pure", msP, csP, resP));

            setHpcApiMode(true);
            double msA = benchMs(() -> a.qr(), runs);
            var qrA = a.qr();
            IMatrix<Double> qA = qrA._1, rA = qrA._2;
            double csA = frobeniusNorm(qA) + frobeniusNorm(rA);
            double resA = frobeniusNorm(a.sub(qA.mmul(rA))) / frobeniusNorm(a);
            rows.add(row("linalg", "qr", n + "x" + n, "java_hpc_api", msA, csA, resA));

            if (nativeOk) {
                double[][] ad = a.toDoubleArray();
                double[] msD = {0};
                QrResult qrD = benchMsDirect(() -> YishapeHpc.qr(ad), runs, msD);
                if (qrD != null && qrD.ok()) {
                    IMatrix<Double> qDm = Linalg.matrix(qrD.q());
                    IMatrix<Double> rDm = Linalg.matrix(qrD.r());
                    double csD = frobeniusNorm(qDm) + frobeniusNorm(rDm);
                    double resD = frobeniusNorm(a.sub(qDm.mmul(rDm))) / frobeniusNorm(a);
                    rows.add(row("linalg", "qr", n + "x" + n, "java_hpc_direct", msD[0], csD, resD));
                }
            }

            // Commons Math 4 — include getQ/getR in timing for fair comparison.
            // IMPORTANT (fixed 2026-05-15): CM4's QRDecomposition constructor lazily evaluates
            // Q and R. Without getQ()+getR() in the timed region, CM4 appears ~2x faster because
            // it only measures Householder reduction (O(n³/3)) while ours includes Q+R
            // construction (total 2n³/3). All other CM4 decompositions (SVD, Eigen, Cholesky, LU)
            // compute results eagerly in their constructors — only QR has this lazy behavior.
            {
                RealMatrix aC = toCM4(a);
                QRDecomposition[] qrHolder = new QRDecomposition[1];
                double msC = benchMs(() -> {
                    QRDecomposition qr = new QRDecomposition(aC);
                    qr.getQ(); qr.getR();
                    qrHolder[0] = qr;
                }, runs);
                QRDecomposition qrC = qrHolder[0];
                RealMatrix qC = qrC.getQ();
                RealMatrix rC = qrC.getR();
                double csC = frobeniusNormCM4(qC) + frobeniusNormCM4(rC);
                IMatrix<Double> qCM = Linalg.matrix(qC.getData());
                IMatrix<Double> rCM = Linalg.matrix(rC.getData());
                double resC = frobeniusNorm(a.sub(qCM.mmul(rCM))) / frobeniusNorm(a);
                rows.add(row("linalg", "qr", n + "x" + n, "java_pure_cm4", msC, csC, resC));
            }

            System.out.printf("  QR %d: pure=%.1f api=%.1f%n", n, msP, msA);
        }
    }

    // ============ 4. Cholesky ============

    private static void benchCholesky(List<Row> rows) {
        System.out.println("\n=== Cholesky ===");
        for (int n : new int[]{50, 100, 200, 500, 1000, 1500}) {
            IMatrix<Double> a = spdMatrix(n, SEED + 4);
            int runs = (n >= 500) ? 1 : REPEAT;

            HpcSwitch.disable();
            double msP = benchMs(() -> a.cholesky(), runs);
            IMatrix<Double> lP = a.cholesky();
            double csP = frobeniusNorm(lP);
            double resP = frobeniusNorm(a.sub(lP.mmul(lP.transpose()))) / frobeniusNorm(a);
            rows.add(row("linalg", "cholesky", n + "x" + n, "java_pure", msP, csP, resP));

            setHpcApiMode(true);
            double msA = benchMs(() -> a.cholesky(), runs);
            IMatrix<Double> lA = a.cholesky();
            double csA = frobeniusNorm(lA);
            double resA = frobeniusNorm(a.sub(lA.mmul(lA.transpose()))) / frobeniusNorm(a);
            rows.add(row("linalg", "cholesky", n + "x" + n, "java_hpc_api", msA, csA, resA));

            if (nativeOk) {
                double[][] ad = a.toDoubleArray();
                double[] msD = {0};
                CholeskyResult cholD = benchMsDirect(() -> YishapeHpc.cholesky(ad), runs, msD);
                if (cholD != null && cholD.ok()) {
                    IMatrix<Double> lD = Linalg.matrix(cholD.lLower());
                    double csD = frobeniusNorm(lD);
                    double resD = frobeniusNorm(a.sub(lD.mmul(lD.transpose()))) / frobeniusNorm(a);
                    rows.add(row("linalg", "cholesky", n + "x" + n, "java_hpc_direct", msD[0], csD, resD));
                }
            }

            // Commons Math 4
            {
                RealMatrix aC = toCM4(a);
                double msC = benchMs(() -> new CholeskyDecomposition(aC), runs);
                CholeskyDecomposition cholC = new CholeskyDecomposition(aC);
                RealMatrix lC = cholC.getL();
                double csC = frobeniusNormCM4(lC);
                IMatrix<Double> lCM = Linalg.matrix(lC.getData());
                double resC = frobeniusNorm(a.sub(lCM.mmul(lCM.transpose()))) / frobeniusNorm(a);
                rows.add(row("linalg", "cholesky", n + "x" + n, "java_pure_cm4", msC, csC, resC));
            }

            System.out.printf("  Cholesky %d: pure=%.1f api=%.1f%n", n, msP, msA);
        }
    }

    // ============ 5. LU ============

    private static void benchLu(List<Row> rows) {
        System.out.println("\n=== LU ===");
        for (int n : new int[]{50, 100, 200, 500, 1000, 1500}) {
            IMatrix<Double> a = genMatrix(n, SEED + 5 + n);
            int runs = (n >= 500) ? 1 : REPEAT;

            HpcSwitch.disable();
            double msP = benchMs(() -> a.lu(), runs);
            var luP = a.lu();
            IMatrix<Double> lP = luP._1, uP = luP._2;
            double csP = frobeniusNorm(lP) + frobeniusNorm(uP);
            double resP = frobeniusNorm(a.sub(lP.mmul(uP))) / frobeniusNorm(a);
            rows.add(row("linalg", "lu", n + "x" + n, "java_pure", msP, csP, resP));

            setHpcApiMode(true);
            double msA = benchMs(() -> a.lu(), runs);
            var luA = a.lu();
            IMatrix<Double> lA = luA._1, uA = luA._2;
            double csA = frobeniusNorm(lA) + frobeniusNorm(uA);
            double resA = frobeniusNorm(a.sub(lA.mmul(uA))) / frobeniusNorm(a);
            rows.add(row("linalg", "lu", n + "x" + n, "java_hpc_api", msA, csA, resA));

            if (nativeOk) {
                double[][] ad = a.toDoubleArray();
                double[] msD = {0};
                LuResult luD = benchMsDirect(() -> YishapeHpc.lu(ad), runs, msD);
                if (luD != null && luD.ok()) {
                    IMatrix<Double> lD = Linalg.matrix(luD.l());
                    IMatrix<Double> uD = Linalg.matrix(luD.u());
                    double csD = frobeniusNorm(lD) + frobeniusNorm(uD);
                    // Build permutation matrix from p[]
                    int[] pArr = luD.p();
                    double[][] pMat = new double[n][n];
                    for (int i = 0; i < n; i++) pMat[i][pArr[i]] = 1.0;
                    IMatrix<Double> pD = Linalg.matrix(pMat);
                    IMatrix<Double> reconD = pD.mmul(lD).mmul(uD);
                    double resD = frobeniusNorm(a.sub(reconD)) / frobeniusNorm(a);
                    rows.add(row("linalg", "lu", n + "x" + n, "java_hpc_direct", msD[0], csD, resD));
                }
            }

            // Commons Math 4
            {
                RealMatrix aC = toCM4(a);
                double msC = benchMs(() -> new LUDecomposition(aC), runs);
                LUDecomposition luC = new LUDecomposition(aC);
                RealMatrix lC = luC.getL();
                RealMatrix uC = luC.getU();
                RealMatrix pC = luC.getP();
                double csC = frobeniusNormCM4(lC) + frobeniusNormCM4(uC);
                IMatrix<Double> lCM = Linalg.matrix(lC.getData());
                IMatrix<Double> uCM = Linalg.matrix(uC.getData());
                IMatrix<Double> pCM = Linalg.matrix(pC.getData());
                double resC = frobeniusNorm(a.sub(pCM.mmul(lCM).mmul(uCM))) / frobeniusNorm(a);
                rows.add(row("linalg", "lu", n + "x" + n, "java_pure_cm4", msC, csC, resC));
            }

            System.out.printf("  LU %d: pure=%.1f api=%.1f%n", n, msP, msA);
        }
    }

    // ============ 6. Solve ============

    private static void benchSolve(List<Row> rows) {
        System.out.println("\n=== Solve ===");
        for (int n : new int[]{100, 500, 1000, 1500}) {
            IMatrix<Double> a = spdMatrix(n, SEED + 6);
            IVector<Double> b = randVector(n, SEED + 7);
            int runs = (n >= 1000) ? 1 : REPEAT;

            HpcSwitch.disable();
            double msP = benchMs(() -> a.solve(b), runs);
            IVector<Double> xP = a.solve(b);
            double csP = vectorSum(xP);
            double resP = vectorNorm(a.mmul(xP).sub(b)) / vectorNorm(b);
            rows.add(row("linalg", "solve", "n=" + n, "java_pure", msP, csP, resP));

            setHpcApiMode(true);
            double msA = benchMs(() -> a.solve(b), runs);
            IVector<Double> xA = a.solve(b);
            double csA = vectorSum(xA);
            double resA = vectorNorm(a.mmul(xA).sub(b)) / vectorNorm(b);
            rows.add(row("linalg", "solve", "n=" + n, "java_hpc_api", msA, csA, resA));

            if (nativeOk) {
                double[][] ad = a.toDoubleArray();
                double[] bd = b.toDoubleArray();
                double[] msD = {0};
                DenseSolveResult sD = benchMsDirect(() -> YishapeHpc.solveSquare(ad, bd), runs, msD);
                if (sD != null && sD.ok()) {
                    IVector<Double> xD = Linalg.vector(sD.x());
                    double csD = vectorSum(xD);
                    double resD = vectorNorm(a.mmul(xD).sub(b)) / vectorNorm(b);
                    rows.add(row("linalg", "solve", "n=" + n, "java_hpc_direct", msD[0], csD, resD));
                }
            }

            // Commons Math 4 (solve via LU)
            {
                RealMatrix aC = toCM4(a);
                RealVector bC = toCM4Vec(b);
                double msC = benchMs(() -> new LUDecomposition(aC).getSolver().solve(bC), runs);
                RealVector xC = new LUDecomposition(aC).getSolver().solve(bC);
                double csC = vectorSumCM4(xC);
                IVector<Double> xCV = Linalg.vector(xC.toArray());
                double resC = vectorNorm(a.mmul(xCV).sub(b)) / vectorNorm(b);
                rows.add(row("linalg", "solve", "n=" + n, "java_pure_cm4", msC, csC, resC));
            }

            System.out.printf("  Solve %d: pure=%.1f api=%.1f%n", n, msP, msA);
        }
    }

    // ============ 7. Inverse ============

    private static void benchInverse(List<Row> rows) {
        System.out.println("\n=== Inverse ===");
        for (int n : new int[]{50, 100, 200, 500}) {
            IMatrix<Double> a = spdMatrix(n, SEED + 8);
            int runs = (n >= 200) ? 1 : REPEAT;

            HpcSwitch.disable();
            double msP = benchMs(() -> a.inv(), runs);
            IMatrix<Double> iP = a.inv();
            double csP = frobeniusNorm(iP);
            double resP = frobeniusNorm(a.mmul(iP).sub(Linalg.eye(n))) / n;
            rows.add(row("linalg", "inverse", n + "x" + n, "java_pure", msP, csP, resP));

            setHpcApiMode(true);
            double msA = benchMs(() -> a.inv(), runs);
            IMatrix<Double> iA = a.inv();
            double csA = frobeniusNorm(iA);
            double resA = frobeniusNorm(a.mmul(iA).sub(Linalg.eye(n))) / n;
            rows.add(row("linalg", "inverse", n + "x" + n, "java_hpc_api", msA, csA, resA));

            if (nativeOk) {
                double[][] ad = a.toDoubleArray();
                double[] msD = {0};
                InverseResult iD = benchMsDirect(() -> YishapeHpc.inverse(ad), runs, msD);
                if (iD != null && iD.ok()) {
                    IMatrix<Double> iDM = Linalg.matrix(iD.inv());
                    double csD = frobeniusNorm(iDM);
                    double resD = frobeniusNorm(a.mmul(iDM).sub(Linalg.eye(n))) / n;
                    rows.add(row("linalg", "inverse", n + "x" + n, "java_hpc_direct", msD[0], csD, resD));
                }
            }

            // Commons Math 4 (inverse via LU)
            {
                RealMatrix aC = toCM4(a);
                double msC = benchMs(() -> new LUDecomposition(aC).getSolver().getInverse(), runs);
                RealMatrix iC = new LUDecomposition(aC).getSolver().getInverse();
                double csC = frobeniusNormCM4(iC);
                IMatrix<Double> iCM = Linalg.matrix(iC.getData());
                double resC = frobeniusNorm(a.mmul(iCM).sub(Linalg.eye(n))) / n;
                rows.add(row("linalg", "inverse", n + "x" + n, "java_pure_cm4", msC, csC, resC));
            }

            System.out.printf("  Inverse %d: pure=%.1f api=%.1f%n", n, msP, msA);
        }
    }

    // ============ 8. Eigen Symmetric ============

    private static void benchEigenSymmetric(List<Row> rows) {
        System.out.println("\n=== Eigen Symmetric ===");
        for (int n : new int[]{50, 100, 200, 500}) {
            IMatrix<Double> a = symMatrix(n, SEED + 10);
            int runs = (n >= 200) ? 1 : REPEAT;

            HpcSwitch.disable();
            double msP = benchMs(() -> a.eigen(), runs);
            var eP = a.eigen();
            double csP = vectorSum(eP._1);
            double resP = maxEigenpairResidual(a, eP._1, eP._2);
            rows.add(row("linalg", "eigen_symmetric", n + "x" + n, "java_pure", msP, csP, resP));

            setHpcApiMode(true);
            double msA = benchMs(() -> a.eigen(), runs);
            var eA = a.eigen();
            double csA = vectorSum(eA._1);
            double resA = maxEigenpairResidual(a, eA._1, eA._2);
            rows.add(row("linalg", "eigen_symmetric", n + "x" + n, "java_hpc_api", msA, csA, resA));

            if (nativeOk) {
                double[][] ad = a.toDoubleArray();
                double[] msD = {0};
                SymmetricEigenResult eD = benchMsDirect(() -> YishapeHpc.eigenSymmetric(ad), runs, msD);
                if (eD != null && eD.ok()) {
                    double[] wD = eD.eigenvaluesAscending();
                    double[][] vD = eD.eigenvectors();
                    double csD = Arrays.stream(wD).sum();
                    double resD = maxEigenpairResidualDirect(a, wD, vD);
                    rows.add(row("linalg", "eigen_symmetric", n + "x" + n, "java_hpc_direct", msD[0], csD, resD));
                }
            }

            // Commons Math 4
            {
                RealMatrix aC = toCM4(a);
                double msC = benchMs(() -> new EigenDecomposition(aC), runs);
                EigenDecomposition eigC = new EigenDecomposition(aC);
                double[] wC = eigC.getRealEigenvalues();
                double csC = Arrays.stream(wC).sum();
                double[][] vArr = eigC.getV().getData();
                double resC = maxEigenpairResidualDirect(a, wC, vArr);
                rows.add(row("linalg", "eigen_symmetric", n + "x" + n, "java_pure_cm4", msC, csC, resC));
            }

            System.out.printf("  EigenSym %d: pure=%.1f api=%.1f%n", n, msP, msA);
        }
    }

    // ============ 9. Eigen General ============

    private static void benchEigenGeneral(List<Row> rows) {
        System.out.println("\n=== Eigen General ===");
        for (int n : new int[]{50, 100, 200, 500}) {
            IMatrix<Double> a = genMatrix(n, SEED + 9 + n);
            int runs = (n >= 200) ? 1 : REPEAT;

            HpcSwitch.disable();
            double msP = benchMs(() -> a.eigen(), runs);
            var eP = a.eigen();
            double csP = vectorSum(eP._1);
            double resP = maxEigenpairResidual(a, eP._1, eP._2);
            rows.add(row("linalg", "eigen_general", n + "x" + n, "java_pure", msP, csP, resP));

            setHpcApiMode(true);
            double msA = benchMs(() -> a.eigen(), runs);
            var eA = a.eigen();
            double csA = vectorSum(eA._1);
            double resA = maxEigenpairResidual(a, eA._1, eA._2);
            rows.add(row("linalg", "eigen_general", n + "x" + n, "java_hpc_api", msA, csA, resA));

            if (nativeOk) {
                double[][] ad = a.toDoubleArray();
                double[] msD = {0};
                NonsymmetricEigenResult eD = benchMsDirect(() -> YishapeHpc.eigenNonsymmetric(ad), runs, msD);
                if (eD != null && eD.ok()) {
                    double[] wD = eD.eigenvaluesReal();
                    double[][] vD = eD.eigenvectorsReal();
                    double csD = Arrays.stream(wD).sum();
                    double resD = maxEigenpairResidualDirect(a, wD, vD);
                    rows.add(row("linalg", "eigen_general", n + "x" + n, "java_hpc_direct", msD[0], csD, resD));
                }
            }

            // Commons Math 4 (non-symmetric: complex eigenvalues possible)
            {
                RealMatrix aC = toCM4(a);
                double msC = benchMs(() -> new EigenDecomposition(aC), runs);
                EigenDecomposition eigC = new EigenDecomposition(aC);
                double[] wC = eigC.getRealEigenvalues();
                double[] imagW = eigC.getImagEigenvalues();
                double csC = Arrays.stream(wC).sum();
                boolean hasComplex = Arrays.stream(imagW).anyMatch(v -> Math.abs(v) > 1e-10);
                double[][] vArr = eigC.getV().getData();
                double resC = hasComplex ? Double.NaN : maxEigenpairResidualDirect(a, wC, vArr);
                rows.add(row("linalg", "eigen_general", n + "x" + n, "java_pure_cm4", msC, csC, resC));
            }

            System.out.printf("  EigenGen %d: pure=%.1f api=%.1f%n", n, msP, msA);
        }
    }

    // ============ 10. Optimizers ============

    private static void benchOptimizers(List<Row> rows) {
        System.out.println("\n=== Optimizers ===");
        IVector<Double> init2d = Linalg.vector(new double[]{0.0, 0.0});

        // Pure Java - RereLBFGS
        HpcSwitch.disable();
        IOptimizer rereP = new RereLBFGS(10, 1e-12, 200);
        rereP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        double msRP = benchMs(() -> rereP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD), REPEAT);
        OptResult rResP = rereP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        rows.add(row("optimize", "lbfgs_rosenbrock", "2d", "java_pure_rerelbfgs", msRP, rResP.getOptimalValue(), 0));

        // Pure Java - RustLBFGS (goes through gosh-lbfgs if HPC available)
        IOptimizer rustP = new RustLBFGS(10, 1e-12, 200);
        rustP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        double msRustP = benchMs(() -> rustP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD), REPEAT);
        OptResult rustResP = rustP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        rows.add(row("optimize", "lbfgs_rosenbrock", "2d", "java_pure_rustlbfgs", msRustP, rustResP.getOptimalValue(), 0));

        // CG
        IOptimizer cgP = new RereConjugateGradient(1e-12, 200, 0.5);
        cgP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        double msCgP = benchMs(() -> cgP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD), REPEAT);
        OptResult cgResP = cgP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        rows.add(row("optimize", "cg_rosenbrock", "2d", "java_pure", msCgP, cgResP.getOptimalValue(), 0));

        // OWLQN
        IOptimizer owlP = new RustOWLQN(10, 1e-12, 200, 0.0);
        owlP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        double msOwlP = benchMs(() -> owlP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD), REPEAT);
        OptResult owlResP = owlP.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        rows.add(row("optimize", "owlqn_rosenbrock", "2d", "java_pure", msOwlP, owlResP.getOptimalValue(), 0));

        // HPC mode - RustLBFGS (should use gosh-lbfgs via YishapeHpc)
        setHpcApiMode(true);
        IOptimizer rustH = new RustLBFGS(10, 1e-12, 200);
        rustH.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        double msRustH = benchMs(() -> rustH.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD), REPEAT);
        OptResult rustResH = rustH.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        rows.add(row("optimize", "lbfgs_rosenbrock", "2d", "java_hpc_rustlbfgs", msRustH, rustResH.getOptimalValue(), 0));

        // HPC Direct L-BFGS
        if (nativeOk) {
            double[] x0 = {0.0, 0.0};
            double[] msD = {0};
            LbfgsResult lbfgsD = benchMsDirect(() -> {
                return YishapeHpc.lbfgsMinimize(x0, 10, 1e-12, 200,
                    (x, g) -> {
                        double x0v = x[0], x1v = x[1];
                        double a = 1.0 - x0v, b = x1v - x0v * x0v;
                        if (g != null) {
                            g[0] = -2.0 * a - 400.0 * x0v * b;
                            g[1] = 200.0 * b;
                        }
                        return a * a + 100.0 * b * b;
                    });
            }, REPEAT, msD);
            if (lbfgsD != null && lbfgsD.ok()) {
                rows.add(row("optimize", "lbfgs_rosenbrock", "2d", "java_hpc_direct_lbfgs", msD[0], lbfgsD.fx(), 0));
            }
        }

        // Extended Rosenbrock and Quadratic for key dims
        for (int dim : new int[]{10, 100, 500}) {
            benchOptimizerExtended(rows, dim);
        }
    }

    private static void benchOptimizerExtended(List<Row> rows, int dim) {
        double[] initArr = new double[dim];
        for (int i = 0; i < dim; i++) initArr[i] = (i % 2 == 0) ? -1.5 : 0.5;
        IVector<Double> initX = Linalg.vector(initArr);
        IObjectiveFunction obj = extendedRosenbrockObj(dim);
        IGradientFunction grad = extendedRosenbrockGrad(dim);
        int maxIter = Math.max(10000, dim * 50);
        int runs = (dim >= 500) ? 1 : REPEAT;

        // RereLBFGS Pure
        HpcSwitch.disable();
        IOptimizer rere = new RereLBFGS(10, 1e-12, maxIter);
        rere.optimize(initX, obj, grad);
        double msR = benchMs(() -> rere.optimize(initX, obj, grad), runs);
        OptResult resR = rere.optimize(initX, obj, grad);
        rows.add(row("optimize", "lbfgs_ext_rosenbrock", "d=" + dim, "java_pure_rerelbfgs", msR, resR.getOptimalValue(), 0));

        // RustLBFGS HPC
        setHpcApiMode(true);
        IOptimizer rust = new RustLBFGS(10, 1e-12, maxIter);
        rust.optimize(initX, obj, grad);
        double msRu = benchMs(() -> rust.optimize(initX, obj, grad), runs);
        OptResult resRu = rust.optimize(initX, obj, grad);
        rows.add(row("optimize", "lbfgs_ext_rosenbrock", "d=" + dim, "java_hpc_rustlbfgs", msRu, resRu.getOptimalValue(), 0));
    }

    // ============ 11. LP Grid ============

    private static void benchLpGrid(List<Row> rows) {
        System.out.println("\n=== LP Grid ===");
        int[] ns = {250, 500, 1000, 2000, 3500};
        int[] mMults = {1, 2};

        // HPC Direct HiGHS
        if (nativeOk) {
            for (int mult : mMults) {
                for (int n : ns) {
                    int mLe = n * mult;
                    LpProblem p = LpProblem.build(n, mLe, 424242L + n * 31L + mult);
                    double[] msD = {0};
                    LpNonnegativeResult r = benchMsDirect(() ->
                        YishapeHpc.lpNonnegative(p.c, p.aUb, p.bUb, null, null), REPEAT, msD);
                    if (r != null && r.ok()) {
                        rows.add(row("linprog", "lp_grid", "n=" + n + "_m=" + mLe, "java_hpc_direct_highs", msD[0], r.objective(), 0));
                    }
                }
            }
        }

        // RereSimplex — with adaptive timeout instead of hard coefficient skip
        HpcSwitch.disable();
        for (int mult : mMults) {
            for (int n : ns) {
                int mLe = n * mult;
                LpProblem p = LpProblem.build(n, mLe, 424242L + n * 31L + mult);
                IVector<Double> cj = Linalg.vector(p.c);
                IMatrix<Double> aUb = Linalg.matrix(p.aUb);
                IVector<Double> bUb = Linalg.vector(p.bUb);

                // Adaptive timeout based on estimated tableau memory footprint
                long estTableauDoubles = (long)(mLe + 1) * (n + mLe + 1);
                long estTableauBytes = estTableauDoubles * 8 + 16_000_000L; // doubles + overhead
                int timeoutSec;
                if (estTableauBytes < 50_000_000L)      timeoutSec = 30;
                else if (estTableauBytes < 200_000_000L) timeoutSec = 60;
                else if (estTableauBytes < 500_000_000L) timeoutSec = 120;
                else                                      timeoutSec = 180;

                // Skip only truly massive cases (>1 GB estimated tableau)
                if (estTableauBytes > 1_000_000_000L) {
                    rows.add(row("linprog", "lp_grid", "n=" + n + "_m=" + mLe,
                        "rere_simplex", -1, Double.NaN, 0));
                    System.out.printf("  LP RereSimplex SKIP (tableau>1GB) n=%d m=%d%n", n, mLe);
                    continue;
                }

                Callable<OptResult> solveTask = () -> {
                    OptResult r = new RereSimplexLinProgSolver().solve(cj, aUb, bUb, null, null);
                    if (!r.isConverged()) throw new RuntimeException("not converged");
                    return r;
                };

                try {
                    double ms = benchMs(() -> {
                        try {
                            Future<OptResult> f = LP_EXECUTOR.submit(solveTask);
                            f.get(timeoutSec, TimeUnit.SECONDS);
                        } catch (TimeoutException e) {
                            throw new RuntimeException("lp_timeout", e);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }, REPEAT);
                    Future<OptResult> fCheck = LP_EXECUTOR.submit(solveTask);
                    OptResult res = fCheck.get(timeoutSec, TimeUnit.SECONDS);
                    rows.add(row("linprog", "lp_grid", "n=" + n + "_m=" + mLe,
                        "rere_simplex", ms, res.getOptimalValue(), 0));
                    System.out.printf("  LP RereSimplex n=%d m=%d: %.1f ms%n", n, mLe, ms);
                } catch (TimeoutException e) {
                    rows.add(row("linprog", "lp_grid", "n=" + n + "_m=" + mLe,
                        "rere_simplex", -1, Double.NaN, 0));
                    System.out.printf("  LP RereSimplex TIMEOUT n=%d m=%d (>%ds)%n", n, mLe, timeoutSec);
                } catch (Exception e) {
                    rows.add(row("linprog", "lp_grid", "n=" + n + "_m=" + mLe,
                        "rere_simplex", -1, Double.NaN, 0));
                    System.out.printf("  LP RereSimplex FAIL n=%d m=%d: %s%n", n, mLe,
                        e.getCause() != null ? e.getCause().getMessage() : e.getMessage());
                }
            }
        }
    }

    private static void benchLpSmall(List<Row> rows) {
        System.out.println("\n=== LP Small ===");
        for (int n : new int[]{80, 200}) {
            double[] c = new double[n];
            double[][] aEq = new double[1][n];
            for (int j = 0; j < n; j++) { c[j] = (j * 17 + 41) % 100 / 100.0 + 0.01; aEq[0][j] = 1.0; }
            double[] bEq = {1.0};

            // Java HiGHS Direct
            if (nativeOk) {
                double[] msD = {0};
                LpNonnegativeResult r = benchMsDirect(() ->
                    YishapeHpc.lpNonnegative(c, null, null, aEq, bEq), REPEAT, msD);
                if (r != null && r.ok()) {
                    rows.add(row("linprog", "lp_nonnegative_simplex", "n=" + n, "java_hpc_direct_highs", msD[0], r.objective(), 0));
                }
            }

            // RereSimplex
            HpcSwitch.disable();
            IVector<Double> cj = Linalg.vector(c);
            IMatrix<Double> aEqM = Linalg.matrix(aEq);
            IVector<Double> bEqV = Linalg.vector(bEq);
            double ms = benchMs(() -> {
                OptResult r = new RereSimplexLinProgSolver().solve(cj, null, null, aEqM, bEqV);
                if (!r.isConverged()) throw new RuntimeException("not converged");
            }, REPEAT);
            OptResult res = new RereSimplexLinProgSolver().solve(cj, null, null, aEqM, bEqV);
            rows.add(row("linprog", "lp_nonnegative_simplex", "n=" + n, "rere_simplex", ms, res.getOptimalValue(), 0));
        }
    }

    // ============ 12. Stats / Signal / Vector (no HPC, just pure Java + correctness) ============

    private static void benchStats(List<Row> rows) {
        System.out.println("\n=== Stats ===");
        HpcSwitch.disable();
        for (int n : new int[]{10000, 100000, 1000000}) {
            NormalDistribution norm = Stats.norm();
            int runs = (n >= 1000000) ? 1 : REPEAT;
            final int nn = n;

            double msPdf = benchMs(() -> { for (int i = 0; i < nn; i++) norm.pdf(i * 0.001); }, runs);
            double pdfCheck = 0; for (int i = 0; i < Math.min(nn, 1000); i++) pdfCheck += norm.pdf(i * 0.001);
            rows.add(row("stats", "normal_pdf", String.valueOf(n), "java_pure", msPdf, pdfCheck, 0));

            double msSample = benchMs(() -> norm.sample(nn), runs);
            rows.add(row("stats", "normal_sample", String.valueOf(n), "java_pure", msSample, 0, 0));
        }
        for (int n : new int[]{10000, 100000}) {
            GammaDistribution gamma = Stats.gamma(2, 1);
            int runs = (n >= 100000) ? 1 : REPEAT;
            double ms = benchMs(() -> gamma.sample(n), runs);
            rows.add(row("stats", "gamma_sample", String.valueOf(n), "java_pure", ms, 0, 0));
        }
    }

    private static void benchSignal(List<Row> rows) {
        System.out.println("\n=== Signal ===");
        HpcSwitch.disable();
        for (int n : new int[]{1024, 4096, 16384, 65536}) {
            Complex[] signal = new Complex[n];
            for (int i = 0; i < n; i++) signal[i] = new Complex(Math.sin(2 * Math.PI * i / n), 0);
            int runs = (n >= 65536) ? 1 : REPEAT;
            double ms = benchMs(() -> RereFFT.fft(signal), runs);
            Complex[] result = RereFFT.fft(signal);
            double cs = 0; for (Complex c : result) cs += c.getReal() + c.getImaginary();
            rows.add(row("signal", "fft", String.valueOf(n), "java_pure", ms, cs, 0));
        }
        for (int n : new int[]{1024, 4096, 16384}) {
            double[] data = new double[n];
            for (int i = 0; i < n; i++) data[i] = Math.sin(2 * Math.PI * i / n);
            IVector<Double> signal = Linalg.vector(data);
            int runs = (n >= 16384) ? 1 : REPEAT;
            double ms = benchMs(() -> RereDCT.dct2(signal), runs);
            IVector<Double> result = RereDCT.dct2(signal);
            rows.add(row("signal", "dct", String.valueOf(n), "java_pure", ms, vectorSum(result), 0));
        }
    }

    private static void benchVector(List<Row> rows) {
        System.out.println("\n=== Vector ===");
        HpcSwitch.disable();
        DoubleVectorComputer computer = new DoubleVectorComputer();
        Random rng = new Random(SEED);
        for (int n : new int[]{10000, 100000, 1000000, 10000000}) {
            double[] a = new double[n], b = new double[n];
            for (int i = 0; i < n; i++) { a[i] = rng.nextDouble(); b[i] = rng.nextDouble(); }
            int runs = (n >= 1000000) ? 1 : REPEAT;
            double msAdd = benchMs(() -> computer.binaryOperate(a, b, BinaryOperation.ADD), runs);
            double csAdd = Arrays.stream(a).sum();
            rows.add(row("compute", "vector_add", String.valueOf(n), "java_pure", msAdd, csAdd, 0));
        }
    }

    // ============ Infrastructure helpers ============

    @FunctionalInterface
    interface DirectOp<T> { T call(); }

    private static <T> T benchMsDirect(DirectOp<T> op, int runs, double[] outMs) {
        for (int i = 0; i < WARMUP; i++) { try { op.call(); } catch (Exception e) {} }
        long[] ns = new long[runs];
        T lastResult = null;
        for (int i = 0; i < runs; i++) {
            long t0 = System.nanoTime();
            lastResult = op.call();
            ns[i] = System.nanoTime() - t0;
        }
        Arrays.sort(ns);
        outMs[0] = ns[ns.length / 2] / 1_000_000.0;
        return lastResult;
    }

    private static double maxEigenpairResidual(IMatrix<Double> a, IVector<Double> w, IMatrix<Double> v) {
        int n = w.size();
        double maxRes = 0;
        for (int j = 0; j < Math.min(n, 10); j++) {
            double[] vj = new double[n];
            for (int i = 0; i < n; i++) vj[i] = v.get(i, j);
            IVector<Double> vjv = Linalg.vector(vj);
            IVector<Double> av = a.mmul(vjv);
            double lambda = w.get(j);
            // Use a separate copy: multiplyByScalar mutates backing array in-place,
            // and vjv shares the same vj backing, so scaling vj would corrupt vjv.
            double[] vjLambda = new double[n];
            for (int i = 0; i < n; i++) vjLambda[i] = vj[i] * lambda;
            double res = vectorNorm(av.sub(Linalg.vector(vjLambda)));
            if (res > maxRes) maxRes = res;
        }
        return maxRes;
    }

    private static double maxEigenpairResidualDirect(IMatrix<Double> a, double[] w, double[][] v) {
        int n = w.length;
        double maxRes = 0;
        for (int j = 0; j < Math.min(n, 10); j++) {
            double[] vj = new double[n];
            for (int i = 0; i < n; i++) vj[i] = v[i][j];
            IVector<Double> vjv = Linalg.vector(vj);
            IVector<Double> av = a.mmul(vjv);
            double[] vjLambda = new double[n];
            for (int i = 0; i < n; i++) vjLambda[i] = vj[i] * w[j];
            double res = vectorNorm(av.sub(Linalg.vector(vjLambda)));
            if (res > maxRes) maxRes = res;
        }
        return maxRes;
    }

    private static Row row(String mod, String op, String size, String backend,
                           double ms, double checksum, double residual) {
        return new Row("java", mod, op, size, backend, ms, checksum, residual);
    }

    // ============ Optimizer problem definitions ============

    private static IObjectiveFunction extendedRosenbrockObj(int dim) {
        int pairs = dim / 2;
        return x -> {
            double sum = 0.0;
            for (int p = 0; p < pairs; p++) {
                double v1 = x.get(2 * p), v2 = x.get(2 * p + 1);
                double a = 1.0 - v1, b = v2 - v1 * v1;
                sum += a * a + 100.0 * b * b;
            }
            return sum;
        };
    }

    private static IGradientFunction extendedRosenbrockGrad(int dim) {
        int pairs = dim / 2;
        return x -> {
            double[] g = new double[dim];
            for (int p = 0; p < pairs; p++) {
                double v1 = x.get(2 * p), v2 = x.get(2 * p + 1);
                int i = 2 * p;
                g[i] = -2.0 * (1.0 - v1) - 400.0 * v1 * (v2 - v1 * v1);
                g[i + 1] = 200.0 * (v2 - v1 * v1);
            }
            return Linalg.vector(g);
        };
    }

    static final IObjectiveFunction ROSENBROCK_OBJ = x -> {
        double x0 = x.get(0), x1 = x.get(1);
        double a = 1.0 - x0, b = x1 - x0 * x0;
        return a * a + 100.0 * b * b;
    };

    static final IGradientFunction ROSENBROCK_GRAD = x -> {
        double x0 = x.get(0), x1 = x.get(1);
        return Linalg.vector(new double[]{
            -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
            200.0 * (x1 - x0 * x0)
        });
    };

    // ============ LP helpers ============

    static final class LpProblem {
        final double[] c, bUb;
        final double[][] aUb;

        LpProblem(double[] c, double[][] aUb, double[] bUb) { this.c = c; this.aUb = aUb; this.bUb = bUb; }

        static LpProblem build(int n, int mLe, long seed) {
            double[] c = new double[n];
            for (int j = 0; j < n; j++) c[j] = 1.0 + (j % 11) * 0.003;
            double[][] aUb = new double[mLe][n];
            double[] bUb = new double[mLe];
            for (int i = 0; i < n; i++) { aUb[i][i] = -1.0; bUb[i] = -1.0; }
            Random rng = new Random(seed);
            for (int i = n; i < mLe; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) { double v = rng.nextDouble(); aUb[i][j] = v; sum += v; }
                bUb[i] = sum * 25.0 + 2.0;
            }
            return new LpProblem(c, aUb, bUb);
        }
    }

    record Row(String suite, String module, String operation, String size, String backend,
               double ms, double checksum, double residual) {}

    private static void writeCsv(List<Row> rows) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("suite,module,operation,size,backend,ms,checksum,residual\n");
        for (Row r : rows) {
            sb.append(r.suite).append(',').append(r.module).append(',').append(r.operation).append(',')
              .append(r.size).append(',').append(r.backend).append(',')
              .append(String.format(Locale.ROOT, "%.6f", r.ms)).append(',')
              .append(Double.isNaN(r.checksum) ? "" : String.format(Locale.ROOT, "%.12e", r.checksum)).append(',')
              .append(r.residual == 0 ? "" : String.format(Locale.ROOT, "%.6e", r.residual)).append('\n');
        }
        Files.writeString(OUT, sb.toString(), StandardCharsets.UTF_8,
            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }
}
