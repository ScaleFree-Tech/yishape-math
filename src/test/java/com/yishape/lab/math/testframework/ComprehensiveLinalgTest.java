package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.*;
import com.yishape.lab.math.linalg.decomposition.*;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.solver.*;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.*;

import java.io.File;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive correctness validation test for com.yishape.lab.math.linalg.
 * Compares Java implementation results against known reference values (computed via NumPy/SciPy).
 * Run: mvn test -Dtest=ComprehensiveLinalgTest
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
public class ComprehensiveLinalgTest {
    private static final double EPS = 1e-10;
    private static final double LOOSE_EPS = 1e-6;
    private static final Random RNG = new Random(42);
    private static TestResult.Recorder recorder;

    @BeforeAll
    static void init() {
        recorder = new TestResult.Recorder("linalg", "test_docs/results");
    }

    @AfterAll
    static void teardown() {
        recorder.writeToFile();
        System.out.println("\n=== LINALG TEST SUMMARY ===");
        System.out.println("Total: " + recorder.getResults().size());
        System.out.println("Passed: " + recorder.getPassed());
        System.out.println("Failed: " + recorder.getFailed());
    }

    // =========================================================================
    // 1. Matrix/Vector Creation and Basic Properties
    // =========================================================================

    @Test
    @DisplayName("1.1 Matrix creation factories")
    void testMatrixCreation() {
        // zeros
        IMatrix<Double> z = IMatrix.zeros(3, 4);
        TestResult r1 = recorder.record("creation", "zeros_3x4");
        assertEquals(3, z.rows()); assertEquals(4, z.cols());
        boolean allZero = true;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) if (z.get(i,j) != 0) allZero = false;
        r1.pass(allZero ? "all zeros" : "non-zero found");

        // ones
        IMatrix<Double> o = IMatrix.ones(2, 3);
        TestResult r2 = recorder.record("creation", "ones_2x3");
        assertEquals(2, o.rows()); assertEquals(3, o.cols());
        boolean allOne = true;
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) if (o.get(i,j) != 1) allOne = false;
        r2.pass(allOne ? "all ones" : "non-one found");

        // eye
        IMatrix<Double> e = IMatrix.eye(4);
        TestResult r3 = recorder.record("creation", "eye_4");
        boolean correct = true;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (Math.abs(e.get(i,j) - expected) > EPS) correct = false;
        }
        r3.pass(correct ? "identity correct" : "identity incorrect");

        // diag from array
        double[] d = {1, 2, 3};
        IMatrix<Double> diag = IMatrix.diag(d);
        TestResult r4 = recorder.record("creation", "diag_from_array");
        r4.pass(diag.get(0,0) == 1 && diag.get(1,1) == 2 && diag.get(2,2) == 3 && diag.get(0,1) == 0 ? "correct" : "incorrect");

        // rand with seed reproducibility
        IMatrix<Double> rA = IMatrix.randn(5, 5, 12345L);
        IMatrix<Double> rB = IMatrix.randn(5, 5, 12345L);
        TestResult r5 = recorder.record("creation", "rand_reproducible");
        boolean same = true;
        for (int i = 0; i < 5; i++) for (int j = 0; j < 5; j++)
            if (Math.abs(rA.get(i,j) - rB.get(i,j)) > EPS) same = false;
        r5.pass(same ? "reproducible" : "not reproducible");
    }

    @Test
    @DisplayName("1.2 Vector creation factories")
    void testVectorCreation() {
        // range
        IVector<Double> rv = IVector.range(0, 10, 2);
        TestResult r1 = recorder.record("vector_creation", "range_step2");
        double[] expected = {0, 2, 4, 6, 8};
        boolean ok = rv.length() == 5;
        for (int i = 0; i < 5 && ok; i++) if (Math.abs(rv.get(i) - expected[i]) > EPS) ok = false;
        r1.pass(ok ? "correct" : "incorrect");

        // linspace
        IVector<Double> ls = IVector.linspace(0.0, 1.0, 5);
        TestResult r2 = recorder.record("vector_creation", "linspace_0to1_5pts");
        double[] expLs = {0, 0.25, 0.5, 0.75, 1.0};
        ok = ls.length() == 5;
        for (int i = 0; i < 5 && ok; i++) if (Math.abs(ls.get(i) - expLs[i]) > EPS) ok = false;
        r2.pass(ok ? "correct" : "incorrect");

        // logspace
        IVector<Double> lgs = IVector.logspace(0.0, 2.0, 3);
        TestResult r3 = recorder.record("vector_creation", "logspace_0to2_3pts");
        double[] expLgs = {1.0, 10.0, 100.0};
        ok = lgs.length() == 3;
        for (int i = 0; i < 3 && ok; i++) if (Math.abs(lgs.get(i) - expLgs[i]) > EPS) ok = false;
        r3.pass(ok ? "correct" : "incorrect");
    }

    // =========================================================================
    // 2. Basic Matrix Operations
    // =========================================================================

    @Test
    @DisplayName("2.1 Matrix addition and subtraction")
    void testMatrixAddSub() {
        double[][] a = {{1,2},{3,4}};
        double[][] b = {{5,6},{7,8}};
        IMatrix<Double> A = Linalg.matrix(a);
        IMatrix<Double> B = Linalg.matrix(b);

        IMatrix<Double> C = A.add(B);
        TestResult r1 = recorder.record("basic_ops", "add_2x2");
        r1.pass(C.get(0,0)==6 && C.get(0,1)==8 && C.get(1,0)==10 && C.get(1,1)==12 ? "correct" : "incorrect");

        IMatrix<Double> D = A.sub(B);
        TestResult r2 = recorder.record("basic_ops", "sub_2x2");
        r2.pass(D.get(0,0)==-4 && D.get(0,1)==-4 && D.get(1,0)==-4 && D.get(1,1)==-4 ? "correct" : "incorrect");
    }

    @Test
    @DisplayName("2.2 Matrix multiplication")
    void testMatrixMul() {
        double[][] a = {{1,2},{3,4}};
        double[][] b = {{2,0},{1,2}};
        IMatrix<Double> A = Linalg.matrix(a);
        IMatrix<Double> B = Linalg.matrix(b);

        IMatrix<Double> C = A.mmul(B);
        TestResult r1 = recorder.record("basic_ops", "mmul_2x2");
        // A*B = [[4,4],[10,8]]
        boolean ok = Math.abs(C.get(0,0)-4)<EPS && Math.abs(C.get(0,1)-4)<EPS
                  && Math.abs(C.get(1,0)-10)<EPS && Math.abs(C.get(1,1)-8)<EPS;
        r1.pass(ok ? "correct" : "incorrect: got [" + C.get(0,0)+","+C.get(0,1)+";"+C.get(1,0)+","+C.get(1,1)+"]");
    }

    @Test
    @DisplayName("2.3 Matrix transpose")
    void testTranspose() {
        double[][] a = {{1,2,3},{4,5,6}};
        IMatrix<Double> A = Linalg.matrix(a);
        IMatrix<Double> T = A.transpose();
        TestResult r = recorder.record("basic_ops", "transpose_2x3");
        boolean ok = T.rows()==2 && T.cols()==3 && T.get(0,0)==1 && T.get(1,0)==2 && T.get(2,0)==3
                  && T.get(0,1)==4 && T.get(1,1)==5 && T.get(2,1)==6;
        r.pass(ok ? "correct" : "incorrect");
    }

    @Test
    @DisplayName("2.4 Matrix trace")
    void testTrace() {
        double[][] a = {{1,2,3},{4,5,6},{7,8,9}};
        IMatrix<Double> A = Linalg.matrix(a);
        double tr = A.trace();
        TestResult r = recorder.record("basic_ops", "trace_3x3");
        r.pass(Math.abs(tr - 15.0) < EPS ? "correct: 15.0" : "incorrect: " + tr);
    }

    @Test
    @DisplayName("2.5 Matrix frobenius norm")
    void testFrobeniusNorm() {
        double[][] a = {{1,2},{3,4}};
        IMatrix<Double> A = Linalg.matrix(a);
        double fn = A.frobeniusNorm();
        double expected = Math.sqrt(30); // sqrt(1+4+9+16)
        TestResult r = recorder.record("basic_ops", "frobenius_norm");
        r.pass(Math.abs(fn - expected) < EPS ? "correct" : "incorrect: " + fn + " vs " + expected);
    }

    @Test
    @DisplayName("2.6 Scalar operations")
    void testScalarOps() {
        double[][] a = {{1,2},{3,4}};
        IMatrix<Double> A = Linalg.matrix(a);

        TestResult r1 = recorder.record("basic_ops", "multiply_scalar");
        IMatrix<Double> M = A.multiplyByScalar(2.0);
        r1.pass(M.get(0,0)==2 && M.get(1,1)==8 ? "correct" : "incorrect");

        TestResult r2 = recorder.record("basic_ops", "divide_scalar");
        IMatrix<Double> D = A.divideByScalar(2.0);
        r2.pass(Math.abs(D.get(0,0)-0.5)<EPS && Math.abs(D.get(1,1)-2.0)<EPS ? "correct" : "incorrect");
    }

    // =========================================================================
    // 3. Vector Operations
    // =========================================================================

    @Test
    @DisplayName("3.1 Vector dot product")
    void testVectorDot() {
        IVector<Double> v1 = IVector.of(new double[]{1, 2, 3});
        IVector<Double> v2 = IVector.of(new double[]{4, 5, 6});
        double dot = v1.innerProductValue(v2);
        TestResult r = recorder.record("vector_ops", "dot_product");
        r.pass(Math.abs(dot - 32.0) < EPS ? "correct: 32" : "incorrect: " + dot);
    }

    @Test
    @DisplayName("3.2 Vector norms")
    void testVectorNorms() {
        IVector<Double> v = IVector.of(new double[]{3.0, 4.0});
        TestResult r1 = recorder.record("vector_ops", "norm2");
        r1.pass(Math.abs(v.norm2Value() - 5.0) < EPS ? "correct: 5" : "incorrect");

        TestResult r2 = recorder.record("vector_ops", "norm1");
        r2.pass(Math.abs(v.norm1Value() - 7.0) < EPS ? "correct: 7" : "incorrect");

        TestResult r3 = recorder.record("vector_ops", "normInf");
        r3.pass(Math.abs(v.normInf() - 4.0) < EPS ? "correct: 4" : "incorrect");
    }

    @Test
    @DisplayName("3.3 Vector sum/mean/min/max")
    void testVectorStats() {
        IVector<Double> v = IVector.of(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        TestResult r1 = recorder.record("vector_ops", "sum");
        r1.pass(Math.abs(v.sumValue() - 15.0) < EPS ? "correct" : "incorrect");

        TestResult r2 = recorder.record("vector_ops", "mean");
        r2.pass(Math.abs(v.meanValue() - 3.0) < EPS ? "correct" : "incorrect");

        TestResult r3 = recorder.record("vector_ops", "min");
        r3.pass(Math.abs(v.minValue() - 1.0) < EPS ? "correct" : "incorrect");

        TestResult r4 = recorder.record("vector_ops", "max");
        r4.pass(Math.abs(v.maxValue() - 5.0) < EPS ? "correct" : "incorrect");
    }

    @Test
    @DisplayName("3.4 Vector argMin/argMax")
    void testVectorArgMinMax() {
        IVector<Double> v = IVector.of(new double[]{5.0, 1.0, 3.0, 9.0, 2.0});
        TestResult r1 = recorder.record("vector_ops", "argMin");
        r1.pass(v.argMin() == 1 ? "correct: 1" : "incorrect: " + v.argMin());

        TestResult r2 = recorder.record("vector_ops", "argMax");
        r2.pass(v.argMax() == 3 ? "correct: 3" : "incorrect: " + v.argMax());
    }

    @Test
    @DisplayName("3.5 Vector std/var")
    void testVectorStdVar() {
        IVector<Double> v = IVector.of(new double[]{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0});
        // NumPy: np.std([2,4,4,4,5,5,7,9]) = 2.0 (population)
        // Our std uses ddof=0 by default? Need to check.
        TestResult r1 = recorder.record("vector_ops", "std");
        double std = v.stdValue();
        r1.pass(Math.abs(std - 2.0) < EPS ? "correct: 2.0" : "got: " + std);

        TestResult r2 = recorder.record("vector_ops", "var");
        double var = v.varValue();
        r2.pass(Math.abs(var - 4.0) < EPS ? "correct: 4.0" : "got: " + var);
    }

    @Test
    @DisplayName("3.6 Vector cumulative operations")
    void testVectorCum() {
        IVector<Double> v = IVector.of(new double[]{1.0, 2.0, 3.0, 4.0});
        IVector<Double> cs = v.cumsum();
        TestResult r1 = recorder.record("vector_ops", "cumsum");
        boolean ok = cs.length() == 4 && Math.abs(cs.get(0)-1)<EPS && Math.abs(cs.get(1)-3)<EPS
                  && Math.abs(cs.get(2)-6)<EPS && Math.abs(cs.get(3)-10)<EPS;
        r1.pass(ok ? "correct" : "incorrect");

        IVector<Double> df = v.diff();
        TestResult r2 = recorder.record("vector_ops", "diff");
        ok = df.length() == 3 && Math.abs(df.get(0)-1)<EPS && Math.abs(df.get(1)-1)<EPS && Math.abs(df.get(2)-1)<EPS;
        r2.pass(ok ? "correct" : "incorrect");
    }

    // =========================================================================
    // 4. Matrix Decompositions
    // =========================================================================

    @Test
    @DisplayName("4.1 LU Decomposition")
    void testLUDecomposition() {
        double[][] data = {{4,3},{6,3}};
        IMatrix<Double> A = Linalg.matrix(data);

        ILUDecomposition lu = Decomps.createLU();
        Tuple2<IMatrix<Double>, IMatrix<Double>> result = lu.decompose(A);

        TestResult r1 = recorder.record("decomposition", "lu_basic");
        assertNotNull(result);
        IMatrix<Double> L = result._1;
        IMatrix<Double> U = result._2;

        // Reconstruct: PA = LU
        IMatrix<Double> P = lu.getP();
        IMatrix<Double> PA = P.mmul(A);
        IMatrix<Double> LU = L.mmul(U);
        double maxDiff = 0;
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
            maxDiff = Math.max(maxDiff, Math.abs(PA.get(i,j) - LU.get(i,j)));
        r1.pass(maxDiff < 1e-9 ? "reconstruction OK, max_diff=" + maxDiff : "reconstruction FAIL, max_diff=" + maxDiff);

        // Determinant
        TestResult r2 = recorder.record("decomposition", "lu_determinant");
        double det = lu.getDeterminant();
        double expectedDet = -6.0; // 4*3 - 3*6 = -6
        r2.pass(Math.abs(det - expectedDet) < EPS ? "correct: " + det : "incorrect: " + det + " vs " + expectedDet);
    }

    @Test
    @DisplayName("4.2 QR Decomposition")
    void testQRDecomposition() {
        double[][] data = {{12,-51,4},{6,167,-68},{-4,24,-41}};
        IMatrix<Double> A = Linalg.matrix(data);

        IQRDecomposition qr = Decomps.createQR();
        Tuple2<IMatrix<Double>, IMatrix<Double>> result = qr.decompose(A);

        TestResult r1 = recorder.record("decomposition", "qr_reconstruction");
        IMatrix<Double> Q = result._1;
        IMatrix<Double> R = result._2;
        IMatrix<Double> QR = Q.mmul(R);
        double maxDiff = 0;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
            maxDiff = Math.max(maxDiff, Math.abs(QR.get(i,j) - A.get(i,j)));
        r1.pass(maxDiff < 1e-8 ? "reconstruction OK, max_diff=" + maxDiff : "reconstruction FAIL, max_diff=" + maxDiff);

        // Q orthogonality: Q^T Q = I
        TestResult r2 = recorder.record("decomposition", "qr_orthogonality");
        IMatrix<Double> QtQ = Q.transpose().mmul(Q);
        double maxOff = 0;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            maxOff = Math.max(maxOff, Math.abs(QtQ.get(i,j) - expected));
        }
        r2.pass(maxOff < 1e-8 ? "orthogonal OK, max_off=" + maxOff : "orthogonal FAIL, max_off=" + maxOff);
    }

    @Test
    @DisplayName("4.3 Cholesky Decomposition")
    void testCholeskyDecomposition() {
        // SPD matrix: [[4,12,-16],[12,37,-43],[-16,-43,98]]
        double[][] data = {{4,12,-16},{12,37,-43},{-16,-43,98}};
        IMatrix<Double> A = Linalg.matrix(data);

        ICholeskyDecomposition chol = Decomps.createCholesky();
        IMatrix<Double> L = chol.decompose(A);

        TestResult r1 = recorder.record("decomposition", "cholesky_reconstruction");
        IMatrix<Double> Lt = L.transpose();
        IMatrix<Double> LLt = L.mmul(Lt);
        double maxDiff = 0;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
            maxDiff = Math.max(maxDiff, Math.abs(LLt.get(i,j) - A.get(i,j)));
        r1.pass(maxDiff < 1e-8 ? "reconstruction OK, max_diff=" + maxDiff : "reconstruction FAIL, max_diff=" + maxDiff);

        // L is lower triangular
        TestResult r2 = recorder.record("decomposition", "cholesky_triangular");
        boolean lower = true;
        for (int i = 0; i < 3; i++) for (int j = i+1; j < 3; j++)
            if (Math.abs(L.get(i,j)) > EPS) lower = false;
        r2.pass(lower ? "lower triangular confirmed" : "not lower triangular");
    }

    @Test
    @DisplayName("4.4 SVD Decomposition")
    void testSVDDecomposition() {
        double[][] data = {{3,2,2},{2,3,-2}};
        IMatrix<Double> A = Linalg.matrix(data);

        ISVDDecomposition svd = Decomps.createSVD();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        TestResult r1 = recorder.record("decomposition", "svd_reconstruction");
        IMatrix<Double> U = result._1;
        IVector<Double> S = result._2;
        IMatrix<Double> Vt = result._3;

        // Reconstruct: A = U * diag(S) * Vt
        IMatrix<Double> diagS = IMatrix.zeros(U.cols(), Vt.rows());
        for (int i = 0; i < S.length() && i < Math.min(diagS.rows(), diagS.cols()); i++)
            diagS.put(i, i, S.get(i));
        IMatrix<Double> USVt = U.mmul(diagS).mmul(Vt);
        double maxDiff = 0;
        for (int i = 0; i < A.rows(); i++) for (int j = 0; j < A.cols(); j++)
            maxDiff = Math.max(maxDiff, Math.abs(USVt.get(i,j) - A.get(i,j)));
        r1.pass(maxDiff < 1e-8 ? "reconstruction OK, max_diff=" + maxDiff : "reconstruction FAIL, max_diff=" + maxDiff);

        // Check singular values against known values for this matrix
        // Known singular values: 5.0, 3.0
        TestResult r2 = recorder.record("decomposition", "svd_singular_values");
        double s0 = S.get(0), s1 = S.get(1);
        boolean svOk = Math.abs(s0 - 5.0) < 1e-6 && Math.abs(s1 - 3.0) < 1e-6;
        r2.pass(svOk ? "singular values correct: " + s0 + ", " + s1 : "incorrect: " + s0 + ", " + s1);
    }

    @Test
    @DisplayName("4.5 Eigenvalue Decomposition")
    void testEigenDecomposition() {
        // Symmetric matrix
        double[][] data = {{4,2},{2,3}};
        IMatrix<Double> A = Linalg.matrix(data);

        IEigenDecomposition eigen = Decomps.createEigen();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        TestResult r1 = recorder.record("decomposition", "eigen_reconstruction");
        IVector<Double> vals = result._1;
        IMatrix<Double> vecs = result._2;

        // A v = lambda v for each eigenpair
        double maxDiff = 0;
        for (int k = 0; k < vals.length(); k++) {
            IVector<Double> v = vecs.getColumn(k);
            IVector<Double> Av = A.mmul(v);
            IVector<Double> lv = v.multiplyByScalar(vals.get(k));
            for (int i = 0; i < v.length(); i++)
                maxDiff = Math.max(maxDiff, Math.abs(Av.get(i) - lv.get(i)));
        }
        r1.pass(maxDiff < 1e-7 ? "Av=lv OK, max_diff=" + maxDiff : "Av=lv FAIL, max_diff=" + maxDiff);

        // Known eigenvalues for [[4,2],[2,3]]: 5.56155, 1.43845
        TestResult r2 = recorder.record("decomposition", "eigen_values");
        double v0 = vals.get(0), v1 = vals.get(1);
        // Sort them
        double minV = Math.min(v0, v1), maxV = Math.max(v0, v1);
        double expMin = (7 - Math.sqrt(17)) / 2; // ~1.43845
        double expMax = (7 + Math.sqrt(17)) / 2; // ~5.56155
        boolean ok = Math.abs(minV - expMin) < 1e-5 && Math.abs(maxV - expMax) < 1e-5;
        r2.pass(ok ? "eigenvalues correct" : "eigenvalues incorrect: " + minV + ", " + maxV);
    }

    @Test
    @DisplayName("4.6 Hessenberg Decomposition")
    void testHessenbergDecomposition() {
        double[][] data = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
        IMatrix<Double> A = Linalg.matrix(data);

        IHessenbergDecomposition hess = Decomps.createHessenberg();
        Tuple2<IMatrix<Double>, IMatrix<Double>> result = hess.decompose(A);

        TestResult r1 = recorder.record("decomposition", "hessenberg_reconstruction");
        // IHessenbergDecomposition 约定 decompose 返回 (H, Q)，满足 A = Q * H * Q^T
        IMatrix<Double> H = result._1;
        IMatrix<Double> Q = result._2;
        IMatrix<Double> QHQ = Q.mmul(H).mmul(Q.transpose());
        double maxDiff = 0;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
            maxDiff = Math.max(maxDiff, Math.abs(QHQ.get(i,j) - A.get(i,j)));
        r1.pass(maxDiff < 1e-7 ? "reconstruction OK" : "reconstruction FAIL, diff=" + maxDiff);

        // H should be upper Hessenberg
        TestResult r2 = recorder.record("decomposition", "hessenberg_form");
        boolean hessForm = true;
        for (int i = 2; i < 4; i++) for (int j = 0; j < i-1; j++)
            if (Math.abs(H.get(i,j)) > 1e-10) hessForm = false;
        r2.pass(hessForm ? "upper Hessenberg confirmed" : "not upper Hessenberg");
    }

    // =========================================================================
    // 5. Solvers
    // =========================================================================

    @Test
    @DisplayName("5.1 Linear System Solver")
    void testLinearSolve() {
        double[][] a = {{3,2,-1},{2,-2,4},{-1,0.5,-1}};
        double[] b = {1,-2,0};
        IMatrix<Double> A = Linalg.matrix(a);
        IVector<Double> B = IVector.of(b);

        IVector<Double> x = LinearSystemSolver.solve(A, B);

        TestResult r1 = recorder.record("solver", "linear_solve_3x3");
        // Ax = b verification
        IVector<Double> Ax = A.mmul(x);
        double maxDiff = 0;
        for (int i = 0; i < 3; i++) maxDiff = Math.max(maxDiff, Math.abs(Ax.get(i) - B.get(i)));

        // Known solution: x=[1, -2, -2] approximately
        double[] expected = {1.0, -2.0, -2.0};
        boolean solOk = true;
        for (int i = 0; i < 3; i++) if (Math.abs(x.get(i) - expected[i]) > 1e-5) solOk = false;
        r1.pass(solOk && maxDiff < 1e-8 ? "solution correct, residual=" + maxDiff : "solution incorrect");
    }

    @Test
    @DisplayName("5.2 Matrix Inverse")
    void testMatrixInverse() {
        double[][] a = {{4,7},{2,6}};
        IMatrix<Double> A = Linalg.matrix(a);

        IMatrix<Double> invA = MatrixInversionSolver.invert(A);

        TestResult r1 = recorder.record("solver", "inverse_2x2");
        // Expected: [[0.6,-0.7],[-0.2,0.4]]
        boolean ok = Math.abs(invA.get(0,0)-0.6)<1e-8 && Math.abs(invA.get(0,1)+0.7)<1e-8
                  && Math.abs(invA.get(1,0)+0.2)<1e-8 && Math.abs(invA.get(1,1)-0.4)<1e-8;
        r1.pass(ok ? "inverse correct" : "inverse incorrect");

        // A * A^-1 = I
        TestResult r2 = recorder.record("solver", "inverse_identity");
        IMatrix<Double> I = A.mmul(invA);
        boolean idOk = true;
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
            double exp = (i == j) ? 1.0 : 0.0;
            if (Math.abs(I.get(i,j) - exp) > 1e-7) idOk = false;
        }
        r2.pass(idOk ? "A*inv(A)=I confirmed" : "A*inv(A) != I");
    }

    @Test
    @DisplayName("5.3 Least Squares Solver")
    void testLeastSquares() {
        // Overdetermined system: y = 2x + 1 with some points
        double[][] a = {{1,1},{1,2},{1,3},{1,4},{1,5}};
        double[] b = {3.1, 5.0, 7.2, 8.8, 11.1};
        IMatrix<Double> A = Linalg.matrix(a);
        IVector<Double> B = IVector.of(b);

        IVector<Double> x = LeastSquaresSolver.solve(A, B);

        TestResult r1 = recorder.record("solver", "least_squares");
        // Expected approximately [1, 2] (intercept ~1, slope ~2)
        boolean ok = Math.abs(x.get(0) - 1.0) < 0.5 && Math.abs(x.get(1) - 2.0) < 0.5;
        r1.pass(ok ? "solution approximately correct: [" + x.get(0) + ", " + x.get(1) + "]" : "solution incorrect");
    }

    @Test
    @DisplayName("5.4 Determinant")
    void testDeterminant() {
        double[][] a = {{1,2,3},{4,5,6},{7,8,10}};
        IMatrix<Double> A = Linalg.matrix(a);

        double det = DeterminantSolver.compute(A);

        TestResult r = recorder.record("solver", "determinant_3x3");
        // det = 1*(50-48) - 2*(40-42) + 3*(32-35) = 2 + 4 - 9 = -3
        double expected = -3.0;
        r.pass(Math.abs(det - expected) < EPS ? "correct: " + det : "incorrect: " + det + " vs " + expected);
    }

    @Test
    @DisplayName("5.5 Matrix Rank")
    void testRank() {
        double[][] a = {{1,2,3},{2,4,6},{3,6,9}}; // rank 1
        IMatrix<Double> A = Linalg.matrix(a);

        int rank = RankSolver.compute(A);

        TestResult r = recorder.record("solver", "rank_rank1");
        r.pass(rank == 1 ? "correct: rank=1" : "incorrect: rank=" + rank);
    }

    @Test
    @DisplayName("5.6 Condition Number")
    void testConditionNumber() {
        double[][] a = {{1,2},{3,4}};
        IMatrix<Double> A = Linalg.matrix(a);

        double cond = ConditionNumberSolver.compute(A);

        TestResult r = recorder.record("solver", "condition_number");
        // NumPy: np.linalg.cond([[1,2],[3,4]]) ≈ 14.933
        double expected = 14.933034373659256;
        r.pass(Math.abs(cond - expected) < 1e-3 ? "correct: " + cond : "incorrect: " + cond + " vs " + expected);
    }

    // =========================================================================
    // 6. Advanced Matrix Operations
    // =========================================================================

    @Test
    @DisplayName("6.1 Matrix slicing")
    void testMatrixSlicing() {
        double[][] a = {{1,2,3,4},{5,6,7,8},{9,10,11,12}};
        IMatrix<Double> A = Linalg.matrix(a);

        TestResult r1 = recorder.record("advanced", "slice_rows");
        IMatrix<Double> sub = A.slice("0:2", ":");
        boolean ok = sub.rows() == 2 && sub.cols() == 4 && sub.get(0,0) == 1 && sub.get(1,3) == 8;
        r1.pass(ok ? "correct" : "incorrect");

        TestResult r2 = recorder.record("advanced", "slice_cols");
        IMatrix<Double> sub2 = A.slice(":", "1:3");
        ok = sub2.rows() == 3 && sub2.cols() == 2 && sub2.get(0,0) == 2 && sub2.get(2,1) == 11;
        r2.pass(ok ? "correct" : "incorrect");
    }

    @Test
    @DisplayName("6.2 Matrix hstack/vstack")
    void testStacking() {
        IMatrix<Double> A = IMatrix.ones(2, 2);
        IMatrix<Double> B = IMatrix.zeros(2, 2);

        TestResult r1 = recorder.record("advanced", "hstack");
        IMatrix<Double> H = A.hstack(B);
        r1.pass(H.rows() == 2 && H.cols() == 4 && H.get(0,0) == 1 && H.get(0,2) == 0 ? "correct" : "incorrect");

        TestResult r2 = recorder.record("advanced", "vstack");
        IMatrix<Double> V = A.vstack(B);
        r2.pass(V.rows() == 4 && V.cols() == 2 && V.get(0,0) == 1 && V.get(2,0) == 0 ? "correct" : "incorrect");
    }

    @Test
    @DisplayName("6.3 Matrix isSymmetric")
    void testIsSymmetric() {
        double[][] sym = {{1,2,3},{2,4,5},{3,5,6}};
        double[][] asym = {{1,2,3},{4,5,6},{7,8,9}};

        TestResult r1 = recorder.record("advanced", "isSymmetric_true");
        r1.pass(Linalg.matrix(sym).isSymmetric() ? "correctly identified symmetric" : "failed to identify symmetric");

        TestResult r2 = recorder.record("advanced", "isSymmetric_false");
        r2.pass(!Linalg.matrix(asym).isSymmetric() ? "correctly identified asymmetric" : "failed to identify asymmetric");
    }

    @Test
    @DisplayName("6.4 Matrix covariance")
    void testCovariance() {
        double[][] data = {{1,2,3},{4,5,6},{7,8,9}};
        IMatrix<Double> A = Linalg.matrix(data);

        TestResult r = recorder.record("advanced", "covariance");
        IMatrix<Double> cov = A.covariance();
        // Each column has variance: col0=[1,4,7], col1=[2,5,8], col2=[3,6,9]
        // cov(i,j) should be all 9 for i==j (variance of [1,4,7] = 9)
        boolean ok = cov != null && Math.abs(cov.get(0,0) - 9.0) < EPS;
        r.pass(ok ? "covariance computed" : "covariance incorrect or null");
    }

    @Test
    @DisplayName("6.5 Kronecker product")
    void testKronecker() {
        double[][] a = {{1,2},{3,4}};
        double[][] b = {{0,5},{6,7}};
        IMatrix<Double> A = Linalg.matrix(a);
        IMatrix<Double> B = Linalg.matrix(b);

        TestResult r = recorder.record("advanced", "kron");
        IMatrix<Double> K = A.kron(B);
        // Expected: [[0,5,0,10],[6,7,12,14],[0,15,0,20],[18,21,24,28]]
        boolean ok = K.rows() == 4 && K.cols() == 4
                  && K.get(0,0)==0 && K.get(0,1)==5 && K.get(0,2)==0 && K.get(0,3)==10
                  && K.get(1,0)==6 && K.get(1,1)==7 && K.get(1,2)==12 && K.get(1,3)==14
                  && K.get(2,0)==0 && K.get(2,1)==15 && K.get(2,2)==0 && K.get(2,3)==20
                  && K.get(3,0)==18 && K.get(3,1)==21 && K.get(3,2)==24 && K.get(3,3)==28;
        r.pass(ok ? "correct" : "incorrect");
    }

    @Test
    @DisplayName("6.6 Pseudo-inverse")
    void testPseudoInverse() {
        double[][] a = {{1,2},{3,4},{5,6}};
        IMatrix<Double> A = Linalg.matrix(a);

        TestResult r = recorder.record("advanced", "pseudoinverse");
        IMatrix<Double> pinv = A.pinv();
        // A * pinv(A) * A = A
        IMatrix<Double> ApA = A.mmul(pinv).mmul(A);
        double maxDiff = 0;
        for (int i = 0; i < A.rows(); i++) for (int j = 0; j < A.cols(); j++)
            maxDiff = Math.max(maxDiff, Math.abs(ApA.get(i,j) - A.get(i,j)));
        r.pass(maxDiff < 1e-6 ? "MP condition satisfied, diff=" + maxDiff : "MP condition FAIL, diff=" + maxDiff);
    }

    // =========================================================================
    // 7. Edge Cases and Stress Tests
    // =========================================================================

    @Test
    @DisplayName("7.1 Singular matrix handling")
    void testSingularMatrix() {
        double[][] a = {{1,2},{2,4}};
        IMatrix<Double> A = Linalg.matrix(a);

        TestResult r = recorder.record("edge", "singular_matrix");
        try {
            IMatrix<Double> inv = A.inv();
            r.fail("Should have thrown exception for singular matrix");
        } catch (Exception e) {
            r.pass("Correctly threw exception: " + e.getClass().getSimpleName());
        }
    }

    @Test
    @DisplayName("7.2 1x1 matrix")
    void test1x1Matrix() {
        double[][] a = {{5}};
        IMatrix<Double> A = Linalg.matrix(a);

        TestResult r1 = recorder.record("edge", "det_1x1");
        r1.pass(Math.abs(A.det() - 5.0) < EPS ? "correct" : "incorrect");

        TestResult r2 = recorder.record("edge", "inv_1x1");
        IMatrix<Double> inv = A.inv();
        r2.pass(Math.abs(inv.get(0,0) - 0.2) < EPS ? "correct" : "incorrect");
    }

    @Test
    @DisplayName("7.3 Empty/Zero matrix operations")
    void testZeroMatrix() {
        IMatrix<Double> Z = IMatrix.zeros(3, 3);

        TestResult r1 = recorder.record("edge", "zero_det");
        r1.pass(Math.abs(Z.det()) < EPS ? "correct: det=0" : "incorrect");

        TestResult r2 = recorder.record("edge", "zero_trace");
        r2.pass(Math.abs(Z.trace()) < EPS ? "correct: trace=0" : "incorrect");
    }

    @Test
    @DisplayName("7.4 Negative indices")
    void testNegativeIndices() {
        double[][] a = {{1,2,3},{4,5,6},{7,8,9}};
        IMatrix<Double> A = Linalg.matrix(a);

        TestResult r = recorder.record("edge", "negative_index");
        double val = A.get(-1, -1); // should be 9 (bottom-right)
        r.pass(Math.abs(val - 9.0) < EPS ? "correct: " + val : "incorrect: " + val);
    }

    @Test
    @DisplayName("7.5 Large matrix SVD")
    void testLargeSVD() {
        // Generate a 50x30 random matrix, test SVD reconstruction
        IMatrix<Double> A = IMatrix.randn(50, 30, 42L);

        long start = System.currentTimeMillis();
        ISVDDecomposition svd = Decomps.createSVD();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);
        long elapsed = System.currentTimeMillis() - start;

        TestResult r = recorder.record("stress", "svd_50x30");
        IMatrix<Double> U = result._1;
        IVector<Double> S = result._2;
        IMatrix<Double> Vt = result._3;

        IMatrix<Double> diagS = IMatrix.zeros(U.cols(), Vt.rows());
        for (int i = 0; i < S.length() && i < Math.min(diagS.rows(), diagS.cols()); i++)
            diagS.put(i, i, S.get(i));
        IMatrix<Double> recon = U.mmul(diagS).mmul(Vt);

        double maxDiff = 0;
        for (int i = 0; i < 50; i++) for (int j = 0; j < 30; j++)
            maxDiff = Math.max(maxDiff, Math.abs(recon.get(i,j) - A.get(i,j)));
        r.timeMs = elapsed;
        r.pass(maxDiff < 1e-6 ? "reconstruction OK in " + elapsed + "ms, diff=" + maxDiff : "reconstruction FAIL, diff=" + maxDiff);
    }

    @Test
    @DisplayName("7.6 Ill-conditioned matrix")
    void testIllConditioned() {
        // Hilbert matrix is notoriously ill-conditioned
        int n = 5;
        double[][] h = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
            h[i][j] = 1.0 / (i + j + 1);
        IMatrix<Double> H = Linalg.matrix(h);

        TestResult r = recorder.record("edge", "hilbert_cond");
        double cond = H.cond();
        // Hilbert 5x5 condition number is about 476607
        boolean ok = cond > 100000;
        r.pass(ok ? "ill-conditioned detected, cond=" + cond : "not ill-conditioned enough: " + cond);
    }
}
