package com.yishape.lab.math.benchmarks;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.linalg.sparse.ISparseLinearSolver;
import com.yishape.lab.math.linalg.sparse.impl.SparseBICGSTABSolver;
import com.yishape.lab.math.linalg.sparse.impl.SparseConjugateGradientSolver;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
public class CGSolverBenchmark {

    @Param({"529", "1024"})
    int n;

    private ISparseMatrix A;
    private IVector<Double> b;
    private ISparseLinearSolver cgSolver;
    private ISparseLinearSolver bicgstabSolver;

    @Setup
    public void setup() {
        int m = (int) Math.sqrt(n);
        n = m * m;
        int nnz = n * 5;
        int[] rows = new int[nnz];
        int[] cols = new int[nnz];
        double[] vals = new double[nnz];
        int idx = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                int r = i * m + j;
                rows[idx] = r; cols[idx] = r; vals[idx] = 4.0; idx++;
                if (j > 0)     { rows[idx] = r; cols[idx] = r - 1; vals[idx] = -1.0; idx++; }
                if (j < m - 1) { rows[idx] = r; cols[idx] = r + 1; vals[idx] = -1.0; idx++; }
                if (i > 0)     { rows[idx] = r; cols[idx] = r - m; vals[idx] = -1.0; idx++; }
                if (i < m - 1) { rows[idx] = r; cols[idx] = r + m; vals[idx] = -1.0; idx++; }
            }
        }
        A = ISparseMatrix.fromCOO(rows, cols, vals, n, n);

        b = IDoubleVector.rand(n);
        cgSolver = new SparseConjugateGradientSolver();
        bicgstabSolver = new SparseBICGSTABSolver();
    }

    @Benchmark
    public IVector<Double> cgSolve() {
        return cgSolver.solve(A, b);
    }

    @Benchmark
    public IVector<Double> bicgstabSolve() {
        return bicgstabSolver.solve(A, b);
    }

    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
            .include(CGSolverBenchmark.class.getSimpleName())
            .build();
        new Runner(opt).run();
    }
}
