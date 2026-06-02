package com.yishape.lab.math.benchmarks;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
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
public class SparseMatrixMultiplyBenchmark {

    @Param({"500", "1000"})
    int n;

    @Param({"0.01", "0.05", "0.10"})
    double density;

    private ISparseMatrix sparseA, sparseB;
    private IDoubleMatrix dense;

    @Setup
    public void setup() {
        dense = IDoubleMatrix.rand(n, n, 123L);
        // create sparse by thresholding a random dense matrix
        double threshold = 1.0 - density;
        IDoubleMatrix randA = IDoubleMatrix.rand(n, n, 42L);
        IDoubleMatrix randB = IDoubleMatrix.rand(n, n, 99L);
        double[][] daA = new double[n][n];
        double[][] daB = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                daA[i][j] = randA.get(i, j) > threshold ? randA.get(i, j) : 0;
                daB[i][j] = randB.get(i, j) > threshold ? randB.get(i, j) : 0;
            }
        }
        sparseA = ISparseMatrix.fromDense(daA);
        sparseB = ISparseMatrix.fromDense(daB);
    }

    @Benchmark
    public ISparseMatrix sparseSparseMultiply() {
        return sparseA.multiply(sparseB);
    }

    @Benchmark
    public IMatrix<Double> sparseDenseMultiply() {
        return sparseA.multiplyDense(dense);
    }

    @Benchmark
    public IMatrix<Double> denseSparseMultiply() {
        return dense.multiply(sparseA);
    }

    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
            .include(SparseMatrixMultiplyBenchmark.class.getSimpleName())
            .build();
        new Runner(opt).run();
    }
}
