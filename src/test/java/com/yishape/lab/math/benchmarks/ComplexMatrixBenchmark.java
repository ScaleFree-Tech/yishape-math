package com.yishape.lab.math.benchmarks;

import com.yishape.lab.math.linalg.complex.IComplexMatrix;
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
public class ComplexMatrixBenchmark {

    @Param({"50", "100"})
    int n;

    private IComplexMatrix a, b;

    @Setup
    public void setup() {
        double[][] reA = new double[n][n];
        double[][] imA = new double[n][n];
        double[][] reB = new double[n][n];
        double[][] imB = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                reA[i][j] = Math.sin(i + j + 1);
                imA[i][j] = Math.cos(i + j + 1);
                reB[i][j] = Math.cos(i - j + 1);
                imB[i][j] = Math.sin(i - j + 1);
            }
        }
        a = IComplexMatrix.fromRealImag(reA, imA);
        b = IComplexMatrix.fromRealImag(reB, imB);
    }

    @Benchmark
    public IComplexMatrix multiply() {
        return a.multiply(b);
    }

    @Benchmark
    public IComplexMatrix lu() {
        var lu = a.lu();
        return lu._2;
    }

    @Benchmark
    public IComplexMatrix qr() {
        var qr = a.qr();
        return qr._2;
    }

    @Benchmark
    public IComplexMatrix svd() {
        var svd = a.svd();
        return svd._1; // U matrix
    }

    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
            .include(ComplexMatrixBenchmark.class.getSimpleName())
            .build();
        new Runner(opt).run();
    }
}
