package com.yishape.lab.math.benchmarks;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IMatrix;
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
public class DecompositionBenchmark {

    @Param({"100", "200"})
    int n;

    private IDoubleMatrix a;
    private IDoubleMatrix b;

    @Setup
    public void setup() {
        a = IDoubleMatrix.rand(n, n, 42L);
        // make it SPD for Cholesky
        a = a.mmul(a.transposeNew()).add(IDoubleMatrix.eye(n).multiplyByScalar(n));
        b = IDoubleMatrix.rand(n, 1, 99L);
    }

    @Benchmark
    public IMatrix<Double> cholesky() {
        return a.cholesky();
    }

    @Benchmark
    public IMatrix<Double> lu() {
        var lu = a.lu();
        return lu._2;
    }

    @Benchmark
    public IMatrix<Double> qr() {
        var qr = a.qr();
        return qr._2;
    }

    @Benchmark
    public IMatrix<Double> solve() {
        return a.solve(b);
    }

    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
            .include(DecompositionBenchmark.class.getSimpleName())
            .build();
        new Runner(opt).run();
    }
}
