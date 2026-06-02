package com.yishape.lab.math.benchmarks;

import com.yishape.lab.math.linalg.IDoubleMatrix;
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
public class MatrixMultiplyBenchmark {

    @Param({"64", "128", "256", "512"})
    int n;

    private IDoubleMatrix a, b;

    @Setup
    public void setup() {
        a = IDoubleMatrix.rand(n, n, 42L);
        b = IDoubleMatrix.rand(n, n, 99L);
    }

    @Benchmark
    public IDoubleMatrix gemm() {
        return a.mmul(b);
    }

    @Benchmark
    public IDoubleMatrix add() {
        return a.add(b);
    }

    @Benchmark
    public IDoubleMatrix transpose() {
        return a.transposeNew();
    }

    public static void main(String[] args) throws Exception {
        Options opt = new OptionsBuilder()
            .include(MatrixMultiplyBenchmark.class.getSimpleName())
            .build();
        new Runner(opt).run();
    }
}
