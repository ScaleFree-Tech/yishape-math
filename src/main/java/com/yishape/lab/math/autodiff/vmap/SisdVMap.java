package com.yishape.lab.math.autodiff.vmap;

import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

import com.yishape.lab.math.compute.DoubleFlatGemm;
import com.yishape.lab.math.compute.SISDDoubleComputer;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import static com.yishape.lab.math.compute.ops.ReduceOperation.MAX;
import static com.yishape.lab.math.compute.ops.ReduceOperation.MEAN;
import static com.yishape.lab.math.compute.ops.ReduceOperation.MIN;
import static com.yishape.lab.math.compute.ops.ReduceOperation.PROD;
import static com.yishape.lab.math.compute.ops.ReduceOperation.STANDARD_DEVIATION;
import static com.yishape.lab.math.compute.ops.ReduceOperation.SUM;
import static com.yishape.lab.math.compute.ops.ReduceOperation.VARIANCE;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.RereDoubleMatrix;
import com.yishape.lab.util.YishapeLogger;

/**
 * Pure Java vmap backend. Always available — the final fallback.
 * Uses ForkJoin for large batches in reductions and delegates element-wise
 * operations to {@link SISDDoubleComputer}.
 *
 * @author lteb2
 */
final class SisdVMap implements IVMap {

    private static final YishapeLogger log = YishapeLogger.getLogger(SisdVMap.class);

    private static final int PARALLEL_REDUCE_THRESHOLD = 256;
    private static final long PARALLEL_ELEMENT_THRESHOLD = 100_000L;

    private static final SISDDoubleComputer SISD = new SISDDoubleComputer();

    SisdVMap() {}

    // ---- stack / unstack ----

    @Override
    public double[] stack(double[][] arrays) {
        if (arrays == null || arrays.length == 0) {
            throw new IllegalArgumentException("stack requires non-empty input");
        }
        int n = arrays.length;
        int d = arrays[0].length;
        double[] result = new double[n * d];
        for (int i = 0; i < n; i++) {
            if (arrays[i].length != d) {
                throw new IllegalArgumentException(
                    "Dimension mismatch in stack: expected " + d + ", got " + arrays[i].length);
            }
            System.arraycopy(arrays[i], 0, result, i * d, d);
        }
        return result;
    }

    @Override
    public double[][] unstack(double[] flat, int n, int d) {
        if (flat == null) {
            throw new IllegalArgumentException("unstack requires non-null input");
        }
        double[][] result = new double[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(flat, i * d, result[i], 0, d);
        }
        return result;
    }

    // ---- batchReduce ----

    @Override
    public double[] batchReduce(double[] flat, int n, int d, ReduceOperation op) {
        if (flat == null || n <= 0 || d <= 0) {
            throw new IllegalArgumentException("batchReduce: invalid dimensions n=" + n + " d=" + d);
        }
        double[] result = new double[n];
        long total = (long) n * d;

        if (total >= PARALLEL_ELEMENT_THRESHOLD && n >= PARALLEL_REDUCE_THRESHOLD) {
            reduceRowsParallel(flat, n, d, op, result);
        } else {
            reduceRowsSequential(flat, n, d, op, result);
        }
        return result;
    }

    private static void reduceRowsSequential(double[] flat, int n, int d,
                                              ReduceOperation op, double[] out) {
        switch (op) {
            case SUM -> {
                for (int i = 0; i < n; i++) out[i] = kahanSum(flat, i * d, d);
            }
            case MEAN -> {
                for (int i = 0; i < n; i++) out[i] = kahanSum(flat, i * d, d) / d;
            }
            case MIN -> {
                for (int i = 0; i < n; i++) {
                    int off = i * d;
                    double v = flat[off];
                    for (int j = 1; j < d; j++) if (flat[off + j] < v) v = flat[off + j];
                    out[i] = v;
                }
            }
            case MAX -> {
                for (int i = 0; i < n; i++) {
                    int off = i * d;
                    double v = flat[off];
                    for (int j = 1; j < d; j++) if (flat[off + j] > v) v = flat[off + j];
                    out[i] = v;
                }
            }
            case VARIANCE -> {
                for (int i = 0; i < n; i++) {
                    int off = i * d;
                    double mean = kahanSum(flat, off, d) / d;
                    out[i] = kahanSsd(flat, off, d, mean) / (d - 1);
                }
            }
            case STANDARD_DEVIATION -> {
                for (int i = 0; i < n; i++) {
                    int off = i * d;
                    double mean = kahanSum(flat, off, d) / d;
                    out[i] = Math.sqrt(kahanSsd(flat, off, d, mean) / (d - 1));
                }
            }
            case PROD -> {
                for (int i = 0; i < n; i++) {
                    int off = i * d;
                    double p = 1.0;
                    for (int j = 0; j < d; j++) p *= flat[off + j];
                    out[i] = p;
                }
            }
        }
    }

    private static double kahanSum(double[] arr, int off, int len) {
        double sum = 0.0, c = 0.0;
        for (int j = 0; j < len; j++) {
            double y = arr[off + j] - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    private static double kahanSsd(double[] arr, int off, int len, double mean) {
        double ssd = 0.0, c = 0.0;
        for (int j = 0; j < len; j++) {
            double diff = arr[off + j] - mean;
            double t = diff * diff - c;
            double y = ssd + t;
            c = (y - ssd) - t;
            ssd = y;
        }
        return ssd;
    }

    private static void reduceRowsParallel(double[] flat, int n, int d,
                                            ReduceOperation op, double[] out) {
        ForkJoinPool pool = RereDoubleMatrix.getThreadPool();
        int numTasks = Math.min(pool.getParallelism(), (n + 63) / 64);
        int chunkSize = (n + numTasks - 1) / numTasks;
        @SuppressWarnings("unchecked")
        Future<?>[] futures = new Future[numTasks];
        for (int t = 0; t < numTasks; t++) {
            final int from = t * chunkSize;
            final int to = Math.min(from + chunkSize, n);
            if (from >= to) break;
            futures[t] = pool.submit(() -> {
                double[] local = new double[to - from];
                reduceRowsSequential(flat, to - from, d, op, local);
                System.arraycopy(local, 0, out, from, local.length);
            });
        }
        for (Future<?> f : futures) {
            if (f != null) { try { f.get(); } catch (Exception e) { throw new RuntimeException(e); } }
        }
    }

    // ---- binaryOperate / unaryOp ----

    @Override
    public double[] binaryOperate(double[] a, double[] b, BinaryOperation op) {
        return SISD.binaryOperate(a, b, op);
    }

    @Override
    public double[] unaryOp(double[] a, UniversalOperation op, double param) {
        return SISD.universalOperate(a, op, param);
    }

    // ---- batchMatmul ----

    @Override
    public double[] batchMatmul(double[] a, double[] b, int batch, int m, int k, int n) {
        return DoubleFlatGemm.flatMmulBatched(a, b, batch, m, k, n);
    }

    // ---- DL ops ----

    @Override
    public double[] batchSoftmax(double[] flat, int batch, int dim) {
        double[] out = new double[flat.length];
        for (int i = 0; i < batch; i++) {
            int off = i * dim;
            double max = flat[off];
            for (int j = 1; j < dim; j++) {
                if (flat[off + j] > max) max = flat[off + j];
            }
            double sum = 0.0;
            for (int j = 0; j < dim; j++) {
                double v = Math.exp(flat[off + j] - max);
                out[off + j] = v;
                sum += v;
            }
            for (int j = 0; j < dim; j++) {
                out[off + j] /= sum;
            }
        }
        return out;
    }

    @Override
    public double[] batchLayerNorm(double[] x, double[] gamma, double[] beta, int batch, int dim, float eps) {
        double[] out = new double[x.length];
        for (int i = 0; i < batch; i++) {
            int off = i * dim;
            double mean = kahanSum(x, off, dim) / dim;
            double var = kahanSsd(x, off, dim, mean) / dim;
            double invStd = 1.0 / Math.sqrt(var + eps);
            for (int j = 0; j < dim; j++) {
                out[off + j] = (x[off + j] - mean) * invStd * gamma[j] + beta[j];
            }
        }
        return out;
    }

    @Override
    public double[] im2col(double[] input, int C, int H, int W, int kH, int kW, int stride, int pad) {
        int outH = (H + 2 * pad - kH) / stride + 1;
        int outW = (W + 2 * pad - kW) / stride + 1;
        int cols = outH * outW;
        int rows = C * kH * kW;
        double[] out = new double[rows * cols];

        for (int c = 0; c < C; c++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    int row = (c * kH + kh) * kW + kw;
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            int hIn = oh * stride + kh - pad;
                            int wIn = ow * stride + kw - pad;
                            double v = 0.0;
                            if (hIn >= 0 && hIn < H && wIn >= 0 && wIn < W) {
                                v = input[(c * H + hIn) * W + wIn];
                            }
                            out[row * cols + oh * outW + ow] = v;
                        }
                    }
                }
            }
        }
        return out;
    }

    @Override
    public double[] batchDropout(double[] flat, double p, long seed) {
        if (p <= 0.0) {
            return flat.clone();
        }
        double[] out = new double[flat.length];
        double scale = 1.0 / (1.0 - p);
        Random rng = new Random(seed);
        // Generate mask in chunks to balance speed vs memory
        int len = flat.length;
        int chunk = Math.min(len, 4096);
        double[] maskChunk = new double[chunk];
        for (int off = 0; off < len; off += chunk) {
            int cur = Math.min(chunk, len - off);
            for (int j = 0; j < cur; j++) {
                maskChunk[j] = rng.nextDouble();
            }
            for (int j = 0; j < cur; j++) {
                out[off + j] = (maskChunk[j] < p) ? 0.0 : flat[off + j] * scale;
            }
        }
        return out;
    }
}
