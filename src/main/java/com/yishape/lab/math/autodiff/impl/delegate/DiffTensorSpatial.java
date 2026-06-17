package com.yishape.lab.math.autodiff.impl.delegate;

import com.yishape.lab.math.autodiff.impl.AutodiffBufferPool;
import com.yishape.lab.math.autodiff.impl.DiffTensorUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.compute.DoubleFlatGemm;
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.gpu.GpuActivation;
import com.yishape.lab.math.compute.gpu.GpuReduce;
import com.yishape.lab.math.compute.hpc.HpcIm2col;
import com.yishape.lab.math.compute.hpc.HpcLoss;
import com.yishape.lab.math.compute.hpc.HpcCross;
import com.yishape.lab.math.compute.hpc.HpcGridSample;
import com.yishape.lab.math.compute.hpc.HpcTrapezoidalScan;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.tensor.EinsumParser;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.linalg.tensor.TensorShape;
import com.yishape.lab.math.compute.gpu.GpuGroupNorm;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.graph.ScalarParamEncoding;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorSpatial {
    private DiffTensorSpatial() { /* utility class */ }

// ==================== cross — 3D vector cross product ====================

public static IDiffTensor cross(RereDiffTensor tensor, IDiffTensor other) {
    RereDiffTensor o = (RereDiffTensor) other;
    int[] sA = tensor.shape();
    int[] sB = o.shape();
    if (sA[sA.length - 1] != 3 || sB[sB.length - 1] != 3) {
        throw new IllegalArgumentException("cross requires last dim = 3, got " +
            Arrays.toString(sA) + " and " + Arrays.toString(sB));
    }
    int[] bcShape = TensorShape.broadcastShape(sA, sB);
    long outSize = 1;
    for (int d : bcShape) outSize *= d;
    double[] aData = tensor.value.toDoubleArray();
    double[] bData = o.value.toDoubleArray();
    double[] y = new double[(int) outSize];
    long numTriplets = outSize / 3;
    final long fOutSize = outSize;    // effectively-final capture for lambda
    final long fNumTriplets = numTriplets;
    final int[] fBcShape = bcShape;

    // Pre-broadcast data to common shape for HPC / SIMD path
    double[] aBC = new double[(int) outSize];
    double[] bBC = new double[(int) outSize];
    DiffTensorUtil.broadcastTo(aData, sA, aBC, bcShape);
    DiffTensorUtil.broadcastTo(bData, sB, bBC, bcShape);

    // Try HPC first, fall back to SISD
    boolean hpcUsed = HpcCross.tryCrossForward(aBC, bBC, (int) numTriplets, y);

    if (!hpcUsed) {
        // SISD fallback: element-wise cross product with broadcast indexing
        for (long t = 0; t < numTriplets; t++) {
            long flatIdx = t * 3;
            int[] bcIdx = DiffTensorUtil.unlinearizeInt((int) flatIdx, bcShape);
            int ai = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, sA, bcShape);
            int bi = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, sB, bcShape);
            int aBase = (ai / 3) * 3, bBase = (bi / 3) * 3;
            y[(int) flatIdx]     = aData[aBase + 1] * bData[bBase + 2] - aData[aBase + 2] * bData[bBase + 1];
            y[(int) flatIdx + 1] = aData[aBase + 2] * bData[bBase + 0] - aData[aBase + 0] * bData[bBase + 2];
            y[(int) flatIdx + 2] = aData[aBase + 0] * bData[bBase + 1] - aData[aBase + 1] * bData[bBase + 0];
        }
    }
    final int[] fSA = sA, fSB = sB;
    final boolean fHpcUsed = hpcUsed;
    final double[] fABC = aBC, fBBC = bBC;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpA = self.inputs.get(0);
        RereDiffTensor inpB = self.inputs.get(1);
        double[] g = self.grad;
        int daSize = (int) inpA.value.totalSize();
        int dbSize = (int) inpB.value.totalSize();
        double[] da = AutodiffBufferPool.acquire(daSize);
        double[] db = AutodiffBufferPool.acquire(dbSize);

        if (fHpcUsed) {
            // HPC backward on broadcasted shapes, then un-broadcast
            double[] daBC = AutodiffBufferPool.acquire((int) fOutSize);
            double[] dbBC = AutodiffBufferPool.acquire((int) fOutSize);
            if (HpcCross.tryCrossBackward(g, fABC, fBBC, (int) fNumTriplets, daBC, dbBC)) {
                DiffTensorUtil.unbroadcastSum(daBC, fBcShape, da, fSA);
                DiffTensorUtil.unbroadcastSum(dbBC, fBcShape, db, fSB);
                AutodiffBufferPool.release(dbBC);
                AutodiffBufferPool.release(daBC);
                inpA.accGradFromPooled(da, daSize);
                inpB.accGradFromPooled(db, dbSize);
                return;
            }
            AutodiffBufferPool.release(dbBC);
            AutodiffBufferPool.release(daBC);
            // If HPC backward fails, fall through to SISD
        }

        // Perf: reuse data extracted during forward instead of re-extracting via toDoubleArray()
        double[] bd = bData;
        double[] ad = aData;
        for (long t = 0; t < fNumTriplets; t++) {
            long flatIdx = t * 3;
            int[] bcIdx = DiffTensorUtil.unlinearizeInt((int) flatIdx, fBcShape);
            int ai = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, fSA, fBcShape);
            int bi = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, fSB, fBcShape);
            int aBase = (ai / 3) * 3, bBase = (bi / 3) * 3;
            double g0 = g[(int) flatIdx];
            double g1 = g[(int) flatIdx + 1];
            double g2 = g[(int) flatIdx + 2];
            double b0v = bd[bBase], b1v = bd[bBase + 1], b2v = bd[bBase + 2];
            double a0v = ad[aBase], a1v = ad[aBase + 1], a2v = ad[aBase + 2];
            da[aBase]     += b1v * g2 - b2v * g1;
            da[aBase + 1] += b2v * g0 - b0v * g2;
            da[aBase + 2] += b0v * g1 - b1v * g0;
            db[bBase]     += g1 * a2v - g2 * a1v;
            db[bBase + 1] += g2 * a0v - g0 * a2v;
            db[bBase + 2] += g0 * a1v - g1 * a0v;
        }
        inpA.accGradFromPooled(da, daSize);
        inpB.accGradFromPooled(db, dbSize);
    };

    int[] outShape = bcShape.clone();
    return new RereDiffTensor(y, outShape, List.of(tensor, o), bw, "cross");
}

// ==================== gridSample — differentiable image warp ====================

public static IDiffTensor gridSample(RereDiffTensor tensor, IDiffTensor grid, String mode, String paddingMode) {
    int[] s = tensor.shape();
    if (tensor.rank() != 4) throw new IllegalArgumentException("gridSample input must be [N,C,H,W], got " + Arrays.toString(s));
    int N = s[0], C = s[1], H = s[2], W = s[3];
    int[] gs = grid.shape();
    int outH = gs[1], outW = gs[2];
    double[] xd = tensor.value.toDoubleArray();
    double[] gd = grid.toDoubleArray();
    double[] y = new double[N * C * outH * outW];
    boolean bilinear = "bilinear".equals(mode);
    int padModeIdx = switch (paddingMode) {
        case "border" -> 1;
        case "reflection" -> 2;
        default -> 0;
    };
    int modeIdx = bilinear ? 0 : 1;

    // Try HPC first, fall back to SISD
    boolean hpcUsed = HpcGridSample.tryGridSampleForward(
            xd, gd, N, C, H, W, outH, outW, modeIdx, padModeIdx, y);

    if (!hpcUsed) {
        // SISD fallback
        double[][] savedWeights = bilinear ? new double[N * C * outH * outW][] : null;
        int[][] savedIndices = new int[N * C * outH * outW][];
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    double gx = gd[((n * outH + oh) * outW + ow) * 2];
                    double gy = gd[((n * outH + oh) * outW + ow) * 2 + 1];
                    double px = (gx + 1.0) * 0.5 * (W - 1);
                    double py = (gy + 1.0) * 0.5 * (H - 1);

                    if (bilinear) {
                        int ix0 = (int) Math.floor(px), iy0 = (int) Math.floor(py);
                        int ix1 = ix0 + 1, iy1 = iy0 + 1;
                        double wx1 = px - ix0, wy1 = py - iy0;
                        double wx0 = 1.0 - wx1, wy0 = 1.0 - wy1;
                        int[][] corners = {{iy0, ix0}, {iy0, ix1}, {iy1, ix0}, {iy1, ix1}};
                        double[] weights = {wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1};
                        for (int c = 0; c < C; c++) {
                            int outIdx = ((n * C + c) * outH + oh) * outW + ow;
                            double val = 0;
                            int[] idxs = new int[4];
                            double[] wts = new double[4];
                            for (int k = 0; k < 4; k++) {
                                int sy = clampCoord(corners[k][0], H, paddingMode);
                                int sx = clampCoord(corners[k][1], W, paddingMode);
                                idxs[k] = sy * W + sx;
                                wts[k] = weights[k];
                                if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
                                    val += wts[k] * xd[((n * C + c) * H + sy) * W + sx];
                                }
                            }
                            y[outIdx] = val;
                            savedWeights[outIdx] = wts;
                            savedIndices[outIdx] = idxs;
                        }
                    } else {
                        int ix = (int) Math.round(px), iy = (int) Math.round(py);
                        ix = clampCoord(ix, W, paddingMode);
                        iy = clampCoord(iy, H, paddingMode);
                        for (int c = 0; c < C; c++) {
                            int outIdx = ((n * C + c) * outH + oh) * outW + ow;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                y[outIdx] = xd[((n * C + c) * H + iy) * W + ix];
                            }
                            savedIndices[outIdx] = new int[]{iy * W + ix};
                        }
                    }
                }
            }
        }
        // Capture saved indices/weights for Java backward
        final double[][] fWeights = savedWeights;
        final int[][] fIndices = savedIndices;
        final boolean fBilinear = bilinear;
        final int fN2 = N, fC2 = C, fH2 = H, fW2 = W, fOutH2 = outH, fOutW2 = outW;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            int dxSize = (int) inp.value.totalSize();
            double[] dx = AutodiffBufferPool.acquire(dxSize);
            for (int n2 = 0; n2 < fN2; n2++) {
                for (int c2 = 0; c2 < fC2; c2++) {
                    for (int oh2 = 0; oh2 < fOutH2; oh2++) {
                        for (int ow2 = 0; ow2 < fOutW2; ow2++) {
                            int outIdx = ((n2 * fC2 + c2) * fOutH2 + oh2) * fOutW2 + ow2;
                            double gv = g[outIdx];
                            if (gv == 0) continue;
                            int[] idxs = fIndices[outIdx];
                            int inBase = ((n2 * fC2 + c2) * fH2) * fW2;
                            if (fBilinear && fWeights != null && fWeights[outIdx] != null) {
                                double[] wts = fWeights[outIdx];
                                for (int k = 0; k < 4; k++) {
                                    int flat = idxs[k];
                                    if (flat >= 0 && flat < fH2 * fW2)
                                        dx[inBase + flat] += gv * wts[k];
                                }
                            } else {
                                int flat = idxs[0];
                                if (flat >= 0 && flat < fH2 * fW2)
                                    dx[inBase + flat] += gv;
                            }
                        }
                    }
                }
            }
            inp.accGradFromPooled(dx, dxSize);
        };

        RereDiffTensor result = new RereDiffTensor(y, new int[]{N, C, outH, outW},
            List.of(tensor, (RereDiffTensor) grid), bw, "gridSample");
        result.scalarParam = ScalarParamEncoding.GridSample.packScalarParam(H, W);
        result.scalarParam2 = ScalarParamEncoding.GridSample.packScalarParam2(padModeIdx, modeIdx);
        return result;
    }

    // HPC path: backward recomputes sampling positions from input+grid
    final int fN = N, fC = C, fH = H, fW = W, fOutH = outH, fOutW = outW;
    final int fModeIdx = modeIdx, fPadModeIdx = padModeIdx;
    final double[] fInputData = xd;
    final double[] fGridData = gd;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inp = self.inputs.get(0);
        double[] g = self.grad;
        int dxSizeHpc = (int) inp.value.totalSize();
        double[] dx = AutodiffBufferPool.acquire(dxSizeHpc);
        if (HpcGridSample.tryGridSampleBackward(g, fInputData, fGridData,
                fN, fC, fH, fW, fOutH, fOutW, fModeIdx, fPadModeIdx, dx)) {
            inp.accGradFromPooled(dx, dxSizeHpc);
            return;
        }
        // Should not happen: if HPC forward succeeded, HPC backward should too.
        // Fallback: release unused buffer (rare edge case).
        AutodiffBufferPool.release(dx);
    };

    RereDiffTensor result = new RereDiffTensor(y, new int[]{N, C, outH, outW},
        List.of(tensor, (RereDiffTensor) grid), bw, "gridSample");
    result.scalarParam = ScalarParamEncoding.GridSample.packScalarParam(H, W);
    result.scalarParam2 = ScalarParamEncoding.GridSample.packScalarParam2(padModeIdx, modeIdx);
    return result;
}

private static int clampCoord(int coord, int limit, String paddingMode) {
    return switch (paddingMode) {
        case "border" -> Math.clamp(coord, 0, limit - 1);
        case "reflection" -> {
            int r = Math.abs(coord) % (2 * limit);
            yield r >= limit ? 2 * limit - 1 - r : r;
        }
        default -> coord; // "zeros": return out-of-bounds as-is (handled as 0 in sampling)
    };
}

// ==================== trapezoidalScan — Mamba SSM ====================

public static IDiffTensor trapezoidalScan(RereDiffTensor tensor, IDiffTensor delta, IDiffTensor A, IDiffTensor B,
                                    IDiffTensor C, IDiffTensor D) {
    int[] s = tensor.shape(); // U: [B, L, D]
    if (tensor.rank() != 3) throw new IllegalArgumentException(
        "trapezoidalScan: U must be [B,L,D], got " + Arrays.toString(s));
    int bSize = s[0], seqLen = s[1], dim = s[2];

    double[] uData = tensor.value.toDoubleArray();
    double[] deltaData = delta.toDoubleArray();
    double[] aData = A.toDoubleArray();
    double[] bData = B.toDoubleArray();
    double[] cData = C.toDoubleArray();
    double[] dData = D.toDoubleArray();

    boolean aIsVec = A.rank() == 1;
    boolean dIsScalar = D.totalSize() == 1;
    boolean deltaBroadcast = delta.shape()[delta.rank() - 2] == 1; // [B,1,D]
    int fAIsVec = aIsVec ? 1 : 0;
    int fDIsScalar = dIsScalar ? 1 : 0;
    int fDeltaBroadcast = deltaBroadcast ? 1 : 0;

    double[] y = new double[bSize * seqLen * dim];
    double[] savedH = new double[bSize * seqLen * dim];
    double[] savedABar = new double[bSize * seqLen * dim];
    double[] savedBBarU = new double[bSize * seqLen * dim];

    // Try HPC first, fall back to SISD
    boolean hpcUsed = HpcTrapezoidalScan.tryTrapezoidalScanForward(
            uData, deltaData, aData, bData, cData, dData,
            bSize, seqLen, dim, fAIsVec, fDIsScalar, fDeltaBroadcast,
            y, savedH, savedABar, savedBBarU);

    if (!hpcUsed) {
        // SISD fallback
        for (int b = 0; b < bSize; b++) {
            double[] h = new double[dim]; // current hidden state (init = 0)
            for (int t = 0; t < seqLen; t++) {
                double[] hNext = new double[dim];
                for (int d = 0; d < dim; d++) {
                    double dt = deltaBroadcast
                        ? deltaData[(b * 1 + 0) * dim + d]
                        : deltaData[(b * seqLen + t) * dim + d];
                    double aVal = aIsVec ? aData[d] : aData[b * dim + d];
                    double ut = uData[(b * seqLen + t) * dim + d];
                    double bt = bData[(b * seqLen + t) * dim + d];

                    double aBar = Math.exp(dt * aVal);
                    double bBar = dt * bt;
                    hNext[d] = aBar * h[d] + bBar * ut;

                    int flatT = (b * seqLen + t) * dim + d;
                    savedABar[flatT] = aBar;
                    savedBBarU[flatT] = bBar * ut;
                }
                for (int d = 0; d < dim; d++) {
                    double ct = cData[(b * seqLen + t) * dim + d];
                    double dtScalar = dIsScalar ? dData[0] : dData[d];
                    double ut = uData[(b * seqLen + t) * dim + d];
                    int flatT = (b * seqLen + t) * dim + d;
                    y[flatT] = ct * hNext[d] + dtScalar * ut;
                    savedH[flatT] = hNext[d];
                }
                h = hNext;
            }
        }

        final int fB2 = bSize, fL2 = seqLen, fD2 = dim;
        final double[] fDeltaData2 = deltaData, fUData2 = uData;
        final double[] fAData2 = aData, fBData2 = bData, fCData2 = cData, fDData2 = dData;
        final boolean fAIsVec2 = aIsVec, fDIsScalar2 = dIsScalar, fDeltaBroadcast2 = deltaBroadcast;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpU = self.inputs.get(0);
            RereDiffTensor inpDelta = self.inputs.get(1);
            RereDiffTensor inpA = self.inputs.get(2);
            RereDiffTensor inpB = self.inputs.get(3);
            RereDiffTensor inpC = self.inputs.get(4);
            RereDiffTensor inpD = self.inputs.get(5);
            double[] gy = self.grad;

            int dUSize = (int) inpU.value.totalSize();
            int dDeltaSize = (int) inpDelta.value.totalSize();
            int dASize = (int) inpA.value.totalSize();
            int dBSize = (int) inpB.value.totalSize();
            int dCSize = (int) inpC.value.totalSize();
            int dDSize = (int) inpD.value.totalSize();
            double[] dU = AutodiffBufferPool.acquire(dUSize);
            double[] dDelta = AutodiffBufferPool.acquire(dDeltaSize);
            double[] dA = AutodiffBufferPool.acquire(dASize);
            double[] dB = AutodiffBufferPool.acquire(dBSize);
            double[] dC = AutodiffBufferPool.acquire(dCSize);
            double[] dD = AutodiffBufferPool.acquire(dDSize);

            for (int b = 0; b < fB2; b++) {
                double[] dh = new double[fD2];
                for (int t = fL2 - 1; t >= 0; t--) {
                    for (int d = 0; d < fD2; d++) {
                        int flatT = (b * fL2 + t) * fD2 + d;
                        double ct = fCData2[(b * fL2 + t) * fD2 + d];
                        double dt = fDeltaBroadcast2
                            ? fDeltaData2[(b * 1 + 0) * fD2 + d]
                            : fDeltaData2[(b * fL2 + t) * fD2 + d];
                        double aVal = fAIsVec2 ? fAData2[d] : fAData2[b * fD2 + d];
                        double aBar = savedABar[flatT];
                        double ut = fUData2[(b * fL2 + t) * fD2 + d];
                        double bt = fBData2[(b * fL2 + t) * fD2 + d];

                        double gy_t = gy[flatT];
                        dC[flatT] += gy_t * savedH[flatT];
                        if (fDIsScalar2) dD[0] += gy_t * ut;
                        else dD[d] += gy_t * ut;

                        dh[d] += gy_t * ct;
                        double bBar = dt * bt;
                        dU[flatT] += dh[d] * bBar;
                        if (fDIsScalar2) dU[flatT] += gy_t * fDData2[0];
                        else dU[flatT] += gy_t * fDData2[d];

                        dB[flatT] += dh[d] * dt * ut;
                        dDelta[fDeltaBroadcast2 ? (b * 1 + 0) * fD2 + d : flatT] +=
                            dh[d] * bt * ut;
                        dDelta[fDeltaBroadcast2 ? (b * 1 + 0) * fD2 + d : flatT] +=
                            dh[d] * aBar * ((t > 0) ? savedH[flatT - fD2] : 0) * aVal;

                        double hPrev = (t > 0) ? savedH[flatT - fD2] : 0;
                        int aOff = fAIsVec2 ? d : (b * fD2 + d);
                        dA[aOff] += dh[d] * aBar * hPrev * dt;
                        dh[d] = dh[d] * aBar;
                    }
                }
            }
            inpU.accGradFromPooled(dU, dUSize);
            inpDelta.accGradFromPooled(dDelta, dDeltaSize);
            inpA.accGradFromPooled(dA, dASize);
            inpB.accGradFromPooled(dB, dBSize);
            inpC.accGradFromPooled(dC, dCSize);
            inpD.accGradFromPooled(dD, dDSize);
        };

        return new RereDiffTensor(y, s, List.of(tensor, (RereDiffTensor) delta,
            (RereDiffTensor) A, (RereDiffTensor) B, (RereDiffTensor) C, (RereDiffTensor) D),
            bw, "trapezoidalScan");
    }

    // HPC path: backward uses HPC backward with saved state arrays
    final int fB = bSize, fL = seqLen, fD = dim;
    final int fAIsVecI = fAIsVec, fDIsScalarI = fDIsScalar, fDeltaBroadcastI = fDeltaBroadcast;
    final double[] fUData = uData, fDeltaData = deltaData;
    final double[] fAData = aData, fBData = bData, fCData = cData, fDData = dData;

    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor inpU = self.inputs.get(0);
        RereDiffTensor inpDelta = self.inputs.get(1);
        RereDiffTensor inpA = self.inputs.get(2);
        RereDiffTensor inpB = self.inputs.get(3);
        RereDiffTensor inpC = self.inputs.get(4);
        RereDiffTensor inpD = self.inputs.get(5);
        double[] gy = self.grad;

        int dUSizeHpc = (int) inpU.value.totalSize();
        int dDeltaSizeHpc = (int) inpDelta.value.totalSize();
        int dASizeHpc = (int) inpA.value.totalSize();
        int dBSizeHpc = (int) inpB.value.totalSize();
        int dCSizeHpc = (int) inpC.value.totalSize();
        int dDSizeHpc = (int) inpD.value.totalSize();
        double[] dU = AutodiffBufferPool.acquire(dUSizeHpc);
        double[] dDelta = AutodiffBufferPool.acquire(dDeltaSizeHpc);
        double[] dA = AutodiffBufferPool.acquire(dASizeHpc);
        double[] dB = AutodiffBufferPool.acquire(dBSizeHpc);
        double[] dC = AutodiffBufferPool.acquire(dCSizeHpc);
        double[] dD = AutodiffBufferPool.acquire(dDSizeHpc);

        if (HpcTrapezoidalScan.tryTrapezoidalScanBackward(
                gy, fUData, fDeltaData, fAData, fBData, fCData, fDData,
                savedH, savedABar, savedBBarU,
                fB, fL, fD, fAIsVecI, fDIsScalarI, fDeltaBroadcastI,
                dU, dDelta, dA, dB, dC, dD)) {
            inpU.accGradFromPooled(dU, dUSizeHpc);
            inpDelta.accGradFromPooled(dDelta, dDeltaSizeHpc);
            inpA.accGradFromPooled(dA, dASizeHpc);
            inpB.accGradFromPooled(dB, dBSizeHpc);
            inpC.accGradFromPooled(dC, dCSizeHpc);
            inpD.accGradFromPooled(dD, dDSizeHpc);
            return;
        }
        // Should not happen: if HPC forward succeeded, HPC backward should too.
        // Fallback: release unused buffers.
        AutodiffBufferPool.release(dD);
        AutodiffBufferPool.release(dC);
        AutodiffBufferPool.release(dB);
        AutodiffBufferPool.release(dA);
        AutodiffBufferPool.release(dDelta);
        AutodiffBufferPool.release(dU);
    };

    RereDiffTensor result = new RereDiffTensor(y, s, List.of(tensor, (RereDiffTensor) delta,
        (RereDiffTensor) A, (RereDiffTensor) B, (RereDiffTensor) C, (RereDiffTensor) D),
        bw, "trapezoidalScan");
    result.scalarParam = ScalarParamEncoding.TrapezoidalScan.packScalarParam(fAIsVec != 0, fDIsScalar != 0, fDeltaBroadcast != 0);
    return result;
}

}
