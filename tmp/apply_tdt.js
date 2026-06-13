const fs = require('fs');
const path = 'E:/work/yishape-math/src/main/java/com/yishape/lab/math/autodiff/impl/TangentDiffTensor.java';
let content = fs.readFileSync(path, 'utf8');

// 1. Add imports
content = content.replace(
  'import com.yishape.lab.math.linalg.tensor.ITensor;',
  'import com.yishape.lab.math.linalg.tensor.ITensor;\nimport com.yishape.lab.math.compute.DoubleVectorComputer;\nimport com.yishape.lab.math.compute.gpu.GpuActivation;\nimport com.yishape.lab.math.compute.ops.BinaryOperation;\nimport com.yishape.lab.math.compute.ops.UniversalOperation;\nimport com.yishape.lab.math.compute.ops.ReduceOperation;'
);

// 2. Add helper methods after tangentBinary
const tangentBinaryEnd = '    private IDoubleTensor tangentBinary(IDoubleTensor other, java.util.function.BinaryOperator<double[]> fn) {\n        double[] a = tangent.toDoubleArray();\n        double[] b = other.toDoubleArray();\n        return new RereDoubleTensor(fn.apply(a, b), tangent.shape());\n    }\n\n    // ---- arithmetic with variables ----';

const newHelpers = `    private IDoubleTensor tangentBinary(IDoubleTensor other, java.util.function.BinaryOperator<double[]> fn) {
        double[] a = tangent.toDoubleArray();
        double[] b = other.toDoubleArray();
        return new RereDoubleTensor(fn.apply(a, b), tangent.shape());
    }
    /** Accelerated binary op via DoubleVectorComputer (GPU/HPC/SIMD/SISD). */
    private IDoubleTensor tvBin(IDoubleTensor other, BinaryOperation op) {
        double[] a = tangent.toDoubleArray();
        double[] b = other.toDoubleArray();
        return new RereDoubleTensor(TVC.binaryOperate(a, b, op), tangent.shape());
    }
    /** Accelerated scalar multiply via DoubleVectorComputer. */
    private IDoubleTensor tvScl(double scalar, BinaryOperation op) {
        double[] a = tangent.toDoubleArray();
        return new RereDoubleTensor(TVC.binaryOperate(a, scalar, op), tangent.shape());
    }
    /** Unary JVP: result = g'(xv) * tangent, with GPU acceleration. */
    private IDoubleTensor tvUnary(Function<double[], double[]> primalDeriv, Function<double[], double[]> gpuPrimalDeriv) {
        double[] xv = this.primal.value().toDoubleArray();
        double[] d = tangent.toDoubleArray();
        double[] deriv = gpuPrimalDeriv.apply(xv);
        if (deriv == null) deriv = primalDeriv.apply(xv);
        double[] r = TVC.binaryOperate(deriv, d, BinaryOperation.MULTIPLY);
        return new RereDoubleTensor(r, tangent.shape());
    }
    /** Unary activation JVP via GpuActivation -> SIMD -> SISD. */
    private IDoubleTensor tvAct(Function<double[], double[]> fn, Function<double[], double[]> gpuFn) {
        double[] d = tangent.toDoubleArray();
        double[] r = gpuFn.apply(d);
        if (r == null) r = fn.apply(d);
        return new RereDoubleTensor(r, tangent.shape());
    }

    // ---- GPU primal-derivative wrappers for tvUnary ----
    private static double[] gpuNeg(double[] xv) {
        double[] ones = TVC.fill(xv.length, 1.0);
        return TVC.negate(ones);
    }
    private static double[] gpuAbsDeriv(double[] xv) {
        return TVC.sign(xv);
    }
    private static double[] gpuSqrtDeriv(double[] xv) {
        double[] sv = GpuActivation.trySqrt(xv);
        if (sv == null) return null;
        double[] ones = TVC.fill(sv.length, 1.0);
        double[] inv = TVC.binaryOperate(ones, sv, BinaryOperation.DIVIDE);
        return TVC.binaryOperate(inv, 0.5, BinaryOperation.MULTIPLY);
    }
    private static double[] gpuExpDeriv(double[] xv) { return GpuActivation.tryExp(xv); }
    private static double[] gpuLogDeriv(double[] xv) {
        double[] ones = TVC.fill(xv.length, 1.0);
        return TVC.binaryOperate(ones, xv, BinaryOperation.DIVIDE);
    }
    private static double[] gpuSinDeriv(double[] xv) { return GpuActivation.tryCos(xv); }
    private static double[] gpuCosDeriv(double[] xv) {
        double[] sv = GpuActivation.trySin(xv);
        if (sv == null) return null;
        return TVC.binaryOperate(sv, -1.0, BinaryOperation.MULTIPLY);
    }
    private static double[] gpuTanhDeriv(double[] xv) {
        double[] tv = GpuActivation.tryTanh(xv);
        if (tv == null) return null;
        double[] sq = TVC.binaryOperate(tv, tv, BinaryOperation.MULTIPLY);
        double[] ones = TVC.fill(tv.length, 1.0);
        return TVC.binaryOperate(ones, sq, BinaryOperation.SUBTRACT);
    }
    private static double[] gpuSigmoidDeriv(double[] xv) {
        double[] sv = GpuActivation.trySigmoid(xv);
        if (sv == null) return null;
        double[] ones = TVC.fill(sv.length, 1.0);
        double[] omS = TVC.binaryOperate(ones, sv, BinaryOperation.SUBTRACT);
        return TVC.binaryOperate(sv, omS, BinaryOperation.MULTIPLY);
    }
    private static double[] gpuReluDeriv(double[] xv) {
        double[] rv = GpuActivation.tryRelu(xv);
        if (rv == null) return null;
        double[] ones = TVC.fill(rv.length, 1.0);
        return TVC.binaryOperate(TVC.binaryOperate(TVC.sign(rv), 0.5, BinaryOperation.ADD), 0.5, BinaryOperation.MULTIPLY);
    }
    private static double[] gpuReciprocalDeriv(double[] xv) {
        double[] sq = TVC.binaryOperate(xv, xv, BinaryOperation.MULTIPLY);
        double[] minusOnes = TVC.fill(sq.length, -1.0);
        return TVC.binaryOperate(minusOnes, sq, BinaryOperation.DIVIDE);
    }
    private static double[] gpuTanDeriv(double[] xv) {
        double[] cx = TVC.universalOperate(xv, UniversalOperation.COS, 0.0);
        if (cx == null) return null;
        double[] sq = TVC.binaryOperate(cx, cx, BinaryOperation.MULTIPLY);
        double[] ones = TVC.fill(sq.length, 1.0);
        return TVC.binaryOperate(ones, sq, BinaryOperation.DIVIDE);
    }
    private static double[] gpuSoftplusDeriv(double[] xv) {
        double[] sv = GpuActivation.trySigmoid(xv);
        if (sv == null) return null;
        return sv;
    }
    private static double[] gpuEluDeriv(double[] xv) {
        double[] ev = GpuActivation.tryElu(xv);
        if (ev == null) return null;
        double[] ones = TVC.fill(ev.length, 1.0);
        double[] aExp = TVC.binaryOperate(ev, ones, BinaryOperation.SUBTRACT);
        aExp = TVC.binaryOperate(aExp, -1.0, BinaryOperation.MULTIPLY);
        double[] s = TVC.sign(xv);
        double[] mask = TVC.binaryOperate(TVC.binaryOperate(s, 0.5, BinaryOperation.ADD), 0.5, BinaryOperation.MULTIPLY);
        return TVC.binaryOperate(ones, TVC.binaryOperate(aExp, mask, BinaryOperation.MULTIPLY), BinaryOperation.SUBTRACT);
    }
    private static double[] gpuLeakyReluDeriv(double[] xv) {
        double[] rv = GpuActivation.tryLeakyRelu(xv);
        if (rv == null) return null;
        double[] ones = TVC.fill(rv.length, 1.0);
        double[] alpha = TVC.fill(rv.length, 0.01);
        double[] diff = TVC.binaryOperate(rv, xv, BinaryOperation.SUBTRACT);
        double[] scaled = TVC.binaryOperate(alpha, diff, BinaryOperation.DIVIDE);
        return TVC.binaryOperate(ones, scaled, BinaryOperation.SUBTRACT);
    }
    private static double[] gpuHardtanhDeriv(double[] xv) {
        double[] ones = TVC.fill(xv.length, 1.0);
        double[] lo = TVC.fill(xv.length, -1.0);
        double[] hi = TVC.fill(xv.length, 1.0);
        double[] gtLo = TVC.binaryOperate(xv, lo, BinaryOperation.SUBTRACT);
        double[] ltHi = TVC.binaryOperate(hi, xv, BinaryOperation.SUBTRACT);
        double[] cond = TVC.binaryOperate(gtLo, ltHi, BinaryOperation.MULTIPLY);
        return TVC.binaryOperate(cond, ones, BinaryOperation.MULTIPLY);
    }
    private static double[] gpuPowDeriv(double[] xv) {
        return null;
    }
    private static double[] gpuSiluDeriv(double[] xv) {
        double[] sig = GpuActivation.trySigmoid(xv);
        if (sig == null) return null;
        double[] ones = TVC.fill(sig.length, 1.0);
        double[] omS = TVC.binaryOperate(ones, sig, BinaryOperation.SUBTRACT);
        double[] xiSig = TVC.binaryOperate(xv, sig, BinaryOperation.MULTIPLY);
        double[] term2 = TVC.binaryOperate(xiSig, omS, BinaryOperation.MULTIPLY);
        return TVC.binaryOperate(sig, term2, BinaryOperation.ADD);
    }
    private static double[] gpuGeluDeriv(double[] xv) {
        double[] gelu = GpuActivation.tryGelu(xv);
        if (gelu == null) return null;
        double[] scaled = TVC.binaryOperate(xv, 1.702, BinaryOperation.MULTIPLY);
        return TVC.universalOperate(scaled, UniversalOperation.SIGMOID, 0.0);
    }
    private static double[] gpuMishDeriv(double[] xv) {
        double[] sig = GpuActivation.trySigmoid(xv);
        if (sig == null) return null;
        double[] sp = TVC.universalOperate(xv, UniversalOperation.LOG, 0.0);
        double[] tanhSp = TVC.universalOperate(sp, UniversalOperation.TANH, 0.0);
        double[] ones = TVC.fill(tanhSp.length, 1.0);
        double[] dTanhSp = TVC.binaryOperate(ones, tanhSp, BinaryOperation.SUBTRACT);
        dTanhSp = TVC.binaryOperate(dTanhSp, tanhSp, BinaryOperation.MULTIPLY);
        dTanhSp = TVC.binaryOperate(dTanhSp, sig, BinaryOperation.MULTIPLY);
        double[] xDt = TVC.binaryOperate(xv, dTanhSp, BinaryOperation.MULTIPLY);
        return TVC.binaryOperate(tanhSp, xDt, BinaryOperation.ADD);
    }
    private static double[] gpuSeluDeriv(double[] xv) {
        double alpha = 1.6732632423543772848170429916717;
        double scale = 1.0507009873554804934193349852946;
        double[] expX = TVC.universalOperate(xv, UniversalOperation.EXP, 0.0);
        double[] aExp = TVC.binaryOperate(expX, alpha, BinaryOperation.MULTIPLY);
        double[] ones = TVC.fill(xv.length, 1.0);
        double[] s = TVC.sign(xv);
        double[] mask = TVC.binaryOperate(TVC.binaryOperate(s, 0.5, BinaryOperation.ADD), 0.5, BinaryOperation.MULTIPLY);
        double[] deriv = TVC.binaryOperate(ones, aExp, BinaryOperation.SUBTRACT);
        deriv = TVC.binaryOperate(deriv, mask, BinaryOperation.MULTIPLY);
        return TVC.binaryOperate(deriv, scale, BinaryOperation.MULTIPLY);
    }

    // ---- arithmetic with variables ----`;

content = content.replace(tangentBinaryEnd, newHelpers);

// 3. Fix mul()
const mulOld = '        double[] aV = this.primal.value().toDoubleArray();\n        double[] bV = o.primal.value().toDoubleArray();\n        double[] aT = this.tangent.toDoubleArray();\n        double[] bT = o.tangent.toDoubleArray();\n        double[] t = new double[aT.length];\n        for (int i = 0; i < t.length; i++) t[i] = aT[i] * bV[i] + aV[i] * bT[i];\n        return new TangentDiffTensor(p, new RereDoubleTensor(t, p.shape()), List.of(this, o), p);';
const mulNew = '        IDoubleTensor dB = this.tangent.mul(o.primal.value());\n        IDoubleTensor adB = this.primal.value().mul(o.tangent);\n        IDiffTensor tSum = dB.add(adB);\n        return new TangentDiffTensor(p, tSum, List.of(this, o), p);';
content = content.replace(mulOld, mulNew);

// 4. Fix div()
const divOld = '        double[] aV = this.primal.value().toDoubleArray();\n        double[] bV = o.primal.value().toDoubleArray();\n        double[] aT = this.tangent.toDoubleArray();\n        double[] bT = o.tangent.toDoubleArray();\n        double[] t = new double[aT.length];\n        for (int i = 0; i < t.length; i++) t[i] = (aT[i] * bV[i] - aV[i] * bT[i]) / (bV[i] * bV[i]);\n        return new TangentDiffTensor(p, new RereDoubleTensor(t, p.shape()), List.of(this, o), p);';
const divNew = '        IDoubleTensor dB = this.tangent.mul(o.primal.value());\n        IDoubleTensor adB = this.primal.value().mul(o.tangent);\n        IDiffTensor num = dB.sub(adB);\n        IDoubleTensor den = o.primal.value().mul(o.primal.value());\n        IDiffTensor tDiv = num.div(den);\n        return new TangentDiffTensor(p, tDiv, List.of(this, o), p);';
content = content.replace(divOld, divNew);

// 5. Fix rdiv()
const rdivOld = '    @Override public IDiffTensor rdiv(double scalar) {\n        RereDiffTensor p = (RereDiffTensor) primal.rdiv(scalar);\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / scalar; return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const rdivNew = '    @Override public IDiffTensor rdiv(double scalar) {\n        RereDiffTensor p = (RereDiffTensor) primal.rdiv(scalar);\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] sq = TVC.binaryOperate(xv, xv, BinaryOperation.MULTIPLY);\n            double[] minusScalar = TVC.fill(sq.length, -scalar);\n            return TVC.binaryOperate(minusScalar, sq, BinaryOperation.DIVIDE);\n        }, xv -> null);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(rdivOld, rdivNew);

// 6. Fix reciprocal()
const recipOld = '    @Override public IDiffTensor reciprocal() {\n        RereDiffTensor p = (RereDiffTensor) primal.reciprocal();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / (xv[i] * xv[i]); return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const recipNew = '    @Override public IDiffTensor reciprocal() {\n        RereDiffTensor p = (RereDiffTensor) primal.reciprocal();\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] sq = TVC.binaryOperate(xv, xv, BinaryOperation.MULTIPLY);\n            double[] minusOnes = TVC.fill(sq.length, -1.0);\n            return TVC.binaryOperate(minusOnes, sq, BinaryOperation.DIVIDE);\n        }, TangentDiffTensor::gpuReciprocalDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(recipOld, recipNew);

// 7. Fix neg()
const negOld = '    @Override public IDiffTensor neg() {\n        RereDiffTensor p = (RereDiffTensor) primal.neg();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = -d[i]; return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const negNew = '    @Override public IDiffTensor neg() {\n        RereDiffTensor p = (RereDiffTensor) primal.neg();\n        IDoubleTensor t = tvScl(-1.0, BinaryOperation.MULTIPLY);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(negOld, negNew);

// 8. Fix abs()
const absOld = '    @Override public IDiffTensor abs() {\n        RereDiffTensor p = (RereDiffTensor) primal.abs();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] >= 0 ? 1.0 : -1.0); return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const absNew = '    @Override public IDiffTensor abs() {\n        RereDiffTensor p = (RereDiffTensor) primal.abs();\n        IDoubleTensor t = tvUnary(xv -> TVC.sign(xv), TangentDiffTensor::gpuAbsDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(absOld, absNew);

// 9. Fix sqrt()
const sqrtOld = '    @Override public IDiffTensor sqrt() {\n        RereDiffTensor p = (RereDiffTensor) primal.sqrt();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / (2.0 * sv[i]); return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const sqrtNew = '    @Override public IDiffTensor sqrt() {\n        RereDiffTensor p = (RereDiffTensor) primal.sqrt();\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] half = TVC.fill(xv.length, 0.5);\n            double[] sqrtX = TVC.universalOperate(xv, UniversalOperation.SQRT, 0.0);\n            return TVC.binaryOperate(half, sqrtX, BinaryOperation.DIVIDE);\n        }, TangentDiffTensor::gpuSqrtDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(sqrtOld, sqrtNew);

// 10. Fix exp()
const expOld = '    @Override public IDiffTensor exp() {\n        RereDiffTensor p = (RereDiffTensor) primal.exp();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * ev[i]; return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const expNew = '    @Override public IDiffTensor exp() {\n        RereDiffTensor p = (RereDiffTensor) primal.exp();\n        IDoubleTensor t = tvUnary(xv -> TVC.universalOperate(xv, UniversalOperation.EXP, 0.0), TangentDiffTensor::gpuExpDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(expOld, expNew);

// 11. Fix log()
const logOld = '    @Override public IDiffTensor log() {\n        RereDiffTensor p = (RereDiffTensor) primal.log();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / xv[i]; return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const logNew = '    @Override public IDiffTensor log() {\n        RereDiffTensor p = (RereDiffTensor) primal.log();\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] ones = TVC.fill(xv.length, 1.0);\n            return TVC.binaryOperate(ones, xv, BinaryOperation.DIVIDE);\n        }, TangentDiffTensor::gpuLogDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(logOld, logNew);

// 12. Fix sin()
const sinOld = '    @Override public IDiffTensor sin() {\n        RereDiffTensor p = (RereDiffTensor) primal.sin();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * cv[i]; return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const sinNew = '    @Override public IDiffTensor sin() {\n        RereDiffTensor p = (RereDiffTensor) primal.sin();\n        IDoubleTensor t = tvUnary(xv -> TVC.universalOperate(xv, UniversalOperation.COS, 0.0), TangentDiffTensor::gpuSinDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(sinOld, sinNew);

// 13. Fix cos()
const cosOld = '    @Override public IDiffTensor cos() {\n        RereDiffTensor p = (RereDiffTensor) primal.cos();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = -d[i] * sv[i]; return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const cosNew = '    @Override public IDiffTensor cos() {\n        RereDiffTensor p = (RereDiffTensor) primal.cos();\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] sx = TVC.universalOperate(xv, UniversalOperation.SIN, 0.0);\n            return TVC.binaryOperate(sx, -1.0, BinaryOperation.MULTIPLY);\n        }, TangentDiffTensor::gpuCosDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(cosOld, cosNew);

// 14. Fix tan()
const tanOld = '    @Override public IDiffTensor tan() {\n        RereDiffTensor p = (RereDiffTensor) primal.tan();\n        IDoubleTensor t = tangentUnary(d -> {\n            double[] tv = tangentUnary(d2 -> { double[] r2 = new double[d2.length]; for (int j = 0; j < d2.length; j++) r2[j] = d2[j] * d2[j]; return r2; }).apply(d);\n            double[] ones = new double[d.length];\n            for (int i = 0; i < d.length; i++) ones[i] = 1.0;\n            return tvBin(new RereDoubleTensor(ones, tangent.shape()), BinaryOperation.SUBTRACT);\n        });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const tanNew = '    @Override public IDiffTensor tan() {\n        RereDiffTensor p = (RereDiffTensor) primal.tan();\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] cx = TVC.universalOperate(xv, UniversalOperation.COS, 0.0);\n            double[] sq = TVC.binaryOperate(cx, cx, BinaryOperation.MULTIPLY);\n            double[] ones = TVC.fill(sq.length, 1.0);\n            return TVC.binaryOperate(ones, sq, BinaryOperation.DIVIDE);\n        }, TangentDiffTensor::gpuTanDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(tanOld, tanNew);

// 15. Fix square()
const sqOld = '    @Override public IDiffTensor square() {\n        RereDiffTensor p = (RereDiffTensor) primal.square();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * 2.0 * xv[i]; return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const sqNew = '    @Override public IDiffTensor square() {\n        RereDiffTensor p = (RereDiffTensor) primal.square();\n        double[] xv = this.primal.value().toDoubleArray();\n        double[] twoX = TVC.binaryOperate(xv, 2.0, BinaryOperation.MULTIPLY);\n        IDoubleTensor t = tvBin(new RereDoubleTensor(twoX, p.shape()), BinaryOperation.MULTIPLY);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(sqOld, sqNew);

// 16. Fix sigmoid()
const sigOld = '    @Override public IDiffTensor sigmoid() {\n        RereDiffTensor p = (RereDiffTensor) primal.sigmoid();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * sv[i] * (1.0 - sv[i]); return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const sigNew = '    @Override public IDiffTensor sigmoid() {\n        RereDiffTensor p = (RereDiffTensor) primal.sigmoid();\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] sv = TVC.universalOperate(xv, UniversalOperation.SIGMOID, 0.0);\n            double[] ones = TVC.fill(sv.length, 1.0);\n            double[] omS = TVC.binaryOperate(ones, sv, BinaryOperation.SUBTRACT);\n            return TVC.binaryOperate(sv, omS, BinaryOperation.MULTIPLY);\n        }, TangentDiffTensor::gpuSigmoidDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(sigOld, sigNew);

// 17. Fix relu()
const reluOld = '    @Override public IDiffTensor relu() {\n        RereDiffTensor p = (RereDiffTensor) primal.relu();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] > 0 ? 1.0 : 0.0); return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const reluNew = '    @Override public IDiffTensor relu() {\n        RereDiffTensor p = (RereDiffTensor) primal.relu();\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] ones = TVC.fill(xv.length, 1.0);\n            double[] s = TVC.sign(xv);\n            return TVC.binaryOperate(TVC.binaryOperate(s, 0.5, BinaryOperation.ADD), 0.5, BinaryOperation.MULTIPLY);\n        }, TangentDiffTensor::gpuReluDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(reluOld, reluNew);

// 18. Fix tanh()
const tanhOld = '    @Override public IDiffTensor tanh() {\n        RereDiffTensor p = (RereDiffTensor) primal.tanh();\n        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (1.0 - tv[i] * tv[i]); return r; });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const tanhNew = '    @Override public IDiffTensor tanh() {\n        RereDiffTensor p = (RereDiffTensor) primal.tanh();\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] tv = TVC.universalOperate(xv, UniversalOperation.TANH, 0.0);\n            double[] sq = TVC.binaryOperate(tv, tv, BinaryOperation.MULTIPLY);\n            double[] ones = TVC.fill(tv.length, 1.0);\n            return TVC.binaryOperate(ones, sq, BinaryOperation.SUBTRACT);\n        }, TangentDiffTensor::gpuTanhDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(tanhOld, tanhNew);

// 19. Fix silu()
const siluOld = '    @Override public IDiffTensor silu() {\n        RereDiffTensor p = (RereDiffTensor) primal.silu();\n        IDoubleTensor t = tangentUnary(d -> {\n            double[] sig = tangentUnary(d2 -> { double[] r2 = new double[d2.length]; for (int j = 0; j < d2.length; j++) r2[j] = 1.0 / (1.0 + Math.exp(-d2[j])); return r2; }).apply(d);\n            double[] ones = new double[d.length];\n            for (int i = 0; i < d.length; i++) ones[i] = 1.0;\n            double[] omS = tvBin(new RereDoubleTensor(ones, tangent.shape()), BinaryOperation.SUBTRACT);\n            double[] xiSig = tvBin(new RereDoubleTensor(xv, tangent.shape()), BinaryOperation.MULTIPLY);\n            return tvBin(new RereDoubleTensor(xiSig.toDoubleArray(), tangent.shape()), BinaryOperation.MULTIPLY);\n        });\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
const siluNew = '    @Override public IDiffTensor silu() {\n        RereDiffTensor p = (RereDiffTensor) primal.silu();\n        IDoubleTensor t = tvUnary(xv -> {\n            double[] sig = TVC.universalOperate(xv, UniversalOperation.SIGMOID, 0.0);\n            double[] ones = TVC.fill(sig.length, 1.0);\n            double[] omS = TVC.binaryOperate(ones, sig, BinaryOperation.SUBTRACT);\n            double[] xiSig = TVC.binaryOperate(xv, sig, BinaryOperation.MULTIPLY);\n            double[] term2 = TVC.binaryOperate(xiSig, omS, BinaryOperation.MULTIPLY);\n            return TVC.binaryOperate(sig, term2, BinaryOperation.ADD);\n        }, TangentDiffTensor::gpuSiluDeriv);\n        return new TangentDiffTensor(p, t, List.of(this), p);\n    }';
content = content.replace(siluOld, siluNew);

fs.writeFileSync(path, content, 'utf8');
console.log('Done! File written.');
