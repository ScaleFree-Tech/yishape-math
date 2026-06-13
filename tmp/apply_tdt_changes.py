import re

path = r'E:\work\yishape-math\src\main\java\com\yishape\lab\math\autodiff\impl\TangentDiffTensor.java'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add imports
old_import = 'import com.yishape.lab.math.linalg.tensor.ITensor;'
new_import = old_import + '\nimport com.yishape.lab.math.compute.DoubleVectorComputer;\nimport com.yishape.lab.math.compute.gpu.GpuActivation;\nimport com.yishape.lab.math.compute.ops.BinaryOperation;\nimport com.yishape.lab.math.compute.ops.UniversalOperation;\nimport com.yishape.lab.math.compute.ops.ReduceOperation;'
content = content.replace(old_import, new_import, 1)

# 2. Add helper methods after tangentBinary
old_block = '''    private IDoubleTensor tangentBinary(IDoubleTensor other, java.util.function.BinaryOperator<double[]> fn) {
        double[] a = tangent.toDoubleArray();
        double[] b = other.toDoubleArray();
        return new RereDoubleTensor(fn.apply(a, b), tangent.shape());
    }

    // ---- arithmetic with variables ----'''

new_block = '''    private IDoubleTensor tangentBinary(IDoubleTensor other, java.util.function.BinaryOperator<double[]> fn) {
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
        return null; // handled per-call via scalar in tvUnary
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

    // ---- arithmetic with variables ----'''

content = content.replace(old_block, new_block, 1)

# 3. Fix mul() - replace for-loop with tensor ops
mul_old = '''        double[] aV = this.primal.value().toDoubleArray();
        double[] bV = o.primal.value().toDoubleArray();
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        double[] t = new double[aT.length];
        for (int i = 0; i < t.length; i++) t[i] = aT[i] * bV[i] + aV[i] * bT[i];
        return new TangentDiffTensor(p, new RereDoubleTensor(t, p.shape()), List.of(this, o), p);'''

mul_new = '''        IDoubleTensor dB = this.tangent.mul(o.primal.value());
        IDoubleTensor adB = this.primal.value().mul(o.tangent);
        IDiffTensor tSum = dB.add(adB);
        return new TangentDiffTensor(p, tSum, List.of(this, o), p);'''

content = content.replace(mul_old, mul_new, 1)

# 4. Fix div() - replace for-loop with tensor ops
div_old = '''        double[] aV = this.primal.value().toDoubleArray();
        double[] bV = o.primal.value().toDoubleArray();
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        double[] t = new double[aT.length];
        for (int i = 0; i < t.length; i++) t[i] = (aT[i] * bV[i] - aV[i] * bT[i]) / (bV[i] * bV[i]);
        return new TangentDiffTensor(p, new RereDoubleTensor(t, p.shape()), List.of(this, o), p);'''

div_new = '''        IDoubleTensor dB = this.tangent.mul(o.primal.value());
        IDoubleTensor adB = this.primal.value().mul(o.tangent);
        IDiffTensor num = dB.sub(adB);
        IDoubleTensor den = o.primal.value().mul(o.primal.value());
        IDiffTensor tDiv = num.div(den);
        return new TangentDiffTensor(p, tDiv, List.of(this, o), p);'''

content = content.replace(div_old, div_new, 1)

# 5. Fix rdiv()
rdiv_old = '''    @Override public IDiffTensor rdiv(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.rdiv(scalar);
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / scalar; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

rdiv_new = '''    @Override public IDiffTensor rdiv(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.rdiv(scalar);
        IDoubleTensor t = tvUnary(xv -> {
            double[] sq = TVC.binaryOperate(xv, xv, BinaryOperation.MULTIPLY);
            double[] minusScalar = TVC.fill(sq.length, -scalar);
            return TVC.binaryOperate(minusScalar, sq, BinaryOperation.DIVIDE);
        }, xv -> null);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(rdiv_old, rdiv_new, 1)

# 6. Fix reciprocal()
recip_old = '''    @Override public IDiffTensor reciprocal() {
        RereDiffTensor p = (RereDiffTensor) primal.reciprocal();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / (xv[i] * xv[i]); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

recip_new = '''    @Override public IDiffTensor reciprocal() {
        RereDiffTensor p = (RereDiffTensor) primal.reciprocal();
        IDoubleTensor t = tvUnary(xv -> {
            double[] sq = TVC.binaryOperate(xv, xv, BinaryOperation.MULTIPLY);
            double[] minusOnes = TVC.fill(sq.length, -1.0);
            return TVC.binaryOperate(minusOnes, sq, BinaryOperation.DIVIDE);
        }, TangentDiffTensor::gpuReciprocalDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(recip_old, recip_new, 1)

# 7. Fix neg()
neg_old = '''    @Override public IDiffTensor neg() {
        RereDiffTensor p = (RereDiffTensor) primal.neg();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = -d[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

neg_new = '''    @Override public IDiffTensor neg() {
        RereDiffTensor p = (RereDiffTensor) primal.neg();
        IDoubleTensor t = tvScl(-1.0, BinaryOperation.MULTIPLY);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(neg_old, neg_new, 1)

# 8. Fix abs()
abs_old = '''    @Override public IDiffTensor abs() {
        RereDiffTensor p = (RereDiffTensor) primal.abs();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] >= 0 ? 1.0 : -1.0); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

abs_new = '''    @Override public IDiffTensor abs() {
        RereDiffTensor p = (RereDiffTensor) primal.abs();
        IDoubleTensor t = tvUnary(xv -> TVC.sign(xv), TangentDiffTensor::gpuAbsDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(abs_old, abs_new, 1)

# 9. Fix sqrt()
sqrt_old = '''    @Override public IDiffTensor sqrt() {
        RereDiffTensor p = (RereDiffTensor) primal.sqrt();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / (2.0 * sv[i]); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

sqrt_new = '''    @Override public IDiffTensor sqrt() {
        RereDiffTensor p = (RereDiffTensor) primal.sqrt();
        IDoubleTensor t = tvUnary(xv -> {
            double[] half = TVC.fill(xv.length, 0.5);
            double[] sqrtX = TVC.universalOperate(xv, UniversalOperation.SQRT, 0.0);
            return TVC.binaryOperate(half, sqrtX, BinaryOperation.DIVIDE);
        }, TangentDiffTensor::gpuSqrtDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(sqrt_old, sqrt_new, 1)

# 10. Fix exp()
exp_old = '''    @Override public IDiffTensor exp() {
        RereDiffTensor p = (RereDiffTensor) primal.exp();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * ev[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

exp_new = '''    @Override public IDiffTensor exp() {
        RereDiffTensor p = (RereDiffTensor) primal.exp();
        IDoubleTensor t = tvUnary(xv -> TVC.universalOperate(xv, UniversalOperation.EXP, 0.0), TangentDiffTensor::gpuExpDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(exp_old, exp_new, 1)

# 11. Fix log()
log_old = '''    @Override public IDiffTensor log() {
        RereDiffTensor p = (RereDiffTensor) primal.log();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / xv[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

log_new = '''    @Override public IDiffTensor log() {
        RereDiffTensor p = (RereDiffTensor) primal.log();
        IDoubleTensor t = tvUnary(xv -> {
            double[] ones = TVC.fill(xv.length, 1.0);
            return TVC.binaryOperate(ones, xv, BinaryOperation.DIVIDE);
        }, TangentDiffTensor::gpuLogDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(log_old, log_new, 1)

# 12. Fix sin()
sin_old = '''    @Override public IDiffTensor sin() {
        RereDiffTensor p = (RereDiffTensor) primal.sin();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * cv[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

sin_new = '''    @Override public IDiffTensor sin() {
        RereDiffTensor p = (RereDiffTensor) primal.sin();
        IDoubleTensor t = tvUnary(xv -> TVC.universalOperate(xv, UniversalOperation.COS, 0.0), TangentDiffTensor::gpuSinDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(sin_old, sin_new, 1)

# 13. Fix cos()
cos_old = '''    @Override public IDiffTensor cos() {
        RereDiffTensor p = (RereDiffTensor) primal.cos();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = -d[i] * sv[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

cos_new = '''    @Override public IDiffTensor cos() {
        RereDiffTensor p = (RereDiffTensor) primal.cos();
        IDoubleTensor t = tvUnary(xv -> {
            double[] sx = TVC.universalOperate(xv, UniversalOperation.SIN, 0.0);
            return TVC.binaryOperate(sx, -1.0, BinaryOperation.MULTIPLY);
        }, TangentDiffTensor::gpuCosDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(cos_old, cos_new, 1)

# 14. Fix tan()
tan_old = '''    @Override public IDiffTensor tan() {
        RereDiffTensor p = (RereDiffTensor) primal.tan();
        IDoubleTensor t = tangentUnary(d -> {
            double[] tv = tangentUnary(d2 -> { double[] r2 = new double[d2.length]; for (int j = 0; j < d2.length; j++) r2[j] = d2[j] * d2[j]; return r2; }).apply(d);
            double[] ones = new double[d.length];
            for (int i = 0; i < d.length; i++) ones[i] = 1.0;
            return tvBin(new RereDoubleTensor(ones, tangent.shape()), BinaryOperation.SUBTRACT);
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

tan_new = '''    @Override public IDiffTensor tan() {
        RereDiffTensor p = (RereDiffTensor) primal.tan();
        IDoubleTensor t = tvUnary(xv -> {
            double[] cx = TVC.universalOperate(xv, UniversalOperation.COS, 0.0);
            double[] sq = TVC.binaryOperate(cx, cx, BinaryOperation.MULTIPLY);
            double[] ones = TVC.fill(sq.length, 1.0);
            return TVC.binaryOperate(ones, sq, BinaryOperation.DIVIDE);
        }, TangentDiffTensor::gpuTanDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(tan_old, tan_new, 1)

# 15. Fix square()
sq_old = '''    @Override public IDiffTensor square() {
        RereDiffTensor p = (RereDiffTensor) primal.square();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * 2.0 * xv[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

sq_new = '''    @Override public IDiffTensor square() {
        RereDiffTensor p = (RereDiffTensor) primal.square();
        double[] xv = this.primal.value().toDoubleArray();
        double[] twoX = TVC.binaryOperate(xv, 2.0, BinaryOperation.MULTIPLY);
        IDoubleTensor t = tvBin(new RereDoubleTensor(twoX, p.shape()), BinaryOperation.MULTIPLY);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(sq_old, sq_new, 1)

# 16. Fix sigmoid()
sig_old = '''    @Override public IDiffTensor sigmoid() {
        RereDiffTensor p = (RereDiffTensor) primal.sigmoid();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * sv[i] * (1.0 - sv[i]); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

sig_new = '''    @Override public IDiffTensor sigmoid() {
        RereDiffTensor p = (RereDiffTensor) primal.sigmoid();
        IDoubleTensor t = tvUnary(xv -> {
            double[] sv = TVC.universalOperate(xv, UniversalOperation.SIGMOID, 0.0);
            double[] ones = TVC.fill(sv.length, 1.0);
            double[] omS = TVC.binaryOperate(ones, sv, BinaryOperation.SUBTRACT);
            return TVC.binaryOperate(sv, omS, BinaryOperation.MULTIPLY);
        }, TangentDiffTensor::gpuSigmoidDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(sig_old, sig_new, 1)

# 17. Fix relu()
relu_old = '''    @Override public IDiffTensor relu() {
        RereDiffTensor p = (RereDiffTensor) primal.relu();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] > 0 ? 1.0 : 0.0); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

relu_new = '''    @Override public IDiffTensor relu() {
        RereDiffTensor p = (RereDiffTensor) primal.relu();
        IDoubleTensor t = tvUnary(xv -> {
            double[] ones = TVC.fill(xv.length, 1.0);
            double[] s = TVC.sign(xv);
            return TVC.binaryOperate(TVC.binaryOperate(s, 0.5, BinaryOperation.ADD), 0.5, BinaryOperation.MULTIPLY);
        }, TangentDiffTensor::gpuReluDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(relu_old, relu_new, 1)

# 18. Fix tanh()
tanh_old = '''    @Override public IDiffTensor tanh() {
        RereDiffTensor p = (RereDiffTensor) primal.tanh();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (1.0 - tv[i] * tv[i]); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

tanh_new = '''    @Override public IDiffTensor tanh() {
        RereDiffTensor p = (RereDiffTensor) primal.tanh();
        IDoubleTensor t = tvUnary(xv -> {
            double[] tv = TVC.universalOperate(xv, UniversalOperation.TANH, 0.0);
            double[] sq = TVC.binaryOperate(tv, tv, BinaryOperation.MULTIPLY);
            double[] ones = TVC.fill(tv.length, 1.0);
            return TVC.binaryOperate(ones, sq, BinaryOperation.SUBTRACT);
        }, TangentDiffTensor::gpuTanhDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(tanh_old, tanh_new, 1)

# 19. Fix silu()
silu_old = '''    @Override public IDiffTensor silu() {
        RereDiffTensor p = (RereDiffTensor) primal.silu();
        IDoubleTensor t = tangentUnary(d -> {
            double[] sig = tangentUnary(d2 -> { double[] r2 = new double[d2.length]; for (int j = 0; j < d2.length; j++) r2[j] = 1.0 / (1.0 + Math.exp(-d2[j])); return r2; }).apply(d);
            double[] ones = new double[d.length];
            for (int i = 0; i < d.length; i++) ones[i] = 1.0;
            double[] omS = tvBin(new RereDoubleTensor(ones, tangent.shape()), BinaryOperation.SUBTRACT);
            double[] xiSig = tvBin(new RereDoubleTensor(xv, tangent.shape()), BinaryOperation.MULTIPLY);
            return tvBin(new RereDoubleTensor(xiSig.toDoubleArray(), tangent.shape()), BinaryOperation.MULTIPLY);
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

silu_new = '''    @Override public IDiffTensor silu() {
        RereDiffTensor p = (RereDiffTensor) primal.silu();
        IDoubleTensor t = tvUnary(xv -> {
            double[] sig = TVC.universalOperate(xv, UniversalOperation.SIGMOID, 0.0);
            double[] ones = TVC.fill(sig.length, 1.0);
            double[] omS = TVC.binaryOperate(ones, sig, BinaryOperation.SUBTRACT);
            double[] xiSig = TVC.binaryOperate(xv, sig, BinaryOperation.MULTIPLY);
            double[] term2 = TVC.binaryOperate(xiSig, omS, BinaryOperation.MULTIPLY);
            return TVC.binaryOperate(sig, term2, BinaryOperation.ADD);
        }, TangentDiffTensor::gpuSiluDeriv);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }'''

content = content.replace(silu_old, silu_new, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("All changes applied successfully!")
print(f"File size: {len(content)} bytes")
